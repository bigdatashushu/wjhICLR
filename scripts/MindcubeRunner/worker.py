"""One label-free episode, with the sandbox on the subprocess main thread."""
from __future__ import annotations

import ast
import hashlib
from itertools import count
import json
import math
import os
from pathlib import Path
import sys
import time

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "MindcubeRunner"

from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.action_check import check_action
from skill3d.schemas.tool_response import model_tool_response
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

from . import VERSION
from .budget import DualBudget
from .common import Config, digest, read, validate_source, write
from .data import load, pair_identity, problem
from .program import OptionKernel, ProgramFormatError, parse_json, parse_program, wrong_controls
from .protocol import messages
from .tools import ALLOWED, MEASUREMENTS, build_registry, empty_counts
from .transport import ProxyOpener, auth_loader, health
from .traces import POLICY, TraceError, TraceWriter, mark_orphans, storage_check

JOB_SCHEMA = "mindcube-label-free-worker/1"


def feedback(code):
    repair = {
        "control_namespace_error": "Use global ReturnAnswer or YieldObservations; ctx.tools has evidence tools only.",
        "program_missing_action": "End solve with global ReturnAnswer or YieldObservations.",
        "runtime_error": "Correct the Python error; valid answers are current option letters.",
    }.get(code, "Output one closed python fence with only def solve(ctx):.")
    return [{"code": code, "repair": repair}]


def solve(job, client, budget):
    config = Config(**job["config"])
    item, arm = job["item"], job["arm"]
    directory = Path(job["output_directory"])
    cache = load(item, prepared=arm == "skill")
    observations, errors, review_ids, hint_ids, rounds = [], [], [], [], []
    counters, kernel = empty_counts(), None
    status, answer = "no_answer", None
    if arm == "skill":
        registry, scene, counters = build_registry(cache, problem(item), config)
        ordinal = count()
        kernel = OptionKernel(registry, scene, frames=list(cache.frames), labels=item["labels"],
                              cell_timeout_s=config.cell_timeout_s, episode_id=item["sample_id"],
                              call_id_factory=lambda: f"mc{next(ordinal):08d}")
    started = time.monotonic()
    def state():
        return {"review_ids": list(review_ids), "hint_ids": list(hint_ids),
                "errors": errors, "observations": observations, "counters": dict(counters)}

    def checkpoint(trace, row):
        trace.finish(row["status"], state(), row)
        write(directory / f"{row['round']:02d}.execution.json", row, exclusive=True)

    for index in range(config.request_limit(arm)):
        trace = TraceWriter(directory / "traces" / f"{index:02d}", {
            "run_identity": job["run_identity"], "pair_identity": pair_identity(item),
            "sample_id": item["sample_id"], "arm": arm, "round_index": index,
            "config": config.public(), "source_map_sha256": digest(job.get("source", {})),
            "initial_receipt_sha256": digest(job["initial_receipt"]),
            "reference_answer_visible": False})
        trace.event("state_before", state())
        if kernel:
            kernel.trace = trace
        opener = getattr(client, "opener", None)
        # BailianClient keeps the opener in a private host attribute.
        opener = getattr(client, "_opener", opener)
        if isinstance(opener, ProxyOpener):
            opener.trace = trace
        try:
            context, receipt = messages(item, cache, arm=arm, config=config, round_index=index,
                                        observations=observations, errors=errors,
                                        review_ids=review_ids, hint_ids=hint_ids, counters=counters)
        except ValueError as exc:
            status = "context_limit" if str(exc) == "context_limit" else "context_error"
            trace.finish(status, state(), {"sent": False, "error_type": type(exc).__name__})
            break
        if index == 0 and receipt != job["initial_receipt"]:
            trace.finish("initial_input_changed", state(), {"sent": False})
            raise ValueError("initial_input_changed")
        trace.event("prepared", {"messages": context, "receipt": receipt})
        final = receipt["phase"] == "final"
        if final and kernel:
            kernel.set_tools_enabled(False)
        row = {"round": index, "request": receipt, "phase": receipt["phase"],
               "parameters": config.kwargs(), "status": "intent", "automatic_retry": False,
               "trace": f"traces/{index:02d}/trace.json", "trace_schema": POLICY["schema"]}
        rounds.append(row)
        write(directory / f"{index:02d}.intent.json", row, exclusive=True)
        try:
            with budget.request() as usage_receipt:
                row["request_admitted_at"] = time.time()
                trace.event("admitted", {"parameters": config.kwargs()})
                text = client.chat(context, **config.kwargs())
                row["response_received_at"] = time.time()
                trace.event("assistant_content", {"content": text,
                    "compatibility_output_sha256": digest(text),
                    "utf8_sha256": hashlib.sha256(text.encode()).hexdigest()})
                usage = {k: v for k, v in client.last_usage.items()
                         if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                         and type(v) in (int, float) and math.isfinite(v) and v >= 0}
                usage_receipt["usage"] = usage
                actual = client.last_response_identity
                if actual.get("model") != config.model or actual.get("model_reported") is not True:
                    raise ValueError("actual_model_changed")
                # No provider exception text, full prompts, reasoning, or labels.
                row.update(status="received", usage=usage,
                           response_identity={"model": actual["model"], "model_reported": True},
                           output_sha256=digest(text))
        except BaseException as exc:
            row.update(status="trace_error" if trace.failed else "service_error",
                       error_type=type(exc).__name__)
            write(job["stop_file"], {"reason": "service_or_budget_deviation"})
            status = row["status"]
            if not trace.failed:
                checkpoint(trace, row)
            break
        if final:
            try:
                answer, status = parse_json(text, item["labels"]), "answered"
            except (TypeError, ValueError, json.JSONDecodeError):
                status = "invalid_answer"
            row.update(status=status, answer=answer)
            checkpoint(trace, row)
            break
        try:
            program = parse_program(text)
        except ProgramFormatError as exc:
            errors = feedback(str(exc))
            row.update(status="program_parse_error", feedback_errors=errors)
            checkpoint(trace, row)
            continue
        row["program_sha256"] = digest(program)
        trace.event("program", {"source": program, "parser_status": "parsed",
            "compatibility_program_sha256": digest(program),
            "utf8_sha256": hashlib.sha256(program.encode()).hexdigest()})
        guarded = ast_guard(program, allowed_tools=ALLOWED)
        wrong = wrong_controls(program)
        namespace_errors = {f"Unknown tool in the tools namespace: {name}" for name in wrong}
        if ([v for v in guarded.violations if v not in namespace_errors]
                or any(isinstance(node, ast.Name) and
                       node.id in {"ReadObservation", "AnswerPayload"}
                       for node in ast.walk(ast.parse(program)))):
            row["status"], status = "program_rejected", "program_rejected"
            row["guard_violations"] = list(guarded.violations)
            checkpoint(trace, row)
            break
        if wrong or not check_action(program, tool_names=ALLOWED).ok:
            errors = feedback("control_namespace_error" if wrong else "program_missing_action")
            row.update(status="repairable_program_error", feedback_errors=errors, executed=False)
            checkpoint(trace, row)
            continue
        kernel.reset_user_namespace()
        cell = kernel.run_cell(program, yield_on_fallthrough=True)
        if trace.failed:
            raise TraceError("tool_trace_commit_failed")
        trace.event("cell_completed", {"calls": [c.model_dump(mode="json") for c in kernel.cell_calls],
            "results": [r.model_dump(mode="json") for r in kernel.cell_results],
            "authorization_receipts": kernel.authorization_receipts,
            "observation_reads": kernel.observation_reads})
        fresh, fresh_hints, delivered = [], [], []
        for result in kernel.cell_results:
            if result.status == "ok" and not result.error and not result.invalidated_by:
                public = model_tool_response(result)
                public = {k: public[k] for k in ("result_id", "source_tool", "status", "data")}
                public["data"] = {k: v for k, v in public["data"].items()
                                  if k in registry.public_fields[result.tool]}
                observations.append(public)
                delivered.append(public)
                if result.tool in MEASUREMENTS:
                    fresh.append(result.result_id)
                else:
                    fresh_hints.append(result.result_id)
        # A malformed or failed program keeps pending review obligations. A valid
        # completed program consumes the delivered input and carries new IDs on.
        if cell.error_code or cell.answer_untrusted:
            review_ids = list(dict.fromkeys([*review_ids, *fresh]))
            hint_ids = list(dict.fromkeys([*hint_ids, *fresh_hints]))
        else:
            review_ids, hint_ids = fresh, fresh_hints
        row.update(tool_observations=delivered, new_measurement_ids=fresh,
                   new_hint_ids=fresh_hints,
                   unreliable_measurement_ids=[
                       result["result_id"] for result in delivered
                       if result["source_tool"] in MEASUREMENTS
                       and result["data"].get("status") == "unreliable"],
                   tool_counts_after=dict(counters),
                   cell={"terminated": cell.terminated, "error_code": cell.error_code,
                         "error_type": cell.error_type, "error_line": cell.error_line,
                         "answer_untrusted": cell.answer_untrusted,
                         "yielded_result_ids": cell.yielded_result_ids})
        errors, stop_episode = [], False
        if cell.error_code or cell.answer_untrusted:
            errors = [{"code": cell.error_code or "untrusted_answer",
                       "type": cell.error_type, "line": cell.error_line}]
            if cell.answer_untrusted or cell.error_code in ("timeout", "tool_contract"):
                status, row["status"], stop_episode = "run_error", "run_error", True
            else:
                errors.extend(feedback("runtime_error"))
                row["status"] = "repairable_program_error"
        elif cell.answer is not None:
            if fresh:
                row.update(status="measurement_review_required", provisional_answer=cell.answer)
            else:
                answer, status, row["status"], stop_episode = cell.answer, "answered", "answered", True
        else:
            if set(cell.yielded_result_ids) - {r["result_id"] for r in observations}:
                errors = [{"code": "invalid_observation_reference"}]
            row["status"] = "yielded" if cell.terminated == "yield" else "no_answer"
        checkpoint(trace, row)
        if stop_episode:
            break
    usage = {}
    for row in rounds:
        for key, value in row.get("usage", {}).items():
            usage[key] = usage.get(key, 0) + value
    usable_measurements = any(
        result["source_tool"] in MEASUREMENTS
        and result["data"].get("status") != "unreliable"
        and ("depth" in result["data"] or "distance" in result["data"])
        for result in observations
    )
    result = {"schema": VERSION, "sample_id": item["sample_id"], "source_split": "tinybench",
              "source_task_type": item["task_type"], "arm": arm,
              "pair_identity": pair_identity(item), "run_identity": job["run_identity"],
              "status": status, "answer": answer, "request_count": len(rounds),
              "usage": usage, "tool_counts": dict(counters), "rounds": rounds,
              "composition": cache.provenance, "elapsed_seconds": round(time.monotonic() - started, 6),
              "basis": "mixed" if usable_measurements else "visual_estimate",
              "reference_answer_visible": False, "real_labels_available": False,
              "automatic_retry": False, "main_thread_kernel": True}
    result["receipt_sha256"] = digest(result)
    write(directory / "result.json", result, exclusive=True)
    return result


def execute(job):
    if job["schema"] != JOB_SCHEMA:
        raise ValueError("invalid_job_schema")
    validate_source(job["source"])
    config = Config(**job["config"])
    if job.get("trace_policy") != POLICY:
        raise TraceError("frozen_trace_policy_changed")
    storage_check(job["output_directory"])
    loader = auth_loader(job["auth_module_root"])
    health(job["gateway"], loader)
    # A non-secret placeholder satisfies BailianClient's local configured check.
    os.environ["MINDCUBE_PROXY_PLACEHOLDER"] = "local-proxy-configuration"
    client = BailianClient(
        model=config.model, timeout_s=config.timeout_s, max_output_tokens=config.max_tokens,
        api_key_env="MINDCUBE_PROXY_PLACEHOLDER",
        opener=ProxyOpener(job["gateway"], loader,
                           digest({"arm": job["arm"], "input": job["item"]["input_sha256"]}),
                           job["run_identity"]))
    base = Path(job["budget_base"])
    policy = read(base / "smallimage/requests/state.json")["policy"]
    budget = DualBudget(
        RequestGovernor(job["budget_root"], RequestGovernorConfig(
            job["max_requests"], job.get("concurrency", 1))),
        RequestGovernor(base / "smallimage/requests", RequestGovernorConfig(**policy)),
        base, job["stop_file"], campaign=job["campaign"], arm=job["arm"],
        sample_id=job["item"]["sample_id"], run_identity=job["run_identity"])
    return solve(job, client, budget)


def main():
    job = json.load(sys.stdin)
    try:
        execute(job)
        return 0
    except BaseException as exc:
        write(job["stop_file"], {"reason": "worker_failure"})
        write(Path(job["output_directory"]) / "worker_failure.json",
              {"error_type": type(exc).__name__, "automatic_retry": False}, exclusive=True)
        mark_orphans(job["output_directory"], "worker_failure")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
