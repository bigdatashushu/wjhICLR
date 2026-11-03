"""One label-free episode on a subprocess's main thread."""
from __future__ import annotations

import base64
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "Ominirunner"

from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.action_check import check_action
from skill3d.schemas.tool_response import model_tool_response
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

from .common import Config, VERSION, concurrency, digest, read, sha, validate_source, write
from .data import load, pair_identity, problem
from .native import (NativeKernel, ProgramFormatError, control_namespace_errors,
                     parse_json, parse_program, program_feedback, scalar)
from .protocol import messages
from .tools import ALLOWED, MEASUREMENTS, build_registry, empty_counts
from .transport import AUTH_SHA, DualBudget, ProxyOpener
from .review import (final_payload, pending, priority_ids, state,
                     validate as validate_reviews, validate_program)
from .evidence import (EvidenceError, dependency_ids, feedback as evidence_feedback,
                       validate as validate_evidence)

SCHEMA = "omni-label-free-main-thread-worker/9"


def solve(job, client, budget):
    config, item = Config(), job["item"]
    arm, directory = job["arm"], Path(job["output_directory"])
    rgb, frame, cache = load(item, prepared=arm == "skill")
    observations, errors, note, review_ids, rounds = [], [], "", [], []
    reviews = {}
    calculation_ids = []
    answer_evidence = None
    status, answer, raw_answer, kernel = "no_answer", None, None, None
    counters = empty_counts()
    if arm == "skill":
        registry, scene, counters = build_registry(cache, problem(item), config)
        kernel = NativeKernel(registry, scene, frames=[frame], episode_id=item["sample_id"],
                              cell_timeout_s=config.cell_timeout_s)
    started = time.monotonic()
    for index in range(config.request_limit(arm)):
        try:
            context, receipt = messages(
                item, rgb, frame, cache, arm=arm, config=config, round_index=index,
                observations=observations, note=note, errors=errors, review_ids=review_ids,
                counters=counters, reviews=reviews, dependency_ids=calculation_ids)
        except ValueError:
            status = "context_limit"
            break
        if index == 0 and receipt != job["initial_receipt"]:
            raise ValueError("initial_input_changed")
        final = receipt["phase"] == "final"
        if kernel and final:
            kernel.set_tools_enabled(False)
        row = {"round": index, "request": receipt, "parameters": config.kwargs(),
               "phase": receipt["phase"], "tool_counts_before": dict(counters),
               "measurement_review_state": state(reviews),
               "pending_measurement_ids": pending(observations, reviews),
               "next_review_measurement_ids": list(review_ids),
               "calculation_dependency_ids": list(calculation_ids),
               "status": "intent", "automatic_retry": False}
        rounds.append(row)
        # Images and full prompts are task-owned host traces, never ledger fields.
        write(directory / f"{index:02d}.messages.json", context, exclusive=True)
        for n, part in enumerate(p for p in context[0]["content"] if p["type"] == "image_url"):
            url = part["image_url"]["url"]
            if url.startswith("data:image/png"):
                image_name = "depth" if n == 1 else f"review-image-{n}"
                (directory / f"{index:02d}.{image_name}.png").write_bytes(
                    base64.b64decode(url.split(",", 1)[1]))
        write(directory / f"{index:02d}.intent.json", row, exclusive=True)
        try:
            with budget.request() as usage_receipt:
                text = client.chat(context, **config.kwargs())
                usage = {k: v for k, v in client.last_usage.items()
                         if k in {"prompt_tokens", "completion_tokens", "total_tokens"}
                         and type(v) in (int, float) and math.isfinite(v)}
                usage_receipt["usage"] = usage
                actual = client.last_response_identity
                if actual.get("model") != config.model or actual.get("model_reported") is not True:
                    raise ValueError("actual_model_changed")
                row.update(status="received", usage=usage, response_identity=actual,
                           generation=client.last_generation, output=text,
                           thinking=client.last_reasoning_content)
                write(directory / f"{index:02d}.response.json", row, exclusive=True)
        except BaseException as exc:
            # Uncertain inference is retained; never re-enter this episode.
            row.update(status="service_error", error_type=type(exc).__name__,
                       service_error=client.last_service_error)
            write(Path(job["stop_file"]), {"reason": "service_or_budget_deviation"})
            status = "service_error"
            break
        if final:
            try:
                if arm == "skill":
                    raw_answer, submitted, declared = final_payload(text)
                    row["evidence_declaration"] = declared
                    calculation_ids = dependency_ids(declared, observations, calculation_ids)
                    accepted, review_errors = validate_reviews(
                        submitted, observations, receipt["visible_measurement_ids"])
                    reviews.update(accepted)
                    row["measurement_reviews"] = list(accepted.values())
                    row["review_feedback_errors"] = review_errors
                    review_ids = priority_ids(observations, reviews, yielded=review_ids)
                    answer_evidence = validate_evidence(
                        raw_answer, declared, observations, reviews, receipt["visible_measurement_ids"])
                    row["answer_evidence"] = answer_evidence
                else:
                    raw_answer = parse_json(text)
                    answer_evidence = {"kind": "visual_estimate", "basis": "original RGB",
                                       "verified_arithmetic": False}
                answer, status = scalar(raw_answer), "answered"
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                status = "invalid_answer"
                row["answer_feedback_errors"] = evidence_feedback(
                    exc if isinstance(exc, EvidenceError) else
                    str(exc) if type(exc) is ValueError else "final_answer_invalid")
            row.update(status=status, answer=answer, raw_answer=raw_answer,
                       measurement_review_state=state(reviews),
                       pending_measurement_ids=pending(observations, reviews),
                       next_review_measurement_ids=list(review_ids))
            row["calculation_dependency_ids"] = list(calculation_ids)
            write(directory / f"{index:02d}.execution.json", row, exclusive=True)
            break
        try:
            program = parse_program(text)
        except ProgramFormatError as exc:
            errors = program_feedback(exc.code)
            row["feedback_errors"] = errors
            row["status"] = "program_parse_error"
            continue
        row["program"] = program
        guarded = ast_guard(program, allowed_tools=ALLOWED)
        wrong_controls = control_namespace_errors(program)
        namespace_violations = {f"Unknown tool in the tools namespace: {name}"
                                for name in wrong_controls}
        security_violations = [v for v in guarded.violations if v not in namespace_violations]
        if security_violations or "ReadObservation" in program or "AnswerPayload" in program:
            row.update(status="program_rejected", guard_violations=guarded.violations)
            status = "program_rejected"
            break
        if wrong_controls:
            errors = program_feedback("control_namespace_error")
            row.update(status="repairable_namespace_error", feedback_errors=errors,
                       executed=False, guard_violations=guarded.violations)
            continue
        if not check_action(program, tool_names=ALLOWED).ok:
            row["status"] = "program_missing_action"
            errors = program_feedback("program_missing_action")
            row["feedback_errors"] = errors
            continue
        errors = []
        accepted, review_errors = validate_program(
            program, observations, receipt["visible_measurement_ids"])
        kernel.reset_user_namespace()
        cell = kernel.run_cell(program, yield_on_fallthrough=True)
        reviews.update(accepted)
        row["measurement_reviews"] = list(accepted.values())
        row["measurement_review_state"] = state(reviews)
        row["cell"] = asdict(cell)
        fresh = []
        for tool_result in kernel.cell_results:
            if tool_result.status == "ok" and not tool_result.error and not tool_result.invalidated_by:
                response = model_tool_response(tool_result)
                response["data"] = {k: v for k, v in response["data"].items()
                                    if k in registry.public_fields[tool_result.tool]}
                observations.append(response)
                if tool_result.tool in MEASUREMENTS:
                    fresh.append(tool_result.result_id)
        row["tool_receipts"] = [r.model_dump() for r in kernel.cell_results]
        row["tool_counts_after"] = dict(counters)
        row["new_measurement_ids"] = fresh
        review_ids = priority_ids(observations, reviews, fresh, cell.yielded_result_ids)
        row["next_review_measurement_ids"] = list(review_ids)
        row["pending_measurement_ids"] = pending(observations, reviews)
        note = cell.yielded_reason[:200]
        stop_episode = False
        if cell.error_code or cell.answer_untrusted:
            errors = [{"code": cell.error_code, "type": cell.error_type, "line": cell.error_line}]
            if cell.answer_untrusted or cell.error_code in ("timeout", "tool_contract"):
                status, row["status"] = "run_error", "run_error"
                stop_episode = True
            else:
                row["status"] = "repairable_program_error"
        elif cell.answer is not None:
            proposed = kernel.answer_slot.raw_value
            row["submitted_native_answer"] = proposed
            row["evidence_declaration"] = kernel.answer_slot.evidence
            calculation_ids = dependency_ids(kernel.answer_slot.evidence, observations, calculation_ids)
            try:
                answer_evidence = validate_evidence(
                    proposed, kernel.answer_slot.evidence, observations, reviews,
                    receipt["visible_measurement_ids"])
            except (ValueError, TypeError) as exc:
                errors = evidence_feedback(
                    exc if isinstance(exc, EvidenceError) else
                    str(exc) if type(exc) is ValueError else "answer_evidence_invalid")
                row.update(status="answer_evidence_required", provisional_answer=cell.answer,
                           answer_feedback_errors=errors)
            else:
                raw_answer, answer = proposed, cell.answer
                row["answer_evidence"] = answer_evidence
                status, row["status"] = "answered", "answered"
                stop_episode = True
        else:
            valid = {r["result_id"] for r in observations}
            if set(cell.yielded_result_ids) - valid:
                errors = [{"code": "invalid_observation_reference"}]
            row["status"] = "yielded" if cell.terminated == "yield" else "no_answer"
        errors.extend(review_errors)
        if review_errors:
            row["review_feedback_errors"] = review_errors
        row["calculation_dependency_ids"] = list(calculation_ids)
        write(directory / f"{index:02d}.execution.json", row, exclusive=True)
        if stop_episode:
            break
    usage = {}
    for r in rounds:
        for key, value in r.get("usage", {}).items():
            usage[key] = usage.get(key, 0) + value
    result = {
        "schema": VERSION, "sample_id": item["sample_id"], "arm": arm,
        "pair_identity": pair_identity(item), "run_identity": job["run_identity"],
        "status": status, "answer": answer, "raw_answer": raw_answer, "usage": usage,
        "request_count": len(rounds), "tool_counts": counters, "rounds": rounds,
        "measurement_reviews": state(reviews),
        "pending_measurement_ids": pending(observations, reviews),
        "calculation_dependency_ids": list(calculation_ids),
        "answer_evidence": answer_evidence,
        "context_text": {
            "record_threshold": config.text_record_threshold,
            "max_chars": max((r["request"]["text_chars"] for r in rounds), default=0),
            "over_threshold_rounds": [r["round"] for r in rounds
                                     if r["request"]["text_over_threshold"]]},
        "basis": answer_evidence["kind"] if answer_evidence else "unanswered",
        "elapsed_seconds": round(time.monotonic()-started, 6),
        "reference_answer_visible": False, "real_labels_available": False,
        "automatic_retry": False, "main_thread_kernel": True}
    result["receipt_sha256"] = digest(result)
    write(directory / "result.json", result, exclusive=True)
    return result


def execute(job):
    if job["schema"] != SCHEMA or job["config"] != Config().public():
        raise ValueError("worker_contract_changed")
    validate_source(job["source"])
    auth_root = Path(job["auth_module_root"])
    if sha(auth_root / "transport_auth.py") != AUTH_SHA:
        raise ValueError("transport_auth_changed")
    sys.path.insert(0, str(auth_root))
    from transport_auth import load_transport_token
    os.environ["OMNI_PROXY_PLACEHOLDER"] = "local-proxy-configuration"
    client = BailianClient(
        timeout_s=600, max_output_tokens=8192, api_key_env="OMNI_PROXY_PLACEHOLDER",
        opener=ProxyOpener(job["gateway"], load_transport_token,
                           digest({"arm": job["arm"], "input": job["item"]["input_sha256"]}),
                           job["run_identity"]))
    local_policy = job.get("local_budget_policy", {
        "max_requests": Config().trial_request_cap, "max_inflight": 2, "requests_per_second": 0.0})
    shared_policy = job.get("shared_budget_policy", {
        "max_requests": 20000, "max_inflight": 4, "requests_per_second": 0.0})
    concurrency(local_policy["max_inflight"])
    concurrency(shared_policy["max_inflight"])
    if (local_policy["max_requests"] > Config().trial_request_cap
            or shared_policy["max_requests"] != 20000
            or local_policy["requests_per_second"] != 0
            or shared_policy["requests_per_second"] != 0):
        raise ValueError("worker_budget_policy_changed")
    budget = DualBudget(
        RequestGovernor(job["budget_root"], RequestGovernorConfig(**local_policy)),
        RequestGovernor(job["shared_budget_root"], RequestGovernorConfig(**shared_policy)),
        job["stop_file"], job["campaign"], job["arm"], job["item"]["sample_id"], job["run_identity"])
    return solve(job, client, budget)


def main():
    job = json.load(sys.stdin)
    try:
        execute(job)
        return 0
    except BaseException as exc:
        write(Path(job["stop_file"]), {"reason": "worker_failure"})
        write(Path(job["output_directory"]) / "worker_failure.json", {
            "error_type": type(exc).__name__, "automatic_retry": False,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)]}, exclusive=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
