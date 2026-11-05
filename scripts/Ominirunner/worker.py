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
                     parse_json, parse_program, pending_program as unexecuted_program,
                     program_feedback, scalar)
from .protocol import messages
from .tools import ALLOWED, MEASUREMENTS, build_registry, empty_counts
from .transport import AUTH_SHA, DualBudget, ProxyOpener
from .review import (apply_updates, final_payload, pending, priority_ids, state,
                     validate as validate_reviews, validate_program)
from .evidence import (EvidenceError, dependency_ids, feedback as evidence_feedback,
                       validate as validate_evidence)
from .answer_contract import specification, canonical
from .progress import activity, record as progress_record
from .recovery import cell_failure
from .submission import capture as rejected_submission
from .provenance import program_inputs

SCHEMA = "omni-label-free-main-thread-worker/13"


def solve(job, client, budget, *, jev_client=None):
    config, item = Config(), job["item"]
    arm, directory = job["arm"], Path(job["output_directory"])
    reviewer = job.get("measurement_reviewer")
    jev_policy, jev_receipts = None, []
    if reviewer is not None:
        from .jev import Client as JevClient, Policy, host_review, program_reviews
        jev_policy = Policy.parse(reviewer)
        if arm != "skill":
            raise ValueError("jev_requires_skill_arm")
        jev_client = jev_client or JevClient(jev_policy, budget)
    rgb, frame, cache = load(item, prepared=arm == "skill")
    observations, errors, note, review_ids, rounds = [], [], "", [], []
    reviews = {}
    calculation_ids = []
    last_submission = None
    pending_program = None
    answer_evidence = None
    task = specification(item)
    if reviewer is not None:
        task["geometry_verification"] = reviewer["version"]
    progress = []
    status, answer, raw_answer, kernel = "no_answer", None, None, None
    counters = empty_counts()
    if arm == "skill":
        registry, scene, counters = (build_registry(cache, problem(item), config, reviewer)
                                    if reviewer is not None else
                                    build_registry(cache, problem(item), config))
        kernel = NativeKernel(registry, scene, frames=[frame], episode_id=item["sample_id"],
                              cell_timeout_s=config.cell_timeout_s)
    started = time.monotonic()
    for index in range(config.request_limit(arm)):
        try:
            context, receipt = messages(
                item, rgb, frame, cache, arm=arm, config=config, round_index=index,
                observations=observations, note=note, errors=errors, review_ids=review_ids,
                counters=counters, reviews=reviews, dependency_ids=calculation_ids, progress=progress,
                last_submission=last_submission, pending_program=pending_program,
                measurement_reviewer=reviewer)
        except ValueError:
            status = "context_limit"
            break
        if index == 0 and receipt != job["initial_receipt"]:
            raise ValueError("initial_input_changed")
        final = receipt["phase"] == "final"
        if kernel:
            scene.delivered_images = {r["image_id"]: r for r in receipt["image_coordinates"]}
            scene.delivered_measurement_ids = set(receipt["visible_measurement_ids"])
            scene.delivered_reselect_ids = set(receipt.get("reselect_measurement_ids", []))
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
        previous_observations, previous_reviews = list(observations), dict(reviews)

        def close_round():
            if arm == "skill":
                row["last_submission_state"] = last_submission
                row["pending_program_state"] = pending_program
            change = progress_record(previous_observations, observations, previous_reviews,
                                     reviews, row["status"] == "answered")
            row["progress"] = change
            progress.append(change)
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
            close_round()
            break
        if final:
            try:
                if arm == "skill":
                    raw_answer, submitted, declared = final_payload(text)
                    if reviewer is not None and submitted:
                        raise ValueError("host_review_only")
                    row["evidence_declaration"] = declared
                    prior_dependencies = list(calculation_ids)
                    calculation_ids = dependency_ids(declared, observations, calculation_ids)
                    accepted, review_errors = ({}, []) if reviewer is not None else validate_reviews(
                        submitted, observations, receipt["visible_measurement_ids"])
                    apply_updates(reviews, accepted, observations)
                    row["measurement_reviews"] = list(accepted.values())
                    row["review_feedback_errors"] = review_errors
                    review_ids = priority_ids(observations, reviews, yielded=review_ids)
                    answer_evidence = validate_evidence(
                        raw_answer, declared, observations, reviews, receipt["visible_measurement_ids"], task,
                        visible_result_ids=receipt["visible_result_ids"],
                        previous_dependency_ids=prior_dependencies, measurement_reviewer=reviewer)
                    row["answer_evidence"] = answer_evidence
                else:
                    raw_answer = parse_json(text)
                    answer_evidence = {"kind": "visual_estimate", "basis": "original RGB",
                                       "verified_arithmetic": False}
                answer = scalar(canonical(raw_answer, task) if arm == "skill" else raw_answer)
                last_submission = None
                status = "answered"
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
            close_round()
            write(directory / f"{index:02d}.execution.json", row, exclusive=True)
            break
        try:
            program = parse_program(text)
        except ProgramFormatError as exc:
            errors = program_feedback(exc)
            pending_program = unexecuted_program(text, exc)
            row["feedback_errors"] = errors
            row["status"] = "program_parse_error"
            row["executed"] = False
            close_round()
            continue
        pending_program = None
        row["program"] = program
        guarded = ast_guard(program, allowed_tools=ALLOWED)
        wrong_controls = control_namespace_errors(program)
        namespace_violations = {f"Unknown tool in the tools namespace: {name}"
                                for name in wrong_controls}
        security_violations = [v for v in guarded.violations if v not in namespace_violations]
        if security_violations or "ReadObservation" in program or "AnswerPayload" in program:
            row.update(status="program_rejected", guard_violations=guarded.violations)
            status = "program_rejected"
            close_round()
            break
        if wrong_controls:
            errors = program_feedback("control_namespace_error")
            row.update(status="repairable_namespace_error", feedback_errors=errors,
                       executed=False, guard_violations=guarded.violations)
            close_round()
            continue
        if not check_action(program, tool_names=ALLOWED).ok:
            row["status"] = "program_missing_action"
            errors = program_feedback("program_missing_action")
            row["feedback_errors"] = errors
            close_round()
            continue
        errors = []
        accepted, review_errors = (program_reviews(program) if reviewer is not None else
                                  validate_program(program, observations, receipt["visible_measurement_ids"]))
        kernel.reset_user_namespace()
        cell = kernel.run_cell(program, yield_on_fallthrough=True)
        apply_updates(reviews, accepted, observations)
        row["measurement_reviews"] = list(accepted.values())
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
        apply_updates(reviews, {}, observations)
        row["tool_receipts"] = [r.model_dump() for r in kernel.cell_results]
        row["tool_counts_after"] = dict(counters)
        row["new_measurement_ids"] = fresh
        jev_failed = False
        if reviewer is not None:
            row["jev_receipts"] = []
            known = {r["result_id"]: r for r in observations}
            for key in fresh:
                # New facts cannot inherit an old binary decision, even when
                # the proposed object or pixel looks identical.
                receipt_jev = jev_client.review(known[key], rgb)
                update = host_review(known[key], receipt_jev, rgb, jev_policy)
                row["jev_receipts"].append(receipt_jev)
                jev_receipts.append(receipt_jev)
                apply_updates(reviews, {key: update}, observations)
                if receipt_jev["status"] != "reviewed":
                    jev_failed = True
                    write(Path(job["stop_file"]), {"reason": "jev_service_or_budget_deviation"})
                    break
        row["measurement_review_state"] = state(reviews)
        review_ids = priority_ids(observations, reviews, fresh, cell.yielded_result_ids)
        row["next_review_measurement_ids"] = list(review_ids)
        row["pending_measurement_ids"] = pending(observations, reviews)
        note = cell.yielded_reason[:200]
        stop_episode = False
        if jev_failed:
            status, row["status"], stop_episode = "service_error", "service_error", True
        elif cell.error_code or cell.answer_untrusted:
            errors, fatal = cell_failure(row["cell"], row["tool_receipts"], registry)
            row["feedback_errors"] = errors
            if fatal:
                status, row["status"] = "run_error", "run_error"
                stop_episode = True
            else:
                row["status"] = ("repairable_tool_arguments" if cell.error_code == "tool_contract"
                                 else "repairable_program_error")
        elif cell.answer is not None:
            proposed = kernel.answer_slot.raw_value
            row["submitted_native_answer"] = proposed
            row["evidence_declaration"] = kernel.answer_slot.evidence
            prior_dependencies = list(calculation_ids)
            input_origin = program_inputs(program, observations, fresh)
            row["program_input_origin"] = input_origin
            calculation_ids = dependency_ids(kernel.answer_slot.evidence, observations, calculation_ids)
            try:
                answer_evidence = validate_evidence(
                    proposed, kernel.answer_slot.evidence, observations, reviews,
                    receipt["visible_measurement_ids"], task,
                    visible_result_ids=receipt["visible_result_ids"],
                    previous_dependency_ids=prior_dependencies,
                    program_dependency_ids=input_origin["result_ids"], measurement_reviewer=reviewer)
            except (ValueError, TypeError) as exc:
                errors = evidence_feedback(
                    exc if isinstance(exc, EvidenceError) else
                    str(exc) if type(exc) is ValueError else "answer_evidence_invalid")
                row.update(status="answer_evidence_required", provisional_answer=cell.answer,
                           answer_feedback_errors=errors)
                last_submission = rejected_submission(
                    index, proposed, kernel.answer_slot.evidence, errors)
            else:
                raw_answer, answer = proposed, scalar(canonical(proposed, task))
                last_submission = None
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
        close_round()
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
        "public_answer_contract": task if arm == "skill" else None,
        "progress": progress,
        "evidence_activity": activity(observations, reviews, progress),
        "context_text": {
            "record_threshold": config.text_record_threshold,
            "max_chars": max((r["request"]["text_chars"] for r in rounds), default=0),
            "over_threshold_rounds": [r["round"] for r in rounds
                                     if r["request"]["text_over_threshold"]]},
        "basis": answer_evidence["kind"] if answer_evidence else "unanswered",
        "elapsed_seconds": round(time.monotonic()-started, 6),
        "reference_answer_visible": False, "real_labels_available": False,
        "automatic_retry": False, "main_thread_kernel": True}
    if arm == "skill":
        result["last_submission_state"] = last_submission
        result["pending_program_state"] = pending_program
    if reviewer is not None:
        result["measurement_reviewer"] = reviewer
        result["jev_request_count"] = sum(r["request_attempted"] for r in jev_receipts)
        result["jev_usage"] = {k: sum(r["usage"].get(k, 0) for r in jev_receipts)
                               for k in ("input_tokens", "output_tokens")}
        result["total_request_count"] = len(rounds) + result["jev_request_count"]
    result["receipt_sha256"] = digest(result)
    write(directory / "result.json", result, exclusive=True)
    return result


def execute(job):
    runtime_secret = job.pop("_jev_secret", None)
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
    reviewer = job.get("measurement_reviewer")
    extra_cap = 0
    if reviewer is not None:
        from .jev import Policy
        extra_cap = Policy.parse(reviewer).max_requests
        if job["arm"] != "skill" or not {
                "scripts/Ominirunner/jev.py", "scripts/Ominirunner/jev_protocol.py",
                "scripts/Ominirunner/jev_trial.py",
                "scripts/Ominirunner/static_skill_host.txt"} <= set(job["source"]):
            raise ValueError("jev_worker_contract_changed")
        if (reviewer["provider"] == "intern"
                and "scripts/Ominirunner/intern_review.py" not in job["source"]):
            raise ValueError("intern_source_pin_missing")
    batch = job.get("batch_frozen")
    if batch is not None:
        from .common import selected_ids
        from .reuse import request_cap
        ids = selected_ids(batch)
        if (reviewer is None or batch.get("measurement_reviewer") != reviewer
                or digest(batch) != job["run_identity"] or batch["source"] != job["source"]
                or batch["config"] != job["config"] or batch["campaign"] != job["campaign"]
                or job["item"]["sample_id"] not in ids
                or batch["max_requests"] != request_cap(batch["reused_direct"], ids, reviewer)
                or any(job[k] != batch[k] for k in (
                    "gateway", "auth_module_root", "shared_budget_root",
                    "local_budget_policy", "shared_budget_policy"))):
            raise ValueError("host_batch_worker_contract_changed")
        pair = next(p for p in batch["pairs"] if p["sample_id"] == job["item"]["sample_id"])
        if pair["pair_identity"] != pair_identity(job["item"]) or pair["initial_receipts"]["skill"] != job["initial_receipt"]:
            raise ValueError("host_batch_worker_input_changed")
        extra_cap *= len(ids)
    local_policy = job.get("local_budget_policy", {
        "max_requests": Config().trial_request_cap, "max_inflight": 2, "requests_per_second": 0.0})
    shared_policy = job.get("shared_budget_policy", {
        "max_requests": 20000, "max_inflight": 4, "requests_per_second": 0.0})
    concurrency(local_policy["max_inflight"])
    concurrency(shared_policy["max_inflight"])
    if (local_policy["max_requests"] > Config().trial_request_cap + extra_cap
            or shared_policy["max_requests"] != 20000
            or local_policy["requests_per_second"] != 0
            or shared_policy["requests_per_second"] != 0):
        raise ValueError("worker_budget_policy_changed")
    budget = DualBudget(
        RequestGovernor(job["budget_root"], RequestGovernorConfig(**local_policy)),
        RequestGovernor(job["shared_budget_root"], RequestGovernorConfig(**shared_policy)),
        job["stop_file"], job["campaign"], job["arm"], job["item"]["sample_id"], job["run_identity"],
        reserved_cap=batch["max_requests"] if batch else None,
        aggregate_policy=batch.get("aggregate_budget_policy") if batch else None)
    jev_client = None
    if reviewer is not None and runtime_secret is not None:
        from .jev import Client as JevClient
        if not isinstance(runtime_secret, str) or not runtime_secret:
            raise ValueError("jev_credential_unavailable")
        jev_client = JevClient(Policy.parse(reviewer), budget, loader=lambda policy: runtime_secret)
    return solve(job, client, budget, jev_client=jev_client)


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
