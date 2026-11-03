"""Post-run receipt audit; uses public traces and numeric accounting only."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
from pathlib import Path
import sys
import time

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "Ominirunner"

from .budget import audit as budget_audit
from .common import Config, IDS, VERSION, digest, read, sha, validate_source, write
from .data import load, pair_identity, problem
from .protocol import messages
from .native import ProgramFormatError, parse_program, program_feedback, scalar
from .tools import MEASUREMENTS, build_registry, empty_counts
from .transport import health
from .review import (final_payload, pending, priority_ids, state,
                     validate as validate_reviews, validate_program)
from .evidence import (EvidenceError, dependency_ids, feedback as evidence_feedback,
                       validate as validate_evidence)
from .reuse import request_cap, validate as validate_reuse
from . import report as report_module


def replay_contexts(item, result, rgb, frame, cache):
    """Reconstruct each public request without invoking a tool or a model."""
    from skill3d.schemas.tool import ToolResult
    from skill3d.schemas.tool_response import model_tool_response

    config = Config()
    observations, errors, note, review_ids = [], [], "", []
    reviews = {}
    calculation_ids = []
    counters = empty_counts()
    registry = build_registry(cache, problem(item), config)[0] if result["arm"] == "skill" else None

    def check_review_state(row):
        if (row.get("measurement_review_state") != state(reviews)
                or row.get("pending_measurement_ids") != pending(observations, reviews)
                or row.get("next_review_measurement_ids") != review_ids
                or row.get("calculation_dependency_ids") != calculation_ids):
            raise ValueError("round_surface_review_state_changed")

    for index, row in enumerate(result["rounds"]):
        if row["round"] != index or row["tool_counts_before"] != counters:
            raise ValueError("round_order_changed")
        expected, receipt = messages(
            item, rgb, frame, cache, arm=result["arm"], config=config, round_index=index,
            observations=observations, note=note, errors=errors, review_ids=review_ids,
            counters=counters, reviews=reviews, dependency_ids=calculation_ids)
        if row["request"] != receipt:
            raise ValueError("reconstructed_public_context_changed")
        if row["phase"] != receipt["phase"]:
            raise ValueError("phase_changed")
        yield index, expected
        if row["status"] == "program_parse_error":
            try:
                parse_program(row["output"])
            except ProgramFormatError as exc:
                errors = program_feedback(exc.code)
            else:
                raise ValueError("unexpected_parse_failure")
            if errors != row["feedback_errors"]:
                raise ValueError("parse_feedback_changed")
            check_review_state(row)
            continue
        if row["status"] in ("program_missing_action", "repairable_namespace_error"):
            errors = program_feedback("control_namespace_error" if
                                      row["status"] == "repairable_namespace_error"
                                      else "program_missing_action")
            if errors != row["feedback_errors"] or row.get("tool_receipts"):
                raise ValueError("nonexecuted_feedback_changed")
            check_review_state(row)
            continue
        review_errors = []
        final_raw, final_declared, parsed_final = None, None, False
        if result["arm"] == "skill" and (row.get("cell") or row["phase"] == "final"):
            accepted = {}
            if row["phase"] == "final":
                try:
                    final_raw, proposed, final_declared = final_payload(row["output"])
                    parsed_final = True
                    if final_declared != row.get("evidence_declaration"):
                        raise ValueError("final_evidence_declaration_changed")
                    accepted, review_errors = validate_reviews(
                        proposed, observations, receipt["visible_measurement_ids"])
                except (ValueError, TypeError):
                    if row["status"] != "invalid_answer":
                        raise
            else:
                accepted, review_errors = validate_program(
                    row["program"], observations, receipt["visible_measurement_ids"])
            if list(accepted.values()) != row.get("measurement_reviews", []):
                raise ValueError("surface_review_projection_changed")
            if review_errors != row.get("review_feedback_errors", []):
                raise ValueError("surface_review_feedback_changed")
            reviews.update(accepted)
        fresh = []
        for raw in row.get("tool_receipts", []):
            tool = ToolResult.model_validate(raw)
            # Signature binding failures happen before registry admission.
            if not tool.failure_cause or tool.failure_cause.code != "invalid_tool_arguments":
                counters["calls"] += 1
                counters["attempted"] += 1
                counters["measurements"] += int(tool.tool in MEASUREMENTS)
                counters["attempted_measurements"] += int(tool.tool in MEASUREMENTS)
            if tool.status == "ok" and not tool.error and not tool.invalidated_by:
                response = model_tool_response(tool)
                response["data"] = {k: v for k, v in response["data"].items()
                                    if k in registry.public_fields[tool.tool]}
                observations.append(response)
                if tool.tool in MEASUREMENTS:
                    fresh.append(tool.result_id)
        if row.get("cell"):
            review_ids = priority_ids(observations, reviews, fresh, row["cell"]["yielded_result_ids"])
        elif row["phase"] == "final":
            review_ids = priority_ids(observations, reviews, yielded=review_ids)
        if fresh != row.get("new_measurement_ids", []):
            raise ValueError("measurement_receipt_projection_changed")
        if "evidence_declaration" in row:
            # Saved JSON sorts mapping keys; the worker used the original
            # final-output binding order when choosing dependency priorities.
            declaration = final_declared if parsed_final else row["evidence_declaration"]
            calculation_ids = dependency_ids(declaration, observations, calculation_ids)
        check_review_state(row)
        cell = row.get("cell")
        answer_errors = []
        if result["arm"] == "skill" and (
                parsed_final or cell and cell["answer"] is not None
                and not cell["error_code"] and not cell["answer_untrusted"]):
            raw_answer = final_raw if parsed_final else row["submitted_native_answer"]
            try:
                declared_evidence = final_declared if parsed_final else row.get("evidence_declaration")
                checked = validate_evidence(raw_answer, declared_evidence,
                                            observations, reviews, receipt["visible_measurement_ids"])
            except (ValueError, TypeError) as exc:
                answer_errors = evidence_feedback(
                    exc if isinstance(exc, EvidenceError) else
                    str(exc) if type(exc) is ValueError else "answer_evidence_invalid")
                if row["status"] not in ("answer_evidence_required", "invalid_answer"):
                    raise ValueError("invalid_evidence_answer_accepted") from None
                # Native worker appends review feedback to the same list object
                # retained in answer_feedback_errors; final JSON does not.
                recorded_errors = answer_errors + review_errors if cell else answer_errors
                if recorded_errors != row.get("answer_feedback_errors", []):
                    raise ValueError("answer_feedback_changed")
            else:
                if row["status"] != "answered" or checked != row.get("answer_evidence"):
                    raise ValueError("answer_evidence_projection_changed")
                if (scalar(raw_answer) != result["answer"]
                        or checked != result["answer_evidence"]):
                    raise ValueError("episode_answer_evidence_changed")
        if cell:
            errors = []
            if row.get("quota_denials"):
                raise ValueError("unexpected_quota_denial")
            if counters != row["tool_counts_after"]:
                raise ValueError("admitted_tool_counts_changed")
            if cell["error_code"] or cell["answer_untrusted"]:
                errors = [{"code": cell["error_code"], "type": cell["error_type"],
                           "line": cell["error_line"]}]
            elif cell["answer"] is None:
                valid = {r["result_id"] for r in observations}
                if set(cell["yielded_result_ids"]) - valid:
                    errors = [{"code": "invalid_observation_reference"}]
            else:
                errors = answer_errors
            note = cell["yielded_reason"][:200]
            errors.extend(review_errors)
    if result["tool_counts"] != counters:
        raise ValueError("episode_tool_counts_changed")
    if result.get("context_text") != {
            "record_threshold": config.text_record_threshold,
            "max_chars": max((r["request"]["text_chars"] for r in result["rounds"]), default=0),
            "over_threshold_rounds": [r["round"] for r in result["rounds"]
                                     if r["request"]["text_over_threshold"]]}:
        raise ValueError("episode_context_threshold_changed")
    if (state(reviews) != result.get("measurement_reviews", [])
            or pending(observations, reviews) != result.get("pending_measurement_ids", [])
            or calculation_ids != result.get("calculation_dependency_ids", [])):
        raise ValueError("episode_surface_reviews_changed")


def run(root):
    frozen, inputs = read(root / "frozen.json"), read(root / "inputs.json")
    reused = frozen.get("reused_direct", {})
    if (frozen["schema"] != VERSION or frozen["config"] != Config().public()
            or frozen["max_requests"] != request_cap(reused)):
        raise ValueError("audit_contract_changed")
    validate_source(frozen["source"])
    completed = read(root / "completed.json")
    if (completed["status"] != "complete" or completed["episodes"] != 2*len(IDS)
            or tuple(item["sample_id"] for item in inputs) != IDS):
        raise ValueError("trial_incomplete")
    if (completed["run_identity"] != digest(frozen)
            or sha(root / "inputs.json") != frozen["inputs_sha256"]
            or read(root / "preflight.json")["run_identity"] != digest(frozen)):
        raise ValueError("trial_input_identity_changed")
    shared_root = Path(frozen["shared_budget_root"])
    before = read(root / "budget_at_launch.json")
    prefixes = []
    for initial in before["ledgers"]:
        state = read(shared_root.parents[1] / initial["ledger"] / "state.json")
        prefix = state["requests"][:initial["count"]]
        identity = digest([{k: r[k] for k in ("request_id", "category", "context", "started_at")}
                           for r in prefix])
        if identity != initial["admission_prefix_sha256"]:
            raise ValueError("historical_admission_prefix_changed")
        prefixes.append({"ledger": initial["ledger"], "prefix_count": len(prefix), "verified": True})
    local = read(root / "requests/state.json")
    shared = read(shared_root / "state.json")
    charged = [r for r in shared["requests"]
               if r.get("context", {}).get("campaign_id") == frozen["campaign"]]
    local_rows = local["requests"]
    if (len(local_rows) != len(charged) or len(charged) > frozen["max_requests"]
            or local["policy"]["max_requests"] != frozen["max_requests"]):
        raise ValueError("local_shared_charge_mismatch")
    for records in (local_rows, charged):
        if any(r["status"] != "completed" for r in records):
            raise ValueError("uncertain_or_failed_request")
    usage, historical_usage = Counter(), Counter()
    episodes, failures, receipts = [], [], []
    expected_charges = {}
    for item in inputs:
        rgb, frame, cache = load(item)
        for arm in ("direct", "skill"):
            directory = root / "episodes" / (item["sample_id"]+"-"+arm)
            result = read(directory / "result.json")
            reuse = reused.get(item["sample_id"]) if arm == "direct" else None
            expected_identity = reuse["source_run_identity"] if reuse else digest(frozen)
            if reuse:
                context, _ = messages(item, rgb, frame, None, arm="direct", config=Config())
                validate_reuse(reuse, item, context)
                if read(directory / "reuse.json") != reuse:
                    raise ValueError("reuse_copy_provenance_changed")
                for name, expected in reuse["files"].items():
                    if sha(directory / name) != expected:
                        raise ValueError("reuse_copy_changed")
            if (result["receipt_sha256"] != digest({
                    k: v for k, v in result.items() if k != "receipt_sha256"})
                    or result["pair_identity"] != pair_identity(item)
                    or result["run_identity"] != expected_identity
                    or result["reference_answer_visible"] or result["real_labels_available"]
                    or result["automatic_retry"] or result["sample_id"] != item["sample_id"]
                    or result["arm"] != arm):
                raise ValueError("episode_identity_changed")
            if (result["request_count"] != len(result["rounds"])
                    or not 1 <= result["request_count"] <= Config().request_limit(arm)):
                raise ValueError("episode_request_limit")
            if not reuse:
                expected_charges[(item["sample_id"], arm)] = (result["request_count"], result["usage"])
            receipts.append(result["receipt_sha256"])
            reconstructed = {0: context} if reuse else dict(
                replay_contexts(item, result, rgb, frame, cache))
            pending, facts, thinking_count, max_chars = set(), {}, 0, 0
            for row in result["rounds"]:
                index, receipt = row["round"], row["request"]
                delivered = set(receipt.get("reviewed_measurement_ids", []))
                if delivered - pending:
                    raise ValueError("review_delivery_not_pending")
                if (receipt["phase"] == "final" and len(receipt.get("visible_measurement_ids", [])) > 8
                        or len(receipt.get("rgb_crop_sheets", [])) > 2):
                    raise ValueError("evidence_delivery_not_bounded")
                payload = read(directory / f"{index:02d}.messages.json")
                parts = payload[0]["content"]
                if len(payload) != 1 or payload[0]["role"] != "user":
                    raise ValueError("history_accumulated")
                if digest(payload) != receipt["messages_sha256"]:
                    raise ValueError("request_payload_changed")
                if payload != reconstructed[index]:
                    raise ValueError("request_not_reconstructed_from_public_evidence")
                texts = [p["text"] for p in parts if p["type"] == "text"]
                if texts[0] != item["question"] or sum(t.count(item["question"]) for t in texts) != 1:
                    raise ValueError("official_question_changed")
                cap = receipt["text_cap"]
                if (receipt["text_chars"] != sum(map(len, texts))
                        or cap is not None and receipt["text_chars"] > cap):
                    raise ValueError("context_limit_broken")
                if not reuse and (
                        receipt["text_over_threshold"] != (receipt["text_chars"] > Config().text_record_threshold)
                        or receipt["text_over_threshold_chars"] != max(0, receipt["text_chars"]-Config().text_record_threshold)):
                    raise ValueError("context_threshold_record_changed")
                max_chars = max(max_chars, receipt["text_chars"])
                images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
                hashes = [hashlib.sha256(base64.b64decode(u.split(",", 1)[1])).hexdigest()
                          for u in images]
                if (len(images) != receipt["image_count"]
                        or hashes != receipt["image_sha256"] or hashes[0] != item["image_sha256"]):
                    raise ValueError("image_identity_changed")
                if (row["parameters"] != Config().kwargs()
                        or row["response_identity"]["model"] != Config().model
                        or not row["response_identity"]["model_reported"]):
                    raise ValueError("model_or_sampling_changed")
                thinking_count += int(bool(row.get("thinking")))
                if arm == "direct" and ("STATIC SKILL" in "\n".join(texts) or result["tool_counts"]["calls"]):
                    raise ValueError("baseline_extra_evidence")
                for r in row.get("tool_receipts", []):
                    if r["status"] == "ok" and not r["error"] and not r.get("invalidated_by"):
                        if not r["authorization"].get("allowed") or not r["result_id"]:
                            raise ValueError("invalid_tool_receipt")
                        facts[r["result_id"]] = r
                pending = set(row.get("pending_measurement_ids", pending))
                if row.get("status") in ("program_rejected", "run_error", "repairable_program_error",
                                         "repairable_namespace_error", "program_parse_error"):
                    cell = row.get("cell", {})
                    failures.append({
                        "sample_id": item["sample_id"], "arm": arm, "round": index,
                        "status": row["status"], "guard_violations": row.get("guard_violations", []),
                        "error_code": cell.get("error_code"), "error_type": cell.get("error_type"),
                        "feedback_code": next((e["code"] for e in row.get("feedback_errors", [])), "")})
            for key, value in result["usage"].items():
                (historical_usage if reuse else usage)[key] += value
            episodes.append({
                "sample_id": item["sample_id"], "arm": arm, "status": result["status"],
                "reused": bool(reuse), "source_run_identity": expected_identity,
                "request_count": result["request_count"], "max_text_chars": max_chars,
                "over_threshold_rounds": [r["round"] for r in result["rounds"]
                                          if r["request"]["text_chars"] > Config().text_record_threshold],
                "thinking_content_requests": thinking_count,
                "tool_calls": result["tool_counts"]["calls"],
                "rejected_calls": result["tool_counts"]["rejected"],
                "measurements": result["tool_counts"]["measurements"],
                "valid_measurements": sum(r["tool"] in MEASUREMENTS for r in facts.values()),
                "surface_review_decisions": dict(Counter(
                    r["decision"] for r in result.get("measurement_reviews", []))),
                "pending_surface_reviews": len(result.get("pending_measurement_ids", [])),
                "answer_basis": result["basis"],
                "verified_arithmetic": bool((result.get("answer_evidence") or {}).get("verified_arithmetic")),
                "round_statuses": dict(Counter(row["status"] for row in result["rounds"])),
                "receipt_sha256": result["receipt_sha256"]})
    for records in (charged, local_rows):
        expected = Counter()
        grouped = {}
        for row in records:
            expected.update(row.get("usage", {}))
            context = row["context"]
            if (row["category"] != "solver" or context["campaign_id"] != frozen["campaign"]
                    or context["run_identity"] != digest(frozen)):
                raise ValueError("request_charge_identity_changed")
            key = (context["slot_key"], context["variant_id"])
            group = grouped.setdefault(key, {"count": 0, "usage": Counter()})
            group["count"] += 1
            group["usage"].update(row.get("usage", {}))
        if dict(expected) != dict(usage):
            raise ValueError("numeric_usage_mismatch")
        actual = {key: (row["count"], dict(row["usage"])) for key, row in grouped.items()}
        if actual != expected_charges:
            raise ValueError("per_episode_charge_mismatch")
    if sorted(receipts) != sorted(completed["episode_receipt_sha256"]):
        raise ValueError("completed_receipt_set_changed")
    if sum(r["request_count"] for r in episodes if not r["reused"]) != len(charged):
        raise ValueError("request_denominator_mismatch")
    sys.path.insert(0, frozen["auth_module_root"])
    from transport_auth import load_transport_token
    service = health(frozen["gateway"], load_transport_token)
    if service != frozen["service_identity"]:
        raise ValueError("post_run_service_changed")
    after = budget_audit(shared_root.parents[1])
    value = {
        "schema": "omni-post-run-audit/9", "status": "passed", "at": time.time(),
        "run_identity": digest(frozen), "pairs": len(inputs), "episodes": episodes,
        "charged_requests": len(charged), "usage": dict(usage), "model": Config().model,
        "reused_direct": len(reused), "historical_usage": dict(historical_usage),
        "historical_requests": len(reused),
        "total_comparison_requests": len(charged) + len(reused),
        "peak_request_concurrency": local["peak_inflight"],
        "thinking_both_arms": True, "source_pins": len(frozen["source"]),
        "host_only_source": {
            "audit.py": sha(Path(__file__)), "report.py": sha(Path(report_module.__file__))},
        "historical_admission_prefixes": prefixes, "budget_after": after,
        "verified_original_rgb_both_arms": True, "verified_depth_grid_identity": True,
        "verified_next_request_review": True, "verified_fresh_bounded_contexts": True,
        "verified_same_model_surface_review_records": True,
        "verified_persisted_surface_review_state": True,
        "verified_priority_measurement_delivery": True,
        "verified_declared_answer_evidence": True,
        "verified_public_context_reconstruction": True,
        "verified_per_episode_request_and_usage_charges": True,
        "verified_request_and_usage_charges": True, "reference_answer_visible": False}
    value["program_failures"] = failures
    write(root / "audit.json", value, exclusive=True)
    print("audit passed:", len(episodes), "episodes;", len(charged), "charged requests", flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    run(args.root)
