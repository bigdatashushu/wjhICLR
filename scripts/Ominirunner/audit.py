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
from .native import ProgramFormatError, parse_program, program_feedback
from .tools import MEASUREMENTS, build_registry, empty_counts, remaining
from .transport import health


def replay_contexts(item, result, rgb, frame, cache):
    """Reconstruct each public request without invoking a tool or a model."""
    from skill3d.schemas.tool import ToolResult
    from skill3d.schemas.tool_response import model_tool_response

    config = Config()
    observations, errors, note, review_ids = [], [], "", []
    counters = empty_counts()
    registry = build_registry(cache, problem(item), config)[0] if result["arm"] == "skill" else None
    for index, row in enumerate(result["rounds"]):
        if row["round"] != index or row["tool_counts_before"] != counters:
            raise ValueError("round_order_changed")
        expected, receipt = messages(
            item, rgb, frame, cache, arm=result["arm"], config=config, round_index=index,
            observations=observations, note=note, errors=errors, review_ids=review_ids,
            counters=counters)
        if row["request"] != receipt:
            raise ValueError("reconstructed_public_context_changed")
        if row["phase"] != receipt["phase"]:
            raise ValueError("phase_changed")
        yield index, expected
        review_ids = []
        if row["status"] == "program_parse_error":
            try:
                parse_program(row["output"])
            except ProgramFormatError as exc:
                errors = program_feedback(exc.code)
            else:
                raise ValueError("unexpected_parse_failure")
            if errors != row["feedback_errors"]:
                raise ValueError("parse_feedback_changed")
            continue
        if row["status"] in ("program_missing_action", "repairable_namespace_error"):
            errors = program_feedback("control_namespace_error" if
                                      row["status"] == "repairable_namespace_error"
                                      else "program_missing_action")
            if errors != row["feedback_errors"] or row.get("tool_receipts"):
                raise ValueError("nonexecuted_feedback_changed")
            continue
        for raw in row.get("tool_receipts", []):
            tool = ToolResult.model_validate(raw)
            # Signature binding failures happen before registry admission.
            if not tool.failure_cause or tool.failure_cause.code != "invalid_tool_arguments":
                if (counters["calls"] >= config.max_tool_calls
                        or tool.tool in MEASUREMENTS and
                        counters["measurements"] >= config.max_measurements):
                    raise ValueError("tool_receipt_above_episode_quota")
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
                    review_ids.append(tool.result_id)
        if review_ids != row.get("new_measurement_ids", []):
            raise ValueError("measurement_receipt_projection_changed")
        cell = row.get("cell")
        if cell:
            errors = []
            for denial in row.get("quota_denials", []):
                measure = int(denial["tool"] in MEASUREMENTS)
                reason = ("tool_calls_exhausted" if counters["calls"] >= config.max_tool_calls
                          else "measurements_exhausted" if measure and
                          counters["measurements"] >= config.max_measurements else "")
                if (not reason or denial["reason"] != reason or denial["admitted"] is not False
                        or denial["argument_digest"] != digest(denial["args"])
                        or denial["remaining"] != remaining(config, counters)
                        or "result_id" in denial):
                    raise ValueError("quota_denial_changed")
                counters["attempted"] += 1
                counters["attempted_measurements"] += measure
                counters["rejected"] += 1
                counters["rejected_measurements"] += measure
            if counters != row["tool_counts_after"]:
                raise ValueError("admitted_tool_counts_changed")
            if cell["error_code"] or cell["answer_untrusted"]:
                errors = [{"code": cell["error_code"], "type": cell["error_type"],
                           "line": cell["error_line"]}]
            elif row.get("quota_denials"):
                errors = [{"code": "quota_exhausted",
                           "reason": row["quota_denials"][-1]["reason"],
                           "repair": "Review delivered successful evidence and submit final JSON."}]
            elif cell["answer"] is None:
                valid = {r["result_id"] for r in observations}
                if set(cell["yielded_result_ids"]) - valid:
                    errors = [{"code": "invalid_observation_reference"}]
            note = cell["yielded_reason"][:200]
    if result["tool_counts"] != counters:
        raise ValueError("episode_tool_counts_changed")


def run(root):
    frozen, inputs = read(root / "frozen.json"), read(root / "inputs.json")
    if (frozen["schema"] != VERSION or frozen["config"] != Config().public()
            or frozen["max_requests"] != Config().trial_request_cap):
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
    usage = Counter()
    episodes, failures, receipts = [], [], []
    expected_charges = {}
    for item in inputs:
        rgb, frame, cache = load(item)
        for arm in ("direct", "skill"):
            directory = root / "episodes" / (item["sample_id"]+"-"+arm)
            result = read(directory / "result.json")
            if (result["receipt_sha256"] != digest({
                    k: v for k, v in result.items() if k != "receipt_sha256"})
                    or result["pair_identity"] != pair_identity(item)
                    or result["run_identity"] != digest(frozen)
                    or result["reference_answer_visible"] or result["real_labels_available"]
                    or result["automatic_retry"] or result["sample_id"] != item["sample_id"]
                    or result["arm"] != arm):
                raise ValueError("episode_identity_changed")
            if (result["request_count"] != len(result["rounds"])
                    or not 1 <= result["request_count"] <= Config().request_limit(arm)):
                raise ValueError("episode_request_limit")
            expected_charges[(item["sample_id"], arm)] = (result["request_count"], result["usage"])
            receipts.append(result["receipt_sha256"])
            reconstructed = dict(replay_contexts(item, result, rgb, frame, cache))
            pending, facts, thinking_count, max_chars = set(), {}, 0, 0
            for row in result["rounds"]:
                index, receipt = row["round"], row["request"]
                delivered = set(receipt["reviewed_measurement_ids"])
                if pending - delivered:
                    raise ValueError("missing_next_request_measurement_review")
                pending.clear()
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
                if receipt["text_chars"] != sum(map(len, texts)) or receipt["text_chars"] > receipt["text_cap"]:
                    raise ValueError("context_limit_broken")
                max_chars = max(max_chars, receipt["text_chars"])
                images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
                hashes = [hashlib.sha256(base64.b64decode(u.split(",", 1)[1])).hexdigest()
                          for u in images]
                if (len(images) != (1 if arm == "direct" else 2)
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
                pending.update(row.get("new_measurement_ids", []))
                if row.get("status") == "answered" and pending:
                    raise ValueError("unreviewed_measurement_accepted")
                if row.get("status") in ("program_rejected", "run_error", "repairable_program_error",
                                         "repairable_namespace_error", "program_parse_error",
                                         "quota_exhausted"):
                    cell = row.get("cell", {})
                    failures.append({
                        "sample_id": item["sample_id"], "arm": arm, "round": index,
                        "status": row["status"], "guard_violations": row.get("guard_violations", []),
                        "error_code": cell.get("error_code"), "error_type": cell.get("error_type"),
                        "feedback_code": next((e["code"] for e in row.get("feedback_errors", [])), ""),
                        "quota_reason": next((d["reason"] for d in row.get("quota_denials", [])), ""),
                        "quota_rejection": bool(row.get("quota_denials"))})
            if result["status"] == "answered" and pending:
                raise ValueError("measurement_review_missing_at_answer")
            for key, value in result["usage"].items():
                usage[key] += value
            episodes.append({
                "sample_id": item["sample_id"], "arm": arm, "status": result["status"],
                "request_count": result["request_count"], "max_text_chars": max_chars,
                "thinking_content_requests": thinking_count,
                "tool_calls": result["tool_counts"]["calls"],
                "rejected_calls": result["tool_counts"]["rejected"],
                "measurements": result["tool_counts"]["measurements"],
                "valid_measurements": sum(r["tool"] in MEASUREMENTS for r in facts.values()),
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
    if sum(r["request_count"] for r in episodes) != len(charged):
        raise ValueError("request_denominator_mismatch")
    sys.path.insert(0, frozen["auth_module_root"])
    from transport_auth import load_transport_token
    service = health(frozen["gateway"], load_transport_token)
    if service != frozen["service_identity"]:
        raise ValueError("post_run_service_changed")
    after = budget_audit(shared_root.parents[1])
    value = {
        "schema": "omni-post-run-audit/2", "status": "passed", "at": time.time(),
        "run_identity": digest(frozen), "pairs": len(inputs), "episodes": episodes,
        "charged_requests": len(charged), "usage": dict(usage), "model": Config().model,
        "thinking_both_arms": True, "source_pins": len(frozen["source"]),
        "host_only_source": {
            name: sha(Path(__file__).parent / name) for name in ("audit.py", "report.py")},
        "historical_admission_prefixes": prefixes, "budget_after": after,
        "verified_original_rgb_both_arms": True, "verified_depth_grid_identity": True,
        "verified_next_request_review": True, "verified_fresh_bounded_contexts": True,
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
