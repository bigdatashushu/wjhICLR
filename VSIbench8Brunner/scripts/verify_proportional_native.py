#!/usr/bin/env python3
"""Verify completed server-native traces and export small label-free score inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_native_continuation(root, selected, admission):
    directory = root / "harness"
    recovery = root / "recovery_r02"
    plan, started, status = [read(recovery / name) for name in
                             ("plan.json", "started.json", "status.json")]
    if (status["status"] != "complete" or status["pending_solves_completed"] != 37
            or status["preserved_native_questions"] != 63
            or status["issued_model_requests_repeated"] != 0
            or (recovery / "execute_failed.json").exists()
            or sha(recovery / "plan.json") != started["plan_sha256"]
            or plan["sampling_sha256"] != sha(root / "sampling_manifest.json")
            or plan["source_manifest_sha256"] != sha(root / "source_manifest.json")
            or plan["script_sha256"] != sha(root / "resume_fixed100_r02.py")):
        raise ValueError("native continuation has not passed its terminal checks")
    for receipt in (plan, started):
        if (receipt["automatic_retry"] is not False
                or receipt["reference_answers_read"] is not False
                or receipt["reference_answer_visible"] is not False
                or receipt["scientific_configuration_changed"] is not False
                or receipt["source_frozen_modified"] is not False):
            raise ValueError("native continuation changed the fixed protocol")
    preserved = plan["preserved_outcome_sha256"]
    final_ids = {qa for qa, q in selected.items() if q["source_split"] == "final_test"}
    if (len(preserved) != 63 or set(preserved) & final_ids
            or set(plan["pending_qa_ids"]) != set(selected) - set(preserved)
            or set(admission["preserved_outcome_sha256"]) != set(preserved)
            or set(admission["unissued_qa_ids"]) != set(plan["unissued_final_test_qa_ids"])
            or set(plan["gated_qa_ids"]) | set(admission["unissued_qa_ids"]) != final_ids
            or len(plan["gated_qa_ids"]) != 28 or len(admission["unissued_qa_ids"]) != 5):
        raise ValueError("continuation preserved or unissued denominators changed")
    for name, identity in plan["preserved_native_files"].items():
        p = directory / name
        if p.stat().st_size != identity["size"] or sha(p) != identity["sha256"]:
            raise ValueError("a previously issued native response changed")
    original_index = read(recovery / "original_trajectory_index.json")["questions"]
    current_index = read(directory / "trajectory_index.json")["questions"]
    for qa, identity in preserved.items():
        if (sha(directory / "outcomes" / (qa + ".json")) != identity
                or current_index[qa]["sha256"] != original_index[qa]["sha256"]):
            raise ValueError("a previously completed native trajectory changed")
    if len(started["failed_preparation_archives"]) != 4:
        raise ValueError("failed preparation history denominator changed")
    verified_archives = {}
    for archive in started["failed_preparation_archives"]:
        for row in archive["files"]:
            p = directory / row["file"]
            identity = row["identity"]
            if (not p.resolve().is_relative_to(directory.resolve())
                    or p.stat().st_size != identity["size"] or sha(p) != identity["sha256"]
                    or identity != plan["failed_native_files"][row["original_file"]]):
                raise ValueError("failed preparation native evidence changed")
            verified_archives[row["original_file"]] = identity
    if verified_archives != plan["failed_native_files"]:
        raise ValueError("failed native history was not fully preserved")
    old = read(recovery / "original_request_budget.json")
    budget = read(directory / "request_budget.json")
    if (sha(recovery / "original_request_budget.json") != plan["original_budget_sha256"]
            or old["used"] != 9710 or old["max_requests"] != 9710
            or budget["requests"][:9710] != old["requests"]
            or budget["max_requests"] != plan["continued_budget_max"]
            or budget["used"] > budget["max_requests"]
            or len(budget["requests"]) != budget["used"]):
        raise ValueError("continuation preparation budget provenance changed")
    additional = budget["requests"][9710:]
    if len(additional) != plan["additional_preparation_request_cap"]:
        raise ValueError("remaining preparation request denominator changed")
    seen = set()
    counts = {}
    for row in additional:
        qa, provider, digest = row["qa_id"], row["provider"], row["request_sha256"]
        key = qa, provider, digest
        detail = plan["preparation"][qa]
        if (key in seen or row["automatic_retry"] is not False
                or any(r["provider"] == provider and r["request_sha256"] == digest
                       for r in detail["prior_issued"])):
            raise ValueError("continuation repeated an issued model request")
        seen.add(key)
        counts[qa] = counts.get(qa, 0) + 1
    if any(counts.get(qa, 0) != detail["additional_description_binary_cap"]
           for qa, detail in plan["preparation"].items()):
        raise ValueError("per-question remaining preparation scope changed")
    sam2 = read(recovery / "sam2_new_admission.json")["requests"]
    sam_seen, sam_counts = set(), {}
    for row in sam2:
        key = row["qa_id"], row["request_sha256"]
        if key in sam_seen or row["automatic_retry"] is not False:
            raise ValueError("continuation repeated a SAM2 inference request")
        sam_seen.add(key)
        sam_counts[row["qa_id"]] = sam_counts.get(row["qa_id"], 0) + 1
    for qa, detail in started["preparation"].items():
        if sam_counts.get(qa, 0) > detail["new_sam2_batch_cap"]:
            raise ValueError("SAM2 continuation exceeded its fixed frame scope")
        if detail["sam2_replay_batches"]:
            current = read(directory / "sam2_receipts" / (qa + ".json"))
            if (len(current) != len(detail["sam2_replay_batches"])
                    or any(not row.get("native_replay") for row in current)):
                raise ValueError("issued SAM2 batches were not fully replayed")
        replay_root = recovery / "review_replay" / qa
        for cid, row in detail["native_visual"].items():
            p = replay_root / "visual" / (cid + ".json")
            if p.stat().st_size != row["identity"]["size"] or sha(p) != row["identity"]["sha256"]:
                raise ValueError("native description/review replay bytes changed")
    return {
        "schema": "vsi-native-continuation-verification/2",
        "preserved_native_questions": 63, "new_native_questions": 37,
        "archived_zero_call_gates": 28, "previously_unissued_final_questions": 5,
        "failed_preparation_questions_replayed": 4,
        "original_preparation_cap": 9710,
        "additional_preparation_requests": len(additional),
        "continued_preparation_cap": budget["max_requests"],
        "new_sam2_batches": len(sam2),
        "issued_model_requests_repeated": 0,
        "reference_answers_read": False, "reference_answer_visible": False,
        "plan_sha256": sha(recovery / "plan.json"),
        "started_receipt_sha256": sha(recovery / "started.json"),
        "status_sha256": sha(recovery / "status.json"),
    }


def verify_final_test_admission(root, selected):
    path = root / "final_test_admission.json"
    if not path.exists():
        return None
    admission = read(path)
    continued = admission["schema"] == "vsi-fixed-panel-final-test-admission/2"
    deadline = time.monotonic() + 120
    status_path = root / "final_test_admission_status.json"
    while read(status_path)["status"] != "complete":
        failure = (root / "recovery_r02/execute_failed.json" if continued else
                   root / "final_test_admission_failed.json")
        if (failure.exists()
                or time.monotonic() > deadline):
            raise ValueError("final-test admission has not passed its terminal checks")
        time.sleep(0.2)
    status, admission = read(status_path), read(path)
    expected = {qa for qa, q in selected.items() if q["source_split"] == "final_test"}
    if (len(expected) != 33 or set(admission["qa_ids"]) != expected
            or admission["questions"] != 33 or status["completed"] != 33
            or admission["prior_model_requests"] != 0
            or admission["issued_model_requests_repeated"] != 0
            or admission["automatic_retry"] is not False
            or admission["reference_answers_read"] is not False
            or admission["reference_answer_visible"] is not False
            or admission["source_frozen_modified"] is not False
            or admission["scientific_configuration_changed"] is not False
            or admission["configuration_change"] != {
                "allow_final_test": {"before": False, "after": True}}
            or admission["sampling_sha256"] != sha(root / "sampling_manifest.json")):
        raise ValueError("final-test admission scope or protocol changed")
    initial = root / admission["initial_launch"]
    if sha(initial) != admission["initial_launch_sha256"]:
        raise ValueError("original launch receipt changed")
    old_launch = read(initial)
    old_guard = old_launch["identity_guard"]
    if sha(root / old_guard["file"]) != old_guard["sha256"]:
        raise ValueError("original operational guard changed")
    preserved = admission["preserved_outcome_sha256"]
    if not continued and set(preserved) != set(selected) - expected:
        raise ValueError("admission changed previously issued episode scope")
    for qa, identity in preserved.items():
        if sha(root / "harness" / "outcomes" / f"{qa}.json") != identity:
            raise ValueError("a previously issued episode outcome changed")
    archives = admission["archived_episodes"]
    archived_ids = expected - set(admission["unissued_qa_ids"]) if continued else expected
    if {record["qa_id"] for record in archives} != archived_ids or len(archives) != len(archived_ids):
        raise ValueError("archived pre-model admission denominator changed")
    directory = root / "harness"
    for archive in archives:
        if archive["model_requests"] != 0 or archive["reference_answer_visible"] is not False:
            raise ValueError("admission repeated an issued request")
        archived_outcome = None
        for record in archive["files"]:
            file = directory / record["file"]
            identity = record["identity"] if continued else {
                "size": record["bytes"], "sha256": record["sha256"]}
            if (not file.resolve().is_relative_to(directory.resolve())
                    or file.stat().st_size != identity["size"] or sha(file) != identity["sha256"]):
                raise ValueError("archived pre-model trace identity changed")
            if record["original_file"] == f"outcomes/{archive['qa_id']}.json":
                archived_outcome = read(file)
        if archived_outcome is None:
            raise ValueError("archived zero-call outcome is missing")
        old = archived_outcome["outcome"]
        if (archived_outcome["native_requests"] or old["legal_answer"]
                or old["agent_rounds"] != 0 or old["synthesis_source"] != "none"
                or old["notes"] != [
                    "split=final_test：硬约束 9 禁止默认进在线链"
                    "（需 allow_final_test 显式盲评）"]):
            raise ValueError("archived episode was not the exact zero-call split gate")
    if continued:
        admission["native_continuation_verification"] = verify_native_continuation(
            root, selected, admission)
    return admission


def verify(root, arm, selected):
    directory = root / arm
    index = read(directory / "trajectory_index.json")
    state = read(directory / "live_status.json")
    manifest = read(directory / "runtime_identity.json")
    if (state["status"] != "complete" or state["completed"] != len(selected)
            or set(index["questions"]) != set(selected)
            or {p.stem for p in (directory / "outcomes").glob("*.json")} != set(selected)):
        raise ValueError("native verification requires the full completed panel")
    checked, questions, compact = {}, [], {}
    total_requests = 0
    for qa, question in selected.items():
        entry = index["questions"][qa]
        path = directory / entry["file"]
        if not path.resolve().is_relative_to(directory.resolve()) or sha(path) != entry["sha256"]:
            raise ValueError("native trajectory hash mismatch")
        full = read(path)
        complete = full["completeness"]
        if (full["qa_id"] != qa or full["question"] != question
                or full["status"] != "finished"
                or full["reference_answer_visible"] is not False
                or not complete["outcome_written"] or not complete["all_model_requests_returned"]
                or complete["missing_reviewer_request_files"] or complete["incomplete_jsonl_lines"]):
            raise ValueError("native trajectory is incomplete")
        sources = [*full["source_files"], *full["preparation"]["mask_assets"]]
        for source in sources:
            file = directory / source["file"]
            if not file.resolve().is_relative_to(directory.resolve()):
                raise ValueError("native source escapes the arm directory")
            expected = (source["bytes"], source["sha256"])
            if str(file) in checked:
                if checked[str(file)] != expected:
                    raise ValueError("native source identity is inconsistent")
                continue
            if file.stat().st_size != source["bytes"] or sha(file) != source["sha256"]:
                raise ValueError("native source or mask hash mismatch")
            checked[str(file)] = expected
        requests = full["native_solver_requests"]
        if (not requests or len(requests) > (1 if arm == "direct" else 12)
                or full["native_parser_requests"]):
            raise ValueError("native request count exceeds the frozen protocol")
        for item in requests:
            record = item["record"]
            payload, response = record["request"], record["raw_response"]
            request_sha = hashlib.sha256(json.dumps(
                payload, sort_keys=True, ensure_ascii=(arm == "direct"),
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if (record["status"] != "returned" or record["automatic_retry"] is not False
                    or record["reference_answer_visible"] is not False
                    or payload["model"] != "qwen3.8-flash"
                    or response["model"] != "qwen3.8-flash"
                    or payload["enable_thinking"] is not True
                    or payload["seed"] != 137 or payload["temperature"] != 0.
                    or payload["max_tokens"] != 8192
                    or request_sha != record["request_sha256"]
                    or hashlib.sha256(record["raw_response_text"].encode()).hexdigest()
                    != record["raw_response_sha256"]):
                raise ValueError("native model identity, payload or response audit failed")
            if arm == "direct":
                content = payload["messages"][0]["content"]
                if (sum(x.get("type") == "image_url" for x in content) != 32
                        or record["request_sha256"] != manifest["questions"][qa]["payload_sha256"]):
                    raise ValueError("Direct official32 payload changed")
        result = read(directory / "outcomes" / (qa + ".json"))
        if full["result"] != result or result["source_split"] != question["source_split"]:
            raise ValueError("native outcome or source split mismatch")
        outcome = result["outcome"]
        if arm == "direct" and outcome["tool_calls"] != 0:
            raise ValueError("Direct arm used tools")
        candidates = (full["preparation"]["binding"] or {}).get("stats", {}).get(
            "candidate_receipts", [])
        described = []
        for candidate in candidates:
            visual = candidate.get("semantic_visual") or {}
            if not visual:
                continue
            description = visual.get("target_description") or {}
            if (visual.get("status") != "observed" or description.get("status") != "described"
                    or description.get("raw_response", {}).get("model")
                    != "Qwen3-VL-8B-Instruct-FP8"):
                raise ValueError("8B description native response is unavailable or changed")
            described.append(candidate["candidate_id"])
        native_bytes = sum(source["bytes"] for source in sources)
        questions.append({
            "qa_id": qa, "trajectory_sha256": entry["sha256"],
            "trajectory_bytes": path.stat().st_size, "source_files_verified": len(sources),
            "native_source_bytes": native_bytes, "model_requests": len(requests),
            "described_candidates": len(described),
            "program_executions": len(full["program_executions"]),
            "program_errors": sum(bool((r["record"].get("cell") or {}).get("error_code"))
                                  for r in full["program_executions"]),
            "outcome_sha256": sha(directory / "outcomes" / (qa + ".json")),
        })
        compact[qa] = {
            "qa_id": qa, "question_type": result["question_type"],
            "source_split": result["source_split"], "reference_answer_visible": False,
            "outcome": {k: outcome.get(k) for k in (
                "answer", "predicted", "legal_answer", "is_mca", "answer_source",
                "agent_rounds")},
            "tool_calls": sum(r.get("tool_calls", 0) for r in outcome.get("rounds", [])),
            "outcome_sha256": questions[-1]["outcome_sha256"],
        }
        total_requests += len(requests)
        if len(questions) % 10 == 0:
            print(json.dumps({"arm": arm, "native_trajectories_verified": len(questions)}),
                  flush=True)
    receipt = {
        "arm": arm, "location": "Titan", "root": str(directory),
        "checked_at": time.time(), "questions_verified": len(questions),
        "trajectory_index_sha256": sha(directory / "trajectory_index.json"),
        "runtime_identity_sha256": sha(directory / "runtime_identity.json"),
        "unique_native_files_verified": len(checked),
        "unique_native_bytes_verified": sum(v[0] for v in checked.values()),
        "trajectory_bytes_verified": sum(q["trajectory_bytes"] for q in questions),
        "model_requests": total_requests, "questions": questions,
        "reference_answers_read": False, "reference_answer_visible": False,
    }
    write(directory / "native_trajectory_verified.json", receipt)
    write(directory / "compact_score_inputs.json", compact)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    sampling = read(args.root / "sampling_manifest.json")
    selected = {str(r["qa_id"]): r for r in sampling["questions"]}
    if len(selected) != 100:
        raise ValueError("native audit panel changed")
    manifest = read(args.root / "source_manifest.json")
    for source in manifest["files"]:
        path = args.root / "source_frozen" / source["file"]
        if path.stat().st_size != source["bytes"] or sha(path) != source["sha256"]:
            raise ValueError("frozen source changed during execution")
    guard = read(args.root / "harness.pid.json")["identity_guard"]
    if sha(args.root / guard["file"]) != guard["sha256"]:
        raise ValueError("owned operational identity guard changed")
    adapter = guard.get("execution_adapter")
    if adapter and sha(args.root / adapter["file"]) != adapter["sha256"]:
        raise ValueError("owned native request replay adapter changed")
    admission = verify_final_test_admission(args.root, selected)
    receipts = {arm: verify(args.root, arm, selected) for arm in ("direct", "harness")}
    write(args.root / "paired_native_verified.json", {
        "schema": "vsi-paired-native-verification/1", "location": "Titan",
        "checked_at": time.time(), "sampling_sha256": sha(args.root / "sampling_manifest.json"),
        "source_manifest_sha256": sha(args.root / "source_manifest.json"),
        "identity_guard": guard,
        "final_test_admission": admission,
        "reference_answers_read": False, "arms": receipts,
    })


if __name__ == "__main__":
    main()
