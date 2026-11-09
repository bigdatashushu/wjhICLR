#!/usr/bin/env python3
"""Audit complete attempts, including explicitly verified terminal service failures."""
import hashlib
import json
from pathlib import Path
import time

from verify_proportional_native import read, sha, write


def verify(root, arm, selected):
    directory = root / arm
    continuation = root / "continuation_unissued"
    state = read(continuation / "live_status.json")
    finished = read(continuation / "verified.json")
    assert state["status"] == finished["status"] == "complete"
    assert state["completed"] == finished["completed"] == len(selected) == 80
    assert not (continuation / "failed.json").exists()
    from continue_unissued80 import verify_preserved
    plan = read(continuation / "plan.json")
    verify_preserved(plan)
    assert sha(continuation / "plan.json") == finished["plan_sha256"]
    assert sha(root / "continue_unissued80.py") == plan["script_sha256"] == finished["script_sha256"]
    assert sha(directory / "trajectory_index.json") == finished["final_index_sha256"]
    assert sha(directory / "request_budget.json") == finished["final_budget_sha256"]
    assert finished["issued_model_requests_repeated"] == 0
    index = read(directory / "trajectory_index.json")["questions"]
    assert set(index) == set(selected) == {p.stem for p in (directory / "outcomes").glob("*.json")}
    checked, questions, compact = {}, [], {}
    total_requests = returned_requests = unavailable_requests = 0
    failures = {}
    for qa, question in selected.items():
        entry = index[qa]
        path = directory / entry["file"]
        assert path.resolve().is_relative_to(directory.resolve()) and sha(path) == entry["sha256"]
        full = read(path)
        complete = full["completeness"]
        assert full["qa_id"] == qa and full["question"] == question
        assert full["reference_answer_visible"] is False and complete["outcome_written"]
        assert not complete["missing_reviewer_request_files"]
        assert not complete["incomplete_jsonl_lines"]
        result_path = directory / "outcomes" / (qa + ".json")
        result = read(result_path)
        assert result == full["result"] and result["source_split"] == question["source_split"]
        outcome = result["outcome"]
        requests = full["native_solver_requests"]
        assert 1 <= len(requests) <= 12 and not full["native_parser_requests"]
        terminal_failure = any(x["record"]["status"] == "unavailable" for x in requests)
        if terminal_failure:
            assert full["status"] in ("failed", "finished")
            assert outcome["failure_code"] == "service_error"
            assert not outcome["legal_answer"] and outcome["answer"] is None
            assert requests[-1]["record"]["status"] == "unavailable"
            assert all(x["record"]["status"] == "returned" for x in requests[:-1])
            assert not complete["all_model_requests_returned"]
            final = requests[-1]["record"]
            assert final["error_type"] in ("TimeoutError", "HTTPError", "URLError")
            assert final["elapsed_s"] > 0 and final.get("started_at")
            if final["error_type"] == "TimeoutError":
                assert final["transport_timeout_s"] == 900
            failures[qa] = {
                "error_type": final["error_type"], "elapsed_s": final["elapsed_s"],
                "http_status": final.get("http_status"), "request_index": final["request_index"],
                "outcome_sha256": sha(result_path),
                "unavailable_request_sha256": sha(directory / "solver_requests" / qa /
                                                   f"{final['request_index']:03d}.json"),
                "score_policy": "zero in full panel; excluded from valid-service cohort",
            }
        else:
            assert full["status"] == "finished" and complete["all_model_requests_returned"]
        sources = full["source_files"] + full["preparation"]["mask_assets"]
        for source in sources:
            p = directory / source["file"]
            assert p.resolve().is_relative_to(directory.resolve())
            expected = source["bytes"], source["sha256"]
            if str(p) in checked:
                assert checked[str(p)] == expected
            else:
                assert p.stat().st_size == source["bytes"] and sha(p) == source["sha256"]
                checked[str(p)] = expected
        for index_, item in enumerate(requests, 1):
            record = item["record"]
            payload = record["request"]
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            assert digest == record["request_sha256"]
            assert record["request_index"] == index_
            assert record["automatic_retry"] is False and record["reference_answer_visible"] is False
            assert payload["model"] == "qwen3.8-flash" and payload["enable_thinking"] is True
            assert payload["seed"] == 137 and payload["temperature"] == 0
            assert payload["max_tokens"] == 8192
            if record["status"] == "returned":
                response = record["raw_response"]
                assert response["model"] == "qwen3.8-flash"
                assert json.loads(record["raw_response_text"]) == response
                assert hashlib.sha256(record["raw_response_text"].encode()).hexdigest() == record[
                    "raw_response_sha256"]
                returned_requests += 1
            else:
                assert record["status"] == "unavailable" and terminal_failure
                assert index_ == len(requests)
                unavailable_requests += 1
        candidates = full["preparation"]["binding"]["stats"].get("candidate_receipts", [])
        described = []
        for candidate in candidates:
            visual = candidate.get("semantic_visual") or {}
            if visual:
                desc = visual.get("target_description") or {}
                assert visual["status"] == "observed" and desc["status"] == "described"
                assert desc["raw_response"]["model"] == "Qwen3-VL-8B-Instruct-FP8"
                described.append(candidate["candidate_id"])
        questions.append({
            "qa_id": qa, "trajectory_sha256": entry["sha256"],
            "trajectory_bytes": path.stat().st_size, "source_files_verified": len(sources),
            "native_source_bytes": sum(s["bytes"] for s in sources),
            "model_requests": len(requests), "described_candidates": len(described),
            "program_executions": len(full["program_executions"]),
            "program_errors": sum(bool((p["record"].get("cell") or {}).get("error_code"))
                                  for p in full["program_executions"]),
            "outcome_sha256": sha(result_path), "terminal_service_failure": terminal_failure,
        })
        compact[qa] = {
            "qa_id": qa, "question_type": result["question_type"],
            "source_split": result["source_split"], "reference_answer_visible": False,
            "terminal_service_failure": terminal_failure,
            "outcome": {k: outcome.get(k) for k in (
                "answer", "predicted", "legal_answer", "is_mca", "answer_source",
                "agent_rounds", "failure_code")},
            "outcome_sha256": sha(result_path),
        }
        total_requests += len(requests)
        if len(questions) % 10 == 0:
            print(json.dumps({"native_attempts_verified": len(questions),
                              "terminal_service_failures": len(failures)}), flush=True)
    host_errors = {p.stem: read(p) for p in (directory / "host_errors").glob("*.json")}
    assert set(host_errors) <= set(failures)
    assert set(host_errors) == {"5105"}, "unexpected original host error"
    for qa, row in failures.items():
        if qa != "5105":
            terminal = read(continuation / "terminal_service_failures" / (qa + ".json"))
            assert terminal["outcome_sha256"] == row["outcome_sha256"]
            assert terminal["unavailable_request_sha256"] == row["unavailable_request_sha256"]
    receipt = {
        "schema": "vsi-terminal-native-attempt-verification/1",
        "arm": arm, "location": "Titan", "root": str(directory), "checked_at": time.time(),
        "questions_verified": len(questions), "valid_service_questions": 80 - len(failures),
        "terminal_service_failures": failures,
        "trajectory_index_sha256": sha(directory / "trajectory_index.json"),
        "runtime_identity_sha256": sha(directory / "runtime_identity.json"),
        "unique_native_files_verified": len(checked),
        "unique_native_bytes_verified": sum(v[0] for v in checked.values()),
        "trajectory_bytes_verified": sum(q["trajectory_bytes"] for q in questions),
        "model_requests": total_requests, "returned_model_requests": returned_requests,
        "unavailable_model_requests": unavailable_requests,
        "questions": questions, "reference_answers_read": False,
        "reference_answer_visible": False, "verification_added_model_requests": 0,
        "continuation_verified_sha256": sha(continuation / "verified.json"),
    }
    write(directory / "native_trajectory_verified.json", receipt)
    write(directory / "compact_score_inputs.json", compact)
    return receipt
