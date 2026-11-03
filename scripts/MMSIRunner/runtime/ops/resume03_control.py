"""Resume 03 host audits; accept only the two exact retained timeout receipts."""
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

import resume_control as previous

ROOT, OPS = previous.ROOT, previous.OPS
pilot, entry, watcher = previous.pilot, previous.entry, previous.watcher
MANIFEST_SHA = previous.MANIFEST_SHA
manifest_current = previous.manifest_current
operator_locks = previous.operator_locks
fingerprints = previous.fingerprints
retained_audit = previous.retained_audit
health = previous.health
process_receipt = previous.process_receipt
PREFLIGHT = OPS / "resume03_preflight.json"
RECOVERY = OPS / "resume03_recovery.json"
MONITOR = OPS / "resume-monitor-03"
NEW_FAILED_OP = "a4897ce890712f07801ebfd4e230fe0e5d32fb10d3acc9c027dd2da5186bfee2"
FAILURES = {
    previous.FAILED_OP: {
        "ordinal": 97, "job_sha256": previous.FAILED_JOB_SHA,
        "records": previous.FAILED_RECORDS, "ledger_hashes": previous.ERROR_HASHES},
    NEW_FAILED_OP: {
        "ordinal": 155,
        "job_sha256": "2506c6143bd9dbed8c6b5674261c237a14957ead744b4fba02fcbac9aea9c892",
        "records": {
            "001.intent.json": "48e997898587b41e4e75fac74690daed63533ad7b8006f4921d33639ef344bf8",
            "001.response.json": "c26ff0d05f4d8201a56008844497ee8dc404238762304270cc420a64c1b66898"},
        "ledger_hashes": {
            "requests": "24ece5b0f3e1b1f1158bcb1030cbd37516d95d9af16fb63ba14a33eb15bbc7d7",
            "test_budget": "60b898e729d3cfcd5157588e52cd7f433478232c8337ae4b568375f82c9a20e0"}},
}


def ledger_audit(manifest, baseline=None, active_ok=False):
    states = {}
    operations = {j["operation_id"] for j in manifest["jobs"]}
    for name, policy in (("requests", "shared_policy"), ("test_budget", "test_policy")):
        state = json.loads((ROOT / name / "state.json").read_text())
        assert state["policy"] == manifest[policy]
        assert state["policy"]["max_requests"] == 1900 and state["policy"]["max_inflight"] == 4
        assert len(state["requests"]) <= 1900 and state["peak_inflight"] <= 4
        assert [r["request_id"] for r in state["requests"]] == list(range(len(state["requests"])))
        allowed = {"completed", "error", "started"} if active_ok else {"completed", "error"}
        assert all(r["status"] in allowed for r in state["requests"])
        assert sum(r["status"] == "started" for r in state["requests"]) <= 4
        for row in state["requests"]:
            assert row["category"] == "solver"
            assert row["context"]["run_identity"] == manifest["run_identity"]
            assert row["context"]["variant_id"] in operations
        errors = [r for r in state["requests"] if r["status"] == "error"]
        assert len(errors) == 2
        assert {r["context"]["variant_id"]: pilot.sha_value(r) for r in errors} == {
            op: spec["ledger_hashes"][name] for op, spec in FAILURES.items()}
        if baseline:
            hashes = baseline["ledger_prefix_hashes"][name]
            assert [pilot.sha_value(r) for r in state["requests"][:len(hashes)]] == hashes
        states[name] = state
    if not active_ok:
        def counts(rows):
            return Counter(pilot.sha_value([r["context"], r["status"], r.get("usage", {})])
                           for r in rows)
        assert counts(states["requests"]["requests"]) == counts(states["test_budget"]["requests"])
    original_path = Path(manifest["original_budget_root"]) / "state.json"
    original = json.loads(original_path.read_text())
    prefix = json.loads((ROOT / "original_budget_prefix.json").read_text())
    assert original["requests"][:len(prefix["requests"])] == prefix["requests"]
    assert original["policy"]["max_requests"] == 20000
    assert all(r["status"] in {"completed", "error"} for r in original["requests"])
    assert len(original["requests"]) + len(states["requests"]["requests"]) <= 20000
    if baseline:
        assert pilot.sha(original_path) == baseline["original_ledger_sha256"]
    assert not watcher.process_state(pilot.read(Path(manifest["cv3d_root"]) / "startup.json"))["alive"]
    return states, original


def audit_response(job, path):
    response = pilot.read(path)
    identity = response["request_identity"]
    params = identity["parameters"]
    assert params["model"] == "qwen3.8-flash"
    assert params["temperature"] == 0 and params["seed"] == 137
    assert params["enable_thinking"] is True and params["max_tokens"] == 8192
    assert [r["jpeg_sha256"] for r in identity["images_in_order"]] == [r["sha256"] for r in job["images"]]
    if response["status"] == "received":
        assert job["operation_id"] not in FAILURES
        assert response["response_identity"]["model"] == "qwen3.8-flash"
        assert response["response_identity"]["model_reported"] is True
        return identity, True
    spec = FAILURES[job["operation_id"]]
    assert job["arm"] == "direct" and job["ordinal"] == spec["ordinal"]
    assert pilot.sha_value(job) == spec["job_sha256"]
    assert path.name == "001.response.json" and pilot.sha(path) == spec["records"][path.name]
    assert response["status"] == "error" and response["error_type"] == "ServiceUnavailable"
    assert response["service_error"]["reason_code"] == "timeout"
    assert response["service_error"]["request_stage"] == "awaiting_headers"
    assert response["service_error"]["timeout_s"] == 600
    assert response["response_identity"] == {} and response["model_response"] is None
    return identity, False


def result_audit(job, path):
    row = pilot.read(path)
    assert row["operation_id"] == job["operation_id"] and row["job_sha256"] == pilot.sha_value(job)
    assert row["reference_answer_visible"] is False
    result = row["result"]
    assert result["input_sha256"] == job["input_sha256"] and result["source_split"] == "test"
    disk = {str(p.relative_to(ROOT)): pilot.sha(p)
            for p in (ROOT / "request_records" / job["operation_id"]).glob("*.json")}
    assert disk == row["request_records"]
    responses = []
    for relative in disk:
        pilot.read(ROOT / relative)
        if relative.endswith(".response.json"):
            identity, received = audit_response(job, ROOT / relative)
            responses.append((relative, identity, received))
    assert len(responses) == result["request_count"] and len(responses) >= 1
    indexes = {Path(p).name.split(".")[0] for p, _, _ in responses}
    assert indexes == {f"{n:03d}" for n in range(1, len(responses) + 1)}
    assert set(disk) == {str(Path("request_records") / job["operation_id"] / (i + suffix))
                         for i in indexes for suffix in (".intent.json", ".response.json")}
    if job["operation_id"] in FAILURES:
        spec = FAILURES[job["operation_id"]]
        assert {Path(p).name: h for p, h in disk.items()} == spec["records"]
        assert row["status"] == "stopped" and result["status"] == "service_error"
        assert result["answer"] is None and result["request_count"] == 1
        assert row["host_recovery"]["automatic_retry"] is False
        assert not responses[0][2]
    else:
        assert row["status"] == "complete" and all(r[2] for r in responses)
        assert result["request_count"] <= (4 if job["arm"] == "skill" else 1)
    return row, responses


def preflight():
    started = time.monotonic()
    manifest = manifest_current()
    with operator_locks(manifest):
        for path in (ROOT / "startup.json", OPS / "startup.json",
                     previous.MONITOR / "startup.json"):
            assert not watcher.process_state(pilot.read(path))["alive"]
        assert not RECOVERY.exists() and not PREFLIGHT.exists()
        retained_audit(pilot.read(previous.PREFLIGHT), check_cache=True)
        prior = pilot.read(previous.RECOVERY)
        assert pilot.sha(ROOT / prior["timeout_result_path"]) == prior["timeout_result_sha256"]
        states, original = ledger_audit(manifest)
        assert all(len(s["requests"]) == 334 for s in states.values())
        result_hashes, request_hashes, intent_hashes, remaining = {}, {}, {}, []
        counts = Counter()
        jobs = {j["operation_id"]: j for j in manifest["jobs"]}
        assert len(jobs) == 730
        for job in manifest["jobs"]:
            op = job["operation_id"]
            result = ROOT / "results" / job["arm"] / (op + ".json")
            intent = ROOT / "intents" / (op + ".json")
            records = ROOT / "request_records" / op
            if result.exists():
                row, responses = result_audit(job, result)
                assert intent.exists()
                counts[job["arm"]] += 1
                result_hashes[str(result.relative_to(ROOT))] = pilot.sha(result)
                request_hashes.update(row["request_records"])
                for state in states.values():
                    assert sum(r["context"]["variant_id"] == op for r in state["requests"]) == len(responses)
            elif op == NEW_FAILED_OP:
                spec = FAILURES[op]
                assert pilot.sha_value(job) == spec["job_sha256"] and intent.exists()
                assert {p.name for p in records.glob("*.json")} == set(spec["records"])
                for name, expected in spec["records"].items():
                    assert pilot.sha(records / name) == expected
                    pilot.read(records / name)
                    request_hashes[str((records / name).relative_to(ROOT))] = expected
                assert audit_response(job, records / "001.response.json")[1] is False
                assert all(sum(r["context"]["variant_id"] == op for r in s["requests"]) == 1
                           for s in states.values())
            else:
                assert not intent.exists() and not records.exists()
                assert all(not any(r["context"]["variant_id"] == op for r in s["requests"])
                           for s in states.values())
                remaining.append(op)
            if intent.exists():
                assert pilot.read(intent)["job_sha256"] == pilot.sha_value(job)
                intent_hashes[str(intent.relative_to(ROOT))] = pilot.sha(intent)
        assert dict(counts) == {"skill": 162, "direct": 161}
        assert {str(p.relative_to(ROOT)) for arm in ("skill", "direct")
                for p in (ROOT / "results" / arm).glob("*.json")} == set(result_hashes)
        assert len(remaining) == 406 and len(intent_hashes) == 324 and len(request_hashes) == 668
        dispatched = set(jobs) - set(remaining)
        assert {p.stem for p in (ROOT / "intents").glob("*.json")} == dispatched
        assert {p.name for p in (ROOT / "request_records").iterdir()} == dispatched
        assert {str(p.relative_to(ROOT)) for p in (ROOT / "request_records").glob("*/*.json")} == set(request_hashes)
        assert Counter(jobs[op]["arm"] for op in remaining) == {"skill": 203, "direct": 203}
        max_additional = sum(4 if jobs[op]["arm"] == "skill" else 1 for op in remaining)
        assert 334 + max_additional == 1349 <= 1900
        assert len(original["requests"]) + 1900 <= 20000
        old = ROOT.parent / "mmsi-365-test-multiagent-20261101"
        records = {r["input_sha256"]: r for r in pilot.read(old / "preflight.json")["all_test_records"]}
        assert pilot.sha(old / "preflight.json") == pilot.read(ROOT / "preflight.json")["old_preflight_sha256"]
        files = {}
        frozen = json.loads((ROOT / "frozen_skills.json").read_text())
        assert len(frozen) == 6
        unique = {j["input_sha256"]: j for j in manifest["jobs"]}
        assert len(unique) == len(records) == 365
        for index, job in enumerate(unique.values(), 1):
            problem = pilot.problem_for(job["problem"])
            assert pilot.image_identities(problem) == job["images"]
            previous_input = records[job["input_sha256"]]
            folder = Path(job["prepared_root"]) / job["input_sha256"]
            for name, key in (("metadata.json", "metadata_sha256"), ("arrays.npz", "arrays_sha256")):
                path = folder / name
                assert pilot.sha(path) == previous_input[key]
                files[str(path)] = fingerprints(path)
            for name in problem.image_paths:
                files[str(Path(name))] = fingerprints(Path(name))
            if index % 10 == 0 or index == 365:
                pilot.update(OPS / "resume03_preflight_progress.json", {
                    "status": "checking_inputs", "completed": index, "expected": 365,
                    "at": pilot.now(), "model_requests": 0})
        for job in manifest["jobs"]:
            assert job["problem"]["source_split"] == "test"
            if job["arm"] == "skill":
                assert job["skill"] == frozen[job["slot"]]["skill"]
                assert job["allowed_tools"] == ["pair_point_distance", "point_depth", "view_candidates", "view_depth_summary"]
            else:
                assert job["skill"] is None and job["allowed_tools"] == []
        assert ledger_audit(manifest)[0] == states
        code = {str(p.relative_to(ROOT)): pilot.sha(p) for p in (
            OPS / "resume_control.py", OPS / "resume_entry.py", OPS / "resume_monitor.py",
            OPS / "resume_monitor_guard.py", OPS / "resume03_control.py",
            OPS / "resume03_entry.py", OPS / "resume03_monitor.py")}
        pilot.write(PREFLIGHT, {
            "status": "passed", "at": pilot.now(), "manifest_sha256": MANIFEST_SHA,
            "completed": dict(counts), "pending_never_dispatched": remaining,
            "retained_timeout_operations": list(FAILURES), "retained_service_errors": 2,
            "retained_request_count": 334,
            "result_hashes": result_hashes, "request_record_hashes": request_hashes,
            "intent_hashes": intent_hashes, "code_hashes": code,
            "ledger_prefix_hashes": {n: [pilot.sha_value(r) for r in s["requests"]] for n, s in states.items()},
            "original_ledger_sha256": pilot.sha(Path(manifest["original_budget_root"]) / "state.json"),
            "input_file_fingerprints": files, "all_test_input_hashes_verified": 365,
            "gateway": health(), "maximum_total_mmsi_requests": 334 + max_additional,
            "model_requests": 0, "gpu_requests": 0, "reference_answers_read": False,
            "six_skills_unchanged": True, "seconds": time.monotonic() - started})
        pilot.update(OPS / "resume03_preflight_progress.json", {
            "status": "passed", "completed": 365, "at": pilot.now(), "model_requests": 0})


def archive_runtime():
    history = OPS / "history_run02"
    assert not history.exists()
    names = ("startup.json", "progress.json", "failure.json", "exit.json", "scores.json")
    hashes = {name: pilot.sha(ROOT / name) for name in names}
    for name in names:
        pilot.read(ROOT / name)
    history.mkdir()
    for name in names:
        (ROOT / name).rename(history / name)
        assert pilot.sha(history / name) == hashes[name]
    pilot.write(history / "archive.json", {
        "at": pilot.now(), "files": hashes, "original_bytes_preserved": True,
        "request_ledgers_modified": False, "prediction_files_modified": False,
        "prior_monitor_history_preserved": True})
    return hashes


def materialize_timeout(manifest):
    op = NEW_FAILED_OP
    spec = FAILURES[op]
    job = next(j for j in manifest["jobs"] if j["operation_id"] == op)
    records = ROOT / "request_records" / op
    assert audit_response(job, records / "001.response.json")[1] is False
    response = pilot.read(records / "001.response.json")
    problem = pilot.problem_for(job["problem"])
    path = ROOT / "results" / "direct" / (op + ".json")
    pilot.write(path, {
        "status": "stopped", "operation_id": op, "job_sha256": spec["job_sha256"],
        "result": {"benchmark": "mmsi", "sample_id": problem.sample_id, "source_split": "test",
                   "task_type": problem.task_type, "input_sha256": problem.input_sha256,
                   "answer": None, "status": "service_error", "request_count": 1,
                   "attempted_tool_calls": 0, "succeeded_tool_calls": 0, "denied_authorizations": 0,
                   "rounds": [], "answer_parse_method": None},
        "observations": [], "seconds": response["seconds"],
        "request_records": {str((records / n).relative_to(ROOT)): h for n, h in spec["records"].items()},
        "reference_answer_visible": False,
        "host_recovery": {"at": pilot.now(), "automatic_retry": False, "resume_attempt": 3,
                          "source": "existing audited timeout receipt",
                          "model_response_received": False, "failure_scores_zero": True}})
    result_audit(job, path)
    return str(path.relative_to(ROOT)), pilot.sha(path)


def final_audit():
    manifest = manifest_current()
    baseline = pilot.read(PREFLIGHT)
    retained_audit(baseline, check_cache=True)
    states, original = ledger_audit(manifest, baseline)
    recovery = pilot.read(RECOVERY)
    assert pilot.sha(ROOT / recovery["timeout_result_path"]) == recovery["timeout_result_sha256"]
    predictions, pairs, response_counts = {}, {}, Counter()
    for job in manifest["jobs"]:
        path = ROOT / "results" / job["arm"] / (job["operation_id"] + ".json")
        row, responses = result_audit(job, path)
        assert pilot.read(ROOT / "intents" / (job["operation_id"] + ".json"))["job_sha256"] == pilot.sha_value(job)
        predictions[str(path.relative_to(ROOT))] = pilot.sha(path)
        response_counts[job["operation_id"]] = len(responses)
        first = next(r for r in responses if r[0].endswith("001.response.json"))
        pairs.setdefault(job["sample_id"], {})[job["arm"]] = (job, first[1], first[2])
    actual_requests = sum(response_counts.values())
    assert len(predictions) == 730 and len(pairs) == 365
    assert {str(p.relative_to(ROOT)) for arm in ("skill", "direct")
            for p in (ROOT / "results" / arm).glob("*.json")} == set(predictions)
    received_pairs = 0
    for arms in pairs.values():
        assert set(arms) == {"skill", "direct"}
        a, ai, ar = arms["skill"]
        b, bi, br = arms["direct"]
        assert a["input_sha256"] == b["input_sha256"]
        assert ai["parameters"] == bi["parameters"] and ai["images_in_order"] == bi["images_in_order"]
        received_pairs += int(ar and br)
    assert received_pairs == 363
    for state in states.values():
        assert Counter(r["context"]["variant_id"] for r in state["requests"]) == response_counts
        assert len(state["requests"]) == actual_requests
    pilot.write(ROOT / "request_pairing_audit.json", {
        "status": "passed_with_recorded_service_errors", "pairs": 365, "attempted_pairs": 365,
        "both_responses_received_pairs": 363, "requests": actual_requests, "service_errors": 2,
        "service_error_operations": list(FAILURES), "parameters_and_original_images_equal": True,
        "error_response_model_identity_available": False, "automatic_model_retries": 0, "at": pilot.now()})
    pilot.write(ROOT / "predictions_frozen.json", {
        "predictions": predictions, "questions": 365, "episodes": 730,
        "service_errors_retained": 2, "at": pilot.now()})
    pilot.score(final=True)
    scores = pilot.read(ROOT / "scores.json")
    assert scores["status"] == "complete" and len(scores["rows"]) == 730
    assert all(scores["summary"][a]["completed"] == 365 for a in ("skill", "direct"))
    failures = [r for r in scores["rows"] if r["status"] == "service_error"]
    assert len(failures) == 2 and {r["operation_id"] for r in failures} == set(FAILURES)
    assert all(r["score"] == 0 and r["prediction"] is None for r in failures)
    scores.pop("content_sha256")
    scores.update(service_errors=2, successful_response_pairs=363,
                  denominator_policy="all 365 author-test questions per arm; service/invalid/program errors score zero")
    pilot.update(ROOT / "scores.json", scores)
    pilot.write(ROOT / "completion.json", {
        "status": "complete", "completion_kind": "complete_with_recorded_service_errors",
        "episodes": 730, "questions": 365, "service_errors": 2, "successful_response_pairs": 363,
        "at": pilot.now(), "manifest_sha256": MANIFEST_SHA, "reference_answers_read": False,
        "request_usage": pilot.governor(ROOT / "requests", manifest["shared_policy"]).summary(),
        "online_train_requests": 0, "test_only": True, "automatic_model_retries": 0, "resume_attempt": 3,
        "original_plus_mmsi_requests": len(original["requests"]) + actual_requests,
        "request_pairing_audit_sha256": pilot.sha(ROOT / "request_pairing_audit.json"),
        "scores_sha256": pilot.sha(ROOT / "scores.json"),
        "frozen_predictions_sha256": pilot.sha(ROOT / "predictions_frozen.json"),
        "recovery_receipt_sha256": pilot.sha(RECOVERY)})
    pilot.update(ROOT / "progress.json", {
        "status": "complete", "episodes": 730, "service_errors": 2,
        "resume_attempt": 3, "at": pilot.now(), "test_only": True})


if __name__ == "__main__":
    code = 0
    try:
        pilot.write(OPS / "resume03_preflight_startup.json", {
            **process_receipt(), "source_sha256": pilot.sha(__file__), "model_requests": 0})
        preflight()
    except BaseException as exc:
        code = 2
        pilot.write(OPS / "resume03_preflight_failure.json", {
            "at": pilot.now(), "error_type": type(exc).__name__, "model_requests": 0,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)]})
    pilot.write(OPS / "resume03_preflight_exit.json", {
        "returncode": code, "at": pilot.now(), "model_requests": 0})
    sys.exit(code)
