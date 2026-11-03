"""Resume 08: preserve ten timeouts and one observed HTTP400, then continue."""
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

import resume_control as previous
import resume03_control as resume03
import resume04_control as resume04
import resume05_control as resume05
import resume06_control as resume06
import resume07_control as resume07
import timeout_handling08
import timeout_extension06
import timeout_transport06
from shared_budget_audit import audit_shared_state

ROOT, OPS = previous.ROOT, previous.OPS
pilot, entry, watcher = previous.pilot, previous.entry, previous.watcher
MANIFEST_SHA = previous.MANIFEST_SHA
manifest_current = previous.manifest_current
operator_locks = previous.operator_locks
fingerprints = previous.fingerprints
retained_audit = previous.retained_audit
process_receipt = previous.process_receipt
PREFLIGHT = OPS / "resume08_preflight.json"
RECOVERY = OPS / "resume08_recovery.json"
MONITOR = OPS / "resume-monitor-08"
ORIGINAL_PREFIX = OPS / "resume08_original_budget_prefix.json"
TIMEOUT_POLICY = OPS / "resume08_timeout_policy.json"
BOUND_ERRORS = OPS / "resume08_bound_errors.json"
BOUND_ERRORS_SHA = '921a46bcd52a77b917b8bc485b2f4efff0b5ad0a715e95e723e75a88ffaf2823'
assert pilot.sha(BOUND_ERRORS) == BOUND_ERRORS_SHA
bound = pilot.read(BOUND_ERRORS)
assert bound["manifest_sha256"] == MANIFEST_SHA and bound["model_requests"] == 0
FAILURES = bound["failures"]
assert len(FAILURES) == 11 and timeout_handling08.RETAINED_400_OP in FAILURES
for op, spec in resume07.FAILURES.items():
    assert FAILURES[op] == spec

def health():
    timeout_handling08.health()
    value = timeout_transport06.health()
    assert value["source_sha256"] == timeout_extension06.GATEWAY_SOURCE_SHA
    assert value["status"] == "ready" and value["max_inflight"] == 4
    return {k: value[k] for k in (
        "status", "schema_version", "model", "source_sha256", "max_inflight",
        "upstream_timeout_s", "mmsi_timeout_extension")}


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
        error_ops = {r["context"]["variant_id"] for r in errors}
        assert len(error_ops) == len(errors) and set(FAILURES) <= error_ops
        jobs = {j["operation_id"]: j for j in manifest["jobs"]}
        future_allowed = set(baseline["pending_never_dispatched"]) if baseline else set()
        for row in errors:
            op = row["context"]["variant_id"]
            if op in FAILURES:
                assert pilot.sha_value(row) == FAILURES[op]["ledger_hashes"][name]
            else:
                assert op in future_allowed and row["exception_type"] == "ServiceUnavailable"
                timeout_handling08.audit_operation(jobs[op])
                if not active_ok:
                    timeout_handling08.verify_handled(jobs[op])
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
    retained = pilot.read(ORIGINAL_PREFIX) if baseline else None
    if baseline:
        assert pilot.sha(ORIGINAL_PREFIX) == baseline["original_ledger_prefix_sha256"]
    audit_shared_state(
        original, prefix, retained, mmsi_operations=operations,
        mmsi_run_identity=manifest["run_identity"],
        mmsi_requests=len(states["requests"]["requests"]))
    assert not watcher.process_state(pilot.read(Path(manifest["cv3d_root"]) / "startup.json"))["alive"]
    return states, original


"""Audit preserved historical errors separately from future timeout recovery."""


def error_operations(state):
    return {r["context"]["variant_id"] for r in state["requests"] if r["status"] == "error"}


def audit_response(job, path):
    op = job["operation_id"]
    if op in FAILURES:
        if op == timeout_handling08.RETAINED_400_OP:
            return timeout_handling08.validate_retained400(job, path, FAILURES[op]["upstream_receipt"]), False
        return resume07.audit_response(job, path)
    response = pilot.read(path)
    if response["status"] == "received":
        return resume06.audit_response(job, path)
    identity, _ = timeout_handling08.proof_for(job, path)
    return identity, False


def result_audit(job, path):
    op = job["operation_id"]
    if op in FAILURES and op != timeout_handling08.RETAINED_400_OP:
        spec = FAILURES[op]
        assert pilot.sha(path) == spec["result_sha256"] and pilot.sha_value(job) == spec["job_sha256"]
        row, responses = resume07.result_audit(job, path)
        assert {Path(p).name: h for p, h in row["request_records"].items()} == spec["records"]
        if spec.get("kind") == "handled_timeout07":
            assert pilot.sha(resume07.timeout_handling07.HANDLED / (op + ".json")) == spec["handled_receipt_sha256"]
        return row, responses
    row = pilot.read(path)
    assert row["operation_id"] == op and row["job_sha256"] == pilot.sha_value(job)
    assert row["reference_answer_visible"] is False
    result = row["result"]
    assert result["input_sha256"] == job["input_sha256"] and result["source_split"] == "test"
    folder = ROOT / "request_records" / op
    disk = {str(p.relative_to(ROOT)): pilot.sha(p) for p in folder.glob("*.json")}
    assert disk == row["request_records"]
    responses = [(str(p.relative_to(ROOT)), *audit_response(job, p))
                 for p in sorted(folder.glob("*.response.json"))]
    assert len(responses) == result["request_count"]
    assert 1 <= len(responses) <= (4 if job["arm"] == "skill" else 1)
    assert [Path(p).name for p, _, _ in responses] == [
        f"{i:03d}.response.json" for i in range(1, len(responses) + 1)]
    assert {Path(p).name for p in disk} == {
        f"{i:03d}{suffix}" for i in range(1, len(responses) + 1) for suffix in (".intent.json", ".response.json")}
    for relative, identity, received in responses:
        intent = pilot.read(ROOT / relative.replace(".response.", ".intent."))
        assert intent["images"] == job["images"] and intent["reference_answer_visible"] is False
        assert all(identity["parameters"][k] == v for k, v in intent["parameters"].items())
    if result["status"] == "service_error":
        assert row["status"] == "stopped" and result["answer"] is None
        assert all(r[2] for r in responses[:-1]) and responses[-1][2] is False
        if op in FAILURES:
            spec = FAILURES[op]
            assert op == timeout_handling08.RETAINED_400_OP and result["request_count"] == 1
            assert pilot.sha(path) == spec["result_sha256"] and pilot.sha_value(job) == spec["job_sha256"]
            assert {Path(p).name: h for p, h in disk.items()} == spec["records"]
            assert ("host_recovery" in row) is spec["host_recovery_present"]
        else:
            timeout_handling08.verify_handled(job)
    else:
        assert op not in FAILURES and row["status"] == "complete" and all(r[2] for r in responses)
    return row, responses


def pair_statistics(pairs):
    first_received = fully_received = 0
    for arms in pairs.values():
        assert set(arms) == {"skill", "direct"}
        a, ai, ar, aa = arms["skill"]
        b, bi, br, ba = arms["direct"]
        assert a["input_sha256"] == b["input_sha256"]
        assert ai["parameters"] == bi["parameters"] and ai["images_in_order"] == bi["images_in_order"]
        first_received += int(ar and br)
        fully_received += int(aa and ba)
    return first_received, fully_received

def preflight():
    started = time.monotonic()
    tests = pilot.read(OPS / "resume08_test_receipt.json")
    assert tests["status"] == "passed" and tests["model_requests"] == tests["network_requests"] == 0
    manifest = manifest_current()
    with operator_locks(manifest):
        for path in (ROOT / "startup.json", OPS / "startup.json",
                     previous.MONITOR / "startup.json", resume03.MONITOR / "startup.json",
                     resume04.MONITOR / "startup.json", OPS / "resume04_preflight_startup.json",
                     resume05.MONITOR / "startup.json", OPS / "resume05_preflight_startup.json",
                     resume06.MONITOR / "startup.json", OPS / "resume06_preflight_startup.json",
                     resume07.MONITOR / "startup.json", OPS / "resume07_preflight_startup.json"):
            assert not watcher.process_state(pilot.read(path))["alive"]
        assert not RECOVERY.exists() and not PREFLIGHT.exists()
        old_baseline = pilot.read(resume07.PREFLIGHT)
        retained_audit(old_baseline, check_cache=True)
        assert pilot.sha(resume07.ORIGINAL_PREFIX) == old_baseline["original_ledger_prefix_sha256"]
        prior = pilot.read(resume07.RECOVERY)
        assert pilot.sha(ROOT / prior["timeout_result_path"]) == prior["timeout_result_sha256"]
        states, original = ledger_audit(manifest)
        assert all(len(s["requests"]) == 668 for s in states.values())
        for name, state in states.items():
            hashes = old_baseline["ledger_prefix_hashes"][name]
            assert [pilot.sha_value(r) for r in state["requests"][:len(hashes)]] == hashes
        assert all(r["status"] in {"completed", "error"} for r in original["requests"])
        legacy = json.loads((ROOT / "original_budget_prefix.json").read_text())
        shared_audit = audit_shared_state(
            original, legacy, pilot.read(resume07.ORIGINAL_PREFIX),
            mmsi_operations={j["operation_id"] for j in manifest["jobs"]},
            mmsi_run_identity=manifest["run_identity"], mmsi_requests=668)
        assert not ORIGINAL_PREFIX.exists()
        pilot.write(ORIGINAL_PREFIX, original)
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
            else:
                assert not intent.exists() and not records.exists()
                assert all(not any(r["context"]["variant_id"] == op for r in s["requests"])
                           for s in states.values())
                remaining.append(op)
            if intent.exists():
                assert pilot.read(intent)["job_sha256"] == pilot.sha_value(job)
                intent_hashes[str(intent.relative_to(ROOT))] = pilot.sha(intent)
        assert dict(counts) == {"skill": 327, "direct": 326}
        assert {str(p.relative_to(ROOT)) for arm in ("skill", "direct")
                for p in (ROOT / "results" / arm).glob("*.json")} == set(result_hashes)
        assert len(remaining) == 77 and len(intent_hashes) == 653 and len(request_hashes) == 1336
        dispatched = set(jobs) - set(remaining)
        assert {p.stem for p in (ROOT / "intents").glob("*.json")} == dispatched
        assert {p.name for p in (ROOT / "request_records").iterdir()} == dispatched
        assert {str(p.relative_to(ROOT)) for p in (ROOT / "request_records").glob("*/*.json")} == set(request_hashes)
        assert Counter(jobs[op]["arm"] for op in remaining) == {"skill": 38, "direct": 39}
        max_additional = sum(4 if jobs[op]["arm"] == "skill" else 1 for op in remaining)
        assert 668 + max_additional == 859 <= 1900
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
                pilot.update(OPS / "resume08_preflight_progress.json", {
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
        checks = {
            arm: timeout_extension06.zero_request_check(next(
                jobs[op] for op in remaining if jobs[op]["arm"] == arm))
            for arm in ("skill", "direct")}
        assert not TIMEOUT_POLICY.exists()
        prior_policy = pilot.read(resume07.TIMEOUT_POLICY)
        assert set(remaining) <= set(prior_policy["extended_operations"])
        pilot.write(TIMEOUT_POLICY, {
            "status": "authorized_timeout_record_and_continue", "at": pilot.now(),
            "manifest_sha256": MANIFEST_SHA, "run_identity": manifest["run_identity"],
            "original_timeout_s": 600, "upstream_timeout_s": 1200, "client_timeout_s": 1230,
            "extended_operations": prior_policy["extended_operations"],
            "prior_timeout_policy_sha256": pilot.sha(resume07.TIMEOUT_POLICY),
            "continue_after_verified_upstream_timeout": True,
            "new_timeout_failure_score": 0,
            "non_timeout_errors_remain_fatal": True, "retained_operations": sorted(dispatched),
            "gateway_source_sha256": timeout_extension06.GATEWAY_SOURCE_SHA,
            "gateway_port": 27440, "model_payload_unchanged": True,
            "historical_jobs_and_results_unchanged": True, "automatic_model_retries": 0,
            "user_authorization": "继续",
            "retained_observed_http400_operation": timeout_handling08.RETAINED_400_OP,
            "retained_timeouts": 10, "retained_non_timeout_service_errors": 1,
            "zero_request_transport_checks": checks})
        code = {str(p.relative_to(ROOT)): pilot.sha(p) for p in (
            OPS / "resume_control.py", OPS / "resume_entry.py", OPS / "resume_monitor.py",
            OPS / "resume_monitor_guard.py", OPS / "resume03_control.py",
            OPS / "resume03_entry.py", OPS / "resume03_monitor.py",
            OPS / "resume04_control.py", OPS / "resume04_entry.py", OPS / "resume04_monitor.py",
            OPS / "shared_budget_audit.py",
            OPS / "resume05_control.py", OPS / "resume05_entry.py", OPS / "resume05_monitor.py",
            OPS / "resume08_control.py", OPS / "resume08_entry.py", OPS / "resume08_monitor.py",
            OPS / "timeout_extension06.py", OPS / "timeout_transport06.py",
            OPS / "resume06_control.py", OPS / "resume06_entry.py", OPS / "resume06_monitor.py",
            resume06.TIMEOUT_POLICY, OPS / "timeout_handling08.py",
            OPS / "resume08_tests.py", OPS / "resume08_test_receipt.json",
            OPS / "resume07_control.py", OPS / "resume07_entry.py", OPS / "resume07_monitor.py",
            OPS / "resume07_tests.py", OPS / "resume07_test_receipt.json",
            OPS / "timeout_handling07.py", resume07.TIMEOUT_POLICY,
            *resume07.timeout_handling07.PROOFS.glob("*.json"),
            *resume07.timeout_handling07.HANDLED.glob("*.json"),
            *timeout_handling08.PROOFS.glob("*.json"), TIMEOUT_POLICY, BOUND_ERRORS)}
        pilot.write(PREFLIGHT, {
            "status": "passed", "at": pilot.now(), "manifest_sha256": MANIFEST_SHA,
            "completed": dict(counts), "pending_never_dispatched": remaining,
            "retained_timeout_operations": sorted(set(FAILURES) - {timeout_handling08.RETAINED_400_OP}),
            "retained_non_timeout_operations": [timeout_handling08.RETAINED_400_OP], "retained_service_errors": 11,
            "retained_request_count": 668,
            "result_hashes": result_hashes, "request_record_hashes": request_hashes,
            "intent_hashes": intent_hashes, "code_hashes": code,
            "ledger_prefix_hashes": {n: [pilot.sha_value(r) for r in s["requests"]] for n, s in states.items()},
            "original_ledger_sha256": pilot.sha(Path(manifest["original_budget_root"]) / "state.json"),
            "original_ledger_prefix_sha256": pilot.sha(ORIGINAL_PREFIX),
            "original_prefix_requests": len(original["requests"]),
            "verified_other_campaign_requests": shared_audit["other_requests_added_since_initial_freeze"],
            "retained_shared_budget_audit": shared_audit,
            "prior_preflight_sha256": pilot.sha(resume07.PREFLIGHT),
            "bound_timeout_evidence_sha256": BOUND_ERRORS_SHA,
            "shared_ledger_policy": "append-only unrelated campaigns; immutable retained rows; combined cap 20000",
            "input_file_fingerprints": files, "all_test_input_hashes_verified": 365,
            "gateway": health(), "maximum_total_mmsi_requests": 668 + max_additional,
            "timeout_policy_sha256": pilot.sha(TIMEOUT_POLICY),
            "timeout_extension_authorized": True,
            "upstream_timeout_s": 1200, "client_timeout_s": 1230,
            "zero_request_transport_checks": checks,
            "model_requests": 0, "gpu_requests": 0, "reference_answers_read": False,
            "six_skills_unchanged": True, "seconds": time.monotonic() - started})
        pilot.update(OPS / "resume08_preflight_progress.json", {
            "status": "passed", "completed": 365, "at": pilot.now(), "model_requests": 0})


def archive_runtime():
    history = OPS / "history_run07"
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


def final_audit():
    manifest = manifest_current()
    baseline = pilot.read(PREFLIGHT)
    retained_audit(baseline, check_cache=True)
    states, original = ledger_audit(manifest, baseline)
    recovery = pilot.read(RECOVERY)
    assert pilot.sha(ROOT / recovery["timeout_result_path"]) == recovery["timeout_result_sha256"]
    failure_ops = error_operations(states["requests"])
    error_count = len(failure_ops)
    predictions, pairs, response_counts = {}, {}, Counter()
    for job in manifest["jobs"]:
        path = ROOT / "results" / job["arm"] / (job["operation_id"] + ".json")
        row, responses = result_audit(job, path)
        assert pilot.read(ROOT / "intents" / (job["operation_id"] + ".json"))["job_sha256"] == pilot.sha_value(job)
        predictions[str(path.relative_to(ROOT))] = pilot.sha(path)
        response_counts[job["operation_id"]] = len(responses)
        first = next(r for r in responses if r[0].endswith("001.response.json"))
        pairs.setdefault(job["sample_id"], {})[job["arm"]] = (job, first[1], first[2], all(r[2] for r in responses))
    actual_requests = sum(response_counts.values())
    assert len(predictions) == 730 and len(pairs) == 365
    assert {str(p.relative_to(ROOT)) for arm in ("skill", "direct")
            for p in (ROOT / "results" / arm).glob("*.json")} == set(predictions)
    first_received_pairs, received_pairs = pair_statistics(pairs)
    extended_ops = set(pilot.read(TIMEOUT_POLICY)["extended_operations"])
    mixed_wait_pairs = [
        sample for sample, arms in pairs.items()
        if (arms["skill"][0]["operation_id"] in extended_ops)
        != (arms["direct"][0]["operation_id"] in extended_ops)]
    assert len(mixed_wait_pairs) == 1
    for state in states.values():
        assert Counter(r["context"]["variant_id"] for r in state["requests"]) == response_counts
        assert len(state["requests"]) == actual_requests
    pilot.write(ROOT / "request_pairing_audit.json", {
        "status": "passed_with_recorded_service_errors", "pairs": 365, "attempted_pairs": 365,
        "both_responses_received_pairs": first_received_pairs,
        "all_responses_received_pairs": received_pairs, "requests": actual_requests, "service_errors": error_count,
        "timeout_policy_sha256": pilot.sha(TIMEOUT_POLICY),
        "mixed_wait_policy_pairs": mixed_wait_pairs, "same_wait_policy_pairs": 364,
        "service_error_operations": sorted(failure_ops), "parameters_and_original_images_equal": True,
        "error_response_model_identity_available": False, "automatic_model_retries": 0, "at": pilot.now()})
    pilot.write(ROOT / "predictions_frozen.json", {
        "predictions": predictions, "questions": 365, "episodes": 730,
        "service_errors_retained": error_count, "at": pilot.now()})
    pilot.score(final=True)
    scores = pilot.read(ROOT / "scores.json")
    assert scores["status"] == "complete" and len(scores["rows"]) == 730
    assert all(scores["summary"][a]["completed"] == 365 for a in ("skill", "direct"))
    failures = [r for r in scores["rows"] if r["status"] == "service_error"]
    assert len(failures) == error_count and {r["operation_id"] for r in failures} == failure_ops
    assert all(r["score"] == 0 and r["prediction"] is None for r in failures)
    scores.pop("content_sha256")
    scores.update(service_errors=error_count, successful_response_pairs=received_pairs,
                  timeout_policy_sha256=pilot.sha(TIMEOUT_POLICY),
                  mixed_wait_policy_pairs=mixed_wait_pairs,
                  denominator_policy="all 365 author-test questions per arm; service/invalid/program errors score zero")
    pilot.update(ROOT / "scores.json", scores)
    pilot.write(ROOT / "completion.json", {
        "status": "complete", "completion_kind": "complete_with_recorded_service_errors",
        "episodes": 730, "questions": 365, "service_errors": error_count, "successful_response_pairs": received_pairs,
        "timeout_policy_sha256": pilot.sha(TIMEOUT_POLICY),
        "upstream_timeout_s": 1200, "client_timeout_s": 1230,
        "at": pilot.now(), "manifest_sha256": MANIFEST_SHA, "reference_answers_read": False,
        "request_usage": pilot.governor(ROOT / "requests", manifest["shared_policy"]).summary(),
        "online_train_requests": 0, "test_only": True, "automatic_model_retries": 0, "resume_attempt": 8,
        "original_plus_mmsi_requests": len(original["requests"]) + actual_requests,
        "request_pairing_audit_sha256": pilot.sha(ROOT / "request_pairing_audit.json"),
        "scores_sha256": pilot.sha(ROOT / "scores.json"),
        "frozen_predictions_sha256": pilot.sha(ROOT / "predictions_frozen.json"),
        "recovery_receipt_sha256": pilot.sha(RECOVERY)})
    pilot.update(ROOT / "progress.json", {
        "status": "complete", "episodes": 730, "service_errors": error_count,
        "resume_attempt": 8, "at": pilot.now(), "test_only": True})


if __name__ == "__main__":
    code = 0
    try:
        pilot.write(OPS / "resume08_preflight_startup.json", {
            **process_receipt(), "source_sha256": pilot.sha(__file__), "model_requests": 0})
        preflight()
    except BaseException as exc:
        code = 2
        pilot.write(OPS / "resume08_preflight_failure.json", {
            "at": pilot.now(), "error_type": type(exc).__name__, "model_requests": 0,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)]})
    pilot.write(OPS / "resume08_preflight_exit.json", {
        "returncode": code, "at": pilot.now(), "model_requests": 0})
    sys.exit(code)
