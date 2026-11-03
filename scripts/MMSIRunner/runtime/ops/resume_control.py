"""Host-only recovery audits for the single, retained MMSI timeout."""
from collections import Counter
from contextlib import ExitStack, contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
OPS = ROOT / "ops"
for key in list(os.environ):
    if any(s in key.upper() for s in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY")):
        os.environ.pop(key, None)
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
sys.path.insert(0, str(ROOT))
import entry
pilot = entry.pilot
watcher = entry.watcher
transport = entry.audited_eval_transport

MANIFEST_SHA = "b933e38f0b02be6a334cd05ddfb16378ad2548a9198b9e555aa84c696e22a69b"
FAILED_OP = "67849b67c12837a6f575ff533ffa85879b857a9094350ccf8271e853c45298c9"
FAILED_JOB_SHA = "c6a8029d22a3766e56af5c2ece1d5522b142448e84a232c4d7c50ccbf662a1ca"
FAILED_RECORDS = {
    "001.intent.json": "4cde077fe813ded627a7f2fa7c4f3f38415f4971bf0f90d458766362db4bae82",
    "001.response.json": "fec242b4313369dc8b7b0ebb747e86aa18a466ed2c934c9624c61cfdd590cad0",
}
ERROR_HASHES = {
    "requests": "5c7eb0180a4f2d3851542f89d618eef6de18c051cf6d920f44b86914d4068f33",
    "test_budget": "81c445962658897f4ba3070d2ab97f08ce3f2640e69f81b1b8a6b150a635a419",
}
PREFLIGHT = OPS / "resume02_preflight.json"
RECOVERY = OPS / "resume02_recovery.json"
MONITOR = OPS / "resume-monitor-02"


@contextmanager
def operator_locks(manifest):
    with ExitStack() as stack:
        for path in (ROOT / "operator.lock",
                     Path(manifest["original_budget_root"]) / "thinking_pilot_operator.lock"):
            stream = stack.enter_context(path.open("a+b"))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def manifest_current():
    manifest = pilot.read(ROOT / "manifest.json")
    assert manifest["content_sha256"] == MANIFEST_SHA
    assert pilot.read(ROOT / "ready.json")["manifest_sha256"] == MANIFEST_SHA
    assert pilot.read(ROOT / "preflight.json")["status"] == "passed"
    assert pilot.read(ROOT / "preflight_exit.json")["returncode"] == 0
    assert pilot.sha(ROOT / "entry.py") == "c356ca89e9810e56cc93993e81fb138657d6a1d3749a8d8513fdbc1d96d93b9f"
    assert pilot.sha(ROOT / "frozen_skills.json") == "dbc9693fafab9e34267d6f79dedffa3fb0c06a18c55f75bfd3be53f83672a3a9"
    pilot.verify(manifest)
    return manifest


def ledger_audit(manifest, baseline=None, active_ok=False):
    states = {}
    operations = {j["operation_id"] for j in manifest["jobs"]}
    for name, policy in (("requests", "shared_policy"), ("test_budget", "test_policy")):
        state = json.loads((ROOT / name / "state.json").read_text())
        assert state["policy"] == manifest[policy]
        assert state["policy"]["max_requests"] == 1900 and state["policy"]["max_inflight"] == 4
        assert len(state["requests"]) <= 1900
        assert state["peak_inflight"] <= 4
        assert [r["request_id"] for r in state["requests"]] == list(range(len(state["requests"])))
        assert all(r["status"] in ({"completed", "error", "started"} if active_ok else {"completed", "error"})
                   for r in state["requests"])
        for row in state["requests"]:
            assert row["category"] == "solver"
            assert row["context"]["run_identity"] == manifest["run_identity"]
            assert row["context"]["variant_id"] in operations
        errors = [r for r in state["requests"] if r["status"] == "error"]
        assert len(errors) == 1
        assert errors[0]["context"]["variant_id"] == FAILED_OP
        assert pilot.sha_value(errors[0]) == ERROR_HASHES[name]
        if baseline:
            hashes = baseline["ledger_prefix_hashes"][name]
            assert [pilot.sha_value(r) for r in state["requests"][:len(hashes)]] == hashes
        states[name] = state
    if not active_ok:
        a, b = (states[n]["requests"] for n in ("requests", "test_budget"))
        # Nested governors can reserve in different order across workers.
        def counts(rows):
            return Counter(pilot.sha_value([r["context"], r["status"], r.get("usage", {})])
                           for r in rows)
        assert counts(a) == counts(b)
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
        assert response["response_identity"]["model"] == "qwen3.8-flash"
        assert response["response_identity"]["model_reported"] is True
        return identity, True
    assert job["operation_id"] == FAILED_OP and job["arm"] == "direct" and job["ordinal"] == 97
    assert path.name == "001.response.json" and pilot.sha(path) == FAILED_RECORDS[path.name]
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
    for relative, expected in disk.items():
        assert pilot.sha(ROOT / relative) == expected
        pilot.read(ROOT / relative)
        if relative.endswith(".response.json"):
            identity, received = audit_response(job, ROOT / relative)
            responses.append((relative, identity, received))
    assert len(responses) == result["request_count"]
    indexes = {Path(p).name.split(".")[0] for p, _, _ in responses}
    assert indexes == {f"{n:03d}" for n in range(1, len(responses) + 1)}
    assert set(disk) == {str(Path("request_records") / job["operation_id"] / (i + suffix))
                         for i in indexes for suffix in (".intent.json", ".response.json")}
    if job["operation_id"] == FAILED_OP:
        assert row["status"] == "stopped" and result["status"] == "service_error"
        assert result["answer"] is None and result["request_count"] == 1
        assert row["host_recovery"]["automatic_retry"] is False
        assert not responses[0][2]
    else:
        assert row["status"] == "complete" and all(r[2] for r in responses)
    return row, responses


def fingerprints(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def retained_audit(baseline, check_cache=False):
    for key in ("result_hashes", "request_record_hashes", "intent_hashes", "code_hashes"):
        for relative, expected in baseline[key].items():
            assert pilot.sha(ROOT / relative) == expected
    if check_cache:
        for path, expected in baseline["input_file_fingerprints"].items():
            assert fingerprints(Path(path)) == expected


def health():
    manifest = pilot.read(ROOT / "manifest.json")
    value = transport.health()
    assert value["status"] == "ready" and value["model"] == "qwen3.8-flash"
    assert value["source_sha256"] == manifest["audited_gateway_source_sha256"]
    assert value["schema_version"] == manifest["jobs"][0]["proxy_schema"] and value["max_inflight"] == 4
    return {k: value[k] for k in ("status", "schema_version", "model", "source_sha256", "max_inflight")}


def preflight():
    started = time.monotonic()
    manifest = manifest_current()
    with operator_locks(manifest):
        assert not watcher.process_state(pilot.read(ROOT / "startup.json"))["alive"]
        assert not watcher.process_state(pilot.read(OPS / "startup.json"))["alive"]
        assert not RECOVERY.exists() and not PREFLIGHT.exists()
        states, original = ledger_audit(manifest)
        assert len(states["requests"]["requests"]) == 217
        result_hashes, request_hashes, intent_hashes, remaining = {}, {}, {}, []
        counts = Counter()
        jobs = {j["operation_id"]: j for j in manifest["jobs"]}
        assert len(jobs) == 730
        for job in manifest["jobs"]:
            result = ROOT / "results" / job["arm"] / (job["operation_id"] + ".json")
            intent = ROOT / "intents" / (job["operation_id"] + ".json")
            records = ROOT / "request_records" / job["operation_id"]
            if result.exists():
                row, responses = result_audit(job, result)
                assert intent.exists()
                counts[job["arm"]] += 1
                result_hashes[str(result.relative_to(ROOT))] = pilot.sha(result)
                request_hashes.update(row["request_records"])
                assert sum(r["context"]["variant_id"] == job["operation_id"]
                           for r in states["requests"]["requests"]) == len(responses)
            elif job["operation_id"] == FAILED_OP:
                assert pilot.sha_value(job) == FAILED_JOB_SHA and intent.exists()
                assert {p.name for p in records.glob("*.json")} == set(FAILED_RECORDS)
                for name, expected in FAILED_RECORDS.items():
                    assert pilot.sha(records / name) == expected
                    pilot.read(records / name)
                    request_hashes[str((records / name).relative_to(ROOT))] = expected
                assert audit_response(job, records / "001.response.json")[1] is False
            else:
                assert not intent.exists() and not records.exists()
                assert not any(r["context"]["variant_id"] == job["operation_id"]
                               for r in states["requests"]["requests"])
                remaining.append(job["operation_id"])
            if intent.exists():
                assert pilot.read(intent)["job_sha256"] == pilot.sha_value(job)
                intent_hashes[str(intent.relative_to(ROOT))] = pilot.sha(intent)
        assert dict(counts) == {"skill": 104, "direct": 102}
        assert {str(p.relative_to(ROOT)) for arm in ("skill", "direct")
                for p in (ROOT / "results" / arm).glob("*.json")} == set(result_hashes)
        assert len(remaining) == 523 and len(intent_hashes) == 207
        assert len(request_hashes) == 434
        assert {p.stem for p in (ROOT / "intents").glob("*.json")} == {
            jobs[op]["operation_id"] for op in jobs if op not in remaining}
        assert {p.name for p in (ROOT / "request_records").iterdir()} == {
            jobs[op]["operation_id"] for op in jobs if op not in remaining}
        max_additional = sum(4 if jobs[op]["arm"] == "skill" else 1 for op in remaining)
        assert 217 + max_additional <= 1900
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
            previous = records[job["input_sha256"]]
            folder = Path(job["prepared_root"]) / job["input_sha256"]
            for name, key in (("metadata.json", "metadata_sha256"), ("arrays.npz", "arrays_sha256")):
                path = folder / name
                assert pilot.sha(path) == previous[key]
                files[str(path)] = fingerprints(path)
            for name in problem.image_paths:
                files[str(Path(name))] = fingerprints(Path(name))
            if index % 10 == 0 or index == 365:
                pilot.update(OPS / "resume02_preflight_progress.json", {
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
            OPS / "resume_control.py", OPS / "resume_entry.py", OPS / "resume_monitor.py")}
        pilot.write(PREFLIGHT, {
            "status": "passed", "at": pilot.now(), "manifest_sha256": MANIFEST_SHA,
            "completed": dict(counts), "pending_never_dispatched": remaining,
            "retained_timeout_operation": FAILED_OP, "retained_request_count": 217,
            "result_hashes": result_hashes, "request_record_hashes": request_hashes,
            "intent_hashes": intent_hashes, "code_hashes": code,
            "ledger_prefix_hashes": {n: [pilot.sha_value(r) for r in s["requests"]] for n, s in states.items()},
            "original_ledger_sha256": pilot.sha(Path(manifest["original_budget_root"]) / "state.json"),
            "input_file_fingerprints": files, "all_test_input_hashes_verified": 365,
            "gateway": health(), "maximum_total_mmsi_requests": 217 + max_additional,
            "model_requests": 0, "gpu_requests": 0, "reference_answers_read": False,
            "six_skills_unchanged": True, "seconds": time.monotonic() - started})
        pilot.update(OPS / "resume02_preflight_progress.json", {
            "status": "passed", "completed": 365, "at": pilot.now(), "model_requests": 0})


def archive_runtime():
    history = OPS / "history_run01"
    history.mkdir()
    names = ("startup.json", "progress.json", "failure.json", "exit.json", "scores.json")
    hashes = {name: pilot.sha(ROOT / name) for name in names}
    for name in names:
        pilot.read(ROOT / name)
        (ROOT / name).rename(history / name)
        assert pilot.sha(history / name) == hashes[name]
    pilot.write(history / "archive.json", {
        "at": pilot.now(), "files": hashes, "original_bytes_preserved": True,
        "request_ledgers_modified": False, "prediction_files_modified": False})
    return hashes


def materialize_timeout(manifest):
    job = next(j for j in manifest["jobs"] if j["operation_id"] == FAILED_OP)
    records = ROOT / "request_records" / FAILED_OP
    assert audit_response(job, records / "001.response.json")[1] is False
    response = pilot.read(records / "001.response.json")
    problem = pilot.problem_for(job["problem"])
    path = ROOT / "results" / "direct" / (FAILED_OP + ".json")
    pilot.write(path, {
        "status": "stopped", "operation_id": FAILED_OP, "job_sha256": FAILED_JOB_SHA,
        "result": {"benchmark": "mmsi", "sample_id": problem.sample_id, "source_split": "test",
                   "task_type": problem.task_type, "input_sha256": problem.input_sha256,
                   "answer": None, "status": "service_error", "request_count": 1,
                   "attempted_tool_calls": 0, "succeeded_tool_calls": 0, "denied_authorizations": 0,
                   "rounds": [], "answer_parse_method": None},
        "observations": [], "seconds": response["seconds"],
        "request_records": {str((records / n).relative_to(ROOT)): h for n, h in FAILED_RECORDS.items()},
        "reference_answer_visible": False,
        "host_recovery": {"at": pilot.now(), "automatic_retry": False,
                          "source": "existing audited timeout receipt",
                          "model_response_received": False, "failure_scores_zero": True}})
    result_audit(job, path)
    return str(path.relative_to(ROOT)), pilot.sha(path)


def final_audit():
    manifest = manifest_current()
    baseline = pilot.read(PREFLIGHT)
    retained_audit(baseline)
    states, original = ledger_audit(manifest, baseline)
    recovery = pilot.read(RECOVERY)
    assert pilot.sha(ROOT / recovery["timeout_result_path"]) == recovery["timeout_result_sha256"]
    predictions, pairs, response_counts = {}, {}, Counter()
    actual_requests = 0
    for job in manifest["jobs"]:
        path = ROOT / "results" / job["arm"] / (job["operation_id"] + ".json")
        row, responses = result_audit(job, path)
        intent = pilot.read(ROOT / "intents" / (job["operation_id"] + ".json"))
        assert intent["job_sha256"] == pilot.sha_value(job)
        predictions[str(path.relative_to(ROOT))] = pilot.sha(path)
        response_counts[job["operation_id"]] = len(responses)
        actual_requests += len(responses)
        first = next(r for r in responses if r[0].endswith("001.response.json"))
        pairs.setdefault(job["sample_id"], {})[job["arm"]] = (job, first[1], first[2])
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
    assert received_pairs == 364
    for state in states.values():
        assert Counter(r["context"]["variant_id"] for r in state["requests"]) == response_counts
        assert len(state["requests"]) == actual_requests
    pairing = {
        "status": "passed_with_recorded_service_error", "pairs": 365, "attempted_pairs": 365,
        "both_responses_received_pairs": 364, "requests": actual_requests, "service_errors": 1,
        "service_error_operation": FAILED_OP, "parameters_and_original_images_equal": True,
        "error_response_model_identity_available": False, "automatic_model_retries": 0, "at": pilot.now()}
    pilot.write(ROOT / "request_pairing_audit.json", pairing)
    pilot.write(ROOT / "predictions_frozen.json", {
        "predictions": predictions, "questions": 365, "episodes": 730,
        "service_errors_retained": 1, "at": pilot.now()})
    pilot.score(final=True)
    scores = pilot.read(ROOT / "scores.json")
    assert scores["status"] == "complete" and len(scores["rows"]) == 730
    assert all(scores["summary"][a]["completed"] == 365 for a in ("skill", "direct"))
    failures = [r for r in scores["rows"] if r["status"] == "service_error"]
    assert len(failures) == 1 and failures[0]["operation_id"] == FAILED_OP
    assert failures[0]["score"] == 0 and failures[0]["prediction"] is None
    scores.pop("content_sha256")
    scores.update(service_errors=1, successful_response_pairs=364,
                  denominator_policy="all 365 author-test questions per arm; service/invalid/program errors score zero")
    pilot.update(ROOT / "scores.json", scores)
    pilot.write(ROOT / "completion.json", {
        "status": "complete", "completion_kind": "complete_with_recorded_service_error",
        "episodes": 730, "questions": 365, "service_errors": 1, "successful_response_pairs": 364,
        "at": pilot.now(), "manifest_sha256": MANIFEST_SHA, "reference_answers_read": False,
        "request_usage": pilot.governor(ROOT / "requests", manifest["shared_policy"]).summary(),
        "online_train_requests": 0, "test_only": True, "automatic_model_retries": 0,
        "original_plus_mmsi_requests": len(original["requests"]) + actual_requests,
        "request_pairing_audit_sha256": pilot.sha(ROOT / "request_pairing_audit.json"),
        "scores_sha256": pilot.sha(ROOT / "scores.json"),
        "frozen_predictions_sha256": pilot.sha(ROOT / "predictions_frozen.json"),
        "recovery_receipt_sha256": pilot.sha(RECOVERY)})
    pilot.update(ROOT / "progress.json", {
        "status": "complete", "episodes": 730, "service_errors": 1, "at": pilot.now(), "test_only": True})


def process_receipt():
    proc = Path("/proc") / str(os.getpid())
    fields = (proc / "stat").read_text().split(") ", 1)[1].split()
    return {"pid": os.getpid(), "start_ticks": fields[19],
            "command_sha256": pilot.sha(proc / "cmdline"), "at": pilot.now()}


if __name__ == "__main__":
    code = 0
    try:
        pilot.write(OPS / "resume02_preflight_startup.json", {
            **process_receipt(), "source_sha256": pilot.sha(__file__), "model_requests": 0})
        preflight()
    except BaseException as exc:
        code = 2
        pilot.write(OPS / "resume02_preflight_failure.json", {
            "at": pilot.now(), "error_type": type(exc).__name__, "model_requests": 0,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)]})
    pilot.write(OPS / "resume02_preflight_exit.json", {
        "returncode": code, "at": pilot.now(), "model_requests": 0})
    sys.exit(code)
