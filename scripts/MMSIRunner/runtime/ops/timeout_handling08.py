"""Verify upstream timeout evidence and retain one failure without retrying."""
import hashlib
import fcntl
import json
from pathlib import Path
import urllib.request

import pilot
from provider import token

ROOT = pilot.ROOT
OPS = ROOT / "ops"
PROOFS = OPS / "resume08_timeout_proofs"
HANDLED = OPS / "resume08_handled_timeouts"
READER = "http://10.10.0.198:27442"
READER_SOURCE_SHA = "b56537d1beba92972bf5ba11eb06e2cd848f6c3469a44779ae724bb4ad566dc5"
GATEWAY_SOURCE_SHA = "61e7166812e170dd6e2e3446e611c0e7f3a976cf2ee99a5c686c79f06bea09d1"
RUN = "92a55aecff8c3e79090ef2b4fdfd02132185e55bca006479de332eafc04edbb2"
HISTORICAL_OP = "a584a0d562b9bd4fcc9ff892eae31c2be909cac5e3e1bda328ef8ebacf50ba06"


def get(path):
    secret = token()
    request = urllib.request.Request(READER + path, headers={"Authorization": "Bearer " + secret})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=10) as response:
        raw = response.read(65537)
    assert len(raw) <= 65536 and secret.encode() not in raw
    value = json.loads(raw)
    assert value["schema_version"] == "harness3d.timeout-receipt-reader/1"
    assert value["reader_source_sha256"] == READER_SOURCE_SHA
    assert value["gateway_source_sha256"] == GATEWAY_SOURCE_SHA
    assert value["gateway_process_verified"] is True and value["run_identity"] == RUN
    assert value["model_requests"] == 0 and value["response_bodies_read"] is False
    return value


def health():
    value = get("/authenticated-health")
    assert value["status"] == "ready"
    return value


def validate_proof(job, path, proof):
    """Reject an ordinary HTTP 502, stale identity, or unbound request."""
    response = pilot.read(path)
    local_intent = pilot.read(path.with_name(path.name.replace(".response.", ".intent.")))
    index = int(path.name.split(".")[0])
    assert job["run_identity"] == RUN and 1 <= index <= (4 if job["arm"] == "skill" else 1)
    request_id = hashlib.sha256((RUN + ":" + job["operation_id"] + ":" + str(index)).encode()).hexdigest()
    assert proof["status"] == "receipt_available" and proof["request_id"] == request_id
    assert proof["schema_version"] == "harness3d.timeout-receipt-reader/1"
    assert proof["reader_source_sha256"] == READER_SOURCE_SHA
    assert proof["gateway_source_sha256"] == GATEWAY_SOURCE_SHA
    assert proof["gateway_process_verified"] is True and proof["run_identity"] == RUN
    assert proof["model_requests"] == 0 and proof["response_bodies_read"] is False
    identity = response["request_identity"]
    params = identity["parameters"]
    assert params["model"] == "qwen3.8-flash" and params["temperature"] == 0 and params["seed"] == 137
    assert params["enable_thinking"] is True and params["max_tokens"] == 8192
    assert params.get("top_p") is None and params.get("response_format") is None
    assert [i["jpeg_sha256"] for i in identity["images_in_order"]] == [i["sha256"] for i in job["images"]]
    assert local_intent["images"] == job["images"] and local_intent["reference_answer_visible"] is False
    upstream_intent = proof["intent"]
    assert upstream_intent["request_id"] == request_id and upstream_intent["audit_identity"] == {
        "run_identity": RUN, "operation_id": job["operation_id"]}
    assert upstream_intent["payload_sha256"] == identity["wire_payload_sha256"]
    assert all(params[k] == v for k, v in local_intent["parameters"].items())
    assert upstream_intent["upstream_timeout_s"] == 1200
    assert all(upstream_intent["parameters"][k] == params[k] for k in (
        "model", "temperature", "seed", "enable_thinking", "max_tokens"))
    assert response["status"] == "error" and response["error_type"] == "ServiceUnavailable"
    assert response["model_response"] is None and response["response_identity"] == {}
    local = response["service_error"]
    assert local["request_stage"] == "awaiting_headers" and local["timeout_s"] == 1230
    assert (local["reason_code"], local["exception_type"], local["http_status"]) in {
        ("http_error", "HTTPError", 502), ("timeout", "TimeoutError", None)}
    upstream = proof["response"]
    assert upstream["status"] == "failed" and upstream["error_type"] == "ServiceUnavailable"
    assert upstream["automatic_retry"] is False and upstream["body_sha256"] is None
    error = upstream["service_error"]
    assert error["schema"] == "bailian-service-error/1" and error["timeout_s"] == 1200
    assert (error["reason_code"], error["exception_type"], error["http_status"], error["request_stage"]) == (
        "timeout", "TimeoutError", None, "awaiting_headers")
    assert error["elapsed_seconds"] >= 1190
    for name in ("intent_sha256", "response_sha256"):
        assert isinstance(proof[name], str) and len(proof[name]) == 64
    return identity


def proof_for(job, path):
    index = int(path.name.split(".")[0])
    target = PROOFS / (job["operation_id"] + "-" + str(index) + ".json")
    with target.with_suffix(".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return locked_proof(job, path, target, index)


def locked_proof(job, path, target, index):
    if target.exists():
        receipt = pilot.read(target)
        proof = receipt["upstream_receipt"]
        assert receipt["job_sha256"] == pilot.sha_value(job) and receipt["local_response_sha256"] == pilot.sha(path)
        assert receipt["local_intent_sha256"] == pilot.sha(path.with_name(path.name.replace(".response.", ".intent.")))
    else:
        request_id = hashlib.sha256((RUN + ":" + job["operation_id"] + ":" + str(index)).encode()).hexdigest()
        proof = get("/request-receipt/" + request_id)
        validate_proof(job, path, proof)
        pilot.write(target, {"job_sha256": pilot.sha_value(job), "at": pilot.now(),
                             "local_response_sha256": pilot.sha(path),
                             "local_intent_sha256": pilot.sha(path.with_name(path.name.replace(".response.", ".intent."))),
                             "upstream_receipt": proof, "automatic_retry": False})
    return validate_proof(job, path, proof), target


def audit_operation(job):
    """May run between ledger error and worker result publication."""
    folder = ROOT / "request_records" / job["operation_id"]
    responses = sorted(folder.glob("*.response.json"))
    assert 1 <= len(responses) <= (4 if job["arm"] == "skill" else 1)
    assert [p.name for p in responses] == [f"{i:03d}.response.json" for i in range(1, len(responses) + 1)]
    assert {p.name for p in folder.glob("*.json")} == {
        f"{i:03d}{suffix}" for i in range(1, len(responses) + 1) for suffix in (".intent.json", ".response.json")}
    for path in responses[:-1]:
        received = pilot.read(path)
        assert received["status"] == "received"
        assert received["response_identity"]["model"] == "qwen3.8-flash"
        assert received["response_identity"]["model_reported"] is True
    _, proof_path = proof_for(job, responses[-1])
    return responses, proof_path


def ledger_errors(job, require_all=True):
    rows = {}
    for name in ("requests", "test_budget"):
        state = json.loads((ROOT / name / "state.json").read_text())
        errors = [r for r in state["requests"] if r["context"]["variant_id"] == job["operation_id"]
                  and r["status"] == "error"]
        assert len(errors) <= 1
        if require_all:
            assert len(errors) == 1
        if errors:
            row = errors[0]
            assert row["context"]["run_identity"] == RUN and row["exception_type"] == "ServiceUnavailable"
            rows[name] = pilot.sha_value(row)
    return rows


def verify_handled(job):
    receipt = pilot.read(HANDLED / (job["operation_id"] + ".json"))
    assert receipt["operation_id"] == job["operation_id"] and receipt["job_sha256"] == pilot.sha_value(job)
    assert receipt["automatic_retry"] is False and receipt["score_policy"] == "zero"
    responses, proof = audit_operation(job)
    path = ROOT / "results" / job["arm"] / (job["operation_id"] + ".json")
    assert receipt["result_sha256"] == pilot.sha(path) and receipt["proof_sha256"] == pilot.sha(proof)
    assert receipt["ledger_error_hashes"] == ledger_errors(job)
    assert receipt["request_record_hashes"] == {
        str(p.relative_to(ROOT)): pilot.sha(p) for p in responses[0].parent.glob("*.json")}
    return receipt


def handle(job, error):
    assert isinstance(error, Exception)
    assert not (HANDLED / (job["operation_id"] + ".json")).exists()
    responses, proof = audit_operation(job)
    ledgers = ledger_errors(job)
    records = {str(p.relative_to(ROOT)): pilot.sha(p) for p in responses[0].parent.glob("*.json")}
    path = ROOT / "results" / job["arm"] / (job["operation_id"] + ".json")
    if not path.exists():
        assert job["arm"] == "direct" and type(error).__name__ == "ServiceUnavailable"
        problem = job["problem"]
        pilot.write(path, {
            "status": "stopped", "operation_id": job["operation_id"], "job_sha256": pilot.sha_value(job),
            "result": {"benchmark": problem["benchmark"], "sample_id": problem["sample_id"],
                       "source_split": "test", "task_type": problem["task_type"],
                       "input_sha256": job["input_sha256"], "answer": None, "status": "service_error",
                       "request_count": 1, "attempted_tool_calls": 0, "succeeded_tool_calls": 0,
                       "denied_authorizations": 0, "rounds": [], "answer_parse_method": None},
            "observations": [], "seconds": pilot.read(responses[-1])["seconds"],
            "request_records": records, "reference_answer_visible": False,
            "host_recovery": {"kind": "verified_upstream_timeout", "automatic_retry": False,
                              "resume_attempt": 8}})
    row = pilot.read(path)
    assert row["status"] == "stopped" and row["job_sha256"] == pilot.sha_value(job)
    assert row["operation_id"] == job["operation_id"] and row["reference_answer_visible"] is False
    assert row["result"]["status"] == "service_error" and row["result"]["answer"] is None
    assert row["result"]["request_count"] == len(responses) and row["request_records"] == records
    pilot.write(HANDLED / (job["operation_id"] + ".json"), {
        "status": "timeout_recorded_continue", "at": pilot.now(), "operation_id": job["operation_id"],
        "job_sha256": pilot.sha_value(job), "result_sha256": pilot.sha(path),
        "proof_sha256": pilot.sha(proof), "request_record_hashes": records,
        "ledger_error_hashes": ledgers, "automatic_retry": False, "score_policy": "zero",
        "upstream_timeout_s": 1200, "client_timeout_s": 1230})
    verify_handled(job)
    return row

RETAINED_400_OP = "421c444704b3f26d4386d13bea79fd2c31cdbceb2fc40077bb20031aeaaa2715"

def validate_retained400(job, path, proof):
    """Audit one already failed HTTP 400; never used to recover a future error."""
    assert job["operation_id"] == RETAINED_400_OP
    assert pilot.sha_value(job) == "0250663ec20aa208f8e58b11301a001d0a6e7b459c3266eb714dd0ae0f2b67b8"
    assert path.name == "001.response.json"
    assert pilot.sha(path) == "af6431fd7610822670c0b24068a19e2148178e25fed875b416209ddbb5817440"
    assert proof["intent_sha256"] == "78212998e063a8d339bc44bc6dc52d66d28e0912967ca39a4c9a47345bb2a492"
    assert proof["response_sha256"] == "ca1a036978ab0c0ca8590649ba1b5ef3b4bf89981173aa5978def23dca978551"
    response = pilot.read(path)
    local_intent = pilot.read(path.with_name(path.name.replace(".response.", ".intent.")))
    index = int(path.name.split(".")[0])
    assert job["run_identity"] == RUN and 1 <= index <= (4 if job["arm"] == "skill" else 1)
    request_id = hashlib.sha256((RUN + ":" + job["operation_id"] + ":" + str(index)).encode()).hexdigest()
    assert proof["status"] == "receipt_available" and proof["request_id"] == request_id
    assert proof["schema_version"] == "harness3d.timeout-receipt-reader/1"
    assert proof["reader_source_sha256"] == READER_SOURCE_SHA
    assert proof["gateway_source_sha256"] == GATEWAY_SOURCE_SHA
    assert proof["gateway_process_verified"] is True and proof["run_identity"] == RUN
    assert proof["model_requests"] == 0 and proof["response_bodies_read"] is False
    identity = response["request_identity"]
    params = identity["parameters"]
    assert params["model"] == "qwen3.8-flash" and params["temperature"] == 0 and params["seed"] == 137
    assert params["enable_thinking"] is True and params["max_tokens"] == 8192
    assert params.get("top_p") is None and params.get("response_format") is None
    assert [i["jpeg_sha256"] for i in identity["images_in_order"]] == [i["sha256"] for i in job["images"]]
    assert local_intent["images"] == job["images"] and local_intent["reference_answer_visible"] is False
    upstream_intent = proof["intent"]
    assert upstream_intent["request_id"] == request_id and upstream_intent["audit_identity"] == {
        "run_identity": RUN, "operation_id": job["operation_id"]}
    assert upstream_intent["payload_sha256"] == identity["wire_payload_sha256"]
    assert all(params[k] == v for k, v in local_intent["parameters"].items())
    assert upstream_intent["upstream_timeout_s"] == 1200
    assert all(upstream_intent["parameters"][k] == params[k] for k in (
        "model", "temperature", "seed", "enable_thinking", "max_tokens"))
    assert response["status"] == "error" and response["error_type"] == "ServiceUnavailable"
    assert response["model_response"] is None and response["response_identity"] == {}
    local = response["service_error"]
    assert local["request_stage"] == "awaiting_headers" and local["timeout_s"] == 1230
    assert (local["reason_code"], local["exception_type"], local["http_status"]) in {
        ("http_error", "HTTPError", 502), ("timeout", "TimeoutError", None)}
    upstream = proof["response"]
    assert upstream["status"] == "failed" and upstream["error_type"] == "RuntimeError"
    assert upstream["automatic_retry"] is False and upstream["body_sha256"] is None
    error = upstream["service_error"]
    assert error["schema"] == "bailian-service-error/1" and error["timeout_s"] == 1200
    assert (error["reason_code"], error["exception_type"], error["http_status"], error["request_stage"]) == (
        "http_error", "HTTPError", 400, "awaiting_headers")
    assert 0 <= error["elapsed_seconds"] < 2
    for name in ("intent_sha256", "response_sha256"):
        assert isinstance(proof[name], str) and len(proof[name]) == 64
    return identity


