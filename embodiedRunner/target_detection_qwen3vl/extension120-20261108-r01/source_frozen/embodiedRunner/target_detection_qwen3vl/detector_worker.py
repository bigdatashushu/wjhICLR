"""Stdlib SSH worker; authentication is loaded only in memory on the service host."""
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


class Failure(Exception):
    def __init__(self, code):
        self.code = code


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def process_identity(service):
    proc = Path("/proc") / str(service["auth_process_pid"])
    # Fields following ')' begin at proc stat field 3; starttime is field 22.
    stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if (stat[19] != str(service["auth_process_start_ticks"])
            or boot != service["auth_process_boot_id"]):
        raise Failure("qwen_service_process_identity_changed")


def authorize(service):
    process_identity(service)
    for relative, expected in service["model_files_sha256"].items():
        path = Path(service["model_snapshot"]) / relative
        if digest(path.read_bytes()) != expected:
            raise Failure("qwen_model_metadata_identity_changed")
    environment = (Path("/proc") / str(service["auth_process_pid"]) / "environ").read_bytes()
    keys = [entry.partition(b"=")[2] for entry in environment.split(b"\0")
            if entry.startswith(b"VLLM_API_KEY=")]
    if len(keys) != 1 or not keys[0]:
        raise Failure("qwen_service_auth_unavailable")
    process_identity(service)
    return keys[0].decode()


def http(service, key, suffix, payload=None):
    data = canonical(payload) if payload is not None else None
    request = urllib.request.Request(
        service["base_url"].rstrip("/") + suffix, data=data,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=service["timeout_s"]) as response:
            raw = response.read(4 * 1024**2 + 1)
        if len(raw) > 4 * 1024**2:
            raise Failure("qwen_response_too_large")
        if key.encode() in raw:
            raise Failure("qwen_response_contains_authentication")
        return json.loads(raw), digest(raw)
    except urllib.error.HTTPError as exc:
        raise Failure("qwen_http_" + str(exc.code)) from None
    except (urllib.error.URLError, TimeoutError):
        raise Failure("qwen_service_transport_error") from None


def main(value):
    service = value["service"]
    key = authorize(service)
    models, _ = http(service, key, "/models")
    ids = [row["id"] for row in models["data"]]
    if ids != [service["model"]]:
        raise Failure("qwen_served_model_identity_mismatch")
    report = {
        "status": "ok", "model": service["model"], "precision": "FP8",
        "service_identity_sha256": digest(canonical(service)),
        "model_files_sha256": service["model_files_sha256"],
        "automatic_retry": False, "reference_answer_visible": False,
        "detector_requests": 0,
    }
    if value["mode"] == "status":
        return report
    if value["mode"] != "detect" or value["payload"]["model"] != service["model"]:
        raise Failure("invalid_qwen_worker_request")
    started = time.monotonic()
    response, response_hash = http(service, key, "/chat/completions", value["payload"])
    process_identity(service)
    if response.get("model") != service["model"] or len(response["choices"]) != 1:
        raise Failure("qwen_response_identity_mismatch")
    choice = response["choices"][0]
    if choice["finish_reason"] != "stop":
        raise Failure("qwen_detection_output_incomplete")
    output = json.loads(choice["message"]["content"])
    usage = {k: v for k, v in response.get("usage", {}).items()
             if k in ("prompt_tokens", "completion_tokens", "total_tokens") and type(v) is int}
    report.update(output=output, request_sha256=digest(canonical(value["payload"])),
                  response_sha256=response_hash, output_sha256=digest(canonical(output)),
                  detector_requests=1, usage=usage, elapsed_seconds=time.monotonic() - started)
    return report


if __name__ == "__main__":
    try:
        result = main(INPUT)
        print(canonical(result).decode())
    except Failure as exc:
        print(canonical({"status": "error", "code": exc.code, "automatic_retry": False}).decode())
        sys.exit(2)
    except Exception:
        # Never emit exception strings, credentials, provider bodies or environments.
        print('{"status":"error","code":"qwen_worker_protocol_error","automatic_retry":false}')
        sys.exit(2)
