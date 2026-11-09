"""Use the already loaded GPU-1 replica and validate its startup/process/GPU."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

STAGE = "starting"

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def context():
    global STAGE
    STAGE = "native_receipt"
    root = Path("/home/cvailab/wangjh/harness3d/detector-optimization-20261022")
    receipt_path = root / "native-g1/groundingdino_startup.json"
    receipt = json.loads(receipt_path.read_text())
    pin = receipt["process"]
    proc = Path("/proc") / str(pin["pid"])
    command = (proc / "cmdline").read_bytes()
    argv = command.decode().split("\0")
    STAGE = "native_command"
    assert argv[argv.index("--gpu") + 1] == "1", "native_gpu_outside_authorized_set"
    assert argv[argv.index("--port") + 1] == "20030", "gateway_command_identity_mismatch"
    stat = (proc / "stat").read_text().split(") ", 1)[1].split()
    STAGE = "native_lifetime"
    assert int(stat[19]) == int(pin["start_ticks"]), "gateway_process_identity_mismatch"
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip() == pin["boot_id"]
    assert proc.stat().st_uid == os.getuid()
    STAGE = "native_gpu_mapping"
    gpu_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], timeout=15).decode()
    indexes = {line.split(",")[1].strip(): int(line.split(",")[0]) for line in gpu_rows.splitlines()}
    process_rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], timeout=15).decode()
    bindings = [indexes[line.split(",")[1].strip()] for line in process_rows.splitlines()
                if int(line.split(",")[0]) == int(pin["pid"])]
    assert bindings == [1], "native_gpu_outside_authorized_set"
    model = receipt["generation_identity"]
    expected = sha(canonical(model))
    assert model["model"] == "GroundingDINO_SwinB" and model["mode"] == "eval"
    physical = [{"port": 20030, "pid": int(pin["pid"]), "physical_gpu": 1,
                 "command_sha256": sha(command), "receipt_sha256": sha(receipt_path.read_bytes())}]
    return expected, physical


def exchange(path, payload=None, timeout=180):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None
    headers = {"Content-Type": "application/json"}
    body = None if payload is None else canonical(payload)
    request = urllib.request.Request("http://127.0.0.1:20030" + path, data=body, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    start = time.monotonic()
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
        assert len(raw) <= 8 * 1024 * 1024, "invalid_response_payload"
        pins = response.headers.get_all("X-Detector-Model-Sha256", [])
        assert len(pins) <= 1, "duplicate_model_pin"
        return raw, pins[0] if pins else None, time.monotonic() - start


def main():
    global STAGE
    expected, physical = context()
    assert expected == INPUT["model_sha256"], "detector_response_pin_mismatch"
    STAGE = "native_health_request"
    raw, _, elapsed = exchange("/health", timeout=60)
    value = json.loads(raw)
    STAGE = "native_health_validation"
    assert value.get("model_loaded", value.get("status") in ("healthy", "ok", "健康")) is True
    report = {"status": "ok", "model_sha256": expected, "physical_bindings": physical,
              "checked_at": datetime.now(timezone.utc).isoformat(),
              "health_response_sha256": sha(raw), "automatic_retry": False,
              "identity_verified_by": "startup_receipt_process_boot_id_command_and_nvidia_gpu"}
    if INPUT["mode"] == "status":
        report.update(model_requests=0, detector_requests=0,
                      health_keys=list(value), health_response_sha256=sha(raw), health_elapsed_s=elapsed)
        for key in ("status", "healthy", "ready", "enabled", "device", "model_loaded"):
            if isinstance(value.get(key), (str, int, bool)):
                report[key if key != "status" else "native_health_status"] = value[key]
    else:
        STAGE = "native_inference"
        payload = INPUT["payload"]
        raw, pin, elapsed = exchange("/infer", payload)
        assert pin is None or pin == expected, "detector_response_pin_mismatch"
        value = json.loads(raw)
        assert value.get("success") is True and isinstance(value.get("detections"), list)
        ids = set()
        for row in value["detections"]:
            assert isinstance(row["label"], str) and len(row["bbox"]) == 4
            assert all(type(x) in (int, float) and math.isfinite(x) for x in row["bbox"])
            assert (type(row["confidence"]) in (int, float) and math.isfinite(row["confidence"])
                    and 0 <= row["confidence"] <= 1)
            assert type(row["id"]) is int and row["id"] not in ids
            ids.add(row["id"])
            x1, y1, x2, y2 = row["bbox"]
            assert 0 <= x1 < x2 <= 500 and 0 <= y1 < y2 <= 500
        report.update(detector_requests=1, request_sha256=sha(canonical(payload)),
                      response_sha256=sha(raw), elapsed_s=elapsed, detections=value["detections"])
    print(canonical(report).decode())


try:
    main()
except urllib.error.HTTPError as exc:
    print(canonical({"status": "error", "code": "detector_http_error", "http_status": exc.code,
                     "automatic_retry": False}).decode())
except Exception as exc:
    known = {
        "gateway_command_identity_mismatch", "gateway_process_identity_mismatch",
        "gateway_authorization_missing", "native_gpu_outside_authorized_set",
        "native_model_pins_differ", "detector_response_pin_mismatch",
        "invalid_response_payload", "duplicate_model_pin",
    }
    code = exc.args[0] if type(exc) is AssertionError and exc.args and exc.args[0] in known else "detector_preflight_or_protocol_error"
    print(canonical({"status": "error", "code": code,
                     "exception_type": type(exc).__name__, "stage": STAGE,
                     "automatic_retry": False}).decode())
