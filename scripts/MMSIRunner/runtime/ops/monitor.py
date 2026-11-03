"""One-off read-only monitoring until the MMSI rerun exits."""
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import select
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
OPS = ROOT / "ops"
for key in list(os.environ):
    if any(s in key.upper() for s in ("API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY")):
        os.environ.pop(key, None)
sys.path.insert(0, str(ROOT))
import pilot
import watcher


def snapshot():
    state = json.loads((ROOT / "requests/state.json").read_text())
    counts = {arm: len(list((ROOT / "results" / arm).glob("*.json"))) for arm in ("skill", "direct")}
    out = {"status": "monitoring", "at": pilot.now(), "completed": counts,
           "expected": {"skill": 365, "direct": 365}, "model_requests_issued_by_monitor": 0,
           "requests_used": len(state["requests"]),
           "request_statuses": dict(Counter(r["status"] for r in state["requests"]))}
    if (ROOT / "startup.json").exists():
        out["runner_process"] = watcher.process_state(pilot.read(ROOT / "startup.json"))
    if (ROOT / "scores.json").exists():
        scores = pilot.read(ROOT / "scores.json")
        out["scores_status"] = scores["status"]
        out["summary"] = scores["summary"]
    for name in ("failure.json", "exit.json", "completion.json"):
        if (ROOT / name).exists():
            out[name] = pilot.read(ROOT / name)
    if "failure.json" in out or ("exit.json" in out and out["exit.json"]["returncode"] != 0):
        out["status"] = "diagnosis_required"
        out["automatic_model_retries"] = 0
    elif "exit.json" in out:
        out["status"] = "verifying_completion"
    elif out.get("runner_process", {}).get("alive") is False:
        out["status"] = "runner_missing_requires_diagnosis"
    return out


def verify_complete():
    manifest = pilot.read(ROOT / "manifest.json")
    state = pilot.verify(manifest)
    completion = pilot.read(ROOT / "completion.json")
    scores = pilot.read(ROOT / "scores.json")
    pairing = pilot.read(ROOT / "request_pairing_audit.json")
    predictions = pilot.read(ROOT / "predictions_frozen.json")
    assert pilot.read(ROOT / "exit.json")["returncode"] == 0
    assert not watcher.process_state(pilot.read(ROOT / "startup.json"))["alive"]
    assert completion["status"] == "complete" and completion["episodes"] == 730
    assert completion["manifest_sha256"] == manifest["content_sha256"]
    assert scores["status"] == "complete" and len(scores["rows"]) == 730
    assert pairing["status"] == "passed" and pairing["pairs"] == 365
    assert len(predictions["predictions"]) == 730
    for relative, expected in predictions["predictions"].items():
        assert pilot.sha(ROOT / relative) == expected
    for key, name in (("scores_sha256", "scores.json"),
                      ("request_pairing_audit_sha256", "request_pairing_audit.json"),
                      ("frozen_predictions_sha256", "predictions_frozen.json")):
        assert completion[key] == pilot.sha(ROOT / name)
    assert all(r["status"] == "completed" for r in state["requests"])
    assert len(state["requests"]) == pairing["requests"] <= 1900
    assert all(scores["summary"][a]["completed"] == scores["summary"][a]["expected"] == 365
               for a in ("skill", "direct"))
    original = json.loads((ROOT.parent / "smallimage/requests/state.json").read_text())
    assert len(original["requests"]) + len(state["requests"]) <= 20000
    pilot.write(OPS / "final_acceptance.json", {
        "status": "complete_verified", "at": pilot.now(), "questions": 365, "episodes": 730,
        "manifest_sha256": manifest["content_sha256"], "summary": scores["summary"],
        "by_task": scores["by_task"], "requests_used": len(state["requests"]),
        "original_plus_mmsi_requests": len(original["requests"]) + len(state["requests"]),
        "online_train_requests": 0, "model_requests_issued_by_monitor": 0,
        "completion_sha256": pilot.sha(ROOT / "completion.json"),
        "original_ledger_modified_by_monitor": False, "reference_answers_read_by_monitor": False})


def main():
    proc = Path("/proc") / str(os.getpid())
    fields = (proc / "stat").read_text().split(") ", 1)[1].split()
    pilot.write(OPS / "startup.json", {
        "pid": os.getpid(), "start_ticks": fields[19],
        "command_sha256": pilot.sha(proc / "cmdline"), "source_sha256": pilot.sha(__file__),
        "at": pilot.now(), "model_requests_issued_by_monitor": 0})
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    assert fd >= 0
    try:
        assert libc.inotify_add_watch(fd, os.fsencode(ROOT), 0x00000008 | 0x00000080 | 0x00000100) >= 0
        while True:
            value = snapshot()
            pilot.update(OPS / "progress.json", value)
            if value["status"] in ("diagnosis_required", "runner_missing_requires_diagnosis"):
                pilot.write(OPS / "diagnosis.json", value)
                return
            if value["status"] == "verifying_completion":
                verify_complete()
                return
            ready, _, _ = select.select([fd], [], [], 45)
            if ready:
                os.read(fd, 65536)
    finally:
        os.close(fd)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException as exc:
        code = 2
        pilot.write(OPS / "failure.json", {
            "status": "monitor_failure", "at": pilot.now(), "error_type": type(exc).__name__,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)],
            "model_requests_issued_by_monitor": 0})
    pilot.write(OPS / "exit.json", {"returncode": code, "at": pilot.now()})
