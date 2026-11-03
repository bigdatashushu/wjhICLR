"""Read-only resume 05 monitoring and acceptance after the runner actually exits."""
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import select
import sys
import traceback

import resume05_control as control

pilot = control.pilot
ROOT, MONITOR = control.ROOT, control.MONITOR


def snapshot():
    state = json.loads((ROOT / "requests/state.json").read_text())
    counts = {arm: len(list((ROOT / "results" / arm).glob("*.json"))) for arm in ("skill", "direct")}
    start = pilot.read(ROOT / "startup.json")
    assert start["resume_attempt"] == 5
    out = {"status": "monitoring", "at": pilot.now(), "completed": counts,
           "expected": {"skill": 365, "direct": 365}, "resume_attempt": 5,
           "runner_process": control.watcher.process_state(start),
           "requests_used": len(state["requests"]),
           "request_statuses": dict(Counter(r["status"] for r in state["requests"])),
           "model_requests_issued_by_monitor": 0, "automatic_model_retries": 0,
           "retained_service_errors": 3}
    if (ROOT / "scores.json").exists():
        scores = pilot.read(ROOT / "scores.json")
        out["scores"] = {k: scores[k] for k in ("status", "at", "summary")}
    for name in ("failure.json", "exit.json", "completion.json"):
        if (ROOT / name).exists():
            out[name] = pilot.read(ROOT / name)
    if "failure.json" in out or ("exit.json" in out and out["exit.json"]["returncode"] != 0):
        out["status"] = "diagnosis_required"
    elif "exit.json" in out:
        out["status"] = "waiting_for_runner_exit" if out["runner_process"]["alive"] else "verifying_completion"
    elif out["runner_process"]["alive"] is False:
        out["status"] = "runner_missing_requires_diagnosis"
    return out


def accept():
    manifest = control.manifest_current()
    baseline = pilot.read(control.PREFLIGHT)
    control.retained_audit(baseline, check_cache=True)
    states, original = control.ledger_audit(manifest, baseline)
    completion = pilot.read(ROOT / "completion.json")
    pairing = pilot.read(ROOT / "request_pairing_audit.json")
    frozen = pilot.read(ROOT / "predictions_frozen.json")
    scores = pilot.read(ROOT / "scores.json")
    assert pilot.read(ROOT / "exit.json")["returncode"] == 0
    assert not control.watcher.process_state(pilot.read(ROOT / "startup.json"))["alive"]
    assert completion["status"] == "complete" and completion["service_errors"] == 3
    assert completion["episodes"] == 730 and completion["manifest_sha256"] == control.MANIFEST_SHA
    assert pairing["status"] == "passed_with_recorded_service_errors"
    assert pairing["attempted_pairs"] == 365 and pairing["both_responses_received_pairs"] == 362
    assert set(pairing["service_error_operations"]) == set(control.FAILURES)
    assert len(frozen["predictions"]) == 730 and len(scores["rows"]) == 730
    count = 0
    for job in manifest["jobs"]:
        path = ROOT / "results" / job["arm"] / (job["operation_id"] + ".json")
        assert frozen["predictions"][str(path.relative_to(ROOT))] == pilot.sha(path)
        row, responses = control.result_audit(job, path)
        count += len(responses)
    assert count == pairing["requests"] == len(states["requests"]["requests"])
    for key, name in (("scores_sha256", "scores.json"),
                      ("request_pairing_audit_sha256", "request_pairing_audit.json"),
                      ("frozen_predictions_sha256", "predictions_frozen.json")):
        assert completion[key] == pilot.sha(ROOT / name)
    assert completion["recovery_receipt_sha256"] == pilot.sha(control.RECOVERY)
    assert scores["status"] == "complete"
    assert all(scores["summary"][a]["completed"] == scores["summary"][a]["expected"] == 365
               for a in ("skill", "direct"))
    failed = [r for r in scores["rows"] if r["status"] == "service_error"]
    assert len(failed) == 3 and {r["operation_id"] for r in failed} == set(control.FAILURES)
    assert all(r["score"] == 0 for r in failed)
    pilot.write(MONITOR / "final_acceptance.json", {
        "status": "complete_verified_with_recorded_service_errors", "at": pilot.now(),
        "questions": 365, "episodes": 730, "successful_response_pairs": 362, "service_errors": 3,
        "manifest_sha256": control.MANIFEST_SHA, "summary": scores["summary"], "by_task": scores["by_task"],
        "requests_used": count, "original_plus_mmsi_requests": len(original["requests"]) + count,
        "completion_sha256": pilot.sha(ROOT / "completion.json"),
        "model_requests_issued_by_monitor": 0, "reference_answers_read_by_monitor": False,
        "automatic_model_retries": 0, "other_experiments_resumed": False})


def main():
    pilot.write(MONITOR / "startup.json", {
        **control.process_receipt(), "source_sha256": pilot.sha(__file__),
        "model_requests_issued_by_monitor": 0})
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    assert fd >= 0
    try:
        assert libc.inotify_add_watch(fd, os.fsencode(ROOT), 0x00000008 | 0x00000080 | 0x00000100) >= 0
        while True:
            value = snapshot()
            pilot.update(MONITOR / "progress.json", value)
            if value["status"] in {"diagnosis_required", "runner_missing_requires_diagnosis"}:
                pilot.write(MONITOR / "diagnosis.json", value)
                return
            if value["status"] == "verifying_completion":
                accept()
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
        pilot.write(MONITOR / "failure.json", {
            "at": pilot.now(), "error_type": type(exc).__name__, "model_requests_issued_by_monitor": 0,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)]})
    pilot.write(MONITOR / "exit.json", {"returncode": code, "at": pilot.now()})
    sys.exit(code)
