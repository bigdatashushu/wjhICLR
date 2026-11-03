"""Stream verified MMSI progress from existing receipts; no model or GT reads."""
from collections import Counter
import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import select
import sys
import time
import traceback

ROOT = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/mmsi-365-test-rerun-20261101")
sys.path.insert(0, str(ROOT / "ops"))
import resume_control as control
startup = control.pilot.read(ROOT / "startup.json")
RESUME_ATTEMPT = startup.get("resume_attempt", 2)
if RESUME_ATTEMPT == 8:
    import resume08_control as control
elif RESUME_ATTEMPT == 7:
    import resume07_control as control
elif RESUME_ATTEMPT == 6:
    import resume06_control as control
elif RESUME_ATTEMPT == 5:
    import resume05_control as control
elif RESUME_ATTEMPT == 4:
    import resume04_control as control
elif RESUME_ATTEMPT == 3:
    import resume03_control as control
else:
    assert RESUME_ATTEMPT == 2
RETAINED_SERVICE_ERRORS = startup.get("retained_service_errors", 1)
pilot = control.pilot
manifest = pilot.read(ROOT / "manifest.json")
assert manifest["content_sha256"] == control.MANIFEST_SHA
jobs = {j["operation_id"]: j for j in manifest["jobs"]}
job_hashes = {op: pilot.sha_value(job) for op, job in jobs.items()}
cached = {}


def sealed(path, partial_ok=False):
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except json.JSONDecodeError:
        if partial_ok:
            return None, None
        raise
    assert value["content_sha256"] == pilot.sha_value(
        {k: v for k, v in value.items() if k != "content_sha256"})
    return value, hashlib.sha256(raw).hexdigest()


def snapshot():
    for arm in ("skill", "direct"):
        for path in (ROOT / "results" / arm).glob("*.json"):
            stat = path.stat()
            fingerprint = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            op = path.stem
            if op in cached and cached[op]["fingerprint"] == fingerprint:
                continue
            value, sha = sealed(path, partial_ok=True)
            if value is None:
                continue
            job = jobs[op]
            assert job["arm"] == arm and value["operation_id"] == op
            assert value["job_sha256"] == job_hashes[op]
            assert value["reference_answer_visible"] is False
            result = value["result"]
            assert result["input_sha256"] == job["input_sha256"] and result["source_split"] == "test"
            if op in cached:
                assert cached[op]["sha256"] == sha
            cached[op] = {
                "fingerprint": fingerprint, "sha256": sha, "event_id": op,
                "arm": arm, "ordinal": job["ordinal"], "task": result["task_type"],
                "prediction": result.get("answer"), "status": result["status"],
                "requests": result["request_count"], "tools": result["attempted_tool_calls"],
                "successful_tools": result["succeeded_tool_calls"], "seconds": value["seconds"],
                "at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
            }
    scores, _ = sealed(ROOT / "scores.json")
    grades = {r["operation_id"]: r for r in scores["rows"]}
    records, summary = [], {}
    for op, row in cached.items():
        grade = grades.get(op)
        if grade is not None:
            assert grade["prediction_sha256"] == row["sha256"]
            assert grade["prediction"] == row["prediction"] and grade["status"] == row["status"]
            assert grade["score"] in (0, 1)
        records.append({k: v for k, v in row.items() if k != "fingerprint"} |
                       {"score": grade["score"] if grade is not None else None})
    for arm in ("skill", "direct"):
        rows = [r for r in records if r["arm"] == arm]
        graded = [r for r in rows if r["score"] is not None]
        correct = sum(r["score"] for r in graded)
        summary[arm] = {
            "recorded": len(rows), "expected": 365, "graded": len(graded),
            "pending_grade": len(rows) - len(graded), "correct": correct,
            "accuracy_graded": correct / len(graded) if graded else None,
            "service_errors": sum(r["status"] == "service_error" for r in rows),
        }
    state = json.loads((ROOT / "requests/state.json").read_text())
    assert state["policy"] == manifest["shared_policy"]
    statuses = dict(Counter(r["status"] for r in state["requests"]))
    start = pilot.read(ROOT / "startup.json")
    assert start.get("resume_attempt", 2) == RESUME_ATTEMPT
    process = control.watcher.process_state(start)
    phase = "running" if process["alive"] else "runner_exited"
    final = None
    if statuses.get("error", 0) > RETAINED_SERVICE_ERRORS:
        if RESUME_ATTEMPT in (7, 8):
            handled = len(list((control.timeout_handling08.HANDLED if RESUME_ATTEMPT == 8 else control.timeout_handling07.HANDLED).glob("*.json")))
            phase = "handling_timeout" if statuses["error"] > RETAINED_SERVICE_ERRORS + handled else phase
        else:
            phase = "new_service_error"
    if RESUME_ATTEMPT == 8 and (ROOT / "progress.json").exists():
        progress = pilot.read(ROOT / "progress.json")
        if progress["status"] == "draining_after_fatal_error":
            phase = "draining_after_fatal_error"
    if (ROOT / "failure.json").exists():
        phase = "stopped"
    if (ROOT / "exit.json").exists():
        exit_receipt = pilot.read(ROOT / "exit.json")
        phase = "awaiting_acceptance" if exit_receipt["returncode"] == 0 else "stopped"
    acceptance = control.MONITOR / "final_acceptance.json"
    if acceptance.exists():
        final = pilot.read(acceptance)
        expected_status = ("complete_verified_with_recorded_service_errors" if RESUME_ATTEMPT >= 3
                           else "complete_verified_with_recorded_service_error")
        assert final["status"] == expected_status
        assert final["service_errors"] == statuses.get("error", 0)
        assert sum(summary[a]["service_errors"] for a in summary) == final["service_errors"]
        assert all(summary[a]["recorded"] == summary[a]["graded"] == 365 for a in summary)
        phase = "complete_verified"
    elif phase == "awaiting_acceptance" and any(
            (control.MONITOR / name).exists() for name in ("failure.json", "diagnosis.json")):
        phase = "acceptance_failed"
    return {
        "at": pilot.now(), "phase": phase, "summary": summary,
        "requests_used": len(state["requests"]), "requests_limit": state["policy"]["max_requests"],
        "request_statuses": statuses, "runner_process": process,
        "records": sorted(records, key=lambda r: (r["at"], r["arm"], r["ordinal"])),
        "final_acceptance": final, "model_requests_issued_by_monitor": 0,
        "reference_answers_read_by_monitor": False,
        "resume_attempt": RESUME_ATTEMPT, "retained_service_errors": RETAINED_SERVICE_ERRORS, "total_service_errors": statuses.get("error", 0),
    }


def stream():
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    assert fd >= 0
    mask = 0x00000008 | 0x00000080 | 0x00000100
    for folder in (ROOT, ROOT / "results/skill", ROOT / "results/direct",
                   ROOT / "requests", control.MONITOR):
        assert libc.inotify_add_watch(fd, os.fsencode(folder), mask) >= 0
    previous, signature, last_emit = {}, None, 0.0
    try:
        while True:
            value = snapshot()
            records = value.pop("records")
            current = {r["event_id"]: (r["sha256"], r["score"]) for r in records}
            initial = signature is None
            changes = records[-8:] if initial else [
                r | {"score_update": r["event_id"] in previous}
                for r in records if previous.get(r["event_id"]) != current[r["event_id"]]]
            # Timestamp is excluded so unchanged snapshots become 30-second heartbeats.
            comparison = {k: v for k, v in value.items() if k != "at"}
            new_signature = json.dumps(comparison, sort_keys=True, default=str)
            if changes or signature != new_signature or time.monotonic() - last_emit >= 30:
                value.update(initial=initial, records=changes)
                print(json.dumps(value, ensure_ascii=False), flush=True)
                previous, signature, last_emit = current, new_signature, time.monotonic()
            if value["phase"] in {"complete_verified", "stopped", "acceptance_failed", "runner_exited"}:
                return 0 if value["phase"] == "complete_verified" else 2
            ready, _, _ = select.select([fd], [], [], max(0, 30 - (time.monotonic() - last_emit)))
            if ready:
                os.read(fd, 65536)
    finally:
        os.close(fd)


if globals().get("LIVE_ONCE", False):
    print(json.dumps(snapshot(), ensure_ascii=False))
elif __name__ == "__main__":
    try:
        code = stream()
    except BrokenPipeError:
        code = 0
    except BaseException as exc:
        print(json.dumps({
            "phase": "reader_failed", "at": pilot.now(), "error_type": type(exc).__name__,
            "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                          for f in traceback.extract_tb(exc.__traceback__)],
            "model_requests_issued_by_monitor": 0,
        }), flush=True)
        code = 2
    sys.exit(code)
