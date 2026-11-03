"""Read-only recovery status; never print prompts, answers or credentials."""
from collections import Counter
import ctypes
import json
import os
from pathlib import Path
import select
import sys
import time

ROOT = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/mmsi-365-test-rerun-20261101")
sys.path.insert(0, str(ROOT / "ops"))
import resume_control as control
pilot = control.pilot
OPS = control.OPS


def snapshot():
    out = {"at": pilot.now(), "root": str(ROOT), "model_requests": 0, "status": "checking_preflight"}
    for name in ("resume02_preflight_startup.json", "resume02_preflight_progress.json",
                 "resume02_preflight_failure.json", "resume02_preflight_exit.json",
                 "resume02_launch.json", "resume02_launch_failure.json", "resume02_launch_exit.json"):
        if (OPS / name).exists():
            out[name] = pilot.read(OPS / name)
            if name.endswith("startup.json"):
                out[name]["process"] = control.watcher.process_state(out[name])
    if control.PREFLIGHT.exists():
        value = pilot.read(control.PREFLIGHT)
        out["preflight"] = {k: value[k] for k in (
            "status", "at", "manifest_sha256", "completed", "retained_request_count",
            "all_test_input_hashes_verified", "gateway", "maximum_total_mmsi_requests",
            "model_requests", "gpu_requests", "reference_answers_read", "six_skills_unchanged", "seconds")}
        out["preflight"]["remaining_episodes"] = len(value["pending_never_dispatched"])
        out["preflight"]["file_sha256"] = pilot.sha(control.PREFLIGHT)
        out["status"] = "ready_to_resume"
    if control.RECOVERY.exists():
        out["recovery"] = pilot.read(control.RECOVERY)
        out["status"] = "recovering"
        for name in ("startup.json", "progress.json", "failure.json", "exit.json", "completion.json"):
            if (ROOT / name).exists():
                value = pilot.read(ROOT / name)
                if name == "startup.json":
                    value["process"] = control.watcher.process_state(value)
                out[name] = value
        if out.get("startup.json", {}).get("process", {}).get("alive"):
            out["status"] = "running"
        if "failure.json" in out:
            out["status"] = "stopped_requires_diagnosis"
        if "completion.json" in out:
            out["status"] = "complete_with_recorded_service_error"
    if "resume02_preflight_failure.json" in out or "resume02_launch_failure.json" in out:
        out["status"] = "recovery_requires_diagnosis"
    state = json.loads((ROOT / "requests/state.json").read_text())
    out["requests"] = {"used": len(state["requests"]), "policy": state["policy"],
                       "statuses": dict(Counter(r["status"] for r in state["requests"]))}
    out["completed"] = {a: len(list((ROOT / "results" / a).glob("*.json"))) for a in ("skill", "direct")}
    out["new_results"] = 0
    out["new_responses"] = []
    if control.PREFLIGHT.exists() and control.RECOVERY.exists():
        baseline = pilot.read(control.PREFLIGHT)
        manifest = pilot.read(ROOT / "manifest.json")
        for job in manifest["jobs"]:
            op = job["operation_id"]
            if op not in baseline["pending_never_dispatched"]:
                continue
            if (ROOT / "results" / job["arm"] / (op + ".json")).exists():
                out["new_results"] += 1
            for path in sorted((ROOT / "request_records" / op).glob("*.response.json")):
                identity, received = control.audit_response(job, path)
                response = pilot.read(path)
                brief = {"arm": job["arm"], "question": job["ordinal"], "status": response["status"],
                         "seconds": response["seconds"], "received": received,
                         "parameters": identity["parameters"],
                         "original_image_bytes_and_order_verified": True,
                         "actual_model": response["response_identity"].get("model"),
                         "model_reported": response["response_identity"].get("model_reported"),
                         "response_sha256": pilot.sha(path)}
                out["new_responses"].append(brief)
        out["new_responses"] = out["new_responses"][:8]
    if (ROOT / "scores.json").exists():
        scores = pilot.read(ROOT / "scores.json")
        out["scores"] = {k: scores[k] for k in ("status", "at", "summary")}
    out["monitor"] = {}
    for name in ("startup.json", "progress.json", "diagnosis.json", "failure.json",
                 "exit.json", "final_acceptance.json"):
        path = control.MONITOR / name
        if path.exists():
            value = pilot.read(path)
            if name == "startup.json":
                value["process"] = control.watcher.process_state(value)
            out["monitor"][name] = value
    original = json.loads((ROOT.parent / "smallimage/requests/state.json").read_text())
    out["original_budget"] = {"used": len(original["requests"]),
                              "combined_used": len(original["requests"]) + len(state["requests"]),
                              "max_requests": 20000}
    return out


seconds = min(float(globals().get("WAIT_SECONDS", 0)), 45)
deadline = time.monotonic() + seconds
libc = ctypes.CDLL(None, use_errno=True)
fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
assert fd >= 0
try:
    for folder in (ROOT, OPS):
        assert libc.inotify_add_watch(fd, os.fsencode(folder), 0x00000008 | 0x00000080 | 0x00000100) >= 0
    while True:
        value = snapshot()
        if value["new_responses"] or value["status"] in {
                "recovery_requires_diagnosis", "stopped_requires_diagnosis", "complete_with_recorded_service_error"}:
            break
        if time.monotonic() >= deadline:
            break
        ready, _, _ = select.select([fd], [], [], max(0, deadline - time.monotonic()))
        if ready:
            os.read(fd, 65536)
    print(json.dumps(value, ensure_ascii=False))
finally:
    os.close(fd)
