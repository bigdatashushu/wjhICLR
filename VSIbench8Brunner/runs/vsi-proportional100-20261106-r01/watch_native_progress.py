#!/usr/bin/env python3
"""Bounded read-only monitoring of the existing fixed-panel experiment."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import collect_proportional100 as collector

CONTROL = "/Users/bytedance/.cache/harness3d-ssh/vsi-proportional100-persistent"
PROBE = """
import fcntl,json
from pathlib import Path
r=Path(%r); a=r/"harness"; z={}
for name, relative in (
    ("final_test_admission_status.json", "recovery_r02/status.json"),
    ("final_test_admission_failed.json", "recovery_r02/execute_failed.json")):
    p=r/relative
    z[name]=json.loads(p.read_text()) if p.exists() else None
launch=json.loads((r/"harness-recovery-r02.pid.json").read_text())
p=Path("/proc")/str(launch["pid"]); alive=p.exists()
if alive:
    fields=p.joinpath("stat").read_text().rsplit(")",1)[1].split()
    assert fields[19]==launch["process_identity"]["start_ticks"]
    assert Path("/proc/sys/kernel/random/boot_id").read_text().strip()==launch["process_identity"]["boot_id"]
    alive=fields[0]!="Z"
z["admission_alive"]=alive
z["continuation_current"]=json.loads((r/"harness.pid.json").read_text())["pid"]==launch["pid"]
outcomes={p.stem for p in (a/"outcomes").glob("*.json")}
issued={p.parent.name for p in (a/"solver_requests").glob("*/*.json")}
z["answered"]=len(outcomes & issued)
z["legal"]=None
z["gated"]=len(outcomes-issued)
z["host_errors"]=len(list((a/"host_errors").glob("*.json")))
z["scenes_verified"]=len(list((a/"scene_audits").glob("*.json")))
z["bindings_written"]=len(list((a/"bindings").glob("*.json")))
with (a/"request_budget.json").open() as handle:
    fcntl.flock(handle, fcntl.LOCK_SH)
    b=json.load(handle)
z["preparation_requests"]=b["used"]
z["preparation_request_cap"]=b["max_requests"]
z["paired_audit_ready"]=(r/"paired_native_verified.json").exists()
print(json.dumps(z))
""" % collector.REMOTE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=50)
    args = parser.parse_args()
    deadline, previous = time.monotonic() + args.seconds, None
    while True:
        status = collector.query(CONTROL)
        harness = status["arms"]["harness"]
        state = harness["state"]
        extra = json.loads(collector.remote(CONTROL, PROBE))
        output = {
            "direct_native": status["arms"]["direct"]["native_trajectories"],
            "harness_native": harness["native_trajectories"],
            "harness_status": state["status"],
            "harness_alive": harness["alive"],
            "pending": state.get("pending_tasks"),
            "active": len(state.get("active_tasks", [])),
            "answered": extra["answered"],
            "legal": extra["legal"],
            "gated": extra["gated"],
            "scenes_verified": extra["scenes_verified"],
            "bindings_written": extra["bindings_written"],
            "preparation_requests": extra["preparation_requests"],
            "preparation_request_cap": extra["preparation_request_cap"],
            "admission_status": (extra["final_test_admission_status.json"] or {}).get("status"),
            "admission_alive": extra["admission_alive"],
            "host_errors": extra["host_errors"],
            "paired_audit_ready": extra["paired_audit_ready"],
        }
        signature = output
        if signature != previous:
            print(json.dumps(output, ensure_ascii=False), flush=True)
            previous = signature
        if (extra["final_test_admission_failed.json"] is not None
                or extra["continuation_current"] and (
                    extra["host_errors"] or state["status"].startswith("stopped")
                    and not harness["alive"])
                and (extra["final_test_admission_status.json"] or {}).get("status") == "running"):
            if extra["final_test_admission_failed.json"]:
                print(json.dumps(extra["final_test_admission_failed.json"]), flush=True)
            raise RuntimeError("existing experiment requires inspection; no retry issued")
        complete = all(
            arm["state"]["status"] == "complete" and arm["native_trajectories"] == 100
            for arm in status["arms"].values()
        )
        if complete or time.monotonic() >= deadline:
            break
        time.sleep(min(10, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
