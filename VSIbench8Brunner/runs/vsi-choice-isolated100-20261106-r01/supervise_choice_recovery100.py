#!/usr/bin/env python3
"""Finish the authorized paired run after preserving the zero-call failed attempt."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from launch_choice_paired100 import LOCAL, REMOTE, remote, launch
from launch_choice_recovery100 import launch as launch_recovery
from supervise_choice_paired100 import query, wait, launch_policy
from collect_choice_paired100 import collect, score


def query_recovery(phase):
    code = """import json,time
from pathlib import Path
r=Path(%r);phase=%r;p=r/(phase+'_recovery_r02.pid.json')
if not p.exists():
    print(json.dumps({'launched':False}))
else:
    receipt=json.loads(p.read_text());proc=Path('/proc')/str(receipt['pid']);alive=proc.exists()
    if alive:
        fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
        assert fields[19]==receipt['process_identity']['start_ticks']
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==receipt['process_identity']['boot_id']
        alive=fields[0]!='Z'
    state=json.loads((r/'harness/live_status.json').read_text()) if (r/'harness/live_status.json').exists() else {}
    ready=(r/'recovery_compatibility_verified.json').exists() if phase=='compatibility' else (
        state.get('status')=='complete' and state.get('completed')==100)
    print(json.dumps({'phase':phase+'_recovery_r02','launched':True,'alive':alive,
        'ready':ready,'elapsed_s':time.time()-receipt['at'],'completed':state.get('completed',0),
        'active_requests':len(state.get('active_qwen_requests',[]))}))
""" % (REMOTE, phase)
    return json.loads(remote(code, timeout=30))


def wait_recovery(phase):
    deadline = time.monotonic() + 6 * 3600
    previous = None
    while True:
        state = query_recovery(phase)
        key = (state.get("ready"), state.get("completed"), int(state.get("elapsed_s", 0) // 30))
        if key != previous:
            print(json.dumps(state), flush=True)
            (LOCAL / "recovery_supervisor_live_status.json").write_text(
                json.dumps(state, indent=2) + "\n", encoding="utf-8")
            previous = key
        if state.get("ready") and not state.get("alive"):
            return
        if not state.get("launched") or not state.get("alive"):
            raise RuntimeError("owned corrected phase stopped without its success receipt")
        if time.monotonic() > deadline:
            raise RuntimeError("owned corrected phase exceeded its bounded wait")
        time.sleep(5)


def collect_recovery():
    names = ["recovery_compatibility_verified.json", "recovery_r02/admission.json",
             "compatibility_recovery_r02.pid.json", "execute_recovery_r02.pid.json",
             "policy.pid.json", "choice_policy_audit_source_identity.json"]
    code = """import json
from pathlib import Path
r=Path(%r)
print(json.dumps({name:json.loads((r/name).read_text()) for name in %r}))
""" % (REMOTE, names)
    for name, value in json.loads(remote(code, timeout=120)).items():
        path = LOCAL / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()
    with (LOCAL / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for phase in ("compatibility", "execute"):
            if not query_recovery(phase).get("launched"):
                launch_recovery(phase)
            wait_recovery(phase)
        if not query("verify").get("launched"):
            launch("verify")
        wait("verify")
        if not query("policy").get("launched"):
            launch_policy()
        wait("policy")
        collect()
        collect_recovery()
        score(args.metadata)


if __name__ == "__main__":
    main()
