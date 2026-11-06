#!/usr/bin/env python3
"""Wait for the owned input preflight, launch Harness once, then collect the pair."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_proportional100 import LOCAL, REMOTE, ROOT
from collect_proportional100 import remote, write


def probe(control, name):
    code = """import json,time
from pathlib import Path
r=Path(%r); launch=json.loads((r/(%r+'.pid.json')).read_text())
p=Path('/proc')/str(launch['pid']); alive=p.exists()
if alive:
    fields=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
    assert fields[19]==launch['process_identity']['start_ticks']
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==launch['process_identity']['boot_id']
    alive=fields[0]!='Z'
marker=r/'harness/preflight_verified.json'
receipt=json.loads(marker.read_text()) if marker.exists() else None
log=r/(%r+'.log'); progress=None
if log.exists():
    with log.open('rb') as f:
        f.seek(max(0,log.stat().st_size-131072))
        for line in f.read().decode('utf-8',errors='replace').splitlines():
            try: row=json.loads(line)
            except ValueError: continue
            if 'verified' in row: progress={'verified':row['verified'],'total':row['total']}
io={}
if alive:
    io=dict(x.split(': ',1) for x in (p/'io').read_text().splitlines())
print(json.dumps({'checked_at':time.time(),'alive':alive,'process_identity':launch['process_identity'],
    'progress':progress,'read_bytes':int(io.get('read_bytes',0)),
    'receipt':{k:receipt.get(k) for k in ('questions_verified','scenes_verified','sampling_sha256',
        'model_requests','gpu_requests','reference_answers_read')} if receipt else None,
    'harness_launch_exists':(r/'harness.pid.json').exists()}))
""" % (REMOTE, name, name)
    return json.loads(remote(control, code))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--preflight-name", default="harness-preflight02")
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    env = {**os.environ, "VSI8B_SSH_CONTROL_TITAN": args.control_path,
           "PYTHONPATH": str(ROOT / "src") + ":" + str(ROOT / "scripts")}
    with (LOCAL / ".continuation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous, deadline = None, time.monotonic() + 2 * 3600
        while True:
            state = probe(args.control_path, args.preflight_name)
            write(LOCAL / "harness_preflight_live.json", state)
            counts = (state["alive"], (state["progress"] or {}).get("verified", 0),
                      (state["receipt"] or {}).get("questions_verified", 0))
            if counts != previous:
                print(json.dumps(state), flush=True)
                previous = counts
            receipt = state["receipt"]
            if receipt:
                sampling = json.loads((LOCAL / "experiment_spec.json").read_text())
                if (receipt["questions_verified"] != 100
                        or receipt["sampling_sha256"] != sampling["sampling_sha256"]
                        or receipt["model_requests"] != 0 or receipt["gpu_requests"] != 0
                        or receipt["reference_answers_read"] is not False):
                    raise ValueError("owned preflight receipt does not match the fixed panel")
                break
            if not state["alive"]:
                write(LOCAL / "preflight_stopped.json", state)
                raise RuntimeError("owned preflight stopped before verification")
            if time.monotonic() > deadline:
                raise RuntimeError("owned input preflight exceeded its two-hour bound")
            time.sleep(15)
        if not state["harness_launch_exists"]:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/launch_proportional100.py"),
                 "--arm", "harness", "--phase", "execute"],
                cwd=str(ROOT), env=env, check=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/collect_proportional100.py"),
             "--control-path", args.control_path, "--metadata", str(args.metadata)],
            cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
