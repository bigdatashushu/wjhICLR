#!/usr/bin/env python3
"""Inject authorized credentials transiently and launch the fixed Titan pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

from deploy_pilot_services import LAUNCH, ROOT, REMOTE_4090, REMOTE_TITAN, PYTHON_4090
from ssh_from_server import remote_run

PYTHON_TITAN = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "all", "solve"), default="preflight")
    parser.add_argument("--name", default="pilot40")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--solve-workers", type=int, default=16)
    parser.add_argument("--cpu-limit", type=int, default=8)
    parser.add_argument("--api-limit", type=int, default=32)
    parser.add_argument("--wait-for-pid-file")
    args = parser.parse_args()
    retrieve = """import json
from pathlib import Path
root=Path("/home/cvailab/wangjh/VSIbench8Brunner")
result={}
for name,key in [("sam2","VSI_SAM2_TOKEN"),("qwen8b","VLLM_API_KEY")]:
    pid=json.load((root/(name+".pid.json")).open())["pid"]
    env=dict(row.split(b"=",1) for row in (Path("/proc")/str(pid)/"environ").read_bytes().split(b"\\0") if b"=" in row)
    result[key]=env[key.encode()].decode()
print(json.dumps(result))
"""
    code, raw, _ = remote_run("cvailab@10.10.0.198",
                              PYTHON_4090 + " -c " + shlex.quote(retrieve),
                              return_output=True)
    if code:
        raise RuntimeError("owned service credential injection unavailable")
    service_env = json.loads(raw)
    text = (ROOT.parent / "qwenkey.md").read_text()
    keys = set(re.findall(r"\bsk-[A-Za-z0-9._~+/-]+=*", text.strip()))
    if len(keys) != 1:
        raise RuntimeError("authorized Qwen key must be unambiguous")
    key = keys.pop()
    env = {
        "PYTHONPATH":
        "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat:" +
        REMOTE_TITAN + "/src",
        "LD_PRELOAD":
        "/home/wangy/wangjh/harness3d/precompute-20261022-r01/cpu_sqlite/libsqlite3.so.0",
        "DASHSCOPE_API_KEY": key,
        "VSI_SAM2_TOKEN": service_env["VSI_SAM2_TOKEN"],
        "QWEN3VL8B_API_KEY": service_env["VLLM_API_KEY"],
        "VSI_SAM2_ENDPOINT": "http://10.10.0.198:28102",
        "VSI_QWEN8B_BASE_URL": "http://10.10.0.198:28101/v1",
        "VSI_QWEN8B_MODEL": "Qwen3-VL-8B-Instruct-FP8",
        "VSI_REVIEW_ENDPOINT": "http://127.0.0.1:27248",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
    }
    if args.name != "pilot40":
        env["VSI8B_REVIEW_REPLAY_ROOT"] = (
            REMOTE_TITAN + "/runs/vsi8b-pilot40-20261105-r02/review_replay")
    archive = subprocess.check_output([
        "tar", "--no-xattrs", "-czf", "-", "--exclude=./tests", "--exclude=./runs",
        "--exclude=__pycache__",
        "-C", str(ROOT), "."])
    if remote_run("wangy@10.10.0.34", "tar -xzf - -C " + REMOTE_TITAN,
                  input_bytes=archive):
        raise RuntimeError("pilot runtime upload failed")
    run = REMOTE_TITAN + "/runs/vsi8b-pilot40-20261105-r02"
    if args.parallel and args.phase != "all":
        raise ValueError("parallel scheduling requires the full continuation phase")
    command = [
        PYTHON_TITAN, "scripts/pilot40_parallel.py" if args.parallel else "scripts/pilot40.py",
        "--sampling", run + "/sampling_manifest.json", "--run-dir", run,
        "--published-inputs",
        "/data/wangy/wangjh/harness3d/experiment-inputs/vsibench-m4-success-1000-seed137/published_inputs.json",
    ]
    if args.parallel:
        command.extend(["--prepare-workers", str(args.prepare_workers),
                        "--solve-workers", str(args.solve_workers),
                        "--cpu-limit", str(args.cpu_limit), "--api-limit", str(args.api_limit)])
        if args.wait_for_pid_file:
            command.extend(["--wait-for-pid-file", args.wait_for_pid_file])
    else:
        command.extend(["--phase", args.phase])
    spec = {"name": args.name, "root": REMOTE_TITAN, "env": env, "command": command}
    if args.phase == "preflight":
        launch = """import json,os,subprocess,sys
spec=json.load(sys.stdin)
raise SystemExit(subprocess.run(spec["command"],cwd=spec["root"],
                               env={**os.environ,**spec["env"]}).returncode)
"""
    else:
        launch = LAUNCH
    return remote_run("wangy@10.10.0.34", PYTHON_TITAN + " -c " + shlex.quote(launch),
                      input_bytes=json.dumps(spec).encode())


if __name__ == "__main__":
    raise SystemExit(main())
