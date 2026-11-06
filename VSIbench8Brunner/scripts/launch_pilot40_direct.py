#!/usr/bin/env python3
"""Upload only Direct runtime files, precheck without auth, then launch once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex

from deploy_pilot_services import LAUNCH, REMOTE_TITAN, ROOT
from launch_pilot40 import PYTHON_TITAN
from ssh_from_server import remote_run

RUN_NAME = "vsi-direct40-thinking-20261105-r01"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for name in ("pilot40_direct.py", "export_pilot_trajectories.py"):
        target = REMOTE_TITAN + "/scripts/" + name
        upload = "import sys; from pathlib import Path; Path(" + repr(target) + ").write_bytes(sys.stdin.buffer.read())"
        if remote_run("wangy@10.10.0.34", PYTHON_TITAN + " -c " + shlex.quote(upload),
                      input_bytes=(ROOT / "scripts" / name).read_bytes()):
            raise RuntimeError("Direct runtime upload failed")
    command = [
        PYTHON_TITAN, "scripts/pilot40_direct.py",
        "--sampling", REMOTE_TITAN + "/runs/vsi8b-pilot40-20261105-r02/sampling_manifest.json",
        "--run-dir", REMOTE_TITAN + "/runs/" + RUN_NAME,
        "--official-root", "/home/wangy/wangjh/harness3d/four-datasets-20261031/vsi1000/official32",
        "--validator", "/home/wangy/wangjh/harness3d/four-datasets-20261031/official32_pipeline.py",
        "--workers", "16",
    ]
    env = {
        "PYTHONPATH": "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat:" +
                      REMOTE_TITAN + "/src",
        "LD_PRELOAD": "/home/wangy/wangjh/harness3d/precompute-20261022-r01/cpu_sqlite/libsqlite3.so.0",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
    }
    spec = {"root": REMOTE_TITAN, "name": "pilot40-direct01", "command": command, "env": env}
    if args.execute:
        text = (ROOT.parent / "qwenkey.md").read_text()
        keys = set(re.findall(r"\bsk-[A-Za-z0-9._~+/-]+=*", text.strip()))
        if len(keys) != 1:
            raise RuntimeError("authorized Qwen credential is ambiguous")
        env["DASHSCOPE_API_KEY"] = keys.pop()
        command.append("--execute")
        launcher = LAUNCH
    else:
        launcher = """import json,os,subprocess,sys
spec=json.load(sys.stdin)
env={key:value for key,value in os.environ.items() if key not in
    ("DASHSCOPE_API_KEY","QWEN3VL8B_API_KEY","VSI_SAM2_TOKEN","OPENROUTER_API_KEY",
     "MAKERS_MODELS_KEY","VSI8B_SSH_PASSWORD")}
env.update(spec["env"])
raise SystemExit(subprocess.run(spec["command"],cwd=spec["root"],env=env).returncode)
"""
    return remote_run(
        "wangy@10.10.0.34", PYTHON_TITAN + " -c " + shlex.quote(launcher),
        input_bytes=json.dumps(spec).encode())


if __name__ == "__main__":
    raise SystemExit(main())
