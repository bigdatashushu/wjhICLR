#!/usr/bin/env python3
"""Deploy owned model services; credentials travel in SSH stdin, never files."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import secrets
import shlex
import subprocess

from ssh_from_server import remote_run

ROOT = Path(__file__).resolve().parents[1]
PYTHON_3090 = "/home/zhouky/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
REMOTE_3090 = "/home/zhouky/wangjh/VSIbench8Brunner"
REMOTE_TITAN = "/home/wangy/wangjh/VSIbench8Brunner"
PYTHON_4090 = "/home/cvailab/anaconda3/envs/skill3d-exp/bin/python"
REMOTE_4090 = "/home/cvailab/wangjh/VSIbench8Brunner"

LAUNCH = """import json,os,subprocess,sys
from pathlib import Path
spec=json.load(sys.stdin)
root=Path(spec["root"])
pidfile=root/(spec["name"]+".pid.json")
if pidfile.exists():
    old=json.load(pidfile.open())
    proc=Path("/proc/"+str(old["pid"]))
    if proc.exists() and proc.joinpath("stat").read_text().rsplit(")",1)[1].split()[0] != "Z":
        raise RuntimeError("owned service already exists")
env={**os.environ,**spec["env"]}
log=(root/(spec["name"]+".log")).open("ab",buffering=0)
p=subprocess.Popen(spec["command"],cwd=root,env=env,stdin=subprocess.DEVNULL,
                   stdout=log,stderr=log,start_new_session=True)
import hashlib
proc=Path("/proc")/str(p.pid)
process_identity={"pid":p.pid,"start_ticks":proc.joinpath("stat").read_text().rsplit(")",1)[1].split()[19],
    "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
pidfile.write_text(json.dumps({"pid":p.pid,"command":spec["command"],
    "process_identity":process_identity,
    "command_sha256":hashlib.sha256(json.dumps(spec["command"],sort_keys=True,
        separators=(",",":")).encode()).hexdigest()}))
print(json.dumps({"name":spec["name"],"pid":p.pid}))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("sam2", "qwen8b"))
    parser.add_argument("--server", choices=("3090", "4090"), default="4090")
    args = parser.parse_args()
    host = "cvailab@10.10.0.198" if args.server == "4090" else "zhouky@10.10.0.36"
    destination = REMOTE_4090 if args.server == "4090" else REMOTE_3090
    python = PYTHON_4090 if args.server == "4090" else PYTHON_3090
    archive = subprocess.check_output([
        "tar", "--no-xattrs", "-czf", "-", "--exclude=./runs", "--exclude=./tests",
        "--exclude=__pycache__", "-C", str(ROOT), "."])
    for target_host, target_root in (
            (host, destination), ("wangy@10.10.0.34", REMOTE_TITAN)):
        result = remote_run(
            target_host, f"mkdir -p {target_root} && tar -xzf - -C {target_root}",
            input_bytes=archive)
        if result:
            raise RuntimeError("runtime deployment failed")
    specs = [
        {"name": "sam2", "root": REMOTE_3090,
         "env": {
             "CUDA_VISIBLE_DEVICES": "3", "VSI_SAM2_TOKEN": secrets.token_urlsafe(32),
             "PYTHONPATH": REMOTE_3090 + "/src",
             "SKILL3D_SAM2_CHECKPOINT":
             "/home/zhouky/wangjh/harness3d/precompute-20261022-r01/imports/assets/sam2/sam2.1_hiera_large.pt",
             "SKILL3D_SAM2_CONFIG": "configs/sam2.1/sam2.1_hiera_l.yaml"},
         "command": [PYTHON_3090, "serve_sam2.py"]},
        {"name": "qwen8b", "root": REMOTE_3090,
         "env": {"CUDA_VISIBLE_DEVICES": "4,5", "VLLM_API_KEY": secrets.token_urlsafe(32),
                 "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
         "command": [
             PYTHON_3090, "-m", "vllm.entrypoints.openai.api_server",
             "--model", "/nas/wangjh/harness3d/models/Qwen3-VL-8B-Instruct",
             "--served-model-name", "Qwen3-VL-8B-Instruct", "--dtype", "bfloat16",
             "--tensor-parallel-size", "2",
             "--host", "10.10.0.36", "--port", "28101",
             "--max-model-len", "8192", "--max-num-seqs", "2",
             "--limit-mm-per-prompt", '{"image":3}',
             "--mm-processor-kwargs", '{"max_pixels":1048576}',
             "--gpu-memory-utilization", "0.90",
             "--no-enable-log-requests", "--uvicorn-log-level", "warning",
             "--enforce-eager"]},
    ]
    if args.server == "4090":
        for spec in specs:
            spec["root"] = REMOTE_4090
            spec["command"][0] = PYTHON_4090
            spec["env"]["PYTHONPATH"] = REMOTE_4090 + "/src"
        specs[0]["env"].update(
            CUDA_VISIBLE_DEVICES="0",
            SKILL3D_SAM2_CHECKPOINT=
            "/home/cvailab/.cache/huggingface/hub/models--facebook--sam2.1-hiera-large/"
            "snapshots/227a114a2f535cd147f82442e7d2038cdd2e5d68/sam2.1_hiera_large.pt")
        specs[0]["command"].extend(["--host", "10.10.0.198"])
        specs[1]["env"]["CUDA_VISIBLE_DEVICES"] = "0,1"
        command = specs[1]["command"]
        command[command.index("--model") + 1] = (
            "/home/cvailab/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct-FP8/"
            "snapshots/9cdc6310a8cb770ce18efaf4e9935334512aee45")
        command[command.index("--served-model-name") + 1] = "Qwen3-VL-8B-Instruct-FP8"
        command[command.index("--host") + 1] = "10.10.0.198"
        command[command.index("--gpu-memory-utilization") + 1] = "0.34"
    else:
        specs[0]["command"].extend(["--host", "10.10.0.36"])
    for spec in specs:
        if args.only and spec["name"] != args.only:
            continue
        if remote_run(
                host, python + " -c " + shlex.quote(LAUNCH),
                input_bytes=json.dumps(spec).encode()):
            raise RuntimeError("service launch failed")


if __name__ == "__main__":
    main()
