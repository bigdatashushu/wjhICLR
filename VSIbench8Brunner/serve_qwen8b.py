#!/usr/bin/env python3
"""Explicit single-GPU local Qwen3-VL-8B service launcher."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, type=int, help="physical GPU explicitly allocated to this service")
    parser.add_argument("--model-path", required=True, type=Path, help="existing Qwen3-VL-8B model directory")
    parser.add_argument("--model-name", choices=(
        "Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Instruct-FP8"),
        default="Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--python", default=sys.executable, help="existing Python with compatible vLLM")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.model_path.is_dir():
        parser.error("existing model directory does not exist")
    command = [
        args.python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(args.model_path.absolute()), "--served-model-name", args.model_name,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--max-model-len", "8192", "--max-num-seqs", "2",
        "--limit-mm-per-prompt", '{"image":3}', "--gpu-memory-utilization", "0.90",
    ]
    if args.dry_run:
        print("CUDA_VISIBLE_DEVICES=" + str(args.gpu))
        print(command)
        return 0
    occupancy = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True).strip()
    if int(occupancy) > 256:
        parser.error("allocated GPU is occupied; choose an available allocated GPU")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
