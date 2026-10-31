#!/usr/bin/env python3
"""Run read-only measurements on Titan and retain JSONL evidence locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REMOTE_PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
REMOTE_ROOT = "/home/wangy/wangjh/harness3d/four-datasets-20261031/smallimage"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--askpass", type=Path,
                   help="Optional existing private SSH askpass helper; no credential is copied.")
    p.add_argument("--credential-choice", default="1")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    source = (root/"run_tools.py").read_text(encoding="utf-8")
    annotations = json.loads((root/"visual_annotations.json").read_text(encoding="utf-8"))
    matches = json.loads((root/"rgb_correspondences.json").read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output/"raw.jsonl"
    if raw_path.exists():
        raise ValueError("Output already exists; inspect prior evidence before rerunning.")
    identity = {
        "measurement_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "annotation_file_sha256": hashlib.sha256((root/"visual_annotations.json").read_bytes()).hexdigest(),
        "rgb_correspondences_file_sha256": hashlib.sha256((root/"rgb_correspondences.json").read_bytes()).hexdigest(),
        "remote_root": REMOTE_ROOT, "remote_python": REMOTE_PYTHON,
        "requested_questions": args.limit, "started_at_unix": time.time(),
        "model_requests": 0, "gpu_requests": 0,
    }
    (args.output/"execution.json").write_text(json.dumps(identity, indent=2)+"\n", encoding="utf-8")
    payload = (
        "import json,traceback,sys\n"
        "namespace={'__name__':'blink_tool_experiment'}\n"
        f"exec(compile({source!r},'run_tools.py','exec'),namespace)\n"
        "try:\n"
        f" namespace['run']({REMOTE_ROOT!r},{annotations!r},{args.limit!r},{matches!r})\n"
        "except Exception as exc:\n"
        " print(json.dumps({'record_type':'execution_error','type':type(exc).__name__,"
        "'message':str(exc),'frames':[{'file':f.filename,'line':f.lineno}"
        " for f in traceback.extract_tb(exc.__traceback__)]}),flush=True)\n"
        " sys.exit(1)\n"
    )
    env = dict(os.environ)
    opts = ["-T", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
    if args.askpass:
        if not args.askpass.is_file():
            raise ValueError("Existing askpass helper not found.")
        env.update(SSH_ASKPASS=str(args.askpass), SSH_ASKPASS_REQUIRE="force",
                   DISPLAY="harness3d-tool-check:0", H3D_CREDENTIAL_CHOICE=args.credential_choice)
        opts += ["-o", "ControlMaster=no", "-o", "PubkeyAuthentication=no",
                 "-o", "PreferredAuthentications=password", "-o", "NumberOfPasswordPrompts=1"]
    else:
        opts += ["-o", "BatchMode=yes"]
    command = ["ssh", *opts, "wangy@10.10.0.34", REMOTE_PYTHON, "-B", "-"]
    seen, complete = [], False
    with raw_path.open("x", encoding="utf-8") as out:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env)
        proc.stdin.write(payload)
        proc.stdin.close()
        for line in proc.stdout:
            record = json.loads(line)
            out.write(line)
            out.flush()
            if record["record_type"] == "sample":
                sample = record["result"]
                if sample["sample_id"] in seen:
                    raise ValueError("Duplicate sample in measurement stream.")
                seen.append(sample["sample_id"])
                print(json.dumps({"completed": len(seen), "last_sample": sample["sample_id"],
                                  "seconds": round(sample["elapsed_seconds"], 2)}), flush=True)
            elif record["record_type"] == "complete":
                complete = True
                print(json.dumps(record), flush=True)
            elif record["record_type"] == "execution_error":
                print(json.dumps(record), flush=True)
        stderr = proc.stderr.read()
        code = proc.wait()
    identity.update(returncode=code, saved_samples=len(seen), complete=complete,
                    finished_at_unix=time.time(),
                    raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
    # SSH diagnostics contain no request, model prompt, or credential values.
    if code and not seen:
        identity["transport_error_type"] = ("authentication" if "Permission denied" in stderr
                                             else "ssh_or_runtime")
    (args.output/"execution.json").write_text(json.dumps(identity, indent=2)+"\n", encoding="utf-8")
    if code or not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
