#!/usr/bin/env python3
"""Collect immutable completed pilot journals, validate hashes, then score locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from ssh_from_server import remote_run

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/home/wangy/wangjh/VSIbench8Brunner"
RUN_NAME = "vsi8b-pilot40-20261105-r02"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def probe(name):
    code = """
import json,time
from pathlib import Path
root=Path(%s)
run=root/"runs"/%s
launch=json.loads((root/(%s+".pid.json")).read_text())
proc=Path("/proc")/str(launch["pid"])
alive=proc.exists()
if alive:
    stat=proc.joinpath("stat").read_text().rsplit(")",1)[1].split()
    if stat[19] != launch["process_identity"]["start_ticks"]:
        raise RuntimeError("owned collector process identity mismatch")
    alive=stat[0]!="Z"
outcomes=sorted(p.stem for p in (run/"outcomes").glob("*.json"))
index=json.loads((run/"trajectory_index.json").read_text())["questions"]
complete=[qa for qa,row in index.items()
          if row["completeness"]["outcome_written"]
          and row["completeness"]["all_model_requests_returned"]]
live=json.loads((run/"live_status.json").read_text())
print(json.dumps({"at":time.time(),"alive":alive,"outcomes":outcomes,
    "complete_trajectories":sorted(complete),"status":live.get("status"),
    "active_qwen_requests":len(live.get("active_qwen_requests",[])),
    "process_identity":launch["process_identity"]}))
""" % (repr(REMOTE_ROOT), repr(RUN_NAME), repr(name))
    status, stdout, _ = remote_run(
        "wangy@10.10.0.34", "python3 -c " + shlex.quote(code), return_output=True)
    if status:
        raise RuntimeError("pilot collector status query failed")
    return json.loads(stdout)


def validate(root):
    sampling = json.loads((root / "sampling_manifest.json").read_text())
    ids = {str(row["qa_id"]) for row in sampling["questions"]}
    index = json.loads((root / "trajectory_index.json").read_text())
    if len(ids) != 40 or set(index["questions"]) != ids:
        raise ValueError("pilot collection scope changed")
    verified = []
    for qa in sorted(ids):
        row = index["questions"][qa]
        path = root / row["file"]
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != row["sha256"]:
            raise ValueError("collected trajectory identity mismatch")
        full = json.loads(path.read_text())
        complete = full["completeness"]
        if (full["qa_id"] != qa or full["reference_answer_visible"] is not False
                or not complete["outcome_written"]
                or not complete["all_model_requests_returned"]
                or complete["missing_reviewer_request_files"]
                or complete["incomplete_jsonl_lines"]):
            raise ValueError("collected trajectory is incomplete")
        sources = [*full["source_files"], *full["preparation"]["mask_assets"]]
        for source in sources:
            source_path = root / source["file"]
            if (not source_path.resolve().is_relative_to(root.resolve())
                    or source_path.stat().st_size != source["bytes"]
                    or digest(source_path) != source["sha256"]):
                raise ValueError("collected native journal or mask identity mismatch")
        verified.append({"qa_id": qa, "sha256": row["sha256"],
                         "source_files_checked": len(sources)})
    receipt = {
        "schema": "pilot-full-local-collection/1", "at": time.time(),
        "reference_answers_read_during_collection": False,
        "selected_questions": 40, "verified": verified,
    }
    (root / "local_collection_verified.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="pilot40-parallel01")
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    root = ROOT / "runs" / RUN_NAME
    previous = None
    while True:
        state = probe(args.name)
        counts = (len(state["outcomes"]), len(state["complete_trajectories"]),
                  state["active_qwen_requests"], state["status"])
        if counts != previous:
            print(json.dumps({**state, "outcomes": counts[0],
                              "complete_trajectories": counts[1]}), flush=True)
            previous = counts
        if len(state["outcomes"]) == 40 and len(state["complete_trajectories"]) == 40:
            break
        if not state["alive"]:
            (root / "collector_stopped.json").write_text(json.dumps(state, indent=2) + "\n")
            raise RuntimeError("pilot stopped before completion; no scoring or automatic retry")
        time.sleep(10)
    subprocess.run([
        "rsync", "-a", "-e",
        "ssh -o BatchMode=yes -S " + shlex.quote(args.control_path),
        "wangy@10.10.0.34:" + REMOTE_ROOT + "/runs/" + RUN_NAME + "/",
        str(root) + "/"], check=True)
    validate(root)
    env = {key: value for key, value in os.environ.items()
           if key not in ("DASHSCOPE_API_KEY", "QWEN3VL8B_API_KEY", "VSI_SAM2_TOKEN",
                          "OPENROUTER_API_KEY", "MAKERS_MODELS_KEY", "VSI8B_SSH_PASSWORD")}
    env["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "score_pilot40.py"),
                    "--run-dir", str(root), "--metadata", str(args.metadata)],
                   env=env, check=True)


if __name__ == "__main__":
    main()
