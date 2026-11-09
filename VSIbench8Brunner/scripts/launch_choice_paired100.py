#!/usr/bin/env python3
"""Freeze and launch one repaired arm without regenerating prior preparation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

from launch_proportional100 import (
    ROOT, REMOTE_TITAN, PYTHON_TITAN, PUBLISHED, CPU_COMPAT, PRELOAD,
    SECRET_NAMES, credentials, source_files,
)

NAME = "vsi-choice-isolated100-20261106-r01"
LOCAL = ROOT / "runs" / NAME
REMOTE = REMOTE_TITAN + "/runs/" + NAME
OLD = ROOT / "runs/vsi-proportional100-20261106-r01"
CONTROL = "/Users/bytedance/.cache/harness3d-ssh/vsi-proportional100-persistent"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remote(code, data=None, timeout=60):
    result = subprocess.run([
        "ssh", "-S", CONTROL, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=10", "wangy@10.10.0.34",
        PYTHON_TITAN + " -c " + shlex.quote(code),
    ], input=data, capture_output=True, timeout=timeout)
    if result.returncode:
        # Never expose transport stderr or injected environment values.
        raise RuntimeError("choice paired remote step failed")
    return result.stdout


def freeze():
    if LOCAL.exists():
        raise ValueError("candidate experiment already exists; duplicate freeze disabled")
    LOCAL.mkdir()
    frozen = LOCAL / "source_frozen"
    frozen.mkdir()
    files = []
    for path in source_files():
        relative = path.relative_to(ROOT)
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        files.append({"file": str(relative), "sha256": sha(target),
                      "bytes": target.stat().st_size})
    sampling = (OLD / "sampling_manifest.json").read_bytes()
    (LOCAL / "sampling_manifest.json").write_bytes(sampling)
    (LOCAL / "harness").mkdir()
    (LOCAL / "harness/sampling_manifest.json").write_bytes(sampling)
    identity = json.loads((OLD / "harness/runtime_identity.json").read_text())
    spec = {
        "schema": "vsi-choice-repair-paired100/1", "n_questions": 100,
        "sampling_sha256": hashlib.sha256(sampling).hexdigest(),
        "published_inputs_sha256": identity["published_inputs_sha256"],
        "skill_snapshot": identity["skill_snapshot"],
        "reviewer_identity_sha256": identity["reviewer_identity_sha256"],
        "sam2_identity_sha256": identity["sam2_identity_sha256"],
        "description": identity["description"],
        "paired_baseline_root": str(OLD),
        "paired_baseline_server_root": REMOTE_TITAN + "/runs/" + OLD.name,
        "baseline_scores_sha256": sha(OLD / "direct_vs_8b_scores.json"),
        "baseline_native_audit_sha256": sha(OLD / "paired_native_verified.json"),
        "changes": [
            "isolated original32 choice fallback and observer/candidate prompt",
            "partial-ranking recovery scope warnings",
            "explicit validation role and blind final_test admission",
            "cached-only preparation transport; no new perception inference",
        ],
        "evaluation": "diagnostic rerun of the same previously evaluated 100 questions",
        "model": "qwen3.8-flash", "enable_thinking": True, "seed": 137,
        "temperature": 0, "max_tokens": 8192, "timeout_s": 900,
        "max_solver_rounds": 6, "episode_model_request_cap": 12,
        "additional_visual_request_cap": 1, "automatic_retry": False,
        "reference_answer_visible": False, "source_split_preserved": True,
    }
    for name, data in [
        ("experiment_spec.json", spec),
        ("source_manifest.json", {"schema": "vsi-choice-frozen-source/1", "files": files,
                                  "sampling_sha256": spec["sampling_sha256"]}),
    ]:
        (LOCAL / name).write_text(json.dumps(data, indent=2) + "\n")
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        for row in files:
            archive.add(frozen / row["file"], arcname="source_frozen/" + row["file"])
        for name in ("sampling_manifest.json", "experiment_spec.json",
                     "source_manifest.json", "harness/sampling_manifest.json"):
            archive.add(LOCAL / name, arcname=name)
    code = """import io,json,sys,tarfile
from pathlib import Path
r=Path(%r)
assert r.parent.is_dir()
if r.exists(): raise ValueError('candidate server experiment already exists')
r.mkdir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
    archive.extractall(str(r))
print(json.dumps({'status':'candidate_source_frozen','root':str(r)}))
""" % REMOTE
    print(remote(code, archive_buffer.getvalue(), timeout=120).decode(), end="")


def launch(phase):
    execute = phase == "execute"
    env = {
        "PYTHONPATH": CPU_COMPAT + ":" + REMOTE + "/source_frozen/src",
        "LD_PRELOAD": PRELOAD, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
    }
    if execute:
        env.update(credentials("direct"))
    command = [
        PYTHON_TITAN, REMOTE + "/source_frozen/scripts/choice_cached100.py", phase,
        "--sampling", REMOTE + "/sampling_manifest.json",
        "--published-inputs", PUBLISHED, "--run-dir", REMOTE + "/harness",
        "--expected-questions", "100",
    ]
    if execute:
        command.extend(["--preparation-request-cap", "0", "--prepare-workers", "1",
                        "--solve-workers", "12", "--cpu-limit", "8", "--api-limit", "12"])
    data = {"root": REMOTE, "phase": phase, "env": env, "strip": list(SECRET_NAMES),
            "command": command}
    code = """import hashlib,json,os,subprocess,sys,time
from pathlib import Path
s=json.load(sys.stdin);r=Path(s['root']); phase=s['phase']
sys.path.insert(0,str(r/'source_frozen/scripts'));sys.path.insert(0,str(r/'source_frozen/src'))
pidfile=r/(phase+'.pid.json')
if pidfile.exists(): raise ValueError('owned phase already started; duplicate launch disabled')
if phase=='execute':
    p=r/'harness/preflight_verified.json'; receipt=json.loads(p.read_text())
    assert receipt['questions_verified']==100 and receipt['model_requests']==0
    assert receipt['source_manifest_sha256']==hashlib.sha256((r/'source_manifest.json').read_bytes()).hexdigest()
    assert receipt['cached_preparation_manifest_sha256']==hashlib.sha256((r/'cached_preparation_manifest.json').read_bytes()).hexdigest()
    assert not list((r/'harness/solver_requests').glob('*/*.json'))
env={k:v for k,v in os.environ.items() if k not in s['strip']}
env.update(s['env'])
log=(r/(phase+'.log')).open('ab',buffering=0)
p=subprocess.Popen(s['command'],cwd=str(r/'source_frozen'),env=env,stdin=subprocess.DEVNULL,
    stdout=log,stderr=log,start_new_session=True)
proc=Path('/proc')/str(p.pid)
receipt={'phase':phase,'pid':p.pid,'process_identity':{'pid':p.pid,
    'start_ticks':proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],
    'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()},
    'command':s['command'],'command_sha256':hashlib.sha256(json.dumps(s['command'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'automatic_retry':False,'reference_answer_visible':False,'at':time.time()}
pidfile.write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
"""
    print(remote(code, json.dumps(data).encode()).decode(), end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze", "preflight", "execute", "verify"))
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
    else:
        launch(args.phase)


if __name__ == "__main__":
    main()
