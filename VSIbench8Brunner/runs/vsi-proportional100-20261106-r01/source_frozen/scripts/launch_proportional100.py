#!/usr/bin/env python3
"""Freeze and launch the two authorized arms in a separate experiment root."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import tarfile

from deploy_pilot_services import ROOT, REMOTE_TITAN, PYTHON_4090
from launch_pilot40 import PYTHON_TITAN
from ssh_from_server import remote_run

NAME = "vsi-proportional100-20261106-r01"
LOCAL = ROOT / "runs" / NAME
REMOTE = REMOTE_TITAN + "/runs/" + NAME
FROZEN = REMOTE + "/source_frozen"
PUBLISHED = "/data/wangy/wangjh/harness3d/experiment-inputs/vsibench-m4-success-1000-seed137/published_inputs.json"
CPU_COMPAT = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat"
PRELOAD = "/home/wangy/wangjh/harness3d/precompute-20261022-r01/cpu_sqlite/libsqlite3.so.0"
SECRET_NAMES = ("DASHSCOPE_API_KEY", "QWEN3VL8B_API_KEY", "VSI_SAM2_TOKEN",
                "OPENROUTER_API_KEY", "MAKERS_MODELS_KEY", "VSI8B_SSH_PASSWORD",
                "VSI8B_REVIEW_REPLAY_ROOT")

RUN = """import hashlib,json,os,subprocess,sys,time
from pathlib import Path
s=json.load(sys.stdin); root=Path(s['root'])
manifest=json.loads((root/'source_manifest.json').read_text())
for f in manifest['files']:
    p=root/'source_frozen'/f['file']
    assert p.stat().st_size==f['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
assert hashlib.sha256((root/'sampling_manifest.json').read_bytes()).hexdigest()==s['sampling_sha256']
env={k:v for k,v in os.environ.items() if k not in s['strip_env']}
env.update(s['env'])
arm=root/s['arm']
if s['execute']:
    assert (arm/'preflight_verified.json').exists()
    assert json.loads((arm/'preflight_verified.json').read_text())['questions_verified']==100
    pidfile=root/(s['arm']+'.pid.json')
    if pidfile.exists() or (arm/'request_budget.json').exists() or list((arm/'solver_requests').glob('*/*.json')):
        raise ValueError('owned arm already started; automatic rerun disabled')
    log=(root/(s['arm']+'.log')).open('ab',buffering=0)
    p=subprocess.Popen(s['command'],cwd=str(root/'source_frozen'),env=env,
        stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    proc=Path('/proc')/str(p.pid)
    identity={'pid':p.pid,'start_ticks':proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    receipt={'arm':s['arm'],'pid':p.pid,'process_identity':identity,
        'command':s['command'],'at':time.time(),
        'command_sha256':hashlib.sha256(json.dumps(s['command'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'sampling_sha256':s['sampling_sha256'],'automatic_retry':False}
    pidfile.write_text(json.dumps(receipt,indent=2)+'\\n')
    print(json.dumps(receipt))
else:
    raise SystemExit(subprocess.run(s['command'],cwd=str(root/'source_frozen'),env=env).returncode)
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    for directory in ("src", "scripts", "skill_library"):
        for path in sorted((ROOT / directory).rglob("*")):
            if (path.is_file() and not path.is_symlink()
                    and "__pycache__" not in path.parts
                    and path.suffix not in (".pyc", ".log", ".npz")
                    and path.name not in (".env", "credentials.json")):
                yield path
    yield ROOT / "runner.yaml"


def freeze():
    sampling = json.loads((LOCAL / "sampling_manifest.json").read_text())
    source = Path(sampling["source_question_file"])
    if sha(source) != sampling["source_question_sha256"]:
        raise ValueError("selected public source changed")
    frozen = LOCAL / "source_frozen"
    manifest_path = LOCAL / "source_manifest.json"
    if not manifest_path.exists():
        if frozen.exists():
            raise ValueError("partial frozen source exists; retain for diagnosis")
        frozen.mkdir()
        files = []
        for path in source_files():
            relative = path.relative_to(ROOT)
            target = frozen / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            files.append({"file": str(relative), "sha256": sha(target),
                          "bytes": target.stat().st_size})
        manifest = {
            "schema": "vsi-paired-frozen-source/1", "files": files,
            "sampling_sha256": sha(LOCAL / "sampling_manifest.json"),
            "scientific_configuration_changed": False,
            "execution_changes": "panel size and request caps; published-scene official32 reuse",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        (LOCAL / "public_questions.jsonl").write_bytes(source.read_bytes())
        for arm in ("direct", "harness"):
            directory = LOCAL / arm
            directory.mkdir()
            (directory / "sampling_manifest.json").write_bytes(
                (LOCAL / "sampling_manifest.json").read_bytes())
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["files"]:
        if sha(frozen / row["file"]) != row["sha256"]:
            raise ValueError("frozen source has changed")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for row in manifest["files"]:
            archive.add(frozen / row["file"], arcname="source_frozen/" + row["file"])
        for name in ("source_manifest.json", "sampling_manifest.json",
                     "public_questions.jsonl", "experiment_spec.json",
                     "prior_run_inventory.json"):
            archive.add(LOCAL / name, arcname=name)
        for arm in ("direct", "harness"):
            archive.add(LOCAL / arm / "sampling_manifest.json",
                        arcname=arm + "/sampling_manifest.json")
    upload = """import hashlib,io,json,sys,tarfile
from pathlib import Path
r=Path(%r); r.mkdir(parents=True,exist_ok=True)
b=sys.stdin.buffer.read()
if (r/'source_manifest.json').exists():
    with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
        assert (r/'source_manifest.json').read_bytes()==t.extractfile('source_manifest.json').read()
    print(json.dumps({'status':'existing_frozen_source_reused'}))
else:
    with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
        t.extractall(str(r))
    print(json.dumps({'status':'new_frozen_source_uploaded'}))
""" % REMOTE
    if remote_run("wangy@10.10.0.34", PYTHON_TITAN + " -c " + shlex.quote(upload),
                  input_bytes=buffer.getvalue()):
        raise RuntimeError("independent frozen runtime upload failed")


def credentials(arm):
    text = (ROOT.parent / "qwenkey.md").read_text()
    keys = set(re.findall(r"\bsk-[A-Za-z0-9._~+/-]+=*", text.strip()))
    if len(keys) != 1:
        raise RuntimeError("authorized main credential is ambiguous")
    env = {"DASHSCOPE_API_KEY": keys.pop()}
    if arm == "harness":
        retrieve = """import json
from pathlib import Path
r=Path('/home/cvailab/wangjh/VSIbench8Brunner'); result={}
for name,key in [('sam2','VSI_SAM2_TOKEN'),('qwen8b','VLLM_API_KEY')]:
    pid=json.loads((r/(name+'.pid.json')).read_text())['pid']
    e=dict(x.split(b'=',1) for x in (Path('/proc')/str(pid)/'environ').read_bytes().split(b'\\0') if b'=' in x)
    result[key]=e[key.encode()].decode()
print(json.dumps(result))
"""
        code, raw, _ = remote_run(
            "cvailab@10.10.0.198", PYTHON_4090 + " -c " + shlex.quote(retrieve),
            return_output=True)
        if code:
            raise RuntimeError("owned model service credential injection unavailable")
        service = json.loads(raw)
        env.update(VSI_SAM2_TOKEN=service["VSI_SAM2_TOKEN"],
                   QWEN3VL8B_API_KEY=service["VLLM_API_KEY"])
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("direct", "harness"))
    parser.add_argument("--phase", choices=("freeze", "preflight", "execute"), default="preflight")
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze()
        return 0
    if not args.arm:
        parser.error("arm is required after source freezing")
    total = json.loads((LOCAL / "sampling_manifest.json").read_text())["n_questions"]
    if total != 100:
        raise ValueError("authorized new panel scope changed")
    execute = args.phase == "execute"
    sampling = REMOTE + "/sampling_manifest.json"
    arm_root = REMOTE + "/" + args.arm
    common = ["--sampling", sampling, "--run-dir", arm_root, "--expected-questions", "100"]
    if args.arm == "direct":
        command = [PYTHON_TITAN, FROZEN + "/scripts/pilot40_direct.py", *common,
                   "--source-questions", REMOTE + "/public_questions.jsonl",
                   "--official-root", "/home/wangy/wangjh/harness3d/four-datasets-20261031/vsi1000/official32",
                   "--validator", "/home/wangy/wangjh/harness3d/four-datasets-20261031/official32_pipeline.py",
                   "--workers", "16"]
        if execute:
            command.append("--execute")
    else:
        command = [PYTHON_TITAN, FROZEN + "/scripts/" + (
            "pilot40_parallel.py" if execute else "preflight_proportional_harness.py"),
            *common, "--published-inputs", PUBLISHED]
        if execute:
            command.extend(["--preparation-request-cap", "9710", "--prepare-workers", "4",
                            "--solve-workers", "16", "--cpu-limit", "8", "--api-limit", "32"])
    env = {
        "PYTHONPATH": CPU_COMPAT + ":" + FROZEN + "/src",
        "LD_PRELOAD": PRELOAD, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
    }
    if args.arm == "harness":
        env.update({
            "VSI_SAM2_ENDPOINT": "http://10.10.0.198:28102",
            "VSI_QWEN8B_BASE_URL": "http://10.10.0.198:28101/v1",
            "VSI_QWEN8B_MODEL": "Qwen3-VL-8B-Instruct-FP8",
            "VSI_REVIEW_ENDPOINT": "http://127.0.0.1:27248",
        })
    if execute or args.arm == "harness":
        env.update(credentials(args.arm))
    spec = {"root": REMOTE, "arm": args.arm, "execute": execute,
            "command": command, "env": env, "strip_env": list(SECRET_NAMES),
            "sampling_sha256": sha(LOCAL / "sampling_manifest.json")}
    return remote_run(
        "wangy@10.10.0.34", PYTHON_TITAN + " -c " + shlex.quote(RUN),
        input_bytes=json.dumps(spec).encode())


if __name__ == "__main__":
    raise SystemExit(main())
