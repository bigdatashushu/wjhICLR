#!/usr/bin/env python3
"""Deploy and launch unique owned recovery stages with transient credentials."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import launch_proportional100 as launch
from ssh_from_server import remote_run
from collect_proportional100 import remote

CONTROL = "/Users/bytedance/.cache/harness3d-ssh/vsi-proportional100-persistent"
os.environ["VSI8B_SSH_CONTROL_TITAN"] = CONTROL
os.environ.pop("VSI8B_SSH_CONTROL_4090", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("preflight", "execute"), required=True)
    args = parser.parse_args()
    base_script = Path(__file__).with_name("resume_fixed100_r02.py")
    script = (base_script if args.phase == "preflight" else
              base_script.with_name("resume_fixed100_r02_execute.py"))
    deploy = """import sys
from pathlib import Path
p=Path(%r); data=sys.stdin.buffer.read()
assert p.parent.is_dir()
if p.exists(): assert p.read_bytes()==data
else: p.write_bytes(data)
""" % (launch.REMOTE + "/" + script.name)
    remote(CONTROL, deploy, stdin=script.read_bytes(), timeout=30)
    env = {
        "PYTHONPATH": launch.CPU_COMPAT + ":" + launch.FROZEN + "/src",
        "LD_PRELOAD": launch.PRELOAD, "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
        "VSI_SAM2_ENDPOINT": "http://10.10.0.198:28102",
        "VSI_QWEN8B_BASE_URL": "http://10.10.0.198:28101/v1",
        "VSI_QWEN8B_MODEL": "Qwen3-VL-8B-Instruct-FP8",
        "VSI_REVIEW_ENDPOINT": "http://127.0.0.1:27248",
    }
    env.update(launch.credentials("harness"))
    name = "harness-recovery-" + ("preflight-r02" if args.phase == "preflight" else "r02")
    guard = {"file": base_script.name, "sha256": launch.sha(base_script)}
    if args.phase == "execute":
        guard["execution_adapter"] = {"file": script.name, "sha256": launch.sha(script)}
    spec = {"env": env, "strip_env": list(launch.SECRET_NAMES),
            "root": launch.REMOTE, "name": name,
            "phase": args.phase, "script_sha256": launch.sha(script),
            "identity_guard": guard,
            "command": [launch.PYTHON_TITAN, launch.REMOTE + "/" + script.name,
                        "--phase", args.phase]}
    code = """import hashlib,json,os,subprocess,sys,time
from pathlib import Path
s=json.load(sys.stdin); root=Path(s['root']); p=root/(s['name']+'.pid.json')
assert not p.exists(), 'unique owned stage already launched'
script=Path(s['command'][1]); assert hashlib.sha256(script.read_bytes()).hexdigest()==s['script_sha256']
g=s['identity_guard']; assert hashlib.sha256((root/g['file']).read_bytes()).hexdigest()==g['sha256']
if s['phase']=='execute': assert (root/'recovery_r02/plan.json').exists()
env={k:v for k,v in os.environ.items() if k not in s['strip_env']}
env.update(s['env'])
with (root/(s['name']+'.log')).open('ab',buffering=0) as log:
 proc=subprocess.Popen(s['command'],cwd=str(root),env=env,stdin=subprocess.DEVNULL,
     stdout=log,stderr=log,start_new_session=True)
 fields=(Path('/proc')/str(proc.pid)/'stat').read_text().rsplit(')',1)[1].split()
 identity={'pid':proc.pid,'start_ticks':fields[19],
     'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
 receipt={'arm':'harness','phase':s['phase'],'pid':proc.pid,'process_identity':identity,
    'command_sha256':hashlib.sha256(json.dumps(s['command'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
    'sampling_sha256':hashlib.sha256((root/'sampling_manifest.json').read_bytes()).hexdigest(),
    'identity_guard':s['identity_guard'],
    'automatic_retry':False,'at':time.time()}
 p.write_text(json.dumps(receipt,indent=2)+'\\n')
 print(json.dumps(receipt))
"""
    status, raw, _ = remote_run("wangy@10.10.0.34",
        launch.PYTHON_TITAN + " -c " + shlex.quote(code),
        input_bytes=json.dumps(spec).encode(), return_output=True)
    if status:
        raise RuntimeError("owned recovery launch unavailable")
    print(raw.decode())


if __name__ == "__main__":
    main()
