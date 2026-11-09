#!/usr/bin/env python3
"""Advance the authorized fixed panel through preflight, solve, audit and scoring."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

from launch_choice_paired100 import (
    LOCAL, REMOTE, PYTHON_TITAN, CPU_COMPAT, PRELOAD, remote, launch,
)
from collect_choice_paired100 import collect, score


def query(phase):
    code = """import json,time
from pathlib import Path
r=Path(%r);phase=%r;pidfile=r/(phase+'.pid.json')
if not pidfile.exists():
    print(json.dumps({'launched':False}))
else:
    launch=json.loads(pidfile.read_text());proc=Path('/proc')/str(launch['pid'])
    alive=proc.exists()
    if alive:
        fields=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()
        assert fields[19]==launch['process_identity']['start_ticks']
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==launch['process_identity']['boot_id']
        alive=fields[0]!='Z'
    artifact={'cache':r/'cached_preparation_manifest.json',
        'preflight':r/'harness/preflight_verified.json',
        'verify':r/'harness/native_trajectory_verified.json',
        'policy':r/'choice_policy_verified.json'}.get(phase)
    state=json.loads((r/'harness/live_status.json').read_text()) if (r/'harness/live_status.json').exists() else {}
    ready=(state.get('status')=='complete' and state.get('completed')==100) if phase=='execute' else artifact.exists()
    result={'phase':phase,'launched':True,'alive':alive,'ready':ready,'elapsed_s':time.time()-launch['at'],
        'completed':state.get('completed',0),'active_requests':len(state.get('active_qwen_requests',[]))}
    print(json.dumps(result))
""" % (REMOTE, phase)
    return json.loads(remote(code, timeout=30))


def wait(phase):
    previous, deadline = None, time.monotonic() + 6 * 3600
    while True:
        state = query(phase)
        key = (state.get("ready"), state.get("completed"), int(state.get("elapsed_s", 0) // 30))
        if key != previous:
            print(json.dumps(state), flush=True)
            (LOCAL / "supervisor_live_status.json").write_text(json.dumps(state, indent=2) + "\n")
            previous = key
        if state.get("ready") and not state.get("alive"):
            return
        if not state.get("launched") or not state.get("alive"):
            raise RuntimeError("owned experiment phase stopped without its success receipt: " + phase)
        if time.monotonic() > deadline:
            raise RuntimeError("experiment phase exceeded its bounded wait: " + phase)
        time.sleep(5)


def launch_policy():
    path = Path(__file__).with_name("verify_choice_policy.py")
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    code = """import hashlib,json,os,subprocess,sys,time
from pathlib import Path
r=Path(%r);source=sys.stdin.buffer.read()
assert hashlib.sha256(source).hexdigest()==%r
path=r/'verify_choice_policy.py'
assert not path.exists() and not (r/'policy.pid.json').exists()
path.write_bytes(source)
(r/'choice_policy_audit_source_identity.json').write_text(json.dumps({'file':path.name,'sha256':%r})+'\\n')
env=dict(os.environ)
for key in ['DASHSCOPE_API_KEY','QWEN3VL8B_API_KEY','VSI_SAM2_TOKEN','OPENROUTER_API_KEY','MAKERS_MODELS_KEY','VSI8B_SSH_PASSWORD']:
    env.pop(key,None)
env.update(PYTHONPATH=%r+':'+str(r/'source_frozen/src'),LD_PRELOAD=%r,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[%r,str(path),'--root',str(r)]
log=(r/'policy.log').open('ab',buffering=0)
p=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
receipt={'pid':p.pid,'process_identity':{'pid':p.pid,
    'start_ticks':(Path('/proc')/str(p.pid)/'stat').read_text().rsplit(')',1)[1].split()[19],
    'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()},
    'command':command,'at':time.time(),'automatic_retry':False,'reference_answer_visible':False,
    'source_sha256':%r}
(r/'policy.pid.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
""" % (REMOTE, digest, digest, CPU_COMPAT, PRELOAD, PYTHON_TITAN, digest)
    print(remote(code, source).decode(), end="", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    with (LOCAL / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        wait("cache")
        for phase in ("preflight", "execute", "verify"):
            if not query(phase).get("launched"):
                launch(phase)
            wait(phase)
        if not query("policy").get("launched"):
            launch_policy()
        wait("policy")
        collect()
        score(args.metadata)


if __name__ == "__main__":
    main()
