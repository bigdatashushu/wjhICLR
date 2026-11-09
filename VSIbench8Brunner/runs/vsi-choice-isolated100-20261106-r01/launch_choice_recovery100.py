#!/usr/bin/env python3
"""Launch the exact cached-parameter correction outside the frozen solver source."""
import argparse
import hashlib
import json
from pathlib import Path

from launch_choice_paired100 import (
    LOCAL, REMOTE, PYTHON_TITAN, PUBLISHED, CPU_COMPAT, PRELOAD,
    SECRET_NAMES, credentials, remote,
)


def launch(phase):
    source = (LOCAL / "resume_choice_cached100.py").read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    env = {
        "PYTHONPATH": CPU_COMPAT + ":" + REMOTE + "/source_frozen/src:"
                      + REMOTE + "/source_frozen/scripts",
        "LD_PRELOAD": PRELOAD, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
    }
    execute = phase == "execute"
    if execute:
        env.update(credentials("direct"))
    command = [PYTHON_TITAN, REMOTE + "/resume_choice_cached100.py", phase]
    if execute:
        command += [
            "--sampling", REMOTE + "/sampling_manifest.json",
            "--published-inputs", PUBLISHED, "--run-dir", REMOTE + "/harness",
            "--expected-questions", "100", "--preparation-request-cap", "0",
            "--prepare-workers", "1", "--solve-workers", "12",
            "--cpu-limit", "8", "--api-limit", "12",
        ]
    data = {
        "root": REMOTE, "phase": phase, "env": env, "strip": list(SECRET_NAMES),
        "command": command, "source": source.decode("utf-8"), "source_sha256": sha,
    }
    code = """import hashlib,json,os,subprocess,sys,time
from pathlib import Path
s=json.load(sys.stdin);r=Path(s['root']);phase=s['phase'];name=phase+'_recovery_r02'
assert r.is_dir()
pidfile=r/(name+'.pid.json')
assert not pidfile.exists()
path=r/'resume_choice_cached100.py';source=s['source'].encode('utf-8')
assert hashlib.sha256(source).hexdigest()==s['source_sha256']
if path.exists(): assert path.read_bytes()==source
else: path.write_bytes(source)
if phase=='execute':
    pre=json.loads((r/'recovery_compatibility_verified.json').read_text())
    assert pre['questions_verified']==100 and pre['model_requests']==0
    assert pre['recovery_source_sha256']==s['source_sha256']
env={k:v for k,v in os.environ.items() if k not in s['strip']};env.update(s['env'])
log=(r/(name+'.log')).open('ab',buffering=0)
p=subprocess.Popen(s['command'],cwd=str(r/'source_frozen'),env=env,
    stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
proc=Path('/proc')/str(p.pid)
receipt={'phase':name,'pid':p.pid,'process_identity':{'pid':p.pid,
    'start_ticks':proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19],
    'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()},
    'command_sha256':hashlib.sha256(json.dumps(s['command'],sort_keys=True,
        separators=(',',':')).encode()).hexdigest(),
    'source_sha256':s['source_sha256'],'automatic_retry':False,
    'reference_answer_visible':False,'at':time.time()}
pidfile.write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
"""
    print(remote(code, json.dumps(data).encode("utf-8")).decode("utf-8"), end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("compatibility", "execute"))
    launch(parser.parse_args().phase)
