"""Run r06 in distinct frozen pilot and complete-development workspaces."""
import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("r05_control", ROOT / "checks/diagnosis-repair-r05/control.py")
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)

parser = argparse.ArgumentParser()
parser.add_argument("scope", choices=("pilot", "full"))
parser.add_argument("operation", choices=("deploy", "assets", "check", "live", "collect", "collect-completed"))
args = parser.parse_args()
control.HERE = HERE / args.scope
assert control.HERE.is_dir()
control.REMOTE = f"/home/cvailab/wangjh/embodied-action-v2-task-card-{args.scope}-20261106-r06"
control.STEM = f"task-card-{args.scope}-r06"
control.CONFIG = "action-task-card-pilot-r06.json" if args.scope == "pilot" else "action-task-card-15-4090.json"

def deploy():
    import json
    import shlex
    import subprocess
    receipt = control.action_deploy.deploy(
        control.REMOTE, archive_path=ROOT / f"tmp/task-card-{args.scope}-r06-code.tar")
    remote_code = """import sys,json,pathlib
r=pathlib.Path(sys.argv[1])/'embodiedRunner'
sys.path[:0]=[str(r),str(r.parent/'src')]
from action_identity import source_identity
print(json.dumps(source_identity()))
"""
    command = shlex.join(["/home/cvailab/anaconda3/envs/skill3d-exp/bin/python", "-B", "-c",
                          remote_code, control.REMOTE])
    identity = json.loads(subprocess.check_output(control.ssh(command), timeout=60))
    assert identity == control.source_identity()
    control.save("deployment.json", {"receipt": receipt, "source": identity, "remote": control.REMOTE})
    print(json.dumps({"status": "deployed", "source_identity_match": True, "remote": control.REMOTE}))

def collect_completed():
    import json
    import shlex
    import subprocess
    import tarfile
    code = """import sys,tarfile,pathlib,json
r=pathlib.Path(sys.argv[1])/'embodiedRunner'/'runs'
s=sys.argv[2]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
    for p in sorted((r/(s+'-episodes')).glob('trajectory-*/trajectory.json')):
        d=json.loads(p.read_bytes())
        if d['events'] and d['events'][-1]['kind']=='episode_result':
            t.add(p.parent,arcname=str(p.parent.relative_to(r)))
"""
    command = shlex.join(["/home/cvailab/anaconda3/envs/skill3d-exp/bin/python", "-B", "-c",
                          code, control.REMOTE, control.STEM])
    process = subprocess.Popen(control.ssh(command), stdout=subprocess.PIPE)
    files = 0
    with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
        for member in tar:
            target = ROOT / "runs" / member.name
            if not target.resolve().is_relative_to((ROOT / "runs").resolve()):
                raise ValueError("invalid collected path")
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
            elif member.isfile():
                data = tar.extractfile(member).read()
                if target.exists():
                    assert target.read_bytes() == data
                else:
                    with target.open("xb") as f:
                        f.write(data)
                    target.chmod(0o600)
                files += 1
            else:
                raise ValueError("unexpected collected file type")
    assert process.wait(timeout=60) == 0
    print(json.dumps({"status": "immutable_completed_traces_collected", "files": files}))

{"deploy": deploy, "assets": control.assets, "check": lambda: control.launch(True),
 "live": control.launch, "collect": control.collect, "collect-completed": collect_completed}[args.operation]()
