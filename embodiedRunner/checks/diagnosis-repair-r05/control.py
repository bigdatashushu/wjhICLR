"""Scoped deployment and credential-transient launch for the r05 pilot."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "src")]
import action_deploy
from action_contract import public_hash
from action_identity import source_identity
from protocol import canonical

REMOTE = "/home/cvailab/wangjh/embodied-action-v2-task-card-pilot-20261106-r05"
STEM = "task-card-pilot-r05"
CONFIG = "action-task-card-pilot-r05.json"


def ssh(command):
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-S", "/private/tmp/embodied-diagnosis-20261106.sock",
            "cvailab@10.10.0.198", command]


action_deploy.ssh = ssh


def save(name, data):
    with (HERE / name).open("xb") as f:
        f.write(canonical(data) + b"\n")


def deploy():
    receipt = action_deploy.deploy(REMOTE, archive_path=ROOT / "tmp/task-card-pilot-r05-code.tar")
    remote_code = """import sys,json,pathlib
r=pathlib.Path(sys.argv[1])/'embodiedRunner'
sys.path[:0]=[str(r),str(r.parent/'src')]
from action_identity import source_identity
print(json.dumps(source_identity()))
"""
    command = shlex.join(["/home/cvailab/anaconda3/envs/skill3d-exp/bin/python", "-B", "-c",
                          remote_code, REMOTE])
    identity = json.loads(subprocess.check_output(ssh(command), timeout=60))
    assert identity == source_identity()
    save("deployment.json", {"receipt": receipt, "source": identity, "remote": REMOTE})
    print(json.dumps({"status": "deployed", "source_identity_match": True, "remote": REMOTE}))


def launch(check=False):
    import action_live_launch
    assert json.loads((HERE / "deployment.json").read_bytes())["source"] == source_identity()
    sys.argv = ["action_live_launch.py", "--remote", REMOTE, "--config-name", CONFIG,
                "--output-name", STEM + ("-preflight.json" if check else ".json"),
                "--timeout-s", "12000"]
    if check:
        sys.argv.append("--check")
    action_live_launch.main()


def assets():
    code = """import pathlib,json,hashlib,sys
r=pathlib.Path(sys.argv[1])/'embodiedRunner'
p=pathlib.Path('/home/cvailab/wangjh/embodied-action-v2-compare-20261105-r03/embodiedRunner/action-alfred-data')
assert p.is_dir()
d=r/'action-alfred-data'
assert not d.exists() and not d.is_symlink()
d.symlink_to(p,target_is_directory=True)
m=json.loads((r/'action_alfred_data.json').read_bytes())
for row in m['files']:
    assert hashlib.sha256((d/row['path']).read_bytes()).hexdigest()==row['sha256']
print(json.dumps({'status':'linked_verified','files':len(m['files']),'historical_data_modified':False}))
"""
    command = shlex.join(["/home/cvailab/anaconda3/envs/skill3d-exp/bin/python", "-B", "-c",
                          code, REMOTE])
    receipt = json.loads(subprocess.check_output(ssh(command), timeout=60))
    save("asset-link.json", receipt)
    print(json.dumps(receipt))


def collect():
    code = """import sys,tarfile,pathlib
r=pathlib.Path(sys.argv[1])/'embodiedRunner'/'runs'
s=sys.argv[2]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:
    for n in (s+'.json',s+'-preflight.json',s+'-episodes'):
        p=r/n
        if p.exists(): t.add(p,arcname=n)
"""
    command = shlex.join(["/home/cvailab/anaconda3/envs/skill3d-exp/bin/python", "-B", "-c",
                          code, REMOTE, STEM])
    process = subprocess.Popen(ssh(command), stdout=subprocess.PIPE)
    count = 0
    with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
        for member in tar:
            target = ROOT / "runs" / member.name
            if not target.resolve().is_relative_to((ROOT / "runs").resolve()):
                raise ValueError("invalid collected path")
            if member.isdir():
                target.mkdir(mode=0o700, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                data = tar.extractfile(member).read()
                if target.exists():
                    assert target.read_bytes() == data
                else:
                    with target.open("xb") as f:
                        f.write(data)
                    target.chmod(0o600)
                count += 1
            else:
                raise ValueError("unexpected collected file type")
    assert process.wait(timeout=60) == 0
    print(json.dumps({"status": "collected", "files": count}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("deploy", "assets", "check", "live", "collect"))
    args = parser.parse_args()
    {"deploy": deploy, "assets": assets, "check": lambda: launch(True),
     "live": launch, "collect": collect}[args.operation]()
