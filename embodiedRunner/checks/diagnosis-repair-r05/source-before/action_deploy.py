"""Copy only task code into a new isolated server workspace; preserve V1."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import tarfile

from action_identity import HOST_CONFIGS, LOCAL_SOURCES, source_identity
from protocol import ROOT, canonical

HOST = "cvailab@10.10.0.198"
REMOTE_PARENT = "/home/cvailab/wangjh"
DEFAULT_REMOTE = REMOTE_PARENT + "/embodied-action-v2-20261105"


def ssh(command):
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-S", str(ROOT / "checks/ssh4090"), HOST, command]


def deploy(remote=DEFAULT_REMOTE):
    if not remote.startswith(REMOTE_PARENT + "/embodied-action-v2-") or ".." in remote:
        raise ValueError("invalid isolated deployment directory")
    subprocess.run(ssh("ls -d " + shlex.quote(REMOTE_PARENT)), check=True, capture_output=True)
    paths = set(source_identity()["files_sha256"])
    paths.update("embodiedRunner/" + name for name in LOCAL_SOURCES)
    paths.update("embodiedRunner/" + name for name in HOST_CONFIGS)
    paths.update("embodiedRunner/" + name for name in
                 ("action_deploy.py", "action_live_launch.py", "bridge.py", "alfred_pilot_data.json",
                  "config/action-live-4090.json", "checks/architecture-preparation/local-readonly.json"))
    paths.update(str(p.relative_to(ROOT.parent)) for p in (ROOT / "tests").glob("*.py"))
    # Extra source modules are import dependencies; no datasets, runs or keys.
    paths.update(str(p.relative_to(ROOT.parent)) for p in (ROOT.parent / "src").rglob("*.py"))
    paths = {p for p in paths if not p.startswith(
        ("embodiedRunner/upstream/", "embodiedRunner/upstream-household/",
         "embodiedRunner/upstream-manipulation/"))}
    archive = ROOT / "tmp/action-v2-code.tar"
    with tarfile.open(archive, "w") as tar:
        for name in sorted(paths):
            tar.add(ROOT.parent / name, arcname=name, recursive=False)
    bootstrap = """import sys,tarfile,pathlib
r=pathlib.Path(sys.argv[1])
r.mkdir(mode=0o700, exist_ok=False)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as t:
    for m in t:
        p=r/m.name
        if not p.resolve().is_relative_to(r.resolve()) or not m.isfile():
            raise ValueError('invalid code archive')
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as f:
            f.write(t.extractfile(m).read())
b=r/'embodiedRunner'
for name in ('runtime','cache','data','upstream','upstream-household','upstream-manipulation',
             'manipulation-runtime','venv-nav','venv-household'):
    p=b/name
    if not p.exists():
        p.symlink_to(pathlib.Path('/home/cvailab/wangjh/embodiedRunner')/name)
for name in ('checks','tmp','runs','config','state'):
    (b/name).mkdir(exist_ok=True)
print('{"status":"deployed","shared_source_modified":false}')
"""
    command = shlex.join(["/home/cvailab/anaconda3/envs/skill3d-exp/bin/python", "-B",
                          "-c", bootstrap, remote])
    with archive.open("rb") as data:
        subprocess.run(ssh(command), stdin=data, check=True, timeout=120)
    return {"status": "deployed", "remote_root_sha256":
            __import__("hashlib").sha256(remote.encode()).hexdigest(), "files": len(paths)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    print(canonical(deploy(parser.parse_args().remote)).decode())
