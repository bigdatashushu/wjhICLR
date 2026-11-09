"""Package public inputs, use the already-authenticated SSH session, and start."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

from .common import ROOT, file_digest, read, read_keys, write

SSH = ["/usr/bin/ssh", "-F",
       "/Users/bytedance/wangjh/.trae/skills/harness3d-cluster-ops/assets/ssh_config",
       "-S", "/Users/bytedance/.cache/harness3d-ssh/blink-gpt", "-T",
       "-o", "BatchMode=yes", "h3d-titan"]
PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"


def remote_python(remote, mode):
    return ("env PYTHONPATH=" + shlex.quote(remote + "/code") +
            " " + shlex.quote(PYTHON)
            + " -m BLINKGPT.run " + mode + " --root " + shlex.quote(remote))


def ssh(command, payload=None, timeout=None):
    p = subprocess.run([*SSH, command], input=payload, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=timeout)
    if p.returncode:
        # Commands never contain keys; this process has no API-key environment.
        sys.stderr.write(p.stderr.decode())
        raise RuntimeError("ssh_command_failed")
    return p.stdout


def deploy(local, remote, *, update_before_start=False):
    if remote.rstrip("/") != remote or not remote.startswith("/data/wangy/wangjh/harness3d/"):
        raise ValueError("requires_isolated_task_root")
    ssh("ls " + shlex.quote(str(Path(remote).parent)))
    bundle = local / "deployment.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(local / "manifest.json", arcname="manifest.json")
        tar.add(local / "public", arcname="public")
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".txt", ".json"} \
                    and path.relative_to(ROOT).parts[0] not in {"runs", "deployment", "__pycache__"} \
                    and "__pycache__" not in path.parts:
                tar.add(path, arcname="code/BLINKGPT/" + str(path.relative_to(ROOT)))
    if update_before_start:
        guard = " && ".join("test ! -e " + shlex.quote(remote + "/" + name)
                            for name in ("launch.json", "owner.json", "completed.json"))
        guard += " && test ! -d " + shlex.quote(remote + "/requests")
        ssh(guard + " && tar -xzf - -C " + shlex.quote(remote), payload=bundle.read_bytes())
    else:
        # mkdir without -p refuses to overwrite an existing task.
        ssh("mkdir " + shlex.quote(remote) + " && tar -xzf - -C " + shlex.quote(remote),
            payload=bundle.read_bytes())
    write(local / "deployment.json", {"remote_root": remote, "bundle_sha256": file_digest(bundle),
                                     "manifest_sha256": file_digest(local / "manifest.json")},
          once=not update_before_start)
    print(ssh(remote_python(remote, "preflight")).decode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("deploy", "update-before-start", "preflight", "start", "qualify", "snapshot"))
    p.add_argument("--local-root", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    args = p.parse_args()
    if args.mode in ("deploy", "update-before-start"):
        deploy(args.local_root, args.remote_root, update_before_start=args.mode == "update-before-start")
    elif args.mode == "snapshot":
        data = json.loads(ssh(remote_python(args.remote_root, "snapshot")))
        write(args.local_root / "snapshot.json", data)
        print(json.dumps({k: v for k, v in data.items() if k not in {"episodes", "slots"}}, ensure_ascii=False))
    elif args.mode == "preflight":
        print(ssh(remote_python(args.remote_root, "preflight")).decode())
    else:
        keys = read_keys(ROOT.parents[1] / "xcodeapi/credentials.local.env")
        payload = json.dumps({"keys": keys}).encode()
        result = json.loads(ssh(remote_python(args.remote_root, args.mode), payload=payload))
        if args.mode == "start":
            write(args.local_root / "launch.json", result, once=True)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
