#!/usr/bin/env python3
"""Restore the original reviewer transport without restarting its service."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import urllib.request

from ssh_from_server import passwords
from launch_pilot40 import PYTHON_TITAN

EXPECTED = "c16e97ae7e43e8605e201de8388eb9a318bc5f260e05a59e869943a8762c0cb2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-path", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    try:
        with opener("http://127.0.0.1:29248/health", timeout=3) as response:
            health = json.load(response)
        if health["identity_sha256"] != EXPECTED:
            raise ValueError("local forwarding belongs to a different reviewer")
    except OSError:
        base = [
            "ssh", "-f", "-N", "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
            "-o", "ServerAliveCountMax=3", "-L", "127.0.0.1:29248:127.0.0.1:27248",
        ]
        result = subprocess.run(
            [*base, "-o", "BatchMode=yes", "cvailab@10.10.0.198"],
            capture_output=True)
        if result.returncode:
            with tempfile.TemporaryDirectory(prefix="vsi8b-forward-auth-") as directory:
                helper = Path(directory) / "askpass.py"
                helper.write_text(
                    "#!/usr/bin/env python3\nimport os\n"
                    "print(os.environ['VSI8B_SSH_PASSWORD'])\n")
                helper.chmod(0o700)
                for password in passwords(Path(__file__).resolve().parents[2] / "server.md"):
                    result = subprocess.run(
                        [*base, "-o", "NumberOfPasswordPrompts=1", "-o",
                         "PreferredAuthentications=password", "cvailab@10.10.0.198"],
                        env={**os.environ, "SSH_ASKPASS": str(helper),
                             "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": "vsi8b",
                             "VSI8B_SSH_PASSWORD": password},
                        capture_output=True)
                    if result.returncode == 0:
                        break
            if result.returncode:
                raise RuntimeError("reviewer local forwarding authentication failed")
    with opener("http://127.0.0.1:29248/health", timeout=15) as response:
        health = json.load(response)
    if health["identity_sha256"] != EXPECTED:
        raise ValueError("local reviewer identity changed")
    command = [
        "ssh", "-S", args.control_path, "-O", "forward",
        "-o", "ExitOnForwardFailure=yes",
        "-R", "127.0.0.1:27248:127.0.0.1:29248", "wangy@10.10.0.34",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise RuntimeError("Titan reverse forwarding could not be established")
    code = """import json,urllib.request
o=urllib.request.build_opener(urllib.request.ProxyHandler({})).open
with o('http://127.0.0.1:27248/health',timeout=15) as r:
    h=json.load(r)
assert h['identity_sha256']==%r
print(json.dumps({'available':True,'identity_sha256':h['identity_sha256']}))
""" % EXPECTED
    result = subprocess.run(
        ["ssh", "-S", args.control_path, "-o", "BatchMode=yes", "wangy@10.10.0.34",
         PYTHON_TITAN + " -c " + shlex.quote(code)], capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError("Titan reviewer health verification failed")
    receipt = {"checked_at": time.time(), "transport": "Mac local plus Titan reverse SSH",
               "service_restarted": False, "titan_reviewer": json.loads(result.stdout)}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
