#!/usr/bin/env python3
"""Use authorized server.md credentials transiently, never in output or argv."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def passwords(server_file):
    text = Path(server_file).read_text()
    match = re.search(r"密码是([^\n]+)", text)
    if not match:
        return []
    return [value.strip().replace(r"\@", "@").rstrip("\\")
            for value in match[1].split("或") if value.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host")
    parser.add_argument("command")
    parser.add_argument("--stdin-file", type=Path)
    parser.add_argument("--server-file", type=Path,
                        default=Path(__file__).resolve().parents[2] / "server.md")
    args = parser.parse_args()
    input_bytes = args.stdin_file.read_bytes() if args.stdin_file else None
    return remote_run(args.host, args.command, args.server_file, input_bytes)


def remote_run(host, command, server_file=None, input_bytes=None, return_output=False):
    server_file = server_file or Path(__file__).resolve().parents[2] / "server.md"
    base = ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]
    control_name = {
        "wangy@10.10.0.34": "VSI8B_SSH_CONTROL_TITAN",
        "cvailab@10.10.0.198": "VSI8B_SSH_CONTROL_4090",
    }.get(host)
    control = os.environ.get(control_name, "") if control_name else ""
    if control and Path(control).exists():
        base.extend(["-S", control])
    result = subprocess.run([*base, "-o", "BatchMode=yes", host, command],
                            input=input_bytes, capture_output=True)
    if result.returncode != 255:
        if return_output:
            return result.returncode, result.stdout, result.stderr
        sys.stdout.buffer.write(result.stdout)
        sys.stderr.buffer.write(result.stderr)
        return result.returncode
    with tempfile.TemporaryDirectory(prefix="vsi8b-auth-") as temporary:
        helper = Path(temporary) / "askpass.py"
        helper.write_text("#!/usr/bin/env python3\nimport os\n"
                          "print(os.environ['VSI8B_SSH_PASSWORD'])\n")
        helper.chmod(0o700)
        for password in passwords(server_file):
            env = {**os.environ, "SSH_ASKPASS": str(helper),
                   "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": "vsi8b",
                   "VSI8B_SSH_PASSWORD": password}
            result = subprocess.run(
                [*base, "-o", "NumberOfPasswordPrompts=1",
                 "-o", "PreferredAuthentications=password", host, command],
                env=env, capture_output=True, input=input_bytes)
            if result.returncode != 255:
                if password.encode() in result.stdout or password.encode() in result.stderr:
                    raise RuntimeError("remote output contains a credential")
                if return_output:
                    return result.returncode, result.stdout, result.stderr
                sys.stdout.buffer.write(result.stdout)
                sys.stderr.buffer.write(result.stderr)
                return result.returncode
    print("SSH authentication unavailable", file=sys.stderr)
    if return_output:
        return 255, b"", b"SSH authentication unavailable"
    return 255


if __name__ == "__main__":
    raise SystemExit(main())
