"""Owned tmux execution with a transient Unix-socket credential handoff."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import struct
import subprocess
import sys
import tempfile


def launch(root, mode, output, config, credential):
    root = Path(root)
    session = "embodied-" + Path(output).stem
    log = root / "runs" / (Path(output).stem + "-log.jsonl")
    startup = root / "runs" / (Path(output).stem + "-startup.json")
    completion = root / "runs" / (Path(output).stem + "-completion.json")
    if any(path.exists() for path in (log, startup, completion, root / "runs" / output)):
        raise ValueError("detached_output_exists")
    present = subprocess.run(["tmux", "has-session", "-t", "=" + session],
                             capture_output=True).returncode == 0
    if present:
        raise ValueError("detached_session_exists")
    with tempfile.TemporaryDirectory(prefix="embodied-auth-") as directory:
        os.chmod(directory, 0o700)
        address = str(Path(directory) / "handoff.sock")
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(address)
            os.chmod(address, 0o600)
            listener.listen(1)
            listener.settimeout(30)
            command = [sys.executable, "-B", str(root / "action_detach.py"), "--worker",
                       "--socket", address, "--root", str(root), "--mode=" + mode,
                       "--output", output, "--config", config, "--session", session]
            subprocess.run(["tmux", "new-session", "-d", "-s", session, shlex.join(command)],
                           check=True, capture_output=True)
            with listener.accept()[0] as channel:
                channel.settimeout(30)
                peer_pid, peer_uid, _ = struct.unpack(
                    "3i", channel.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if peer_uid != os.getuid():
                    raise ValueError("detached_peer_identity")
                channel.sendall(json.dumps({"api_key": credential}).encode() + b"\n")
                with channel.makefile("rb") as response:
                    receipt = json.loads(response.readline(65537))
                if receipt["pid"] != peer_pid:
                    raise ValueError("detached_worker_identity")
                return receipt


def worker(args):
    root = Path(args.root)
    # Only this owned process receives auth; no secret enters argv or a file.
    with socket.socket(socket.AF_UNIX) as channel:
        channel.settimeout(30)
        channel.connect(args.socket)
        with channel.makefile("rb") as stream:
            credential = json.loads(stream.readline(65537))["api_key"]
        if credential:
            os.environ["DASHSCOPE_API_KEY"] = credential
        sys.path[:0] = [str(root), str(root.parent / "src")]
        from protocol import canonical
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        receipt = {
            "schema": "embodied-detached-startup/1", "status": "started",
            "pid": os.getpid(), "start_ticks": int(fields[19]),
            "cmdline_sha256": hashlib.sha256(Path("/proc/self/cmdline").read_bytes()).hexdigest(),
            "tmux_session": args.session, "mode": args.mode, "output": args.output,
            "config": args.config, "credential_persisted": False,
            "automatic_retry": False,
        }
        startup = root / "runs" / (Path(args.output).stem + "-startup.json")
        with startup.open("xb") as stream:
            os.chmod(startup, 0o600)
            stream.write(canonical(receipt) + b"\n")
        channel.sendall(canonical(receipt) + b"\n")
    log_path = root / "runs" / (Path(args.output).stem + "-log.jsonl")
    code = 0
    with log_path.open("x", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        os.chmod(log_path, 0o600)
        try:
            auth_path = "/home/cvailab/wangjh/harness3d/four-datasets-20261031/transport_auth.py"
            spec = importlib.util.spec_from_file_location("authorized_transport_auth", auth_path)
            auth = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(auth)
            # Do not persist loader diagnostics or inherited credentials.
            with open(os.devnull, "w") as discard, contextlib.redirect_stdout(discard), contextlib.redirect_stderr(discard):
                auth.load_transport_token()
            sys.argv = ["action_run.py", args.mode, "--live-config", str(root / "config" / args.config),
                        "--output", str(root / "runs" / args.output)]
            from action_run import main
            main()
        except SystemExit as error:
            code = error.code if type(error.code) is int else 1
        except BaseException as error:
            code = 1
            print(canonical({"event": "detached_failure",
                             "reason": type(error).__name__.lower()}).decode(), flush=True)
    completion = root / "runs" / (Path(args.output).stem + "-completion.json")
    with completion.open("xb") as stream:
        os.chmod(completion, 0o600)
        stream.write(canonical({
            "schema": "embodied-detached-completion/1", "pid": receipt["pid"],
            "start_ticks": receipt["start_ticks"], "exit_code": code,
            "output_exists": (root / "runs" / args.output).is_file(),
            "automatic_retry": False}) + b"\n")
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--mode", choices=["--live", "--live-check"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--session", required=True)
    raise SystemExit(worker(parser.parse_args()))
