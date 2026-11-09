"""Launch the frozen SR20 trial with an API credential received through stdin."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    os.umask(0o077)
    for name in list(os.environ):
        if any(t in name.upper() for t in (
                "API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "PASSWORD", "SECRET")):
            os.environ.pop(name, None)
    credential = json.loads(sys.stdin.read())["api_key"]
    if not isinstance(credential, str) or not credential:
        raise ValueError("missing_transient_auth")
    compat = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat"
    os.environ.update(
        DASHSCOPE_API_KEY=credential, PYTHONPATH=compat, CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1")
    sys.path[:0] = [str(root / "source/scripts/BLINKRunner"), compat]
    import sr_trial as trial
    manifest = trial.sr.read(root / "manifest.json")
    if trial.sr.read(root / "preparation.json")["manifest_sha256"] != manifest["content_sha256"]:
        raise ValueError("preparation_changed")
    trial.verify(root, manifest)
    if any((root / name).exists() for name in ("launch.json", "startup.json", "supervise.py")):
        raise ValueError("sr_launch_already_exists")
    governor = trial.sr.RequestGovernor(
        root / "budget", trial.sr.RequestGovernorConfig(trial.CAP, trial.CONCURRENCY))
    jobs, continuation = trial.initial_jobs(root, manifest, governor)
    client = trial.FullResponseClient(**manifest["inspection"]["configuration"]["model"])
    request = urllib.request.Request(client.base_url + "/models",
                                    headers={"Authorization": "Bearer " + credential})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read()
        if response.status != 200 or credential.encode() in raw:
            raise ValueError("provider_auth_or_privacy_check_failed")
    trial.sr.write_once(root / "provider_auth.json", {
        "at": trial.now(), "http_status": 200, "model_requests": 0,
        "credential_written_to_disk": False,
    })
    supervisor = '''
import datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
root = Path(RUN_ROOT)
def emit(name, body):
    with (root/name).open("x") as stream:
        json.dump(body, stream, sort_keys=True)
        stream.write("\\n")
command = [sys.executable, "-B", str(root/"source/scripts/BLINKRunner/sr_trial.py"),
           "run", "--root", str(root)]
with (root/"run.log").open("xb") as log:
    runner = subprocess.Popen(command, stdout=log, stderr=log, env=dict(os.environ))
    proc = Path("/proc")/str(runner.pid)
    stat = (proc/"stat").read_text().split(") ",1)[1].split()
    emit("startup.json", {
        "at": datetime.datetime.now().astimezone().isoformat(),
        "runner_pid": runner.pid, "runner_start_ticks": stat[19],
        "supervisor_pid": os.getpid(),
        "command_sha256": hashlib.sha256(json.dumps(command).encode()).hexdigest(),
        "proc_command_sha256": hashlib.sha256((proc/"cmdline").read_bytes()).hexdigest(),
        "max_inflight": 8, "gpu_allocation": [], "automatic_retry": False,
        "credential_written_to_disk": False,
    })
    code = runner.wait()
emit("exit.json", {"at": datetime.datetime.now().astimezone().isoformat(),
                  "returncode": code, "completed": (root/"completed.json").exists(),
                  "automatic_retry": False})
'''.replace("RUN_ROOT", repr(str(root)))
    with (root / "supervise.py").open("x") as stream:
        stream.write(supervisor)
    subprocess.run([
        "tmux", "-L", root.name, "new-session", "-d", "-s", "sr20", "-c", str(root),
        shlex.join([sys.executable, "-B", str(root / "supervise.py")]),
    ], check=True, capture_output=True, timeout=15)
    trial.sr.write_once(root / "launch.json", {
        "at": trial.now(), "status": "launched",
        "manifest_sha256": manifest["content_sha256"], "max_requests": trial.CAP,
        "max_inflight": trial.CONCURRENCY, "episodes": 40, "new_episodes": len(jobs),
        "prior_requests": continuation["prior_requests"] if continuation else 0,
        "reused_direct_episodes": 20 if continuation else 0,
        "tmux_socket": root.name, "tmux_session": "sr20",
        "launcher_sha256": trial.sr.file_sha(Path(__file__)),
        "supervisor_sha256": trial.sr.file_sha(root / "supervise.py"),
        "credential_written_to_disk": False, "automatic_retry": False,
    })
    print(json.dumps({"status": "launched", "root": str(root),
                      "episodes": 40, "new_episodes": len(jobs)}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1)
