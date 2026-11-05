"""Launch a prepared paired run; accept the API credential through stdin only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request

COMPAT = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    os.umask(0o077)
    for name in list(os.environ):
        if any(t in name.upper() for t in (
                "API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY", "PASSWORD", "SECRET")):
            os.environ.pop(name, None)
    credential = json.loads(sys.stdin.read())["api_key"]
    if not isinstance(credential, str) or not credential:
        raise ValueError("missing_transient_auth")
    os.environ.update(DASHSCOPE_API_KEY=credential, PYTHONPATH=COMPAT,
                      PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.path[:0] = [str(root / "driver"), str(root / "code/source/src"), COMPAT]
    import paired
    exp = paired.load_exp(root / "code/source")
    manifest = exp.read(root / "manifest.json")
    ready = exp.read(root / "preparation.json")
    if ready["manifest_sha256"] != manifest["content_sha256"]:
        raise ValueError("preparation_identity_changed")
    paired.verify(exp, root, manifest)
    if any((root / name).exists() for name in ("launch.json", "startup.json", "supervise.py")):
        raise ValueError("launch_already_exists")
    governor = exp.RequestGovernor(root / "budget", exp.RequestGovernorConfig(paired.CAP, 4))
    if governor.summary()["reserved_requests"] != 0:
        raise ValueError("new_run_budget_not_empty")
    client = exp.BailianClient(**manifest["profile"]["model"])
    request = urllib.request.Request(client.base_url + "/models",
                                    headers={"Authorization": "Bearer " + credential})
    with client._opener(request, timeout=20) as response:
        raw = response.read()
        if response.status != 200 or credential.encode() in raw:
            raise ValueError("provider_auth_or_privacy_check_failed")
    exp.write_once(root / "provider_auth.json", {
        "at": paired.now(), "authenticated_http_status": 200,
        "model_requests": 0, "credential_written_to_disk": False,
    })
    supervisor = '''
import datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(RUN_ROOT)
def emit(path, body):
    with path.open("x") as stream:
        json.dump(body, stream, sort_keys=True)
        stream.write("\\n")
command = [sys.executable, "-B", str(ROOT/"driver/paired.py"), "--root", str(ROOT)]
with (ROOT/"run.log").open("xb") as log:
    runner = subprocess.Popen(command, stdout=log, stderr=log, env=dict(os.environ))
    proc = Path("/proc")/str(runner.pid)
    stat = (proc/"stat").read_text().split(") ",1)[1].split()
    emit(ROOT/"startup.json", {
        "at": datetime.datetime.now().astimezone().isoformat(),
        "runner_pid": runner.pid, "runner_start_ticks": stat[19],
        "supervisor_pid": os.getpid(),
        "command_sha256": hashlib.sha256(json.dumps(command).encode()).hexdigest(),
        "proc_command_sha256": hashlib.sha256((proc/"cmdline").read_bytes()).hexdigest(),
        "run_root": str(ROOT), "timeout_s": 600, "gpu_allocation": [],
        "credential_written_to_disk": False, "automatic_retry": False,
    })
    code = runner.wait()
emit(ROOT/"exit.json", {
    "at": datetime.datetime.now().astimezone().isoformat(),
    "returncode": code, "completion_present": (ROOT/"completion.json").exists(),
    "automatic_retry": False,
})
'''
    supervisor = supervisor.replace("RUN_ROOT", repr(str(root)))
    with (root / "supervise.py").open("x") as stream:
        stream.write(supervisor)
    socket = root.name
    command = ["tmux", "-L", socket, "new-session", "-d", "-s", "paired", "-c", str(root),
               shlex.join([sys.executable, "-B", str(root / "supervise.py")])]
    subprocess.run(command, check=True, capture_output=True, timeout=15)
    exp.write_once(root / "launch.json", {
        "at": paired.now(), "status": "launched", "manifest_sha256": manifest["content_sha256"],
        "tmux_socket": socket, "tmux_session": "paired", "questions_per_arm": paired.SIZE,
        "episodes": paired.SIZE * 2, "max_requests": paired.CAP, "max_inflight": 4,
        "driver_sha256": manifest["driver_sha256"],
        "launcher_sha256": exp.file_sha(Path(__file__)),
        "supervisor_sha256": exp.file_sha(root / "supervise.py"),
        "credential_written_to_disk": False, "automatic_retry": False,
    })
    print(json.dumps({"status": "launched", "root": str(root), "episodes": paired.SIZE * 2}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__}))
        raise SystemExit(1)
