"""Start only the prepared 24-question pilot with transient inherited API auth."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request

os.umask(0o077)
ROOT = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031")
PILOT = ROOT/"blink-pilot24-direct-20261031"
RECOVERY = ROOT/"blink-recovery-20261031-1ee33f28c07e"
COMPAT = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat"
SOCKET = "h3d-blink-pilot24-direct-20261031"
for name in list(os.environ):
    if name != "DASHSCOPE_API_KEY" and any(s in name.upper() for s in (
            "API_KEY", "TOKEN", "ENDPOINT", "BASE_URL", "MODELS_KEY")):
        os.environ.pop(name, None)
os.environ.update(PYTHONPATH=COMPAT, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
sys.path[:0] = [str(PILOT), str(RECOVERY/"code/source/src"), COMPAT]
from skill3d.evaluation import blink_experiment as exp
from skill3d.synthesis.bailian_client import BailianClient
from continuation import campaign_class
RevisedCampaign = campaign_class(exp)

try:
    ready = exp.read(PILOT/"ready.json")
    manifest = exp.read(PILOT/"manifest.json")
    if ready["pilot_sha256"] != manifest["content_sha256"]:
        raise ValueError("pilot_ready_identity_changed")
    if (exp.file_sha(PILOT/"pilot.py") != manifest["driver_sha256"]
            or exp.file_sha(PILOT/"continuation.py") != manifest["continuation_driver_sha256"]):
        raise ValueError("pilot_source_changed")
    if (PILOT/"launch.json").exists() or (PILOT/"startup.json").exists():
        raise ValueError("pilot_launch_already_exists")
    campaign = RevisedCampaign(PILOT/"campaign", exp.read(PILOT/"campaign/manifest.json"))
    campaign.verify()
    if campaign.governor.summary() != manifest["budget_start"]:
        raise ValueError("pilot_start_budget_changed")
    client = BailianClient(**campaign.profile["model"])
    if not client.configured:
        raise ValueError("missing_transient_provider_auth")
    request = urllib.request.Request(client.base_url+"/models",
                                    headers={"Authorization": "Bearer "+os.environ["DASHSCOPE_API_KEY"]})
    with client._opener(request, timeout=20) as response:
        if response.status != 200:
            raise ValueError("provider_auth_rejected")
        response.read()
    exp.write_once(PILOT/"provider_auth.json", {
        "at": datetime.datetime.now().astimezone().isoformat(),
        "authenticated_http_status": 200, "model_requests": 0,
        "credential_written_to_disk": False,
    })
    supervisor_source = '''
import datetime, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(PILOT_ROOT)
def emit(path, body):
    with path.open("x") as stream:
        json.dump(body, stream, sort_keys=True)
        stream.write("\\n")
command = [sys.executable, "-B", str(ROOT/"pilot.py"), "--root", str(ROOT)]
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
        "pilot_root": str(ROOT), "timeout_s": 600,
        "credential_written_to_disk": False, "automatic_retry": False,
    })
    code = runner.wait()
emit(ROOT/"exit.json", {
    "at": datetime.datetime.now().astimezone().isoformat(),
    "returncode": code, "completion_present": (ROOT/"completion.json").exists(),
    "automatic_retry": False,
})
'''
    supervisor_source = supervisor_source.replace("PILOT_ROOT", repr(str(PILOT)))
    with (PILOT/"supervise.py").open("x") as stream:
        stream.write(supervisor_source)
    command = ["tmux", "-L", SOCKET, "new-session", "-d", "-s", "pilot", "-c", str(PILOT),
               shlex.join([sys.executable, "-B", str(PILOT/"supervise.py")])]
    subprocess.run(command, check=True, capture_output=True, timeout=15)
    record = exp.write_once(PILOT/"launch.json", {
        "status": "launched", "at": datetime.datetime.now().astimezone().isoformat(),
        "pilot": str(PILOT), "pilot_sha256": manifest["content_sha256"],
        "tmux_socket": SOCKET, "tmux_session": "pilot",
        "questions": 24, "episodes": 24, "arms": manifest["arms"],
        "max_additional_requests": 24, "global_max_requests": 1341,
        "timeout_s": 600, "max_inflight": 4, "source_split": "train",
        "credential_written_to_disk": False, "automatic_retry": False,
        "full_campaign_started": False,
    })
    print(json.dumps(record))
except BaseException as exc:
    print(json.dumps({"status": "stopped", "error_type": type(exc).__name__,
                      "automatic_retry": False, "credential_written_to_disk": False}))
    raise SystemExit(2)
