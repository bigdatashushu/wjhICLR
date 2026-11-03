"""Start independent read-only acceptance monitor for resume 06."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/mmsi-365-test-rerun-20261101")
OPS = ROOT / "ops"
PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
sys.path.insert(0, str(OPS))
import resume08_control as control
with (ROOT / "launcher.lock").open("a+b") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert control.pilot.read(ROOT / "startup.json")["resume_attempt"] == 8
    assert control.watcher.process_state(control.pilot.read(ROOT / "startup.json"))["alive"]
    assert control.pilot.read(control.RECOVERY)["resume_attempt"] == 8
    assert not (control.MONITOR / "startup.json").exists()
    assert not (control.MONITOR / "launch.json").exists()
    sessions = subprocess.run(["tmux", "-L", "h3d-mmsi-rerun-20261101", "list-sessions", "-F", "#{session_name}"],
                              text=True, capture_output=True, check=True)
    assert "resume08-evaluation" in sessions.stdout.splitlines()
    assert "resume08-monitor" not in sessions.stdout.splitlines()
    control.pilot.write(control.MONITOR / "launch.json", {
        "status": "launch_once", "at": control.pilot.now(), "resume_attempt": 8,
        "model_requests_issued_by_monitor": 0, "reference_answers_read_by_monitor": False})
    command = "exec " + shlex.quote(PYTHON) + " -B " + shlex.quote(str(OPS / "resume08_monitor.py"))
    command += " > " + shlex.quote(str(control.MONITOR / "run.log")) + " 2>&1"
    subprocess.run(["tmux", "-L", "h3d-mmsi-rerun-20261101", "new-session", "-d",
                    "-s", "resume08-monitor", command], check=True)
    print(json.dumps({"status": "read_only_monitor_started", "at": control.pilot.now(),
                      "model_requests_issued_by_monitor": 0, "retained_service_errors": 11,
                      "waits_for_actual_runner_exit": True}))
