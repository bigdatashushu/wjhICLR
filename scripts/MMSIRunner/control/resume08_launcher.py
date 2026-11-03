"""Launch this one audited continuation only after its preflight exits."""
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/mmsi-365-test-rerun-20261101")
OPS = ROOT / "ops"
PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
SOCKET = "h3d-mmsi-rerun-20261101"
sys.path.insert(0, str(OPS))
import resume08_control as control
subprocess.run(["ls", "-d", str(ROOT), str(OPS), str(control.MONITOR)], capture_output=True, check=True)
with (ROOT / "launcher.lock").open("a+b") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = control.manifest_current()
    with control.operator_locks(manifest):
        assert not (OPS / "resume08_launch.json").exists()
        assert not control.RECOVERY.exists()
        assert control.pilot.read(OPS / "resume08_preflight_exit.json")["returncode"] == 0
        assert not control.watcher.process_state(control.pilot.read(OPS / "resume08_preflight_startup.json"))["alive"]
        assert not control.watcher.process_state(control.pilot.read(ROOT / "startup.json"))["alive"]
        baseline = control.pilot.read(control.PREFLIGHT)
        assert baseline["status"] == "passed" and baseline["model_requests"] == baseline["gpu_requests"] == 0
        control.retained_audit(baseline, check_cache=True)
        control.ledger_audit(manifest, baseline)
        gateway = control.health()
        sessions = subprocess.run(["tmux", "-L", SOCKET, "list-sessions", "-F", "#{session_name}"],
                                  text=True, capture_output=True)
        if sessions.returncode:
            assert sessions.returncode == 1
        assert not set(sessions.stdout.splitlines()) & {
            "evaluation", "resume02-evaluation", "resume03-evaluation", "resume03-monitor",
            "resume04-preflight", "resume04-evaluation", "resume04-monitor",
            "resume05-preflight", "resume05-evaluation", "resume05-monitor",
            "resume07-preflight", "resume07-evaluation", "resume07-monitor",
            "resume08-preflight", "resume08-evaluation", "resume08-monitor"}
        control.pilot.write(OPS / "resume08_launch.json", {
            "status": "launch_once", "at": control.pilot.now(), "resume_attempt": 8,
            "preflight_sha256": control.pilot.sha(control.PREFLIGHT),
            "retained_predictions": 653, "retained_service_errors": 11, "remaining_episodes": 77,
            "upstream_timeout_s": 1200, "client_timeout_s": 1230,
            "timeout_policy_sha256": control.pilot.sha(control.TIMEOUT_POLICY),
            "max_workers": 4, "automatic_retry": False, "gateway": gateway,
            "other_experiments_resumed": False})
    command = "exec " + shlex.quote(PYTHON) + " -B " + shlex.quote(str(OPS / "resume08_entry.py"))
    command += " > " + shlex.quote(str(OPS / "resume08_run.log")) + " 2>&1"
    subprocess.run(["tmux", "-L", SOCKET, "new-session", "-d",
                    "-s", "resume08-evaluation", command], check=True)
    print(json.dumps({"status": "continuation_launched", "at": control.pilot.now(),
                      "resume_attempt": 8, "max_workers": 4, "retained_service_errors": 11,
                      "remaining_episodes": 77, "automatic_model_retries": 0,
                      "upstream_timeout_s": 1200, "client_timeout_s": 1230,
                      "manifest_sha256": control.MANIFEST_SHA, "other_experiments_resumed": False}))
