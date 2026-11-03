"""Launch resume 02 once after preflight, then attach its independent monitor."""
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time

ROOT = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031/mmsi-365-test-rerun-20261101")
OPS = ROOT / "ops"
PYTHON = "/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/skill3d-exp/bin/python"
sys.path.insert(0, str(OPS))
import resume_control as control
pilot = control.pilot

with (ROOT / "launcher.lock").open("a+b") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = control.manifest_current()
    baseline = pilot.read(control.PREFLIGHT)
    assert baseline["status"] == "passed"
    assert pilot.read(OPS / "resume02_preflight_exit.json")["returncode"] == 0
    assert not control.watcher.process_state(pilot.read(ROOT / "startup.json"))["alive"]
    assert not control.RECOVERY.exists() and not (OPS / "resume02_launch.json").exists()
    control.retained_audit(baseline, check_cache=True)
    control.ledger_audit(manifest, baseline)
    control.health()
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    assert fd >= 0
    try:
        assert libc.inotify_add_watch(fd, os.fsencode(ROOT), 0x00000008 | 0x00000080 | 0x00000100) >= 0
        pilot.write(OPS / "resume02_launch.json", {
            "status": "launch_once", "at": pilot.now(), "manifest_sha256": control.MANIFEST_SHA,
            "preflight_sha256": pilot.sha(control.PREFLIGHT), "max_workers": 4,
            "remaining_never_dispatched": 523, "automatic_retry": False,
            "other_experiments_resumed": False, "model_requests_issued_by_launcher": 0,
            "monitor_guard_sha256": pilot.sha(OPS / "resume_monitor_guard.py")})
        command = "exec " + shlex.quote(PYTHON) + " -B " + shlex.quote(str(OPS / "resume_entry.py"))
        command += " > " + shlex.quote(str(OPS / "resume02_run.log")) + " 2>&1"
        subprocess.run(["tmux", "-L", "h3d-mmsi-rerun-20261101", "new-session", "-d",
                        "-s", "resume02-evaluation", command], check=True)
        deadline = time.monotonic() + 45
        startup = None
        while time.monotonic() < deadline:
            if (OPS / "resume02_launch_failure.json").exists():
                raise RuntimeError("resume_startup_failed")
            if control.RECOVERY.exists() and (ROOT / "startup.json").exists():
                value = pilot.read(ROOT / "startup.json")
                if value.get("resume_attempt") == 2:
                    assert control.watcher.process_state(value)["alive"]
                    startup = value
                    break
            ready, _, _ = select.select([fd], [], [], max(0, deadline - time.monotonic()))
            if ready:
                os.read(fd, 65536)
        assert startup is not None
        command = "exec " + shlex.quote(PYTHON) + " -B " + shlex.quote(str(OPS / "resume_monitor_guard.py"))
        command += " > " + shlex.quote(str(control.MONITOR / "monitor.log")) + " 2>&1"
        subprocess.run(["tmux", "-L", "h3d-mmsi-rerun-20261101", "new-session", "-d",
                        "-s", "resume02-monitor", command], check=True)
        print(json.dumps({"status": "resumed_and_monitor_started", "at": pilot.now(),
                          "runner": startup, "recovery": pilot.read(control.RECOVERY),
                          "other_experiments_resumed": False}))
    finally:
        os.close(fd)
