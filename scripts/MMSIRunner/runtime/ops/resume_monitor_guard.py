"""Wait for actual runner exit before accepting its final completion receipt."""
from pathlib import Path
import sys
import traceback

import resume_monitor as monitor
original_snapshot = monitor.snapshot


def snapshot_after_exit():
    value = original_snapshot()
    if value["status"] == "verifying_completion" and value["runner_process"]["alive"]:
        # exit.json is written immediately before the process exits.
        # The monitor's existing event wait checks the process again.
        value["status"] = "waiting_for_runner_exit"
    return value


monitor.snapshot = snapshot_after_exit
code = 0
try:
    monitor.main()
except BaseException as exc:
    code = 2
    monitor.pilot.write(monitor.MONITOR / "failure.json", {
        "at": monitor.pilot.now(), "error_type": type(exc).__name__,
        "model_requests_issued_by_monitor": 0,
        "locations": [{"file": Path(f.filename).name, "line": f.lineno}
                      for f in traceback.extract_tb(exc.__traceback__)]})
monitor.pilot.write(monitor.MONITOR / "exit.json", {
    "returncode": code, "at": monitor.pilot.now(),
    "guard_source_sha256": monitor.pilot.sha(__file__)})
sys.exit(code)
