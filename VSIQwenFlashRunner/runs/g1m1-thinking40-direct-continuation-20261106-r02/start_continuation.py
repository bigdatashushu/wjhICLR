"""Launch this frozen direct continuation exactly once."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
assert not (root / "launcher_start.json").exists()
(root / "launcher_start.json").write_text(json.dumps({
    "pid": os.getpid(), "started_at": time.time(), "automatic_model_retry": False,
    "enable_thinking": True}, sort_keys=True, indent=2) + "\n")
environment = dict(os.environ)
environment.update(
    LD_PRELOAD="/home/wangy/wangjh/harness3d/precompute-20261022-r01/cpu_sqlite/libsqlite3.so.0",
    PYTHONPATH="/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat",
    OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
    CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
with (root / "direct.log").open("xb") as log:
    process = subprocess.Popen([
        sys.executable, str(root / "controller.py"), "--run-root", str(root),
        "--action", "direct", "--skip-validated-preflight"],
        env=environment, stdout=log, stderr=subprocess.STDOUT)
    code = process.wait()
receipt = {
    "status": "complete" if code == 0 else "stopped", "exit_code": code,
    "pid": process.pid, "finished_at": time.time(), "automatic_model_retry": False}
(root / "exit.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
raise SystemExit(code)
