#!/usr/bin/env python3
"""Start the two frozen thinking arms together and retain separate exits."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest = json.loads((root / "diagnostic.json").read_text())
    assert manifest["runtime"]["enable_thinking"] is True
    assert manifest["comparison"]["system_request_cap"] == 280
    assert manifest["comparison"]["direct_request_cap"] == 40
    assert root.is_dir() and str(root) == manifest["native_root"]
    assert not any((root / name).exists() for name in
                   ("pair_start.json", "solve_start.json", "direct_start.json", "exit.json"))
    spec = importlib.util.spec_from_file_location("_thinking40_controller", root / "controller.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.authenticate("http://10.10.0.198:27035")
    commands = {
        "system": [sys.executable, str(root / "controller.py"), "--run-root", str(root),
                   "--action", "solve", "--skip-validated-preflight"],
        "direct": [sys.executable, str(root / "controller.py"), "--run-root", str(root),
                   "--action", "direct", "--skip-validated-preflight"],
    }
    started = time.time()
    with (root / "pair_start.json").open("x") as stream:
        json.dump({"started_at": started, "pid": os.getpid(),
                   "diagnostic_identity": manifest["identity_sha256"],
                   "enable_thinking": True, "automatic_model_retry": False,
                   "command_sha256": {arm: hashlib.sha256(json.dumps(command).encode()).hexdigest()
                                      for arm, command in commands.items()}}, stream, indent=2)
        stream.write("\n")
    environment = dict(os.environ)
    environment.update(
        LD_PRELOAD="/home/wangy/wangjh/harness3d/precompute-20261022-r01/cpu_sqlite/libsqlite3.so.0",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    # Each controller reads its own exact frozen source; inherited credentials
    # are not needed other than the authenticated gateway token.
    for name in list(environment):
        if name != "HARNESS3D_PERCEPTION_TOKEN" and any(
                marker in name.upper() for marker in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")):
            environment.pop(name)
    processes, logs = {}, {}
    for arm, command in commands.items():
        logs[arm] = (root / (arm + ".log")).open("xb")
        processes[arm] = subprocess.Popen(command, env=environment, stdout=logs[arm],
                                          stderr=subprocess.STDOUT)
    write(root / "execution_note.json", {
        "diagnostic_identity": manifest["identity_sha256"],
        "pids": {arm: process.pid for arm, process in processes.items()},
        "started_at": started, "both_enable_thinking": True,
        "question_preflight": "skipped_per_user_instruction",
        "automatic_model_retry": False, "reference_answer_visible": False})
    exits = {}
    for arm, process in processes.items():
        exits[arm] = process.wait()
        logs[arm].close()
        write(root / (arm + "_exit.json"), {
            "exit_code": exits[arm], "finished_at": time.time(), "pid": process.pid})
    write(root / "exit.json", {
        "status": "complete" if all(code == 0 for code in exits.values()) else "stopped",
        "exits": exits, "started_at": started, "finished_at": time.time(),
        "automatic_model_retry": False})
    raise SystemExit(0 if all(code == 0 for code in exits.values()) else 1)


if __name__ == "__main__":
    main()
