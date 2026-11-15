#!/usr/bin/env python3
"""Launch both frozen paired arms after their zero-call input preflight."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest = json.loads((root / "diagnostic.json").read_text())
    assert str(root) == manifest["native_root"]
    system_thinking = manifest["runtime"]["enable_thinking"]
    direct_thinking = manifest["runtime"].get("direct_enable_thinking", system_thinking)
    assert isinstance(system_thinking, bool) and isinstance(direct_thinking, bool)
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == manifest["launcher_sha256"]
    assert not any((root / name).exists() for name in
                   ("pair_start.json", "solve_start.json", "direct_start.json", "exit.json"))
    per_episode = manifest["runtime"]["question_preflight"] == "per_episode_before_first_call"
    if not per_episode:
        preflight = json.loads((root / "system/preflight_summary.json").read_text())
        assert preflight["status"] == "complete" and preflight["model_requests"] == 0
        assert preflight["completed_preflights"] == manifest["n_questions"]
    commands = {arm: [sys.executable, str(root / "controller.py"), "--run-root", str(root),
                      "--action", action]
                for arm, action in (("system", "solve"), ("direct", "direct"))
                if arm in manifest["comparison"]["arms"]}
    assert commands
    if per_episode and "system" in commands:
        commands["system"].append("--skip-validated-preflight")
    started = time.time()
    write(root / "pair_start.json", {
        "started_at": started, "pid": os.getpid(),
        "diagnostic_identity": manifest["identity_sha256"],
        "command_sha256": {arm: hashlib.sha256(json.dumps(command).encode()).hexdigest()
                           for arm, command in commands.items()}})
    processes, logs = {}, {}
    for arm, command in commands.items():
        logs[arm] = (root / (arm + ".log")).open("xb")
        processes[arm] = subprocess.Popen(command, stdout=logs[arm], stderr=subprocess.STDOUT)
    write(root / "execution_note.json", {
        "diagnostic_identity": manifest["identity_sha256"],
        "pids": {arm: process.pid for arm, process in processes.items()},
        "started_at": started, "both_enable_thinking": system_thinking and direct_thinking,
        "system_enable_thinking": system_thinking, "direct_enable_thinking": direct_thinking,
        "question_preflight": "per_episode_before_first_call" if per_episode else "passed_zero_model_calls",
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
