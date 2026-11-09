#!/usr/bin/env python3
"""Pause an owned serial pilot during outcome export, never during a request."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def identity(pid):
    proc = Path("/proc") / str(pid)
    return {
        "pid": pid,
        "start_ticks": proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.pid_file.read_text())
    expected = receipt["process_identity"]
    pid = receipt["pid"]
    if identity(pid) != expected:
        raise RuntimeError("owned pilot identity mismatch")
    known = set(args.run_dir.joinpath("outcomes").glob("*.json"))
    while Path("/proc", str(pid)).exists():
        new = set(args.run_dir.joinpath("outcomes").glob("*.json")) - known
        if new:
            if identity(pid) != expected:
                raise RuntimeError("owned pilot identity changed")
            os.kill(pid, signal.SIGSTOP)
            pending = [str(p.relative_to(args.run_dir))
                       for kind in ("solver_requests", "parser_requests")
                       for p in args.run_dir.joinpath(kind).glob("*/*.json")
                       if json.loads(p.read_text()).get("status") == "started"]
            if pending:
                os.kill(pid, signal.SIGCONT)
                known.update(new)
            else:
                stopped = {
                    "schema": "pilot-clean-boundary-pause/1",
                    "process_identity": expected,
                    "completed_outcomes": sorted(p.stem for p in new),
                    "pending_model_requests": pending,
                    "clean_episode_boundary": True, "at": time.time(),
                }
                path = args.run_dir / "concurrency_cutover_pause.json"
                path.write_text(json.dumps(stopped, indent=2) + "\n")
                os.kill(pid, signal.SIGTERM)
                os.kill(pid, signal.SIGCONT)
                print(json.dumps(stopped), flush=True)
                return
        time.sleep(0.1)


if __name__ == "__main__":
    main()
