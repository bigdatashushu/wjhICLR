#!/usr/bin/env python3
"""Restart only the pinned operational service; retain model-child custody."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import fullqa_custody as c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--service", choices=["gateway", "supervisor"], required=True)
    args = parser.parse_args()
    os.umask(0o077)
    home = args.home
    identity_path = home / (args.service + "_startup.json")
    program = home / ("fullqa_" + args.service + ".py")
    expected = c.read(home / "source_manifest.json")["files"][program.name]
    with (home / (args.service + "_keeper.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c.durable(home / (args.service + "_keeper_startup.json"),
                  {"at": time.time(), "process": c.process(os.getpid())})
        while True:
            c.require(c.digest(program.read_bytes()) == expected, "service_source_changed")
            status = c.read(home / "current.json") if (home / "current.json").exists() else {}
            if args.service == "supervisor" and status.get("status") == "complete":
                return
            identity = c.read(identity_path)["process"] if identity_path.exists() else None
            if identity is None or not c.live(identity):
                child = subprocess.Popen([sys.executable, "-B", str(program), "--config",
                    str(home / (args.service + "_config.json"))],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, start_new_session=True)
                time.sleep(1)
                c.durable(home / (args.service + "_keeper_restart.json"), {
                    "at": time.time(), "process": c.process(child.pid),
                    "reason": "owned_service_exited", "model_restarted": False})
                # Startup may fail closed on a retained native lock. A retry
                # launches operational code only, never a model attempt.
            time.sleep(10)


if __name__ == "__main__":
    main()
