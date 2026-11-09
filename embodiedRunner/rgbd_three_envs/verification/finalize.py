"""Finalize on the experiment's filesystem completion event; no status polling."""
import json
import argparse
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rgbd_three_envs.transport import child_environment

PACKAGE = ROOT / "rgbd_three_envs"
RUN = PACKAGE / "runs/three-rgbd-smoke-20261107-r02"


def await_summary(run):
    summary = run / "summary.json"
    parent_fd = os.open(run.parent, os.O_RDONLY)
    parent_queue = select.kqueue()
    try:
        parent_queue.control([select.kevent(parent_fd, filter=select.KQ_FILTER_VNODE,
                                            flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                                            fflags=select.KQ_NOTE_WRITE)], 0)
        while not run.exists():
            parent_queue.control(None, 1)
    finally:
        parent_queue.close()
        os.close(parent_fd)
    fd = os.open(run, os.O_RDONLY)
    queue = select.kqueue()
    try:
        event = select.kevent(fd, filter=select.KQ_FILTER_VNODE,
                              flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                              fflags=select.KQ_NOTE_WRITE)
        queue.control([event], 0)
        while not summary.exists():
            # The kernel notifies directory writes. There is no timed retry,
            # sleep or repeated command-status query.
            queue.control(None, 1)
        summary_fd = os.open(summary, os.O_RDONLY)
        try:
            queue.control([select.kevent(summary_fd, filter=select.KQ_FILTER_VNODE,
                                        flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                                        fflags=select.KQ_NOTE_WRITE)], 0)
            while True:
                try:
                    return json.loads(summary.read_text())
                except json.JSONDecodeError:
                    # Creation may be observed before the single JSON write.
                    queue.control(None, 1)
        finally:
            os.close(summary_fd)
    finally:
        queue.close()
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN.name, choices=(
        RUN.name, "three-rgbd-alfred-recovery-20261107-r02"))
    args = parser.parse_args()
    summary = await_summary(PACKAGE / "runs" / args.run_id)
    if summary["status"] != "finished":
        print(json.dumps({"status": "experiment_stopped", "automatic_retry": False}))
        return 2
    verification = PACKAGE / "verification"
    python = "/home/cvailab/wangjh/embodiedRunner/venv-household/bin/python"
    argv = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=10", "cvailab@10.10.0.198",
            shlex.join([python, "-B", "-"])]
    result = subprocess.run(argv, input=(verification / "server_audit.py").read_bytes(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=child_environment(), timeout=60, check=False)
    if result.returncode != 0:
        print(json.dumps({"status": "server_audit_failed", "automatic_retry": False}))
        return 2
    receipt = json.loads(result.stdout)
    assert receipt["status"] == "verified"
    (verification / "server-final-audit.json").write_bytes(result.stdout)
    return subprocess.run([sys.executable, "-B", str(verification / "audit_three.py")],
                          cwd=ROOT, env=child_environment(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
