"""Stop only orphaned processes of the completed failed ALFRED startup."""
import json
import os
from pathlib import Path
import select
import signal
import ctypes
import platform

ROOT = b"/home/cvailab/wangjh/embodied-three-rgbd-20261107-r02/embodiedRunner/rgbd_three_envs/runs/three-rgbd-smoke-20261107-r02/workers/alfred-base-000"

def open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    if platform.machine() != "x86_64":
        raise RuntimeError("unsupported_pidfd_fallback_architecture")
    # This Python build omits the wrapper; the Linux kernel exposes syscall 434.
    library = ctypes.CDLL(None, use_errno=True)
    handle = library.syscall(434, pid, 0)
    if handle < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return handle


def matches(directory):
    try:
        command = (directory / "cmdline").read_bytes()
        cwd = str((directory / "cwd").resolve()).encode()
    except (OSError, PermissionError):
        return False
    return bool(command) and (ROOT in command or cwd.startswith(ROOT))


def main():
    selected, handles = [], {}
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit() and int(directory.name) != os.getpid() and matches(directory):
            pid = int(directory.name)
            try:
                handles[open_pidfd(pid)] = pid
                os.kill(pid, signal.SIGTERM)
                selected.append(pid)
            except ProcessLookupError:
                continue
    remaining = set(handles)
    # pidfds signal process exit; this is a bounded event wait, not a sleep loop.
    import time
    deadline = time.monotonic() + 5
    while remaining and time.monotonic() < deadline:
        readable, _, _ = select.select(list(remaining), [], [], max(0, deadline - time.monotonic()))
        remaining.difference_update(readable)
    forced = []
    for handle in remaining:
        pid = handles[handle]
        if matches(Path("/proc") / str(pid)):
            try:
                os.kill(pid, signal.SIGKILL)
                forced.append(pid)
            except ProcessLookupError:
                pass
    for handle in handles:
        os.close(handle)
    print(json.dumps({"status": "owned_failed_startup_cleanup", "terminated_pids": selected,
                      "force_terminated_owned_pids": forced, "other_runs_touched": False}))


if __name__ == "__main__":
    main()
