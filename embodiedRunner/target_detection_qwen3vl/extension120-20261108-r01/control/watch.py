"""Wait at most fifty seconds for completed-task or controller-state changes."""
import os
from pathlib import Path
import select
import sys
import time
import audit

duration = min(50, max(0, int(sys.argv[1]) if len(sys.argv) > 1 else 50))
initial = audit.status()
directories = [audit.OUT] + [
    path for _, run in audit.roots() if run.exists()
    for path in [run] + [p for p in run.iterdir() if p.is_dir()]]
fds = [os.open(path, os.O_RDONLY) for path in directories]
queue = select.kqueue()
queue.control([select.kevent(fd, filter=select.KQ_FILTER_VNODE,
                            flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                            fflags=select.KQ_NOTE_WRITE) for fd in fds], 0, 0)
deadline = time.monotonic() + duration
try:
    while time.monotonic() < deadline:
        queue.control(None, len(fds), max(0, deadline - time.monotonic()))
        current = audit.status()
        if current["completed"] != initial["completed"] or current["controller"] != initial["controller"]:
            break
    print(audit.canonical(audit.status()).decode(), flush=True)
finally:
    queue.close()
    for fd in fds:
        os.close(fd)
