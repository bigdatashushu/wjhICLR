"""React to coordinator completion, then perform the full local evidence audit."""
import json
import os
from pathlib import Path
import select
import sys

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
import audit


def main():
    path = OUT / "controller.json"
    fd = os.open(path, os.O_RDONLY)
    queue = select.kqueue()
    try:
        queue.control([select.kevent(
            fd, filter=select.KQ_FILTER_VNODE, flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
            fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_EXTEND)], 0)
        while True:
            try:
                receipt = json.loads(path.read_text())
            except json.JSONDecodeError:
                queue.control(None, 1)
                continue
            if receipt["status"] == "finished":
                audit.main()
                return 0
            if receipt["status"] == "stopped_on_operational_failure":
                print(json.dumps({"status": receipt["status"], "automatic_retry": False}))
                return 2
            queue.control(None, 1)
    finally:
        queue.close()
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
