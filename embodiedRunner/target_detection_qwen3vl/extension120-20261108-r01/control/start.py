"""Launch exactly one detached extension controller and retain its exit status."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("/Users/bytedance/wangjh/wjhICLR/embodiedRunner")
OUT = ROOT / "target_detection_qwen3vl/extension120-20261108-r01"
PYTHON = str(ROOT.parent / ".venv/bin/python")


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n", encoding="utf-8")


def main():
    assert (OUT / "frozen.json").exists() and not (OUT / "run").exists()
    assert not (OUT / "controller-startup.json").exists()
    fd = os.open(OUT / "controller.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.close(fd)
    sys.path.insert(0, json.loads((OUT / "frozen.json").read_text())["source_root"])
    from target_detection_r05.transport import child_environment
    with (OUT / "controller.log").open("xb") as log:
        child = subprocess.Popen([PYTHON, "-B", str(Path(__file__).parent / "controller.py"), "control"],
                                 cwd=OUT, env=child_environment(), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, start_new_session=True, close_fds=True)
    receipt = {"pid": child.pid, "detached": True, "selected": 120,
               "created_at": datetime.now(timezone.utc).isoformat(), "automatic_retry": False}
    save(OUT / "controller.json", receipt)
    print(json.dumps({"status": "started", **receipt}), flush=True)
    code = child.wait()
    save(OUT / "controller-exit.json", {"exit_code": code, "at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "controller_exited", "exit_code": code}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
