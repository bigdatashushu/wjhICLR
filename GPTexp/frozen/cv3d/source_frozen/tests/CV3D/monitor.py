"""Independent read-only monitor. Never starts model calls or retries failed work."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main(root, status_source):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cv3d_readonly_status", status_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = Path(root)
    target = root / "ops"
    target.mkdir(exist_ok=True)
    process = Path("/proc") / str(os.getpid())
    startup = {
        "pid": os.getpid(), "at": datetime.now().astimezone().isoformat(),
        "start_ticks": (process / "stat").read_text().split(") ", 1)[1].split()[19],
        "command_sha256": hashlib.sha256((process / "cmdline").read_bytes()).hexdigest(),
        "read_only": True, "model_requests": 0,
    }
    (target / "startup.json").write_text(json.dumps(startup))
    while True:
        value = module.status(root)
        temporary = target / ".progress.json"
        temporary.write_text(json.dumps(value, ensure_ascii=False))
        temporary.replace(target / "progress.json")
        if "failure.json" in value:
            (target / "diagnosis.json").write_text(json.dumps({
                "status": "pipeline_stopped", "at": value["at"],
                "failure": value["failure.json"], "automatic_retry": False,
            }))
            return 2
        if "exit.json" in value or not value.get("process", {}).get("alive", False):
            (target / "observed_exit.json").write_text(json.dumps({
                "status": "complete" if value.get("completion.json", {}).get("status") == "complete"
                else "exited_without_completion",
                "at": value["at"], "model_requests": 0,
            }))
            return 0 if value.get("completion.json", {}).get("status") == "complete" else 2
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
