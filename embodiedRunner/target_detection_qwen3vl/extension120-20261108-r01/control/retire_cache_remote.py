"""Retire only disposable copied Unity caches belonging to audited, exited tasks."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys

payload = json.load(sys.stdin)
namespace = Path(payload["work_root"])
assert str(namespace) == (
    "/home/cvailab/wangjh/embodied-target-detection-20261108-r05/embodiedRunner/"
    "target_detection_r05/runs/qwen3vl8b")
run_id = payload["run_id"]
assert re.fullmatch(r"qwen3vl-extension120-20261108-r01(?:-remaining\d{3})?", run_id)
worker_root = namespace / run_id / "workers"
retired, deferred = [], []
before = shutil.disk_usage(namespace).free
for name in payload["completed_tasks"]:
    assert re.fullmatch(r"(?:alfred|navigation)-base-\d{3}", name)
    task_root = worker_root / name
    assert worker_root in task_root.parents and task_root.resolve() == task_root
    active = False
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cwd = (proc / "cwd").resolve()
            raw = (proc / "cmdline").read_bytes().split(b"\0")
            if cwd == task_root or task_root in cwd.parents or str(task_root).encode() in raw:
                active = True
                break
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    if active:
        deferred.append(name)
        continue
    cache = task_root / ("thor21" if name.startswith("alfred-") else "thor5")
    if cache.exists():
        assert cache.is_dir() and not cache.is_symlink() and task_root in cache.resolve().parents
        shutil.rmtree(cache)
        retired.append(str(cache))
print(json.dumps({
    "checked_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
    "retired_disposable_cache_paths": retired, "deferred_active_tasks": deferred,
    "disk_free_change_bytes": shutil.disk_usage(namespace).free - before,
    "official_logs_and_local_trajectory_preserved": True,
    "original_caches_untouched": True, "model_requests": 0, "simulator_launches": 0,
}))
