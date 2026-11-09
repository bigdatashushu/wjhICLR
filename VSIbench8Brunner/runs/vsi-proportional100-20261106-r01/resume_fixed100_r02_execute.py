#!/usr/bin/env python3
"""Preserve exact request-file locations referenced by native replay receipts."""
import importlib.util
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "audited_fixed100_continuation", ROOT / "resume_fixed100_r02.py")
operation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = operation
spec.loader.exec_module(operation)
original_archive = operation.archive_paths


def archive_with_request_copies(paths, destination):
    records = original_archive(paths, destination)
    for row in records:
        relative = Path(row["original_file"])
        if (relative.parts[0] == "preparation" and "requests" in relative.parts
                and relative.suffix == ".json"):
            source = operation.ARM / row["file"]
            target = operation.ARM / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError("native replay request location already exists")
            shutil.copyfile(source, target)
            if operation.file_identity(target) != row["identity"]:
                raise ValueError("native reviewer request copy changed")
    return records


operation.archive_paths = archive_with_request_copies
if __name__ == "__main__":
    try:
        operation.execute()
    except Exception as exc:
        operation.REC.mkdir(exist_ok=True)
        operation.pilot.write(operation.REC / "execute_failed.json", {
            "error_type": type(exc).__name__, "automatic_retry": False,
            "reference_answer_visible": False, "at": operation.time.time(),
            "stack": [{"file": f.filename, "line": f.lineno, "function": f.name}
                      for f in operation.traceback.extract_tb(exc.__traceback__)]})
        print(operation.json.dumps({
            "status": "recovery_stopped", "phase": "execute",
            "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1)
