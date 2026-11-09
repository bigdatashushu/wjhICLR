"""Verify preserved evidence and frozen bytes, without model or simulator calls."""
from datetime import datetime, timezone
import json
import sys

import report

frozen = report.load(report.OUT / "frozen.json")
before = report.load(report.OUT / "verification/preserved-before.json")
missing, changed = [], []
for name, expected in before.items():
    path = report.ROOT / name
    if not path.is_file():
        missing.append(name)
    elif report.file_digest(path) != expected:
        changed.append(name)
snapshot_changes = [name for name, expected in frozen["snapshot_files"].items()
                    if report.file_digest(report.OUT / name) != expected]
continuation_config_changes = [
    str(path.parent.relative_to(report.OUT)) for path in (report.OUT / "continuations").glob("*/prepared.json")
    if report.file_digest(path.parent / "config.json") != report.load(path)["config_sha256"]]
sources = report.Sources(report.ROOT, report.ROOT / "official_direct/assets/manipulation_assets.json")
source_counts = sources.verify_all()
result = {
    "checked_at": datetime.now(timezone.utc).isoformat(), "preserved_files": len(before),
    "missing": missing, "changed": changed, "snapshot_changes": snapshot_changes,
    "continuation_config_changes": continuation_config_changes,
    "official_source_counts": source_counts, "model_requests": 0, "simulator_launches": 0,
}
suffix = "after" if len(sys.argv) > 1 and sys.argv[1] == "after" else "during"
(report.OUT / "verification" / f"preservation-{suffix}.json").write_bytes(report.canonical(result) + b"\n")
assert not missing and not changed and not snapshot_changes and not continuation_config_changes, "preserved_or_frozen_file_changed"
print(json.dumps(result))
