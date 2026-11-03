#!/usr/bin/env python3
"""Verify that organized MMSI source files still match their recorded bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "source_manifest.json"


def verify(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if not path.is_file():
            failures.append({"path": row["path"], "reason": "missing"})
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != row["bytes"] or digest != row["sha256"]:
            failures.append(
                {
                    "path": row["path"],
                    "reason": "identity_mismatch",
                    "expected_bytes": row["bytes"],
                    "actual_bytes": len(data),
                    "expected_sha256": row["sha256"],
                    "actual_sha256": digest,
                }
            )
    return {
        "status": "passed" if not failures else "failed",
        "checked": len(manifest["files"]),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = verify(args.manifest)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
