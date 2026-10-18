#!/usr/bin/env python3
"""Build/check the complete 24-slot initial-evidence Skill baseline without models."""

import argparse
import json
from pathlib import Path

from skill3d.skills.state_baseline import STATE_SNAPSHOT_ID, build_initial_state_baseline
from skill3d.skills.v11_library import V11LibraryError, write_v11_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", default="skill_library")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(args.library_root)
    snapshot, manifest = build_initial_state_baseline(root, write_sources=not args.check)
    if args.check:
        for path, value in (
            (root / "snapshots" / f"snapshot_{STATE_SNAPSHOT_ID}.json", snapshot),
            (root / snapshot["manifest_ref"], manifest),
        ):
            expected = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
                        + "\n").encode()
            if not path.is_file() or path.read_bytes() != expected:
                raise V11LibraryError(f"状态基线文件过期: {path}")
    else:
        write_v11_snapshot(root, snapshot, manifest, activate=args.activate)
    print(json.dumps({
        "snapshot_id": snapshot["snapshot_id"], "skill_count": 24,
        "manifest_sha256": snapshot["manifest_hash"],
        "active_by_slot": snapshot["active_by_slot"],
        "activated": bool(args.activate and not args.check),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
