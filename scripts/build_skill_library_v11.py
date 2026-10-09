#!/usr/bin/env python3
"""Build or check the complete eight-task v11 format-migration snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skill3d.skills.v11_library import (
    V11LibraryError,
    build_v11_snapshot,
    load_skill_source_v11,
    write_v11_snapshot,
)

DEFAULT_SNAPSHOT_ID = "S0-v11-format-migration"
REPAIR_SNAPSHOT_ID = "S0-v11-contract-repair"
SKILL_SOURCES = (
    ("S01", "object_counting", "count-scene-objects"),
    ("S02", "object_abs_distance", "estimate-object-distance"),
    ("S03", "object_rel_distance", "rank-object-distances"),
    ("S04", "object_size_estimation", "estimate-object-size"),
    ("S05", "room_size_estimation", "estimate-room-area"),
    ("S06", "object_rel_direction", "judge-relative-direction"),
    ("S07", "obj_appearance_order", "order-first-appearances"),
    ("S08", "route_planning", "infer-route-turns"),
)


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", default="skill_library")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--profile", choices=["format-migration", "contract-repair"],
                        default="format-migration")
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = Path(args.library_root)
    repair = args.profile == "contract-repair"
    snapshot_id = args.snapshot_id or (REPAIR_SNAPSHOT_ID if repair else DEFAULT_SNAPSHOT_ID)
    versions = {sid: "1.2.0" if repair and sid in ("S01", "S08") else "1.1.0"
                for sid, _, _ in SKILL_SOURCES}
    sources = [
        load_skill_source_v11(
            root,
            source_ref=f"versions/{skill_id}/{versions[skill_id]}/{name}/SKILL.md",
            skill_id=skill_id,
            version=versions[skill_id],
            question_type=question_type,
        )
        for skill_id, question_type, name in SKILL_SOURCES
    ]
    snapshot, manifest = build_v11_snapshot(
        sources,
        snapshot_id=snapshot_id,
        parent_snapshot_id=DEFAULT_SNAPSHOT_ID if repair else "S0-seed-20260925-v1",
        generation=0,
        created_by="contract_repair" if repair else "format_migration",
        parent_versions={"S01": "S01@1.1.0", "S08": "S08@1.1.0"} if repair else None,
    )
    snapshot_path = root / "snapshots" / f"snapshot_{snapshot_id}.json"
    manifest_path = root / snapshot["manifest_ref"]
    if args.check:
        if not snapshot_path.is_file() or snapshot_path.read_bytes() != _canonical(snapshot):
            raise V11LibraryError(f"v11 snapshot 已过期: {snapshot_path}")
        if not manifest_path.is_file() or manifest_path.read_bytes() != _canonical(manifest):
            raise V11LibraryError(f"v11 manifest 已过期: {manifest_path}")
    else:
        write_v11_snapshot(root, snapshot, manifest, activate=bool(args.activate))
    print(json.dumps({
        "snapshot_id": snapshot["snapshot_id"],
        "manifest_sha256": snapshot["manifest_hash"],
        "active_by_question_type": snapshot["active_by_question_type"],
        "skill_count": len(sources),
        "activated": bool(args.activate and not args.check),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
