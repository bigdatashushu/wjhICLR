"""Immutable interface migration from the 24-slot initial-state baseline."""

import copy
import json
from pathlib import Path

from skill3d.schemas import SkillSpecV11
from skill3d.skills.state_baseline import STATE_SNAPSHOT_ID
from skill3d.skills.v11_library import (
    V11_ENTRY_HISTORICAL, _manifest_for, _write_create_once,
    build_v11_snapshot, canonical_json_sha256, load_skill_source_v11, source_ref_for,
    validate_v11_snapshot,
)

TOOL_SNAPSHOT_ID = "S0-v11-task-tools"


def migrate_distance_source(text: str) -> str:
    """Change only retired distance references and result field names."""
    return (text.replace(
        "`object_distance_m(obj_a, obj_b)`",
        '`object_surface_distance(obj_a, obj_b, unit="m")`')
        .replace("`distance_m`", "`distance`")
        .replace(
            "surface_distance_between_objects can provide distance_normalized",
            'object_surface_distance(obj_a, obj_b, unit="normalized") can provide distance')
        .replace(
            "Do not call object_distance_m or fabricate scale.",
            'Do not request unit="m" or fabricate scale.'))


def build_tool_baseline(library_root: str | Path, *, write_sources: bool = True):
    root = Path(library_root)
    parent = json.loads((root / "snapshots" / f"snapshot_{STATE_SNAPSHOT_ID}.json").read_bytes())
    validate_v11_snapshot(parent, library_root=root)
    sources, parents, origins, history = [], {}, {}, {}
    for key, entry in sorted(parent["entries"].items()):
        source = load_skill_source_v11(
            root, source_ref=entry["source_ref"], skill_id=entry["skill_id"],
            version=entry["version"], question_type=entry["question_type"],
            evidence_state=entry["evidence_state"])
        text = migrate_distance_source(source.spec.skill_md)
        if text == source.spec.skill_md:
            sources.append(source)
            parents[entry["skill_id"]] = entry.get("parent_version")
            origins[entry["skill_id"]] = {
                k: entry[k] for k in ("origin_skill_version", "origin_content_sha256")
                if k in entry}
            continue
        spec = SkillSpecV11(
            **{**source.spec.model_dump(), "version": "1.0.1", "skill_md": text})
        ref = source_ref_for(spec)
        if write_sources:
            _write_create_once(root / ref, text.encode())
        elif not (root / ref).is_file() or (root / ref).read_bytes() != text.encode():
            raise ValueError(f"Tool migration source missing or changed: {ref}")
        sources.append(load_skill_source_v11(
            root, source_ref=ref, skill_id=spec.skill_id, version=spec.version,
            question_type=spec.question_type, evidence_state=spec.evidence_state))
        parents[spec.skill_id] = key
        origins[spec.skill_id] = {
            "origin_skill_version": key, "origin_content_sha256": source.content_sha256}
        history[key] = {**copy.deepcopy(entry), "state": V11_ENTRY_HISTORICAL}
    snapshot, _ = build_v11_snapshot(
        sources, snapshot_id=TOOL_SNAPSHOT_ID, parent_snapshot_id=STATE_SNAPSHOT_ID,
        generation=1, created_by="tool_interface_migration", parent_versions=parents,
        origins=origins, require_complete=True)
    snapshot["entries"].update(history)
    snapshot["skill_versions"] = sorted(snapshot["entries"])
    manifest = _manifest_for(snapshot)
    digest = canonical_json_sha256(manifest)
    snapshot["manifest_hash"] = digest
    manifest["manifest_sha256"] = digest
    validate_v11_snapshot(snapshot, library_root=root)
    return snapshot, manifest
