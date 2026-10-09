"""v9 repository-local Skill library and S0 import contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from skill3d.schemas import SkillSpec
from skill3d.skills.registry import load_legacy_active_skills as load_active_skills
from skill3d.skills.source_compiler import (
    COMPILER_VERSION,
    compile_directory,
    verify_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "skill_library"


def test_s0_bundle_and_sources_cover_all_tasks() -> None:
    bundle = verify_bundle(LIBRARY / "imports" / "harness3d_s0_bundle.json")
    assert bundle["file_count"] == 18
    assert len(bundle["files"]) == 18

    compiled = compile_directory(LIBRARY / "skills", require_all_tasks=True)
    assert len(compiled) == 8
    assert {s.spec.skill_id for s in compiled} == {f"S{i:02d}" for i in range(1, 9)}
    assert {s.spec.applicable_question_types[0] for s in compiled} == {
        "object_counting",
        "object_abs_distance",
        "object_rel_distance",
        "object_size_estimation",
        "room_size_estimation",
        "object_rel_direction",
        "obj_appearance_order",
        "route_planning",
    }
    assert all(item.spec.required_evidence_signature == {"image_2d": "degraded"}
               for item in compiled)


def test_generated_specs_are_strict_runtime_specs() -> None:
    index = json.loads((LIBRARY / "generated" / "index.json").read_text(encoding="utf-8"))
    assert index["compiler_version"] == COMPILER_VERSION
    assert len(index["entries"]) == 8
    for row in index["entries"]:
        generated = LIBRARY / row["generated_path"]
        spec = SkillSpec.model_validate(json.loads(generated.read_text(encoding="utf-8")))
        assert f"{spec.skill_id}@{spec.version}" == row["skill_version"]
        assert row["generated_sha256"]

    raw = json.loads(
        (LIBRARY / "generated" / "S01" / "1.0.0.json").read_text(encoding="utf-8")
    )
    raw["unexpected_document_field"] = True
    with pytest.raises(ValidationError):
        SkillSpec.model_validate(raw)


def test_s0_active_snapshot_loads_from_directory_and_pointer() -> None:
    store = LIBRARY / "snapshots"
    expected_ref = json.loads(
        (store / "active_snapshot.json").read_text(encoding="utf-8")
    )["snapshot_id"]
    from_directory, warnings, ref = load_active_skills(store)
    from_pointer, pointer_warnings, pointer_ref = load_active_skills(
        store / "active_snapshot.json"
    )
    assert not warnings
    assert not pointer_warnings
    assert ref == pointer_ref == expected_ref
    assert len(from_directory) == len(from_pointer) == 8
    assert {s.skill_id for s in from_directory} == {f"S{i:02d}" for i in range(1, 9)}


def test_loader_skips_bad_entry_but_does_not_scan_future_candidates(tmp_path: Path) -> None:
    store = tmp_path / "snapshots"
    store.mkdir()
    valid = json.loads(
        (LIBRARY / "generated" / "S01" / "1.0.0.json").read_text(encoding="utf-8")
    )
    snapshot = {
        "snapshot_id": "test-snapshot",
        "entries": {
            "valid": {"candidate_type": "skill", "spec_content": json.dumps(valid)},
            "bad": {"candidate_type": "skill", "spec_content": "not-json"},
        },
    }
    (store / "snapshot_test-snapshot.json").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    (store / "active_snapshot.json").write_text(
        json.dumps({"snapshot_id": "test-snapshot"}), encoding="utf-8"
    )
    future = tmp_path / "candidates" / "future"
    future.mkdir(parents=True)
    (future / "future.json").write_text(json.dumps(valid), encoding="utf-8")

    skills, warnings, ref = load_active_skills(store)
    assert ref == "test-snapshot"
    assert [s.skill_id for s in skills] == ["S01"]
    assert len(warnings) == 1
    assert "bad" in warnings[0]


def test_missing_active_pointer_is_genesis(tmp_path: Path) -> None:
    skills, warnings, ref = load_active_skills(tmp_path)
    assert skills == []
    assert ref == "genesis"
    assert warnings and "无 active snapshot" in warnings[0]


def test_candidate_record_is_immutable_and_staged_outside_active(tmp_path: Path) -> None:
    from skill3d.schemas import SkillLibraryCandidate
    from skill3d.skills.library import SkillLibraryError, write_candidate_record

    candidate = SkillLibraryCandidate(
        candidate_id="cand-001",
        operation="revise",
        canonical_question_type="object_counting",
        source_split="learning",
        experience_relation="retrieved_and_delivered_used",
        hypothesis="先核对遮挡后重现，再决定是否补检。",
        patch="增加遮挡后重现检查。",
        expected_scope="object_counting",
        expected_effect="减少重复或漏计",
        known_risks="增加观察成本",
        created_at="2026-09-26T00:00:00Z",
    )
    path, digest = write_candidate_record(tmp_path, candidate)
    assert path == tmp_path / "candidates" / "future" / "cand-001.json"
    assert digest
    assert not (tmp_path / "snapshots" / "active_snapshot.json").exists()
    same_path, same_digest = write_candidate_record(tmp_path, candidate)
    assert same_path == path and same_digest == digest
    changed = candidate.model_copy(update={"patch": "different"})
    with pytest.raises(SkillLibraryError):
        write_candidate_record(tmp_path, changed)


def test_strict_promotion_rejects_opaque_skill_content(tmp_path: Path) -> None:
    from skill3d.schemas import CandidateRevision
    from skill3d.skills.promote_atomic import SnapshotValidationError, promote

    candidate = CandidateRevision(
        revision_id="rev-bad",
        root_candidate_id="root-bad",
        parent_version=None,
        candidate_type="skill",
        spec_content="not-a-runtime-spec",
        status="draft",
        induction_trace_refs=[],
        evidence_lineage_ref="",
        created_by="human",
        created_at="2026-09-26T00:00:00Z",
    )
    with pytest.raises(SnapshotValidationError):
        promote(tmp_path, candidate, strict_skill_specs=True)
    assert not (tmp_path / "active_snapshot.json").exists()
