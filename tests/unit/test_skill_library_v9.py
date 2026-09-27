"""Focused v9 Skill library source/derived/active-pointer checks."""

import json
from pathlib import Path

import pytest

from skill3d.schemas import SkillSpec
from skill3d.skills.registry import load_active_skills
from skill3d.skills.source_compiler import (
    SkillSourceError,
    compile_directory,
    compile_skill_source,
    verify_bundle,
)


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "skill_library"
BUNDLE = LIBRARY / "imports" / "harness3d_s0_bundle.json"


def test_s0_sources_compile_to_eight_single_task_specs():
    compiled = compile_directory(LIBRARY / "skills", require_all_tasks=True)
    assert len(compiled) == 8
    assert {item.spec.skill_id for item in compiled} == {f"S{i:02d}" for i in range(1, 9)}
    assert {item.spec.applicable_question_types[0] for item in compiled} == {
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


def test_compile_directory_preserves_metric_evidence_signature_for_candidates(tmp_path):
    source = LIBRARY / "skills" / "estimate-object-distance" / "SKILL.md"
    text = source.read_text(encoding="utf-8")
    text = text.replace("  harness3d-validation: document-checked-runtime-pending\n",
                        "  harness3d-validation: document-checked-runtime-pending\n"
                        "  harness3d-gate-version: metric-evidence-gate-v6\n", 1)
    text = text.replace("  - degraded\n", "  - degraded\n  metric_scale:\n  - available\n", 1)
    target = tmp_path / "estimate-object-distance" / "SKILL.md"
    target.parent.mkdir()
    target.write_text(text, encoding="utf-8")
    spec, _metadata = compile_skill_source(target)
    assert spec.required_evidence_signature == {
        "image_2d": "degraded", "metric_scale": "available"
    }
    assert spec.requires_metric_evidence is True
    assert spec.applicable_gate_version == "metric-evidence-gate-v6"


def test_compile_directory_allows_multiple_skills_for_one_task(tmp_path):
    source = LIBRARY / "skills" / "count-scene-objects" / "SKILL.md"
    text = source.read_text(encoding="utf-8")
    alternate = (
        text.replace("name: count-scene-objects", "name: count-scene-objects-v2", 1)
        .replace("harness3d-skill-id: S01", "harness3d-skill-id: S09", 1)
        .replace("harness3d-version: 1.0.0", "harness3d-version: 1.1.0", 1)
    )
    original = tmp_path / "count-scene-objects" / "SKILL.md"
    alternate_path = tmp_path / "count-scene-objects-v2" / "SKILL.md"
    original.parent.mkdir()
    alternate_path.parent.mkdir()
    original.write_text(text, encoding="utf-8")
    alternate_path.write_text(alternate, encoding="utf-8")

    compiled = compile_directory(tmp_path)
    assert {item.spec.skill_id for item in compiled} == {"S01", "S09"}
    assert {item.spec.applicable_question_types[0] for item in compiled} == {
        "object_counting"
    }


def test_bundle_and_generated_digests_are_stable():
    bundle = verify_bundle(BUNDLE)
    assert bundle["file_count"] == 18
    index = json.loads((LIBRARY / "generated" / "index.json").read_text(encoding="utf-8"))
    for row in index["entries"]:
        source = LIBRARY / row["source_path"]
        generated = LIBRARY / row["generated_path"]
        assert source.is_file()
        assert generated.is_file()
        import hashlib
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row["source_sha256"]
        assert hashlib.sha256(generated.read_bytes()).hexdigest() == row["generated_sha256"]
        SkillSpec.model_validate(json.loads(generated.read_text(encoding="utf-8")))


def test_source_compiler_rejects_missing_fixed_section(tmp_path):
    source = tmp_path / "broken" / "SKILL.md"
    source.parent.mkdir()
    source.write_text(
        "---\nname: broken\ndescription: broken\nmetadata:\n"
        "  harness3d-skill-id: S99\n  harness3d-version: 1.0.0\n"
        "  harness3d-question-type: object_counting\n  harness3d-family: counting\n"
        "  harness3d-source-format: '1.0'\n  harness3d-validation: pending\n---\n"
        "## 目标\ntext\n", encoding="utf-8")
    with pytest.raises(SkillSourceError, match="缺少固定章节"):
        compile_skill_source(source)


def test_active_pointer_loads_s0_from_directory_and_pointer():
    directory_skills, directory_warnings, directory_ref = load_active_skills(LIBRARY / "snapshots")
    pointer_skills, pointer_warnings, pointer_ref = load_active_skills(
        LIBRARY / "snapshots" / "active_snapshot.json")
    assert directory_ref == pointer_ref == "S0-seed-20260925-v1"
    assert directory_warnings == pointer_warnings == []
    assert len(directory_skills) == len(pointer_skills) == 8
    assert {s.skill_id for s in directory_skills} == {f"S{i:02d}" for i in range(1, 9)}


def test_loader_skips_malformed_entry_and_never_scans_future_candidates(tmp_path):
    store = tmp_path / "snapshots"
    store.mkdir()
    (tmp_path / "candidates" / "future").mkdir(parents=True)
    (tmp_path / "candidates" / "future" / "not-active.json").write_text(
        "not a runtime spec", encoding="utf-8")
    valid = json.loads((LIBRARY / "generated" / "S01" / "1.0.0.json").read_text(encoding="utf-8"))
    snapshot = {
        "snapshot_id": "test-snapshot",
        "entries": {
            "valid": {
                "candidate_type": "skill",
                "spec_content": json.dumps(valid, ensure_ascii=False),
            },
            "broken": {
                "candidate_type": "skill",
                "spec_content": "{\"unknown\": true}",
            },
        },
    }
    (store / "snapshot_test-snapshot.json").write_text(
        json.dumps(snapshot), encoding="utf-8")
    (store / "active_snapshot.json").write_text(
        json.dumps({"snapshot_id": "test-snapshot"}), encoding="utf-8")
    skills, warnings, ref = load_active_skills(store)
    assert ref == "test-snapshot"
    assert [s.skill_id for s in skills] == ["S01"]
    assert len(warnings) == 1
    assert "broken" in warnings[0]


def test_missing_active_pointer_is_genesis(tmp_path):
    skills, warnings, ref = load_active_skills(tmp_path)
    assert skills == []
    assert ref == "genesis"
    assert "无 active snapshot" in warnings[0]
