"""v11 lossless Skill source, snapshot, retrieval, and publication contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from skill3d.routing.skill_retriever import retrieve_ex
from skill3d.evolution.campaign import (
    CampaignBlocked,
    CampaignConfig,
    EvolutionCampaignRunner,
)
from skill3d.schemas import (
    SceneState,
    SkillCandidateV11,
    SkillSpec,
    SkillSpecV11,
    parse_skill_markdown,
)
from skill3d.skills.delivery import (
    plan_delivery,
    render_skill_entry,
    skill_content_sha256,
)
from skill3d.skills.promote_atomic import rollback
from skill3d.skills.registry import active_snapshot_provenance, load_active_skills
from skill3d.skills.v11_library import (
    V11LibraryError,
    build_v11_snapshot,
    canonical_json_sha256,
    load_skill_source_v11,
    publish_v11_candidate,
    validate_v11_snapshot,
    write_v11_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "skill_library"
V11_SKILLS = (
    ("S01", "object_counting", "count-scene-objects",
     {"list_objects", "count_objects", "detect_objects"}),
    ("S02", "object_abs_distance", "estimate-object-distance",
     {"object_distance_m"}),
    ("S03", "object_rel_distance", "rank-object-distances",
     {"relative_distance_rank"}),
    ("S04", "object_size_estimation", "estimate-object-size",
     {"object_3d_extent"}),
    ("S05", "room_size_estimation", "estimate-room-area",
     {"plane_fit_room_size"}),
    ("S06", "object_rel_direction", "judge-relative-direction",
     {"relative_direction_of"}),
    ("S07", "obj_appearance_order", "order-first-appearances",
     {"list_objects", "object_visible_frames"}),
    ("S08", "route_planning", "infer-route-turns",
     {"object_centroid", "relative_direction_of", "connectivity_graph"}),
)
S03_SOURCE = (
    LIBRARY / "versions" / "S03" / "1.1.0"
    / "rank-object-distances" / "SKILL.md"
)
S03_REF = "versions/S03/1.1.0/rank-object-distances/SKILL.md"


def _install_s03(library_root: Path):
    target = library_root / S03_REF
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(S03_SOURCE.read_bytes())
    return load_skill_source_v11(
        library_root,
        source_ref=S03_REF,
        skill_id="S03",
        version="1.1.0",
        question_type="object_rel_distance",
    )


def _activate_s03_baseline(library_root: Path):
    source = _install_s03(library_root)
    snapshot, manifest = build_v11_snapshot(
        [source],
        snapshot_id="S0-v11-s03-test",
        parent_snapshot_id="S0-seed-20260925-v1",
    )
    write_v11_snapshot(library_root, snapshot, manifest, activate=True)
    return source, snapshot


@pytest.mark.parametrize(
    ("skill_id", "question_type", "name", "expected_tools"),
    V11_SKILLS,
)
def test_all_v11_sources_are_compact_lossless_methods(
    skill_id,
    question_type,
    name,
    expected_tools,
):
    source_ref = f"versions/{skill_id}/1.1.0/{name}/SKILL.md"
    source = load_skill_source_v11(
        LIBRARY,
        source_ref=source_ref,
        skill_id=skill_id,
        version="1.1.0",
        question_type=question_type,
    )
    front, body = parse_skill_markdown(source.spec.skill_md)
    assert set(front) == {"name", "description"}
    assert front["name"] == name
    assert "## 方法" in body
    assert "## 分支与检查" in body
    assert "## 执行约束" not in body
    assert "## 代码示例" not in body
    assert "## 失败教训" not in body
    assert len(source.spec.skill_md) <= 8000
    assert render_skill_entry(source.spec) == source.spec.skill_md
    assert all(f"`{tool}" in source.spec.skill_md for tool in expected_tools)

    tool_sources = (
        (ROOT / "src/skill3d/tools/geometry_tools.py").read_text(encoding="utf-8")
        + (ROOT / "src/skill3d/tools/vision_tools.py").read_text(encoding="utf-8")
    )
    registered = set(re.findall(r'name="([a-z0-9_]+)"', tool_sources))
    assert expected_tools <= registered


def test_complete_v11_snapshot_has_one_active_skill_for_all_eight_tasks():
    snapshot_path = (
        LIBRARY / "snapshots" / "snapshot_S0-v11-format-migration.json")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    skills = validate_v11_snapshot(snapshot, library_root=LIBRARY)

    assert {skill.skill_id for skill in skills} == {
        skill_id for skill_id, _question_type, _name, _tools in V11_SKILLS}
    assert snapshot["active_by_question_type"] == {
        question_type: f"{skill_id}@1.1.0"
        for skill_id, question_type, _name, _tools in V11_SKILLS
    }
    assert len(snapshot["entries"]) == len(V11_SKILLS) == 8

    manifest = json.loads(
        (LIBRARY / snapshot["manifest_ref"]).read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256")
    assert canonical_json_sha256(manifest) == claimed == snapshot["manifest_hash"]


def test_v11_s03_source_is_the_runtime_body_without_recompilation(tmp_path):
    source = _install_s03(tmp_path)
    expected = S03_SOURCE.read_text(encoding="utf-8")

    assert source.spec.skill_md == expected
    assert source.spec.name == "rank-object-distances"
    assert source.spec.question_type == "object_rel_distance"
    assert "categories_without_detection" in source.spec.body
    assert render_skill_entry(source.spec) == expected
    assert skill_content_sha256(source.spec) == source.content_sha256
    assert source.content_sha256 == (
        "c85e8cadc39b2d447d06d8ea08cc62e7a62581cd7b3f089fce133e9c0da8239d")


@pytest.mark.parametrize("mutator", [
    lambda text: text.replace("\n", "\r\n"),
    lambda text: text + "\n",
])
def test_v11_spec_rejects_noncanonical_line_endings(mutator):
    text = mutator(S03_SOURCE.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="LF|换行"):
        SkillSpecV11(
            skill_id="S03",
            version="1.1.0",
            question_type="object_rel_distance",
            skill_md=text,
        )


def test_v11_snapshot_loads_only_referenced_hash_verified_source(tmp_path):
    source, snapshot = _activate_s03_baseline(tmp_path)

    skills, warnings, snapshot_id = load_active_skills(tmp_path / "snapshots")
    assert warnings == []
    assert snapshot_id == snapshot["snapshot_id"]
    assert len(skills) == 1
    assert isinstance(skills[0], SkillSpecV11)
    assert skills[0].skill_md == source.spec.skill_md
    assert active_snapshot_provenance(tmp_path / "snapshots") == (
        snapshot["snapshot_id"], snapshot["manifest_hash"])

    source_path = tmp_path / S03_REF
    source_path.write_text(
        source_path.read_text(encoding="utf-8").replace(
            "候选列表必须覆盖全部选项", "候选列表可以遗漏选项", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash 不一致"):
        load_active_skills(tmp_path / "snapshots")


def test_v11_source_ref_cannot_escape_library(tmp_path):
    with pytest.raises(V11LibraryError, match="source_ref"):
        load_skill_source_v11(
            tmp_path,
            source_ref="../outside/SKILL.md",
            skill_id="S03",
            version="1.1.0",
            question_type="object_rel_distance",
        )


def test_v11_loader_rejects_manifest_tampering(tmp_path):
    _source, snapshot = _activate_s03_baseline(tmp_path)
    manifest_path = tmp_path / snapshot["manifest_ref"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["active_by_question_type"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest 校验失败"):
        load_active_skills(tmp_path / "snapshots")


def test_v11_snapshot_rejects_multiple_active_skills_for_one_question_type(tmp_path):
    first = _install_s03(tmp_path)
    other_text = first.spec.skill_md.replace(
        "name: rank-object-distances", "name: compare-object-distances", 1)
    other_ref = "versions/S09/1.0.0/compare-object-distances/SKILL.md"
    other_path = tmp_path / other_ref
    other_path.parent.mkdir(parents=True)
    other_path.write_text(other_text, encoding="utf-8")
    second = load_skill_source_v11(
        tmp_path,
        source_ref=other_ref,
        skill_id="S09",
        version="1.0.0",
        question_type="object_rel_distance",
    )
    with pytest.raises(V11LibraryError, match="多个 active"):
        build_v11_snapshot([first, second], snapshot_id="duplicate")


def test_v11_retrieval_is_deterministic_question_type_lookup(tmp_path):
    source = _install_s03(tmp_path)
    scene = SceneState(
        artifact_ref="artifact",
        scene_route="fallback_2d_only",
        evidence_profile=None,
    )
    hits, record = retrieve_ex(
        "Which object is closest to the chair?",
        scene,
        [source.spec],
        question_type="object_rel_distance",
        snapshot_ref="S0-v11-s03-test",
    )

    assert [hit.skill_version for hit in hits] == ["S03@1.1.0"]
    assert record.retrieved_skill_versions == ["S03@1.1.0"]
    assert record.partition_policy == "v11_unique_active_by_question_type"
    assert record.config_version == "ret-v11-deterministic-1"
    assert record.lineage_selections == []
    assert record.candidates[0].score == 1.0
    assert record.candidates[0].matched_evidence_signature == {}


def test_v11_retrieval_rejects_mixed_runtime_formats(tmp_path):
    source = _install_s03(tmp_path)
    legacy = SkillSpec(
        skill_id="legacy",
        version="1.0.0",
        applicable_question_types=["object_rel_distance"],
        skill_family="relative_geometry",
    )
    scene = SceneState(artifact_ref="artifact", scene_route="fallback_2d_only")
    with pytest.raises(ValueError, match="不能混用"):
        retrieve_ex(
            "question",
            scene,
            [source.spec, legacy],
            question_type="object_rel_distance",
        )


def test_v11_publish_replaces_active_and_preserves_parent_for_rollback(tmp_path):
    parent_source, parent_snapshot = _activate_s03_baseline(tmp_path)
    parent_in_memory = load_active_skills(tmp_path / "snapshots")[0][0]
    candidate_md = parent_source.spec.skill_md.replace(
        "重点核查会改变前两名的错误绑定",
        "先核查会改变前两名的错误绑定",
        1,
    )
    candidate = SkillCandidateV11(
        candidate_id="cand-s03-120",
        parent_snapshot_id=parent_snapshot["snapshot_id"],
        parent_skill_version="S03@1.1.0",
        full_skill_spec=SkillSpecV11(
            skill_id="S03",
            version="1.2.0",
            question_type="object_rel_distance",
            skill_md=candidate_md,
        ),
        modification_reason="测试完整 Markdown 发布",
        source_run_ref="run:test",
    )

    promoted, receipt = publish_v11_candidate(tmp_path, candidate)
    assert promoted["entries"]["S03@1.1.0"]["state"] == "historical"
    assert promoted["entries"]["S03@1.2.0"]["state"] == "active"
    assert promoted["active_by_question_type"] == {
        "object_rel_distance": "S03@1.2.0"}
    assert receipt["snapshot_before"] == parent_snapshot["snapshot_id"]
    assert receipt["snapshot_after"] == promoted["snapshot_id"]
    assert receipt["source_diff"].startswith("--- S03@1.1.0")
    assert len(receipt["diff_sha256"]) == 64
    assert promoted["entries"]["S03@1.2.0"]["diff_sha256"] == receipt["diff_sha256"]
    assert (tmp_path / S03_REF).read_text(encoding="utf-8") == parent_source.spec.skill_md
    assert parent_in_memory.version == "1.1.0"

    current, warnings, current_id = load_active_skills(tmp_path / "snapshots")
    assert warnings == []
    assert current_id == promoted["snapshot_id"]
    assert [skill.version for skill in current] == ["1.2.0"]

    rollback(tmp_path / "snapshots", parent_snapshot["snapshot_id"])
    restored, warnings, restored_id = load_active_skills(tmp_path / "snapshots")
    assert warnings == []
    assert restored_id == parent_snapshot["snapshot_id"]
    assert [skill.version for skill in restored] == ["1.1.0"]


def test_v11_static_limit_rejects_complete_source(tmp_path):
    _source, snapshot = _activate_s03_baseline(tmp_path)
    with pytest.raises(V11LibraryError, match="超过服务限制"):
        validate_v11_snapshot(
            snapshot,
            library_root=tmp_path,
            method_context_max_chars=100,
        )


def test_v11_runtime_never_silently_drops_an_oversized_active_skill(tmp_path):
    source = _install_s03(tmp_path)
    with pytest.raises(ValueError, match="必须压缩候选"):
        plan_delivery([source.spec], max_chars=100)


def test_v11_candidate_id_must_be_path_safe(tmp_path):
    source = _install_s03(tmp_path)
    with pytest.raises(ValueError, match="candidate_id"):
        SkillCandidateV11(
            candidate_id="../escape",
            parent_snapshot_id="snapshot",
            parent_skill_version="S03@1.0.0",
            full_skill_spec=source.spec,
        )


def test_v11_active_pointer_contains_only_snapshot_identity(tmp_path):
    _source, snapshot = _activate_s03_baseline(tmp_path)
    pointer = json.loads(
        (tmp_path / "snapshots" / "active_snapshot.json").read_text(encoding="utf-8"))
    assert pointer == {"snapshot_id": snapshot["snapshot_id"]}


def test_legacy_campaign_stops_before_using_v11_active_snapshot(tmp_path):
    runner = EvolutionCampaignRunner(
        CampaignConfig(
            campaign_id="blocked-v11",
            library_root=str(LIBRARY),
            run_root=str(tmp_path),
        ),
        print_fn=lambda *_: None,
    )
    with pytest.raises(CampaignBlocked, match="不支持当前 SkillSpecV11"):
        runner.run()


def test_current_online_loader_rejects_legacy_snapshot_schema(tmp_path):
    store = tmp_path / "snapshots"
    store.mkdir()
    (store / "active_snapshot.json").write_text(
        json.dumps({"snapshot_id": "legacy"}), encoding="utf-8")
    (store / "snapshot_legacy.json").write_text(
        json.dumps({"snapshot_id": "legacy", "schema_version": "runtime-skill-snapshot/1.0"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="只接受 runtime-skill-snapshot/2.0"):
        load_active_skills(store)
