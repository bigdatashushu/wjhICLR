"""Exact initial slots, immutable sources and isolated slot publication."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from skill3d.routing.skill_retriever import retrieve_ex
from skill3d.schemas import EvidenceProfile, SceneState, SkillCandidateV11
from skill3d.schemas.evidence import skill_evidence_state
from skill3d.skills.delivery import plan_delivery
from skill3d.skills.registry import load_active_skills
from skill3d.skills.v11_library import publish_v11_candidate, validate_v11_snapshot
from skill3d.skills.state_baseline import STATE_SNAPSHOT_ID

LIBRARY = Path(__file__).resolve().parents[2] / "skill_library"


@pytest.mark.parametrize(("geometry", "metric", "state"), [
    ("unavailable", "unavailable", "g0m0"),
    ("available", "unavailable", "g1m0"),
    ("available", "available", "g1m1"),
])
def test_exact_state_matches_one_of_three_methods(geometry, metric, state):
    skills = load_active_skills(LIBRARY / "snapshots")[0]
    scene = SceneState(artifact_ref="test", evidence_profile=EvidenceProfile(
        geometry_3d=geometry, metric_scale=metric))
    hits, record = retrieve_ex("question", scene, skills, question_type="object_counting")
    assert skill_evidence_state(scene.evidence_profile) == state
    assert [hit.skill_version for hit in hits] == [f"S01-{state}@1.0.0"]
    assert record.initial_evidence_state == state
    assert len([row for row in record.candidates if row.selected]) == 1


@pytest.mark.parametrize("profile", [
    None, {}, {"geometry_3d": "unavailable", "metric_scale": "available"},
    {"geometry_3d": "degraded", "metric_scale": "unavailable"},
])
def test_missing_or_illegal_states_are_never_g0m0(profile):
    with pytest.raises(ValueError):
        skill_evidence_state(profile)


def test_missing_slot_never_borrows_lower_state():
    skills = load_active_skills(LIBRARY / "snapshots")[0]
    only_visual = [s for s in skills if s.evidence_state == "g0m0"]
    scene = SceneState(artifact_ref="test", evidence_profile=EvidenceProfile(
        geometry_3d="available", metric_scale="available"))
    hits, record = retrieve_ex("q", scene, only_visual, question_type="object_counting")
    assert hits == [] and not record.retrieved_skill_versions
    unknown = SceneState(artifact_ref="test")
    hits, record = retrieve_ex("q", unknown, skills, question_type="object_counting")
    assert not hits
    assert any(row.reason_code == "evidence_state_unknown" for row in record.candidates)


def test_baseline_and_single_slot_publication_preserve_other_23(tmp_path):
    library = tmp_path / "library"
    shutil.copytree(LIBRARY, library)
    skills, _, _ = load_active_skills(library / "snapshots")
    before = json.loads((library / "snapshots" / f"snapshot_{STATE_SNAPSHOT_ID}.json").read_text())
    assert len(skills) == 24
    parent = next(s for s in skills if s.skill_id == "S03-g1m0")
    candidate = SkillCandidateV11(
        candidate_id="slot-test", parent_snapshot_id=before["snapshot_id"],
        parent_skill_version=f"{parent.skill_id}@{parent.version}",
        full_skill_spec=parent.model_copy(update={
            "version": "1.1.0",
            "skill_md": parent.skill_md + "\nRecheck the reference before submission.\n",
        }))
    after, receipt = publish_v11_candidate(library, candidate)
    for spec in skills:
        key = after["active_by_slot"][spec.question_type][spec.evidence_state]
        if spec.skill_id == parent.skill_id:
            assert key == candidate.candidate_skill_version
        else:
            assert key == f"{spec.skill_id}@{spec.version}"
            assert after["entries"][key]["content_sha256"] == spec.content_sha256
    assert receipt["evidence_state"] == "g1m0"
    assert len(validate_v11_snapshot(after, library_root=library)) == 24
    source = library / before["entries"][candidate.parent_skill_version]["source_ref"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == parent.content_sha256


def test_delivery_rejects_multiple_slots():
    skills = load_active_skills(LIBRARY / "snapshots")[0]
    with pytest.raises(ValueError, match="最多交付"):
        plan_delivery(skills[:2])


def test_nonmetric_question_with_scale_selects_g1m1_without_metric_tools():
    from skill3d.reconstruction_gate.evidence_profile import evaluate_metric_gate
    from skill3d.tools import REGISTRY

    gate = evaluate_metric_gate(
        scene_route="full_3d", main_gate_passed=True, fusion_status="success",
        scale_self_consistency=.02, valid_frame_ratio=1., metric_scale=1.,
        question_type="object_counting")
    scene = SceneState(
        artifact_ref="test", scene_route="full_3d", question_tool_scope="full_3d",
        evidence_profile=EvidenceProfile(geometry_3d="available", metric_scale="available"),
        question_type="object_counting", metric_evidence_gate_result=gate)
    skills = load_active_skills(LIBRARY / "snapshots")[0]
    hits, _ = retrieve_ex("q", scene, skills, question_type="object_counting")
    assert [h.skill_version for h in hits] == ["S01-g1m1@1.0.0"]
    assert not scene.metric_gate_passed
    assert not scene.metric_task_authorized("object_counting")
    names = REGISTRY.names_for_scope("full_3d", evidence_profile=scene.evidence_profile)
    assert not any(REGISTRY.is_metric_tool(name) for name in names)
