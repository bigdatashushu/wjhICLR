"""Two public capabilities, local candidate checks, and shared authorization."""

from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from skill3d.online.recovery import downgrade_profile
from skill3d.reconstruction_gate.evidence_profile import (
    M5EvidenceSummary, build_evidence_profile, evaluate_metric_gate,
)
from skill3d.schemas import EvidenceProfile, ObjectRecord, ProgramExecutionTrace, SceneState
from skill3d.schemas.evidence import CAPABILITIES
from skill3d.tools import REGISTRY, SceneHandle
from skill3d.tools.contract import ArtifactUnavailableError, available_artifacts_for


def _gate(**overrides):
    return evaluate_metric_gate(**{
        "scene_route": "full_3d", "main_gate_passed": True,
        "fusion_status": "success", "scale_self_consistency": .02,
        "valid_frame_ratio": 1., "metric_scale": 2.,
        "question_type": "object_abs_distance", **overrides,
    })


def _artifact(**overrides):
    return SimpleNamespace(**{
        "quality_status": "computed",
        "quality": SimpleNamespace(main_gate_passed=True, overall_quality=.9,
                                  diagnostic_warnings=["rotation_jump"]),
        "scale_fusion_status": "success", "metric_scale": 2.,
        "world_up": [0., 1., 0.], "handedness": "right",
        **overrides,
    })


def _handle(*, geometry=True, metric=True, up=True, objects=(), strict=False):
    route = "full_3d" if geometry else "fallback_2d_only"
    profile = EvidenceProfile(geometry_3d="available" if geometry else "unavailable",
                              metric_scale="available" if metric and geometry else "unavailable")
    state = SceneState(
        artifact_ref="fixture", artifact=_artifact(world_up=[0., 1., 0.] if up else None),
        scene_route=route, question_tool_scope=(
            "metric_enabled" if metric and geometry else route),
        question_type="object_abs_distance", evidence_profile=profile,
        metric_evidence_gate_result=_gate(main_gate_passed=geometry),
        available_artifacts=set(available_artifacts_for(
            route, metric_question=True, gate_passed=metric)),
    )
    return SceneHandle(state, objects=list(objects), c2w_list=np.eye(4)[None],
                       intrinsics=np.eye(3)[None], objects_materialized=True,
                       require_question_binding=strict)


def _visible(handle):
    return REGISTRY.names_for_scope(
        handle.question_tool_scope, handle.available_artifacts,
        evidence_profile=handle.evidence_profile, gate_passed=handle.metric_gate_passed,
        local_facts=handle.local_facts, question_type=handle.question_type)


def test_schema_has_only_two_binary_capabilities():
    assert CAPABILITIES == ("geometry_3d", "metric_scale")
    for extra in ("image_2d", "temporal", "object_grounding", "object_detection",
                  "world_frame", "track_consensus"):
        with pytest.raises(ValidationError):
            EvidenceProfile(geometry_3d="available", metric_scale="unavailable",
                            **{extra: "available"})
    with pytest.raises(ValidationError):
        EvidenceProfile(geometry_3d="degraded", metric_scale="unavailable")
    with pytest.raises(ValidationError):
        EvidenceProfile(geometry_3d="unavailable", metric_scale="available")


def test_m5_diagnostics_and_soft_warnings_do_not_split_public_states():
    profile = build_evidence_profile(
        artifact=_artifact(), scene_route="full_3d",
        question_type="object_rel_distance",
        gate=_gate(question_type="object_rel_distance"),
        m5=M5EvidenceSummary(m5_ran=True, detection_fault=True),
    )
    assert profile.as_signature() == {"geometry_3d": "available", "metric_scale": "available"}
    assert profile.producer_subvalues["geometry_3d"]["warnings"] == ["rotation_jump"]
    assert profile.producer_subvalues["candidate_diagnostics"]["detection_fault"]


def test_scale_self_consistency_failure_preserves_geometry():
    profile = build_evidence_profile(
        artifact=_artifact(), scene_route="full_3d", question_type="object_abs_distance",
        gate=_gate(scale_self_consistency=.4))
    assert profile.as_signature() == {"geometry_3d": "available", "metric_scale": "unavailable"}
    assert profile.state_reasons["metric_scale"] == "quality_gate_not_passed"


def test_recovery_is_binary_and_only_geometry_cascades_to_scale():
    profile = EvidenceProfile(geometry_3d="available", metric_scale="available")
    scale, _ = downgrade_profile(profile, "metric_scale")
    assert scale.as_signature() == {"geometry_3d": "available", "metric_scale": "unavailable"}
    geometry, _ = downgrade_profile(profile, "geometry_3d")
    assert set(geometry.as_signature().values()) == {"unavailable"}
    for fact in ("world_frame", "object_detection"):
        same, changed = downgrade_profile(profile, fact)
        assert same == profile and changed == {}


def test_local_world_failure_hides_and_blocks_only_dependent_tools():
    handle = _handle(up=False)
    visible = _visible(handle)
    assert "relative_direction_of" not in visible
    assert "plane_fit_room_size" not in visible
    assert {"object_centroid", "object_surface_distance", "list_objects"} <= set(visible)
    with pytest.raises(ArtifactUnavailableError, match="world_frame"):
        REGISTRY.call_tool("relative_direction_of", {
            "observer_id": "a", "facing_at_id": "b", "target_id": "c"}, handle,
            model_call=False)


def test_2d_candidates_survive_geometry_and_object_point_failures(tmp_path):
    mask = np.ones((2, 2, 2), dtype=np.uint8)
    ref = tmp_path / "mask.npy"
    np.save(ref, mask)
    obj = ObjectRecord(obj_id="chair_1", category_name="chair",
                       visible_frames=[0, 1], mask_per_frame=str(ref),
                       centroid_world=[], pointcloud_world="")
    handle = _handle(geometry=False, objects=[obj], strict=True)
    assert {"list_objects", "object_visible_frames", "inspect_frames"} <= set(_visible(handle))
    assert "object_centroid" not in _visible(handle)
    assert REGISTRY.call_tool("list_objects", {}, handle).status == "ok"
    frames = REGISTRY.call_tool("object_visible_frames", {"obj_id": "chair_1"}, handle)
    assert frames.value == "[0, 1]"
    assert frames.resolved_object_ids == ["chair_1"]


def test_single_object_does_not_require_whole_category(monkeypatch):
    obj = ObjectRecord(obj_id="chair_1", category_name="chair", centroid_world=[1., 2., 3.])
    handle = _handle(objects=[obj], strict=True)
    monkeypatch.setattr(handle, "binding_usable", lambda oid: oid == "chair_1")
    monkeypatch.setattr(handle, "category_binding_complete", lambda name: False)
    selected = REGISTRY.call_tool("object_centroid", {"obj_id": "chair_1"}, handle)
    assert selected.status == "ok"
    assert selected.object_dependencies == ["chair_1"]
    assert selected.resolved_object_ids == ["chair_1"]
    count = REGISTRY.call_tool("count_objects", {"category_name": "chair"}, handle,
                               model_call=False)
    assert count.status == "failed"
    rank = REGISTRY.call_tool("relative_distance_rank", {
        "reference": "chair_1", "candidate_categories": ["table", "sofa"]}, handle,
        model_call=False)
    assert rank.payload["status"] == "incomplete_candidates"
    assert rank.payload["closest_category"] is None


def _trace(result):
    return ProgramExecutionTrace(program_id="test", calls=[], results=[result],
                                 stdout_tail="", error_code=None, steps=1, wallclock_s=0.)


def test_verifier_checks_selected_object_without_certifying_reference(monkeypatch):
    from skill3d.verifier.geometry_oracle import geometry_verify

    objects = [
        ObjectRecord(obj_id="chair_good", category_name="chair", centroid_world=[1., 2., 3.]),
        ObjectRecord(obj_id="chair_bad", category_name="chair", centroid_world=[]),
    ]
    handle = _handle(objects=objects, strict=True)
    monkeypatch.setattr(handle, "binding_usable", lambda oid: oid == "chair_good")
    monkeypatch.setattr(handle, "category_binding_complete", lambda name: False)
    selected = REGISTRY.call_tool("object_centroid", {"obj_id": "chair_good"}, handle)
    verified = geometry_verify(_trace(selected), handle)
    assert verified.passed
    assert verified.check_status["binding_integrity"] == "pass"
    failed = REGISTRY.call_tool("object_centroid", {"obj_id": "chair_bad"}, handle)
    assert failed.status == "failed"
    assert failed.resolved_object_ids == ["chair_bad"]
    assert handle.evidence_profile.as_signature() == {
        "geometry_3d": "available", "metric_scale": "available"}


def test_reprojection_does_not_require_world_up():
    from skill3d.verifier.geometry_oracle import geometry_verify, WORLD_FRAME_TOOLS

    obj = ObjectRecord(obj_id="chair", category_name="chair",
                       centroid_world=[0., 0., 2.], bbox=[-1., -1., 1., 1., 1., 3.])
    handle = _handle(up=False, objects=[obj])
    projected = REGISTRY.call_tool("reproject", {"p3d": [0., 0., 2.], "frame_idx": 0}, handle)
    assert projected.status == "ok"
    verified = geometry_verify(_trace(projected), handle)
    assert verified.passed
    assert verified.check_status["world_frame"] == "not_applicable"
    assert set(WORLD_FRAME_TOOLS) == {
        name for name in REGISTRY.names()
        if "world_frame" in REGISTRY.spec(name).requires_local_facts}


def test_evidence_history_records_context_changes_and_final_state():
    from skill3d.adapters.episode_source import load_synthetic_items
    from skill3d.online.runner import OnlineRunConfig, run_episode

    item = load_synthetic_items("inner_validation", question_types=["object_counting"],
                                frame_size=(120, 160), limit=1)[0]
    result = run_episode(item.episode, item.pixels, OnlineRunConfig(mode="mock_light"),
                         geometry=item.geometry)
    trace = result.episode_trace
    assert trace.evidence_history
    assert trace.evidence_history[0]["trigger"] == "round:1"
    assert trace.evidence_history[-1]["evidence_states"] == trace.evidence_states
    assert trace.evidence_history[0]["candidates"]
    for row in trace.evidence_history:
        assert set(row["evidence_states"]) == {"geometry_3d", "metric_scale"}
