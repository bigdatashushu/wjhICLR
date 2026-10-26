"""Synthetic regressions for graph bindings and optional metric dependencies."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.infra.m5_identity import content_sha256
from skill3d.online.recovery import (
    cascade_invalidate, collect_validated, downgrade_profile, invalidate_objects,
)
from skill3d.online.submission import invalid_reference_issues
from skill3d.reconstruction_gate.evidence_profile import evaluate_metric_gate
from skill3d.schemas import (
    AnswerPayload, EvidenceProfile, ObjectRecord, ProgramExecutionTrace, SceneState,
    ToolResult,
)
from skill3d.segmentation.binding_validation import validate_binding
from skill3d.tools import REGISTRY, SceneHandle
from skill3d.tools.contract import ConfidenceGateError, authorize_tool_call
from skill3d.verifier.geometry_oracle import geometry_verify


def _object(oid, centroid=(1., 0., 2.), **overrides):
    return ObjectRecord(obj_id=oid, category_name="chair",
                        centroid_world=list(centroid), **overrides)


def _handle(objects=(), *, metric=False, strict=False, task="route_planning"):
    gate = evaluate_metric_gate(
        scene_route="full_3d", main_gate_passed=True, fusion_status="success",
        scale_self_consistency=.02, valid_frame_ratio=1., metric_scale=2.,
        question_type=task,
    )
    state = SceneState(
        artifact_ref="synthetic",
        artifact=SimpleNamespace(world_up=[0., 1., 0.], handedness="right"),
        scene_route="full_3d",
        question_tool_scope="metric_enabled" if metric else "full_3d",
        question_type=task,
        evidence_profile=EvidenceProfile(
            geometry_3d="available", metric_scale="available" if metric else "unavailable"),
        metric_evidence_gate_result=gate,
    )
    handle = SceneHandle(
        state, objects=list(objects), metric_scale=2.,
        c2w_list=np.eye(4)[None], intrinsics=np.eye(3)[None],
        objects_materialized=True, require_question_binding=strict,
    )
    x, z = np.meshgrid(np.linspace(-2., 2., 12), np.linspace(-2., 2., 12))
    floor = np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))
    handle.set_point_map(np.concatenate([floor, [[-2., 2., -2.], [2., 2., 2.]]]))
    return handle


def _trace(*results):
    return ProgramExecutionTrace(
        program_id="synthetic", calls=[], results=list(results),
        stdout_tail="", steps=len(results), wallclock_s=0., error_code=None,
    )


def _derived(rid, *inputs):
    return ToolResult(
        result_id=rid, source_tool="synthetic_derived", tool="synthetic_derived",
        status="ok", value='{"computed": 1}', payload={"computed": 1},
        result_dependencies=list(inputs),
    )


@pytest.mark.parametrize("tool", ["connectivity_graph", "list_objects"])
def test_deeply_nested_raw_json_is_rejected_without_a_host_exception(tool):
    raw = ('{"nodes":["start"],"start_node":"start","notes":'
           + "[" * 1200 + "0" + "]" * 1200 + "}")
    result = ToolResult(
        result_id="synthetic-nested", source_tool=tool, tool=tool, status="ok", value=raw,
    )
    verified = geometry_verify(_trace(result), _handle())
    assert not verified.passed
    assert "valid_json" in verified.violations


@pytest.mark.parametrize("kind", ["unverified", "conflict", "shape", "nan", "inf"])
def test_graph_rejects_any_invalid_node_without_filtering(kind):
    strict = kind == "unverified"
    bad = _object("bad")
    if kind == "conflict":
        bad = bad.model_copy(update={"binding_flags": [{
            "code": "semantic_label_mask_conflict", "audit_sha256": "synthetic-audit",
        }]})
    elif kind in ("shape", "nan", "inf"):
        bad = bad.model_copy(update={"centroid_world": {
            "shape": [], "nan": [float("nan"), 0., 0.],
            "inf": [float("inf"), 0., 0.],
        }[kind]})
    handle = _handle([bad], strict=strict)
    if not strict:
        handle = _handle([_object("good"), bad])
    profile = handle.evidence_profile.model_dump()
    result = REGISTRY.call_tool("connectivity_graph", {}, handle)
    assert result.status == "failed"
    assert result.payload is None
    assert result.error_code == ("confidence_gate" if kind in (
        "unverified", "conflict") else "domain_value")
    assert result.failure_cause.scope == "local_object"
    assert result.failure_cause.object_ids == ["bad"]
    assert result.failure_cause.shared_premises == []
    assert result.resolved_object_ids[-1] == "bad"
    assert result.object_dependencies == []
    assert handle.evidence_profile.model_dump() == profile
    if kind == "conflict":
        assert result.failure_cause.audit_ids == ["synthetic-audit"]
        assert REGISTRY.call_tool("object_centroid", {"obj_id": "good"}, handle).status == "ok"


def test_direct_handle_graph_also_checks_binding():
    handle = _handle([_object("unverified")], strict=True)
    with pytest.raises(ConfidenceGateError, match="verified M5 binding"):
        handle.connectivity_graph()


def test_graph_records_every_node_without_requiring_category_coverage(monkeypatch):
    handle = _handle([_object("a"), _object("b", (-1., 0., -1.))])
    monkeypatch.setattr(handle, "category_binding_complete", lambda name: False)
    result = REGISTRY.call_tool("connectivity_graph", {}, handle)
    assert result.status == "ok"
    assert result.payload["nodes"] == ["start", "a", "b"]
    assert result.resolved_object_ids == ["a", "b"]
    assert result.object_dependencies == ["a", "b"]
    assert "metric_scale" not in result.authorization["dependency_refs"]
    verified = geometry_verify(_trace(result), handle)
    assert verified.passed
    assert verified.check_status["binding_integrity"] == "pass"


def test_strict_graph_uses_existing_m5_binding_and_rechecks_changed_centroid(tmp_path):
    mask_path, points_path = tmp_path / "mask.npy", tmp_path / "points.npy"
    np.save(mask_path, np.ones((2, 1, 1), dtype=np.uint8))
    np.save(points_path, np.array([[0., 0., 0.], [2., 0., 4.]]))
    observation = {"observed_object": "chair", "visibility": "clear",
                   "mask_purity": "clean", "mask_completeness": "whole"}
    identity = {"candidate_id": "a", "model": {"name": "synthetic"}}
    obj = _object(
        "a", grounding_status="question_driven",
        mask_per_frame=str(mask_path), pointcloud_world=str(points_path),
        visible_frames=[0, 1], mask_support_ratio=1., bbox=[0., 0., 0., 2., 0., 4.],
        track_id="a", candidate_ids=["a"],
        question_roles=[{"category": "chair", "role": "waypoint", "source_text": "chair"}],
        binding_provenance=[{
            "candidate_id": "a", "category": "chair",
            "semantic_provider": "bailian", "semantic_policy": "visual-quality-and-relation",
            "semantic_relation": "match",
            "semantic_visual_request_sha256": content_sha256(identity),
            "semantic_relation_request_sha256": "b" * 64,
            "semantic_observation": observation,
            "semantic_observation_sha256": content_sha256(observation),
            "semantic_request_identity": identity,
            "semantic_model_identity_sha256": content_sha256(identity["model"]),
        }],
    )
    validation = validate_binding(obj, generation_sha256="c" * 64)
    assert validation.status == "validated", validation.errors
    obj = obj.model_copy(update={"binding_validation": validation})
    handle = _handle([obj], strict=True)
    assert handle.binding_usable("a")
    graph = REGISTRY.call_tool("connectivity_graph", {}, handle)
    assert graph.status == "ok"
    assert graph.object_dependencies == ["a"]
    assert geometry_verify(_trace(graph), handle).passed
    handle._objects["a"] = obj.model_copy(update={"centroid_world": [9., 0., 2.]})
    failed = REGISTRY.call_tool("connectivity_graph", {}, handle)
    assert failed.status == "failed"
    assert failed.failure_cause.scope == "local_object"
    assert failed.failure_cause.object_ids == ["a"]
    assert not geometry_verify(_trace(graph), handle).passed


def test_room_graph_with_materialized_empty_objects_remains_valid():
    handle = _handle([], strict=True, task="room_size_estimation")
    result = REGISTRY.call_tool("connectivity_graph", {}, handle)
    assert result.status == "ok"
    assert result.payload["nodes"] == ["start"]
    assert result.object_dependencies == result.resolved_object_ids == []
    assert geometry_verify(_trace(result), handle).passed


@pytest.mark.parametrize("dependencies", [[], ["good"], ["good", "bad"]])
def test_m11_rechecks_graph_nodes_independently_of_dependency_metadata(dependencies):
    handle = _handle([_object("good"), _object("bad", (-1., 0., -1.))])
    result = REGISTRY.call_tool("connectivity_graph", {}, handle)
    assert result.status == "ok"
    handle._objects["bad"] = handle._objects["bad"].model_copy(update={
        "binding_flags": [{"code": "semantic_label_mask_conflict"}],
    })
    forged = result.model_copy(update={
        "object_dependencies": dependencies, "resolved_object_ids": ["good"],
    })
    verified = geometry_verify(_trace(forged), handle)
    assert not verified.passed
    assert verified.check_status["binding_integrity"] == "fail"
    issue = next(issue for issue in verified.issues if issue.check == "binding_integrity")
    assert issue.failure_cause.scope == "local_object"
    assert issue.failure_cause.object_ids == ["bad"]
    assert issue.confirmed_shared_premise is None


def test_m11_rejects_graph_nodes_without_strict_m5_binding():
    handle = _handle([_object("unverified")])
    result = REGISTRY.call_tool("connectivity_graph", {}, handle)
    handle.require_question_binding = True
    forged = result.model_copy(update={"object_dependencies": []})
    verified = geometry_verify(_trace(forged), handle)
    assert not verified.passed
    assert "binding_integrity" in verified.violations


def test_m11_rejects_unknown_graph_node():
    handle = _handle([_object("good")])
    result = REGISTRY.call_tool("connectivity_graph", {}, handle)
    payload = {**result.payload, "nodes": ["start", "missing"]}
    forged = result.model_copy(update={"payload": payload, "value": json.dumps(payload)})
    assert not geometry_verify(_trace(forged), handle).passed


def test_m11_cannot_treat_an_object_as_the_synthetic_start():
    handle = _handle([_object("bad")])
    result = REGISTRY.call_tool("connectivity_graph", {}, handle)
    handle._objects["bad"] = handle._objects["bad"].model_copy(update={
        "binding_flags": [{"code": "semantic_label_mask_conflict"}],
    })
    payload = {**result.payload, "nodes": ["bad"], "start_node": "bad"}
    forged = result.model_copy(update={
        "payload": payload, "value": json.dumps(payload), "object_dependencies": [],
    })
    assert not geometry_verify(_trace(forged), handle).passed


def test_object_invalidation_revokes_graph_and_transitive_results_only():
    handle = _handle([_object("a"), _object("b")],
                     metric=True, task="room_size_estimation")
    graph = REGISTRY.call_tool("connectivity_graph", {}, handle)
    a = REGISTRY.call_tool("object_centroid", {"obj_id": "a"}, handle)
    b = REGISTRY.call_tool("object_centroid", {"obj_id": "b"}, handle)
    listing = REGISTRY.call_tool("list_objects", {}, handle)
    derived = _derived("derived", graph.result_id)
    grandchild = _derived("grandchild", derived.result_id)
    ledger = SimpleNamespace(tool_results=[graph, a, b, listing, derived, grandchild])
    before = copy.deepcopy(graph.payload)
    invalidated = invalidate_objects(ledger, ["a"], reason="synthetic-object-a")
    assert set(invalidated) == {
        graph.result_id, a.result_id, derived.result_id, grandchild.result_id,
    }
    assert {obs.result_id for obs in collect_validated(ledger)} == {
        b.result_id, listing.result_id,
    }
    assert ledger.tool_results[0].payload == before
    assert not ledger.tool_results[2].invalidated_by
    assert handle.evidence_profile.metric_scale == "available"
    assert invalidate_objects(ledger, ["a"], reason="again") == []


def test_optional_metric_centroid_has_runtime_not_static_scale_dependency():
    handle = _handle([_object("a")], metric=True, task="object_abs_distance")
    spec = REGISTRY.spec("object_centroid")
    assert "metric_scale" not in spec.requires_evidence
    assert "scale" not in spec.requires_artifacts
    preflight = authorize_tool_call(
        "object_centroid", spec, scope=handle.question_tool_scope,
        profile=handle.evidence_profile, available_artifacts=handle.available_artifacts,
        gate=handle.metric_evidence_gate_result, question_type=handle.question_type,
        local_facts=handle.local_facts, gate_passed=handle.metric_gate_passed,
    )
    assert preflight.allowed
    assert "metric_scale" not in preflight.dependency_refs
    result = REGISTRY.call_tool("object_centroid", {"obj_id": "a"}, handle)
    assert result.status == "ok"
    assert result.payload["centroid_normalized"] == [1., 0., 2.]
    assert result.payload["centroid_metric"] == [2., 0., 4.]
    assert {"metric_scale", "scale"} <= set(result.authorization["dependency_refs"])
    assert result.authorization["allowed"]
    assert result.authorization["metric_gate_result"]["status"] == "pass"
    assert result.authorization["metric_gate_result"]["gate_passed"] is True


def test_runtime_scale_consumption_is_authorized_and_denial_is_receipted(monkeypatch):
    handle = _handle([_object("a")], task="object_abs_distance")

    def consume_unauthorized_scale(handle, obj_id):
        handle._record_dependency("metric_scale")
        handle._record_dependency("scale")
        return {"centroid_normalized": [1., 0., 2.], "centroid_metric": [2., 0., 4.]}

    monkeypatch.setattr(REGISTRY.get("object_centroid"), "fn", consume_unauthorized_scale)
    result = REGISTRY.call_tool("object_centroid", {"obj_id": "a"}, handle)
    assert result.status == "failed"
    assert result.payload is None
    assert result.authorization["allowed"] is False
    assert {"metric_scale", "scale"} <= set(result.authorization["dependency_refs"])
    assert "tool_contract" in result.authorization["reason_codes"]


def test_scale_invalidation_revokes_metric_centroid_and_transitive_results():
    handle = _handle([_object("a")], metric=True, task="object_abs_distance")
    metric = REGISTRY.call_tool("object_centroid", {"obj_id": "a"}, handle)
    listing = REGISTRY.call_tool("list_objects", {}, handle)
    normalized_handle = _handle([_object("a")], task="object_abs_distance")
    normalized = REGISTRY.call_tool("object_centroid", {"obj_id": "a"}, normalized_handle)
    derived = _derived("derived", metric.result_id)
    grandchild = _derived("grandchild", derived.result_id)
    ledger = SimpleNamespace(tool_results=[
        metric, listing, normalized, grandchild, derived,
    ])
    old_payload, old_value = copy.deepcopy(metric.payload), metric.value
    invalidated, tools = cascade_invalidate(ledger, "metric_scale", registry=REGISTRY)
    assert set(invalidated) == {metric.result_id, derived.result_id, grandchild.result_id}
    assert set(tools) == {"object_centroid", "synthetic_derived"}
    assert {obs.result_id for obs in collect_validated(ledger)} == {
        listing.result_id, normalized.result_id,
    }
    assert ledger.tool_results[0].payload == old_payload
    assert ledger.tool_results[0].value == old_value
    assert ledger.tool_results[0].invalidated_by == ["metric_scale"]
    ledger.answer_slot = SimpleNamespace(payload=AnswerPayload(
        value=1., unit="m", basis="tool_derived",
        used_result_ids=[grandchild.result_id],
        derivation={"op": "field", "input_result_ids": [grandchild.result_id],
                    "parameters": {"field": "computed"}},
    ))
    ledger.unresolved_result_ids = set()
    issues = invalid_reference_issues(ledger)
    assert [(issue.check, issue.result_id) for issue in issues] == [
        ("invalidated_reference", grandchild.result_id),
    ]
    assert cascade_invalidate(ledger, "metric_scale", registry=REGISTRY) == ([], [])
    profile, _ = downgrade_profile(handle.evidence_profile, "metric_scale")
    handle._state = handle._state.model_copy(update={
        "evidence_profile": profile, "question_tool_scope": "full_3d",
    })
    fresh = REGISTRY.call_tool("object_centroid", {"obj_id": "a"}, handle)
    assert fresh.status == "ok"
    assert fresh.payload["centroid_metric"] is None
    assert fresh.payload["centroid_normalized"] == old_payload["centroid_normalized"]
    assert "metric_scale" not in fresh.authorization["dependency_refs"]
    assert "scale" not in fresh.authorization["dependency_refs"]
    assert geometry_verify(_trace(fresh), handle).passed


@pytest.mark.parametrize("scale", [None, 0., -1., float("nan"), float("inf")])
def test_centroid_without_metric_output_survives_scale_invalidation(scale):
    handle = _handle([_object("a")], metric=True, task="object_abs_distance")
    handle._metric_scale = scale
    result = REGISTRY.call_tool("object_centroid", {"obj_id": "a"}, handle)
    assert result.status == "ok"
    assert result.payload["centroid_metric"] is None
    assert "metric_scale" not in result.authorization["dependency_refs"]
    ledger = SimpleNamespace(tool_results=[result])
    assert cascade_invalidate(ledger, "metric_scale", registry=REGISTRY) == ([], [])
    assert collect_validated(ledger)[0].payload["centroid_normalized"] == [1., 0., 2.]


def test_consumed_scale_dependency_does_not_leak_to_next_call():
    handle = _handle([_object("a")], metric=True, task="object_abs_distance")
    metric = REGISTRY.call_tool("object_centroid", {"obj_id": "a"}, handle)
    assert "metric_scale" in metric.authorization["dependency_refs"]
    listing = REGISTRY.call_tool("list_objects", {}, handle)
    assert "metric_scale" not in listing.authorization["dependency_refs"]
    profile, _ = downgrade_profile(handle.evidence_profile, "metric_scale")
    handle._state = handle._state.model_copy(update={"evidence_profile": profile})
    normalized = REGISTRY.call_tool("object_centroid", {"obj_id": "a"}, handle)
    assert normalized.status == "ok"
    assert normalized.payload["centroid_metric"] is None
    assert "metric_scale" not in normalized.authorization["dependency_refs"]
