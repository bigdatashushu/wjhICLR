"""Public task interfaces and parameter-dependent distance authorization."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.online.recovery import cascade_invalidate, downgrade_profile
from skill3d.reconstruction_gate.evidence_profile import evaluate_metric_gate
from skill3d.schemas import AnswerPayload, EvidenceProfile, ObjectRecord, SceneState
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.tools import REGISTRY, SceneHandle
from skill3d.tools.contract import (
    KNOWN_ARTIFACTS, ToolContractError, authorize_tool_call,
)
from skill3d.tools.task_policy import MODEL_TOOLS, TASK_TOOLS
from skill3d.verifier.derivation import replay_derivation
from skill3d.verifier.geometry_oracle import check_unit_consistent


def _context(task="object_abs_distance", state="g1m1"):
    geometry, scale = state != "g0m0", state == "g1m1"
    metric = scale and task in (
        "object_abs_distance", "object_size_estimation", "room_size_estimation")
    return {
        "scope": ("fallback_2d_only" if not geometry
                  else "metric_enabled" if metric else "full_3d"),
        "profile": EvidenceProfile(
            geometry_3d="available" if geometry else "unavailable",
            metric_scale="available" if scale else "unavailable"),
        "gate_passed": metric,
        "question_type": task,
        "available_artifacts": KNOWN_ARTIFACTS,
        "local_facts": {"world_frame": True},
    }


@pytest.mark.parametrize(("task", "sizes"), [
    ("object_counting", (4, 7, 7)),
    ("object_abs_distance", (4, 7, 7)),
    ("object_rel_distance", (4, 5, 5)),
    ("object_size_estimation", (4, 6, 7)),
    ("room_size_estimation", (4, 7, 8)),
    ("object_rel_direction", (4, 7, 7)),
    ("obj_appearance_order", (4, 4, 4)),
    ("route_planning", (4, 8, 8)),
])
def test_each_task_state_exposes_its_authorized_interfaces(task, sizes):
    for state, size in zip(("g0m0", "g1m0", "g1m1"), sizes):
        ctx = _context(task, state)
        visible = REGISTRY.names_for_scope(
            ctx["scope"], ctx["available_artifacts"],
            evidence_profile=ctx["profile"], gate_passed=ctx["gate_passed"],
            question_type=task, local_facts=ctx["local_facts"])
        assert len(visible) == size
        assert set(visible) <= TASK_TOOLS[task]
        assert all(authorize_tool_call(name, REGISTRY.spec(name), **ctx).allowed
                   for name in visible)
    assert len(MODEL_TOOLS) == 13
    assert set(REGISTRY.model_names()) == MODEL_TOOLS


def _handle(tmp_path, task="object_abs_distance"):
    gate = evaluate_metric_gate(
        scene_route="full_3d", main_gate_passed=True, fusion_status="success",
        scale_self_consistency=.02, valid_frame_ratio=1., metric_scale=2.,
        question_type=task)
    objects = []
    rng = np.random.default_rng(0)
    for oid, x in (("a", 0.), ("b", 3.)):
        pts = rng.normal(0., .08, (400, 3)) + [x, 0., 2.]
        path = tmp_path / f"{oid}.npy"
        np.save(path, pts)
        objects.append(ObjectRecord(
            obj_id=oid, category_name=oid, centroid_world=[x, 0., 2.],
            pointcloud_world=str(path)))
    ctx = _context(task)
    scene = SceneState(
        artifact_ref="test", scene_route="full_3d", question_type=task,
        question_tool_scope=ctx["scope"], available_artifacts=set(KNOWN_ARTIFACTS),
        evidence_profile=ctx["profile"], metric_evidence_gate_result=gate)
    return SceneHandle(scene, objects=objects, c2w_list=np.eye(4)[None],
                       intrinsics=np.eye(3)[None], metric_scale=2.)


def test_distance_unit_controls_receipts_and_metric_conversion(tmp_path):
    handle = _handle(tmp_path)
    results = []
    for unit in ("normalized", "m"):
        result = REGISTRY.call_tool(
            "object_surface_distance", {"obj_a": "a", "obj_b": "b", "unit": unit}, handle)
        assert result.status == "ok"
        assert result.resolved_object_ids == ["a", "b"]
        assert result.object_dependencies == ["a", "b"]
        assert ("metric_scale" in result.authorization["dependency_refs"]) == (unit == "m")
        assert ("scale" in result.authorization["dependency_refs"]) == (unit == "m")
        assert result.payload["unit"] == unit
        results.append(result)
    assert results[1].payload["distance"] == pytest.approx(results[0].payload["distance"] * 2.)
    metric = results[1]
    payload = AnswerPayload(
        value=metric.payload["distance"] * 100, unit="cm", basis="tool_derived",
        used_result_ids=[metric.result_id], derivation={
            "op": "convert", "input_result_ids": [metric.result_id],
            "parameters": {"field": "distance", "from_unit": "m", "to_unit": "cm"}})
    assert replay_derivation(payload, [metric]).passed
    normalized = results[0]
    forged = payload.model_copy(update={
        "used_result_ids": [normalized.result_id],
        "derivation": {**payload.derivation, "input_result_ids": [normalized.result_id]}})
    assert not replay_derivation(forged, [normalized]).passed


def test_public_hidden_and_wrong_task_tools_cannot_execute(tmp_path):
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(tmp_path), frames=[])
    for code in (
        "euclidean_distance([0,0,0], [1,0,0])",
        "camera_object_distance('a')",
        "object_distance_m('a', 'b')",
        "surface_distance_between_objects('a','b')",
        "robust_distance('a','b')",
        "exists_in_scene('a')",
        "count_objects('a')",
    ):
        cell = kernel.run_cell(code)
        assert cell.error_code == "tool_contract"
        assert "task_denied" in kernel.authorization_receipts[-1]["reason_codes"]
        assert kernel.tool_results[-1].status == "failed"


def test_metric_downgrade_invalidates_only_metric_mode_and_allows_normalized(tmp_path):
    handle = _handle(tmp_path)
    rows = [REGISTRY.call_tool(
        "object_surface_distance", {"obj_a": "a", "obj_b": "b", "unit": unit}, handle)
        for unit in ("normalized", "m")]
    ledger = SimpleNamespace(tool_results=rows)
    invalid, _ = cascade_invalidate(ledger, "metric_scale", registry=REGISTRY)
    assert invalid == [rows[1].result_id]
    assert not ledger.tool_results[0].invalidated_by
    profile, _ = downgrade_profile(handle.evidence_profile, "metric_scale")
    handle._state = handle._state.model_copy(update={
        "evidence_profile": profile, "question_tool_scope": "full_3d"})
    with pytest.raises(ToolContractError):
        REGISTRY.call_tool("object_surface_distance",
                           {"obj_a": "a", "obj_b": "b", "unit": "m"}, handle)
    assert REGISTRY.call_tool(
        "object_surface_distance", {"obj_a": "a", "obj_b": "b"}, handle).status == "ok"


def test_docs_only_advertise_allowed_unit_modes():
    for task, expected in (
        ("object_abs_distance", "allowed units: normalized, m"),
        ("object_rel_distance", "- relative_distance_rank(")):
        ctx = _context(task)
        docs = REGISTRY.docs(
            ctx["scope"], ctx["available_artifacts"], evidence_profile=ctx["profile"],
            gate_passed=ctx["gate_passed"], question_type=task,
            local_facts=ctx["local_facts"])
        assert expected in docs
        assert "- object_distance_m(" not in docs
        assert "- surface_distance_between_objects(" not in docs
        if task == "object_rel_distance":
            assert "- object_surface_distance(" not in docs
            assert "- object_centroid(" not in docs
            assert "- reproject(" not in docs


def test_distance_unit_mismatch_is_rejected_by_verifier(tmp_path):
    from skill3d.schemas import ProgramExecutionTrace
    handle = _handle(tmp_path)
    result = REGISTRY.call_tool(
        "object_surface_distance", {"obj_a": "a", "obj_b": "b", "unit": "m"}, handle)
    bad = result.model_copy(update={
        "value": json.dumps({**result.payload, "unit": "normalized"})})
    trace = ProgramExecutionTrace(
        program_id="test", calls=[], results=[bad], stdout_tail="", steps=1,
        error_code=None, wallclock_s=0.)
    assert not check_unit_consistent(trace, handle)


@pytest.mark.parametrize("unit", ["cm", None, [], 1])
def test_unknown_unit_modes_are_refused_before_execution(tmp_path, unit):
    handle = _handle(tmp_path)
    with pytest.raises(ToolContractError, match="unit must"):
        REGISTRY.call_tool("object_surface_distance",
                           {"obj_a": "a", "obj_b": "b", "unit": unit}, handle)


def test_nonmetric_g1m1_keeps_geometry_but_rejects_metric_distance(tmp_path):
    handle = _handle(tmp_path, task="object_rel_distance")
    assert handle.evidence_profile.metric_scale == "available"
    for unit in ("normalized", "m"):
        with pytest.raises(ToolContractError, match="not exposed"):
            REGISTRY.call_tool("object_surface_distance",
                               {"obj_a": "a", "obj_b": "b", "unit": unit}, handle)


def test_tool_snapshot_changes_only_two_slot_sources_and_keeps_parents():
    from pathlib import Path
    from skill3d.skills.tool_baseline import build_tool_baseline

    library = Path(__file__).resolve().parents[2] / "skill_library"
    parent = json.loads(
        (library / "snapshots/snapshot_S0-v11-initial-state-skills.json").read_text())
    snapshot, _ = build_tool_baseline(library, write_sources=False)
    changed = []
    for task, states in parent["active_by_slot"].items():
        for state, key in states.items():
            new_key = snapshot["active_by_slot"][task][state]
            if new_key != key:
                changed.append((task, state))
                assert snapshot["entries"][key]["state"] == "historical"
                assert snapshot["entries"][new_key]["parent_version"] == key
                assert snapshot["entries"][key]["content_sha256"] == parent["entries"][key]["content_sha256"]
            else:
                assert snapshot["entries"][key]["content_sha256"] == parent["entries"][key]["content_sha256"]
    assert changed == [("object_abs_distance", "g1m0"), ("object_abs_distance", "g1m1")]
