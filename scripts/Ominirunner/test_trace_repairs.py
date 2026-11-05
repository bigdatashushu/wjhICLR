"""Reproduce repaired15 failures without model requests or reference answers."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from Ominirunner import worker
from Ominirunner.audit import replay_contexts
from Ominirunner.common import Config
from Ominirunner.evidence import EvidenceError, validate as validate_evidence
from Ominirunner.protocol import crop_sheets, messages, overlay
from Ominirunner.review import apply_updates, validate as validate_reviews
from Ominirunner.test_contracts import (
    dimension, fenced, offline_episode, offline_only, point, reviewed, scene_data, task,
)
from skill3d.tools.contract import DomainValueError


@pytest.mark.parametrize("call,code,field,charged", [
    ('ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[60,20])',
     "omni_invalid_input", "b_xy", 1),
    ('ctx.tools.point_depth(view_index=0,x=10,y=10,intended_targets=[{"target":"chair","surface":"top"}])',
     "invalid_tool_arguments", "arguments", 0),
    ('ctx.tools.point_depth(view_index=0,x=10)', "invalid_tool_arguments", "arguments", 0),
    ('ctx.tools.point_depth(view_index=0,x=10,y=10,targets=[])', "omni_invalid_input", "targets", 1),
    ('ctx.tools.point_depth(view_index=1,x=10,y=10)', "omni_invalid_input", "view_index", 1),
    ('ctx.tools.point_depth(view_index=0,x=10,y=10,replaces="UNSEEN")',
     "omni_invalid_input", "replaces", 1),
])
def test_located_input_failure_recovers_in_new_cell_and_replays(
        tmp_path, monkeypatch, scene_data, call, code, field, charged):
    result = offline_episode(tmp_path, monkeypatch, scene_data, scene_data[0], [
        fenced(call + "\nReturnAnswer(999)"),
        fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10,targets=[{"target":"chair","surface":"center"}])\n'
               'YieldObservations([r["result_id"]])'),
        fenced('ReturnAnswer(3,evidence={"kind":"visual_estimate","basis":"RGB estimate"})')])
    first = result["rounds"][0]
    assert result["status"] == "answered" and result["answer"] == "3" and result["raw_answer"] == 3
    assert result["request_count"] == 3
    assert first["status"] == "repairable_tool_arguments" and first["cell"]["answer_untrusted"]
    assert first["cell"]["answer"] is None
    failure = first["tool_receipts"][0]
    error = first["feedback_errors"][0]
    assert failure["status"] == "failed" and error["failed_result_id"] == failure["result_id"]
    assert error["code"] == code and error["field"] == field
    assert result["tool_counts"]["calls"] == 1 + charged
    assert failure["result_id"] not in result["rounds"][1]["request"]["visible_result_ids"]
    assert not result["rounds"][1]["cell"]["answer_untrusted"]
    if "intended_targets" in call:
        assert error["unexpected_fields"] == ["intended_targets"] and "targets" in error["allowed_fields"]
    changed = deepcopy(result)
    changed["rounds"][0]["status"] = "run_error"
    with pytest.raises(ValueError, match="cell_failure_recovery_changed"):
        list(replay_contexts(scene_data[0], changed, *scene_data[1:4]))


def test_partial_success_survives_input_repair_but_failed_receipt_is_never_a_fact(
        tmp_path, monkeypatch, scene_data):
    result = offline_episode(tmp_path, monkeypatch, scene_data, scene_data[0], [
        fenced('ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
               'ctx.tools.point_depth(view_index=0,x=-1,y=10)\nReturnAnswer(999)'),
        fenced('ReturnAnswer(3,evidence={"kind":"visual_estimate","basis":"RGB estimate"})')])
    receipts = result["rounds"][0]["tool_receipts"]
    assert [r["status"] for r in receipts] == ["ok", "failed"]
    assert result["rounds"][1]["request"]["visible_result_ids"] == [receipts[0]["result_id"]]
    assert result["rounds"][0]["cell"]["answer_untrusted"] and result["raw_answer"] == 3


def test_catching_input_failure_does_not_authorize_same_cell_answer(tmp_path, monkeypatch, scene_data):
    result = offline_episode(tmp_path, monkeypatch, scene_data, scene_data[0], [
        fenced('try:\n    ctx.tools.point_depth(view_index=0,x=-1,y=10)\n'
               'except Exception:\n    ReturnAnswer(999)')])
    assert result["status"] == "run_error" and result["request_count"] == 1
    assert result["answer"] is None and result["rounds"][0]["cell"]["answer_untrusted"]


@pytest.mark.parametrize("failure", ["confidence", "unknown_domain", "missing_geometry"])
def test_quality_and_unlocated_contract_failures_still_stop(
        tmp_path, monkeypatch, scene_data, failure):
    cache = scene_data[3]
    if failure == "confidence":
        cache.geometries[0]["valid_mask"][8:13, 8:13] = False
    elif failure == "unknown_domain":
        original = worker.build_registry
        def unlocated(*args, **kwargs):
            registry, scene, counters = original(*args, **kwargs)
            def bad(handle, view_index, x, y):
                raise DomainValueError("point_depth", "unlocated implementation failure")
            registry.get("point_depth").fn = bad
            return registry, scene, counters
        monkeypatch.setattr(worker, "build_registry", unlocated)
    if failure == "missing_geometry":
        # Keep aligned depth renderable while making the point map invalid.
        cache.geometries[0]["points"] = cache.geometries[0]["points"][:10]
        call = 'ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[20,20])'
    else:
        call = 'ctx.tools.point_depth(view_index=0,x=10,y=10)'
    result = offline_episode(tmp_path, monkeypatch, scene_data, scene_data[0], [
        fenced(call + "\nReturnAnswer(999)")])
    assert result["status"] == "run_error" and result["answer"] is None
    assert result["request_count"] == 1 and result["rounds"][0]["cell"]["answer_untrusted"]
    assert result["rounds"][0]["feedback_errors"][0]["code"] == "tool_contract"


def test_usable_restatement_preserves_surface_but_downgrade_and_replacement_clear_trust(scene_data):
    row = point(scene_data[-1], "A")
    reviews = reviewed([row])
    surfaces = deepcopy(reviews["A"]["observed_surfaces"])
    apply_updates(reviews, {"A": {"result_id": "A", "decision": "usable"}}, [row])
    assert reviews["A"]["observed_surfaces"] == surfaces
    apply_updates(reviews, {"A": {"result_id": "A", "decision": "uncertain"}}, [row])
    apply_updates(reviews, {"A": {"result_id": "A", "decision": "usable"}}, [row])
    assert "observed_surfaces" not in reviews["A"]
    replacement = point(scene_data[-1], "B", xy=(11, 10))
    replacement["data"]["replaces_result_id"] = "A"
    apply_updates(reviews, reviewed([row]), [row, replacement])
    assert reviews["A"]["decision"] == "resample" and reviews["A"]["exclusion"] == "replaced"


def test_nested_wrong_decision_normalizes_and_visibility_error_identifies_endpoint(scene_data):
    row, _ = dimension(scene_data[-1], "W", "desk", "width")
    update = reviewed([row])["W"]
    update["decision"] = "wrong"
    update["observed_surfaces"][0]["decision"] = "wrong target"
    update["observed_surfaces"][0]["surface"] = "face_left"
    accepted, errors = validate_reviews([update], [row], ["W"])
    assert errors == []
    surface = accepted["W"]["observed_surfaces"][0]
    assert surface["decision"] == "resample" and surface["raw_decision"] == "wrong target"
    assert surface["surface"] == "left" and surface["raw_surface"] == "face_left"
    update = reviewed([row])["W"]
    update["observed_surfaces"][1]["visibility"] = "partial"
    accepted, errors = validate_reviews([update], [row], ["W"])
    assert accepted == {} and len(errors) == 1
    assert errors[0]["field"] == "observed_surfaces[1].visibility"
    assert errors[0]["actual"] == "partial" and errors[0]["measurement_result_id"] == "W"
    assert errors[0]["allowed"] == ["visible", "cropped", "occluded", "unclear"]


def test_component_surface_is_not_reinterpreted_as_whole_object_extremum(scene_data):
    row, _ = dimension(scene_data[-1], "H", "fireplace", "height")
    update = reviewed([row])["H"]
    update["observed_surfaces"][0]["surface"] = "mantel_top"
    accepted, errors = validate_reviews([update], [row], ["H"])
    assert not accepted and errors[0]["field"] == "observed_surfaces[0].surface"
    assert errors[0]["actual"] == "mantel_top" and errors[0]["expected"] == "top"


def test_anchor_feedback_preserves_real_endpoint_order(scene_data):
    row, binding = dimension(scene_data[-1], "L", "desk", "length")
    row["data"]["intended_targets"].reverse()
    contract = task("If the desk is 2m long, what is its length?", "float")
    evidence = {"kind": "calculation", "expression": "s*l/l", "bindings": {"l": binding},
                "scale": {"name": "s", "value": 2, "unit": "m", "anchor_binding": "l"}}
    with pytest.raises(EvidenceError) as caught:
        validate_evidence(2, evidence, [row], reviewed([row]), ["L"], contract)
    error = caught.value.record
    assert error["measurement_result_id"] == "L" and error["point_index"] == 0
    assert error["actual"] == "back" and error["expected"] == "front"
    binding["geometry"]["anchors"].reverse()
    assert validate_evidence(2, evidence, [row], reviewed([row]), ["L"], contract)["verified_arithmetic"]


def test_crop_budget_respects_dependency_order_before_marker_archive_order(scene_data):
    rgb = scene_data[1]
    rows = [point(scene_data[-1], f"P{n}", xy=(n+2, 10)) for n in range(20)]
    _, display = overlay(rgb, rows)
    priority = ["P19", "P18", *[f"P{n}" for n in range(18)]]
    sheets = crop_sheets(rgb, display["markers"], priority)
    markers = [m for _, page in sheets for m in page["markers"]]
    assert len(sheets) == 2 and len(markers) == 16
    assert [m["result_id"] for m in markers] == priority[:16]
    assert markers[0]["pixel_xy"] == rows[-1]["data"]["pixel_xy"]


def test_error_target_crops_remain_visible_when_already_usable(scene_data):
    row = point(scene_data[-1], "A")
    _, receipt = messages(*scene_data[:4], arm="skill", config=Config(), round_index=5,
                          observations=[row], reviews=reviewed([row]),
                          errors=[{"code": "measurement_surface_required", "measurement_result_id": "A"}])
    assert receipt["reviewed_measurement_ids"] == []
    assert receipt["context_packing"]["crop_priority_ids"] == ["A"]
    assert receipt["rgb_crop_sheets"][0]["markers"][0]["result_id"] == "A"


def test_final_json_restatement_keeps_full_reviews_and_dependency_crops(
        tmp_path, monkeypatch, scene_data):
    item = {**scene_data[0], "question": "If the desk is 2m wide, what is its width?", "task_type": "float"}
    sample = fenced('r=ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[20,20],'
                    'targets=[{"target":"desk","surface":"left_edge"},{"target":"desk","surface":"face_right"}])\n'
                    'YieldObservations([r["result_id"]])')
    def declaration(facts):
        key = facts[0]["result_id"]
        binding = {"result_id": key, "quantity": "width", "geometry": {
            "coverage": "complete", "axis": "object_horizontal",
            "anchors": ["left", "right"], "objects": ["desk", "desk"]}}
        return {"kind": "calculation", "expression": "s*w/w", "bindings": {"w": binding},
                "scale": {"name": "s", "value": 2, "unit": "m", "anchor_binding": "w"}}
    def wrong_arithmetic(facts):
        updates = reviewed(facts)
        return fenced(f'measurement_reviews = {list(updates.values())!r}\n'
                      f'ReturnAnswer(2.1,evidence={declaration(facts)!r})')
    def final(facts):
        assert facts[0]["observed_surfaces"][0]["surface"] == "left"
        return json.dumps({"answer": 2, "evidence": declaration(facts),
                           "measurement_reviews": [{"result_id": facts[0]["result_id"], "decision": "usable"}]})
    result = offline_episode(tmp_path, monkeypatch, scene_data, item, [
        sample, wrong_arithmetic, *[fenced('YieldObservations([])')]*3, final])
    assert result["status"] == "answered" and result["request_count"] == 6
    assert result["answer"] == "2" and result["answer_evidence"]["verified_arithmetic"]
    assert result["answer_evidence"]["validation_layers"]["independent_visual_geometry"] is False
    assert len(result["measurement_reviews"][0]["observed_surfaces"]) == 2
    receipt = result["rounds"][-1]["request"]
    assert receipt["reviewed_measurement_ids"] == []
    assert receipt["rgb_crop_sheets"][0]["markers"][0]["result_id"] == result["calculation_dependency_ids"][0]
