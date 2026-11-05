"""Origins and ordered count ratios tested with no model or label access."""
from copy import deepcopy

import pytest

from .evidence import EvidenceError, validate
from .provenance import program_inputs
from .test_contracts import (
    comparison, entity, fenced, offline_episode, offline_only, point, reviewed,
    scene_data, scope, task,
)


def rgb_origin(abandoned=()):
    return {"source": "rgb_only", "referenced_result_ids": [],
            "abandoned_result_ids": list(abandoned),
            "abandonment_reason": "Ignore the rejected geometry; estimate from original RGB." if abandoned else ""}


def test_noncount_scope_checks_identity_without_total_count_equation():
    contract = task("Estimate the visual proportion.", "float")
    evidence = {"kind": "visual_estimate", "basis": "RGB proportion",
                "object_scope": scope([entity("E1", [], "rgb_only"), entity("E2", [], "rgb_only")])}
    checked = validate(1.5, evidence, [], {}, [], contract)
    assert checked["scope_validation"]["counts"] == {"chair": 2}
    bad = deepcopy(evidence)
    bad["object_scope"]["entities"][1]["entity_id"] = "E1"
    with pytest.raises(EvidenceError, match="object_scope_identity_invalid"):
        validate(1.5, bad, [], {}, [], contract)
    with pytest.raises(EvidenceError, match="object_scope_count_mismatch"):
        validate(1.5, evidence, [], {}, [], task("How many chairs?", "int"))


def ratio_scope():
    entities = []
    for index in range(5):
        e = entity(f"E{index}", [], "rgb_only")
        e["category"] = "pillow" if index < 3 else "remote control"
        entities.append(e)
    return {"kind": "visual_estimate", "basis": "RGB shows three pillows and two remotes",
            "object_scope": scope(entities),
            "count_groups": {"numerator": ["E0", "E1", "E2"], "denominator": ["E3", "E4"]}}


def test_ratio_uses_ordered_groups_and_preserves_original_answer():
    contract = task("What is the ratio of pillows to remote controls?", "float")
    checked = validate(1.5, ratio_scope(), [], {}, [], contract)
    assert checked["scope_validation"]["ratio"]["calculated_value"] == 1.5
    with pytest.raises(EvidenceError, match="object_scope_ratio_mismatch"):
        validate(5, ratio_scope(), [], {}, [], contract)
    for change in ("duplicate", "zero", "unknown", "overlap"):
        bad = ratio_scope()
        groups = bad["count_groups"]
        if change == "duplicate":
            groups["numerator"].append("E0")
        elif change == "zero":
            groups["denominator"] = []
        elif change == "unknown":
            groups["denominator"] = ["missing"]
        else:
            groups["denominator"].append("E0")
        with pytest.raises(EvidenceError, match="object_scope_ratio_invalid"):
            validate(1.5, bad, [], {}, [], contract)


def test_ratio_keeps_rgb_category_shorthand_and_rejects_swapped_groups():
    contract = task("What is the ratio of pillows to remote controls?", "float")
    evidence = ratio_scope()
    for entity in evidence["object_scope"]["entities"][3:]:
        entity["category"] = "remote"
    assert validate(1.5, evidence, [], {}, [], contract)["scope_validation"]["ratio"]["calculated_value"] == 1.5
    evidence["count_groups"] = {"numerator": ["E3", "E4"], "denominator": ["E0", "E1", "E2"]}
    with pytest.raises(EvidenceError, match="object_scope_ratio_invalid"):
        validate(2 / 3, evidence, [], {}, [], contract)


def test_rejected_dependency_cannot_disappear_on_kind_change(scene_data):
    row = point(scene_data[-1], "A")
    evidence = {"kind": "visual_estimate", "basis": "RGB estimate"}
    with pytest.raises(EvidenceError, match="answer_provenance_required"):
        validate(1.5, evidence, [row], {}, ["A"], previous_dependency_ids=["A"])
    evidence["provenance"] = rgb_origin(["A"])
    # Equality with an old numeric answer alone is not evidence of derivation.
    checked = validate(1.5, evidence, [row], {}, ["A"], previous_dependency_ids=["A"])
    assert checked["origin_validation"]["source"] == "rgb_only"
    assert checked["origin_validation"]["origin_independently_verified"] is False
    with pytest.raises(EvidenceError, match="answer_provenance_conflict"):
        validate(1.5, evidence, [row], {}, ["A"], previous_dependency_ids=["A"],
                 program_dependency_ids=["A"])


def test_tool_estimate_cannot_use_blocked_replaced_or_wrong_quantity(scene_data):
    row = point(scene_data[-1], "A")
    evidence = {"kind": "tool_estimate", "basis": "Estimate informed by point depth",
                "expression": "z", "bindings": {"z": {"result_id": "A", "quantity": "camera_z"}},
                "provenance": {"source": "tool_informed", "referenced_result_ids": ["A"],
                               "abandoned_result_ids": [], "abandonment_reason": ""}}
    with pytest.raises(EvidenceError, match="calculation_measurement_not_usable"):
        validate(row["data"]["depth"], evidence, [row], {}, ["A"])
    replacement = point(scene_data[-1], "B")
    replacement["data"]["replaces_result_id"] = "A"
    with pytest.raises(EvidenceError, match="calculation_measurement_replaced"):
        validate(row["data"]["depth"], evidence, [row, replacement], reviewed([row]), ["A"])
    with pytest.raises(EvidenceError, match="task_quantity_mismatch"):
        validate(row["data"]["depth"], evidence, [row], reviewed([row]), ["A"],
                 task("How many meters do I have to walk?", "float"))


def test_straight_line_inputs_are_provable_but_literals_are_not(scene_data):
    row = point(scene_data[-1], "A")
    derived = fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
                     'z=r["data"]["depth"]\nanswer=z*2\nReturnAnswer(answer)')
    assert program_inputs(derived.removeprefix("```python\n").removesuffix("\n```"), [row], ["A"])["result_ids"] == ["A"]
    literal = fenced('answer=1.5*2\nReturnAnswer(answer)')
    assert program_inputs(literal.removeprefix("```python\n").removesuffix("\n```"), [row], ["A"])["result_ids"] == []
    dynamic = "def solve(ctx):\n    for r in []:\n        pass\n    ReturnAnswer(1)"
    assert program_inputs(dynamic, [row])["coverage"] == "partial_straight_line"


def test_worker_and_audit_require_abandonment_then_allow_independent_rgb(
        tmp_path, monkeypatch, scene_data):
    sample = fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10,'
                    'targets=[{"target":"chair","surface":"center"}])\nYieldObservations([r["result_id"]])')
    def rejected(facts):
        key = facts[0]["result_id"]
        return fenced(f'ReturnAnswer(1.5,evidence={{"kind":"calculation","expression":"z",'
                      f'"bindings":{{"z":{{"result_id":{key!r},"quantity":"camera_z"}}}}}})')
    def independent(facts):
        evidence = {"kind": "visual_estimate", "basis": "Independent RGB estimate",
                    "provenance": rgb_origin([facts[0]["result_id"]])}
        return fenced(f"ReturnAnswer(1.5,evidence={evidence!r})")
    result = offline_episode(tmp_path, monkeypatch, scene_data, scene_data[0], [
        sample, rejected, fenced('ReturnAnswer(1.5,evidence={"kind":"visual_estimate","basis":"RGB estimate"})'),
        independent])
    assert result["status"] == "answered" and result["request_count"] == 4
    assert result["rounds"][2]["answer_feedback_errors"][0]["code"] == "answer_provenance_required"
    assert result["answer_evidence"]["origin_validation"]["source"] == "rgb_only"


def test_worker_detects_explicit_current_tool_derivation_as_visual(
        tmp_path, monkeypatch, scene_data):
    def fallback(facts):
        evidence = {"kind": "visual_estimate", "basis": "Independent RGB estimate",
                    "provenance": rgb_origin([facts[0]["result_id"]])}
        return fenced(f"ReturnAnswer(1.5,evidence={evidence!r})")
    result = offline_episode(tmp_path, monkeypatch, scene_data, scene_data[0], [
        fenced('r=ctx.tools.point_depth(view_index=0,x=10,y=10)\n'
               'answer=r["data"]["depth"]*2\n'
               'ReturnAnswer(answer,evidence={"kind":"visual_estimate","basis":"RGB estimate"})'),
        fallback])
    assert result["status"] == "answered"
    first = result["rounds"][0]
    assert first["answer_feedback_errors"][0]["code"] == "answer_provenance_required"
    assert first["program_input_origin"]["result_ids"] == first["new_measurement_ids"]
