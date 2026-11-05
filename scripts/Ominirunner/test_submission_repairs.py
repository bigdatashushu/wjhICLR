"""Tracefixed15 contract failures, using synthetic evidence and offline clients."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from Ominirunner.audit import replay_contexts
from Ominirunner.common import Config, digest, read, sha, validate_source
from Ominirunner.evidence import EvidenceError, validate as validate_evidence
from Ominirunner.protocol import messages, task_output
from Ominirunner.submission import MAX_CHARS, capture, delivery, size
from Ominirunner.test_contracts import (
    dimension, entity, fenced, offline_episode, offline_only, point, reviewed,
    scene_data, scope, task,
)
from skill3d.sandbox.ast_guard import ast_guard
from Ominirunner.native import parse_program
from Ominirunner.tools import ALLOWED


def texts(payload):
    return "\n".join(p["text"] for p in payload[0]["content"] if p["type"] == "text")


def submitted_template(contract, final):
    text = task_output(contract, final)
    if final:
        return json.loads(text.split("\n", 1)[1].split("\n", 1)[0])
    program = parse_program(re.findall(r"```python\n[\s\S]*?```", text)[0])
    assert ast_guard(program, allowed_tools=ALLOWED).ok
    nodes = ast.parse(program).body[0].body
    evidence = ast.literal_eval(nodes[0].value)
    answer = ast.literal_eval(nodes[-1].value.args[0])
    return {"answer": answer, "evidence": evidence}


@pytest.mark.parametrize("question,kind", [
    ("How many chairs are present?", "int"),
    ("How many holes are in the chair back?", "int"),
    ("What is the volume ratio of sofa and chair?", "float"),
    ("What is the volume of the sofa?", "float"),
    ("Which object is closer to the camera: chair or tv?", "str"),
    ("How tall is the chair?", "float"),
])
@pytest.mark.parametrize("final", [False, True])
def test_complete_task_visual_templates_pass_real_validator(question, kind, final):
    contract = task(question, kind)
    value = submitted_template(contract, final)
    checked = validate_evidence(value["answer"], value["evidence"], [], {}, [], contract)
    assert checked["kind"] == "visual_estimate"
    if contract["quantity"] == "count":
        assert checked["scope_validation"]["counts"] == {"requested entity": 1}
        assert "Zero count uses entities=[]" in task_output(contract, final)
    if contract["quantity"].startswith("volume"):
        assert all(isinstance(v, str) for v in checked["assumptions"].values())
    if final:
        assert value["measurement_reviews"] == [] and "def solve(ctx)" not in task_output(contract, final)


@pytest.mark.parametrize("assumptions,field", [
    (None, "evidence.assumptions"),
    ({}, "evidence.assumptions"),
    ({"object_scope": "one sofa"}, "evidence.assumptions"),
    ({"object_scope": "one sofa", "volume_model": "cuboid", "extra": "x"}, "evidence.assumptions"),
    ({"object_scope": {"include": "sofa"}, "volume_model": "cuboid"}, "evidence.assumptions.object_scope"),
    ({"object_scope": 1, "volume_model": "cuboid"}, "evidence.assumptions.object_scope"),
    ({"object_scope": "  ", "volume_model": "cuboid"}, "evidence.assumptions.object_scope"),
    ({"object_scope": "sofa", "volume_model": []}, "evidence.assumptions.volume_model"),
    ({"object_scope": "sofa", "volume_model": "x" * 201}, "evidence.assumptions.volume_model"),
])
def test_volume_124_feedback_locates_type_fields_and_lengths(assumptions, field):
    contract = task("What is the volume ratio of sofa and chair?", "float")
    with pytest.raises(EvidenceError) as caught:
        validate_evidence(12.5, {"kind": "visual_estimate", "basis": "RGB cuboids",
                                "assumptions": assumptions}, [], {}, [], contract)
    error = caught.value.record
    assert error["code"] == "object_scope_invalid" and error["field"] == field
    assert "both nonempty strings" in error["repair"]
    if field.endswith(".volume_model") and isinstance(assumptions["volume_model"], str):
        assert error["actual_chars"] == 201 and error["max_chars"] == 200


@pytest.mark.parametrize("basis", ["x" * 222, "x" * 242, "", None, ["RGB"]])
def test_basis_128_feedback_preserves_original_input_and_reports_length(basis):
    evidence = {"kind": "visual_estimate", "basis": basis}
    before = deepcopy(evidence)
    with pytest.raises(EvidenceError) as caught:
        validate_evidence("O2", evidence, [], {}, [],
                          task("Which object is closer to the camera: chair or tv?"))
    error = caught.value.record
    assert error["field"] == "evidence.basis" and error["max_chars"] == 200
    assert error["actual_chars"] == (len(basis) if isinstance(basis, str) else None)
    assert "<=120" in error["repair"] and "len(evidence['basis'])" in error["repair"]
    assert evidence == before


@pytest.mark.parametrize("which,field", [
    ("missing", "evidence.object_scope"),
    ("entities", "evidence.object_scope.entities"),
    ("basis", "evidence.object_scope.entities[0].basis"),
    ("included", "evidence.object_scope.entities[0].included"),
])
def test_count_145_44_feedback_locates_missing_or_bad_scope(which, field):
    evidence = {"kind": "visual_estimate", "basis": "RGB holes",
                "object_scope": scope([entity("hole1", [], "rgb_only")])}
    if which == "missing":
        evidence.pop("object_scope")
    elif which == "entities":
        evidence["object_scope"]["entities"] = {}
    elif which == "basis":
        evidence["object_scope"]["entities"][0]["basis"] = ""
    else:
        evidence["object_scope"]["entities"][0]["included"] = 1
    with pytest.raises(EvidenceError) as caught:
        validate_evidence(1, evidence, [], {}, [], task("How many holes are in the chair back?", "int"))
    assert caught.value.record["field"] == field


@pytest.mark.parametrize("quantity", ["distance", "object_height", 1, None, []])
def test_119_pair_quantity_enum_is_checked_before_task_role(scene_data, quantity):
    row, binding = dimension(scene_data[-1], "H", "chair", "height")
    binding["quantity"] = quantity
    with pytest.raises(EvidenceError) as caught:
        validate_evidence(1, {"kind": "calculation", "expression": "h", "bindings": {"h": binding}},
                          [row], reviewed([row]), ["H"], task("What is the height of the chair?", "float"))
    error = caught.value.record
    assert error["code"] == "distance_quantity_invalid" and error["actual"] == quantity
    assert error["field"] == "evidence.bindings.h.quantity"
    assert error["allowed_quantities"] == ["chord", "height", "width", "length", "diameter"]


def test_119_valid_quantity_with_wrong_task_role_is_a_separate_error(scene_data):
    row = point(scene_data[-1], "Z")
    evidence = {"kind": "calculation", "expression": "z",
                "bindings": {"z": {"result_id": "Z", "quantity": "camera_z"}}}
    with pytest.raises(EvidenceError) as caught:
        validate_evidence(1, evidence, [row], reviewed([row]), ["Z"],
                          task("What is the height of the chair?", "float"))
    error = caught.value.record
    assert error["code"] == "task_quantity_mismatch" and error["actual"] == "camera_z"
    assert error["allowed_quantities"] == ["height"]


@pytest.mark.parametrize("public,question,unit,value", [
    ("lamp is 0.6m tall", "how wide is the bookshelf?", "m", 0.6),
    ("red chair is 1 mile tall", "what is the height of the left chair in feet?", "ft", 5280.0),
])
def test_calculation_templates_register_scale_and_allow_cross_dimension_anchor(
        scene_data, public, question, unit, value):
    contract = task(f"If the {public}, {question}", "float")
    examples = re.findall(r"```python\n[\s\S]*?```", task_output(contract))
    assert len(examples) == 2
    program = parse_program(examples[1])
    assert ast_guard(program, allowed_tools=ALLOWED).ok
    declaration = ast.literal_eval(ast.parse(program).body[0].body[-2].value)
    assert declaration["expression"] == "s*t/a"
    assert declaration["scale"] == {"name": "s", "value": value, "unit": unit, "anchor_binding": "a"}
    rows = []
    for name, binding in declaration["bindings"].items():
        target = contract["operands"][1 if name == "a" else 0]["target"]
        binding["geometry"]["objects"] = [target, target]
        row, _ = dimension(scene_data[-1], binding["result_id"], target, binding["quantity"])
        row["data"]["distance"] = 1.2345678901234567 if name == "a" else 3.141592653589793
        rows.append(row)
    result = validate_evidence(value * 3.141592653589793 / 1.2345678901234567, declaration,
                              rows, reviewed(rows), [r["result_id"] for r in rows], contract)
    assert result["verified_arithmetic"] and result["scale"]["value"] == value


def test_252_scale_only_repair_preserves_full_precision_but_does_not_certify_geometry(scene_data):
    a, ab = dimension(scene_data[-1], "A", "red chair", "height")
    t, tb = dimension(scene_data[-1], "T", "left chair", "height")
    a["data"]["distance"] = 1.2345678901234567
    t["data"]["distance"] = a["data"]["distance"] * (1 + 8391.89770596981 / 5280)
    rows = [a, t]
    answer = 5280 * (t["data"]["distance"] / a["data"]["distance"] - 1)
    evidence = {"kind": "calculation", "expression": "5280*(t/a-1)",
                "bindings": {"t": tb, "a": ab}}
    contract = task("If the red chair is 1 mile tall, how many feet must it rise to match the left chair's height?", "float")
    with pytest.raises(EvidenceError, match="task_unit_mismatch"):
        validate_evidence(answer, evidence, rows, reviewed(rows), ["A", "T"], contract)
    evidence.update(expression="s*(t/a-1)", scale={"name": "s", "value": 5280.0,
                                                "unit": "ft", "anchor_binding": "a"})
    checked = validate_evidence(answer, evidence, rows, reviewed(rows), ["A", "T"], contract)
    assert checked["calculated_value"] == answer
    assert math.isclose(answer, 8391.89770596981, rel_tol=1e-12)
    assert checked["validation_layers"]["independent_visual_geometry"] is False


def test_388_changing_to_height_requires_geometry_and_a_registered_scale(scene_data):
    row, binding = dimension(scene_data[-1], "H", "lamp", "height")
    binding.pop("geometry")
    evidence = {"kind": "calculation", "expression": "1.4*h/h", "bindings": {"h": binding}}
    contract = task("If the lamp is 1.4m tall, what is its height?", "float")
    with pytest.raises(EvidenceError) as caught:
        validate_evidence(1.4, evidence, [row], reviewed([row]), ["H"], contract)
    assert caught.value.record["field"] == "evidence.bindings.h.geometry"
    binding["geometry"] = dimension(scene_data[-1], "H", "lamp", "height")[1]["geometry"]
    with pytest.raises(EvidenceError, match="task_unit_mismatch"):
        validate_evidence(1.4, evidence, [row], reviewed([row]), ["H"], contract)
    evidence.update(expression="s*h/h", scale={"name": "s", "value": 1.4, "unit": "m", "anchor_binding": "h"})
    assert validate_evidence(1.4, evidence, [row], reviewed([row]), ["H"], contract)["verified_arithmetic"]


def test_count_rejected_list_can_be_revised_from_12_to_18_without_any_tool_calls(
        tmp_path, monkeypatch, scene_data):
    item = {**scene_data[0], "question": "How many holes are in the chair back?", "task_type": "int"}
    def count_evidence(count):
        return {"kind": "visual_estimate", "basis": "RGB holes",
                "object_scope": scope([entity(f"hole{n}", [], "rgb_only") for n in range(count)])}
    # The old list is a rejected hypothesis and can be revised, even when no
    # candidate detections or measurement calls were used.
    result = offline_episode(tmp_path, monkeypatch, scene_data, item, [
        fenced('ReturnAnswer(12,evidence={"kind":"visual_estimate","basis":"RGB holes"})'),
        fenced(f'ReturnAnswer(18,evidence={count_evidence(12)!r})'),
        fenced(f'ReturnAnswer(18,evidence={count_evidence(18)!r})')])
    assert result["status"] == "answered" and result["raw_answer"] == 18
    assert result["tool_counts"]["calls"] == 0 and result["last_submission_state"] is None
    first = result["rounds"][0]["last_submission_state"]
    assert first["answer"] == 12 and first["status"] == "rejected_model_hypothesis"
    second = result["rounds"][1]["last_submission_state"]
    assert second["answer"] == 18 and second["round"] == 1
    assert len(second["evidence"]["object_scope"]["entities"]) == 12
    text = texts(read(tmp_path / "02.messages.json"))
    assert "new evidence may change the answer or entity list" in text
    assert "LAST REJECTED SUBMISSION" in text and "\nTOOL FACTS []" in text


def test_volume_nested_scope_recovers_with_rejected_submission_context(tmp_path, monkeypatch, scene_data):
    item = {**scene_data[0], "question": "What is the volume ratio of sofa and chair?", "task_type": "float"}
    bad = {"kind": "visual_estimate", "basis": "RGB cuboids",
           "assumptions": {"object_scope": {"include": "sofa and chair"}, "volume_model": "cuboids"}}
    good = deepcopy(bad)
    good["assumptions"]["object_scope"] = "one sofa and chair, excluding footstool"
    result = offline_episode(tmp_path, monkeypatch, scene_data, item, [
        fenced(f"ReturnAnswer(12.5,evidence={bad!r})"),
        fenced(f"ReturnAnswer(12.5,evidence={good!r})")])
    snapshot = result["rounds"][0]["last_submission_state"]
    assert snapshot["evidence"] == bad and snapshot["rejection_errors"][0]["field"].endswith(".object_scope")
    text = texts(read(tmp_path / "01.messages.json"))
    delivered = json.loads(text.split("\nLAST REJECTED SUBMISSION ", 1)[1].split("\n", 1)[0])
    assert delivered == snapshot
    assert result["raw_answer"] == 12.5 and result["request_count"] == 2


def test_rejected_calculation_precision_state_persistence_and_audit_tamper_detection(
        tmp_path, monkeypatch, scene_data):
    item = {**scene_data[0], "question": "What is the height ratio of chair and desk?", "task_type": "float"}
    sample = fenced(
        't=ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[30,30],'
        'targets=[{"target":"chair","surface":"top"},{"target":"chair","surface":"base"}])\n'
        'a=ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[20,20],'
        'targets=[{"target":"desk","surface":"top"},{"target":"desk","surface":"base"}])\n'
        'YieldObservations([t["result_id"],a["result_id"]])')
    def declaration(facts):
        by_target = {f["data"]["intended_targets"][0]["target"]: f for f in facts}
        bindings = {}
        # Reversed insertion order, including a dynamic dict() assignment.
        for name, target in [("t", "chair"), ("a", "desk")]:
            binding = dimension(scene_data[-1], by_target[target]["result_id"], target, "height")[1]
            bindings[name] = binding
        return {"kind": "calculation", "expression": "t/a", "bindings": bindings}
    def reject(facts):
        evidence = declaration(facts)
        return fenced(f'measurement_reviews={list(reviewed(facts).values())!r}\n'
                      f'evidence={evidence!r}\nevidence["bindings"]=dict(evidence["bindings"])\n'
                      'ReturnAnswer(2.000012345678901,evidence=evidence)')
    def final(facts):
        evidence = declaration(facts)
        by_key = {f["result_id"]: f["data"]["distance"] for f in facts}
        answer = by_key[evidence["bindings"]["t"]["result_id"]] / by_key[evidence["bindings"]["a"]["result_id"]]
        return json.dumps({"answer": answer, "measurement_reviews": [], "evidence": evidence})
    result = offline_episode(tmp_path, monkeypatch, scene_data, item,
        [sample, reject, *[fenced('YieldObservations([])')]*3, final])
    snapshot = result["rounds"][1]["last_submission_state"]
    assert snapshot["answer"] == 2.000012345678901 and snapshot["evidence"]["expression"] == "t/a"
    assert result["rounds"][4]["last_submission_state"] == snapshot
    assert result["status"] == "answered" and result["last_submission_state"] is None
    final_context = texts(read(tmp_path / "05.messages.json"))
    assert "2.000012345678901" in final_context
    assert all(len(r["observed_surfaces"]) == 2 for r in result["measurement_reviews"])
    assert len(result["rounds"][-1]["request"]["visible_measurement_ids"]) == 2
    for change, expected in [
        ("snapshot", "round_last_submission_changed"),
        ("receipt", "reconstructed_public_context_changed"),
        ("episode", "episode_last_submission_changed"),
    ]:
        changed = deepcopy(result)
        if change == "snapshot":
            changed["rounds"][1]["last_submission_state"]["answer"] = 999
        elif change == "receipt":
            changed["rounds"][2]["request"]["last_submission"]["state_sha256"] = "forged"
        else:
            changed["last_submission_state"] = snapshot
        with pytest.raises(ValueError, match=expected):
            list(replay_contexts(item, changed, *scene_data[1:4]))


@pytest.mark.parametrize("failure", [
    'ctx.tools.point_depth(view_index=0,x=-1,y=10)\nReturnAnswer(999)',
    'try:\n    ctx.tools.point_depth(view_index=0,x=-1,y=10)\nexcept Exception:\n    ReturnAnswer(999)',
])
def test_failed_cell_answer_never_enters_rejected_submission_state(tmp_path, monkeypatch, scene_data, failure):
    repairable = not failure.startswith("try:")
    outputs = [fenced(failure)]
    if repairable:
        outputs.append(fenced('ReturnAnswer(3)'))
    result = offline_episode(tmp_path, monkeypatch, scene_data, scene_data[0], outputs)
    assert result["rounds"][0]["cell"]["answer_untrusted"]
    assert all(r["last_submission_state"] is None for r in result["rounds"])
    assert result["last_submission_state"] is None


def test_rejected_state_survives_format_and_input_errors_without_adopting_failed_answer(
        tmp_path, monkeypatch, scene_data):
    item = {**scene_data[0], "question": "How many chairs are present?", "task_type": "int"}
    good = {"kind": "visual_estimate", "basis": "RGB count",
            "object_scope": scope([entity("E1", [], "rgb_only")])}
    result = offline_episode(tmp_path, monkeypatch, scene_data, item, [
        fenced('ReturnAnswer(12,evidence={"kind":"visual_estimate","basis":"RGB count"})'),
        "malformed response",
        fenced('ctx.tools.point_depth(view_index=0,x=-1,y=10)\nReturnAnswer(999)'),
        fenced(f'ReturnAnswer(1,evidence={good!r})')])
    snapshot = result["rounds"][0]["last_submission_state"]
    assert snapshot["answer"] == 12 and snapshot["round"] == 0
    assert result["rounds"][1]["last_submission_state"] == result["rounds"][2]["last_submission_state"] == snapshot
    assert result["rounds"][2]["cell"]["answer_untrusted"] and result["raw_answer"] == 1
    assert result["rounds"][3]["request"]["last_submission"]["source_sha256"] == snapshot["source_sha256"]


@pytest.mark.parametrize("basis", ["x" * 20000, "中" * 20000])
def test_snapshot_bounds_escaped_json_without_mutating_or_rounding_fields(basis):
    evidence = {"kind": "visual_estimate", "basis": basis}
    before = deepcopy(evidence)
    answer = 1.2345678901234567
    errors = [{"code": "visual_basis_required_max200", "field": "evidence.basis", "actual_chars": len(basis)}]
    snapshot = capture(3, answer, evidence, errors)
    assert size(snapshot) <= MAX_CHARS and evidence == before
    assert snapshot["answer"] == answer
    assert snapshot["source_sha256"] == digest({"answer": answer, "evidence": evidence, "rejection_errors": errors})
    assert snapshot["omitted"] == [{"field": "evidence.basis", "sha256": digest(basis),
                                     "reason": "submission_budget"}]
    compact = delivery(snapshot, 400)
    assert size(compact) <= 400 and compact["source_sha256"] == snapshot["source_sha256"]
    assert "answer" not in compact and compact["omitted"][0]["reason"] == "context_budget"


def test_final_seven_full_precision_dependencies_take_priority_over_huge_rejected_scope(scene_data):
    item, rgb, frame, cache = scene_data[:4]
    item = {**item, "question": "What is the height of the chair?", "task_type": "float"}
    rows = [dimension(scene_data[-1], f"H{n}", "chair", "height")[0] for n in range(7)]
    for n, row in enumerate(rows):
        for patch in ("a_depth_samples", "b_depth_samples"):
            row["data"][patch]["values"] = [
                [math.sqrt(2) + (n*25+y*5+x)*0.0000123456789 for x in range(5)] for y in range(5)]
    before = deepcopy(rows)
    ids = [r["result_id"] for r in rows]
    large = capture(4, 1.2345678901234567,
                    {"kind": "visual_estimate", "basis": "x"*20000, "object_scope": scope([
                        entity(f"E{n}", [], "rgb_only") for n in range(128)])},
                    [{"code": "visual_basis_required_max200"}])
    context, receipt = messages(item, rgb, frame, cache, arm="skill", config=Config(),
        round_index=5, observations=rows, reviews=reviewed(rows), dependency_ids=ids, last_submission=large)
    assert receipt["visible_measurement_ids"] == ids and not receipt["context_packing"]["omitted_dependency_ids"]
    assert receipt["text_chars"] <= 20000 and rows == before
    assert receipt["last_submission"]["state_sha256"] == digest(large)
    assert "LAST REJECTED SUBMISSION" in texts(context)


def test_direct_message_payload_ignores_new_state_and_contains_only_original_evidence(scene_data):
    args = scene_data[:4]
    original, receipt = messages(*args, arm="direct", config=Config())
    changed, other = messages(*args, arm="direct", config=Config(),
                             last_submission=capture(0, 99, None, []))
    assert changed == original and other == receipt
    assert "LAST REJECTED SUBMISSION" not in texts(original) and "last_submission" not in receipt
    assert receipt["image_count"] == 1 and receipt["visible_result_ids"] == []


def test_source_pin_requires_new_snapshot_module():
    directory = ROOT / "scripts/Ominirunner"
    required = ["worker", "review", "evidence", "sampling", "answer_contract", "coordinates",
                "instances", "progress", "recovery", "provenance", "task_evidence"]
    pins = {f"scripts/Ominirunner/{name}.py": sha(directory / f"{name}.py") for name in required}
    with pytest.raises(ValueError, match="solver_pin_missing"):
        validate_source(pins)
    pins["scripts/Ominirunner/submission.py"] = sha(directory / "submission.py")
    validate_source(pins)
