"""Public task-expression regressions; no dataset labels or online clients."""
from __future__ import annotations

import builtins
from pathlib import Path
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Ominirunner.answer_contract import canonical, specification


def task(question, task_type="float", options=None, **metadata):
    return specification({"question": question, "task_type": task_type,
                          "options": options or [], **metadata})


def op(role, quantity, target):
    return {"role": role, "quantity": quantity, "target": target}


def labels(contract):
    return [row["label"] for row in contract["allowed_answers"]]


def test_location_count_alternatives_are_not_yes_no():
    contract = task("Are there more chairs at the dining table or the kitchen counter?", "str")
    assert labels(contract) == ["dining table", "kitchen counter"]
    assert contract["quantity"] == "count" and contract["unit"] == "label"
    assert contract["operation"] == "comparison"
    assert contract["comparison"] == {"operator": "argmax"}
    assert contract["operands"] == [
        op("candidate", "count", "chairs at dining table"),
        op("candidate", "count", "chairs at kitchen counter")]
    assert canonical("O2", contract) == "kitchen counter"
    with pytest.raises(ValueError, match="answer_option_invalid"):
        canonical("yes", contract)


@pytest.mark.parametrize("choices", [
    "the chair, the table or the cabinet",
    "the chair, the table, or the cabinet",
])
def test_three_dimension_alternatives_remain_ordered(choices):
    contract = task(f"Which object has the largest 3D height: {choices}?", "str")
    assert labels(contract) == ["chair", "table", "cabinet"]
    assert contract["operation"] == "comparison"
    assert contract["comparison"] == {"operator": "argmax"}
    assert contract["operands"] == [op("candidate", "height", v)
                                     for v in ("chair", "table", "cabinet")]


@pytest.mark.parametrize("question,quantity,target,scale_quantity,scale_target", [
    ("If the 3D height of the bench that the purple pillow is resting on is 5 feet, "
     "what is the 3D width of the television in feet?",
     "width", "television", "height", "bench that the purple pillow is resting on"),
    ("If the 3D width of the sink is 4 meters, what is the length of the stove top?",
     "length", "stove top", "width", "sink"),
    ("If the bowl on the table has a 3D diameter of 4 meters, "
     "how long is the table in 3D in meters?",
     "length", "table", "diameter", "bowl on the table"),
    ("If the TV is 1.9m wide, how tall is the TV stand?",
     "height", "tv stand", "width", "tv"),
])
def test_scale_and_requested_dimension_have_distinct_targets(
        question, quantity, target, scale_quantity, scale_target):
    contract = task(question)
    assert contract["quantity"] == quantity
    assert contract["operation"] == "measure"
    assert contract["operands"] == [op("target", quantity, target),
                                     op("scale_anchor", scale_quantity, scale_target)]
    span = contract["public_metric_values"][0]
    assert span["quantity"] == scale_quantity and span["target"] == scale_target
    assert span["operand_index"] == 1


def test_mixed_dimension_ratio_keeps_numerator_and_denominator():
    contract = task(
        "What is the ratio of the total 3D width of the white cabinets on the right "
        "to the 3D height of the chair closest to the camera?")
    assert contract["quantity"] == "dimension_ratio" and contract["unit"] == "ratio"
    assert contract["operation"] == "ratio"
    assert contract["operands"] == [
        op("numerator", "width", "white cabinets on the right"),
        op("denominator", "height", "chair closest to the camera")]
    assert contract["reference"] is None and contract["distance_relation"] is None


@pytest.mark.parametrize("source_quantity,target_quantity,expected", [
    ("width", "width", "width_ratio"), ("width", "height", "dimension_ratio"),
])
def test_multiplicative_factor_is_dimensionless_and_ordered(source_quantity, target_quantity, expected):
    contract = task(
        f"What number would you have to multiply the {source_quantity} of the small stool "
        f"such that it matches the {target_quantity} of the fireplace?")
    assert contract["quantity"] == expected and contract["unit"] == "ratio"
    assert contract["operation"] == "ratio"
    assert contract["operands"] == [
        op("numerator", target_quantity, "fireplace"),
        op("denominator", source_quantity, "small stool")]


def test_dimension_comparison_records_stacked_height_multiplier():
    contract = task(
        "Is the 3D width of the table larger than the height of three stacked microwaves?", "str")
    assert labels(contract) == ["yes", "no"]
    assert contract["quantity"] == "dimension_comparison" and contract["unit"] == "label"
    assert contract["operation"] == "comparison"
    assert contract["operands"] == [op("left", "width", "table"), op("right", "height", "microwaves")]
    assert contract["comparison"] == {"operator": ">", "factor": 3}


def test_count_ratio_is_not_a_visual_judgment():
    contract = task("What is the ratio of pillows to remote controls? Answer with a decimal")
    assert contract["quantity"] == "count_ratio" and contract["unit"] == "ratio"
    assert contract["operation"] == "ratio"
    assert contract["operands"] == [op("numerator", "count", "pillows"),
                                     op("denominator", "count", "remote controls")]


def test_fixed_origin_distance_does_not_use_entire_or_phrase_as_reference():
    contract = task(
        "Is the rightmost chair closer to the table or the wooden dresser? Options: {table, dresser}", "str")
    assert labels(contract) == ["table", "dresser"]
    assert contract["quantity"] == "object_distance_order"
    assert contract["reference"] == "rightmost chair"
    assert contract["operands"] == [op("origin", "object_distance", "rightmost chair"),
                                     op("candidate", "object_distance", "table"),
                                     op("candidate", "object_distance", "wooden dresser")]
    assert contract["distance_relation"] == {
        "kind": "fixed_origin", "origin": "rightmost chair", "candidates": ["table", "wooden dresser"]}


@pytest.mark.parametrize("reference,quantity,measured", [
    ("camera", "camera_order", "camera_z"),
    ("armchair", "object_distance_order", "object_distance"),
])
def test_common_reference_distance_retains_independent_candidates(reference, quantity, measured):
    contract = task(f"Which object is closer to the {reference}: the tv or the chair?", "str")
    assert labels(contract) == ["tv", "chair"]
    assert contract["quantity"] == quantity and contract["reference"] == reference
    assert contract["operands"] == [op("candidate", measured, "tv"),
                                     op("candidate", measured, "chair"),
                                     op("reference", measured, reference)]
    assert contract["distance_relation"] == {
        "kind": "common_reference", "reference": reference, "candidates": ["tv", "chair"]}


def test_boolean_camera_order_binds_subject_and_than_object():
    contract = task("Is the kettle farther away from the camera than the chicken?", "str")
    assert labels(contract) == ["yes", "no"]
    assert contract["quantity"] == "camera_order" and contract["reference"] == "camera"
    assert contract["comparison"] == {"operator": ">"}
    assert contract["operands"] == [op("left", "camera_z", "kettle"),
                                     op("right", "camera_z", "chicken"),
                                     op("reference", "camera_z", "camera")]


@pytest.mark.parametrize("dimension", ["height", "width", "length", "volume"])
def test_legacy_homogeneous_ratio_contract(dimension):
    contract = task(f"What is the {dimension} ratio of the chair and desk?")
    assert contract["quantity"] == dimension + "_ratio"
    assert contract["unit"] == "ratio" and contract["operation"] == "ratio"
    assert contract["operands"] == [op("numerator", dimension, "chair"),
                                     op("denominator", dimension, "desk")]
    assert {"quantity", "unit", "reference", "allowed_answers", "public_metric_values"} <= contract.keys()


def test_combined_denominator_is_not_split_into_answer_choices():
    contract = task(
        "What is the ratio of the height of the wooden table to the combined height of the TV and the TV table?")
    assert labels(contract) == []
    assert contract["operands"] == [op("numerator", "height", "wooden table"),
                                     op("denominator", "height", "tv"),
                                     op("denominator", "height", "tv table")]


def test_walking_request_and_scale_have_separate_destinations():
    contract = task(
        "If I have to walk 3.2 meters to get to the leftmost lamp, "
        "how many meters do I have to walk to get to the closest barstool?")
    assert contract["quantity"] == "walking_distance" and contract["unit"] == "m"
    assert contract["operands"] == [op("target", "walking_distance", "closest barstool"),
                                     op("scale_anchor", "walking_distance", "leftmost lamp")]


def test_two_scales_keep_local_quantities_targets_and_conversion():
    contract = task(
        "If the shelf is 2m wide and the lamp is 3 feet tall, how long is the sofa in inches?")
    assert contract["quantity"] == "length" and contract["unit"] == "in"
    assert contract["operands"] == [op("target", "length", "sofa"), op("scale_anchor", "width", "shelf"),
                                     op("scale_anchor", "height", "lamp")]
    scales = contract["public_metric_values"]
    assert [span["operand_index"] for span in scales] == [1, 2]
    assert scales[0]["answer_unit_value"] == 2 / .0254
    assert scales[1]["answer_unit_value"] == pytest.approx(36)


def test_anaphoric_dimension_binds_public_anchor_without_synonym_inference():
    contract = task("If the Side__Table is 2m wide, what is its width?")
    assert contract["operands"] == [op("target", "width", "side table"),
                                     op("scale_anchor", "width", "side table")]


def test_raising_in_feet_is_height_not_entity_count():
    contract = task("If the height of the lamp is 1 mile, how many feet must it be raised?")
    assert contract["quantity"] == "height" and contract["unit"] == "ft"
    assert contract["operands"] == [op("target", "height", "lamp"),
                                     op("scale_anchor", "height", "lamp")]
    assert contract["public_metric_values"][0]["answer_unit_value"] == 5280


@pytest.mark.parametrize("condition", [
    "If the height of the chair is 2m",
    "If the width of the cabinet is 2m",
    "If the length of the bench is 2m",
])
def test_request_does_not_follow_condition_keyword_priority(condition):
    contract = task(f"{condition}, what is the width of the oven?")
    assert contract["quantity"] == "width"
    assert contract["operands"][0] == op("target", "width", "oven")


def test_explicit_option_labels_do_not_replace_qualified_task_targets():
    contract = task(
        "Which object is closer to the camera: the gray chair or the red table?",
        "str", ["chair", "table"])
    assert labels(contract) == ["chair", "table"]
    assert [row["target"] for row in contract["operands"][:2]] == ["gray chair", "red table"]


def test_implicit_option_labels_preserve_case_but_targets_are_normalized():
    contract = task("Which object is closer to the camera: the TV or the Side_Table?", "str")
    assert labels(contract) == ["TV", "Side_Table"]
    assert [row["target"] for row in contract["operands"][:2]] == ["tv", "side table"]
    assert canonical("O1", contract) == "TV"


def test_sample_metadata_cannot_change_task_semantics():
    question = "What is the ratio of the width of the cabinet to the height of the chair?"
    assert task(question, sample_id="133") == task(question, sample_id="another-public-id")


def test_parser_requires_no_file_or_network_access(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("task parsing must be pure")

    monkeypatch.setattr(builtins, "open", denied)
    monkeypatch.setattr(Path, "open", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    assert task("What is the width of the cabinet?")["operands"] == [op("target", "width", "cabinet")]


@pytest.mark.parametrize("bad", [["tv", "TV"], [""], ["x"] * 17, [1]])
def test_option_validation_remains_strict(bad):
    with pytest.raises(ValueError, match="public_options_invalid"):
        specification({"question": "Choose an object.", "task_type": "str", "options": bad})
