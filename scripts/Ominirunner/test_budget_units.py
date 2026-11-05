"""Offline regressions for final text budget and imperial length contracts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from Ominirunner.common import Config
from Ominirunner.evidence import EvidenceError, validate as validate_evidence
from Ominirunner.protocol import messages
from Ominirunner.test_contracts import (
    dimension, fenced, offline_episode, offline_only, reviewed, scene_data, task,
)


@pytest.mark.parametrize("span,unit", [
    ("2 m", "m"), ("2 meters", "m"),
    ("2 feet", "ft"), ("2 foot", "ft"), ("2 ft", "ft"),
    ("2 inches", "in"), ("2 inch", "in"), ("2in", "in"), ("2 in", "in"),
    ("2 yards", "yd"), ("2 yard", "yd"), ("2 yd", "yd"),
    ("2 miles", "mi"), ("2 mile", "mi"), ("2 mi", "mi"),
])
def test_public_length_aliases_preserve_native_value_and_unit(span, unit):
    contract = task(f"If the lamp is {span} tall, what is its height?", "float")
    public = contract["public_metric_values"][0]
    assert public["value"] == 2 and public["unit"] == unit and public["quantity"] == "height"
    assert contract["unit"] == unit


@pytest.mark.parametrize("asked_text,unit", [
    ("how many feet must it be raised?", "ft"),
    ("what is its height in inches?", "in"),
    ("what is its height in yards?", "yd"),
    ("what is its height in miles?", "mi"),
    ("what is its height in meters?", "m"),
    ("how tall is it (ft)?", "ft"),
])
def test_explicit_output_unit_overrides_the_given_span_unit(asked_text, unit):
    contract = task(f"If the height of the lamp is 1 mile, {asked_text}", "float")
    assert contract["unit"] == unit
    public = contract["public_metric_values"][0]
    assert public["value"] == 1 and public["unit"] == "mi"
    if unit != "mi":
        assert public["answer_unit"] == unit


@pytest.mark.parametrize("unit", ["feet", "inches", "yards", "miles"])
def test_how_many_units_tall_is_a_dimension_not_an_entity_count(unit):
    contract = task(f"How many {unit} tall is the lamp?", "float")
    assert contract["quantity"] == "height" and contract["unit"] != "count"


def test_preposition_in_is_not_an_inch_scale():
    assert task("How many chairs are present? There is 1 in the image.", "int")[
        "public_metric_values"] == []


def test_mile_to_feet_conversion_is_public_and_full_precision():
    contract = task(
        "If the red chair is 1 mile tall, how many feet must it rise to match the other chair's height?",
        "float")
    assert contract["unit"] == "ft"
    span = contract["public_metric_values"][0]
    assert span["unit"] == "mi" and span["value"] == 1
    assert span["answer_unit_value"] == 5280.0 and span["answer_unit"] == "ft"


@pytest.mark.parametrize("public,declared,unit", [
    ("1 mile", 5280.0, "ft"),
    ("5 feet", 1.524, "m"),
    ("12 inches", 1.0, "ft"),
    ("1 yard", 36.0, "in"),
    ("1 meter", 1 / 0.3048, "ft"),
])
def test_equivalent_public_scale_converts_before_arithmetic(scene_data, public, declared, unit):
    row, binding = dimension(scene_data[-1], "H", "lamp", "height")
    contract = task(f"If the lamp is {public} tall, what is its height in {unit}?", "float")
    evidence = {"kind": "calculation", "expression": "s*h/h", "bindings": {"h": binding},
                "scale": {"name": "s", "value": declared, "unit": unit, "anchor_binding": "h"}}
    assert validate_evidence(declared, evidence, [row], reviewed([row]), ["H"], contract)[
        "verified_arithmetic"] is True


@pytest.mark.parametrize("value,unit,error", [
    (1.0, "ft", "scale_anchor_invalid"),
    (5280.0, "mi", "scale_anchor_invalid"),
    (5281.0, "ft", "scale_anchor_invalid"),
    (1.0, "mi", "task_unit_mismatch"),
    (1609.344, "m", "task_unit_mismatch"),
    (5280.0, "feet", "scale_anchor_invalid"),
])
def test_conversion_does_not_allow_wrong_scale_or_output_unit(scene_data, value, unit, error):
    row, binding = dimension(scene_data[-1], "H", "lamp", "height")
    contract = task("If the lamp is 1 mile tall, what is its height in feet?", "float")
    evidence = {"kind": "calculation", "expression": "s*h/h", "bindings": {"h": binding},
                "scale": {"name": "s", "value": value, "unit": unit, "anchor_binding": "h"}}
    with pytest.raises(EvidenceError, match=error):
        validate_evidence(value, evidence, [row], reviewed([row]), ["H"], contract)


def test_foot_input_does_not_change_a_dimensionless_height_ratio(scene_data):
    a, ab = dimension(scene_data[-1], "A", "lamp", "height")
    b, bb = dimension(scene_data[-1], "B", "chair", "height")
    rows = [a, b]
    contract = task("If the lamp is 5 feet tall, what is the height ratio of lamp and chair?", "float")
    assert contract["unit"] == "ratio"
    evidence = {"kind": "calculation", "expression": "a/b", "bindings": {"a": ab, "b": bb}}
    assert validate_evidence(1, evidence, rows, reviewed(rows), ["A", "B"], contract)[
        "verified_arithmetic"]


@pytest.mark.parametrize("final", [False, True])
def test_worker_and_audit_use_feet_after_mile_conversion(tmp_path, monkeypatch, scene_data, final):
    item = {**scene_data[0], "task_type": "float",
            "question": "If the red chair is 1 mile tall, how many feet must it rise to match the left chair's height?"}
    sample = fenced(
        'r=ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[20,20],'
        'targets=[{"target":"red chair","surface":"top"},{"target":"red chair","surface":"base"}])\n'
        'l=ctx.tools.pair_point_distance(view_index=0,a_xy=[10,10],b_xy=[30,30],'
        'targets=[{"target":"left chair","surface":"top"},{"target":"left chair","surface":"base"}])\n'
        'YieldObservations([r["result_id"],l["result_id"]],reason="Review chair extrema")')

    def finish(facts):
        by_target = {row["data"]["intended_targets"][0]["target"]: row for row in facts}
        bindings = {}
        # Keep binding order deterministic to isolate the unit-conversion path.
        for name, target in (("l", "left chair"), ("r", "red chair")):
            bindings[name] = {
                "result_id": by_target[target]["result_id"], "quantity": "height",
                "geometry": {"coverage": "complete", "axis": "object_vertical",
                             "anchors": ["top", "base"], "objects": [target, target]}}
        scale = task(item["question"], "float")["public_metric_values"][0]["answer_unit_value"]
        r = by_target["red chair"]["data"]["distance"]
        l = by_target["left chair"]["data"]["distance"]
        answer = scale * (l / r - 1)
        evidence = {"kind": "calculation", "expression": "s*(l/r-1)", "bindings": bindings,
                    "scale": {"name": "s", "value": scale, "unit": "ft", "anchor_binding": "r"}}
        reviews = list(reviewed(facts).values())
        if final:
            return json.dumps({"answer": answer, "measurement_reviews": reviews, "evidence": evidence})
        return fenced(
            f"measurement_reviews={reviews!r}\n"
            f"s={scale!r}\nr={r!r}\nl={l!r}\n"
            f"ReturnAnswer(s*(l/r-1),evidence={evidence!r})")

    outputs = [sample]
    if final:
        outputs += [fenced('YieldObservations([],reason="Await final")')] * 4
    result = offline_episode(tmp_path, monkeypatch, scene_data, item, outputs + [finish])
    assert result["status"] == "answered"
    assert math.isclose(result["raw_answer"], 5280, rel_tol=1e-12)
    assert result["public_answer_contract"]["unit"] == "ft"
    assert result["answer_evidence"]["scale"]["value"] == 5280
    assert result["answer_evidence"]["validation_layers"]["task_unit_contract"] is True


def test_only_skill_final_text_limit_increases(scene_data):
    item, rgb, frame, cache = scene_data[:4]
    config = Config()
    assert config.text_caps == (None, None, None, None, None, 20000)
    assert config.text_record_threshold == 10000
    for arm, index, cap in (("direct", 0, 10000), ("skill", 0, None), ("skill", 5, 20000)):
        _, receipt = messages(item, rgb, frame, cache, arm=arm, config=config, round_index=index)
        assert receipt["text_cap"] == cap


def test_twenty_thousand_chars_retains_seven_full_precision_dependencies(scene_data):
    item, rgb, frame, cache = scene_data[:4]
    item = {**item, "question": "What is the height of the chair?", "task_type": "float"}
    rows = [dimension(scene_data[-1], f"H{index}", "chair", "height")[0] for index in range(7)]
    for index, row in enumerate(rows):
        for patch in ("a_depth_samples", "b_depth_samples"):
            row["data"][patch]["values"] = [
                [math.sqrt(2) + (index * 25 + y * 5 + x) * 0.0000123456789 for x in range(5)]
                for y in range(5)]
    before = deepcopy(rows)
    ids = [row["result_id"] for row in rows]
    reviews = reviewed(rows)
    old = replace(Config(), text_caps=(None, None, None, None, None, 10000))
    _, old_receipt = messages(item, rgb, frame, cache, arm="skill", config=old, round_index=5,
                              observations=rows, reviews=reviews, dependency_ids=ids)
    context, receipt = messages(item, rgb, frame, cache, arm="skill", config=Config(), round_index=5,
                                observations=rows, reviews=reviews, dependency_ids=ids)
    assert old_receipt["context_packing"]["omitted_dependency_ids"]
    assert receipt["visible_measurement_ids"] == ids
    assert receipt["context_packing"]["omitted_dependency_ids"] == []
    assert 10000 < receipt["text_chars"] <= 20000 and receipt["text_over_threshold"]
    assert rows == before
    text = "\n".join(part.get("text", "") for part in context[0]["content"])
    facts = json.loads(text.split("\nTOOL FACTS ", 1)[1].split("\n", 1)[0])
    assert [fact["data"] for fact in facts] == [
        {key: value for key, value in row["data"].items()
         if key not in ("coordinate_convention", "definition")} for row in rows]
