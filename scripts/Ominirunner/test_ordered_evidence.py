"""Public operand order, full-dimension roles and applicable executable templates."""
import ast
from copy import deepcopy
import json
import re

import pytest

from .evidence import EvidenceError, validate
from .native import parse_program
from .protocol import task_output
from .review import validate as validate_reviews
from .test_contracts import dimension, offline_only, point, reviewed, scene_data, task


def span(scene, key, target, quantity):
    if quantity != "diameter":
        return dimension(scene, key, target, quantity)
    row, binding = dimension(scene, key, target, "width")
    row["data"]["intended_targets"] = [
        {"target": target, "surface": "rim", "extent_role": role, "extent_basis": "RGB full diameter"}
        for role in ("rim_a", "rim_b")]
    binding.update(quantity="diameter", geometry={"coverage": "complete",
        "axis": "object_diameter", "anchors": ["rim_a", "rim_b"], "objects": [target, target]})
    return row, binding


def full_reviews(rows):
    submitted = []
    for row in rows:
        surfaces = []
        for target in row["data"]["intended_targets"]:
            surfaces.append({"target": target["target"], "observed_surface": target["surface"],
                "extent_role": target.get("extent_role", target["surface"]),
                "extent_basis": target.get("extent_basis", "RGB full same-object axis extremum"),
                "scope_completeness": "complete", "issues": [],
                "decision": "usable", "visibility": "visible"})
        submitted.append({"result_id": row["result_id"], "decision": "usable", "observed_surfaces": surfaces})
    result, errors = validate_reviews(submitted, rows, [r["result_id"] for r in rows])
    assert not errors
    return result


@pytest.mark.parametrize("question,targets,quantities", [
    ("What is the ratio of the total 3D width of the white cabinets on the right to the 3D height of the chair closest to the camera?",
     ("white cabinets on the right", "chair closest to the camera"), ("width", "height")),
    ("What number would you have to multiply the width of the small stool such that it matches the height of the fireplace?",
     ("fireplace", "small stool"), ("height", "width")),
])
def test_mixed_ratio_allows_both_dimensions_but_rejects_reversed_formula(
        scene_data, question, targets, quantities):
    rows, bindings = [], {}
    for name, target, q in zip(("n", "d"), targets, quantities):
        row, binding = span(scene_data[-1], name.upper(), target, q)
        rows.append(row)
        bindings[name] = binding
    rows[0]["data"]["distance"], rows[1]["data"]["distance"] = 4.0, 2.0
    evidence = {"kind": "calculation", "expression": "n/d", "bindings": bindings}
    contract = task(question, "float")
    assert validate(2.0, evidence, rows, full_reviews(rows), ["N", "D"], contract)["verified_arithmetic"]
    evidence["expression"] = "d/n"
    with pytest.raises(EvidenceError, match="task_expression_mismatch"):
        validate(.5, evidence, rows, full_reviews(rows), ["N", "D"], contract)


def test_comparison_requires_the_stacked_multiplier(scene_data):
    w, wb = span(scene_data[-1], "W", "table", "width")
    h, hb = span(scene_data[-1], "H", "microwaves", "height")
    w["data"]["distance"], h["data"]["distance"] = 7, 2
    rows = [w, h]
    contract = task("Is the 3D width of the table larger than the height of three stacked microwaves?")
    evidence = {"kind": "tool_comparison", "bindings": {"w": wb, "h": hb}, "left": "w",
                "right": "3*h", "operator": ">", "answers": {"true": "O1", "false": "O2"}}
    assert validate("O1", evidence, rows, full_reviews(rows), ["W", "H"], contract)["verified_arithmetic"]
    evidence["right"] = "h"
    with pytest.raises(EvidenceError, match="task_expression_mismatch"):
        validate("O1", evidence, rows, full_reviews(rows), ["W", "H"], contract)


def test_diameter_is_a_full_diameter_anchor_and_not_a_motion_span(scene_data):
    t, tb = span(scene_data[-1], "T", "table", "length")
    a, ab = span(scene_data[-1], "A", "bowl on the table", "diameter")
    t["data"]["distance"], a["data"]["distance"] = 3, 2
    rows = [t, a]
    contract = task("If the bowl on the table has a 3D diameter of 4 meters, how long is the table in 3D in meters?", "float")
    evidence = {"kind": "calculation", "expression": "s*t/a", "bindings": {"t": tb, "a": ab},
                "scale": {"name": "s", "value": 4, "unit": "m", "anchor_binding": "a"}}
    assert validate(6, evidence, rows, full_reviews(rows), ["T", "A"], contract)["verified_arithmetic"]
    reversed_scale = deepcopy(evidence)
    reversed_scale["expression"] = "s*a/t"
    with pytest.raises(EvidenceError, match="task_expression_mismatch"):
        validate(4 * 2 / 3, reversed_scale, rows, full_reviews(rows), ["T", "A"], contract)
    bad = deepcopy(evidence)
    bad["bindings"]["a"]["geometry"]["coverage"] = "partial"
    with pytest.raises(EvidenceError, match="complete_dimension_not_declared"):
        validate(6, bad, rows, full_reviews(rows), ["T", "A"], contract)


def test_fixed_origin_pairs_bind_distinct_candidates_in_either_endpoint_order(scene_data):
    rows = []
    for key, target, distance in (("A", "table", 2), ("B", "wooden dresser", 4)):
        row, _ = dimension(scene_data[-1], key, target, "height")
        row["data"]["intended_targets"] = [
            {"target": "rightmost chair", "surface": "center"}, {"target": target, "surface": "center"}]
        row["data"]["distance"] = distance
        if key == "B":
            row["data"]["intended_targets"].reverse()
        rows.append(row)
    contract = task("Is the rightmost chair closer to the table or the wooden dresser? Options: {table, dresser}")
    evidence = {"kind": "tool_comparison", "left": "a", "right": "b", "operator": "<",
                "answers": {"true": "O1", "false": "O2"}, "bindings": {
                    name: {"result_id": name.upper(), "quantity": "chord", "reference": "rightmost chair"}
                    for name in ("a", "b")}}
    assert validate("O1", evidence, rows, reviewed(rows), ["A", "B"], contract)["verified_arithmetic"]
    evidence["bindings"]["b"]["reference"] = "table or the wooden dresser"
    with pytest.raises(EvidenceError, match="task_quantity_mismatch") as caught:
        validate("O1", evidence, rows, reviewed(rows), ["A", "B"], contract)
    assert caught.value.record["expected"] == "rightmost chair"


def test_component_surface_and_extent_role_are_independent(scene_data):
    row, binding = dimension(scene_data[-1], "H", "fireplace", "height")
    row["data"]["intended_targets"][0] = {"target": "fireplace", "surface": "mantel_top",
        "extent_role": "top", "extent_basis": "RGB full fireplace upper extremum"}
    rows = [row]
    evidence = {"kind": "calculation", "expression": "h", "bindings": {"h": binding}}
    contract = task("What is the height of the fireplace?", "float")
    reviews = full_reviews(rows)
    reviews["H"]["observed_surfaces"][0]["extent_basis"] = "RGB shows the full upper boundary on this fireplace"
    # A fresh observation explains its own evidence; it need not copy an
    # earlier sampling hypothesis's wording.
    accepted, errors = validate_reviews(list(reviews.values()), rows, ["H"])
    assert not errors
    reviews = accepted
    checked = validate(row["data"]["distance"], evidence, rows, reviews, ["H"], contract)
    assert checked["validation_layers"]["independent_visual_geometry"] is False
    assert reviews["H"]["observed_surfaces"][0]["observed_surface"] == "mantel_top"
    reviews["H"]["observed_surfaces"][0]["extent_role"] = "left"
    with pytest.raises(EvidenceError, match="complete_dimension_not_declared"):
        validate(row["data"]["distance"], evidence, rows, reviews, ["H"], contract)


@pytest.mark.parametrize("question", [
    "Are there more chairs at the dining table or the kitchen counter?",
    "Which object has the largest 3D height: the chair, the table or the cabinet?",
])
def test_text_choices_remain_available_for_direct_rgb(question):
    contract = task(question)
    checked = validate("O2", {"kind": "visual_estimate", "basis": "Original RGB judgment"}, [], {}, [], contract)
    assert checked["kind"] == "visual_estimate" and not checked["verified_arithmetic"]


@pytest.mark.parametrize("question", [
    "What is the ratio of the width of the cabinet to the height of the chair?",
    "What is the ratio of the height of the wooden table to the combined height of the TV and the TV table?",
    "If the bowl on the table has a 3D diameter of 4 meters, how long is the table in 3D in meters?",
    "Is the 3D width of the table larger than the height of three stacked microwaves?",
])
def test_complete_calculation_templates_pass_host_validation(scene_data, question):
    contract = task(question, "str" if question.startswith("Is") else "float")
    code = re.findall(r"```python\n[\s\S]*?```", task_output(contract))[-1]
    body = ast.parse(parse_program(code)).body[0].body
    declared = ast.literal_eval(body[-2].value)
    # The template's values and binding identities are illustrative, but its
    # nested fields, targets, axes, units and task expression are all real schema.
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in body[:-3]}
    rows = []
    for name, binding in declared["bindings"].items():
        row, _ = span(scene_data[-1], binding["result_id"], binding["geometry"]["objects"][0], binding["quantity"])
        row["data"]["distance"] = values[name]
        rows.append(row)
    expression = ast.Expression(body[-3].value)
    answer = eval(compile(expression, "<public-template>", "eval"), {"__builtins__": {}}, values)
    assert validate(answer, declared, rows, full_reviews(rows), [r["result_id"] for r in rows], contract)["verified_arithmetic"]
    final_template = task_output(contract, final=True).splitlines()[-1]
    assert json.loads(final_template)["evidence"] == declared
