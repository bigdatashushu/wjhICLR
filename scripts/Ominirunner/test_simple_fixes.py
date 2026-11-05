"""Offline regressions for bounded interface and feedback fixes."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from Ominirunner.evidence import EvidenceError, validate as validate_evidence
from Ominirunner.protocol import messages
from Ominirunner.common import Config
from Ominirunner.review import validate as validate_reviews
from Ominirunner.test_contracts import (
    dimension, entity, fenced, offline_episode, offline_only, point, reviewed,
    scene_data, scope, task,
)


@pytest.mark.parametrize("question,expected", [
    ("If the lamp is 0.6m tall, how wide is the bookshelf?", ["height"]),
    ("If the shelf is 2 m wide, how tall is the lamp?", ["width"]),
    ("If the sofa is 3 meters long, how wide is the chair?", ["length"]),
    ("If the cabinet is 1 m high, how long is the shelf?", ["height"]),
    ("If the 3D length of the sink is 20 meters, what is the height of the oven?", ["length"]),
    ("If the height of the lamp is 0.6m, how wide is the bookshelf?", ["height"]),
    ("If the width of the shelf is 2m, how tall is the lamp?", ["width"]),
    ("If its width is stated and its height is 1m, how wide is the shelf?", ["height"]),
    ("If the shelf is 2m wide and the lamp is 0.6m tall, how long is the sofa?", ["width", "height"]),
    ("If you walk 2m forward, how wide is the door?", ["walking_distance"]),
    ("If the lamp moves 2m forward, how wide is the bookshelf?", ["motion"]),
    ("Estimate the shelf width. A chair moves 2m forward.", ["motion"]),
])
def test_public_span_dimension_stays_with_its_local_description(question, expected):
    contract = task(question, "float")
    assert [span["quantity"] for span in contract["public_metric_values"]] == expected


def test_correct_height_anchor_can_scale_requested_width(scene_data):
    h, hb = dimension(scene_data[-1], "H", "lamp", "height")
    w, wb = dimension(scene_data[-1], "W", "bookshelf", "width")
    rows = [h, w]
    contract = task("If the lamp is 0.6m tall, how wide is the bookshelf?", "float")
    assert contract["quantity"] == "width"
    evidence = {"kind": "calculation", "expression": "s*w/h",
                "bindings": {"h": hb, "w": wb},
                "scale": {"name": "s", "value": 0.6, "unit": "m", "anchor_binding": "h"}}
    answer = 0.6 * w["data"]["distance"] / h["data"]["distance"]
    assert validate_evidence(answer, evidence, rows, reviewed(rows), ["H", "W"], contract)[
        "verified_arithmetic"] is True
    evidence["scale"]["anchor_binding"] = "w"
    with pytest.raises(EvidenceError, match="scale_anchor_invalid"):
        validate_evidence(answer, evidence, rows, reviewed(rows), ["H", "W"], contract)


@pytest.mark.parametrize("sampled,public", [
    ("side_table", "side table"),
    ("Side__Table", "SIDE table"),
    ("side table", "side_table"),
    ("side_table", "side  table"),
])
def test_scale_name_accepts_separator_and_case_variants_only(scene_data, sampled, public):
    row, binding = dimension(scene_data[-1], "H", sampled, "height")
    contract = task(f"If the {public} is 1.2m tall, what is its height?", "float")
    evidence = {"kind": "calculation", "expression": "s*h/h", "bindings": {"h": binding},
                "scale": {"name": "s", "value": 1.2, "unit": "m", "anchor_binding": "h"}}
    assert validate_evidence(1.2, evidence, [row], reviewed([row]), ["H"], contract)[
        "verified_arithmetic"]


@pytest.mark.parametrize("sampled,public", [
    ("side_table", "bedside table"),
    ("table", "tabletop"),
    ("chair", "armchair"),
    ("__", "side table"),
])
def test_scale_name_does_not_accept_substrings_or_empty_names(scene_data, sampled, public):
    row, binding = dimension(scene_data[-1], "H", sampled, "height")
    contract = task(f"If the {public} is 1.2m tall, what is its height?", "float")
    evidence = {"kind": "calculation", "expression": "s*h/h", "bindings": {"h": binding},
                "scale": {"name": "s", "value": 1.2, "unit": "m", "anchor_binding": "h"}}
    with pytest.raises(EvidenceError, match="scale_anchor_invalid"):
        validate_evidence(1.2, evidence, [row], reviewed([row]), ["H"], contract)


def candidate_observation():
    return {"result_id": "V", "source_tool": "view_candidates",
            "data": {"candidates": [
                {"candidate_id": f"C{index}", "category": "chair"} for index in range(10)]}}


@pytest.mark.parametrize("ids,relation,expected", [
    ([], "single", "rgb_only"),
    (["C0"], "same_object", "single"),
    (["C0", "C1"], "single", ["same_object", "part_whole"]),
])
def test_relationship_feedback_identifies_the_actual_entity_field(ids, relation, expected):
    evidence = {"kind": "visual_estimate", "basis": "RGB count",
                "object_scope": scope([entity("E", ids, relation)])}
    with pytest.raises(EvidenceError, match="object_scope_merge_requires_rgb") as caught:
        validate_evidence(1, evidence, [candidate_observation()], {}, [],
                          task("How many chairs are there?", "int"), visible_result_ids=["V"])
    record = caught.value.record
    assert record["field"] == "evidence.object_scope.entities[0].relationship"
    assert record["candidate_count"] == len(ids) and record["actual_relationship"] == relation
    assert record.get("expected_relationship", record.get("expected_relationships")) == expected
    assert str(expected[0] if isinstance(expected, list) else expected) in record["repair"]


def test_candidate_visibility_does_not_grant_measurement_visibility(scene_data):
    row = point(scene_data[-1], "A")
    evidence = {"kind": "calculation", "expression": "z",
                "bindings": {"z": {"result_id": "A", "quantity": "camera_z"}}}
    with pytest.raises(EvidenceError, match="calculation_measurement_not_delivered"):
        validate_evidence(row["data"]["depth"], evidence, [row], reviewed([row]), [],
                          visible_result_ids=["A", "V"])


@pytest.mark.parametrize("candidate,visible", [("C0", []), ("C9", ["V"]), ("FAKE", ["V"])])
def test_count_rejects_undelivered_hidden_and_forged_candidates(candidate, visible):
    evidence = {"kind": "visual_estimate", "basis": "RGB count",
                "object_scope": scope([entity("E", [candidate])])}
    with pytest.raises(EvidenceError, match="object_scope_identity_invalid"):
        validate_evidence(1, evidence, [candidate_observation()], {}, ["V"],
                          task("How many chairs are there?", "int"), visible_result_ids=visible)


def test_missing_surfaces_feedback_locates_review_declaration(scene_data):
    row, binding = dimension(scene_data[-1], "H", "lamp", "height")
    reviews = {"H": {"result_id": "H", "decision": "usable"}}
    evidence = {"kind": "calculation", "expression": "h", "bindings": {"h": binding}}
    with pytest.raises(EvidenceError, match="measurement_surface_required") as caught:
        validate_evidence(row["data"]["distance"], evidence, [row], reviews, ["H"],
                          task("What is the height of the lamp?", "float"))
    record = caught.value.record
    assert record["measurement_result_id"] == "H"
    assert record["review_field"] == "measurement_reviews[].observed_surfaces"
    assert "inside each measurement_reviews record" in record["repair"]


def test_host_geometry_metadata_is_not_delivered_as_scene_information(scene_data):
    item, rgb, frame, cache = scene_data[:4]
    original = task(item["question"], item["task_type"])
    for round_index in (0, 5):
        context, receipt = messages(item, rgb, frame, cache, arm="skill",
                                    config=Config(), round_index=round_index)
        text = "\n".join(part.get("text", "") for part in context[0]["content"])
        public = json.loads(text.split("\nPUBLIC ANSWER/TASK CONTRACT ", 1)[1].split("\n", 1)[0])
        assert "geometry_verification" not in public
        assert "same_model_surface_declaration" not in text
        assert public == {key: value for key, value in original.items()
                          if key != "geometry_verification"}
    assert original["geometry_verification"] == "same_model_surface_declaration"


@pytest.mark.parametrize("round_index", [0, 5])
def test_delivered_review_example_passes_the_host_schema(scene_data, round_index):
    item, rgb, frame, cache = scene_data[:4]
    context, _ = messages(item, rgb, frame, cache, arm="skill",
                          config=Config(), round_index=round_index)
    text = "\n".join(part.get("text", "") for part in context[0]["content"])
    example = json.loads(text.split(
        "\nQuantitative review example: measurement_reviews=", 1)[1].split(". For", 1)[0])
    row = point(scene_data[-1], "A")
    row["data"]["intended_targets"][0]["surface"] = "top"
    example[0]["result_id"] = "A"
    reviews, errors = validate_reviews(example, [row], ["A"])
    assert errors == [] and reviews["A"]["observed_surfaces"][0]["surface"] == "top"
    assert "For pairs include both surfaces in a/b order" in text


@pytest.mark.parametrize("final", [False, True])
def test_worker_and_audit_accept_visible_candidates_without_measurements(
        tmp_path, monkeypatch, scene_data, final):
    item = {**scene_data[0], "question": "How many chairs are there?", "task_type": "int"}
    sample = fenced(
        'r=ctx.tools.view_candidates(view_index=0,category="chair")\n'
        'YieldObservations([r["result_id"]],reason="Review RGB entities")')

    def finish(facts):
        candidate = facts[0]["data"]["candidates"][0]["candidate_id"]
        evidence = {"kind": "visual_estimate", "basis": "One synthetic RGB chair",
                    "object_scope": scope([entity("E", [candidate])])}
        if final:
            return json.dumps({"answer": 1, "measurement_reviews": [], "evidence": evidence})
        return fenced(f"ReturnAnswer(1, evidence={evidence!r})")

    outputs = [sample]
    if final:
        outputs += [fenced('YieldObservations([],reason="Await final")')] * 4
    outputs.append(finish)
    result = offline_episode(tmp_path, monkeypatch, scene_data, item, outputs)
    assert result["status"] == "answered" and result["answer"] == "1"
    assert result["request_count"] == (6 if final else 2)
    assert result["rounds"][-1]["request"]["visible_measurement_ids"] == []
    assert len(result["rounds"][-1]["request"]["visible_result_ids"]) == 1
    assert result["answer_evidence"]["scope_validation"]["candidate_ids"] == ["chair"]


def test_worker_and_audit_repair_rgb_relationship_without_merging(tmp_path, monkeypatch, scene_data):
    item = {**scene_data[0], "question": "How many chairs are there?", "task_type": "int"}
    bad = {"kind": "visual_estimate", "basis": "One RGB chair",
           "object_scope": scope([entity("E", [], "single")])}
    good = deepcopy(bad)
    good["object_scope"]["entities"][0]["relationship"] = "rgb_only"
    result = offline_episode(tmp_path, monkeypatch, scene_data, item, [
        fenced(f"ReturnAnswer(1,evidence={bad!r})"),
        fenced(f"ReturnAnswer(1,evidence={good!r})")])
    assert result["request_count"] == 2 and result["answer"] == "1"
    first = result["rounds"][0]["answer_feedback_errors"][0]
    assert first["field"] == "evidence.object_scope.entities[0].relationship"
    assert first["expected_relationship"] == "rgb_only"
    assert result["tool_counts"]["calls"] == 0
