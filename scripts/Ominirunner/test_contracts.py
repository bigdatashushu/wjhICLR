"""Seven-fix regressions using synthetic RGB/geometry and an offline client only."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import socket
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from Ominirunner import worker
from Ominirunner.answer_contract import canonical, specification
from Ominirunner.audit import replay_contexts
from Ominirunner.check import fixture, fenced
from Ominirunner.common import Config, digest, read
from Ominirunner.coordinates import original_descriptor, resolve
from Ominirunner.data import problem
from Ominirunner.evidence import EvidenceError, validate as validate_evidence
from Ominirunner.instances import candidate_facts, iou, validate_scope
from Ominirunner.progress import record, summary, unresolved
from Ominirunner.protocol import crop_sheets, messages, overlay
from Ominirunner.review import apply_updates, pending, validate as validate_reviews
from Ominirunner.sampling import pair_point_distance, point_depth
from Ominirunner.tools import build_registry
from skill3d.tools.contract import DomainValueError


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("live transport forbidden in contract tests")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(worker, "BailianClient", denied)


@pytest.fixture
def scene_data():
    item, rgb, frame, cache = fixture()
    registry, scene, _ = build_registry(cache, problem(item), Config())
    return item, rgb, frame, cache, registry, scene


def task(question, task_type="str", options=None):
    return specification({"question": question, "task_type": task_type, "options": options or []})


def point(scene, key="A", target="chair", xy=(10, 10)):
    data = point_depth(scene, 0, *xy)
    data.update(intended_targets=[{"target": target, "surface": "center"}],
                coordinate_transforms=[None], replaces_result_id=None)
    return {"result_id": key, "source_tool": "point_depth", "status": "ok", "data": data}


def dimension(scene, key, target, quantity):
    anchors = {"height": ["top", "base"], "width": ["left", "right"],
               "length": ["front", "back"]}[quantity]
    axes = {"height": "object_vertical", "width": "object_horizontal", "length": "object_length"}
    data = pair_point_distance(scene, 0, [10, 10], [20, 20])
    data.update(intended_targets=[{"target": target, "surface": anchor} for anchor in anchors],
                coordinate_transforms=[None, None], replaces_result_id=None)
    row = {"result_id": key, "source_tool": "pair_point_distance", "status": "ok", "data": data}
    binding = {"result_id": key, "quantity": quantity, "geometry": {
        "coverage": "complete", "axis": axes[quantity], "anchors": anchors,
        "objects": [target, target]}}
    return row, binding


def reviewed(rows):
    submitted = [{"result_id": row["result_id"], "decision": "usable", "observed_surfaces": [
        {**target, "decision": "usable", "visibility": "visible"}
        for target in row["data"]["intended_targets"]]} for row in rows]
    reviews, errors = validate_reviews(submitted, rows, [r["result_id"] for r in rows])
    assert errors == []
    return reviews


CAMERA_TASK = task("Which object is closer to the camera: chair or tv?")


def comparison():
    return {"kind": "tool_comparison", "bindings": {
        "a": {"result_id": "A", "quantity": "camera_z", "reference": "camera"},
        "b": {"result_id": "B", "quantity": "camera_z", "reference": "camera"}},
        "left": "a", "right": "b", "operator": "<", "answers": {"true": "O1", "false": "O2"}}


@pytest.mark.parametrize("question,options,expected", [
    ("Choose from tv, chair.", [], ["tv", "chair"]),
    ("Which direction? Options: {front, front right, right, back right, back, back left, left, front left}",
     [], ["front", "front right", "right", "back right", "back", "back left", "left", "front left"]),
    ("Which object is closer to the camera: the tv or the chair?", [], ["tv", "chair"]),
    ("Can the chair fit?", [], ["yes", "no"]),
    ("Select a public object.", ["tv", "chair"], ["tv", "chair"]),
])
def test_public_options(question, options, expected):
    contract = task(question, options=options)
    assert [r["label"] for r in contract["allowed_answers"]] == expected
    assert canonical("O1", contract) == expected[0]
    assert canonical(expected[0].upper(), contract) == expected[0]
    with pytest.raises(ValueError, match="answer_option_invalid"):
        canonical("the " + expected[0], contract)
    with pytest.raises(ValueError, match="answer_option_required"):
        canonical(1, contract)


@pytest.mark.parametrize("bad", [["tv", "TV"], [""], ["x"] * 17, [1]])
def test_public_options_reject_ambiguous_labels(bad):
    with pytest.raises(ValueError, match="public_options_invalid"):
        task("Choose an object.", options=bad)


@pytest.mark.parametrize("alias", ["wrong", "wrong target"])
def test_wrong_review_is_excluded_and_replaced(scene_data, alias):
    scene = scene_data[-1]
    old = point(scene)
    reviews, errors = validate_reviews([{"result_id": "A", "decision": alias}], [old], ["A"])
    assert errors == [] and reviews["A"]["decision"] == "resample"
    assert reviews["A"]["raw_decision"] == alias
    assert unresolved([old], reviews)[0]["result_id"] == "A"
    evidence = {"kind": "calculation", "expression": "z", "bindings": {
        "z": {"result_id": "A", "quantity": "camera_z"}}}
    with pytest.raises(EvidenceError, match="calculation_measurement_not_usable"):
        validate_evidence(old["data"]["depth"], evidence, [old], reviews, ["A"])
    new = point(scene, "B", xy=(11, 11))
    new["data"]["replaces_result_id"] = "A"
    observations = [old, new]
    apply_updates(reviews, reviewed([new]), observations)
    assert pending(observations, reviews) == []
    assert reviews["A"]["exclusion"] == "replaced" and reviews["A"]["raw_decision"] == alias
    apply_updates(reviews, reviewed([old]), observations)
    assert reviews["A"]["decision"] == "resample"
    with pytest.raises(EvidenceError, match="calculation_measurement_replaced"):
        validate_evidence(old["data"]["depth"], evidence, observations, reviews, ["A", "B"])
    assert old["data"]["depth"] == point(scene)["data"]["depth"]


def test_unknown_review_and_wrong_surface_cannot_be_usable(scene_data):
    row = point(scene_data[-1])
    reviews, errors = validate_reviews([{"result_id": "A", "decision": "maybe"}], [row], ["A"])
    assert not reviews and errors[0]["code"] == "review_decision_invalid"
    submitted = [{"result_id": "A", "decision": "usable", "observed_surfaces": [
        {"target": "wall", "surface": "center", "decision": "usable", "visibility": "visible"}]}]
    assert validate_reviews(submitted, [row], ["A"])[1][0]["code"] == "review_surface_target_mismatch"


@pytest.mark.parametrize("operator,first,second,relation", [
    ("<", 1, 2, "true"), ("<=", 2, 2, "true"), (">", 1, 2, "false"),
    (">=", 2, 2, "true"), ("==", 2, 2, "true"), ("!=", 1, 2, "true"),
    ("three_way", 1, 2, "less"), ("three_way", 2, 2, "equal"),
    ("three_way", 3, 2, "greater"),
])
def test_tool_comparison_recomputes_all_relations(scene_data, operator, first, second, relation):
    scene = scene_data[-1]
    a, b = point(scene), point(scene, "B", "tv")
    a["data"]["depth"], b["data"]["depth"] = first, second
    rows = [a, b]
    evidence = comparison()
    evidence["operator"] = operator
    if operator == "three_way":
        evidence["answers"] = {"less": "O1", "equal": "O2", "greater": "O3"}
        contract = task("Choose from chair, equal, tv.")
    else:
        contract = CAMERA_TASK
    answer = canonical(evidence["answers"][relation], contract)
    verified = validate_evidence(answer, evidence, rows, reviewed(rows), ["A", "B"], contract)
    assert verified["comparison_result"] == relation
    assert verified["verified_arithmetic"] is True


@pytest.mark.parametrize("mutation,code", [
    ("label", "tool_comparison_answer_mismatch"),
    ("mapping", "answer_option_invalid"),
    ("undelivered", "calculation_measurement_not_delivered"),
    ("unusable", "calculation_measurement_not_usable"),
    ("reference", "task_quantity_mismatch"),
    ("units", "comparison_unit_mismatch"),
    ("threshold", "comparison_unit_mismatch"),
    ("unused", "expression_bindings_unused"),
    ("unsafe", "expression_node_forbidden"),
    ("missing_surface", "measurement_surface_required"),
])
def test_comparison_rejects_invalid_support(scene_data, mutation, code):
    scene = scene_data[-1]
    rows = [point(scene), point(scene, "B", "tv", (20, 20))]
    evidence, reviews, visible, answer = comparison(), reviewed(rows), ["A", "B"], "chair"
    if mutation == "label":
        answer = "tv"
    elif mutation == "mapping":
        evidence["answers"]["false"] = "the tv"
    elif mutation == "undelivered":
        visible.remove("B")
    elif mutation == "unusable":
        reviews["B"]["decision"] = "uncertain"
    elif mutation == "reference":
        evidence["bindings"]["b"]["reference"] = "wall"
    elif mutation == "units":
        rows[1]["data"]["unit"] = "m"
    elif mutation == "threshold":
        evidence["right"] = "2"
        evidence["left"] = "a+b"
    elif mutation == "unused":
        evidence["right"] = "a"
    elif mutation == "unsafe":
        evidence["left"] = "__import__('os').system('true')"
    elif mutation == "missing_surface":
        reviews["A"].pop("observed_surfaces")
    with pytest.raises(EvidenceError, match=code):
        validate_evidence(answer, evidence, rows, reviews, visible, CAMERA_TASK)


def test_normalized_original_and_crop_mapping_share_raw_values(scene_data):
    item, rgb, frame, cache, registry, scene = scene_data
    original = original_descriptor(rgb, frame.shape[1], frame.shape[0])
    scene.delivered_images = {"rgb-original": original}
    fn = registry.get("point_depth").fn
    data = fn(scene, 0, 1, 1, image_id="rgb-original", coordinate_space="normalized",
              targets=[{"target": "chair", "surface": "center"}])
    assert data["pixel_xy"] == [49, 39]
    assert data["depth"] == cache.geometries[0]["depth"][39, 49]
    assert data["coordinate_transforms"][0]["image_sha256"] == original["image_sha256"]
    row = point(scene)
    _, markers = overlay(rgb, [row])
    sheets = crop_sheets(rgb, markers["markers"], ["A"])
    crop = sheets[0][1]["markers"][0]
    scene.delivered_images[crop["image_id"]] = crop
    local = [10-crop["source_box_xyxy"][0], 10-crop["source_box_xyxy"][1]]
    mapped = fn(scene, 0, *local, image_id=crop["image_id"], coordinate_space="image_pixel")
    assert mapped["pixel_xy"] == row["data"]["pixel_xy"]
    assert mapped["depth"] == row["data"]["depth"]
    scene.delivered_measurement_ids = {"A"}
    replacement = fn(scene, 0, *local, image_id=crop["image_id"], coordinate_space="image_pixel",
                     replaces="A")
    assert replacement["replaces_result_id"] == "A"
    with pytest.raises(DomainValueError):
        fn(scene, 0, 1, 1, replaces="UNSEEN")


@pytest.mark.parametrize("width,height", [(1, 1), (101, 37), (37, 101)])
def test_normalized_extreme_pixels(width, height):
    descriptor = original_descriptor(b"synthetic", width, height)
    for xy, expected in [([0, 0], [0, 0]), ([1, 1], [width-1, height-1])]:
        assert resolve(xy, "rgb-original", "normalized", {"rgb-original": descriptor})[0] == expected


@pytest.mark.parametrize("xy,image_id,space", [
    ([1.1, 0], "rgb-original", "normalized"), ([-0.1, 0], "rgb-original", "normalized"),
    ([0, 0], "unknown", "normalized"), ([50, 0], "rgb-original", "image_pixel"),
    ([0, float("nan")], "rgb-original", "normalized"), ([0, 0], "rgb-original", "guessed"),
])
def test_bad_coordinates_fail_before_sampling(scene_data, xy, image_id, space):
    rgb, frame = scene_data[1:3]
    with pytest.raises(DomainValueError):
        resolve(xy, image_id, space, {"rgb-original": original_descriptor(rgb, 50, 40)})


def test_nonzero_crop_offset_and_outside_content():
    crop = {"image_sha256": "crop", "width": 192, "height": 100,
            "content_box_xyxy": [0, 0, 192, 100], "source_box_xyxy": [400, 200, 592, 300]}
    assert resolve([1, 1], "crop-M1", "normalized", {"crop-M1": crop})[0] == [591, 299]
    with pytest.raises(DomainValueError):
        resolve([192, 50], "crop-M1", "image_pixel", {"crop-M1": crop})
    with pytest.raises(DomainValueError):
        resolve([1, 1], "crop-M1", "original_pixel", {"crop-M1": crop})


def test_public_scale_converts_complete_dimensions_and_retains_layers(scene_data):
    scene = scene_data[-1]
    h, hb = dimension(scene, "H", "chair", "height")
    w, wb = dimension(scene, "W", "desk", "width")
    rows = [h, w]
    contract = task("If the desk is 2 m wide, what is the height of the chair?", "float")
    evidence = {"kind": "calculation", "expression": "s*h/w", "bindings": {"h": hb, "w": wb},
                "scale": {"name": "s", "value": 2, "unit": "m", "anchor_binding": "w"}}
    answer = 2*h["data"]["distance"]/w["data"]["distance"]
    checked = validate_evidence(answer, evidence, rows, reviewed(rows), ["H", "W"], contract)
    assert checked["validation_layers"]["task_unit_contract"] is True
    assert checked["validation_layers"]["independent_visual_geometry"] is False
    bad = deepcopy(evidence)
    bad["scale"]["value"] = 3
    with pytest.raises(EvidenceError, match="scale_anchor_invalid"):
        validate_evidence(answer, bad, rows, reviewed(rows), ["H", "W"], contract)
    evidence.pop("scale")
    evidence["expression"] = "2*h/w"
    with pytest.raises(EvidenceError, match="task_unit_mismatch"):
        validate_evidence(answer, evidence, rows, reviewed(rows), ["H", "W"], contract)


@pytest.mark.parametrize("quantity,word,anchors", [
    ("width", "wide", ["left", "right"]),
    ("length", "long", ["front", "back"]),
])
def test_width_length_tasks_use_full_extrema_and_public_span(scene_data, quantity, word, anchors):
    scene = scene_data[-1]
    row, binding = dimension(scene, "D", "desk", quantity)
    contract = task(f"If the desk is 2 m {word}, what is its {quantity}?", "float")
    assert contract["quantity"] == quantity and contract["public_metric_values"][0]["quantity"] == quantity
    evidence = {"kind": "calculation", "expression": "s*d/d", "bindings": {"d": binding},
                "scale": {"name": "s", "value": 2, "unit": "m", "anchor_binding": "d"}}
    assert validate_evidence(2, evidence, [row], reviewed([row]), ["D"], contract)["verified_arithmetic"]
    evidence["bindings"]["d"]["geometry"]["anchors"] = list(reversed(anchors))
    with pytest.raises(EvidenceError, match="complete_dimension_not_declared"):
        validate_evidence(2, evidence, [row], reviewed([row]), ["D"], contract)


@pytest.mark.parametrize("question", [
    "How far would you walk if the pillow is 2.2 m long?",
    "If the chair moves 2 m, would it hit the desk?",
    "What is the height of the chair?",
])
def test_camera_z_cannot_establish_other_task_quantities(scene_data, question):
    row = point(scene_data[-1])
    evidence = {"kind": "calculation", "expression": "z", "bindings": {
        "z": {"result_id": "A", "quantity": "camera_z"}}}
    contract = task(question, "float", options=["yes", "no"] if "hit" in question else [])
    answer = "yes" if "hit" in question else row["data"]["depth"]
    with pytest.raises(EvidenceError, match="task_quantity_mismatch"):
        validate_evidence(answer, evidence, [row], reviewed([row]), ["A"], contract)


@pytest.mark.parametrize("visibility", ["cropped", "occluded", "unclear"])
def test_incomplete_extrema_cannot_be_declared_full_height(scene_data, visibility):
    row, binding = dimension(scene_data[-1], "H", "chair", "height")
    reviews = reviewed([row])
    reviews["H"]["observed_surfaces"][0]["visibility"] = visibility
    evidence = {"kind": "calculation", "expression": "h", "bindings": {"h": binding}}
    with pytest.raises(EvidenceError, match="complete_dimension_not_declared"):
        validate_evidence(row["data"]["distance"], evidence, [row], reviews, ["H"],
                          task("What is the height of the chair?", "float"))


def test_height_ratio_is_continuous_and_selected_pixel_needs_own_review(scene_data):
    scene = scene_data[-1]
    a, ab = dimension(scene, "A", "chair", "height")
    b, bb = dimension(scene, "B", "desk", "height")
    a["data"]["distance"], b["data"]["distance"] = 1.25, 2.75
    rows = [a, b]
    evidence = {"kind": "calculation", "expression": "a/b", "bindings": {"a": ab, "b": bb}}
    ratio_task = task("What is the height ratio of the chair and desk?", "float")
    assert validate_evidence(1.25/2.75, evidence, rows, reviewed(rows), ["A", "B"], ratio_task)[
        "calculated_value"] == 1.25/2.75
    with pytest.raises(EvidenceError, match="answer_calculation_mismatch"):
        validate_evidence(1, evidence, rows, reviewed(rows), ["A", "B"], ratio_task)
    rows = [point(scene), point(scene, "B", "tv", (20, 20))]
    evidence = comparison()
    evidence["bindings"]["a"]["sample_xy"] = [11, 10]
    with pytest.raises(EvidenceError, match="measurement_surface_required"):
        validate_evidence("chair", evidence, rows, reviewed(rows), ["A", "B"], CAMERA_TASK)
    reviews = reviewed(rows)
    reviews["A"]["observed_surfaces"][0]["sample_xy"] = [11, 10]
    assert validate_evidence("chair", evidence, rows, reviews, ["A", "B"], CAMERA_TASK)[
        "comparison_result"] == "true"


def test_object_distance_requires_common_reference(scene_data):
    scene = scene_data[-1]
    rows = []
    for key, target in [("A", "tv"), ("B", "chair")]:
        row, _ = dimension(scene, key, target, "height")
        row["data"]["intended_targets"] = [
            {"target": target, "surface": "center"}, {"target": "armchair", "surface": "center"}]
        rows.append(row)
    rows[1]["data"]["distance"] *= 2
    evidence = comparison()
    evidence["bindings"] = {k.lower(): {"result_id": k, "quantity": "chord", "reference": "armchair"}
                            for k in ("A", "B")}
    contract = task("Which is closer to the armchair: tv or chair?")
    assert validate_evidence("tv", evidence, rows, reviewed(rows), ["A", "B"], contract)[
        "comparison_result"] == "true"
    evidence["bindings"]["b"]["reference"] = "wall"
    with pytest.raises(EvidenceError, match="task_quantity_mismatch"):
        validate_evidence("tv", evidence, rows, reviewed(rows), ["A", "B"], contract)


def entity(key, candidates, relationship="single", included=True):
    return {"entity_id": key, "category": "chair", "candidate_ids": candidates,
            "relationship": relationship, "basis": "RGB shows a distinct chair", "included": included}


def scope(entities):
    return {"include": "physical chairs", "exclude": "reflections and parts", "entities": entities}


def test_high_iou_candidates_and_unconfirmed_categories_remain_clues(scene_data):
    cache = scene_data[3]
    cache.candidates = [
        {"candidate_id": "C1", "view_index": 0, "category": "chair", "box_xyxy": [0, 0, 100, 100],
         "category_reviewed": True, "category_confirmed": True, "review": {"identity": "intern"}},
        {"candidate_id": "C2", "view_index": 0, "category": "chair", "box_xyxy": [0, 0, 100, 76.9],
         "category_reviewed": True, "category_confirmed": False, "review_status": "not_established"}]
    before = deepcopy(cache.candidates)
    facts = candidate_facts(cache, 0)
    assert len(facts["candidates"]) == 2 and iou(*[r["box_xyxy"] for r in before]) > .7
    assert facts["overlap_groups"][0]["relationship"] == "unresolved"
    assert facts["candidates"][0]["lineage"]["nms"]["status"] == "not_recorded_in_cache"
    assert facts["candidates"][1]["category_confirmed"] is False
    assert facts["exhaustive"] is False and cache.candidates == before
    observations = [{"result_id": "V", "source_tool": "view_candidates", "data": facts}]
    # Two genuinely separate RGB entities remain separate even above an NMS threshold.
    assert validate_scope(scope([entity("E1", ["C1"]), entity("E2", ["C2"])]),
                          observations, ["V"], 2)["counts"] == {"chair": 2}
    # A specific same-object RGB declaration can instead describe a duplicate/part.
    duplicate = entity("E", ["C1", "C2"], "same_object")
    duplicate["basis"] = "RGB: same chair backrest, legs and seat in both boxes"
    assert validate_scope(scope([duplicate]), observations, ["V"], 1)["counts"] == {"chair": 1}
    duplicate["basis"] = "overlap"
    with pytest.raises(ValueError, match="object_scope_merge_requires_rgb"):
        validate_scope(scope([duplicate]), observations, ["V"], 1)
    with pytest.raises(ValueError, match="object_scope_identity_invalid"):
        validate_scope(scope([entity("E1", ["C1"]), entity("E2", ["C1"])]), observations, ["V"], 2)


def test_count_scope_and_volume_assumptions_are_required_without_tools(scene_data):
    count_task = task("How many chairs are present?", "int")
    with pytest.raises(EvidenceError, match="answer_evidence_required"):
        validate_evidence(1, None, [], {}, [], count_task)
    evidence = {"kind": "visual_estimate", "basis": "RGB count", "object_scope": scope([
        entity("E", [], "rgb_only")])}
    assert validate_evidence(1, evidence, [], {}, [], count_task)["scope_validation"]["counts"] == {"chair": 1}
    with pytest.raises(EvidenceError, match="object_scope_count_mismatch"):
        validate_evidence(2, evidence, [], {}, [], count_task)
    volume_task = task("What is the volume ratio of the sofa and chair?", "float")
    for missing in (None, {"kind": "visual_estimate", "basis": "RGB estimate"}):
        with pytest.raises(EvidenceError):
            validate_evidence(12.5, missing, [], {}, [], volume_task)
    estimate = {"kind": "visual_estimate", "basis": "RGB bounding shapes", "assumptions": {
        "object_scope": "sofa includes its footstool; chair excludes side table",
        "volume_model": "bounding cuboids for both targets"}}
    assert validate_evidence(12.5, estimate, [], {}, [], volume_task)["verified_arithmetic"] is False


def test_tool_volume_formula_records_fixed_scope_and_shape(scene_data):
    scene = scene_data[-1]
    rows, bindings = [], {}
    for target, prefix in (("sofa", "a"), ("chair", "b")):
        for quantity in ("height", "width", "length"):
            key = prefix + quantity[0]
            row, binding = dimension(scene, key, target, quantity)
            rows.append(row)
            bindings[key] = binding
    evidence = {"kind": "calculation", "bindings": bindings,
                "expression": "ah*aw*al/(bh*bw*bl)",
                "assumptions": {"object_scope": "one sofa and one chair, excluding footstool",
                                "volume_model": "full bounding cuboids"}}
    contract = task("What is the volume ratio of the sofa and chair?", "float")
    ids = [r["result_id"] for r in rows]
    checked = validate_evidence(1.0, evidence, rows, reviewed(rows), ids, contract)
    assert checked["assumptions"] == evidence["assumptions"] and checked["verified_arithmetic"]
    evidence.pop("assumptions")
    with pytest.raises(EvidenceError, match="object_scope_invalid"):
        validate_evidence(1.0, evidence, rows, reviewed(rows), ids, contract)


def test_unshown_candidate_ids_cannot_enter_scope(scene_data):
    candidates = [{"candidate_id": f"C{n}", "category": "chair", "box_xyxy": [0, 0, 1, 1],
                   "view_index": 0} for n in range(10)]
    observation = {"result_id": "V", "source_tool": "view_candidates",
                   "data": {"candidates": candidates}}
    with pytest.raises(ValueError, match="object_scope_identity_invalid"):
        validate_scope(scope([entity("E", ["C9"])]), [observation], ["V"], 1)


def test_repeated_measurement_and_notes_do_not_create_progress(scene_data):
    old = point(scene_data[-1])
    repeated = deepcopy(old)
    repeated["result_id"] = "B"
    repeated["data"]["coordinate_transforms"] = [{"image_id": "crop-M1"}]
    repeated["data"]["replaces_result_id"] = "A"
    assert record([old], [old, repeated], {}, {})["made_progress"] is False
    changed = record([old], [old], {}, {"A": {"result_id": "A", "decision": "resample"}})
    assert changed["made_progress"] is True and changed["changed_review_ids"] == ["A"]
    assert record([old], [old], {}, {}, answered=True)["answer_contract_passed"] is True
    bulky = {**changed, "new_observation_ids": [f"NEW{n}" for n in range(10000)],
             "novel_observation_ids": [f"NEW{n}" for n in range(10000)]}
    assert len(json.dumps(summary(bulky))) < 200
    item, rgb, frame, cache = scene_data[:4]
    context, receipt = messages(item, rgb, frame, cache, arm="skill", config=Config(), round_index=5,
                                observations=[old], progress=[bulky], note="unverified model hypothesis")
    text = "\n".join(p.get("text", "") for p in context[0]["content"])
    assert receipt["text_chars"] <= Config().text_caps[-1] and "NEW9999" not in text
    assert "MODEL HYPOTHESIS (model_note_unverified; not a tool fact)" in text
    assert "finish from usable evidence or RGB" in text


def test_final_retains_metric_dependencies_with_large_progress_history(scene_data):
    item, rgb, frame, cache, registry, scene = scene_data
    item = {**item, "question": "If the desk is 2 m wide, what is the height of the chair?",
            "task_type": "float"}
    rows = [dimension(scene, "H", "chair", "height")[0],
            dimension(scene, "W", "desk", "width")[0]]
    progress = record([], rows, {}, reviewed(rows))
    progress["new_observation_ids"] = [f"NEW{n}" for n in range(10000)]
    progress["novel_observation_ids"] = progress["new_observation_ids"]
    _, receipt = messages(item, rgb, frame, cache, arm="skill", config=Config(), round_index=5,
                          observations=rows, reviews=reviewed(rows), dependency_ids=["H", "W"],
                          progress=[progress])
    assert receipt["text_chars"] <= Config().text_caps[-1]
    assert receipt["visible_measurement_ids"] == ["H", "W"]
    assert receipt["context_packing"]["omitted_dependency_ids"] == []


class OfflineClient:
    def __init__(self, outputs):
        self.outputs, self.count = list(outputs), 0
        self.last_usage = {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20}
        self.last_response_identity = {"model": Config().model, "model_reported": True}
        self.last_generation = {}
        self.last_reasoning_content = None
        self.last_service_error = {}

    def chat(self, context, **kwargs):
        assert kwargs == Config().kwargs()
        text = "\n".join(p.get("text", "") for p in context[0]["content"])
        facts = json.loads(text.split("\nTOOL FACTS ", 1)[1].split("\n", 1)[0])
        output = self.outputs[self.count]
        self.count += 1
        return output(facts) if callable(output) else output


class OfflineBudget:
    @contextmanager
    def request(self):
        yield {"usage": {}}


def offline_episode(tmp_path, monkeypatch, scene_data, item, outputs):
    rgb, frame, cache = scene_data[1:4]
    monkeypatch.setattr(worker, "load", lambda *args, **kwargs: (rgb, frame, cache))
    client = OfflineClient(outputs)
    job = {"item": item, "arm": "skill", "output_directory": str(tmp_path),
           "run_identity": "offline-contracts", "stop_file": str(tmp_path / "STOP"),
           "initial_receipt": messages(item, rgb, frame, cache, arm="skill", config=Config())[1]}
    result = worker.solve(job, client, OfflineBudget())
    assert not (tmp_path / "STOP").exists()
    stored = read(tmp_path / "result.json")
    assert digest({k: v for k, v in stored.items() if k != "receipt_sha256"}) == stored["receipt_sha256"]
    replayed = dict(replay_contexts(item, stored, rgb, frame, cache))
    for index, payload in replayed.items():
        assert payload == read(tmp_path / f"{index:02d}.messages.json")
    assert client.count == result["request_count"] <= 6
    return stored


def test_worker_maps_option_id_and_replay_detects_contract_tampering(tmp_path, monkeypatch, scene_data):
    item = {**scene_data[0], "question": "Which object is closer to the camera: tv or chair?",
            "task_type": "str"}
    result = offline_episode(tmp_path, monkeypatch, scene_data, item, [fenced('ReturnAnswer("O1")')])
    assert result["raw_answer"] == "O1" and result["answer"] == "tv"
    assert result["usage"] == {"prompt_tokens": 13, "completion_tokens": 7, "total_tokens": 20}
    changed = deepcopy(result)
    changed["public_answer_contract"]["allowed_answers"][0]["label"] = "the tv"
    with pytest.raises(ValueError, match="episode_public_contract_or_progress_changed"):
        list(replay_contexts(item, changed, *scene_data[1:4]))
    changed = deepcopy(result)
    changed["raw_answer"] = "the tv"
    with pytest.raises(ValueError, match="episode_answer_evidence_changed"):
        list(replay_contexts(item, changed, *scene_data[1:4]))
    changed = deepcopy(result)
    changed["evidence_activity"]["no_progress_rounds"] += 1
    with pytest.raises(ValueError, match="episode_public_contract_or_progress_changed"):
        list(replay_contexts(item, changed, *scene_data[1:4]))


def test_worker_repairs_illegal_article_from_public_options(tmp_path, monkeypatch, scene_data):
    item = {**scene_data[0], "question": "Choose from tv, chair.", "task_type": "str"}
    result = offline_episode(tmp_path, monkeypatch, scene_data, item, [
        fenced('ReturnAnswer("the tv")'), fenced('ReturnAnswer("O1")')])
    assert result["answer"] == "tv" and result["raw_answer"] == "O1"
    assert result["rounds"][0]["answer_feedback_errors"][0]["code"] == "answer_option_invalid"
    assert result["progress"][0]["made_progress"] is False
    assert result["evidence_activity"]["no_progress_rounds"] == 1


def test_worker_wrong_point_replacement_and_tool_comparison_replay(tmp_path, monkeypatch, scene_data):
    item = {**scene_data[0], "question": "Which object is closer to the camera: chair or tv?",
            "task_type": "str"}
    sample = fenced(
        'r=ctx.tools.point_depth(view_index=0,x=10,y=10,targets=[{"target":"chair","surface":"center"}])\n'
        'YieldObservations([r["result_id"]],reason="Review chair point")')

    def replace(facts):
        key = facts[0]["result_id"]
        return fenced(f'measurement_reviews = [{{"result_id":{key!r},"decision":"wrong target"}}]\n'
                      f'a=ctx.tools.point_depth(view_index=0,x=11,y=10,targets=[{{"target":"chair","surface":"center"}}],replaces={key!r})\n'
                      'b=ctx.tools.point_depth(view_index=0,x=20,y=20,targets=[{"target":"tv","surface":"center"}])\n'
                      'YieldObservations([a["result_id"],b["result_id"]],reason="Review replacements")')

    def finish(facts):
        rows = [r for r in facts if r["data"]["pixel_xy"] != [10, 10]]
        ids = {r["data"]["intended_targets"][0]["target"]: r["result_id"] for r in rows}
        evidence = comparison()
        evidence["bindings"]["a"]["result_id"] = ids["chair"]
        evidence["bindings"]["b"]["result_id"] = ids["tv"]
        updates = reviewed(rows)
        return fenced(f'measurement_reviews = {list(updates.values())!r}\n'
                      f'ReturnAnswer("O1", evidence={evidence!r})')

    result = offline_episode(tmp_path, monkeypatch, scene_data, item, [sample, replace, finish])
    assert result["status"] == "answered" and result["answer"] == "chair"
    assert result["answer_evidence"]["kind"] == "tool_comparison"
    old = next(r for r in result["measurement_reviews"] if r.get("exclusion") == "replaced")
    assert old["raw_decision"] == "wrong target" and old["decision"] == "resample"
    assert old["result_id"] not in result["pending_measurement_ids"]
    assert old["result_id"] not in result["answer_evidence"]["result_ids"]
    assert result["evidence_activity"]["replacement_measurements"] == 1
    assert result["evidence_activity"]["usable_novel_replacements"] == 1
    assert result["evidence_activity"]["answer_contract_passes"] == 1
    changed = deepcopy(result)
    changed["rounds"][1]["progress"]["novel_observation_ids"] = []
    with pytest.raises(ValueError, match="round_progress_changed"):
        list(replay_contexts(item, changed, *scene_data[1:4]))
