"""Offline surface/extent-role regressions using synthetic measurement records."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import socket
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from Ominirunner.coordinates import extent_requirement, intent, surface_name
from Ominirunner.review import apply_updates, pending, validate, validate_program
from skill3d.tools.contract import DomainValueError


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("live transport forbidden in surface-role tests")
    monkeypatch.setattr(socket.socket, "connect", denied)


def measurement(key="A", surfaces=("top", "base"), target="fireplace", replaces=None):
    data = {"intended_targets": [{"target": target, "surface": name} for name in surfaces],
            "replaces_result_id": replaces}
    if len(surfaces) == 1:
        tool = "point_depth"
        data["pixel_xy"] = [10, 10]
    else:
        tool = "pair_point_distance"
        data.update(a_xy=[10, 10], b_xy=[20, 20])
    return {"result_id": key, "source_tool": tool, "data": data}


def review(row, decision="usable"):
    return {"result_id": row["result_id"], "decision": decision, "observed_surfaces": [
        {**target, "decision": decision, "visibility": "visible"}
        for target in row["data"]["intended_targets"]]}


def declared(surface, role, basis="object_vertical"):
    return {"target": "fireplace", "observed_surface": surface, "extent_role": role,
            "extent_basis": basis, "scope_completeness": "complete",
            "decision": "usable", "visibility": "visible", "issues": []}


def test_sampling_intent_keeps_legacy_and_accepts_separate_roles_without_mutation():
    values = [{"target": "desk", "surface": "face_left"},
              {"target": "desk", "surface": "front", "extent_role": "left",
               "extent_basis": "object_horizontal"}]
    original = deepcopy(values)
    assert intent(values, 2) == [{**values[0], "surface": "left"}, values[1]]
    assert values == original
    assert intent(None, 1) is None


@pytest.mark.parametrize("bad", [
    [], {}, [{"target": "desk"}], [{"target": "desk", "surface": "front", "extra": "x"}],
    [{"target": "desk", "surface": "front", "extent_role": "mantel_top"}],
    [{"target": "desk", "surface": "front", "extent_role": None}],
    [{"target": "desk", "surface": "front", "extent_basis": ""}],
    [{"target": "desk", "surface": "front", "extent_basis": " " * 3}],
    [{"target": "desk", "surface": "front", "extent_basis": "x" * 81}],
])
def test_sampling_intent_rejects_invalid_optional_fields(bad):
    with pytest.raises(DomainValueError):
        intent(bad, 1)


@pytest.mark.parametrize("name,role", [
    ("top", "top"), ("base", "base"), ("left", "left"), ("right", "right"),
    ("face_left", "left"), ("face_right", "right"),
    ("mantel_top", None), ("front", None), ("back", None),
])
def test_only_legacy_canonical_surfaces_imply_roles(name, role):
    assert extent_requirement({"surface": name}) == (role, None)
    assert surface_name("mantel_top") == "mantel_top"
    assert surface_name("back") == "back"


@pytest.mark.parametrize("legacy,physical,role,basis", [
    ("top", "mantel_top", "top", "object_vertical"),
    ("left", "front", "left", "object_horizontal"),
    ("left", "back", "left", "object_horizontal"),
])
def test_legacy_endpoint_migration_keeps_actual_physical_surface(legacy, physical, role, basis):
    row = measurement(surfaces=(legacy,))
    submitted = review(row)
    submitted["observed_surfaces"] = [declared(physical, role, basis)]
    original = deepcopy(submitted)
    accepted, errors = validate([submitted], [row], ["A"])
    assert errors == []
    normalized = accepted["A"]["observed_surfaces"][0]
    assert normalized["observed_surface"] == normalized["surface"] == physical
    assert normalized["extent_role"] == role and normalized["extent_basis"] == basis
    assert submitted == original
    assert validate([accepted["A"]], [row], ["A"]) == (accepted, [])


@pytest.mark.parametrize("legacy,physical", [("top", "mantel_top"), ("left", "front"), ("left", "back")])
def test_physical_name_alone_cannot_establish_a_legacy_role(legacy, physical):
    row = measurement(surfaces=(legacy,))
    submitted = review(row)
    submitted["observed_surfaces"][0]["surface"] = physical
    accepted, errors = validate([submitted], [row], ["A"])
    assert accepted == {}
    assert errors[0]["field"] == "observed_surfaces[0].surface"
    assert errors[0]["expected_target"] == "fireplace"
    assert errors[0]["intended_surface"] == legacy
    assert errors[0]["observed_surface"] == physical
    assert errors[0]["required_role"] == legacy
    assert errors[0]["measurement_result_id"] == "A" and errors[0]["point_index"] == 0
    assert errors[0]["index"] == 0


@pytest.mark.parametrize("missing", ["extent_role", "extent_basis", "scope_completeness"])
def test_migration_requires_explicit_role_basis_and_scope(missing):
    row = measurement(surfaces=("top",))
    submitted = review(row)
    surface = declared("mantel_top", "top")
    surface.pop(missing)
    submitted["observed_surfaces"] = [surface]
    accepted, errors = validate([submitted], [row], ["A"])
    assert accepted == {} and errors[0]["field"].startswith("observed_surfaces[0].")


def test_new_sampling_intent_checks_actual_surface_then_role_and_basis():
    row = measurement(surfaces=("front",))
    row["data"]["intended_targets"] = intent([
        {"target": "fireplace", "surface": "front", "extent_role": "left",
         "extent_basis": "object_horizontal"}], 1)
    submitted = review(row)
    submitted["observed_surfaces"] = [declared("front", "left", "object_horizontal")]
    accepted, errors = validate([submitted], [row], ["A"])
    assert errors == [] and accepted["A"]["observed_surfaces"][0]["surface"] == "front"
    for field, bad in (("observed_surface", "back"), ("extent_role", "right"),
                       ("target", "wall")):
        changed = deepcopy(submitted)
        changed["observed_surfaces"][0][field] = bad
        accepted, errors = validate([changed], [row], ["A"])
        assert accepted == {} and errors[0]["field"] == f"observed_surfaces[0].{field}"
        assert errors[0]["expected_target"] == "fireplace"
        assert errors[0]["intended_surface"] == "front"
        assert errors[0]["required_role"] == "left"
        assert errors[0]["required_basis"] == "object_horizontal"


def test_pair_role_validation_keeps_endpoint_order_and_identifies_second_point():
    row = measurement()
    submitted = review(row)
    submitted["observed_surfaces"] = [declared("mantel_top", "top"), declared("foot", "base")]
    assert validate([submitted], [row], ["A"])[1] == []
    submitted["observed_surfaces"][1]["extent_role"] = "top"
    accepted, errors = validate([submitted], [row], ["A"])
    assert not accepted
    assert errors[0]["field"] == "observed_surfaces[1].extent_role"
    assert errors[0]["point_index"] == 1 and errors[0]["required_role"] == "base"
    assert errors[0]["actual"] == "top" and errors[0]["expected"] == "base"


def test_diameter_rim_roles_are_explicit_and_do_not_change_physical_surfaces():
    row = measurement(surfaces=("near_rim", "far_rim"))
    row["data"]["intended_targets"] = intent([
        {"target": "fireplace", "surface": surface, "extent_role": role,
         "extent_basis": "object_diameter"}
        for surface, role in zip(("near_rim", "far_rim"), ("rim_a", "rim_b"))], 2)
    submitted = review(row)
    submitted["observed_surfaces"] = [
        declared(surface, role, "object_diameter")
        for surface, role in zip(("near_rim", "far_rim"), ("rim_a", "rim_b"))]
    accepted, errors = validate([submitted], [row], ["A"])
    assert errors == []
    assert [surface["extent_role"] for surface in accepted["A"]["observed_surfaces"]] == ["rim_a", "rim_b"]
    assert [surface["observed_surface"] for surface in accepted["A"]["observed_surfaces"]] == [
        "near_rim", "far_rim"]
    assert extent_requirement({"surface": "rim_a"}) == (None, None)


@pytest.mark.parametrize("visibility", ["cropped", "occluded", "unclear"])
def test_unusable_visibility_is_rejected_even_without_extent_roles(visibility):
    row = measurement(surfaces=("center",))
    submitted = review(row)
    submitted["observed_surfaces"][0]["visibility"] = visibility
    accepted, errors = validate([submitted], [row], ["A"])
    assert not accepted and errors[0]["field"] == "observed_surfaces[0].visibility"
    submitted["decision"] = submitted["observed_surfaces"][0]["decision"] = "uncertain"
    assert validate([submitted], [row], ["A"])[1] == []


@pytest.mark.parametrize("field,bad", [
    ("scope_completeness", "incomplete"), ("scope_completeness", "unclear"),
    ("issues", ["possibly another object"]), ("target", "wall"),
])
def test_explicit_visual_doubts_or_wrong_object_cannot_be_usable(field, bad):
    row = measurement(surfaces=("top",))
    submitted = review(row)
    submitted["observed_surfaces"] = [declared("mantel_top", "top")]
    submitted["observed_surfaces"][0][field] = bad
    accepted, errors = validate([submitted], [row], ["A"])
    assert not accepted and errors[0]["field"] == f"observed_surfaces[0].{field}"
    submitted["decision"] = submitted["observed_surfaces"][0]["decision"] = "resample"
    accepted, errors = validate([submitted], [row], ["A"])
    assert not errors and accepted["A"]["observed_surfaces"][0][field] == bad


def test_measurement_level_issues_cannot_hide_behind_usable_surfaces():
    row = measurement(surfaces=("center",))
    submitted = review(row)
    submitted["issues"] = ["target identity remains uncertain"]
    assert validate([submitted], [row], ["A"])[1][0]["field"] == "issues"
    submitted["decision"] = "uncertain"
    accepted, errors = validate([submitted], [row], ["A"])
    assert errors == [] and accepted["A"]["issues"] == submitted["issues"]


@pytest.mark.parametrize("field,bad", [
    ("observed_surface", " "), ("extent_role", []), ("extent_role", "mantel_top"),
    ("extent_basis", False), ("scope_completeness", "partial"),
    ("issues", "unclear"), ("issues", [""]), ("issues", [None]),
    ("issues", ["x" * 201]), ("issues", ["doubt"] * 17),
    ("sample_xy", [True, 10]),
])
def test_invalid_surface_fields_are_located(field, bad):
    row = measurement(surfaces=("top",))
    submitted = review(row)
    submitted["observed_surfaces"] = [declared("mantel_top", "top")]
    submitted["observed_surfaces"][0][field] = bad
    accepted, errors = validate([submitted], [row], ["A"])
    assert not accepted and errors[0]["field"] == f"observed_surfaces[0].{field}"
    assert errors[0]["measurement_result_id"] == "A" and errors[0]["point_index"] == 0


def test_legacy_surface_alias_is_preserved_and_cannot_disagree_with_observed_surface():
    row = measurement(surfaces=("left",))
    submitted = review(row)
    submitted["observed_surfaces"][0]["surface"] = "face_left"
    accepted, errors = validate([submitted], [row], ["A"])
    assert errors == []
    normalized = accepted["A"]["observed_surfaces"][0]
    assert normalized["surface"] == normalized["observed_surface"] == normalized["extent_role"] == "left"
    assert normalized["raw_surface"] == "face_left"
    submitted["observed_surfaces"][0].update(surface="top", observed_surface="mantel_top",
                                          extent_role="top")
    accepted, errors = validate([submitted], [row], ["A"])
    assert not accepted and errors[0]["field"] == "observed_surfaces[0].surface"


def test_replacement_requires_own_delivered_visible_review_and_keeps_apply_updates_behavior():
    old = measurement("A", surfaces=("center",))
    new = measurement("B", surfaces=("center",), replaces="A")
    observations = [old, new]
    reviews, errors = validate([review(old)], [old], ["A"])
    assert errors == []
    accepted, errors = validate([{"result_id": "B", "decision": "usable"}], observations, ["B"])
    assert not accepted and errors[0]["code"] == "review_replacement_surfaces_required"
    accepted, errors = validate([review(new)], observations, ["A"])
    assert not accepted and errors[0]["code"] == "review_not_delivered"
    bad = review(new)
    bad["observed_surfaces"][0]["visibility"] = "occluded"
    assert validate([bad], observations, ["B"])[0] == {}
    apply_updates(reviews, {}, observations)
    assert reviews["A"]["decision"] == "resample" and "B" not in reviews
    assert pending(observations, reviews) == ["B"]
    accepted, errors = validate([review(new)], observations, ["B"])
    assert errors == []
    apply_updates(reviews, accepted, observations)
    assert reviews["B"]["decision"] == "usable"
    assert reviews["A"]["exclusion"] == "replaced" and pending(observations, reviews) == []
    accepted, errors = validate([review(old)], observations, ["A"])
    assert errors == []
    apply_updates(reviews, accepted, observations)
    assert reviews["A"]["decision"] == "resample" and reviews["A"]["replaced_by_result_id"] == "B"
    assert validate([{"result_id": "A", "decision": "usable"}], [old], ["A"])[1] == []


def test_record_failures_are_independent_and_literal_program_supports_new_schema():
    a = measurement("A", surfaces=("top",))
    b = measurement("B", surfaces=("top",))
    bad, good = review(a), review(b)
    bad["observed_surfaces"][0]["surface"] = "mantel_top"
    good["observed_surfaces"] = [declared("mantel_top", "top")]
    program = f"def solve(ctx):\n    measurement_reviews = {[bad, good]!r}\n"
    accepted, errors = validate_program(program, [a, b], ["A", "B"])
    assert list(accepted) == ["B"]
    assert len(errors) == 1 and errors[0]["index"] == 0
    assert errors[0]["measurement_result_id"] == "A"
