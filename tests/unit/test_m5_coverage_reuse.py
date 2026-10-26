"""Coverage reuse is local to one call and never keyed by object ID or status."""

import copy

import pytest

from test_question_binding import harness
from skill3d.schemas.reconstruction import ObjectRecord, SceneState
from skill3d.segmentation import binding_validation as bv, question_binding as qb
from skill3d.tools.scene_handle import SceneHandle


def count_validations(monkeypatch):
    calls = []
    original = bv.validate_binding

    def counted(obj, **kwargs):
        calls.append((obj, kwargs))
        return original(obj, **kwargs)

    monkeypatch.setattr(bv, "validate_binding", counted)
    return calls


def test_refresh_coverage_evaluates_each_entry_once_for_all_requirements(harness, monkeypatch):
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    changed = objects[0].model_copy(update={"centroid_world": [0., 0., 0.]})
    entries = [objects[0], changed]
    grounding = copy.deepcopy(stats["grounding"])
    grounding["requirements"] *= 3
    calls = count_validations(monkeypatch)
    refreshed = qb.refresh_coverage(grounding, entries)
    assert len(calls) == len(entries)
    assert refreshed["n_new"] == 1
    for row in refreshed["requirements"]:
        assert row["matched_object_ids"] == [obj.obj_id for obj in entries]
        assert row["usable_object_ids"] == [objects[0].obj_id]
    assert grounding == stats["grounding"] | {
        "requirements": stats["grounding"]["requirements"] * 3}


@pytest.mark.parametrize("requirements", [[], [{"name": "unmatched", "scan_complete": True}]])
def test_unmatched_objects_still_count_once(harness, monkeypatch, requirements):
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    calls = count_validations(monkeypatch)
    coverage = qb.refresh_coverage({"requirements": requirements}, objects)
    assert coverage["n_new"] == 1
    assert coverage["status"] == "unavailable"
    assert len(calls) == 1


def test_refresh_has_no_cross_call_cache_and_tools_revalidate(harness, monkeypatch):
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    obj = objects[0]
    calls = count_validations(monkeypatch)
    assert qb.refresh_coverage(stats["grounding"], objects)["n_new"] == 1
    assert qb.refresh_coverage(stats["grounding"], objects)["n_new"] == 1
    assert len(calls) == 2
    handle = SceneHandle(
        SceneState(artifact_ref="synthetic"), objects=objects,
        objects_materialized=True, require_question_binding=True)
    assert handle.binding_usable(obj.obj_id)
    assert len(calls) == 3
    obj.binding_provenance[0]["semantic_observation"]["mask_purity"] = "mixed"
    assert obj.binding_validation.status == "validated"
    assert qb.refresh_coverage(stats["grounding"], objects)["n_new"] == 0
    assert not handle.binding_usable(obj.obj_id)
    assert len(calls) == 5


def test_bind_explanation_uses_computed_count_without_extra_validation(harness, monkeypatch):
    calls = []
    original = bv.validate_binding

    def counted(obj, **kwargs):
        calls.append((obj, kwargs))
        return original(obj, **kwargs)

    monkeypatch.setattr(bv, "validate_binding", counted)
    monkeypatch.setattr(qb, "validate_binding", counted)
    objects, stats, notes = qb.bind_question_objects(**harness.args)
    assert len(objects) == 1
    candidate_calls = [row for row in calls if "/candidates/" in row[0].mask_per_frame]
    final_calls = [row for row in calls if "/objects/" in row[0].mask_per_frame]
    assert len(candidate_calls) == 2
    assert len(final_calls) == 2
    assert f"候选→{stats['grounding']['n_new']} 可用实例" in notes[0]
    assert sum(bool(kwargs.get("check_receipt")) for _, kwargs in final_calls) == 1


def test_legacy_flags_and_blocking_scan_semantics_are_preserved():
    objects = [
        ObjectRecord(obj_id="base", category_name="chair",
                     binding_provenance=[{"source": "base_list", "source_label": "chair"}]),
        ObjectRecord(obj_id="flagged", category_name="chair",
                     binding_flags=[{"reason": "unavailable"}]),
    ]
    coverage = qb.refresh_coverage({"requirements": [
        {"name": "chair", "scan_complete": True,
         "binding_failures": [{"blocking": False}, {"blocking": True}]},
        {"name": "chair", "scan_complete": False},
    ], "all_present": True, "miss": []}, objects)
    assert coverage["n_new"] == 1
    assert coverage["status"] == "unavailable"
    assert all(row["usable_object_ids"] == ["base"] for row in coverage["requirements"])
    assert coverage["requirements"][0]["blocking_failures"] == [{"blocking": True}]
    assert "all_present" not in coverage and "miss" not in coverage
