"""Deterministic M11 replay of declarative answer derivations."""

import json

import pytest

from skill3d.schemas import AnswerPayload, ToolResult
from skill3d.verifier.derivation import replay_derivation


def _result(result_id, value, *, tool="count_objects", raw_value=None, **overrides):
    values = {
        "result_id": result_id,
        "source_tool": tool,
        "tool": tool,
        "status": "ok",
        "value": json.dumps(value) if raw_value is None else raw_value,
        "request_digest": f"digest-{result_id}",
    }
    values.update(overrides)
    return ToolResult(**values)


def _payload(value, unit, op, input_ids, parameters):
    return AnswerPayload(
        value=value,
        unit=unit,
        basis="tool_derived",
        used_result_ids=list(input_ids),
        derivation={
            "op": op,
            "input_result_ids": list(input_ids),
            "parameters": parameters,
        },
    )


@pytest.mark.parametrize(("payload", "results", "question_type", "computed"), [
    (
        _payload(3, "count", "field", ["r1"], {"field": "count"}),
        [_result("r1", {"count": 3})],
        "object_counting",
        3,
    ),
    (
        _payload(250, "cm", "convert", ["r1"], {
            "field": "extent_longest_metric", "from_unit": "m", "to_unit": "cm",
        }),
        [_result("r1", {"extent_longest_metric": 2.5}, tool="object_3d_extent")],
        "object_size_estimation",
        250.0,
    ),
    (
        _payload(2, "count", "count", ["r1"], {"field": "items"}),
        [_result("r1", {"items": [{"id": 1}, {"id": 2}]}, tool="list_objects")],
        "object_counting",
        2,
    ),
    (
        _payload('["B", "A"]', "option", "sort", ["r1", "r2"], {
            "labels": ["A", "B"], "descending": False,
        }),
        [
            _result("r1", 8, tool="object_visible_frames"),
            _result("r2", 3, tool="object_visible_frames"),
        ],
        "obj_appearance_order",
        ["B", "A"],
    ),
    (
        _payload("B", "option", "argmin", ["r1"], {}),
        [_result("r1", {"A": 2.0, "B": 1.0}, tool="relative_distance_rank")],
        "object_rel_distance",
        "B",
    ),
    (
        _payload("B", "option", "option_map", ["r1"], {
            "field": "direction", "mapping": {"left": "A", "right": "B"},
        }),
        [_result("r1", {"direction": "right"}, tool="relative_direction_of")],
        "object_rel_direction",
        "B",
    ),
])
def test_registered_operations_replay_to_the_declared_answer(
        payload, results, question_type, computed):
    replay = replay_derivation(
        payload, results, question_type=question_type, options=["first", "second"])

    assert replay.passed, replay.issues
    assert replay.performed
    assert replay.computed_value == computed


def test_wrong_computed_answer_is_rejected():
    replay = replay_derivation(
        _payload(4, "count", "field", ["r1"], {"field": "count"}),
        [_result("r1", {"count": 3})],
        question_type="object_counting",
    )

    assert not replay.passed
    assert {issue.code for issue in replay.issues} == {"value_mismatch"}


@pytest.mark.parametrize(("result", "code"), [
    (_result("r1", {"count": 3}, status="failed", error="failed"),
     "result_not_successful"),
    (_result("r1", {"count": 3}, invalidated_by=["M11:old"]),
     "result_invalidated"),
    (_result("r1", {"count": 3}, raw_value="{not-json"), "result_not_json"),
])
def test_unusable_tool_results_fail_closed(result, code):
    replay = replay_derivation(
        _payload(3, "count", "field", ["r1"], {"field": "count"}),
        [result],
        question_type="object_counting",
    )

    assert not replay.passed
    assert code in {issue.code for issue in replay.issues}


def test_derivation_parameters_cannot_contain_code_or_arbitrary_keys():
    replay = replay_derivation(
        _payload(3, "count", "field", ["r1"], {
            "field": "count", "code": "open('/tmp/x', 'w')",
        }),
        [_result("r1", {"count": 3})],
        question_type="object_counting",
    )

    assert not replay.passed
    assert {issue.code for issue in replay.issues} == {"invalid_parameters"}


def test_answer_unit_must_match_both_task_and_derived_field():
    replay = replay_derivation(
        _payload(2.0, "cm", "field", ["r1"], {"field": "distance_m"}),
        [_result("r1", {"distance_m": 2.0}, tool="object_distance_m")],
        question_type="object_abs_distance",
    )

    assert not replay.passed
    assert {issue.code for issue in replay.issues} == {
        "canonical_unit_mismatch", "derived_unit_mismatch",
    }


def test_unknown_source_unit_cannot_be_claimed_as_metric():
    replay = replay_derivation(
        _payload(2.0, "m", "field", ["r1"], {}),
        [_result("r1", 2.0, tool="euclidean_distance")],
        question_type="object_abs_distance",
    )

    assert not replay.passed
    assert "unit_unverified" in {issue.code for issue in replay.issues}


def test_mixed_without_derivation_does_not_claim_replay():
    payload = AnswerPayload(
        value="A", unit="option", basis="mixed", used_result_ids=["r1"])

    replay = replay_derivation(
        payload,
        [_result("r1", {"closest_category": "chair"},
                 tool="relative_distance_rank")],
        question_type="object_rel_distance",
        options=["chair", "table"],
    )

    assert replay.passed
    assert not replay.performed
