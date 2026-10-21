"""Process telemetry must not invent visual facts or leak sample content."""

import json

import pytest

from skill3d.evolution.process_feedback import (
    extract_process_facts, legacy_process_facts, safe_process_dict, score_band,
)


def test_direct_visual_success_records_received_frames_without_tool_selection():
    outcome = {
        "final_state": "answer",
        "rounds": [{"executed": True, "program_source": "ReturnAnswer('A')"}],
        "image_ledger": {
            "images": [{"kind": "frame", "delivered_rounds": [1], "observed_rounds": [1]}
                       for _ in range(32)],
            "rounds": [{"delivered": True, "original_image_ids": list(range(32)),
                        "derived_image_ids": []}],
        },
    }
    facts = extract_process_facts(outcome)
    assert facts.input_observation.original_frames_response_completed == 32
    assert facts.execution.tool_calls_by_name == {}
    assert facts.referents.resolved_tool_object_count == 0
    assert facts.referents.candidate_count is None
    assert facts.referents.explicit_selection_count is None
    assert facts.execution.answer_submitted


def test_list_count_is_not_resolved_selection_and_error_chain_is_retained():
    facts = extract_process_facts({
        "final_state": "run_error",
        "rounds": [
            {"index": 1, "executed": True, "error_code": "runtime_error",
             "execution_diagnostics": {"error_type": "TypeError"},
             "results": [{"source_tool": "list_objects", "status": "ok",
                          "value": '[{"obj_id":"private-id"},{"obj_id":"another"}]'}]},
            {"index": 2, "trigger": "finalize", "program_source": "answer = 2.5",
             "error_code": "ast_violation"},
        ],
    })
    assert facts.referents.candidate_count == 2
    assert facts.referents.resolved_tool_object_count == 0
    assert facts.execution.answer_assignment_without_submit
    assert facts.terminal_error == "finalization_missing_submission"
    assert len(facts.error_chain) == 2
    assert facts.first_actionable_error == "tool_argument_error"
    assert facts.error_chain[0].return_type == "list"
    assert "private-id" not in json.dumps(safe_process_dict(facts.model_dump()))


@pytest.mark.parametrize(("program", "expected"), [
    ("answer = 'A'", "finalization_missing_submission"),
    ("r = inspect_frames([0])\nYieldObservations([], 'read')", "finalization_tool_call_disallowed"),
])
def test_legacy_finalization_is_reclassified_without_extracting_answer(program, expected):
    facts = extract_process_facts({
        "rounds": [{"trigger": "finalize", "error_code": "ast_violation",
                    "program_source": program}],
    })
    assert facts.terminal_error == expected
    assert "value" not in safe_process_dict(facts.model_dump())["execution"]


def test_whitelist_removes_adversarial_sample_strings_from_nested_fields():
    facts = extract_process_facts({"rounds": [{"error_code": "unknown"}]}).model_dump()
    secret = "qa-abcdef scene-private reference_answer=937.777"
    facts["error_chain"][0].update(
        event_type=secret, tool_name=secret, exception_type=secret,
        argument_shape={"raw_question": secret, "boxes_length": secret})
    facts["execution"]["tool_calls_by_name"] = {secret: 1}
    facts["first_actionable_error"] = secret
    facts["observation_quality"]["required_fields_omitted"] = [secret]
    assert secret not in json.dumps(safe_process_dict(facts))
    with pytest.raises(ValueError):
        safe_process_dict({**facts, "reference_answer": secret})


def test_absent_legacy_images_remain_unknown_and_score_bands_are_separate():
    facts = legacy_process_facts({"selected_object_count": 0, "final_state": "answer"})
    assert facts.input_observation.original_frames_sent is None
    assert facts.referents.candidate_count is None
    assert facts.referents.explicit_selection_count is None
    assert [score_band(s, available) for s, available in
            [(1, True), (.9, True), (0, True), (0, False)]] == [
        "full_score", "partial_score", "zero_score_answered", "no_answer"]
