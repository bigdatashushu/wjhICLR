"""Strict finalization accepts data, preserves M11, and never infers an answer."""

import json

import pytest

from skill3d.online.submission import (
    SubmissionFormatError, finalization_program, parse_finalization,
)
from skill3d.sandbox.action_check import check_action
from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import ProgramExecutionTrace, ToolResult
from skill3d.tools import REGISTRY
from skill3d.verifier.derivation import replay_derivation
from skill3d.verifier.geometry_oracle import geometry_verify
from test_geometry_tools import _handle, _scene


def answer(**changes):
    return {"action": "answer", "value": 3, "basis": "visual_estimate",
            "used_result_ids": [], "derivation": None, **changes}


def parse(raw, **changes):
    args = dict(question_type="object_counting", options=[],
                valid_result_ids=[], readable_images=True, **changes)
    return parse_finalization(json.dumps(raw) if not isinstance(raw, str) else raw, **args)


@pytest.mark.parametrize("raw", [
    "answer = 3", "ReturnAnswer(3)", "```json\n{}\n```", '{"action":"answer"}',
    json.dumps(answer()) + json.dumps(answer()),
    json.dumps(answer()).replace('"value": 3', '"value": 3, "value": 5'),
    answer(value=True), answer(value=None), answer(value=[3]), answer(value="3"),
    answer(value=-1), answer(value=3.2), answer(value=float("nan")), answer(value=float("inf")),
    json.dumps(answer()).replace('"value": 3', '"value": 1e999'),
    answer(unit="m"), answer(basis="unknown"), answer(action="tool"),
    answer(extra="print('injected')"), answer(used_result_ids=["fake"]),
    answer(basis="tool_derived"), answer(basis="mixed", derivation={"op": "exec"}),
])
def test_invalid_json_and_answer_contract_never_produce_a_program(raw):
    with pytest.raises(SubmissionFormatError):
        parse(raw)


def test_option_letters_and_images_are_validated_without_labels():
    payload = parse_finalization(
        json.dumps(answer(value="B")), question_type="object_rel_direction",
        options=["A. left", "B. right"], valid_result_ids=[], readable_images=True)
    assert payload.value == "B" and payload.unit == "option"
    with pytest.raises(SubmissionFormatError):
        parse_finalization(
            json.dumps(answer(value="C")), question_type="object_rel_direction",
            options=["A. left", "B. right"], valid_result_ids=[], readable_images=True)
    with pytest.raises(SubmissionFormatError, match="readable images"):
        parse_finalization(json.dumps(answer()), question_type="object_counting",
                           options=[], valid_result_ids=[], readable_images=False)


def test_host_program_and_direct_submission_have_identical_m11_results():
    payload = parse(answer())
    program = finalization_program(payload)
    assert ast_guard(program.program_source, allowed_tools=set()).ok
    outputs = []
    for source in (
        program.program_source,
        "ReturnAnswer(AnswerPayload(value=3, unit='count', basis='visual_estimate'))",
    ):
        kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
        kernel.set_tools_enabled(False)
        cell = kernel.run_cell(source)
        assert cell.error is None and cell.answer == "3"
        assert not kernel.tool_results
        trace = ProgramExecutionTrace(
            program_id="p", calls=[], results=[], stdout_tail="", error_code=None,
            steps=1, wallclock_s=0)
        outputs.append((
            geometry_verify(trace, kernel._scene, cell.answer).model_dump(),
            replay_derivation(kernel.answer_slot.payload, [],
                              question_type="object_counting").model_dump(),
        ))
    assert outputs[0] == outputs[1]
    assert outputs[0][0]["passed"] and outputs[0][1]["passed"]


def test_valid_json_does_not_make_wrong_derivation_acceptable():
    raw = answer(
        basis="tool_derived", value=4, used_result_ids=["valid"],
        derivation={"op": "count", "input_result_ids": ["valid"], "parameters": {}})
    payload = parse_finalization(
        json.dumps(raw), question_type="object_counting", options=[],
        valid_result_ids=["valid"], readable_images=True)
    result = ToolResult(result_id="valid", tool="list_objects", source_tool="list_objects",
                        status="ok", value='["one","two"]', request_digest="d")
    replay = replay_derivation(payload, [result], question_type="object_counting")
    assert not replay.passed
    assert "value_mismatch" in {issue.code for issue in replay.issues}


def test_derivation_strings_are_data_and_not_executed(tmp_path):
    code = f"open({str(tmp_path / 'sentinel')!r}, 'w')"
    payload = parse_finalization(json.dumps(answer(
        basis="tool_derived", used_result_ids=["r"],
        derivation={"op": "field", "input_result_ids": ["r"], "parameters": {"field": code}})),
        question_type="object_counting", options=[], valid_result_ids=["r"], readable_images=True)
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    cell = kernel.run_cell(finalization_program(payload).program_source)
    assert cell.answer == "3" and not (tmp_path / "sentinel").exists()
    result = ToolResult(result_id="r", source_tool="list_objects", status="ok",
                        tool="list_objects", value="[]", request_digest="d")
    assert not replay_derivation(payload, [result], question_type="object_counting").passed


@pytest.mark.parametrize("source", ["answer = 3", "print(3)", "list_objects()"])
def test_missing_action_is_separate_from_ast_legality(source):
    assert ast_guard(source).ok
    assert check_action(source).error_code == "program_missing_action"
    assert check_action("answer = 3", finalizing=True).error_code == "finalization_missing_submission"


def test_conditional_control_is_left_to_runtime_and_final_tools_are_distinct():
    conditional = "if False:\n    ReturnAnswer(3)"
    assert check_action(conditional).ok
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    cell = kernel.run_cell(conditional)
    assert cell.terminated is None and cell.answer is None
    result = check_action("ctx.tools.list_objects()\nReturnAnswer(3)", finalizing=True,
                          tool_names=set(REGISTRY.names()))
    assert result.error_code == "finalization_tool_call_disallowed"
