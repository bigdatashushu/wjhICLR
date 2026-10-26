"""Submission quarantine and malformed derivation boundaries, without models."""

import json

import pytest

from skill3d.online.recovery import collect_validated
from skill3d.online.submission import (
    SubmissionFormatError,
    accepted_result_ids,
    invalid_reference_issues,
    parse_finalization,
    submission_scope,
)
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import AnswerPayload, ProgramExecutionTrace, ToolResult
from skill3d.tools import REGISTRY
from skill3d.verifier.derivation import replay_derivation
from skill3d.verifier.geometry_oracle import geometry_verify
from test_geometry_tools import _handle, _scene


def _kernel():
    return RestrictedNamespaceKernel(
        REGISTRY, _handle(_scene(question_type="object_counting")))


def _result(rid, *, tool="list_objects", value=None, **updates):
    return ToolResult(
        result_id=rid, source_tool=tool, tool=tool, status="ok",
        value=json.dumps([] if value is None else value),
        request_digest=f"digest-{rid}", **updates)


def _trace(results=()):
    return ProgramExecutionTrace(
        program_id="quarantine", calls=[], results=list(results),
        stdout_tail="", error_code=None, steps=1, wallclock_s=0)


def _submit(kernel, **updates):
    arguments = {"value": 1, "unit": "count", "basis": "mixed", **updates}
    cell = kernel.run_cell(f"ReturnAnswer(AnswerPayload(**{arguments!r}))")
    assert cell.error_code is None and cell.terminated == "answer"
    assert isinstance(kernel.answer_slot.payload, AnswerPayload)
    return kernel.answer_slot.payload


@pytest.mark.parametrize("basis", ["mixed", "tool_derived"])
@pytest.mark.parametrize("reference", ["declared", "derivation_only"])
def test_unresolved_explicit_reference_is_structured_and_does_not_revoke_result(
        basis, reference):
    kernel = _kernel()
    quarantined = _result("quarantined")
    kernel.tool_results = [quarantined]
    kernel.unresolved_result_ids = {"quarantined"}
    _submit(
        kernel, basis=basis,
        used_result_ids=["quarantined"] if reference == "declared" else [],
        derivation={"op": "count", "input_result_ids": ["quarantined"],
                    "parameters": {}})

    assert submission_scope(kernel, _trace()).results == []
    issues = invalid_reference_issues(kernel)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.check == "unresolved_reference"
    assert issue.result_id == "quarantined" and issue.tool == "list_objects"
    assert issue.failure_cause.code == "unresolved_reference"
    assert issue.failure_cause.scope == "local_result"
    assert issue.failure_cause.result_ids == ["quarantined"]
    assert not issue.failure_cause.shared_premises
    assert issue.confirmed_shared_premise is None and not issue.invalidate_result
    assert quarantined.status == "ok" and not quarantined.invalidated_by


@pytest.mark.parametrize("submission", ["short", "implicit", "visual", "explicit"])
def test_scope_excludes_quarantine_but_keeps_current_and_other_valid_results(submission):
    kernel = _kernel()
    old = _result("old")
    current = _result("current")
    kernel.tool_results = [
        _result("quarantined"), old, current,
        _result("failed", error="synthetic failure"),
        _result("revoked", invalidated_by=["synthetic-local"]),
    ]
    kernel.unresolved_result_ids = {"quarantined"}
    if submission == "short":
        assert kernel.run_cell("ReturnAnswer(1)").error_code is None
    elif submission == "visual":
        _submit(kernel, basis="visual_estimate")
    else:
        _submit(kernel, used_result_ids=["old"] if submission == "explicit" else [])

    trace = _trace([current])
    before = [result.result_id for result in trace.results]
    scope = submission_scope(kernel, trace)
    expected = ["current"] if submission == "visual" else ["old", "current"]
    assert [result.result_id for result in scope.results] == expected
    assert accepted_result_ids(kernel, scope) == expected
    assert [result.result_id for result in trace.results] == before
    assert invalid_reference_issues(kernel) == []


@pytest.mark.parametrize("basis", ["mixed", "visual_estimate"])
def test_current_bad_measurement_cannot_be_hidden_by_old_reference_or_visual_basis(basis):
    kernel = _kernel()
    current = _result("current", tool="euclidean_distance", value=-1)
    kernel.tool_results = [_result("quarantined"), _result("old"), current]
    kernel.unresolved_result_ids = {"quarantined"}
    _submit(kernel, basis=basis, used_result_ids=["old"] if basis == "mixed" else [])

    scope = submission_scope(kernel, _trace([current]))
    assert "current" in accepted_result_ids(kernel, scope)
    verified = geometry_verify(scope, kernel._scene)
    assert not verified.passed and "no_negative_distance" in verified.violations
    assert any(issue.result_id == "current" for issue in verified.issues)


def test_reader_and_json_still_reject_quarantine_while_visual_has_empty_scope():
    kernel = _kernel()
    observed = kernel.run_cell("listed = list_objects()\nYieldObservations([listed['result_id']])")
    assert observed.error_code is None and observed.terminated == "yield"
    rid = kernel.tool_results[0].result_id
    kernel.reset_user_namespace()
    kernel.unresolved_result_ids = {rid}
    read = kernel.run_cell(f"ReadObservation({rid!r})")
    assert read.error_type == "ValueError" and read.answer_untrusted
    assert "unresolved" in read.error_message
    assert kernel.authorization_receipts[-1]["reason_code"] == "observation_read_rejected"

    valid_ids = [obs.result_id for obs in collect_validated(kernel)
                 if obs.result_id not in kernel.unresolved_result_ids]
    raw = {"action": "answer", "value": 1, "basis": "mixed",
           "used_result_ids": [rid], "derivation": None}
    with pytest.raises(SubmissionFormatError) as rejected:
        parse_finalization(
            json.dumps(raw), question_type="object_counting", options=[],
            valid_result_ids=valid_ids, readable_images=True)
    assert rejected.value.code == "answer_contract_rejected"

    kernel.reset_user_namespace()
    payload = _submit(kernel, basis="visual_estimate")
    scope = submission_scope(kernel, _trace())
    assert scope.results == [] and invalid_reference_issues(kernel) == []
    assert geometry_verify(scope, kernel._scene).passed
    assert replay_derivation(payload, kernel.tool_results, question_type="object_counting").passed


@pytest.mark.parametrize("input_ids", [
    1, True, {"r": 1}, None, "r", [], [1], ["r", None], [["r"]],
], ids=["int", "bool", "dict", "none", "string", "empty", "number_item",
        "none_item", "nested_list"])
def test_malformed_derivation_is_rejected_before_replay_and_preserved_for_audit(input_ids):
    kernel = _kernel()
    derivation = {"op": "count", "input_result_ids": input_ids, "parameters": {}}
    payload = _submit(kernel, derivation=derivation)
    assert payload.derivation == derivation

    replay = replay_derivation(payload, [], question_type="object_counting")
    assert not replay.passed and not replay.performed
    assert replay.computed_value is None and replay.input_result_ids == []
    assert replay.expected_value == 1 and replay.expected_unit == "count"
    assert [issue.code for issue in replay.issues] == ["payload_contract"] * len(payload.problems())
    assert [issue.reason for issue in replay.issues] == payload.problems()
    assert replay.issues[0].reason == (
        "derivation.input_result_ids must be a list of nonempty strings")
    assert payload.derivation == derivation
    assert kernel.answer_slot.payload is payload


def test_payload_contract_failure_keeps_unit_diagnostic_without_scanning_ledger():
    class UnreadableLedger(tuple):
        def __iter__(self):
            raise AssertionError("Malformed payload must not scan the tool ledger")

    kernel = _kernel()
    payload = _submit(
        kernel, unit="m",
        derivation={"op": "count", "input_result_ids": 1, "parameters": {}})
    replay = replay_derivation(payload, UnreadableLedger(), question_type="object_counting")
    assert not replay.passed and not replay.performed
    assert replay.op == "count" and replay.expected_unit == "m"
    assert [issue.code for issue in replay.issues] == [
        "payload_contract", "canonical_unit_mismatch"]


def test_valid_derivation_still_replays_and_duplicate_inputs_still_fail_closed():
    kernel = _kernel()
    result = _result("valid", value=[{"obj_id": "synthetic"}])
    kernel.tool_results = [result]
    for ids, expected in [(["valid"], True), (["valid", "valid"], False)]:
        kernel.reset_user_namespace()
        payload = _submit(
            kernel, basis="tool_derived", used_result_ids=["valid"],
            derivation={"op": "count", "input_result_ids": ids, "parameters": {}})
        replay = replay_derivation(payload, kernel.tool_results, question_type="object_counting")
        assert replay.passed is expected and replay.performed
        if expected:
            assert replay.computed_value == 1
        else:
            assert {issue.code for issue in replay.issues} == {"duplicate_input_result_id"}
