"""M11 结构化归因与跨轮提交范围。"""

import json

import numpy as np
import pytest

from skill3d.online.recovery import collect_validated, invalidate_results
from skill3d.online.submission import invalid_reference_issues, submission_scope
from skill3d.sandbox.kernel import AnswerTerminate, RestrictedNamespaceKernel
from skill3d.schemas import AnswerPayload, ProgramExecutionTrace, ToolResult
from skill3d.tools import REGISTRY
from skill3d.verifier.geometry_oracle import geometry_verify

from test_geometry_tools import _handle, _scene
from test_geometry_tools import _ranking_handle
from skill3d.tools.geometry_tools import relative_distance_rank
from skill3d.verifier.derivation import replay_derivation


class _MetricState:
    def metric_task_authorized(self, question_type):
        return False


class _VerifierHandle:
    world_up = np.asarray([0.0, 0.0, 1.0])
    handedness = "right"
    metric_gate_passed = False
    question_type = "object_abs_distance"
    _state = _MetricState()

    def scene_bbox(self):
        return np.asarray([-2.0, -2.0, -2.0]), np.asarray([2.0, 2.0, 2.0])


def _result(result_id, value, *, tool="object_distance_m"):
    return ToolResult(
        result_id=result_id, source_tool=tool, status="ok", tool=tool,
        value=json.dumps(value), request_digest=f"digest-{result_id}")


def test_geometry_verify_locates_every_failed_check_at_result_level():
    result = _result("bad-distance", {
        "distance_m": -1.0, "distance_metric": -1.0,
        "distance_normalized": -0.5,
    })
    trace = ProgramExecutionTrace(
        program_id="p", calls=[], results=[result], stdout_tail="",
        error_code=None, steps=1, wallclock_s=0.0)

    verify = geometry_verify(trace, _VerifierHandle())

    assert verify.passed is False
    assert set(verify.violations) == {"no_negative_distance", "unit_consistent"}
    assert {i.result_id for i in verify.issues} == {"bad-distance"}
    assert {i.tool for i in verify.issues} == {"object_distance_m"}
    assert all(i.confirmed_shared_premise is None for i in verify.issues)


def test_submission_scope_keeps_cross_round_results_unless_answer_is_visual_only():
    handle = _handle(_scene())
    kernel = RestrictedNamespaceKernel(REGISTRY, handle)
    old = _result("old", 1.0, tool="euclidean_distance")
    current = _result("current", 2.0, tool="euclidean_distance")
    kernel.tool_results.extend([old, current])
    trace = ProgramExecutionTrace(
        program_id="p", calls=[], results=[current], stdout_tail="",
        error_code=None, steps=1, wallclock_s=0.0)

    with pytest.raises(AnswerTerminate):
        kernel.answer_slot("1")
    mixed = submission_scope(kernel, trace)
    assert [r.result_id for r in mixed.results] == ["old", "current"]

    kernel.reset_user_namespace()
    with pytest.raises(AnswerTerminate):
        kernel.answer_slot(
            AnswerPayload(
                value=1, unit="m", basis="visual_estimate",
                used_result_ids=[], derivation=None))
    visual = submission_scope(kernel, trace)
    assert [r.result_id for r in visual.results] == ["current"]

    kernel.reset_user_namespace()
    with pytest.raises(AnswerTerminate):
        kernel.answer_slot(
            AnswerPayload(
                value=1, unit="m", basis="tool_derived",
                used_result_ids=["old"],
                derivation={"op": "field", "input_result_ids": ["old"],
                            "parameters": {}}))
    declared = submission_scope(kernel, trace)
    assert [r.result_id for r in declared.results] == ["old", "current"]


def test_local_invalidation_removes_only_the_named_result_from_recovery():
    handle = _handle(_scene())
    kernel = RestrictedNamespaceKernel(REGISTRY, handle)
    kernel.tool_results.extend([
        _result("bad", -1.0, tool="euclidean_distance"),
        _result("good", 2.0, tool="euclidean_distance"),
    ])

    assert invalidate_results(kernel, ["bad"], reason="M11:test") == ["bad"]
    assert [o.result_id for o in collect_validated(kernel)] == ["good"]


def test_repeated_tool_requests_have_unique_replayable_occurrence_ids():
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    result = kernel.run_cell(
        "euclidean_distance([0,0,0], [1,0,0])\n"
        "euclidean_distance([0,0,0], [1,0,0])")
    assert result.error_code is None
    a, b = kernel.tool_results
    assert a.request_digest == b.request_digest
    assert a.result_id != b.result_id
    assert a.authorization["result_id"] == a.result_id
    assert b.authorization["result_id"] == b.result_id
    replay = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    replay.run_cell(
        "euclidean_distance([0,0,0], [1,0,0])\n"
        "euclidean_distance([0,0,0], [1,0,0])")
    assert [r.result_id for r in replay.tool_results] == [a.result_id, b.result_id]


@pytest.mark.parametrize("kind", ["missing", "failed", "invalidated"])
def test_mixed_submission_rejects_unusable_references(kind):
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    if kind != "missing":
        result = _result("reference", 1.0)
        if kind == "failed":
            result = result.model_copy(update={"status": "error", "error": "failure"})
        else:
            result = result.model_copy(update={"invalidated_by": ["M11"]})
        kernel.tool_results.append(result)
    with pytest.raises(AnswerTerminate):
        kernel.answer_slot(AnswerPayload(
            value=1, unit="m", basis="mixed", used_result_ids=["reference"]))
    issues = invalid_reference_issues(kernel)
    assert len(issues) == 1
    assert issues[0].check == f"{kind}_reference"


@pytest.mark.parametrize("raw", [
    "{broken", "NaN", '{"distance_normalized": NaN}',
    '{"distance_normalized": "unknown"}',
])
def test_m11_rejects_malformed_or_nonfinite_measurements(raw):
    result = _result("bad", None, tool="robust_distance")
    result.value = raw
    trace = ProgramExecutionTrace(
        program_id="p", calls=[], results=[result], stdout_tail="",
        error_code=None, steps=1, wallclock_s=0.0)
    verified = geometry_verify(trace, _VerifierHandle())
    assert not verified.passed
    assert {issue.result_id for issue in verified.issues} == {"bad"}


@pytest.mark.parametrize("mutation", ["missing", "omitted_option", "tie"])
def test_rank_acceptance_rejects_incomplete_result_without_revoking_shared_geometry(mutation):
    handle = _ranking_handle(missing=mutation == "missing")
    rank = relative_distance_rank(handle, "desk", ["chair", "lamp"])
    if mutation == "tie":
        rank["ranking"][1]["distance_normalized"] = rank["ranking"][0]["distance_normalized"]
    options = ["chair", "lamp", "table"] if mutation == "omitted_option" else ["chair", "lamp"]
    result = _result("rank", rank, tool="relative_distance_rank")
    trace = ProgramExecutionTrace(
        program_id="p", calls=[], results=[result], stdout_tail="",
        error_code=None, steps=1, wallclock_s=0.0)
    verified = geometry_verify(trace, handle, options=options)
    assert not verified.passed and "complete_ranking" in verified.violations
    assert all(i.confirmed_shared_premise is None for i in verified.issues)
    kernel = RestrictedNamespaceKernel(REGISTRY, handle)
    kernel.tool_results.append(result)
    with pytest.raises(AnswerTerminate):
        kernel.answer_slot(AnswerPayload(value="A", unit="option", basis="visual_estimate"))
    # A pure visual answer cannot hide a rank produced in the same cell.
    assert not geometry_verify(submission_scope(kernel, trace), handle, options=options).passed
    # A subsequent zero-tool answer may explicitly discard historical geometry.
    next_trace = trace.model_copy(update={"results": []})
    assert geometry_verify(submission_scope(kernel, next_trace), handle, options=options).passed


def test_rank_derivation_checks_actual_option_coverage_and_mapping():
    handle = _ranking_handle()
    rank = relative_distance_rank(handle, "desk", ["chair", "lamp"])
    result = _result("rank", rank, tool="relative_distance_rank")
    def replay(value, mapping, options):
        return replay_derivation(
            AnswerPayload(
                value=value, unit="option", basis="tool_derived", used_result_ids=["rank"],
                derivation={"op": "option_map", "input_result_ids": ["rank"],
                            "parameters": {"field": "closest_category", "mapping": mapping}}),
            [result], question_type="object_rel_distance", options=options)
    assert replay("A", {"chair": "A", "lamp": "B"}, ["chair", "lamp"]).passed
    assert replay("B", {"chair": "B", "lamp": "A"}, ["lamp", "chair"]).passed
    assert not replay("A", {"chair": "A"}, ["chair", "lamp", "table"]).passed
    wrong = replay("B", {"chair": "B", "lamp": "A"}, ["chair", "lamp"])
    assert "ranking_option_mismatch" in {i.code for i in wrong.issues}
