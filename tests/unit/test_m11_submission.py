"""M11 结构化归因与跨轮提交范围。"""

import json

import numpy as np
import pytest

from skill3d.online.recovery import collect_validated, invalidate_results
from skill3d.online.submission import submission_scope
from skill3d.sandbox.kernel import AnswerTerminate, RestrictedNamespaceKernel
from skill3d.schemas import AnswerPayload, ProgramExecutionTrace, ToolResult
from skill3d.tools import REGISTRY
from skill3d.verifier.geometry_oracle import geometry_verify

from test_geometry_tools import _handle, _scene


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
