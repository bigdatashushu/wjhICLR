"""M11 结构化归因与跨轮提交范围。"""

import json

import numpy as np
import pytest

from skill3d.online.recovery import collect_validated, invalidate_results
from skill3d.online.submission import invalid_reference_issues, submission_scope
from skill3d.sandbox.kernel import AnswerTerminate, RestrictedNamespaceKernel
from skill3d.schemas import AnswerPayload, ObjectRecord, ProgramExecutionTrace, ToolResult
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
    world_frame_usable = True
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
    assert verify.check_status["no_negative_distance"] == "fail"
    assert verify.check_details["no_negative_distance"]["executed_result_ids"] == ["bad-distance"]
    assert verify.check_status["reprojection"] == "not_applicable"


def test_m11_rejects_result_that_bypasses_binding_integrity_gate():
    obj = ObjectRecord(
        obj_id="obj_0", category_name="television",
        centroid_world=[0.0, 0.0, 2.0],
        binding_flags=[{
            "code": "semantic_label_mask_conflict",
            "audit_sha256": "a" * 64,
        }])
    handle = _handle(_scene(objs=[obj]), objs=[obj])
    result = _result(
        "bypassed", {
            "centroid_normalized": [0.0, 0.0, 2.0],
            "centroid_metric": None,
            "category_name": "television",
            "track_id": None,
        }, tool="object_centroid")
    result.args = {"obj_id": "obj_0"}
    trace = ProgramExecutionTrace(
        program_id="p", calls=[], results=[result], stdout_tail="",
        error_code=None, steps=1, wallclock_s=0.0)
    verified = geometry_verify(trace, handle)
    assert not verified.passed
    assert verified.violations == ["binding_integrity"]
    assert verified.issues[0].result_id == "bypassed"


def test_empty_geometry_contract_pass_does_not_claim_checks_executed():
    trace = ProgramExecutionTrace(
        program_id="visual", calls=[], results=[], stdout_tail="",
        error_code=None, steps=1, wallclock_s=0)
    verified = geometry_verify(trace, _VerifierHandle())
    assert verified.passed and all(verified.checks.values())
    assert set(verified.check_status.values()) == {"not_applicable"}
    assert all(d["execution_count"] == 0 for d in verified.check_details.values())


def test_missing_bbox_is_not_performed_and_missing_projection_artifacts_fail():
    class NoGeometry(_VerifierHandle):
        def scene_bbox(self):
            raise ValueError("no object bbox")

    result = _result("projection", [1, 2], tool="reproject")
    result.args = {"p3d": [0, 0, 1], "frame_idx": 0}
    trace = ProgramExecutionTrace(
        program_id="p", calls=[], results=[result], stdout_tail="",
        error_code=None, steps=1, wallclock_s=0)
    verified = geometry_verify(trace, NoGeometry())
    assert not verified.passed
    assert verified.check_status["inside_bbox"] == "not_performed"
    assert verified.check_details["inside_bbox"]["execution_count"] == 0
    assert verified.check_status["reprojection"] == "fail"
    assert verified.check_details["reprojection"]["execution_count"] == 1


def test_used_flags_only_include_successful_final_references():
    from skill3d.online.submission import used_evidence_flags
    from test_geometry_tools import _obj
    obj = _obj("chair", "chair", (0, 0, 1)).model_copy(update={"duplicate_suspect": True})
    handle = _handle(_scene(objs=[obj]), objs=[obj])
    kernel = RestrictedNamespaceKernel(REGISTRY, handle)
    result = _result("used", {"distance_m": 2, "degradation_flags": [
        {"code": "geometry_3d_degraded", "audit_id": "audit-a"}],
        "duplicate_suspect": True}).model_copy(update={
            "degraded_evidence": ["geometry_3d"], "object_dependencies": ["chair"]})
    kernel.tool_results = [
        result,
        result.model_copy(update={"result_id": "unreferenced"}),
        result.model_copy(update={"result_id": "revoked", "invalidated_by": ["local"]}),
        result.model_copy(update={"result_id": "failed", "status": "failed"}),
    ]
    flags = used_evidence_flags(kernel, ["used", "revoked", "failed"], handle)
    assert {f["result_id"] for f in flags} == {"used"}
    assert {f["code"] for f in flags} == {
        "geometry_3d_degraded", "geometry_3d", "suspect_duplicate"}
    assert next(f for f in flags if f["code"] == "geometry_3d_degraded")["audit_id"] == "audit-a"
    assert next(f for f in flags if f.get("object_id"))["object_id"] == "chair"
    assert used_evidence_flags(kernel, [], handle) == []


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


def test_object_failure_revokes_only_consumed_geometry_and_explicit_derivatives():
    from skill3d.online.recovery import invalidate_objects

    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    kernel.tool_results.extend([
        _result("bad", 1).model_copy(update={"object_dependencies": ["tv"]}),
        _result("good", 2).model_copy(update={"object_dependencies": ["chair"]}),
        _result("room", 3, tool="plane_fit_room_size"),
        _result("derived", 4).model_copy(update={"result_dependencies": ["bad"]}),
        _result("listed", [], tool="list_objects"),
    ])
    assert invalidate_objects(kernel, ["tv"], reason="semantic_label_mask_conflict") == [
        "bad", "derived"]
    assert [o.result_id for o in collect_validated(kernel)] == ["good", "room", "listed"]
    assert kernel._scene.evidence_profile.metric_scale == "unavailable"


def test_binding_cause_survives_registry_kernel_and_cannot_revoke_unrelated_results():
    from skill3d.online.recovery import invalidate_objects
    from test_geometry_tools import _obj

    good = _obj("chair", "chair", (1, 0, 0))
    bad = _obj("tv", "television", (0, 0, 2), binding_flags=[{
        "code": "semantic_label_mask_conflict", "audit_sha256": "b" * 64}])
    handle = _handle(_scene(metric=True, question_type="object_size_estimation",
                           objs=[good, bad]), objs=[good, bad], metric_scale=2)
    kernel = RestrictedNamespaceKernel(REGISTRY, handle)
    cell = kernel.run_cell("object_centroid('chair')\nobject_3d_extent('tv')")
    assert cell.error_code == "tool_contract"
    assert kernel.tool_results[-1].error_code == "confidence_gate"
    cause = kernel.tool_results[-1].failure_cause
    assert cell.contract_violations[-1]["failure_cause"] == cause.model_dump(mode="json")
    assert cause.object_ids == ["tv"] and not cause.shared_premises
    assert invalidate_objects(kernel, cause.object_ids, reason=cause.code) == []
    assert len(collect_validated(kernel)) == 1
    assert handle.evidence_profile.metric_scale == "available"


def test_shared_missing_facts_and_multiple_downgrades_have_precise_union():
    from skill3d.online.recovery import downgrade_profile, premise_of_failure
    from skill3d.tools.contract import ArtifactUnavailableError

    cause = ArtifactUnavailableError("test", ["poses", "scale"]).failure_cause
    assert cause.shared_premises == ["geometry_3d", "metric_scale"]
    assert premise_of_failure(
        error_code="confidence_gate", tool="object_3d_extent",
        requires_evidence=("metric_scale",)) is None
    assert premise_of_failure(error_code="tool_contract", tool="unknown") is None
    profile = _scene(metric=True, question_type="object_size_estimation").evidence_profile
    first, changed = downgrade_profile(profile, "geometry_3d")
    second, other = downgrade_profile(first, "metric_scale", excluded_capabilities=changed)
    assert second.metric_scale == "unavailable" and not other


def test_repeated_tool_requests_have_unique_replayable_occurrence_ids():
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    result = kernel.run_cell(
        "list_objects()\n"
        "list_objects()")
    assert result.error_code is None
    a, b = kernel.tool_results
    assert a.request_digest == b.request_digest
    assert a.result_id != b.result_id
    assert a.authorization["result_id"] == a.result_id
    assert b.authorization["result_id"] == b.result_id
    replay = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    replay.run_cell(
        "list_objects()\n"
        "list_objects()")
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
