"""A missing yield can return real observations, but cannot create an answer."""

import pytest
import json

from skill3d.sandbox.action_check import permits_observation_fallthrough
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import ToolResult
from skill3d.tools import REGISTRY
from test_geometry_tools import _handle, _scene


@pytest.mark.parametrize("source", [
    "answer = 7", "print(7)", "# ReturnAnswer(7)",
    "ReturnAnswer", "if False:\n    ReturnAnswer(7)",
    "result = list_objects()\nanswer = AnswerPayload(value=7, unit='count', basis='mixed')",
])
def test_answer_or_diagnostic_intent_is_never_adapted(source):
    assert not permits_observation_fallthrough(source, tool_names=set(REGISTRY.model_names()))


@pytest.mark.parametrize("source", [
    "result = list_objects()\nprint(result['data'])",
    "def solve(ctx):\n    result = ctx.tools.list_objects()\n    print(result['data'])",
])
def test_successful_evidence_yields_once_without_changing_or_submitting_code(source):
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    assert permits_observation_fallthrough(source, tool_names=set(REGISTRY.model_names()))
    cell = kernel.run_cell(source, yield_on_fallthrough=True)
    assert cell.host_observation_yield and cell.terminated == "yield"
    assert cell.answer is None and not kernel.answer_slot.given
    assert len(kernel.cell_results) == 1
    assert cell.yielded_result_ids == [kernel.cell_results[0].result_id]
    assert kernel.cell_results[0].status == "ok"


@pytest.mark.parametrize("source", [
    "print(7)", "answer = 7",
    "if False:\n    list_objects()",
    "def unused():\n    list_objects()",
])
def test_no_current_observation_cannot_use_a_historical_result_to_yield(source):
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    kernel.run_cell("list_objects()")
    kernel.reset_user_namespace()
    cell = kernel.run_cell(source, yield_on_fallthrough=True)
    assert not cell.host_observation_yield and cell.terminated is None
    assert cell.answer is None and kernel.cell_results == []


def test_failed_or_caught_call_is_not_a_successful_observation_and_history_is_local():
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    failed = kernel.run_cell(
        "try:\n    exists_in_scene('missing')\nexcept Exception:\n    pass\nlist_objects()",
        yield_on_fallthrough=True)
    assert failed.answer_untrusted and not failed.host_observation_yield
    assert failed.contract_violations
    kernel.reset_user_namespace()
    recovered = kernel.run_cell("list_objects()", yield_on_fallthrough=True)
    assert recovered.host_observation_yield and not recovered.answer_untrusted
    assert recovered.contract_violations == []
    assert len(kernel.contract_violations) == 1


def test_valid_page_can_yield_but_a_failed_page_cannot():
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    kernel.run_cell("list_objects()")
    rid = kernel.cell_results[0].result_id
    kernel.reset_user_namespace()
    page = kernel.run_cell(f"ReadObservation({rid!r})", yield_on_fallthrough=True)
    assert page.host_observation_yield and page.yielded_result_ids == [rid]
    kernel.reset_user_namespace()
    failed = kernel.run_cell(
        f"ReadObservation({rid!r})\ntry:\n    ReadObservation('other-episode')\nexcept Exception:\n    pass",
        yield_on_fallthrough=True)
    assert not failed.host_observation_yield and failed.terminated is None
    assert failed.answer_untrusted
    assert kernel.authorization_receipts[-1]["allowed"] is False


@pytest.mark.parametrize("runtime_error", [False, True])
def test_fault_or_caught_runtime_result_does_not_become_an_implicit_observation(monkeypatch, runtime_error):
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    payload = {"status": "fault", "service_healthy": False, "detections": []}
    result = ToolResult(
        tool="detect_objects", source_tool="detect_objects", status="ok",
        value=json.dumps(payload), payload=payload, request_digest="failed-service",
        error="service failed" if runtime_error else None)
    monkeypatch.setattr(REGISTRY, "call_tool", lambda *args, **kwargs: result)
    cell = kernel.run_cell(
        "try:\n    detect_objects([0], ['chair'])\nexcept Exception:\n    pass",
        yield_on_fallthrough=True)
    assert not cell.host_observation_yield and cell.terminated is None
    assert cell.answer is None
    assert cell.answer_untrusted is runtime_error
