"""F04: all model-visible values share an ID without altering raw evidence."""

import json

import pytest

from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import AnswerPayload, ToolResult
from skill3d.schemas.tool_response import (
    TOOL_RESPONSE_VERSION, TOOL_RESPONSE_FIELDS, decode_recorded_response,
    model_tool_response,
)
from skill3d.synthesis.execution_contract import review_method_contract
from skill3d.tools import REGISTRY
from skill3d.verifier.derivation import replay_derivation
from test_geometry_tools import _handle, _scene


@pytest.mark.parametrize("data", [
    [{"obj_id": "chair"}], [], 3.25, True, False,
    {"distance": 2.0, "result_id": "business-not-host", "status": "incomplete_candidates"},
])
def test_kernel_wraps_each_data_type_and_keeps_raw_receipt(monkeypatch, data):
    raw = ToolResult(
        tool="list_objects", source_tool="list_objects", result_id="registry-id",
        request_digest="same-request", value=json.dumps(data),
        payload=data if isinstance(data, dict) else None,
        evidence_version="frozen-evidence", authorization={"allowed": True})
    monkeypatch.setattr(REGISTRY, "call_tool", lambda *a, **kw: raw)
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    cell = kernel.run_cell(
        'res = list_objects()\nshow(res)\nYieldObservations([res["result_id"]], "read data")')
    assert cell.error is None and cell.terminated == "yield"
    received = kernel.show_log[0]
    assert set(received) == TOOL_RESPONSE_FIELDS
    assert received["data"] == data and type(received["data"]) is type(data)
    assert received["schema_version"] == TOOL_RESPONSE_VERSION
    assert received["evidence_version"] == "frozen-evidence"
    recorded = kernel.tool_results[0]
    assert recorded.result_id == received["result_id"] == cell.yielded_result_ids[0]
    assert recorded.result_id != raw.result_id
    assert recorded.value == raw.value and recorded.payload == raw.payload
    assert recorded.model_response_schema_version == TOOL_RESPONSE_VERSION
    if isinstance(received["data"], dict):
        assert received["result_id"] != received["data"]["result_id"]
        received["data"]["distance"] = 99
        assert json.loads(recorded.value)["distance"] == 2


@pytest.mark.parametrize("changes", [
    {"status": "failed"}, {"error": "service unavailable"},
    {"error_code": "domain_value"}, {"invalidated_by": ["revoked"]},
    {"authorization": {"allowed": False}}, {"result_id": ""},
])
def test_no_success_envelope_for_failed_invalidated_or_denied_receipts(changes):
    result = ToolResult(result_id="r", tool="list_objects", value="[]")
    with pytest.raises(ValueError, match="valid successful"):
        model_tool_response(result.model_copy(update=changes))


def test_kernel_failed_return_raises_and_never_returns_an_ok_envelope(monkeypatch):
    failed = ToolResult(tool="list_objects", status="failed", error="detector failure")
    monkeypatch.setattr(REGISTRY, "call_tool", lambda *a, **kw: failed)
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()))
    cell = kernel.run_cell("res = list_objects()\nshow(res)\nYieldObservations([], 'read')")
    assert cell.error and not kernel.show_log
    assert kernel.tool_results[0].status == "failed"


def test_legacy_decoder_requires_explicit_version_and_never_guesses_data_field():
    legacy = {"data": [1], "result_id": "old-business-id"}
    assert decode_recorded_response(legacy, schema_version="tool-response/1.0") is legacy
    with pytest.raises(ValueError):
        decode_recorded_response(legacy, schema_version=TOOL_RESPONSE_VERSION)
    with pytest.raises(ValueError):
        decode_recorded_response(legacy, schema_version="unspecified")
    new = model_tool_response(ToolResult(result_id="host", tool="list_objects",
                                        value=json.dumps(legacy)))
    assert decode_recorded_response(new, schema_version=TOOL_RESPONSE_VERSION) == legacy


def test_m11_field_paths_still_start_at_raw_data_not_at_model_envelope():
    result = ToolResult(result_id="r", tool="count_objects", value='{"count": 3}',
                        model_response_schema_version=TOOL_RESPONSE_VERSION)
    response = model_tool_response(result)
    answer = AnswerPayload(
        value=response["data"]["count"], unit="count", basis="tool_derived",
        used_result_ids=[response["result_id"]],
        derivation={"op": "field", "input_result_ids": [response["result_id"]],
                    "parameters": {"field": "count"}})
    replay = replay_derivation(answer, [result])
    assert replay.passed
    bad = answer.model_copy(update={"derivation": {
        "op": "field", "input_result_ids": ["r"], "parameters": {"field": "data.count"}}})
    assert not replay_derivation(bad, [result]).passed


@pytest.mark.parametrize("source", [
    'r = list_objects()\nYieldObservations([r["result_id"]], "inspect")',
    'r = list_objects()\nitems = r["data"]\nshow(items[0]["obj_id"])',
    'r = object_centroid("chair")\nshow(r["data"]["centroid_normalized"])',
    'r = object_centroid("chair")\nr = {"custom": 1}\nshow(r["custom"])',
])
def test_legal_envelope_examples_and_ambiguous_reassignments_pass(source):
    assert review_method_contract("```python\n" + source + "\n```") == []


@pytest.mark.parametrize("source", [
    'r = list_objects()\nitems = r["data"]\nshow(items["result_id"])',
    'r = object_centroid("chair")\nshow(r["centroid_normalized"])',
    'r = ctx.tools.count_objects("chair")\nshow(r.get("count"))',
    'show(list_objects()[0])',
])
def test_definite_return_shape_conflicts_are_caught_before_gate(source):
    issues = review_method_contract("```python\n" + source + "\n```")
    assert any(i.code == "return_schema_mismatch" and i.line for i in issues)


def test_catalog_separates_model_and_raw_data_schemas():
    row = REGISTRY.interface_catalog(["list_objects"])[0]
    assert row["return_schema_ref"] == TOOL_RESPONSE_VERSION
    assert row["data_annotation"] == "list[dict]"
    assert set(row["return_schema"]["properties"]) == TOOL_RESPONSE_FIELDS
