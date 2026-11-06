"""Synthetic offline checks for the three VSI execution-interface improvements."""

import copy
import hashlib
import importlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from skill3d.online import runner
from skill3d.online.submission import parse_finalization, finalization_program
from skill3d.online.vsi_execution_contract import (
    MEASUREMENT_FIELDS, OBSERVATION_EXAMPLE, PAGING_EXAMPLE, VSI_INTERFACE_NAMES,
    VSI_INTERFACE_SHA256, VSI_INTERFACE_VERSION, finalization_example,
    program_instructions, visual_example,
)
from skill3d.online.vsi_kernel import RestrictedNamespaceKernel
from skill3d.sandbox.action_check import check_action
from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.receipt import ReceiptChain
from skill3d.schemas import CANONICAL_UNIT_BY_QUESTION_TYPE, ToolResult
from skill3d.schemas.tool_response import model_tool_response
from skill3d.synthesis.program_assembler import assemble_program
from skill3d.tools import REGISTRY
from skill3d.verifier.derivation import replay_derivation

ROOT = Path(__file__).resolve().parents[1]


def scene(task="object_counting"):
    return SimpleNamespace(
        question_type=task, frame="world", scene_route="fallback_2d_only",
        question_tool_scope="fallback_2d_only", summary="Synthetic scene",
        available_artifacts=set(), grounding_coverage={}, local_facts={},
        evidence_profile=None, metric_evidence_gate_result=None,
    )


def kernel(task="object_counting"):
    return RestrictedNamespaceKernel(REGISTRY, scene(task), episode_id="synthetic")


def add_result(value, *, tool="list_objects", result_id="host-r1", **updates):
    return ToolResult(
        tool=tool, source_tool=tool, result_id=result_id,
        value=json.dumps(value), request_digest="synthetic-digest", **updates)


def rows(n=22, *, long=False):
    return [{"obj_id": f"chair-{i}", "visible_frames": [i],
             "binding_flags": [{"code": "x" * 850}] if long else [],
             "track_id": None} for i in range(n)]


def read_round(value, *, fields=None, limit=8):
    value.reset_user_namespace()
    code = (
        f'page = ReadNextObservation("host-r1", fields={fields!r}, limit={limit})\n'
        'YieldObservations([page["result_id"]], "Inspect the next page")')
    cell = value.run_cell(code)
    assert cell.error is None, cell.error
    assert cell.terminated == "yield"
    return copy.deepcopy(value._ns["page"])


@pytest.mark.parametrize("raw", [[{"obj_id": "chair-0"}], {"count": 3}, 1.25])
def test_tool_data_retains_shape_and_cannot_mutate_host_evidence(raw):
    value = kernel()
    result = add_result(raw)
    value.tool_results.append(result)
    response = model_tool_response(result)
    data = value._tool_data(response)
    assert type(data) is type(raw) and data == raw
    if isinstance(data, list):
        data[0]["obj_id"] = "changed"
    elif isinstance(data, dict):
        data["count"] = 999
    assert value.tool_results[0].value == json.dumps(raw)
    assert not value.tool_calls and not value.observation_reads
    assert value.cell_vsi_interface_calls[-1]["name"] == "ToolData"


def test_forged_or_edited_envelope_cannot_read_historical_list_without_paging():
    value = kernel()
    result = add_result(rows())
    value.tool_results.append(result)
    response = model_tool_response(result)
    for forged in (
        {k: v for k, v in response.items() if k != "data"},
        {**response, "data": []},
        {**response, "source_tool": "count_objects"},
    ):
        with pytest.raises(ValueError):
            value._tool_data(forged)
    assert len(value.vsi_interface_calls) == 3
    assert all(not r["allowed"] for r in value.vsi_interface_calls)


def test_truncated_pages_resume_without_skipping_or_rerunning():
    value = kernel()
    result = add_result(rows(long=True)[::-1])
    value.tool_results.append(result)
    first = read_round(value, fields=["obj_id", "binding_flags"], limit=8)
    assert 0 < first["returned_range"][1] < 8  # actual char-budget truncation
    first_end = first["returned_range"][1]
    assert first["vsi_read_progress"]["next_cursor"] == first_end
    seen = [r["obj_id"] for r in first["data"]]
    prior_end = first_end
    for _round in range(22):
        page = read_round(value, limit=8)  # selection inherits across resets
        assert page["returned_range"][0] == prior_end
        assert [json.loads(result.value)[p[0]]["obj_id"]
                for p in page["source_item_paths"]] == [r["obj_id"] for r in page["data"]]
        seen.extend(r["obj_id"] for r in page["data"])
        prior_end = page["returned_range"][1]
        if page["vsi_read_progress"]["coverage_complete"]:
            break
    assert seen == [f"chair-{i}" for i in range(22)]
    assert page["next_cursor"] is None and page["coverage_complete"] is False
    assert page["vsi_read_progress"]["covered_ranges"] == [[0, 22]]
    assert value.tool_results == [result] and not value.tool_calls
    assert all(r["allowed"] for r in value.authorization_receipts)
    assert all("vsi_read_progress" not in page and len(json.dumps(
        page, separators=(",", ":"), ensure_ascii=False)) <= 4000
        for page in value.observation_reads)


def test_tail_read_does_not_hide_gap_and_field_selection_has_separate_progress():
    value = kernel()
    value.tool_results.append(add_result(rows()))
    value.mark_cell_start()
    tail = value._read_observation("host-r1", ["obj_id"], offset=18, limit=8)
    assert tail["next_cursor"] is None and not tail["coverage_complete"]
    assert tail["vsi_read_progress"]["next_cursor"] == 0
    page = read_round(value, fields=["obj_id"], limit=8)
    assert page["returned_range"] == [0, 8]
    page = read_round(value, fields=["visible_frames"], limit=8)
    assert page["returned_range"] == [0, 8]
    page = read_round(value, fields=["obj_id"], limit=8)
    assert page["returned_range"] == [8, 16]
    page = read_round(value, fields=["obj_id"], limit=8)
    assert page["vsi_read_progress"]["covered_ranges"] == [[0, 22]]
    assert page["vsi_read_progress"]["coverage_complete"]


def test_reader_data_is_always_list_including_scalar_and_named_collection():
    value = kernel()
    value.tool_results.append(add_result({"count": 3}, tool="count_objects"))
    page = read_round(value, fields=["count"])
    page["data"][0]["count"] = 999
    value.reset_user_namespace()
    assert value._tool_data(page) == [{"count": 3}]
    value.reset_episode_ledger()
    detections = [{"frame_id": i} for i in range(40)]
    value.tool_results.append(add_result({"detections": detections}, tool="detect_objects"))
    page = read_round(value, fields=["detections"])
    assert page["data"] == detections[:8]
    assert page["data_path"] == ["detections"]
    assert page["source_item_paths"] == [["detections", i] for i in range(8)]
    page = read_round(value)
    assert page["returned_range"] == [8, 16]


def test_helper_shares_reader_budget_and_rejects_private_fields():
    value = kernel()
    value.tool_results.append(add_result(rows()))
    code = (
        'ReadObservation("host-r1", fields=["obj_id"], limit=1)\n'
        'ReadNextObservation("host-r1", fields=["obj_id"], limit=1)\n'
        'ReadNextObservation("host-r1", fields=["obj_id"], limit=1)\n'
        'ReadNextObservation("host-r1", fields=["obj_id"], limit=1)\n'
        'YieldObservations(["host-r1"], "Inspect")')
    cell = value.run_cell(code)
    assert cell.error and "budget exhausted" in cell.error
    assert cell.answer_untrusted and len(value.observation_reads) == 3
    value.reset_user_namespace()
    cell = value.run_cell('ReadNextObservation("host-r1", fields=["ground_truth"])')
    assert cell.error and cell.answer_untrusted
    assert len(value.observation_reads) == 3 and not value.tool_calls


@pytest.mark.parametrize(("task", "raw", "expected"), [
    ("object_abs_distance", {"distance": 1.234567890123, "unit": "m"}, 1.234567890123),
    ("object_size_estimation", {"extent_longest_metric": 0.1234567890123}, 12.34567890123),
    ("room_size_estimation", {"room_area_m2": 19.234567890123}, 19.234567890123),
    ("object_counting", {"count": 3}, 3),
])
def test_measurement_submission_uses_real_id_full_precision_and_unchanged_replay(task, raw, expected):
    value = kernel(task)
    tool, field, _, unit = MEASUREMENT_FIELDS[task]
    result = add_result(raw, tool=tool)
    value.tool_results.append(result)
    response = model_tool_response(result)
    response["data"][field] = 999  # helper must read raw host value, never this edit
    payload = value._measurement_answer(response)
    assert payload.value == pytest.approx(expected, rel=0, abs=1e-13)
    assert payload.unit == unit and payload.used_result_ids == ["host-r1"]
    assert payload.derivation["parameters"]["field"] == field
    assert replay_derivation(payload, [result], question_type=task).passed
    cell = value.run_cell('ReturnAnswer(MeasurementAnswer("host-r1"))')
    assert not cell.error and cell.terminated == "answer"
    assert value.answer_slot.payload == payload
    assert not value.cell_calls and not value.observation_reads
    assert value.tool_results[0] == result
    rounded = payload.model_copy(update={"value": round(float(payload.value), 2)})
    if rounded.value != payload.value:
        replay = replay_derivation(rounded, [result], question_type=task)
        assert not replay.passed and "value_mismatch" in {i.code for i in replay.issues}


@pytest.mark.parametrize(("task", "tool", "raw"), [
    ("object_abs_distance", "object_surface_distance", {"distance": 1, "unit": "normalized"}),
    ("object_abs_distance", "object_surface_distance", {"distance_normalized": 1}),
    ("object_abs_distance", "object_surface_distance", {"distance": True, "unit": "m"}),
    ("object_abs_distance", "object_surface_distance", {"distance": -1, "unit": "m"}),
    ("object_abs_distance", "object_surface_distance", {"distance": float("inf"), "unit": "m"}),
    ("object_abs_distance", "object_surface_distance", {"distance": 1, "unit": "m", "status": "fault"}),
    ("object_abs_distance", "object_surface_distance", {"distance": 1, "unit": "m", "service_healthy": False}),
    ("object_size_estimation", "object_3d_extent", {"extent_normalized": [1, 2, 3]}),
    ("object_size_estimation", "object_surface_distance", {"distance": 1, "unit": "m"}),
    ("room_size_estimation", "plane_fit_room_size", {"room_diagonal_normalized": 5}),
    ("object_counting", "count_objects", {"count": 2.5}),
    ("object_rel_distance", "object_surface_distance", {"distance": 1, "unit": "m"}),
])
def test_measurement_invalid_data_does_not_turn_into_a_number(task, tool, raw):
    value = kernel(task)
    value.tool_results.append(add_result(raw, tool=tool))
    cell = value.run_cell('ReturnAnswer(MeasurementAnswer("host-r1"))')
    assert cell.error and cell.answer is None and cell.answer_untrusted
    assert value.cell_vsi_interface_calls[-1]["allowed"] is False


@pytest.mark.parametrize("updates", [
    {"status": "failed", "error": "synthetic"}, {"invalidated_by": ["geometry_3d"]},
    {"error_code": "confidence_gate"}, {"authorization": {"allowed": False}},
    {"unresolved": True}, {"missing": True},
])
def test_failed_or_withdrawn_result_is_revalidated_on_every_helper_call(updates):
    value = kernel("object_abs_distance")
    changes = dict(updates)
    unresolved = changes.pop("unresolved", False)
    missing = changes.pop("missing", False)
    result = add_result({"distance": 1, "unit": "m"}, tool="object_surface_distance", **changes)
    if not missing:
        value.tool_results.append(result)
    if unresolved:
        value.unresolved_result_ids.add("host-r1")
    for code in (
        'ReturnAnswer(MeasurementAnswer("host-r1"))',
        'ReadNextObservation("host-r1")',
    ):
        value.reset_user_namespace()
        cell = value.run_cell(code)
        assert cell.error and cell.answer is None and cell.answer_untrusted
    assert not value.observation_reads


@pytest.mark.parametrize("name", sorted(VSI_INTERFACE_NAMES))
def test_helpers_are_reserved_host_calls_and_disabled_during_finalization(name):
    assert ast_guard(f'{name}("host-r1")', allowed_tools=set()).ok
    assert ast_guard(f'{name}("host-r1")', allowed_tools=set()).allowed_tool_calls == []
    for code in (
        f"{name} = 1", f"def {name}():\n    pass",
        f"def solve({name}):\n    pass", f"f = lambda {name}: 1",
        f"items = [{name} for {name} in []]",
        f"try:\n    pass\nexcept ValueError as {name}:\n    pass",
    ):
        assert not ast_guard(code, allowed_tools=set()).ok
    value = kernel()
    with pytest.raises(ValueError, match="reserved"):
        value.inject(**{name: 1})
    value.reset_user_namespace()
    value.set_tools_enabled(False)
    cell = value.run_cell(f'{name}("host-r1")')
    assert cell.error and "ordinary-phase" in cell.error and cell.answer_untrusted
    assert not value.tool_calls


@pytest.mark.parametrize("code", [
    "import os", "open('synthetic')", "scene.depth_maps", "frames[0].ctypes",
    "import numpy as np\nnp.fromfile('synthetic')", "eval('1')",
    "class Escape:\n    pass", "try:\n    pass\nexcept:\n    pass",
])
def test_existing_sandbox_denials_remain_intact(code):
    assert not ast_guard(code).ok


@pytest.mark.parametrize("task", sorted(CANONICAL_UNIT_BY_QUESTION_TYPE))
def test_complete_prompt_examples_compile_terminate_and_parse_in_their_actual_phase(task):
    value = kernel(task)
    for source in re.findall(r"```python\n(.*?)```", program_instructions(task), re.S):
        compile(source, "<synthetic-prompt-example>", "exec")
        assert ast_guard(source).ok
        assert check_action(source, tool_names=set(REGISTRY.names())).ok
    cell = value.run_cell(visual_example(task))
    assert not cell.error and cell.terminated == "answer"
    raw = finalization_example(task)
    payload = parse_finalization(raw, question_type=task, options=["A. first", "B. second"],
                                 valid_result_ids=[], readable_images=True)
    assert type(json.loads(raw)["value"]) is not bool
    value.reset_user_namespace()
    value.set_tools_enabled(False)
    cell = value.run_cell(finalization_program(payload).program_source)
    assert not cell.error and cell.terminated == "answer"
    assert not value.tool_calls


def test_runner_executes_local_kernel_and_reset_retains_reader_state():
    episode = SimpleNamespace(qa_id="synthetic", question_type="object_counting")
    cfg = runner.OnlineRunConfig(mode="real", online_model_enabled=False)
    receipts = ReceiptChain("synthetic")
    value, cell, trace = runner._execute_program(
        episode, assemble_program(visual_example("object_counting"), []),
        scene(), [], cfg, receipts)
    assert isinstance(value, RestrictedNamespaceKernel) and cell.answer == "1"
    value.tool_results.append(add_result(rows()))
    program = assemble_program(PAGING_EXAMPLE.replace("ACTUAL_RESULT_ID", "host-r1"), [])
    value, cell, trace = runner._execute_program(episode, program, scene(), [], cfg, receipts, kernel=value)
    assert cell.terminated == "yield" and not trace.calls and not trace.results
    assert value.observation_reads[-1]["returned_range"] == [0, 8]
    value, cell, trace = runner._execute_program(episode, program, scene(), [], cfg, receipts, kernel=value)
    assert value.observation_reads[-1]["returned_range"] == [8, 16]
    assert [r["name"] for r in value.cell_vsi_interface_calls] == ["ReadNextObservation", "ToolData"]
    json.dumps(value.cell_vsi_interface_calls)
    feedback = runner._yield_feedback(cell, value, scene(), cfg)
    assert '"covered_ranges": [[0, 16]]' in feedback


def test_source_report_and_runner_prompt_have_truthful_supplement_identity():
    from skill3d.infra.vsi_runner_source import source_report

    report = source_report()
    for name in ("skill3d.online.vsi_kernel", "skill3d.online.vsi_execution_contract",
                 "skill3d.sandbox.ast_guard", "skill3d.online.runner"):
        assert Path(report["local_module_paths"][name]).is_relative_to(ROOT / "src")
    assert report["model_requests"] == 0
    assert Path(importlib.import_module("skill3d.sandbox.kernel").__file__).is_relative_to(
        ROOT.parent / "src")  # shared runtime stays read-only
    episode = SimpleNamespace(question="Synthetic question", question_type="object_counting", options=[])
    prompt, _plan = runner._build_prompt_ex(episode, scene(), scene(), [], None)
    assert VSI_INTERFACE_VERSION in prompt and OBSERVATION_EXAMPLE in prompt
    assert len(VSI_INTERFACE_SHA256) == 64 and runner.PROMPT_TEMPLATE_VERSION.endswith("+vsi-interfaces-1")
    assert hashlib.sha256(prompt.encode()).hexdigest()


@pytest.mark.parametrize("finalizing", [False, True])
def test_actual_synthesis_request_uses_phase_format_and_separate_contract_identity(finalizing):
    import numpy as np
    from skill3d.synthesis.execution_contract import ExecutionContract

    class OfflineClient:
        last_generation = {}
        last_usage = {}

        def set_request_context(self, context):
            self.context = context

        def chat(self, messages, **_kwargs):
            self.messages = messages
            return (finalization_example("object_size_estimation") if finalizing
                    else "```python\n" + visual_example("object_size_estimation") + "```")

    client = OfflineClient()
    episode = SimpleNamespace(
        qa_id="synthetic", scene_name="synthetic", frame_set=None,
        question="Estimate the largest dimension in centimeters.",
        question_type="object_size_estimation", options=[])
    cfg = runner.OnlineRunConfig(mode="real", online_model_enabled=False)
    result = runner._synthesize(
        episode, scene("object_size_estimation"), scene("object_size_estimation"),
        [], cfg, client, pixels=[np.zeros((8, 8, 3), dtype=np.uint8)],
        trigger="finalize" if finalizing else "initial")
    assert result.program is not None, result.note
    context = client.context
    assert context["vsi_interface_version"] == VSI_INTERFACE_VERSION
    assert context["vsi_interface_contract_sha256"] == VSI_INTERFACE_SHA256
    assert context["execution_contract_sha256"] == ExecutionContract(
        cfg.retrieval_policy.method_context_max_chars).sha256
    assert context["tool_execution_allowed"] is (not finalizing)
    assert context["prompt_version"].endswith("+vsi-interfaces-1")
    if finalizing:
        assert finalization_example(episode.question_type) in result.prompt
        assert "no markdown fences, Python" in result.prompt
    else:
        assert "ToolData(response)" in result.prompt
        assert OBSERVATION_EXAMPLE in result.prompt
    assert check_action(result.program.program_source, finalizing=finalizing,
                        tool_names=set(REGISTRY.names())).ok
