"""State, prompt examples, and bounded repairs of the actual compact solver."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re

import pytest

from skill3d.small_image.cv3d_context import CompactContext, FEEDBACK_VERSION
from skill3d.small_image.cv3d_protocol import FrozenCV3DSkills, program_contract, VERSION
from skill3d.sandbox.ast_guard import ast_guard
from test_cv3d_compact_program import config, prepared, prompt, solver
from test_small_image_runner import ANSWER, YIELD, FakeClient, problem


def context_text(client, index):
    return client.calls[index][0][0]["content"][0]["text"]


def feedback(text):
    return json.loads(text.split("Valid observations: ", 1)[1].split("\n\n", 1)[0])


def progress(text):
    return json.loads(text.split("Progress: ", 1)[1].split("\n", 1)[0])


def expanded(packet):
    full = []
    for item in packet["observations"]:
        copied = {**packet["common"], **deepcopy(item)}
        copied.pop("round")
        copied["data"] = {
            **packet["tool_constants"].get(item["source_tool"], {}), **copied["data"]}
        full.append(copied)
    return full


def test_prompt_example_executes_actual_kernel_and_instructions_have_one_owner(tmp_path):
    root = Path(__file__).resolve().parents[1]
    skills = FrozenCV3DSkills(
        root / "skill_library/candidates/cv3d-compact-testdev-2/frozen_skills.json")
    for task in ("Depth", "Distance"):
        p = replace(problem(tmp_path), task_type=task)
        example = re.search(r"```python\n(.*?)```", program_contract(task), re.S)[1]
        assert ast_guard(example, allowed_tools={"view_depth_summary"}).ok
        client = FakeClient([example] * 5 + ['{"answer":"B"}'])
        client.build_payload = lambda messages, **kwargs: {"messages": messages, **kwargs}
        from skill3d.small_image.runner import SmallImageSolver
        result = SmallImageSolver(
            client, config=config(capture_responses=True), skill_text=skills.texts[task], official_prompt=prompt(p))(
                p, prepared=prepared(p))
        assert result.status == "answered" and result.succeeded_tool_calls == 5
        assert result.request_count == 6
        for i in range(6):
            texts = [part["text"] for part in client.calls[i][0][0]["content"] if part["type"] == "text"]
            joined = "\n".join(texts)
            assert joined.count("Darker grayscale") == 1
            assert joined.count("0 <= x < 10") == 1
            assert f"round {i+1}/6" in joined
            assert joined.count(skills.texts[task]) <= 1
        first, final = context_text(client, 0), context_text(client, 5)
        assert "No previous round" in first and "variables have been cleared" not in first
        for forbidden in ("def solve(ctx)", "ReturnAnswer", "YieldObservations", "Program tools:"):
            assert forbidden not in final
        assert "## Final assessment" in final
        assert "## Measure useful uncertainty" not in final
        assert "## Measure the relation" not in final
        from skill3d.small_image.blink_protocol import sha_value
        from skill3d.small_image.cv3d_protocol import skill_for_stage
        assert result.rounds[-1]["delivered_skill_scope"] == "final_assessment"
        assert result.rounds[-1]["delivered_skill_sha256"] == sha_value(
            skill_for_stage(skills.texts[task], finalizing=True).strip())
        assert "Tools are disabled" in final and '"answer"' in final
        assert "depth image" in skills.texts[task]


@pytest.mark.parametrize("bad,code", [
    ("this is not a program", "program_parse_error"),
    ("def solve(ctx):\n    x = 1", "program_missing_action"),
    ("def solve(ctx):\n    ctx.tools.ReturnAnswer('A')", "interface_misuse"),
    ("def solve(ctx):\n    r = ctx.tools.view_depth_summary()\n    x = r.data\n    YieldObservations([])", "interface_misuse"),
    ("def solve(ctx):\n    ctx.ReturnAnswer('A')", "interface_misuse"),
    ("def solve(ctx):\n    r = ctx.tools.view_depth_summary()\n    x = r.status\n    YieldObservations([])", "interface_misuse"),
    ("def solve(ctx):\n    ReturnAnswer(AnswerPayload(value='invalid', unit='option', basis='mixed'))", "invalid_answer"),
    ("def solve(ctx):\n    YieldObservations(['invented'])", "invalid_observation_reference"),
])
def test_correctable_error_receives_specific_feedback_without_extra_budget(tmp_path, bad, code):
    p = problem(tmp_path)
    client = FakeClient([bad, ANSWER])
    result = solver(client, p)(p, prepared=prepared(p))
    assert result.status == "answered" and result.request_count == 2
    assert result.rounds[0]["status"] != "answered"
    state = progress(context_text(client, 1))
    assert state["last_error"]["code"] == code
    assert state["last_error"]["hint"]
    assert len(state["previous_rounds"]) == 1


def test_repair_after_successful_measurement_preserves_raw_and_compact_evidence(tmp_path):
    p = problem(tmp_path)
    bad = ("def solve(ctx):\n"
           "    r = ctx.tools.point_depth(0, 5, 5)\n"
           "    x = r.status\n"
           "    YieldObservations([r['result_id']])")
    client = FakeClient([bad, ANSWER])
    result = solver(client, p)(p, prepared=prepared(p))
    assert result.status == "answered" and result.succeeded_tool_calls == 1
    assert result.rounds[0]["status"] == "run_error"
    raw = result.rounds[0]["observations"]
    packet = feedback(context_text(client, 1))
    assert packet["schema"] == FEEDBACK_VERSION
    assert expanded(packet) == raw
    assert packet["observations"][0]["data"]["depth"] == pytest.approx(1.5)
    assert packet["observations"][0]["round"] == 1
    assert progress(context_text(client, 1))["last_error"]["line"] == 3


def test_five_failed_programs_use_five_slots_then_json_finalization(tmp_path):
    p = problem(tmp_path)
    client = FakeClient(["def solve(ctx): ("] * 5 + ['{"answer":"B"}', ANSWER])
    result = solver(client, p)(p, prepared=prepared(p))
    assert result.status == "answered" and len(client.calls) == 6
    final = context_text(client, 5)
    assert "Return valid Python" not in final and "def solve(ctx)" not in final
    assert len(progress(final)["previous_rounds"]) == 5
    assert progress(final)["last_error"] == {"code": "program_parse_error"}


@pytest.mark.parametrize("bad,status", [
    ("import os\n" + ANSWER, "program_rejected"),
    ("def solve(ctx):\n    r = ctx.tools.point_depth(0, 9999, 3)\n    YieldObservations([])", "run_error"),
    (TimeoutError(), "service_error"),
])
def test_security_tool_and_service_failures_stay_terminal(tmp_path, bad, status):
    p = problem(tmp_path)
    client = FakeClient([bad, ANSWER])
    result = solver(client, p)(p, prepared=prepared(p))
    assert result.status == status and len(client.calls) == 1


def test_notes_are_bounded_unverified_and_do_not_contaminate_observations(tmp_path):
    p = problem(tmp_path)
    note = "red target at selected surface; next compare blue. " + "x" * 500
    program = YIELD.replace("YieldObservations([response['result_id']])",
                            f"YieldObservations([response['result_id']], reason={note!r})")
    client = FakeClient([program, ANSWER])
    result = solver(client, p)(p, prepared=prepared(p))
    text = context_text(client, 1)
    state = progress(text)
    assert state["model_notes"][0]["kind"] == "model_note_unverified"
    assert state["model_notes"][0]["text"] == note[:400]
    assert state["model_notes"][0]["result_ids"] == [
        result.rounds[0]["observations"][0]["result_id"]]
    assert note[:30] not in json.dumps(feedback(text))
    assert program not in text and "model_notes" in text


def test_metadata_compaction_is_lossless_for_multiple_tools_and_changed_definitions():
    source = {
        "result_id": "one", "source_tool": "point_depth", "status": "ok",
        "schema_version": "tool-response/2.0", "evidence_version": "small-image-tools/1",
        "data": {"pixel_xy": [1.2, 2.4], "depth": 3.287484645843506, "n_valid": 25,
                 "unit": "per_view_relative", "coordinate_convention": "original pixels",
                 "definition": "local median"},
    }
    records = [source, {**deepcopy(source), "result_id": "two"},
               {**deepcopy(source), "result_id": "three", "source_tool": "pair_point_distance"}]
    records[2]["data"] = {"distance": 1.23456789, "a_xy": [2, 3], "b_xy": [4, 5],
                          "unit": "per_view_relative", "definition": "3D Euclidean"}
    ctx = CompactContext()
    ctx.add_observations(records[:2], 1)
    ctx.add_observations(records[2:], 2)
    assert expanded(ctx.feedback()) == records
    assert len(json.dumps(ctx.feedback())) < len(json.dumps(records))
    assert ctx.feedback()["observations"][2]["round"] == 2
    changed = {**deepcopy(source), "result_id": "four", "evidence_version": "other-version"}
    changed["data"]["definition"] = "different definition"
    ctx.add_observations([changed], 3)
    assert expanded(ctx.feedback()) == records + [changed]
    assert "evidence_version" not in ctx.feedback()["common"]
    assert "definition" not in ctx.feedback()["tool_constants"]["point_depth"]
    with pytest.raises(ValueError, match="observation identity changed"):
        ctx.add_observations([{**source, "data": {}}], 4)
    ctx.record(1, "yielded", note="same note", result_ids=["one"])
    ctx.record(2, "yielded", note="same note", result_ids=["one"])
    assert len(ctx.notes) == 1 and ctx.notes[0]["round"] == 2
    assert len(ctx.history) == 2
