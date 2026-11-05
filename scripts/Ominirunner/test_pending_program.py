"""Fresh requests retain complete unexecuted plans without relaxing the parser."""
from copy import deepcopy
import json

import pytest

from .audit import replay_contexts
from .native import ProgramFormatError, parse_program, pending_program
from .test_contracts import fenced, offline_episode, offline_only, scene_data


def long_sampling_plan():
    return ("def solve(ctx):\n"
            "    measurement_reviews = []\n"
            "    # " + "original plan identity " * 30 + "\n"
            '    r = ctx.tools.point_depth(view_index=0,x=10,y=10,'
            'targets=[{"target":"chair","surface":"center"}])\n'
            '    YieldObservations([r["result_id"]],reason="Review actual target")')


@pytest.mark.parametrize("wrapper", ["xml", "prefix", "tail"])
def test_long_unexecuted_plan_is_delivered_whole_then_executes_once(
        tmp_path, monkeypatch, scene_data, wrapper):
    program = long_sampling_plan()
    valid = "```python\n" + program + "\n```"
    invalid = ("<program>\n" + program + "\n</program>" if wrapper == "xml" else
               "Here is the plan:\n" + valid if wrapper == "prefix" else valid + "\n</think>")
    result = offline_episode(tmp_path, monkeypatch, scene_data, scene_data[0], [
        invalid, valid,
        fenced('ReturnAnswer(3,evidence={"kind":"visual_estimate","basis":"Independent RGB estimate"})')])
    first, second = result["rounds"][:2]
    assert first["status"] == "program_parse_error" and first["executed"] is False
    assert first["tool_counts_before"]["calls"] == 0 and not first.get("tool_receipts")
    assert first["pending_program_state"]["code"] == program
    assert second["request"]["pending_program"]["code_delivered"]
    assert second["pending_program_state"] is None and result["pending_program_state"] is None
    context = json.loads((tmp_path / "01.messages.json").read_text())
    text = context[0]["content"][1]["text"]
    snapshot = json.loads(text.split("\nUNEXECUTED MODEL PROGRAM:", 1)[1].split("\n", 1)[1].split("\n", 1)[0])
    assert snapshot["code"] == program and snapshot["execution_authorized"] is False
    assert result["status"] == "answered" and result["tool_counts"]["calls"] == 1
    changed = deepcopy(result)
    changed["rounds"][0]["pending_program_state"]["code"] += "\n# altered"
    with pytest.raises(ValueError, match="round_pending_program_changed"):
        list(replay_contexts(scene_data[0], changed, *scene_data[1:4]))


def test_ambiguous_fences_remain_quoted_invalid_response():
    output = fenced("ReturnAnswer(1)") + "\n" + fenced("ReturnAnswer(2)")
    with pytest.raises(ProgramFormatError) as caught:
        parse_program(output)
    snapshot = pending_program(output, caught.value)
    assert snapshot["source"] == "invalid_response" and snapshot["code"] == output
    assert snapshot["execution_authorized"] is False


def test_final_keeps_unexecuted_identity_without_program_code_or_execution(
        tmp_path, monkeypatch, scene_data):
    program = long_sampling_plan()
    invalid = "<program>\n" + program + "\n</program>"
    result = offline_episode(tmp_path, monkeypatch, scene_data, scene_data[0], [
        *[invalid] * 5, '{"answer":3,"measurement_reviews":[],"evidence":{"kind":"visual_estimate","basis":"RGB estimate"}}'])
    assert result["status"] == "answered" and result["tool_counts"]["calls"] == 0
    assert result["pending_program_state"]["code"] == program
    request = result["rounds"][-1]["request"]
    assert request["phase"] == "final" and request["pending_program"]["code_delivered"] is False
    text = json.loads((tmp_path / "05.messages.json").read_text())[0]["content"][1]["text"]
    assert '"code_omitted":"tools_closed"' in text and "def solve(ctx)" not in text
