"""New finalizer runs inside the existing solver budget and M11 rejection path."""

import json

import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig
from skill3d.synthesis.request_context import current_request_phase
from test_tool_contract_recovery import _write_v6_artifact


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    item = load_synthetic_items(
        "inner_validation", question_types=["room_size_estimation"],
        frame_size=(120, 160), n_frames=32, seed=0)[0]
    artifact = _write_v6_artifact(tmp_path_factory.mktemp("finalizer"), with_poses=True)
    return item, artifact


def response(**updates):
    return json.dumps({"action": "answer", "value": 3, "basis": "visual_estimate",
                       "used_result_ids": [], "derivation": None, **updates})


class Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((current_request_phase(), messages, kwargs))
        assert len(self.calls) <= len(self.replies), "unexpected extra request"
        return self.replies[len(self.calls) - 1]


def run(source, tmp_path, replies):
    item, artifact = source
    client = Client(replies)
    out = runner.run_episode(
        item.episode, item.pixels, OnlineRunConfig(
            mode="real", reuse_artifact=artifact, max_solver_rounds=2,
            finalization_rounds=1, eval_visual_fallback=False,
            trace_dir=str(tmp_path / "traces")), llm=client)
    return out, client


def test_assignment_failure_then_json_answer_uses_same_budget_and_m11(source, tmp_path):
    out, client = run(source, tmp_path, ["answer = 3", response()])
    assert out.answer == "3" and out.final_state == "answer"
    assert len(client.calls) == out.agent_rounds == 2
    assert [call[0] for call in client.calls] == ["program", "finalization"]
    assert out.rounds[0]["error_code"] == "program_missing_action"
    assert out.rounds[0]["executed"] is False
    assert out.rounds[1]["submission_accepted"] is True and out.verify.passed
    assert out.program_trace.calls == []
    last = out.rounds[1]
    assert last["finalization"]["host_generated_program"]
    assert last["finalization"]["response_format_enforced_by_service"] is False
    assert last["response_text"] == response()
    messages = client.calls[1][1]
    assert len(messages) == 1 and len(messages[0]["content"]) == 33
    text = messages[0]["content"][0]["text"]
    assert "answer = 3" not in text and "## Reference method" not in text
    assert "No repair request is added" in text
    assert "omit_complete_method" in text
    assert len(out.image_ledger["rounds"]) == 2


@pytest.mark.parametrize("bad", [
    "answer = 3", "ReturnAnswer(3)", response(value=float("nan")),
    response(used_result_ids=["made-up"]), response(unit="m"),
    response(action="tool"), response(basis="tool_derived"),
])
def test_bad_finalization_has_no_repair_or_tool_execution(source, tmp_path, bad):
    out, client = run(source, tmp_path, ["answer = 3", bad])
    assert out.final_state == "run_error" and out.answer is None
    assert len(client.calls) == out.agent_rounds == 2
    assert out.rounds[1]["executed"] is False
    assert out.failure_code in (
        "model_output_format_error", "answer_contract_rejected", "finalization_tool_call_disallowed")
    assert out.terminal_failure["termination_reason"] == "finalization_failed"
    assert out.mra_value == 0


def test_host_generated_submission_still_uses_geometry_acceptance(source, tmp_path, monkeypatch):
    from skill3d.verifier.geometry_oracle import GeometryVerifyResult

    calls = []
    def reject(trace, handle, answer, **kwargs):
        calls.append(answer)
        return GeometryVerifyResult(passed=False, checks={"synthetic": False},
                                    violations=["synthetic"], issues=[])
    monkeypatch.setattr(runner, "geometry_verify", reject)
    out, client = run(source, tmp_path, ["answer = 3", response()])
    assert len(client.calls) == 2 and calls == ["3"]
    assert out.final_state == "run_error" and out.answer is None
    assert out.failure_code == "geometry_verification_rejected"
    assert out.rounds[-1]["rejected_submission"]["given"] is True
    assert out.rounds[-1]["submission_accepted"] is False
