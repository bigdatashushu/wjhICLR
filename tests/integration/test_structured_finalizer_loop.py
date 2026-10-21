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


def test_missing_yield_executes_requested_observation_in_first_request(source, tmp_path):
    out, client = run(source, tmp_path, [
        "result = list_objects()\nprint(result['data'])", response()])
    assert len(client.calls) == 2 and out.final_state == "answer"
    first = out.rounds[0]
    assert first["executed"] and first["observation_adapter"]["yielded"]
    assert first["observation_adapter"]["answer_inference"] is False
    assert first["observation_adapter"]["result_ids"]
    assert first["program_source"] == "result = list_objects()\nprint(result['data'])"
    assert out.yield_count == 1 and not out.answer_untrusted


def test_caught_failure_prediction_is_rejected_and_independent_visual_can_recover(source, tmp_path):
    bad = (
        "try:\n    exists_in_scene('anything')\nexcept Exception:\n    pass\n"
        "ReturnAnswer(AnswerPayload(value=3, unit='m2', basis='visual_estimate'))")
    out, client = run(source, tmp_path, [bad, response(value=4)])
    first = out.rounds[0]
    assert first["rejected_submission"]["answer"] == "3"
    assert first["submission_accepted"] is False
    assert first["submission_rejection"]["answer_untrusted"] is True
    assert first["execution_diagnostics"]["answer_untrusted"] is True
    assert first["error_code"] == "tool_precondition_rejected"
    assert out.answer == "4" and out.final_state == "answer"
    assert not out.answer_untrusted and out.verify.passed
    assert len(client.calls) == 2
    assert out.rounds[1]["results"] == [] and out.used_result_ids == []
    assert out.authorization_receipts[0]["allowed"] is False


def test_six_request_cap_counts_missing_actions_and_finalizer(source, tmp_path):
    item, artifact = source
    client = Client(["answer = 3"] * 5 + [response()])
    out = runner.run_episode(
        item.episode, item.pixels, OnlineRunConfig(
            mode="real", reuse_artifact=artifact, max_solver_rounds=6,
            finalization_rounds=1, max_retries_per_operation=10,
            eval_visual_fallback=False, trace_dir=str(tmp_path / "six")), llm=client)
    assert len(client.calls) == out.agent_rounds == 6
    assert [phase for phase, _, _ in client.calls] == ["program"] * 5 + ["finalization"]
    assert all(not row["executed"] for row in out.rounds[:5])
    assert out.answer == "3" and out.final_state == "answer"


def test_rejected_prediction_program_is_audit_only_and_not_in_ordinary_recovery(source, tmp_path):
    item, artifact = source
    marker = "AUDIT_ONLY_REJECTED_PROGRAM"
    bad = (
        f"# {marker}\ntry:\n    exists_in_scene('anything')\nexcept Exception:\n    pass\n"
        "ReturnAnswer(AnswerPayload(value=3, unit='m2', basis='visual_estimate'))")
    client = Client([bad, "ReturnAnswer(AnswerPayload(value=4, unit='m2', basis='visual_estimate'))"])
    out = runner.run_episode(
        item.episode, item.pixels, OnlineRunConfig(
            mode="real", reuse_artifact=artifact, max_solver_rounds=3,
            finalization_rounds=1, eval_visual_fallback=False,
            trace_dir=str(tmp_path / "independent")), llm=client)
    assert out.answer == "4" and not out.answer_untrusted
    assert len(client.calls) == 2 and client.calls[1][0] == "program"
    assert out.rounds[0]["recovery_previous_program_omitted"] == "rejected_submission"
    assert marker in out.rounds[0]["response_text"]
    assert marker not in json.dumps(client.calls[1][1])
    assert out.rounds[1]["submission_accepted"] is True
