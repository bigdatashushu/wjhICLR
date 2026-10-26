"""Real kernel/M11 recovery using synthetic artifacts and a bounded fake client."""

import json

import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.synthesis.request_context import current_request_phase
from skill3d.trace.store import TraceStore
from test_tool_contract_recovery import _write_v6_artifact


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    item = load_synthetic_items(
        "inner_validation", question_types=["room_size_estimation"],
        frame_size=(120, 160), n_frames=32, seed=0)[0]
    artifact = _write_v6_artifact(
        tmp_path_factory.mktemp("submission-boundary-artifact"), with_poses=True)
    return item, artifact


@pytest.fixture
def kernels(monkeypatch):
    instances = []

    class RecordingKernel(RestrictedNamespaceKernel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cell_starts = []
            self.resets = []
            instances.append(self)

        def snapshot(self):
            return {
                "given": self.answer_slot.given, "answer": self.answer_slot.answer,
                "payload": self.answer_slot.payload,
                "adapter_problems": list(self.answer_slot.adapter_problems),
                "names": set(self._ns), "unresolved": set(self.unresolved_result_ids),
                "invalidated": {r.result_id: list(r.invalidated_by or [])
                                for r in self.tool_results},
            }

        def run_cell(self, code, **kwargs):
            self.cell_starts.append(self.snapshot())
            return super().run_cell(code, **kwargs)

        def reset_user_namespace(self):
            super().reset_user_namespace()
            self.resets.append(self.snapshot())

    def no_live_client(*args, **kwargs):
        raise AssertionError("Live model client construction is forbidden in these tests")

    monkeypatch.setattr(runner, "RestrictedNamespaceKernel", RecordingKernel)
    monkeypatch.setattr(runner, "_make_online_client", no_live_client)
    return instances


class Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((current_request_phase(), messages, kwargs))
        assert len(self.calls) <= len(self.replies), "unexpected extra model request"
        reply = self.replies[len(self.calls) - 1]
        return reply(messages) if callable(reply) else reply


class NoObjectModel:
    def chat(self, *args, **kwargs):
        raise AssertionError("Synthetic room-area episodes need no object model")


def _json_answer(**updates):
    return json.dumps({
        "action": "answer", "value": 4, "basis": "visual_estimate",
        "used_result_ids": [], "derivation": None, **updates})


def _ordinary_answer(**updates):
    fields = {"value": 3, "unit": "m2", "basis": "mixed", **updates}
    return f"staged_marker = 'audit-only'\nReturnAnswer(AnswerPayload(**{fields!r}))"


def _run(source, tmp_path, replies, *, rounds, finalization_rounds=1, retries=2):
    item, artifact = source
    client = Client(replies)
    store = TraceStore(tmp_path / "traces")
    out = runner.run_episode(
        item.episode, item.pixels,
        OnlineRunConfig(
            mode="real", reuse_artifact=artifact,
            max_solver_rounds=rounds, finalization_rounds=finalization_rounds,
            max_retries_per_operation=retries, eval_visual_fallback=False,
            deterministic_replay=True, trace_dir=str(tmp_path / "traces")),
        llm=client, m5_llm=NoObjectModel(), trace_store=store)
    saved = json.loads((store.root / "episode_trace.jsonl").read_text())
    assert saved["rounds"] == out.episode_trace.model_dump(mode="json")["rounds"]
    return out, client


def _assert_cleared(snapshot):
    assert not snapshot["given"] and snapshot["answer"] is None
    assert snapshot["payload"] is None and snapshot["adapter_problems"] == []
    assert "staged_marker" not in snapshot["names"]


def _quarantine_prefix(kernels):
    def poison(messages):
        rid = kernels[0].tool_results[0].result_id
        assert rid in json.dumps(messages)
        return _ordinary_answer(used_result_ids=[rid, "nonexistent-synthetic-result"])

    return [
        "listed = list_objects()\nYieldObservations([listed['result_id']], 'synthetic listing')",
        poison,
    ]


def _assert_natural_quarantine(out, kernel):
    first, second = out.rounds[:2]
    rid = first["results"][0]["result_id"]
    assert first["results"][0]["status"] == "ok"
    assert first["results"][0]["authorization"]["allowed"] is True
    assert second["verification_result_ids"] == [rid]
    assert second["submission_accepted"] is False
    assert second["invalidated_result_ids"] == []
    assert rid in kernel.unresolved_result_ids
    assert kernel.tool_results[0].status == "ok"
    assert not kernel.tool_results[0].invalidated_by
    return rid


def test_ordinary_quarantined_reference_is_rejected_then_visual_recovers(
        source, tmp_path, kernels):
    def reuse(messages):
        rid = kernels[0].tool_results[0].result_id
        assert rid in kernels[0].unresolved_result_ids
        _assert_cleared(kernels[0].snapshot())
        return _ordinary_answer(used_result_ids=[rid])

    out, client = _run(
        source, tmp_path, [*_quarantine_prefix(kernels), reuse, _json_answer()], rounds=4)
    assert len(kernels) == 1
    kernel = kernels[0]
    rid = _assert_natural_quarantine(out, kernel)
    rejected = out.rounds[2]
    assert rejected["submission_accepted"] is False
    assert rejected["error_code"] == "geometry_verification_rejected"
    assert rejected["verification_result_ids"] == []
    assert rejected["rejected_submission"]["payload"]["used_result_ids"] == [rid]
    assert rejected["rejected_submission"]["answer"] == "3"
    assert rejected["rejected_submission"]["given"] is True
    issue = next(i for i in rejected["geometry_verify"]["issues"]
                 if i["check"] == "unresolved_reference")
    assert issue["result_id"] == rid and issue["invalidate_result"] is False
    assert issue["failure_cause"]["code"] == "unresolved_reference"
    assert rejected["geometry_verify"]["derivation_replay"]["passed"] is True
    assert rejected["invalidated_result_ids"] == []
    assert rejected["downgraded_capabilities"] == {}
    _assert_cleared(kernel.cell_starts[3])
    assert rid in kernel.cell_starts[3]["unresolved"]
    assert out.answer == "4" and out.final_state == "answer" and out.verify.passed
    assert out.rounds[-1]["results"] == [] and out.used_result_ids == []
    assert out.rounds[-1]["submission_accepted"] is True
    assert len(client.calls) == out.agent_rounds == 4
    assert [phase for phase, _, _ in client.calls] == ["program"] * 3 + ["finalization"]
    assert out.episode_trace is not None and not out.episode_trace.label_access


def test_json_quarantined_reference_remains_rejected_without_extra_request(
        source, tmp_path, kernels):
    def reuse(messages):
        rid = kernels[0].tool_results[0].result_id
        assert rid in kernels[0].unresolved_result_ids
        _assert_cleared(kernels[0].snapshot())
        return _json_answer(value=3, basis="mixed", used_result_ids=[rid])

    out, client = _run(
        source, tmp_path, [*_quarantine_prefix(kernels), reuse], rounds=3)
    kernel = kernels[0]
    rid = _assert_natural_quarantine(out, kernel)
    rejected = out.rounds[-1]
    assert rejected["executed"] is False
    assert rejected["error_code"] == "answer_contract_rejected"
    assert rejected["finalization"]["host_generated_program"] is False
    assert json.loads(rejected["response_text"])["used_result_ids"] == [rid]
    assert out.final_state == "run_error" and out.answer is None
    assert out.failure_code == "answer_contract_rejected"
    assert out.terminal_failure["termination_reason"] == "finalization_failed"
    assert len(client.calls) == out.agent_rounds == 3
    assert [phase for phase, _, _ in client.calls] == ["program", "program", "finalization"]
    _assert_cleared(kernel.snapshot())


@pytest.mark.parametrize("input_ids", [1, True, {"r": 1}, None, "r"],
                         ids=["int", "bool", "dict", "none", "string"])
@pytest.mark.parametrize("recovery", ["ordinary", "json"])
def test_malformed_derivation_clears_staging_and_recovers_within_original_budget(
        source, tmp_path, kernels, input_ids, recovery):
    derivation = {"op": "count", "input_result_ids": input_ids, "parameters": {}}
    next_reply = (_json_answer() if recovery == "json"
                  else _ordinary_answer(value=4, basis="visual_estimate"))
    out, client = _run(
        source, tmp_path, [_ordinary_answer(derivation=derivation), next_reply],
        rounds=2 if recovery == "json" else 3)
    kernel = kernels[0]
    rejected = out.rounds[0]
    replay = rejected["geometry_verify"]["derivation_replay"]
    assert rejected["executed"] and rejected["submission_accepted"] is False
    assert rejected["error_code"] == "geometry_verification_rejected"
    assert rejected["execution_diagnostics"]["error_type"] == ""
    assert rejected["execution_diagnostics"]["traceback_text"] == ""
    assert rejected["rejected_submission"]["payload"]["derivation"] == derivation
    assert not replay["passed"] and not replay["performed"]
    assert replay["computed_value"] is None and replay["input_result_ids"] == []
    assert [i["code"] for i in replay["issues"]] == ["payload_contract"]
    assert replay["issues"][0]["reason"] == (
        "derivation.input_result_ids must be a list of nonempty strings")
    assert rejected["geometry_verify"]["check_status"]["answer_contract"] == "fail"
    assert rejected["geometry_verify"]["check_status"]["derivation_replay"] == "not_performed"
    assert rejected["invalidated_result_ids"] == []
    assert rejected["recovery_previous_program_omitted"] == "rejected_submission"
    assert "M11 rejected" in out.rounds[1]["feedback"]
    assert "staged_marker" not in json.dumps(client.calls[1][1])
    for snapshot in kernel.resets:
        _assert_cleared(snapshot)
    _assert_cleared(kernel.cell_starts[1])
    assert out.final_state == "answer" and out.answer == "4" and out.verify.passed
    assert out.used_result_ids == [] and out.program_trace.calls == []
    assert len(client.calls) == out.agent_rounds == 2
    assert [phase for phase, _, _ in client.calls] == [
        "program", "finalization" if recovery == "json" else "program"]


@pytest.mark.parametrize("input_ids", [1, True, {"r": 1}, None, "r"],
                         ids=["int", "bool", "dict", "none", "string"])
def test_malformed_derivation_at_budget_limit_is_audited_and_cleared(
        source, tmp_path, kernels, input_ids):
    derivation = {"op": "count", "input_result_ids": input_ids, "parameters": {}}
    out, client = _run(
        source, tmp_path, [_ordinary_answer(derivation=derivation)],
        rounds=1, finalization_rounds=0, retries=0)
    assert len(client.calls) == out.agent_rounds == 1
    assert out.final_state == "run_error" and out.answer is None
    assert out.failure_code == "geometry_verification_rejected"
    assert out.rounds[0]["rejected_submission"]["payload"]["derivation"] == derivation
    assert out.rounds[0]["submission_accepted"] is False
    _assert_cleared(kernels[0].snapshot())
