import copy
import hashlib
import threading
from collections import Counter
from dataclasses import replace

import pytest

from skill3d.synthesis.request_audit import payload_bytes
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig
from .client import NativeToolClient
from .common import read_json, write_json
from .inference import prepare_request, solve
from .data import PublicInput
from .paired import schedule
from .concurrency import revise_policy
from .replay import replay_session
from .test_simplified import FakeClient, fixture, native_call, reply


class PayloadFakeClient(FakeClient):
    def payload(self, messages, tools):
        return NativeToolClient(self.config).payload(messages, tools)


def paused_session(tmp_path):
    public, prepared_path, _ = fixture(tmp_path)
    stop = threading.Event()

    class PausingClient(PayloadFakeClient):
        def chat_tools(self, messages, tools):
            response = super().chat_tools(messages, tools)
            stop.set()
            return response

    client = PausingClient([reply(calls=[
        native_call("sample", view_index=0, x=5, y=5),
        native_call("bad", view_index=0, x=999, y=5),
    ])])
    governor = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(30, 8))
    _, _, _, _, identity = prepare_request(public, prepared_path)
    output = tmp_path / "episode"
    result = solve(
        public, prepared_path, client=client, governor=governor,
        output_directory=output, stop_event=stop, expected_identity=identity)
    assert result["status"] == "cancelled" and result["request_count"] == 1
    (output / "result.json").unlink()
    return public, prepared_path, output, identity, governor


def test_continuation_preserves_tool_results_and_does_not_repeat_request(tmp_path):
    public, prepared, output, identity, governor = paused_session(tmp_path)
    prior = (output / "0000.intent.json").read_bytes()
    client = PayloadFakeClient([reply("B")])
    result = solve(
        public, prepared, client=client, governor=governor, output_directory=output,
        expected_identity=identity, resume=True)
    assert result["answer"] == "B"
    assert result["request_count"] == governor.summary()["reserved_requests"] == 2
    assert result["attempted_tool_calls"] == 2 and result["succeeded_tool_calls"] == 1
    assert result["continuation_verified_requests"] == 1
    assert (output / "0000.intent.json").read_bytes() == prior
    assert len(client.messages) == 1 and len(client.messages[0]) == 4
    digest = hashlib.sha256(payload_bytes(client.payload(
        client.messages[0], prepare_request(public, prepared)[2]))).hexdigest()
    assert digest == read_json(output / "0001.intent.json")["wire_payload_sha256"]


def test_continuation_rejects_tampered_measurement_before_model_request(tmp_path):
    public, prepared, output, identity, governor = paused_session(tmp_path)
    traces = read_json(output / "0000.tools.json")
    traces[0]["response"]["data"]["depth"] = 999
    write_json(output / "0000.tools.json", traces)
    client = PayloadFakeClient([reply("A")])
    with pytest.raises(AssertionError):
        solve(public, prepared, client=client, governor=governor,
              output_directory=output, expected_identity=identity, resume=True)
    assert not client.messages and governor.summary()["reserved_requests"] == 1


def test_continuation_rejects_unresolved_http_attempt(tmp_path):
    public, prepared, output, identity, governor = paused_session(tmp_path)
    (output / "0000.response.json").unlink()
    with pytest.raises(FileNotFoundError):
        replay_session(
            public, output, prepared, arm="tools",
            client=PayloadFakeClient([]), expected_identity=identity)
    assert governor.summary()["reserved_requests"] == 1


def test_eight_workers_admit_four_requests_per_arm(tmp_path, monkeypatch):
    from . import paired
    public, _, _ = fixture(tmp_path)
    inputs = [PublicInput(replace(public.problem, sample_id=f"case-{i}"),
                          public.official_prompt) for i in range(4)]
    identities = {
        arm: {p.problem.sample_id: {"fixture": arm} for p in inputs}
        for arm in ("tools", "direct")
    }
    raw = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(30, 8))
    barrier = threading.Barrier(8)
    active = Counter()
    observed = []
    lock = threading.Lock()

    def worker(public_input, _, *, arm, governor, **kwargs):
        with governor.request("solver", {"variant_id": arm}):
            with lock:
                active[arm] += 1
                observed.append(dict(active))
            barrier.wait(timeout=5)
            with lock:
                active[arm] -= 1
        return {
            "sample_id": public_input.problem.sample_id, "answer": "A",
            "status": "answered", "request_count": 1,
        }

    monkeypatch.setattr(paired, "solve", worker)
    scores, _ = schedule(
        inputs, identities, {}, output=tmp_path / "paired", governor=raw,
        client_factory=lambda *_: None, answers={p.problem.sample_id: "A" for p in inputs},
        concurrency=8)
    assert scores["complete"]
    assert {"tools": 4, "direct": 4} in observed
    assert raw.summary()["peak_inflight"] == 8


def test_policy_revision_keeps_request_history_and_total_budget(tmp_path):
    ops = tmp_path / "ops"
    ops.mkdir()
    raw = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(6000, 4))
    with raw.request("solver", {"variant_id": "direct"}) as receipt:
        receipt["usage"] = {"total_tokens": 123}
    before = read_json(tmp_path / "budget/state.json")
    revise_policy(tmp_path, ops)
    revised = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(6000, 8))
    after = read_json(tmp_path / "budget/state.json")
    assert after["requests"] == before["requests"]
    assert revised.summary()["reserved_requests"] == 1
    assert revised.summary()["usage"]["total_tokens"] == 123
    assert after["policy"] == {**before["policy"], "max_inflight": 8}
    assert read_json(ops / "budget-before.json") == before
