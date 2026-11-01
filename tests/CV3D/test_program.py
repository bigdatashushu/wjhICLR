"""Fake HTTP tests of actual main solver integration and request admission."""
import base64
from dataclasses import replace
import hashlib
import io
import json
import threading
from types import SimpleNamespace

import pytest

from skill3d.small_image.runner import SmallImageRunnerConfig
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

from .common import InferenceConfig, digest, read_json
from .data import PublicInput
from .governor import SharedSlots
from .paired import schedule
from .program import ProgramClient, PROTOCOL, prepare_request, solve
from .test_simplified import fixture


YIELD = """def solve(ctx):
    r = ctx.tools.point_depth(view_index=0, x=5, y=5)
    YieldObservations([r['result_id']])
"""


def settings():
    return SmallImageRunnerConfig(
        prompt_profile="cv3d-compact-program/1", evaluation_role="test_development",
        max_requests=6, max_solve_rounds=5, max_tokens=8192, enable_thinking=True,
        original_image_bytes=True, explicit_final_answer=True, capture_responses=True,
        expected_model="qwen3.8-flash")


def frozen():
    text = "Ground colored boxes and sample visible object surfaces."
    return SimpleNamespace(
        texts={task: text for task in ("Depth", "Distance")},
        identities={task: hashlib.sha256(text.encode()).hexdigest()
                    for task in ("Depth", "Distance")}, manifest_sha256="a" * 64)


def run_program(tmp_path, monkeypatch, responses):
    public, prepared_path, prepared = fixture(tmp_path)
    payloads, wire = [], []
    iterator = iter(responses)
    monkeypatch.setenv("PROGRAM_TEST_KEY", "PRIVATE_KEY_SENTINEL")

    def opener(request, **kwargs):
        payloads.append(json.loads(request.data))
        wire.append(request.data)
        content = next(iterator)
        if isinstance(content, Exception):
            raise content
        return io.BytesIO(json.dumps({
            "model": "qwen3.8-flash", "id": "fake-1", "created": 1,
            "choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": content, "reasoning_content": "private thinking"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }).encode())

    client = ProgramClient(InferenceConfig(), api_key_env="PROGRAM_TEST_KEY", opener=opener)
    raw = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(50, 8))
    wrapped = SharedSlots(raw, reserve_direct=1)
    identity = prepare_request(public, prepared_path, arm="tools",
                               frozen=frozen(), config=settings())[-1]
    stop = threading.Event()
    row = solve(
        public, prepared_path, arm="tools", client=client, governor=wrapped,
        output_directory=tmp_path / "episode", frozen=frozen(), config=settings(),
        expected_identity=identity, stop_event=stop)
    return row, payloads, raw, public, wire, stop


def test_six_requests_passive_trace_and_original_bytes(tmp_path, monkeypatch):
    row, payloads, raw, public, wire, stop = run_program(
        tmp_path, monkeypatch, [YIELD] * 5 + ['{"answer":"B","reasoning":"compare"}'])
    assert row["status"] == "answered" and row["answer"] == "B"
    assert row["request_count"] == len(payloads) == 6
    assert row["succeeded_tool_calls"] == row["attempted_tool_calls"] == 5
    assert row["usage"]["total_tokens"] == 84
    assert row["program_result"]["finalization_requests"] == 1
    assert raw.summary()["reserved_requests"] == 6
    assert not stop.is_set()
    for ordinal, payload in enumerate(payloads):
        assert payload["max_tokens"] == 8192 and payload["enable_thinking"] is True
        assert payload["temperature"] == 0 and payload["seed"] == 137
        assert "tools" not in payload and "response_format" not in payload
        assert len(payload["messages"]) == 1
        parts = payload["messages"][0]["content"]
        images = [p for p in parts if p["type"] == "image_url"]
        assert len(images) == 1
        assert base64.b64decode(images[0]["image_url"]["url"].split(",")[1]) == (
            __import__("pathlib").Path(public.problem.image_paths[0]).read_bytes())
        intent = read_json(tmp_path / f"episode/{ordinal:04d}.intent.json")
        assert intent["wire_payload_sha256"] == hashlib.sha256(wire[ordinal]).hexdigest()
        assert "PRIVATE_KEY_SENTINEL" not in json.dumps(intent)
    trace = read_json(tmp_path / "episode/0000.tools.json")[0]
    assert trace["response"]["data"]["depth"] == pytest.approx(1.5)
    assert trace["arguments"] == {"view_index": 0, "x": 5, "y": 5}
    assert trace["authorization"]["allowed"]
    continuation = payloads[1]["messages"][0]["content"][0]["text"]
    assert '"depth": 1.5' in continuation and "private thinking" not in continuation
    assert row["receipt_sha256"] == digest({k: v for k, v in row.items()
                                          if k != "receipt_sha256"})
    state = read_json(tmp_path / "budget/state.json")
    assert all(r["context"]["variant_id"] == "tools" for r in state["requests"])
    assert "private thinking" not in json.dumps(state)


def test_program_service_error_stops_without_retry(tmp_path, monkeypatch):
    row, payloads, raw, _, _, stop = run_program(
        tmp_path, monkeypatch, [TimeoutError(), '{"answer":"A"}'])
    assert row["status"] == "service_error" and stop.is_set()
    assert len(payloads) == raw.summary()["error_requests"] == 1


def test_program_call_failure_keeps_trace_and_does_not_stop_campaign(tmp_path, monkeypatch):
    row, payloads, _, _, _, stop = run_program(
        tmp_path, monkeypatch, [YIELD.replace("x=5", "x=9999")])
    assert row["status"] == "run_error" and len(payloads) == 1 and not stop.is_set()
    trace = read_json(tmp_path / "episode/0000.tools.json")[0]
    assert trace["status"] in {"error", "failed"} and trace["arguments"]["x"] == 9999


def test_eight_workers_remain_four_per_arm(tmp_path):
    public, _, _ = fixture(tmp_path)
    inputs = [PublicInput(replace(public.problem, sample_id=f"case-{i}"), public.official_prompt)
              for i in range(10)]
    identities = {arm: {p.problem.sample_id: {} for p in inputs}
                  for arm in ("tools", "direct")}
    barrier = threading.Barrier(8)
    mutex = threading.Lock()
    running, peaks, starts = {"tools": 0, "direct": 0}, {"tools": 0, "direct": 0}, []

    def worker(public, _, *, arm, **kwargs):
        with mutex:
            running[arm] += 1
            peaks[arm] = max(peaks[arm], running[arm])
            starts.append(arm)
            initial = len(starts) <= 8
        if initial:
            barrier.wait(timeout=10)
        with mutex:
            running[arm] -= 1
        return {"sample_id": public.problem.sample_id, "status": "answered", "answer": "A"}

    governor = RequestGovernor(tmp_path / "schedule-budget", RequestGovernorConfig(100, 8))
    result, _ = schedule(
        inputs, identities, {}, output=tmp_path / "paired", governor=governor,
        client_factory=lambda *_: None, answers={p.problem.sample_id: "A" for p in inputs},
        concurrency=8, lend_idle=False, solver=worker, protocol=PROTOCOL)
    assert peaks == {"tools": 4, "direct": 4}
    assert starts[:8].count("tools") == starts[:8].count("direct") == 4
    assert result["complete"] and result["protocol"] == PROTOCOL
