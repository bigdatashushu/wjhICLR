from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest

from skill3d.small_image.datasets import SmallImageProblem
from skill3d.small_image.precompute import SmallImagePrecomputer
from skill3d.small_image.storage import save_prepared
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

from .client import NativeToolClient, ServiceError
from .common import InferenceConfig, read_json
from .data import PublicInput, build_messages, load_inputs, render_depth
from . import evaluate
from .evaluate import _resume_results, summarize
from .inference import parse_final_answer, prepare_request, solve
from .tools import ToolDispatcher, tool_instructions, tool_schemas
from .paired import schedule
from .governor import SharedSlots


def fixture(tmp_path, *, task="Depth"):
    image_path = tmp_path / "original.png"
    yy, xx = np.indices((12, 16))
    rgb = np.stack([xx * 10, yy * 10, xx + yy], axis=-1).astype(np.uint8)
    Image.fromarray(rgb).save(image_path)
    p = SmallImageProblem(
        "cv3d", "CVBench_3D_fixture", "train", "Which object is closer?",
        ("table", "bookcase"), (str(image_path),), task)
    z = 1 + xx / 10
    computer = SmallImagePrecomputer(
        depth_estimator=lambda _: {
            "depth": z, "valid_mask": np.ones_like(z, dtype=bool),
            "points": np.stack([xx, yy, z], axis=-1),
        }, model_identities={"depth_estimator": {"model": "fixture"}})
    prepared = computer.process(p, categories=())
    directory = tmp_path / "prepared"
    save_prepared(prepared, directory)
    return PublicInput(p, p.question + "\n(A) table\n(B) bookcase"), directory, prepared


def native_call(call_id, name="point_depth", **args):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}


def reply(text=None, calls=()):
    return {
        "message": {"role": "assistant", "content": text,
                    "reasoning_content": "fixture reasoning",
                    **({"tool_calls": list(calls)} if calls else {})},
        "finish_reason": "tool_calls" if calls else "stop",
        "model": "qwen3.8-flash",
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


class FakeClient:
    config = InferenceConfig()

    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def chat_tools(self, messages, tools):
        self.messages.append(copy.deepcopy(messages))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def run_fixture(tmp_path, responses, *, budget=100, task="Depth"):
    public, prepared_path, prepared = fixture(tmp_path, task=task)
    governor = RequestGovernor(
        tmp_path / "budget", RequestGovernorConfig(budget, 1))
    client = FakeClient(responses)
    result = solve(
        public, prepared_path, client=client, governor=governor,
        output_directory=tmp_path / "episode", stop_event=threading.Event())
    return result, client, governor, public, prepared


def test_exact_prompt_image_order_and_original_bytes(tmp_path):
    public, directory, _ = fixture(tmp_path)
    _, messages, tools, png, identity = prepare_request(public, directory)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    parts = messages[0]["content"]
    assert parts[0]["text"] == public.official_prompt + "\n\n" + tool_instructions("Depth")
    assert "ORIGINAL RGB IMAGE" in parts[1]["text"]
    assert "PREDICTED DEPTH MAP" in parts[3]["text"]
    assert base64.b64decode(parts[2]["image_url"]["url"].split(",", 1)[1]) == (
        Path(public.problem.image_paths[0]).read_bytes())
    assert base64.b64decode(parts[4]["image_url"]["url"].split(",", 1)[1]) == png
    assert Image.open(io.BytesIO(png)).size == (16, 12)
    assert len(tools) == 3 and identity["artifact_sha256"]
    assert "Skill" not in parts[0]["text"] and "def solve" not in parts[0]["text"]


def test_depth_display_invalid_pixels_and_raw_unchanged():
    depth = np.arange(20, dtype=float).reshape(4, 5) + 1
    depth[0, 0] = np.nan
    original = depth.copy()
    png, _ = render_depth(depth, np.ones_like(depth, dtype=bool))
    rgb = np.asarray(Image.open(io.BytesIO(png)))
    assert rgb[0, 0].tolist() == [255, 0, 255]
    assert rgb[0, 1, 0] < rgb[-1, -1, 0]
    np.testing.assert_equal(depth, original)
    # A constant map remains usable, without division by zero.
    constant, _ = render_depth(np.ones((4, 5)), np.ones((4, 5), dtype=bool))
    assert np.asarray(Image.open(io.BytesIO(constant)))[0, 0].tolist() == [128] * 3


def test_more_than_two_tool_turns_and_large_batch_are_allowed(tmp_path):
    calls = [native_call(f"batch-{i}", view_index=0, x=5, y=5) for i in range(20)]
    responses = [
        reply(calls=calls),
        reply(calls=[native_call("next-1", view_index=0, x=6, y=5)]),
        reply(calls=[native_call("next-2", view_index=0, x=7, y=5)]),
        reply(calls=[native_call("next-3", view_index=0, x=8, y=5)]),
        reply("B"),
    ]
    result, client, governor, _, _ = run_fixture(tmp_path, responses)
    assert result["status"] == "answered" and result["answer"] == "B"
    assert result["attempted_tool_calls"] == result["succeeded_tool_calls"] == 23
    assert result["request_count"] == governor.summary()["reserved_requests"] == 5
    second = client.messages[1]
    assert second[1]["role"] == "assistant" and len(second[1]["tool_calls"]) == 20
    assert second[1]["reasoning_content"] == "fixture reasoning"
    observation = json.loads(second[2]["content"])
    assert observation["status"] == "ok" and observation["data"]["depth"] == 1.5
    assert second[2]["tool_call_id"] == "batch-0"


def test_invalid_answer_ends_session_without_second_question(tmp_path):
    result, client, governor, _, _ = run_fixture(
        tmp_path, [reply("Maybe either choice."), reply("A")])
    assert result["status"] == "invalid_answer" and result["answer"] is None
    assert len(client.messages) == governor.summary()["reserved_requests"] == 1


def test_service_failure_is_recorded_once_and_not_retried(tmp_path):
    result, client, governor, _, _ = run_fixture(
        tmp_path, [ServiceError("http_error", http_status=429), reply("A")])
    assert result["status"] == "service_error" and result["http_status"] == 429
    assert len(client.messages) == 1
    assert governor.summary()["error_requests"] == 1


def test_global_budget_preserved_without_per_question_tool_cap(tmp_path):
    result, client, governor, _, _ = run_fixture(
        tmp_path, [
            reply(calls=[native_call("one", view_index=0, x=5, y=5)]), reply("A")
        ], budget=1)
    assert result["status"] == "budget_exhausted"
    assert len(client.messages) == governor.summary()["reserved_requests"] == 1
    assert result["attempted_tool_calls"] == 1


def test_tool_scope_and_invalid_arguments_are_returned_to_model(tmp_path):
    public, _, prepared = fixture(tmp_path)
    dispatcher = ToolDispatcher(prepared, public.problem)
    observation, _ = dispatcher.call(native_call(
        "out", view_index=0, x=999, y=5))
    assert observation["status"] == "error" and observation["data"] is None
    observation, _ = dispatcher.call(native_call(
        "boolean", view_index=True, x=5, y=5))
    assert observation["status"] == "error"
    observation, _ = dispatcher.call(native_call(
        "not-depth-tool", "pair_point_distance", view_index=0, a_xy=[5, 5], b_xy=[8, 5]))
    assert observation["status"] == "error"
    assert len(tool_schemas("Distance")) == 4
    oversized_number = native_call("nonfinite", view_index=0, x=5, y=5)
    oversized_number["function"]["arguments"] = '{"view_index":0,"x":1e999,"y":5}'
    observation, trace = dispatcher.call(oversized_number)
    assert observation["status"] == "error" and trace["arguments"] is None


def test_distance_reuses_local_median_points(tmp_path):
    public, _, prepared = fixture(tmp_path, task="Distance")
    observation, trace = ToolDispatcher(prepared, public.problem).call(native_call(
        "distance", "pair_point_distance", view_index=0, a_xy=[3, 4], b_xy=[8, 4]))
    assert observation["status"] == "ok"
    assert observation["data"]["distance"] == pytest.approx(np.sqrt(25.25))
    assert trace["authorization"]["allowed"]


def test_preflight_rejects_changed_original_bytes_without_request(tmp_path):
    public, directory, _ = fixture(tmp_path)
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(public.problem.image_paths[0])
    with pytest.raises(ValueError, match="bytes changed"):
        prepare_request(public, directory)


def test_denominator_includes_illegal_error_and_unrun_questions(tmp_path):
    public, _, _ = fixture(tmp_path)
    from dataclasses import replace

    inputs = [PublicInput(replace(public.problem, sample_id=f"case-{i}"),
                          public.official_prompt) for i in range(4)]
    answers = {f"case-{i}": "A" for i in range(4)}
    results = {
        "case-0": {"sample_id": "case-0", "status": "answered", "answer": "A"},
        "case-1": {"sample_id": "case-1", "status": "invalid_answer", "answer": None},
        "case-2": {"sample_id": "case-2", "status": "host_error", "answer": None},
    }
    score = summarize(inputs, answers, results)
    assert score["overall"]["accuracy_full_denominator"] == 0.25
    assert score["overall"]["total_questions"] == 4
    assert score["overall"]["statuses"]["not_run"] == 1
    assert not score["complete"]


@pytest.mark.parametrize(("text", "expected"), [
    ("A", "A"), ("(B)", "B"), ("bookcase", "B"),
    ('{"answer":"A"}', "A"), ("Final answer: B", "B"),
    ("Explanation.\n(B) bookcase", "B"),
    ("A\nB", None), ("A or B", None), ("Maybe the table?", None),
])
def test_answer_parser(text, expected):
    assert parse_final_answer(text, ("table", "bookcase")) == expected


def test_official_prompt_lookup_keeps_labels_out(tmp_path):
    root = tmp_path / "dataset"
    (root / "CVBench/3D").mkdir(parents=True)
    (root / "skill3d_splits").mkdir()
    image = root / "CVBench/3D/000001.png"
    Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(image)
    prompt = "Which object is closer?\n(A) table\n(B) bookcase"
    source = {
        "id": "CVBench_3D_000001", "task": "Depth", "image": ["CVBench/3D/000001.png"],
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "PRIVATE_LABEL_SENTINEL"},
        ],
    }
    (root / "skill3d_splits/cv3d_train.jsonl").write_text(json.dumps(source) + "\n")
    official = tmp_path / "test_3d.jsonl"
    official.write_text(json.dumps({
        "type": "3D", "task": "Depth", "question": "Which object is closer?",
        "choices": ["table", "bookcase"], "prompt": prompt,
        "answer": "PRIVATE_LABEL_SENTINEL",
    }) + "\n")
    public = load_inputs(root, official, split="train")[0]
    assert public.official_prompt == prompt
    assert "PRIVATE_LABEL_SENTINEL" not in json.dumps(public.identity())


def test_native_http_payload_and_no_credentials_in_receipts(tmp_path, monkeypatch):
    captured, wire_bytes = [], []

    class HTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({
                "model": "qwen3.8-flash",
                "choices": [{"finish_reason": "stop", "message": {
                    "role": "assistant", "content": "A"}}],
                "usage": {"total_tokens": 7},
            }).encode()

    def opener(request, **kwargs):
        captured.append(json.loads(request.data))
        wire_bytes.append(request.data)
        return HTTPResponse()

    monkeypatch.setenv("CV3D_FIXTURE_KEY", "PRIVATE_CREDENTIAL_SENTINEL")
    client = NativeToolClient(InferenceConfig(), api_key_env="CV3D_FIXTURE_KEY", opener=opener)
    public, directory, _ = fixture(tmp_path)
    _, messages, tools, _, _ = prepare_request(public, directory)
    response = client.chat_tools(messages, tools)
    assert response["message"]["content"] == "A"
    payload = captured[0]
    assert payload["enable_thinking"] is True
    assert payload["temperature"] == 0 and payload["seed"] == 137
    assert payload["max_tokens"] == 8192 and payload["tool_choice"] == "auto"
    assert payload["tools"] == tools and "response_format" not in payload
    assert "PRIVATE_CREDENTIAL_SENTINEL" not in json.dumps(client.last_request_identity)
    assert client.last_request_identity["wire_payload_sha256"] == hashlib.sha256(
        wire_bytes[0]).hexdigest()
    governor = RequestGovernor(
        tmp_path / "wire-budget", RequestGovernorConfig(10, 1))
    result = solve(
        public, directory, client=client, governor=governor,
        output_directory=tmp_path / "wire-episode")
    assert result["answer"] == "A"
    intent = read_json(tmp_path / "wire-episode/0000.intent.json")
    assert intent["wire_payload_sha256"] == hashlib.sha256(wire_bytes[1]).hexdigest()


def evaluation_fixture(tmp_path):
    """Create one label-bearing host source and one completely separate public input."""
    root = tmp_path / "dataset"
    (root / "CVBench/3D").mkdir(parents=True)
    (root / "skill3d_splits").mkdir()
    image = root / "CVBench/3D/000001.png"
    Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)).save(image)
    prompt = "Which object is closer?\n(A) table\n(B) bookcase"
    source = {
        "id": "CVBench_3D_000001", "task": "Depth", "image": ["CVBench/3D/000001.png"],
        "conversations": [
            {"from": "human", "value": prompt}, {"from": "gpt", "value": "A"},
        ],
    }
    (root / "skill3d_splits/cv3d_train.jsonl").write_text(json.dumps(source) + "\n")
    official = tmp_path / "test_3d.jsonl"
    official.write_text(json.dumps({
        "type": "3D", "task": "Depth", "question": "Which object is closer?",
        "choices": ["table", "bookcase"], "prompt": prompt, "answer": "(A)",
    }) + "\n")
    public = load_inputs(root, official, split="train")[0]
    prepared = SmallImagePrecomputer(
        depth_estimator=lambda rgb: {
            "depth": np.ones(rgb.shape[:2]), "valid_mask": np.ones(rgb.shape[:2], bool),
        }, model_identities={"depth_estimator": {"model": "fixture"}}).process(
            public.problem, categories=())
    save_prepared(prepared, tmp_path / "cache/one")
    return public, [
        "--dataset-root", str(root), "--official-annotations", str(official),
        "--prepared-root", str(tmp_path / "cache"),
        "--output-dir", str(tmp_path / "run"), "--split", "train",
        "--max-requests", "20", "--concurrency", "1",
    ]


def test_full_preflight_execute_resume_and_score_cli(tmp_path, monkeypatch):
    public, args = evaluation_fixture(tmp_path)
    output = tmp_path / "run"
    assert evaluate.main(args) == 0
    assert read_json(output / "preflight.json")["model_requests"] == 0
    assert not (output / "budget").exists()
    clients = []

    class EvaluationClient(FakeClient):
        endpoint_hash = NativeToolClient(InferenceConfig()).endpoint_hash
        configured = True

        def __init__(self, config, **_):
            super().__init__([
                reply(calls=[native_call("one", view_index=0, x=5, y=5)]), reply("A")
            ])
            self.config = config
            clients.append(self)

    monkeypatch.setattr(evaluate, "NativeToolClient", EvaluationClient)
    assert evaluate.main(args + ["--resume", "--execute"]) == 0
    scores = read_json(output / "scores.json")
    assert scores["complete"] and scores["overall"]["accuracy_full_denominator"] == 1
    assert scores["budget"]["reserved_requests"] == 2
    assert sum(len(c.messages) for c in clients) == 2
    preview = read_json(output / "request_preview.json")
    assert preview["messages"][0]["content"][0]["text"].startswith(public.official_prompt)
    assert evaluate.main(args + ["--resume", "--execute"]) == 0
    assert sum(len(c.messages) for c in clients) == 2  # Existing result is never repeated.
    assert evaluate.main(args + ["--score-only"]) == 0
    assert read_json(output / "scores.json")["budget"]["reserved_requests"] == 2


def test_failed_results_can_be_scored_but_not_automatically_resumed(tmp_path):
    result, _, _, public, _ = run_fixture(
        tmp_path, [ServiceError("http_error", http_status=500)])
    directory = tmp_path / "episodes" / public.problem.sample_id
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(json.dumps(result))
    identities = {public.problem.sample_id: result["input_identity"]}
    with pytest.raises(ValueError, match="diagnosis"):
        _resume_results([public], identities, tmp_path)
    rows = _resume_results([public], identities, tmp_path, for_resume=False)
    assert rows[public.problem.sample_id]["status"] == "service_error"
    scores = summarize([public], {public.problem.sample_id: "A"}, rows)
    assert scores["overall"]["total_questions"] == 1
    assert scores["overall"]["accuracy_full_denominator"] == 0
    assert not scores["complete"]


def test_direct_uses_exact_official_prompt_one_image_and_no_tool_parameters(tmp_path):
    public, directory, _ = fixture(tmp_path)
    prepared, messages, tools, png, _ = prepare_request(public, arm="direct")
    assert prepared is None and png is None and tools == []
    assert len(messages) == 1
    assert messages[0]["content"][0]["text"] == public.official_prompt
    assert len(messages[0]["content"]) == 2
    client = NativeToolClient(InferenceConfig())
    payload = client.payload(messages, [])
    assert "tools" not in payload and "tool_choice" not in payload
    fake = FakeClient([reply("A"), reply("B")])
    governor = RequestGovernor(tmp_path / "direct-budget", RequestGovernorConfig(10, 1))
    result = solve(
        public, None, arm="direct", client=fake, governor=governor,
        output_directory=tmp_path / "direct")
    assert result["answer"] == "A" and result["request_count"] == 1
    assert not (tmp_path / "direct/depth.png").exists()


def test_paired_scheduler_shares_budget_and_preserves_same_question(tmp_path):
    public, directory, _ = fixture(tmp_path)
    identities = {
        arm: {public.problem.sample_id: prepare_request(
            public, directory, arm=arm)[-1]} for arm in ("tools", "direct")}
    raw = RequestGovernor(tmp_path / "paired-budget", RequestGovernorConfig(10, 4))
    governor = SharedSlots(raw, reserve_direct=1)
    clients = {}
    def factory(arm, _):
        clients[arm] = FakeClient(
            [reply(calls=[native_call("one", view_index=0, x=5, y=5)]), reply("B")]
            if arm == "tools" else [reply("A")])
        return clients[arm]
    score, rows = schedule(
        [public], identities, {public.problem.input_sha256: directory},
        output=tmp_path / "paired", governor=governor, client_factory=factory,
        answers={public.problem.sample_id: "A"})
    assert score["complete"]
    assert score["paired"]["direct_only"] == 1
    assert raw.summary()["reserved_requests"] == 3
    assert rows["tools"][public.problem.sample_id]["request_count"] == 2
    assert rows["direct"][public.problem.sample_id]["request_count"] == 1


def test_tool_budget_cannot_consume_reserved_direct_request(tmp_path):
    from skill3d.synthesis.request_governor import RequestBudgetExhausted
    raw = RequestGovernor(tmp_path / "reserve-budget", RequestGovernorConfig(2, 4))
    wrapped = SharedSlots(raw, reserve_direct=1)
    with wrapped.request("solver", {"variant_id": "tools"}):
        pass
    with pytest.raises(RequestBudgetExhausted):
        with wrapped.request("solver", {"variant_id": "tools"}):
            pytest.fail("reserved direct budget was consumed")
    with wrapped.request("solver", {"variant_id": "direct"}):
        pass
    assert raw.summary()["reserved_requests"] == 2


def test_stopped_run_does_not_admit_a_waiting_request(tmp_path):
    from skill3d.synthesis.request_governor import RequestBudgetExhausted
    raw = RequestGovernor(tmp_path / "stopped-budget", RequestGovernorConfig(10, 4))
    stop = threading.Event()
    stop.set()
    wrapped = SharedSlots(raw, stop_event=stop)
    with pytest.raises(RequestBudgetExhausted):
        with wrapped.request("solver", {"variant_id": "tools"}):
            pytest.fail("request admitted after stop")
    assert raw.summary()["reserved_requests"] == 0
