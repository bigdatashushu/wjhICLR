"""Fake HTTP tests of actual main solver integration and request admission."""
import base64
from dataclasses import replace
import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from skill3d.small_image.runner import SmallImageRunnerConfig
from skill3d.small_image.cv3d_protocol import FrozenCV3DSkills
from skill3d.small_image.storage import load_prepared
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

from .common import InferenceConfig, digest, read_json
from .data import PublicInput, render_depth
from .governor import SharedSlots
from .paired import schedule
from .program import ProgramClient, PROTOCOL, prepare_request, solve
from .program_paired import source_identity
from .program_worker import ProcessSolver, RunStop
from .test_simplified import fixture


YIELD = """def solve(ctx):
    r = ctx.tools.point_depth(view_index=0, x=5, y=5)
    YieldObservations([r['result_id']])
"""


def settings():
    return SmallImageRunnerConfig(
        prompt_profile="cv3d-compact-program/2", evaluation_role="test_development",
        max_requests=6, max_solve_rounds=5, max_tokens=8192, enable_thinking=True,
        original_image_bytes=True, explicit_final_answer=True, capture_responses=True,
        send_depth_image=True,
        expected_model="qwen3.8-flash")


def frozen(profile="cv3d-compact-program/2"):
    text = "Ground colored boxes and sample visible object surfaces."
    return SimpleNamespace(
        prompt_profile=profile,
        texts={task: text for task in ("Depth", "Distance")},
        identities={task: hashlib.sha256(text.encode()).hexdigest()
                    for task in ("Depth", "Distance")}, manifest_sha256="a" * 64)


def test_legacy_program_identity_uses_original_rgb_without_v2_context(tmp_path):
    from dataclasses import replace

    public, prepared_path, _ = fixture(tmp_path)
    legacy = replace(settings(), prompt_profile="cv3d-compact-program/1", send_depth_image=False)
    _, messages, _, png, identity = prepare_request(
        public, prepared_path, arm="tools", frozen=frozen(legacy.prompt_profile), config=legacy)
    assert identity["prompt_profile"] == legacy.prompt_profile
    assert "context_protocol" not in identity and "feedback_protocol" not in identity
    assert "image_input_protocol" not in identity and png is None
    assert len([part for part in messages[0]["content"] if part["type"] == "image_url"]) == 1
    with pytest.raises(ValueError, match="manifest does not match"):
        prepare_request(public, prepared_path, arm="tools", frozen=frozen(), config=legacy)


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
    prepared = load_prepared(tmp_path / "prepared", problem=public.problem)
    expected_png, expected_display = render_depth(
        prepared.geometries[0]["depth"], prepared.geometries[0]["valid_mask"])
    for ordinal, payload in enumerate(payloads):
        assert payload["max_tokens"] == 8192 and payload["enable_thinking"] is True
        assert payload["temperature"] == 0 and payload["seed"] == 137
        assert "tools" not in payload and "response_format" not in payload
        assert len(payload["messages"]) == 1
        parts = payload["messages"][0]["content"]
        images = [p for p in parts if p["type"] == "image_url"]
        assert len(images) == 2
        assert base64.b64decode(images[0]["image_url"]["url"].split(",")[1]) == (
            __import__("pathlib").Path(public.problem.image_paths[0]).read_bytes())
        depth_png = base64.b64decode(images[1]["image_url"]["url"].split(",")[1])
        assert depth_png == expected_png == (tmp_path / "episode/depth_preview.png").read_bytes()
        assert Image.open(io.BytesIO(depth_png)).size == (16, 12)
        assert "PREDICTED DEPTH MAP" in parts[0]["text"]
        assert "not a second camera view" in parts[0]["text"]
        intent = read_json(tmp_path / f"episode/{ordinal:04d}.intent.json")
        assert len(intent["images"]) == 2
        assert intent["wire_payload_sha256"] == hashlib.sha256(wire[ordinal]).hexdigest()
        assert "PRIVATE_KEY_SENTINEL" not in json.dumps(intent)
    assert row["input_identity"]["depth_display"] == expected_display
    assert row["input_identity"]["images"][1]["sha256"] == hashlib.sha256(expected_png).hexdigest()
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


def test_direct_remains_original_only_and_old_rgb_identity_cannot_be_reused(tmp_path, monkeypatch):
    public, prepared_path, _ = fixture(tmp_path)
    _, direct_messages, direct_tools, png, _ = prepare_request(
        public, prepared_path, arm="direct", frozen=frozen(), config=settings())
    assert png is None and direct_tools == []
    assert len([part for part in direct_messages[0]["content"] if part["type"] == "image_url"]) == 1
    old_identity = prepare_request(
        public, prepared_path, arm="tools", frozen=frozen(),
        config=replace(settings(), send_depth_image=False))[-1]
    new_identity = prepare_request(
        public, prepared_path, arm="tools", frozen=frozen(), config=settings())[-1]
    assert old_identity["initial_messages_sha256"] != new_identity["initial_messages_sha256"]
    monkeypatch.setenv("PROGRAM_TEST_KEY", "OFFLINE_ONLY")

    def forbidden(*args, **kwargs):
        pytest.fail("changed input identity admitted a model request")

    client = ProgramClient(InferenceConfig(), api_key_env="PROGRAM_TEST_KEY", opener=forbidden)
    governor = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(50, 8))
    with pytest.raises(ValueError, match="input changed after preflight"):
        solve(
            public, prepared_path, arm="tools", client=client, governor=governor,
            output_directory=tmp_path / "episode", frozen=frozen(), config=settings(),
            expected_identity=old_identity)
    assert governor.summary()["reserved_requests"] == 0


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


@pytest.fixture
def fake_gateway(tmp_path):
    envelopes = []
    responses = [YIELD] * 5 + ['{"answer":"B"}']
    counts = {}
    mutex = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            envelope = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with mutex:
                operation = envelope["audit_identity"]["operation_id"]
                step = counts.get(operation, 0)
                counts[operation] = step + 1
                envelopes.append(envelope)
            if isinstance(responses[step], Exception):
                self.send_error(503)
                return
            data = json.dumps({
                "model": "qwen3.8-flash", "id": "process-fake",
                "choices": [{"finish_reason": "stop", "message": {
                    "role": "assistant", "content": responses[step]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    (tmp_path / "transport_auth.py").write_text(
        "def load_transport_token():\n    return 'OFFLINE_TEST_TRANSPORT'\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", envelopes, responses
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def process_fixture(tmp_path, gateway):
    skill_path = (Path(__file__).resolve().parents[2] /
                  "skill_library/candidates/cv3d-compact-testdev-2/frozen_skills.json")
    skills = FrozenCV3DSkills(skill_path)
    config = {
        "timeout_s": 10, "gateway": gateway, "auth_module_root": str(tmp_path),
        "skills_manifest": str(skill_path),
    }
    # Host-only label sentinel cannot enter worker transport, budget or process receipts.
    manifest = {"source": source_identity(), "host_scoring_sentinel": "HOST_ONLY_GT"}
    raw = RequestGovernor(tmp_path / "process-budget", RequestGovernorConfig(50, 8))
    stop = RunStop(tmp_path / "stop")
    wrapped = SharedSlots(raw, reserve_direct=1, stop_event=stop, stop_on_request_error=True)
    solver = ProcessSolver(config, manifest, settings(), skills)

    def run(task, ordinal):
        directory = tmp_path / f"case-{ordinal}"
        directory.mkdir()
        public, prepared, _ = fixture(directory, task=task)
        identity = prepare_request(
            public, prepared, arm="tools", frozen=skills, config=settings())[-1]
        return solver(
            public, prepared, arm="tools", client=None, governor=wrapped,
            output_directory=directory / "episode", stop_event=stop, expected_identity=identity)

    return run, raw, stop


def test_concurrent_processes_execute_kernel_and_six_requests(tmp_path, fake_gateway):
    gateway, envelopes, _ = fake_gateway
    run, raw, stop = process_fixture(tmp_path, gateway)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(run, "Depth", 0), pool.submit(run, "Distance", 1)
        rows = [a.result(), b.result()]
    assert all(row["status"] == "answered" and row["answer"] == "B" for row in rows)
    assert all(row["succeeded_tool_calls"] == 5 and row["request_count"] == 6 for row in rows)
    assert raw.summary()["reserved_requests"] == len(envelopes) == 12
    assert not stop.is_set()
    processes = [read_json(tmp_path / f"case-{i}/episode/process.json") for i in range(2)]
    assert len({record["pid"] for record in processes}) == 2
    assert all(record["pid"] != __import__("os").getpid() for record in processes)
    assert "HOST_ONLY_GT" not in json.dumps(envelopes)
    assert all(envelope["payload"]["seed"] == 137 for envelope in envelopes)
    assert all(envelope["payload"]["enable_thinking"] is True for envelope in envelopes)
    depth_images = {}
    for envelope in envelopes:
        operation = envelope["audit_identity"]["operation_id"]
        images = [part for part in envelope["payload"]["messages"][0]["content"]
                  if part["type"] == "image_url"]
        assert len(images) == 2
        image_pair = [part["image_url"]["url"] for part in images]
        assert depth_images.setdefault(operation, image_pair) == image_pair


def test_process_service_failure_stops_other_process_before_http(tmp_path, fake_gateway):
    gateway, envelopes, responses = fake_gateway
    responses[0] = TimeoutError()
    run, raw, stop = process_fixture(tmp_path, gateway)
    first = run("Depth", 0)
    assert first["status"] == "service_error" and stop.path.exists()
    second = run("Distance", 1)
    assert second["status"] == "budget_exhausted" and second["request_count"] == 0
    assert len(envelopes) == raw.summary()["reserved_requests"] == 1
    assert raw.summary()["error_requests"] == 1
