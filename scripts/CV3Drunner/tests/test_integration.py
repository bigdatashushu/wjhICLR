from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest

from skill3d.small_image.datasets import freeze_manifest
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig
from CV3D.common import InferenceConfig, digest, read_json
from CV3D.governor import SharedSlots
from CV3D.inference import prepare_request as old_direct_prepare
from CV3D.program_worker import RunStop
from CV3Drunner.config import RunnerConfig
from CV3Drunner.program import ProgramClient, prepare_request, solve
from CV3Drunner.run import preflight, source_identity, select_inputs
from CV3Drunner.worker import ProcessSolver, validate_source
from fixtures import fixture, frozen
from test_runner import MEASURE_ANSWER, ANSWER


def test_preflight_pins_every_input_and_is_zero_request_on_repeat(tmp_path, monkeypatch):
    selected = [fixture(tmp_path / "CVBench/3D" / f"case-{i}", task=task, split=split, sample_id=f"case-{i}")
                for i, (task, split) in enumerate(
                    [("Depth", "train"), ("Distance", "train"), ("Depth", "test")])]
    problems = [public.problem for public, _, _ in selected]
    freeze_manifest(problems, tmp_path / "public.json")
    official = tmp_path / "official.jsonl"
    official.write_text("\n".join(json.dumps({
        "type": "3D", "task": public.problem.task_type, "question": public.problem.question,
        "choices": list(public.problem.options), "prompt": public.official_prompt})
        for public, _, _ in selected))
    answers = tmp_path / "answers.json"
    # Only file hashing is needed; its contents must not be parsed in preflight.
    answers.write_text("HOST_ONLY_UNREADABLE_LABEL_SENTINEL")
    config = {
        "output": str(tmp_path / "output"), "concurrency": 8, "max_requests": 21,
        "timeout_s": 10, "public_manifest": str(tmp_path / "public.json"),
        "dataset_root": str(tmp_path), "official_annotations": str(official),
        "prepared_root": str(tmp_path), "answers_path": str(answers),
        "selection": {"train_ids": ["case-0", "case-1"], "test_ids": ["case-2"]},
    }

    def forbidden(*args, **kwargs):
        pytest.fail("preflight attempted a model or GPU request")

    monkeypatch.setattr(ProgramClient, "chat", forbidden)
    first = preflight(config)
    receipt = (tmp_path / "output/preflight.json").read_bytes()
    second = preflight(config)
    assert first[4] == second[4]
    assert (tmp_path / "output/preflight.json").read_bytes() == receipt
    assert read_json(tmp_path / "output/preflight.json")["model_requests"] == 0
    assert not (tmp_path / "output/budget").exists()
    manifest = first[4]
    assert "scripts/CV3Drunner/worker.py" in manifest["source"]
    assert "server.md" not in manifest["source"]
    assert "HOST_ONLY_UNREADABLE_LABEL_SENTINEL" not in json.dumps(manifest)
    assert manifest["request_identities"]["tools"]["case-0"]["annotation_catalog_sha256"]
    changed = {**config, "max_requests": 28}
    with pytest.raises(ValueError, match="manifest changed"):
        preflight(changed)


def test_prepare_identity_and_direct_baseline_match_existing_exactly(tmp_path):
    public, directory, _ = fixture(tmp_path)
    new = prepare_request(public, directory, arm="direct", config=RunnerConfig(), frozen=frozen())
    old = old_direct_prepare(public, arm="direct")
    assert new == old
    one = prepare_request(public, directory, arm="tools", config=RunnerConfig(), frozen=frozen())[-1]
    two = prepare_request(public, directory, arm="tools",
                          config=replace(RunnerConfig(), send_depth_image=False), frozen=frozen())[-1]
    assert one["initial_messages_sha256"] != two["initial_messages_sha256"]
    assert one["prompt_profile"] == "cv3d-box-review-program/3"


def test_changed_identity_cannot_admit_model_request(tmp_path, monkeypatch):
    public, directory, _ = fixture(tmp_path)
    monkeypatch.setenv("OFFLINE_PROGRAM_KEY", "OFFLINE_ONLY")

    def forbidden(*args, **kwargs):
        pytest.fail("changed identity admitted a model request")

    client = ProgramClient(InferenceConfig(), api_key_env="OFFLINE_PROGRAM_KEY", opener=forbidden)
    governor = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(20, 1))
    identity = prepare_request(
        public, directory, arm="tools", config=RunnerConfig(), frozen=frozen())[-1]
    with pytest.raises(ValueError, match="changed after preflight"):
        solve(public, directory, arm="tools", client=client, governor=governor,
              output_directory=tmp_path / "episode", frozen=frozen(), config=RunnerConfig(),
              expected_identity={**identity, "annotation_catalog_sha256": "0" * 64})
    assert governor.summary()["reserved_requests"] == 0


def test_trace_retains_kernel_result_identity_when_program_times_out(tmp_path, monkeypatch):
    public, directory, _ = fixture(tmp_path)
    monkeypatch.setenv("OFFLINE_PROGRAM_KEY", "OFFLINE_ONLY")
    config = replace(RunnerConfig(), cell_timeout_s=.02)

    def opener(request, **kwargs):
        source = MEASURE_ANSWER.replace(
            "    ReturnAnswer", "    while True:\n        pass\n    ReturnAnswer")
        return __import__("io").BytesIO(json.dumps({
            "model": "qwen3.8-flash",
            "choices": [{"finish_reason": "stop", "message": {"content": source}}],
        }).encode())

    client = ProgramClient(InferenceConfig(), api_key_env="OFFLINE_PROGRAM_KEY", opener=opener)
    governor = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(20, 1))
    identity = prepare_request(public, directory, arm="tools", config=config, frozen=frozen())[-1]
    stop = threading.Event()
    row = solve(public, directory, arm="tools", client=client, governor=governor,
                output_directory=tmp_path / "episode", frozen=frozen(), config=config,
                expected_identity=identity, stop_event=stop)
    assert row["status"] == "run_error" and row["request_count"] == 1
    assert not stop.is_set()
    trace = read_json(tmp_path / "episode/0000.tools.json")[0]
    assert trace["response"]["result_id"] == (
        row["program_result"]["rounds"][0]["tool_result_receipts"][0]["result_id"])


@pytest.fixture
def fake_gateway(tmp_path):
    envelopes, counts = [], {}
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
            content = MEASURE_ANSWER if step < 5 else '{"answer":"B","reasoning":"reviewed"}'
            data = json.dumps({
                "model": "qwen3.8-flash", "id": "offline-process",
                "choices": [{"finish_reason": "stop", "message": {
                    "role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    (tmp_path / "transport_auth.py").write_text(
        "def load_transport_token():\n    return 'OFFLINE_TRANSPORT'\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", envelopes
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_fresh_processes_execute_main_thread_kernel_with_review_and_shared_budget(tmp_path, fake_gateway):
    gateway, envelopes = fake_gateway
    skills = frozen()
    config = {
        "timeout_s": 10, "gateway": gateway, "auth_module_root": str(tmp_path),
        "skills_manifest": str(Path(__file__).resolve().parents[1] / "skills/frozen_skills.json"),
    }
    raw = RequestGovernor(tmp_path / "budget", RequestGovernorConfig(50, 8))
    stop = RunStop(tmp_path / "stop")
    wrapped = SharedSlots(raw, reserve_direct=2, stop_event=stop, stop_on_request_error=True)
    solver = ProcessSolver(
        config, {"source": source_identity(), "host_scoring_sentinel": "HOST_ONLY_GT"},
        RunnerConfig(), skills)

    def execute(task, ordinal):
        public, prepared, _ = fixture(tmp_path / f"case-{ordinal}", task=task, sample_id=f"case-{ordinal}")
        identity = prepare_request(
            public, prepared, arm="tools", frozen=skills, config=RunnerConfig())[-1]
        return solver(public, prepared, arm="tools", client=None, governor=wrapped,
                      output_directory=tmp_path / f"episode-{ordinal}", stop_event=stop,
                      expected_identity=identity)

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(execute, "Depth", 0), pool.submit(execute, "Distance", 1)
        rows = [a.result(), b.result()]
    assert all(row["status"] == "answered" and row["answer"] == "B" for row in rows)
    assert all(row["request_count"] == 6 and row["succeeded_tool_calls"] == 5 for row in rows)
    assert len(envelopes) == raw.summary()["reserved_requests"] == 12
    assert not stop.is_set()
    processes = [read_json(tmp_path / f"episode-{i}/process.json") for i in range(2)]
    assert len({p["pid"] for p in processes}) == 2 and all(p["main_thread_kernel"] for p in processes)
    for ordinal, row in enumerate(rows):
        root = tmp_path / f"episode-{ordinal}"
        assert (root / "0005.sampling_overlay.png").exists()
        trace = read_json(root / "0000.tools.json")[0]
        assert trace["response"]["result_id"] == row["program_result"]["rounds"][0]["observations"][0]["result_id"]
        assert row["receipt_sha256"] == digest({k: v for k, v in row.items() if k != "receipt_sha256"})
    assert "HOST_ONLY_GT" not in json.dumps(envelopes)
    for operation in {e["audit_identity"]["operation_id"] for e in envelopes}:
        session = [e["payload"]["messages"][0]["content"] for e in envelopes
                   if e["audit_identity"]["operation_id"] == operation]
        assert len([p for p in session[0] if p["type"] == "image_url"]) == 2
        assert all(len([p for p in content if p["type"] == "image_url"]) == 3 for content in session[1:])


def test_source_pin_and_selection_reject_drift_before_transport(tmp_path):
    source = source_identity()
    validate_source(source)
    with pytest.raises(ValueError, match="source changed"):
        validate_source({**source, "scripts/CV3Drunner/runner.py": "0" * 64})
    public, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="outside"):
        select_inputs([public], {"train_ids": ["missing"], "test_ids": ["missing"]})


def test_diagnostic_selection_is_exactly_ten_test_questions_without_training(tmp_path):
    inputs = [fixture(
        tmp_path / f"diagnostic-{i}", task="Depth" if i % 2 == 0 else "Distance",
        split="test", sample_id=f"diagnostic-{i}")[0] for i in range(10)]
    ids = [p.problem.sample_id for p in inputs]
    pilot, test = select_inputs(inputs, {"test_ids": ids}, diagnostic=True)
    assert pilot == [] and test == inputs
    with pytest.raises(ValueError, match="ten unique"):
        select_inputs(inputs, {"test_ids": ids[:-1]}, diagnostic=True)
    with pytest.raises(ValueError, match="outside"):
        select_inputs(inputs, {"test_ids": ids[:-1] + ["missing"]}, diagnostic=True)
    with pytest.raises(ValueError, match="test_ids only"):
        select_inputs(inputs, {"train_ids": [], "test_ids": ids}, diagnostic=True)


def test_full_test_selection_has_no_extra_training_episodes(tmp_path):
    public, _, _ = fixture(tmp_path, split="test")
    inputs = [replace(public, problem=replace(
        public.problem, sample_id=f"full-{i}", task_type="Depth" if i < 420 else "Distance"))
        for i in range(840)]
    train = replace(public, problem=replace(public.problem, source_split="train", sample_id="train"))
    pilot, selected = select_inputs([train, *inputs], test_only=True)
    assert pilot == [] and selected == inputs
    with pytest.raises(ValueError, match="all 840"):
        select_inputs(inputs[:-1], test_only=True)
    with pytest.raises(ValueError, match="subset"):
        select_inputs(inputs, {"test_ids": ["full-0"]}, test_only=True)
