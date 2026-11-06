"""No service calls: check isolation, CPU/API admission, and export races."""
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import urllib.request


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
pilot = importlib.import_module("pilot40")
exporter = importlib.import_module("export_pilot_trajectories")


def fake_episode(root, qa, cpu, api, count, lock, ready, events):
    pilot.CPU_SLOT, pilot.API_SLOT = cpu, api
    pilot.export_question = lambda *a, **kw: None
    cfg = SimpleNamespace(
        online_model="qwen3.8-flash", online_base_url="https://invalid.local/v1",
        online_api_key_env="PILOT_TEST_KEY", online_timeout_s=900,
        temperature=0., max_tokens=123, online_model_revision="test")
    client = pilot.RecordingClient(cfg, root / "solver_requests" / qa)

    class Response:
        status = 200

        def __enter__(self):
            with lock:
                count.value += 1
                if count.value == 2:
                    ready.set()
            if not ready.wait(5):
                raise RuntimeError("CPU slot was held while waiting for the provider")
            return self

        def __exit__(self, *args):
            with lock:
                count.value -= 1

        def read(self):
            return b'{"choices":[{"message":{"content":"ok","reasoning_content":"native"}}]}'

    client.native_opener = lambda *a, **kw: Response()
    request = urllib.request.Request("https://invalid.local/v1",
                                     data=b'{"enable_thinking":true}')
    with pilot.cpu_admission():
        client.record(request, timeout=900)
    events.put(qa)


def test_waiting_provider_releases_cpu_and_allows_concurrent_api(tmp_path):
    ctx = mp.get_context("fork")
    cpu, api = ctx.BoundedSemaphore(1), ctx.BoundedSemaphore(2)
    count, lock, ready, events = ctx.Value("i", 0), ctx.Lock(), ctx.Event(), ctx.Queue()
    processes = [ctx.Process(target=fake_episode,
                             args=(tmp_path, qa, cpu, api, count, lock, ready, events))
                 for qa in ("1", "2")]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
    assert all(p.exitcode == 0 for p in processes)
    assert {events.get(timeout=1) for _ in processes} == {"1", "2"}
    for qa in ("1", "2"):
        receipt = json.loads((tmp_path / "solver_requests" / qa / "001.json").read_text())
        assert receipt["status"] == "returned"
        assert receipt["transport_timeout_s"] == 900
        assert receipt["request"] == {"enable_thinking": True}
        assert receipt["reference_answer_visible"] is False
        assert receipt["automatic_retry"] is False
        assert receipt["raw_response"]["choices"][0]["message"]["reasoning_content"] == "native"


def test_parallel_exports_merge_index_and_per_episode_topics(tmp_path, monkeypatch):
    exporter.write_json(tmp_path / "sampling_manifest.json", {
        "n_questions": 2, "questions": [{"qa_id": qa, "question": "public"} for qa in ("1", "2")]})
    for qa in ("1", "2"):
        exporter.write_json(tmp_path / "solver_requests" / qa / "001.json", {"status": "returned"})
        exporter.write_json(tmp_path / "outcomes" / f"{qa}.json", {"outcome": {"rounds": []}})
        trace = tmp_path / "solver_trace" / qa / "rounds.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text(json.dumps({"qa_id": qa, "event": "isolated"}) + "\n")
    legacy = tmp_path / "solver_trace" / "rounds.jsonl"
    legacy.write_text(json.dumps({"qa_id": "1", "event": "legacy"}) + "\n")
    barrier = threading.Barrier(2)
    native_digest = exporter.digest

    def concurrent_digest(path):
        if path.parent.name == "outcomes":
            barrier.wait(timeout=5)
        return native_digest(path)

    monkeypatch.setattr(exporter, "digest", concurrent_digest)
    errors = []

    def export(qa):
        try:
            exporter.export_question(tmp_path, qa)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=export, args=(qa,)) for qa in ("1", "2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not errors
    index = json.loads((tmp_path / "trajectory_index.json").read_text())
    assert set(index["questions"]) == {"1", "2"}
    for qa in ("1", "2"):
        full = json.loads((tmp_path / "trajectories" / f"{qa}.json").read_text())
        assert full["completeness"]["all_model_requests_returned"]
        assert any(r["event"] == "isolated" for r in full["trace_topics"]["rounds"])
    full = json.loads((tmp_path / "trajectories" / "1.json").read_text())
    assert len(full["trace_topics"]["rounds"]) == 2


def test_incremental_snapshot_keeps_native_journals_and_secret_guard(tmp_path, monkeypatch):
    native = tmp_path / "solver_requests" / "1" / "001.json"
    exporter.write_json(native, {"status": "started", "request": {"content": "public"}})
    destination = exporter.export_question(tmp_path, "1", incremental=True)
    value = json.loads(destination.read_text())
    assert value["journal_files"][0]["file"] == "solver_requests/1/001.json"
    assert json.loads(native.read_text())["request"] == {"content": "public"}
    monkeypatch.setenv("DASHSCOPE_API_KEY", "synthetic-private-test-key")
    import pytest
    with pytest.raises(ValueError, match="credentials"):
        exporter.write_json(tmp_path / "unsafe.json", {"data": "synthetic-private-test-key"})
    assert not (tmp_path / "unsafe.json").exists()


def test_refuses_to_resend_an_unfinished_episode(tmp_path):
    path = tmp_path / "scene_audits" / "scene.json"
    exporter.write_json(path, {})
    exporter.write_json(tmp_path / "solver_requests" / "1" / "001.json", {"status": "returned"})
    provider = SimpleNamespace(entries={"1": {"scene_id": "scene"}})
    import pytest
    with pytest.raises(RuntimeError, match="automatic retry disabled"):
        pilot.solve_question(tmp_path, None, provider, None, {}, "1")
