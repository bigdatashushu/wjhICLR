import base64
import io
import json
import urllib.error

import pytest

from MindcubeRunner.common import Config, read, digest
from MindcubeRunner.data import load
from MindcubeRunner.protocol import messages
from MindcubeRunner.replay import verify
from MindcubeRunner.traces import TraceError, TraceWriter, atomic, mark_orphans, response_fields, unwire
from MindcubeRunner.transport import ProxyOpener
from MindcubeRunner.worker import solve
from skill3d.synthesis.bailian_client import BailianClient

from test_adapter import snapshot, FakeBudget, episode, program


@pytest.mark.parametrize("reasoning", ["trace reasoning Ω\n", "", None, "ABSENT"])
def test_presence_preserves_null_empty_and_missing(reasoning):
    message = {"content": '{"answer":"B"}'}
    choice = {"message": message}
    if reasoning != "ABSENT":
        message["reasoning_content"] = reasoning
        choice["finish_reason"] = reasoning
    body = {"choices": [choice], "model": "qwen3.8-flash",
            "usage": {"total_tokens": 1, "completion_tokens_details": {"reasoning_tokens": 1}}}
    value = response_fields(json.dumps(body).encode())
    field = value["reasoning_content"]
    if reasoning == "ABSENT":
        assert field == {"presence": "absent_in_observed_provider_body"}
        assert value["finish_reason"] == field
    else:
        assert field == {"presence": "present", "value": reasoning}
        assert value["finish_reason"]["value"] == reasoning
    assert value["usage"]["value"] == body["usage"]
    assert response_fields(b'{"error":"upstream_failed"}', http_status=502)[
        "reasoning_content"] == {"presence": "not_observed_at_boundary"}


@pytest.mark.parametrize("arm,item_index", [("skill", 0), ("skill", 1), ("direct", 0)])
def test_real_client_roundtrip_bytes_and_program(snapshot, tmp_path, monkeypatch, arm, item_index):
    item = snapshot[2][item_index]
    source = program('ReturnAnswer("B")') if arm == "skill" else '{"answer":"B"}'
    body = json.dumps({"model": "qwen3.8-flash", "id": "offline",
        "choices": [{"message": {"role": "assistant", "content": source,
                                "reasoning_content": " Ω\nfull reasoning "},
                     "finish_reason": "stop"}],
        "usage": {"total_tokens": 12, "completion_tokens_details": {"reasoning_tokens": 8}}},
        ensure_ascii=False).encode()
    class HTTP:
        def open(self, request, timeout):
            return io.BytesIO(body)
    monkeypatch.setattr("MindcubeRunner.transport.urllib.request.build_opener", lambda *a: HTTP())
    monkeypatch.setenv("OFFLINE_KEY", "offline-placeholder")
    client = BailianClient(api_key_env="OFFLINE_KEY", opener=ProxyOpener(
        "http://127.0.0.1:1234", lambda: "offline-transport-secret", "operation", "test"))
    directory = tmp_path / "episode"
    directory.mkdir()
    config = Config()
    job = {"item": item, "arm": arm, "output_directory": str(directory),
        "stop_file": str(tmp_path / "STOP.json"), "config": config.public(), "run_identity": "test",
        "initial_receipt": messages(item, load(item), arm=arm, config=config)[1]}
    result = solve(job, client, FakeBudget())
    path = directory / "traces/00"
    report = verify(path)
    assert report["wire_request_verified"] and report["provider_body_observed"]
    assert report["images"] == len(item["views"]) * (2 if arm == "skill" else 1)
    events = read(path / "trace.json")["events"]
    response = next(e["data"] for e in events if e["kind"] == "response_body")
    assert unwire(response["body"]) == body
    assert response["fields"]["reasoning_content"]["value"] == " Ω\nfull reasoning "
    if arm == "skill":
        assert report["program_preserved"]
        assert next(e["data"]["source"] for e in events if e["kind"] == "program") in source
    assert result["answer"] == "B"
    assert "offline-transport-secret" not in (path / "trace.json").read_text()


@pytest.mark.parametrize("failure", ["http", "json", "model", "read"])
def test_failure_body_and_terminal_not_lost(snapshot, tmp_path, monkeypatch, failure):
    body = (b'{"error":"upstream_failed"}' if failure == "http" else
            b"invalid json" if failure == "json" else json.dumps({
                "model": "changed-model", "choices": [{"message": {"content": '{"answer":"B"}'},
                                                     "finish_reason": None}]}).encode())
    class Broken(io.BytesIO):
        def read(self, *args):
            raise TimeoutError()
    class HTTP:
        def open(self, request, timeout):
            if failure == "http":
                raise urllib.error.HTTPError("http://127.0.0.1", 502, "offline", {}, io.BytesIO(body))
            return Broken(body) if failure == "read" else io.BytesIO(body)
    monkeypatch.setattr("MindcubeRunner.transport.urllib.request.build_opener", lambda *a: HTTP())
    monkeypatch.setenv("OFFLINE_KEY", "offline-placeholder")
    client = BailianClient(api_key_env="OFFLINE_KEY", opener=ProxyOpener(
        "http://127.0.0.1:1234", lambda: "offline-transport-secret", "operation", "test"))
    item = snapshot[2][0]
    directory = tmp_path / "episode"
    directory.mkdir()
    config = Config()
    job = {"item": item, "arm": "direct", "output_directory": str(directory),
        "stop_file": str(tmp_path / "STOP.json"), "config": config.public(), "run_identity": "test",
        "initial_receipt": messages(item, load(item), arm="direct", config=config)[1]}
    result = solve(job, client, FakeBudget())
    path = directory / "traces/00"
    assert result["status"] == "service_error" and verify(path)["status"] == "terminal"
    events = read(path / "trace.json")["events"]
    bodies = [e["data"] for e in events if e["kind"] == "response_body"]
    if failure == "read":
        assert not bodies
        assert any(e["kind"] == "response_unavailable" for e in events)
    else:
        assert unwire(bodies[0]["body"]) == body
        if failure != "model":
            assert bodies[0]["fields"]["model"]["presence"] == "not_observed_at_boundary"
    assert (directory / "00.execution.json").is_file()


def test_tool_parameters_results_and_review_state(snapshot, tmp_path):
    result, _, _, directory = episode(tmp_path, snapshot[2][0], [
        program('r = ctx.tools.point_depth(0, 10, 10)\nReturnAnswer("A")'),
        program('ReturnAnswer("B")')])
    first = read(directory / "traces/00/trace.json")
    started = next(e["data"] for e in first["events"] if e["kind"] == "tool_started")
    finished = next(e["data"] for e in first["events"] if e["kind"] == "tool_finished")
    assert started["positional_args"] == [0, 10, 10]
    assert finished["calls"][0]["args"] == {"view_index": 0, "x": 10, "y": 10}
    assert finished["results"][0]["result_id"] == result["rounds"][0]["new_measurement_ids"][0]
    assert json.loads(finished["results"][0]["value"])["sample_stability"]["point"]["reliable"]
    assert first["events"][-1]["data"]["state_after"]["review_ids"]
    assert verify(directory / "traces/00")["program_preserved"]
    second = read(directory / "traces/01/trace.json")
    assert second["events"][1]["data"]["review_ids"] == first["events"][-1]["data"]["state_after"]["review_ids"]


def test_tool_binding_failure_preserves_original_positionals(snapshot, tmp_path):
    _, _, _, directory = episode(tmp_path, snapshot[2][0], [
        program('ctx.tools.point_depth(0, 10, 10, 99)\nReturnAnswer("A")')])
    events = read(directory / "traces/00/trace.json")["events"]
    start = next(e["data"] for e in events if e["kind"] == "tool_started")
    end = next(e["data"] for e in events if e["kind"] == "tool_finished")
    assert start["positional_args"] == [0, 10, 10, 99]
    assert end["error"] and end["results"][0]["status"] == "failed"


def test_orphan_chain_tamper_and_atomic_exclusive(tmp_path):
    writer = TraceWriter(tmp_path / "traces/00", {"sample_id": "test"})
    writer.event("state_before", {"review_ids": ["r1"]})
    mark_orphans(tmp_path, "process_exit")
    mark_orphans(tmp_path, "process_exit")
    assert verify(writer.directory)["status"] == "interrupted"
    path = writer.directory / "events/000001.json"
    value = read(path)
    value["data"]["review_ids"] = []
    atomic(path, value)
    with pytest.raises(TraceError, match="event_hash_changed"):
        verify(writer.directory)
    with pytest.raises(FileExistsError):
        atomic(path, {}, exclusive=True)


def test_privacy_and_commit_failure_stop_before_publication(tmp_path, monkeypatch):
    writer = TraceWriter(tmp_path / "trace", {"sample_id": "test"}, secrets=("SENTINEL_SECRET",))
    for data in ({"gt_answer": "GT_SENTINEL"}, {"body": "SENTINEL_SECRET"}):
        with pytest.raises(TraceError):
            writer.event("prepared", data)
    saved = (writer.directory / "trace.json").read_text()
    assert "GT_SENTINEL" not in saved and "SENTINEL_SECRET" not in saved
    import MindcubeRunner.traces as module
    monkeypatch.setattr(module.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        writer.event("prepared", {"messages": []})
    assert writer.failed
    assert len(list((writer.directory / "events").glob("*.json"))) == 1
