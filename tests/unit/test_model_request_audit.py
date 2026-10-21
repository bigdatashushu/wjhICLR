import base64
import hashlib
import json

import pytest

from skill3d.evaluation.gate_process_metrics import episode_process_metrics
from skill3d.evolution.fitset_stream_v11 import SeededGateClient
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_audit import describe_request
from test_bailian_client import Response
from test_fitset_stream_v11 import make_runner


def test_effective_wire_identity_includes_clamp_seed_pixels_and_backend(monkeypatch, tmp_path):
    captured = []
    def opener(request, timeout):
        captured.append(request.data)
        return Response({
            "model": "served-model", "system_fingerprint": "backend-a",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        })
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-only-in-header")
    raw = BailianClient(opener=opener, max_output_tokens=400, model_revision="frozen-rev")
    client = SeededGateClient(raw, 138, tmp_path / "audit.json")
    client.set_audit_context({"shared_input_sha256": "f" * 64})
    client.set_request_context({"phase": "program", "images_in_order": [{"frame_slot": 31}]})
    jpeg = b"synthetic-jpeg-bytes"
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "sensitive question text"},
        {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
    ]}]
    assert client.chat(messages, max_tokens=8192, seed=137) == "ok"
    record = json.loads((tmp_path / "audit-001.json").read_text())
    identity = record["request_identity"]
    assert identity["wire_payload_sha256"] == hashlib.sha256(captured[0]).hexdigest()
    assert identity["parameters"]["seed"] == 138
    assert identity["parameters"]["max_tokens"] == 400
    assert identity["parameters"]["top_p"] is None
    assert identity["images_in_order"][0]["jpeg_sha256"] == hashlib.sha256(jpeg).hexdigest()
    response = json.loads((tmp_path / "audit-response-001.json").read_text())
    assert response["actual_request_identity"] == identity
    assert response["response_identity"]["backend_fingerprint"] == "backend-a"
    assert response["usage"]["prompt_tokens"] == 10
    assert response["elapsed_s"] >= 0
    assert "sensitive question text" not in json.dumps(record)
    assert "secret-only-in-header" not in json.dumps([record, response, raw.last_generation])


def test_request_hash_detects_text_image_order_and_sampling_changes():
    def image(value):
        return {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(value).decode()}}
    payload = {"seed": 137, "messages": [{"role": "user", "content": [
        image(b"first"), image(b"second")]}]}
    first = describe_request(payload, model_identity={})
    reversed_payload = {**payload, "messages": [{"role": "user", "content": [
        image(b"second"), image(b"first")]}]}
    reversed_audit = describe_request(reversed_payload, model_identity={})
    assert first["wire_payload_sha256"] != reversed_audit["wire_payload_sha256"]
    assert first["images_in_order"][0]["jpeg_sha256"] == (
        reversed_audit["images_in_order"][1]["jpeg_sha256"])
    assert first["wire_payload_sha256"] != describe_request(
        {**payload, "seed": 138}, model_identity={})["wire_payload_sha256"]


def test_missing_fingerprint_stays_unknown_and_failed_request_is_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    raw = BailianClient(opener=lambda *a, **k: Response({
        "choices": [{"message": {"content": "ok"}}]}))
    raw.chat([])
    assert raw.last_response_identity["backend_fingerprint"] is None
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    client = SeededGateClient(raw, 137, tmp_path / "failed.json")
    with pytest.raises(Exception):
        client.chat([])
    response = json.loads((tmp_path / "failed-response-001.json").read_text())
    assert response["status"] == "error" and response["usage"] == {}
    assert response["response_identity"] == {} and raw.last_response_identity == {}


def test_gate_balances_submission_order_and_preserves_legacy_metrics(tmp_path):
    runner = make_runner(tmp_path)
    runner.run(resume=False)
    path = next((runner.root / "workers").glob("*/campaigns/*/gate_schedule.json"))
    schedule = json.loads(path.read_text())["submission_order"]
    assert [row["arm"] for row in schedule[:4]] == ["parent", "candidate", "candidate", "parent"]
    assert [row["arm"] for row in schedule[16:18]] == ["candidate", "parent"]
    decision = json.loads((path.parent / "decision.json").read_text())
    assert decision["admission_program_error_rule"] == "legacy_any_round_error_code"
    assert decision["passed"]
    for arm in decision["arms"]:
        for name in ("parent", "candidate"):
            metrics = arm[name]["process_metrics"]
            assert metrics["request_count"] == 8
            assert metrics["token_totals_complete"] is False


def test_insufficient_gate_retains_hashed_decision_without_model_requests(tmp_path):
    runner = make_runner(tmp_path, n=24, reference="1")
    runner.gate_panels["object_counting/g0m0"] = runner.gate_panels["object_counting/g0m0"][:2]
    runner.run(resume=False)
    decisions = list((runner.root / "workers").glob("*/campaigns/*/decision.json"))
    assert len(decisions) == 1
    decision = json.loads(decisions[0].read_text())
    assert not decision["passed"] and decision["identity_sha256"]
    assert not list((runner.root / "workers").glob("*/campaigns/*/gate/*/*/request_seeds/*"))
    stream = json.loads(next((runner.root / "workers").glob("*/buffer/stream.json")).read_text())
    assert [row["outcome"] for row in stream["attempts"]] == ["rejected", "duplicate"]


def test_descriptive_error_metrics_do_not_reclassify_gate_errors():
    outcome = {"final_state": "answer", "rounds": [
        {"error_code": "tool_precondition_rejected"},
        {"error_code": "", "executed": True, "terminated": "answer", "results": []},
    ]}
    metrics = episode_process_metrics(outcome, [{"status": "completed", "elapsed_s": 0.2}])
    assert metrics["error_events"]["tool_precondition_rejected"] == 1
    assert not metrics["token_totals_complete"]


def test_online_program_and_finalizer_receipts_have_actual_frame_mapping(monkeypatch, tmp_path):
    import numpy as np
    from skill3d.adapters.frame_set import build_frame_set
    from skill3d.online import runner
    from skill3d.schemas import SceneState, VSIBenchEpisode
    from skill3d.tools.scene_handle import SceneHandle

    replies = iter(["ReturnAnswer(3)", json.dumps({
        "action": "answer", "value": 3, "basis": "visual_estimate",
        "used_result_ids": [], "derivation": None})])
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    client = SeededGateClient(BailianClient(opener=lambda *a, **k: Response({
        "choices": [{"message": {"content": next(replies)}}]})),
        137, tmp_path / "online.json")
    episode = VSIBenchEpisode(
        qa_id="fixture", scene_name="fixture-scene", dataset="scannet",
        question_type="object_counting", question="Count the objects.",
        ground_truth="private-label-sentinel", options=None, frames=[], split="inner_validation",
        frame_set=build_frame_set(64))
    scene = SceneState(artifact_ref="a", scene_route="fallback_2d_only")
    handle = SceneHandle(scene)
    cfg = runner.OnlineRunConfig(mode="real", max_images=32)
    pixels = [np.full((8, 8, 3), i, np.uint8) for i in range(32)]
    runner._configure_image_ledger(handle, cfg, episode)
    runner._attach_episode_frames(handle, episode, pixels)
    for index, trigger in ((1, "initial"), (2, "finalize")):
        result = runner._synthesize(
            episode, scene, handle, [], cfg, client, pixels=pixels,
            round_index=index, trigger=trigger)
        assert result.program is not None, result.note
    first = json.loads((tmp_path / "online-001.json").read_text())
    last = json.loads((tmp_path / "online-002.json").read_text())
    assert len(first["request_context"]["images_in_order"]) == 32
    assert first["request_context"]["images_in_order"][-1]["frame_slot"] == 31
    assert first["request_context"]["tool_directory_sha256"]
    assert last["request_context"]["phase"] == "finalization"
    assert last["request_context"]["tool_execution_allowed"] is False
    assert "private-label-sentinel" not in json.dumps([first, last])
