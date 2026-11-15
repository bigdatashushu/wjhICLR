"""Native Messages preserves complete SSE, cache usage and high/low RPC."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import threading
from time import monotonic

import pytest
from xcodeapi.errors import TransportError
from xcodeapi.transport import HTTPResult, MessagesStreamTransport
from skill3d.synthesis.xcode_client import XcodeMessagesClient, get_xcode_client_class
from skill3d.synthesis.request_audit import describe_request

OPTIONS = dict(model="claude-sonnet-5", base_url="https://xcode.best/v1",
               api_key_env="XCODE_CLAUDE_API_KEY", timeout_s=1800, max_output_tokens=8192)


def event(kind, **fields):
    return ("data: " + json.dumps({"type": kind, **fields}) + "\n\n").encode()


def test_messages_sse_preserves_thinking_signature_and_cache_usage():
    raw = event("message_start", message={
        "id": "message", "model": OPTIONS["model"], "content": [],
        "usage": {"input_tokens": 3, "cache_read_input_tokens": 7,
                  "cache_creation_input_tokens": 11, "output_tokens": 1}})
    raw += event("content_block_start", index=0, content_block={"type": "thinking", "thinking": ""})
    raw += event("content_block_delta", index=0, delta={"type": "thinking_delta", "thinking": "reason"})
    raw += event("content_block_delta", index=0, delta={"type": "signature_delta", "signature": "signed"})
    raw += event("content_block_stop", index=0)
    raw += event("content_block_start", index=1, content_block={"type": "text", "text": ""})
    raw += event("content_block_delta", index=1, delta={"type": "text_delta", "text": "OK"})
    raw += event("content_block_stop", index=1)
    raw += event("message_delta", delta={"stop_reason": "end_turn"}, usage={"output_tokens": 5})
    transport = MessagesStreamTransport("offline", "bearer", 10)
    with pytest.raises(TransportError, match="without a terminal"):
        transport._read_terminal(io.BytesIO(raw), monotonic())
    body = transport._read_terminal(io.BytesIO(raw + event("message_stop")), monotonic())
    assert body["content"][0] == {"type": "thinking", "thinking": "reason", "signature": "signed"}
    assert body["content"][1]["text"] == "OK"
    assert body["usage"] == {"input_tokens": 3, "cache_read_input_tokens": 7,
                             "cache_creation_input_tokens": 11, "output_tokens": 5}


def test_messages_rpc_effort_rotation_pixels_and_usage(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "messages_gateway", Path(__file__).resolve().parents[1] / "scripts/model_rpc_gateway.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private = tmp_path / "private.json"
    private.write_text(json.dumps({"keys": [
        {"key_id": "first", "api_key": "offline-first"},
        {"key_id": "second", "api_key": "offline-second"}]}))
    private.chmod(0o600)
    monkeypatch.setenv("VSI_XCODE_KEY_POOL", str(private))
    monkeypatch.setenv("HARNESS3D_PERCEPTION_TOKEN", "offline-rpc")
    sent = []

    def reply(transport, method, endpoint, payload):
        sent.append(payload["output_config"]["effort"])
        assert endpoint.endswith("/messages") and payload["thinking"] == {"type": "adaptive"}
        assert "seed" not in payload and "reasoning" not in payload
        return HTTPResult({"id": "message", "model": OPTIONS["model"],
            "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "cache_read_input_tokens": 7,
                      "cache_creation_input_tokens": 11, "output_tokens": 5}}, 1)

    monkeypatch.setattr(MessagesStreamTransport, "request", reply)
    gateway = module.PerceptionGateway({**OPTIONS, "provider": "xcodeapi"}, tmp_path / "cache")
    server = module.make_server(gateway, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for effort in ("high", "low"):
            client = get_xcode_client_class(OPTIONS["model"], remote=True)(
                **OPTIONS, reasoning_effort=effort,
                endpoint=f"http://127.0.0.1:{server.server_address[1]}", transport_retries=0)
            messages = [{"role": "system", "content": "System"}, {"role": "user", "content": [
                {"type": "text", "text": "Question"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YWJj"}}]}]
            audit = describe_request(client.build_payload(messages), model_identity=client.audit_identity())
            assert audit["images_in_order"][0]["jpeg_sha256"] == hashlib.sha256(b"abc").hexdigest()
            assert audit["text_parts"][0]["role"] == "system"
            assert client.chat(messages, max_tokens=8192, seed=137) == "OK"
            assert client.last_usage == {"prompt_tokens": 21, "completion_tokens": 5,
                "total_tokens": 26, "cached_input_tokens": 7, "cache_creation_input_tokens": 11}
        assert sent == ["high", "low"]
        assert gateway.health()["key_usage"] == {"first": 1, "second": 1}
        assert list(gateway._db.execute("SELECT effort,response_effort FROM xcode_evidence")) == [
            ("high", None), ("low", None)]
    finally:
        server.shutdown()
        server.server_close()
        gateway.close()


def test_messages_paid_usage_survives_missing_answer(monkeypatch):
    from xcodeapi.errors import ProtocolError
    client = XcodeMessagesClient(**OPTIONS, reasoning_effort="high")
    monkeypatch.setattr(MessagesStreamTransport, "request", lambda *a, **k: HTTPResult({
        "id": "message", "model": OPTIONS["model"], "content": [], "stop_reason": "max_tokens",
        "usage": {"input_tokens": 3, "cache_read_input_tokens": 7,
                  "cache_creation_input_tokens": 11, "output_tokens": 8192}}, 1))
    with pytest.raises(ProtocolError):
        client.send_native(client.build_payload([{"role": "user", "content": "Question"}]), api_key="offline")
    assert client.last_usage["total_tokens"] == 8213
