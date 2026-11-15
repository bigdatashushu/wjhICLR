"""DeepSeek native max/low RPC, final SSE usage and model revision evidence."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import threading
from time import monotonic

import pytest
from xcodeapi.errors import ProtocolError, TransportError
from xcodeapi.transport import ChatStreamTransport, ResponsesStreamTransport, HTTPResult
from skill3d.synthesis.xcode_client import DeepSeekChatClient, DeepSeekResponsesClient, get_xcode_client_class
from skill3d.synthesis.request_audit import describe_request

OPTIONS = dict(model="deepseek-v4.1-flash", model_revision="deepseek-v4.1-flash-1017",
               base_url="https://xcode.best/v1", api_key_env="XCODE_GPT_API_KEY",
               timeout_s=1800, max_output_tokens=8192)


def chunk(**fields):
    return ("data: " + json.dumps(fields) + "\n\n").encode()


def test_chat_terminal_usage_thinking_and_truncation():
    prefix = chunk(id="chat", model=OPTIONS["model_revision"], choices=[
        {"index": 0, "delta": {"reasoning_content": "reason"}, "finish_reason": None}])
    prefix += chunk(choices=[{"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}])
    prefix += chunk(choices=[], usage={"prompt_tokens": 12, "completion_tokens": 8,
        "total_tokens": 20, "completion_tokens_details": {"reasoning_tokens": 6}})
    transport = ChatStreamTransport("offline", "bearer", 10)
    with pytest.raises(TransportError, match="without a terminal"):
        transport._read_terminal(io.BytesIO(prefix), monotonic())
    body = transport._read_terminal(io.BytesIO(prefix + b"data: [DONE]\n\n"), monotonic())
    assert body["choices"][0]["message"] == {
        "role": "assistant", "content": "OK", "reasoning_content": "reason"}
    assert body["usage"]["total_tokens"] == 20
    with pytest.raises(ProtocolError, match="preceded"):
        transport._read_terminal(io.BytesIO(b"data: [DONE]\n\n"), monotonic())


def test_deepseek_rpc_pixels_effort_revision_and_paid_usage(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "deepseek_gateway", Path(__file__).resolve().parents[1] / "scripts/model_rpc_gateway.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private = tmp_path / "private.json"
    private.write_text(json.dumps({"keys": [
        {"key_id": "first", "api_key": "offline-first"},
        {"key_id": "second", "api_key": "offline-second"}]}))
    private.chmod(0o600)
    monkeypatch.setenv("VSI_XCODE_KEY_POOL", str(private))
    monkeypatch.setenv("HARNESS3D_PERCEPTION_TOKEN", "offline-rpc")
    efforts = []

    def reply(transport, method, endpoint, payload):
        efforts.append(payload["reasoning"]["effort"])
        assert endpoint.endswith("/responses")
        assert payload["stream"] is True and payload["store"] is False
        return HTTPResult({"id": "chat", "model": OPTIONS["model_revision"],
            "status": "completed", "output": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "reason"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "OK"}]}],
            "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20,
                      "output_tokens_details": {"reasoning_tokens": 6}}}, 1)

    monkeypatch.setattr(ResponsesStreamTransport, "request", reply)
    gateway = module.PerceptionGateway({**OPTIONS, "provider": "xcodeapi"}, tmp_path / "cache")
    server = module.make_server(gateway, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for effort in ("max", "low"):
            client = get_xcode_client_class(OPTIONS["model"], remote=True)(
                **OPTIONS, reasoning_effort=effort,
                endpoint=f"http://127.0.0.1:{server.server_address[1]}", transport_retries=0)
            messages = [{"role": "system", "content": "System"}, {"role": "user", "content": [
                {"type": "text", "text": "Question"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YWJj"}}]}]
            payload = client.build_payload(messages, max_tokens=8192, seed=137)
            audit = describe_request(payload, model_identity=client.audit_identity())
            assert payload["reasoning"] == {"effort": effort}
            assert audit["parameters"]["reasoning"] == {"effort": effort}
            assert "seed" not in payload and "enable_thinking" not in payload
            assert audit["images_in_order"][0]["jpeg_sha256"] == hashlib.sha256(b"abc").hexdigest()
            assert client.chat(messages, max_tokens=8192, seed=137) == "OK"
            assert client.last_usage == {"prompt_tokens": 12, "completion_tokens": 8,
                                        "total_tokens": 20, "reasoning_tokens": 6}
            assert client.last_response_identity["model"] == OPTIONS["model_revision"]
            assert client.returned_effort is None
        assert efforts == ["max", "low"]
        assert gateway.health()["key_usage"] == {"first": 1, "second": 1}
        assert list(gateway._db.execute("SELECT effort,response_model FROM xcode_evidence")) == [
            ("max", OPTIONS["model_revision"]), ("low", OPTIONS["model_revision"])]
    finally:
        server.shutdown()
        server.server_close()
        gateway.close()
    client = DeepSeekChatClient(**OPTIONS, reasoning_effort="max")
    monkeypatch.setattr(ChatStreamTransport, "request", lambda *a, **k: HTTPResult({
        "id": "chat", "model": OPTIONS["model_revision"],
        "choices": [{"message": {"content": "", "reasoning_content": "exhausted"},
                     "finish_reason": "length"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8192, "total_tokens": 8204}}, 1))
    with pytest.raises(ProtocolError):
        client.send_native(client.build_payload([{"role": "user", "content": "Question"}]),
                           api_key="offline")
    assert client.last_usage["total_tokens"] == 8204


def test_deepseek_responses_requires_revision_and_rejects_conflicting_effort(monkeypatch):
    client = DeepSeekResponsesClient(**OPTIONS, reasoning_effort="max")
    assert get_xcode_client_class(OPTIONS["model"]) is DeepSeekResponsesClient
    payload = client.build_payload([{"role": "user", "content": "Question"}])
    base = {"id": "response", "model": OPTIONS["model_revision"], "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}],
            "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}}
    for changed in ({"model": "deepseek-flash"}, {"reasoning": {"effort": "low"}}):
        monkeypatch.setattr(ResponsesStreamTransport, "request",
            lambda *a, **k: HTTPResult({**base, **changed}, 1))
        with pytest.raises(ValueError, match="mismatch"):
            client.send_native(payload, api_key="offline")
        assert client.last_usage["total_tokens"] == 20
    monkeypatch.setattr(ResponsesStreamTransport, "request", lambda *a, **k: HTTPResult(base, 1))
    assert client.send_native(payload, api_key="offline") == "OK"


def test_chat_empty_done_records_upstream_shape_without_accepting_answer():
    transport = ChatStreamTransport("offline", "bearer", 10)
    with pytest.raises(ProtocolError, match="preceded"):
        transport._read_terminal(io.BytesIO(b": keepalive\n\ndata: [DONE]\n\n"), monotonic())
    observation = transport.last_transport_metadata["chat_stream"]
    assert observation["comment_count"] == 1 and observation["done_seen"]
    assert observation["event_count"] == 0
    assert not observation["model_seen"] and not observation["finish_seen"]


def test_responses_disconnect_and_done_preserve_safe_progress_without_answer():
    prefix = b": keepalive\n\n" + chunk(type="response.created")
    prefix += chunk(type="response.reasoning_summary_text.delta", delta="private reasoning")
    prefix += chunk(type="response.output_text.delta", delta="B")
    transport = ResponsesStreamTransport("offline", "bearer", 10)
    for ending, error, message in (
        (b"", TransportError, "without a terminal"),
        (b"data: [DONE]\n\n", ProtocolError, "authoritative terminal"),
    ):
        with pytest.raises(error, match=message):
            transport._read_terminal(io.BytesIO(prefix + ending), monotonic())
        observation = transport.last_transport_metadata["responses_stream"]
        assert observation["event_count"] == 3 and observation["comment_count"] == 1
        assert observation["reasoning_text_characters"] == len("private reasoning")
        assert observation["output_text_characters"] == 1
        assert not observation["terminal_seen"]
        assert observation["done_seen"] is bool(ending)
        assert "private reasoning" not in json.dumps(transport.last_transport_metadata)
