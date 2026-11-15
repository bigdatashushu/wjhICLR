"""Verify real native bytes, effort dispatch and paid failure usage offline."""
import json
from xcodeapi.transport import HTTPResult
from skill3d.cluster.xcode_client import RemoteXcodeClient
from skill3d.synthesis.xcode_client import XcodeResponsesClient

OPTIONS = dict(model="gpt-5.6-terra", base_url="https://xcode.best/v1",
               api_key_env="XCODE_GPT_API_KEY", timeout_s=1800, max_output_tokens=8192)


def test_native_rpc_preserves_pixels_and_effort():
    from skill3d.synthesis.request_audit import describe_request
    for effort in ("high", "low"):
        client = RemoteXcodeClient(**OPTIONS, reasoning_effort=effort,
                                   endpoint="http://127.0.0.1:27066")
        payload = client.build_payload([
            {"role": "system", "content": "System"},
            {"role": "user", "content": [
                {"type": "text", "text": "Question"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YWJj"}}]}],
            max_tokens=8192, seed=137, enable_thinking=False)
        assert payload["reasoning"] == {"effort": effort}
        assert payload["stream"] is (effort == "high")
        assert "seed" not in payload and "enable_thinking" not in payload
        assert payload["input"][0]["content"][1]["image_url"] == "data:image/jpeg;base64,YWJj"
        audit = describe_request(payload, model_identity=client.audit_identity(seed=137))
        import hashlib
        assert audit["images_in_order"][0]["jpeg_sha256"] == hashlib.sha256(b"abc").hexdigest()
        assert audit["model_identity"]["seed"] is None
        assert audit["parameters"]["max_output_tokens"] == 8192


def test_usage_survives_missing_answer(monkeypatch):
    from xcodeapi.errors import ProtocolError
    import pytest
    client = XcodeResponsesClient(**OPTIONS, reasoning_effort="high")
    payload = client.build_payload([{"role": "user", "content": "Question"}], max_tokens=8192)
    reply = HTTPResult({"id": "response", "model": OPTIONS["model"],
                        "reasoning": {"effort": "high"}, "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "output": [], "usage": {"input_tokens": 20, "output_tokens": 8192,
                                                  "total_tokens": 8212}}, 1)
    monkeypatch.setattr("skill3d.synthesis.xcode_client.ResponsesStreamTransport.request",
                        lambda *args, **kwargs: reply)
    with pytest.raises(ProtocolError):
        client.send_native(payload, api_key="offline-test")
    assert client.last_usage["total_tokens"] == 8212


def test_native_gateway_round_trip_and_key_rotation(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import threading
    spec = importlib.util.spec_from_file_location(
        "xcode_gateway_test", Path(__file__).resolve().parents[1] / "scripts/model_rpc_gateway.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private = tmp_path / "private.json"
    private.write_text(json.dumps({"keys": [
        {"key_id": "first", "api_key": "test-first"},
        {"key_id": "second", "api_key": "test-second"}]}))
    private.chmod(0o600)
    monkeypatch.setenv("VSI_XCODE_KEY_POOL", str(private))
    monkeypatch.setenv("HARNESS3D_PERCEPTION_TOKEN", "offline-rpc-test")
    def reply(transport, method, endpoint, payload):
        return HTTPResult({
            "id": "response", "model": OPTIONS["model"],
            "reasoning": payload["reasoning"], "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}],
            "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}}, 1)
    monkeypatch.setattr("skill3d.synthesis.xcode_client.JSONTransport.request", reply)
    monkeypatch.setattr("skill3d.synthesis.xcode_client.ResponsesStreamTransport.request", reply)
    gateway = module.PerceptionGateway({**OPTIONS, "provider": "xcodeapi"},
                                       tmp_path / "cache", max_concurrency=32)
    server = module.make_server(gateway, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for effort in ("high", "low"):
            client = RemoteXcodeClient(**OPTIONS, reasoning_effort=effort,
                endpoint=f"http://127.0.0.1:{server.server_address[1]}", transport_retries=0)
            assert client.chat([{"role": "user", "content": "Question"}], max_tokens=8192,
                               seed=137) == "OK"
            assert client.last_usage["total_tokens"] == 25
        health = gateway.health()
        assert health["upstream"]["started"] == health["upstream"]["succeeded"] == 2
        assert health["key_usage"] == {"first": 1, "second": 1}
    finally:
        server.shutdown()
        server.server_close()
        gateway.close()


def test_native_http_failure_retains_safe_status_without_retry(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    from xcodeapi.errors import HTTPAPIError

    spec = importlib.util.spec_from_file_location(
        "xcode_failure_gateway_test",
        Path(__file__).resolve().parents[1] / "scripts/model_rpc_gateway.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private = tmp_path / "private.json"
    private.write_text(json.dumps({"keys": [{"key_id": "first", "api_key": "offline-secret"}]}))
    private.chmod(0o600)
    monkeypatch.setenv("VSI_XCODE_KEY_POOL", str(private))
    monkeypatch.setenv("HARNESS3D_PERCEPTION_TOKEN", "offline-rpc-test")
    attempts = []

    def reject(*args, **kwargs):
        attempts.append(1)
        raise HTTPAPIError(429)

    monkeypatch.setattr("skill3d.synthesis.xcode_client.ResponsesStreamTransport.request", reject)
    client = XcodeResponsesClient(**OPTIONS, reasoning_effort="high")
    payload = client.build_payload([{"role": "user", "content": "Question"}], max_tokens=8192)
    gateway = module.PerceptionGateway({**OPTIONS, "provider": "xcodeapi"}, tmp_path / "cache")
    try:
        from skill3d.synthesis.request_audit import payload_bytes
        import hashlib
        status, reply = gateway._invoke(
            "independent-attempt", payload, client.audit_identity(),
            hashlib.sha256(payload_bytes(payload)).hexdigest())
        assert status == 503 and b"offline-secret" not in reply
        assert gateway._db.execute(
            "SELECT error_type,state,usage_status FROM upstream_audit").fetchone() == (
                "HTTP429", "error", "unknown")
        assert len(attempts) == 1
    finally:
        gateway.close()
