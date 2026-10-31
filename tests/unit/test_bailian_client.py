from __future__ import annotations

import io
import errno
import http.client
import json
import traceback
import urllib.error

import pytest

from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis import bailian_client as bailian
from skill3d.synthesis.client_errors import ServiceUnavailable
from skill3d.synthesis.program_assembler import assemble_program_ex
from skill3d.online.runner import OnlineRunConfig, _make_online_client


class Response:
    def __init__(self, payload):
        self.body = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body.read()


def test_connect_prefers_ipv4_and_falls_back_without_global_socket_changes(monkeypatch):
    addresses = [
        (bailian.socket.AF_INET6, bailian.socket.SOCK_STREAM, 0, "", ("::1", 443)),
        (bailian.socket.AF_INET, bailian.socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443)),
    ]
    monkeypatch.setattr(bailian.socket, "getaddrinfo", lambda *args: addresses)
    attempts = []

    class Connection:
        def __init__(self, family, *_):
            self.family = family
            self.closed = False
            self.timeouts = []
            attempts.append(self)

        def settimeout(self, timeout):
            self.timeouts.append(timeout)

        def connect(self, address):
            if self.family == bailian.socket.AF_INET:
                raise OSError("test IPv4 unavailable")

        def close(self):
            self.closed = True

    monkeypatch.setattr(bailian.socket, "socket", Connection)
    connected = bailian._connect_addresses(("example", 443), 180)
    assert [row.family for row in attempts] == [
        bailian.socket.AF_INET, bailian.socket.AF_INET6,
    ]
    assert attempts[0].closed
    assert connected is attempts[1]
    assert connected.timeouts == [5, 180]


def test_bailian_reads_key_at_request_time_and_records_public_identity(monkeypatch):
    captured = {}

    def opener(request, timeout):
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response({
            "id": "response-id", "model": "qwen3.8-flash", "created": 1,
            "choices": [{"message": {"content": '{"ok":true}'},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4,
                      "total_tokens": 7},
        })

    client = BailianClient(opener=opener, model_revision="2026-09-08")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-value")
    result = client.chat(
        [{"role": "user", "content": "Return JSON"}],
        response_format={"type": "json_object"}, enable_thinking=False,
        temperature=0.0, seed=137)
    assert result == '{"ok":true}'
    assert captured["authorization"] == "Bearer secret-value"
    assert captured["payload"]["model"] == "qwen3.8-flash"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["enable_thinking"] is False
    assert client.last_usage["total_tokens"] == 7
    identity = client.audit_identity(seed=137)
    assert identity["cache_eligible"] and identity["model_revision"] == "2026-09-08"
    assert "secret" not in json.dumps(identity)


def test_bailian_missing_or_rejected_key_is_service_unavailable(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    client = BailianClient(opener=lambda *_args, **_kwargs: None)
    with pytest.raises(ServiceUnavailable, match="DASHSCOPE_API_KEY"):
        client.chat([{"role": "user", "content": "hello"}])

    def rejected(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://example", 401, "unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setenv("DASHSCOPE_API_KEY", "do-not-leak")
    with pytest.raises(ServiceUnavailable) as error:
        BailianClient(opener=rejected).chat([{"role": "user", "content": "hello"}])
    assert "do-not-leak" not in str(error.value)


@pytest.mark.parametrize("failure,expected_code,expected_type", [
    (TimeoutError("sensitive request detail"), "timeout", "TimeoutError"),
    (urllib.error.URLError(TimeoutError("sensitive request detail")), "timeout", "TimeoutError"),
    (urllib.error.URLError(OSError(errno.ETIMEDOUT, "sensitive request detail")),
     "timeout", "TimeoutError"),
    (urllib.error.URLError("sensitive request detail"), "transport_error", "URLError"),
    (OSError("sensitive request detail"), "transport_error", "OSError"),
    (http.client.RemoteDisconnected("sensitive request detail"),
     "transport_error", "HTTPException"),
])
def test_transport_failure_records_safe_stage_without_retry(
        monkeypatch, failure, expected_code, expected_type):
    calls = []
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-secret-only")

    def opener(request, timeout):
        calls.append(timeout)
        raise failure

    client = BailianClient(opener=opener, timeout_s=600)
    with pytest.raises(ServiceUnavailable) as error:
        client.chat([{"role": "user", "content": "sensitive prompt"}])
    assert calls == [600]
    diagnostic = client.last_service_error
    assert diagnostic["reason_code"] == expected_code
    assert diagnostic["exception_type"] == expected_type
    assert diagnostic["request_stage"] == "awaiting_headers"
    assert diagnostic["http_status"] is None
    assert diagnostic["timeout_s"] == 600 and diagnostic["elapsed_seconds"] >= 0
    assert error.value.__suppress_context__ and error.value.__cause__ is None
    rendered = json.dumps(diagnostic) + "".join(traceback.format_exception_only(error.value))
    for forbidden in ("sensitive", "test-secret-only", "Authorization", "https://"):
        assert forbidden not in rendered


def test_body_timeout_is_distinct_and_diagnostic_resets_after_success(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")

    class TimedOut(Response):
        def read(self):
            raise TimeoutError("private body detail")

    responses = iter([TimedOut({}), Response({
        "model": "qwen3.8-flash",
        "choices": [{"message": {"content": "B"}, "finish_reason": "stop"}],
    })])
    calls = []

    def opener(request, timeout):
        calls.append(timeout)
        return next(responses)

    client = BailianClient(opener=opener, timeout_s=600)
    with pytest.raises(ServiceUnavailable):
        client.chat([{"role": "user", "content": "test"}])
    assert calls == [600]
    assert client.last_service_error["reason_code"] == "timeout"
    assert client.last_service_error["request_stage"] == "reading_body"
    assert client.chat([{"role": "user", "content": "test"}]) == "B"
    assert calls == [600, 600] and client.last_service_error == {}


@pytest.mark.parametrize("status,error_type", [
    (400, RuntimeError), (401, ServiceUnavailable), (408, ServiceUnavailable),
    (429, ServiceUnavailable), (500, ServiceUnavailable), (503, ServiceUnavailable),
])
def test_http_error_preserves_status_without_reading_error_body_or_retry(
        monkeypatch, status, error_type):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-secret-only")
    calls = []

    class SensitiveBody(io.BytesIO):
        def read(self, *args):
            pytest.fail("provider error body must not be read")

    body = SensitiveBody(b"test-secret-only private-provider-body")

    def opener(request, timeout):
        calls.append(timeout)
        raise urllib.error.HTTPError(
            "https://private-endpoint", status, "private-provider-detail", {}, body)

    client = BailianClient(opener=opener, timeout_s=600)
    with pytest.raises(error_type) as error:
        client.chat([{"role": "user", "content": "private prompt"}])
    assert calls == [600] and body.closed
    assert client.last_service_error["http_status"] == status
    assert client.last_service_error["reason_code"] == "http_error"
    assert client.last_service_error["request_stage"] == "awaiting_headers"
    assert error.value.__suppress_context__ and error.value.__cause__ is None
    rendered = json.dumps(client.last_service_error) + "".join(
        traceback.format_exception_only(error.value))
    assert "private" not in rendered and "test-secret-only" not in rendered


@pytest.mark.parametrize("raw", [b"private malformed JSON", b"\xff", b"[]"])
def test_invalid_response_records_safe_diagnostic_without_provider_text(monkeypatch, raw):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    client = BailianClient(opener=lambda *args, **kwargs: io.BytesIO(raw))
    with pytest.raises(RuntimeError) as error:
        client.chat([{"role": "user", "content": "test"}])
    assert client.last_service_error["reason_code"] == "invalid_response"
    assert "private" not in str(error.value)


def test_runner_uses_one_flash_client_for_all_online_phases(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    cfg = OnlineRunConfig(max_tokens=8192)
    client = _make_online_client(cfg)
    assert isinstance(client, BailianClient)
    assert client.model == "qwen3.8-flash"
    assert client.max_output_tokens == 8192
    assert _make_online_client(
        OnlineRunConfig(online_model_enabled=False)) is None


def test_bailian_multimodal_program_response_matches_m8_contract(monkeypatch):
    captured = {}

    def opener(request, timeout):
        captured["payload"] = json.loads(request.data)
        return Response({
            "choices": [{"message": {
                "content": "```python\nReturnAnswer('A')\n```"},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        })

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-only")
    client = BailianClient(opener=opener, max_output_tokens=8192)
    text = client.chat([{
        "role": "user",
        "content": [
            {"type": "text", "text": "Write the program."},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64,AA=="}},
        ],
    }], max_tokens=8192, seed=137)
    program, recovered = assemble_program_ex(text, [])
    assert not recovered
    assert program.program_source.strip() == "ReturnAnswer('A')"
    assert captured["payload"]["model"] == "qwen3.8-flash"
    assert captured["payload"]["max_tokens"] == 8192
