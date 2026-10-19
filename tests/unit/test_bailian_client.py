from __future__ import annotations

import io
import json
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
