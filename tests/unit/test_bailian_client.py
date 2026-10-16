from __future__ import annotations

import io
import json
import urllib.error

import pytest

from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.vllm_client import ServiceUnavailable


class Response:
    def __init__(self, payload):
        self.body = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body.read()


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
