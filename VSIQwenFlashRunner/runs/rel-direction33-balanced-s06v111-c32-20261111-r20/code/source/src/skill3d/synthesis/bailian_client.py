"""Minimal OpenAI-compatible client for Alibaba Bailian Qwen models."""

from __future__ import annotations

import errno
import hashlib
import http.client
import json
import os
import socket
import time
from typing import Optional
import urllib.error
import urllib.request

from .client_errors import ServiceUnavailable
from .request_audit import describe_request, payload_bytes

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"


def _connect_addresses(address, timeout, source_address=None):
    """Prefer IPv4 and bound each connect attempt without altering TLS checks."""
    host, port = address
    deadline = time.monotonic() + float(timeout)
    addresses = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
    addresses.sort(key=lambda row: row[0] != socket.AF_INET)
    last_error = None
    for family, socktype, protocol, _, sockaddr in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        connection = socket.socket(family, socktype, protocol)
        try:
            connection.settimeout(min(5.0, remaining))
            if source_address is not None:
                connection.bind(source_address)
            connection.connect(sockaddr)
            connection.settimeout(float(timeout))
            return connection
        except OSError as exc:
            last_error = exc
            connection.close()
    if last_error is not None:
        raise last_error
    raise TimeoutError("Bailian connection address budget exhausted")


class _HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_addresses


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(
            _HTTPSConnection, request, context=self._context,
            check_hostname=self._check_hostname)


def _base_url(value: str) -> str:
    base = str(value or DEFAULT_BASE_URL).strip().rstrip("/")
    if not base.endswith("/v1"):
        raise ValueError("Bailian base_url must end with /v1")
    if not base.startswith("https://"):
        raise ValueError("Bailian base_url must use HTTPS")
    return base


def _transport_error(exc):
    """Classify known exception types without inspecting their text or payload."""
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    timed_out = isinstance(reason, TimeoutError) or (
        isinstance(reason, OSError) and reason.errno == errno.ETIMEDOUT)
    if timed_out:
        return "timeout", "TimeoutError"
    if isinstance(exc, urllib.error.URLError):
        return "transport_error", "URLError"
    if isinstance(exc, http.client.HTTPException):
        return "transport_error", "HTTPException"
    return "transport_error", "OSError"


class BailianClient:
    """Shared Qwen client for M5, M8 and evaluation visual fallback.

    The API key is resolved at request time and is never retained in receipts or
    object identity. The model/base URL are public configuration; only a SHA-256
    of the endpoint is exposed to audit records.
    """

    provider = "bailian"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        timeout_s: float = 180.0,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        model_revision: str = "",
        opener=None,
    ) -> None:
        if not str(model).strip():
            raise ValueError("Bailian model must be non-empty")
        if not str(api_key_env).strip():
            raise ValueError("Bailian API key environment name must be non-empty")
        if timeout_s <= 0 or max_output_tokens <= 0:
            raise ValueError("Bailian timeout and max_output_tokens must be positive")
        self.model = str(model).strip()
        self.base_url = _base_url(base_url)
        self.api_key_env = str(api_key_env).strip()
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.model_revision = str(model_revision or "")
        self.endpoint_hash = hashlib.sha256(self.base_url.encode()).hexdigest()
        self._opener = opener or urllib.request.build_opener(_HTTPSHandler()).open
        self.last_usage: dict = {}
        self.last_generation: dict = {}
        self.last_response_identity: dict = {}
        self.last_request_identity: dict = {}
        self.last_reasoning_content: str | None = None
        self.last_service_error: dict = {}

    def _record_service_error(self, code, stage, started, exception_type, http_status=None):
        # Only locally generated codes and numeric values enter this receipt.
        # No URL, headers, provider body, exception text, or credential is retained.
        self.last_service_error = {
            "schema": "bailian-service-error/1",
            "reason_code": code,
            "request_stage": stage,
            "exception_type": exception_type,
            "http_status": http_status,
            "elapsed_seconds": round(max(0.0, time.monotonic() - started), 6),
            "timeout_s": self.timeout_s,
        }

    @property
    def configured(self) -> bool:
        return bool(os.environ.get(self.api_key_env, "").strip())

    def check_service(self) -> tuple[bool, str]:
        if not self.configured:
            return False, f"{self.api_key_env} is not configured"
        return True, (
            f"configured provider={self.provider} model={self.model} "
            f"endpoint_hash={self.endpoint_hash}")

    def audit_identity(self, *, seed: Optional[int] = None) -> dict:
        """Public model identity. Mutable aliases deliberately disable disk reuse."""
        return {
            "enabled": True,
            "provider": self.provider,
            "model": self.model,
            "model_revision": self.model_revision,
            "endpoint_sha256": self.endpoint_hash,
            "seed": seed,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "cache_eligible": bool(self.model_revision),
        }

    def build_payload(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
        enable_thinking: bool = False,
        top_p: Optional[float] = None,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": min(int(max_tokens), self.max_output_tokens),
            "enable_thinking": bool(enable_thinking),
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if response_format is not None:
            payload["response_format"] = response_format
        if top_p is not None:
            payload["top_p"] = float(top_p)
        return payload

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
        enable_thinking: bool = False,
        top_p: Optional[float] = None,
    ) -> str:
        started = time.monotonic()
        self.last_service_error = {}
        payload = self.build_payload(
            messages, max_tokens=max_tokens, seed=seed, temperature=temperature,
            response_format=response_format, enable_thinking=enable_thinking, top_p=top_p)
        self.last_request_identity = describe_request(
            payload, model_identity=self.audit_identity(seed=seed))
        self.last_usage = {}
        self.last_response_identity = {}
        self.last_reasoning_content = None
        self.last_generation = {
            "mode": "nonstream", "usage_status": "unknown",
            "provider": self.provider, "endpoint_sha256": self.endpoint_hash,
            "request_identity": self.last_request_identity,
        }
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            self._record_service_error(
                "missing_credentials", "configuration", started, "ServiceUnavailable")
            raise ServiceUnavailable(f"{self.api_key_env} is not configured")
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=payload_bytes(payload),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        stage = "awaiting_headers"
        try:
            with self._opener(request, timeout=self.timeout_s) as response:
                stage = "reading_body"
                raw = response.read()
                stage = "decoding_body"
                body = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            self._record_service_error(
                "http_error", stage, started, "HTTPError", http_status=int(exc.code))
            # Reading an error body can block again and obscure the HTTP status.
            exc.close()
            message = f"Bailian HTTP {exc.code}"
            if exc.code in {401, 403, 404, 408, 409, 429} or exc.code >= 500:
                raise ServiceUnavailable(message) from None
            raise RuntimeError(message) from None
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            code, exception_type = _transport_error(exc)
            self._record_service_error(code, stage, started, exception_type)
            raise ServiceUnavailable(f"Bailian {code} at {stage}") from None
        except (UnicodeError, ValueError) as exc:
            self._record_service_error(
                "invalid_response", stage, started,
                "UnicodeError" if isinstance(exc, UnicodeError) else "ValueError")
            raise RuntimeError("invalid Bailian response encoding or JSON") from None
        failure_message = "invalid Bailian response"
        try:
            if not isinstance(body, dict):
                raise ValueError("expected response object")
            usage = body.get("usage") or {}
            self.last_usage = dict(usage) if isinstance(usage, dict) else {}
            self.last_generation["usage_status"] = (
                "measured" if self.last_usage else "unknown")
            self.last_generation["finish_reason"] = str(
                body["choices"][0].get("finish_reason") or "")
            self.last_response_identity = {
                "id": str(body.get("id") or ""),
                "model": str(body.get("model") or self.model),
                "created": body.get("created"),
                "backend_fingerprint": body.get("system_fingerprint"),
                "model_reported": bool(body.get("model")),
            }
            self.last_generation["response_identity"] = self.last_response_identity
            message = body["choices"][0]["message"]
            reasoning = message.get("reasoning_content")
            self.last_reasoning_content = reasoning if isinstance(reasoning, str) else None
            content = message.get("content")
            if not isinstance(content, str) or not content:
                failure_message += ": empty content"
                raise ValueError("empty content")
            return content
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            self._record_service_error(
                "invalid_response", "validating_body", started, "ValueError")
            raise RuntimeError(failure_message) from None
