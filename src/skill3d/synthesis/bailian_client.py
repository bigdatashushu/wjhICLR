"""Minimal OpenAI-compatible client for Alibaba Bailian Qwen models."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional
import urllib.error
import urllib.request

from .vllm_client import ServiceUnavailable

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"


def _base_url(value: str) -> str:
    base = str(value or DEFAULT_BASE_URL).strip().rstrip("/")
    if not base.endswith("/v1"):
        raise ValueError("Bailian base_url must end with /v1")
    if not base.startswith("https://"):
        raise ValueError("Bailian base_url must use HTTPS")
    return base


def _scrub(text: str, secret: str) -> str:
    result = str(text)
    if secret:
        result = result.replace(secret, "<redacted>")
    return result


class BailianClient:
    """Qwen client with the same small ``chat`` contract used by M5.

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
        self._opener = opener or urllib.request.urlopen
        self.last_usage: dict = {}
        self.last_generation: dict = {}
        self.last_response_identity: dict = {}

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

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        response_format: Optional[dict] = None,
        enable_thinking: bool = False,
    ) -> str:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise ServiceUnavailable(f"{self.api_key_env} is not configured")
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
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        self.last_usage = {}
        self.last_generation = {
            "mode": "nonstream", "usage_status": "unknown",
            "provider": self.provider, "endpoint_sha256": self.endpoint_hash,
        }
        self.last_response_identity = {}
        try:
            with self._opener(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _scrub(exc.read().decode("utf-8", errors="replace"), api_key)
            message = f"Bailian HTTP {exc.code}: {detail[:1200]}"
            if exc.code in {401, 403, 404, 408, 409, 429} or exc.code >= 500:
                raise ServiceUnavailable(message) from exc
            raise RuntimeError(message) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ServiceUnavailable(
                _scrub(f"{type(exc).__name__}: {exc}", api_key)) from exc
        try:
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content:
                raise ValueError("empty content")
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
            }
            return content
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid Bailian response: {exc}") from exc
