"""Independent experiment client for Xcode's three wire protocols."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .config import DEFAULT_CREDENTIALS, Profile, load_profiles, resolve_api_key
from .content import convert_messages
from .errors import ConfigurationError, ProtocolError, RequestLimitError
from .transport import HTTPResult, JSONTransport


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    cached_input_tokens: int | None
    cache_creation_input_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    protocol: str
    requested_model: str
    returned_model: str | None
    text: str = field(repr=False)
    reasoning: str = field(repr=False)
    tool_calls: tuple[dict, ...] = field(repr=False)
    finish_reason: str | None
    usage: Usage
    elapsed_seconds: float
    request_id: str | None
    # Native blocks, signatures and opaque reasoning items remain available in
    # memory for continuation. Nothing is persisted or printed automatically.
    raw: dict = field(repr=False)

    def metadata(self) -> dict:
        """Numeric experiment metadata only; no response text or request body."""
        return {
            "protocol": self.protocol,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "total_tokens": self.usage.total_tokens,
            "reasoning_tokens": self.usage.reasoning_tokens,
            "cached_input_tokens": self.usage.cached_input_tokens,
            "cache_creation_input_tokens": self.usage.cache_creation_input_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "tool_call_count": len(self.tool_calls),
        }


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _mapping(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ProtocolError("Provider response has an invalid object field.")
    return value


def _blocks(value: Any) -> list[dict]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ProtocolError("Provider response has an invalid content list.")
    return value


def normalize_response(profile: Profile, reply: HTTPResult) -> GenerationResult:
    data = reply.body
    if not isinstance(data, dict) or data.get("error") is not None:
        raise ProtocolError("Provider returned an API error or invalid response.")
    usage_data = _mapping(data.get("usage", {}))
    text, reasoning, calls = [], [], []
    finish = None
    if profile.protocol == "responses":
        if data.get("status") in ("failed", "cancelled", "queued", "in_progress"):
            raise ProtocolError("Responses request did not produce a final result.")
        for item in _blocks(data.get("output")):
            if item.get("type") == "message":
                for block in _blocks(item.get("content")):
                    if block.get("type") == "output_text" and isinstance(block.get("text"), str):
                        text.append(block["text"])
                    elif block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
                        text.append(block["refusal"])
            elif item.get("type") == "reasoning":
                for block in _blocks(item.get("summary", [])):
                    if isinstance(block.get("text"), str):
                        reasoning.append(block["text"])
            elif item.get("type") == "function_call":
                calls.append(copy.deepcopy(item))
        finish = data.get("status")
        if finish == "incomplete":
            finish = _mapping(data.get("incomplete_details") or {}).get("reason", "incomplete")
    elif profile.protocol == "messages":
        for block in _blocks(data.get("content")):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text.append(block["text"])
            elif block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
                reasoning.append(block["thinking"])
            elif block.get("type") == "tool_use":
                calls.append(copy.deepcopy(block))
        finish = data.get("stop_reason")
    else:
        choices = _blocks(data.get("choices"))
        if len(choices) != 1:
            raise ProtocolError("Expected exactly one completion choice.")
        message = _mapping(choices[0].get("message"))
        content = message.get("content")
        if isinstance(content, str):
            text.append(content)
        elif isinstance(content, list):
            for block in _blocks(content):
                if isinstance(block.get("text"), str):
                    text.append(block["text"])
        elif content is not None:
            raise ProtocolError("Invalid completion message content.")
        if isinstance(message.get("refusal"), str):
            text.append(message["refusal"])
        for name in ("reasoning_content", "reasoning"):
            if isinstance(message.get(name), str):
                reasoning.append(message[name])
                break
        calls.extend(copy.deepcopy(_blocks(message.get("tool_calls", []))))
        finish = choices[0].get("finish_reason")
    if not text and not calls:
        raise ProtocolError("Provider returned no answer text, refusal or tool call.")

    input_tokens = _count(usage_data.get("input_tokens", usage_data.get("prompt_tokens")))
    output_tokens = _count(usage_data.get("output_tokens", usage_data.get("completion_tokens")))
    cached = _count(usage_data.get("cache_read_input_tokens"))
    created = _count(usage_data.get("cache_creation_input_tokens"))
    input_details = _mapping(usage_data.get("input_tokens_details", usage_data.get("prompt_tokens_details", {})) or {})
    output_details = _mapping(usage_data.get("output_tokens_details", usage_data.get("completion_tokens_details", {})) or {})
    if cached is None:
        cached = _count(input_details.get("cached_tokens"))
    # Anthropic input_tokens excludes cache-read/cache-write tokens.
    if profile.protocol == "messages" and input_tokens is not None:
        input_tokens += (cached or 0) + (created or 0)
    total = _count(usage_data.get("total_tokens"))
    if total is None and input_tokens is not None and output_tokens is not None:
        total = input_tokens + output_tokens
    returned_model = data.get("model") if isinstance(data.get("model"), str) else None
    return GenerationResult(
        profile.protocol, profile.model, returned_model, "".join(text), "\n".join(reasoning),
        tuple(calls), finish if isinstance(finish, str) else None,
        Usage(input_tokens, output_tokens, total, _count(output_details.get("reasoning_tokens")), cached, created),
        reply.elapsed_seconds, reply.request_id, copy.deepcopy(data),
    )


class XcodeClient:
    """Synchronous client; independent calls may share it across worker threads.

    Constructing, preparing, or inspecting the client makes no network requests
    and does not read the private key file. Each send/list_models uses one attempt.
    """

    def __init__(
        self, profile: Profile, *, credentials_file: str | Path = DEFAULT_CREDENTIALS,
        max_requests: int | None = None,
    ):
        if max_requests is not None and (
            isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 0
        ):
            raise ConfigurationError("max_requests must be a nonnegative integer or None.")
        self.profile = profile
        self.credentials_file = Path(credentials_file)
        self.max_requests = max_requests
        self._attempts = 0
        self._lock = Lock()

    @classmethod
    def from_profile(
        cls, name: str, *, config_file: str | Path | None = None, model: str | None = None,
        credentials_file: str | Path = DEFAULT_CREDENTIALS, max_requests: int | None = None,
    ) -> XcodeClient:
        profiles = load_profiles(config_file)
        if name not in profiles:
            raise ConfigurationError("Unknown profile name.")
        return cls(profiles[name].with_model(model), credentials_file=credentials_file, max_requests=max_requests)

    @property
    def attempts(self) -> int:
        with self._lock:
            return self._attempts

    def prepare(
        self, messages: list[dict], *, max_output_tokens: int = 1024,
        temperature: float | None = None, extra_body: dict | None = None,
    ) -> dict:
        """Build a native request body offline, without loading credentials."""
        self.profile.require_model()
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise ProtocolError("max_output_tokens must be a positive integer.")
        if temperature is not None and (
            isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature) or not 0 <= temperature <= 2
        ):
            raise ProtocolError("temperature must be finite and between 0 and 2.")
        if self.profile.protocol == "messages" and temperature is not None and temperature > 1:
            raise ProtocolError("Anthropic temperature must be between 0 and 1.")
        converted, system = convert_messages(messages, self.profile.protocol)
        payload = {"model": self.profile.model, "stream": False}
        if self.profile.protocol == "responses":
            payload.update(input=converted, max_output_tokens=max_output_tokens, store=False)
            if system:
                payload["instructions"] = system
        elif self.profile.protocol == "messages":
            payload.update(messages=converted, max_tokens=max_output_tokens)
            if system:
                payload["system"] = system
        else:
            if system:
                converted.insert(0, {"role": "system", "content": system})
            payload.update(messages=converted, max_tokens=max_output_tokens)
        if temperature is not None:
            payload["temperature"] = temperature
        if extra_body is not None:
            reserved = {"model", "stream", "input", "messages", "instructions", "system",
                        "max_tokens", "max_output_tokens", "max_completion_tokens", "temperature"}
            if not isinstance(extra_body, dict) or reserved.intersection(extra_body):
                raise ProtocolError("extra_body cannot replace model, conversation or core generation settings.")
            payload.update(copy.deepcopy(extra_body))
        self._validate_payload(payload)
        return payload

    def _validate_payload(self, payload: dict):
        self.profile.require_model()
        if not isinstance(payload, dict) or payload.get("model") != self.profile.model:
            raise ProtocolError("Native payload model must match the configured experiment model.")
        if payload.get("stream", False) is not False:
            raise ProtocolError("Streaming is not supported by this JSON client.")
        if self.profile.protocol == "responses" and payload.get("background", False) is not False:
            raise ProtocolError("Background Responses requests are not supported.")
        if self.profile.protocol == "chat_completions" and payload.get("n", 1) != 1:
            raise ProtocolError("Use one completion per experiment request.")
        if self.profile.protocol == "responses":
            content = payload.get("input")
            if not (
                (isinstance(content, str) and content)
                or (isinstance(content, list) and content)
            ):
                raise ProtocolError("Responses payload requires nonempty input text or items.")
        else:
            if not isinstance(payload.get("messages"), list) or not payload["messages"]:
                raise ProtocolError("Native payload requires a nonempty messages list.")
            if self.profile.protocol == "messages":
                tokens = payload.get("max_tokens")
                if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                    raise ProtocolError("Anthropic native payload requires positive max_tokens.")
        try:
            json.dumps(payload, allow_nan=False)
        except (ValueError, TypeError):
            raise ProtocolError("Request must be valid JSON.") from None

    def _request(self, method: str, endpoint: str, payload: dict | None = None) -> HTTPResult:
        key = resolve_api_key(self.profile, self.credentials_file)
        transport = JSONTransport(key, self.profile.auth, self.profile.timeout_seconds)
        with self._lock:
            if self.max_requests is not None and self._attempts >= self.max_requests:
                raise RequestLimitError("Client request-attempt limit reached.")
            self._attempts += 1
        return transport.request(method, endpoint, payload)

    def send(self, payload: dict) -> GenerationResult:
        """Send an exact native payload, including native tools/continuations."""
        snapshot = copy.deepcopy(payload)
        self._validate_payload(snapshot)
        return normalize_response(self.profile, self._request("POST", self.profile.endpoint, snapshot))

    def generate(self, messages: list[dict], **options) -> GenerationResult:
        return self.send(self.prepare(messages, **options))

    def list_models(self) -> list[dict]:
        """Authenticated GET /v1/models. Some token groups may not expose it."""
        reply = self._request("GET", self.profile.api_base + "/models")
        models = _blocks(reply.body.get("data"))
        if any(not isinstance(item.get("id"), str) for item in models):
            raise ProtocolError("Invalid model-list entry.")
        return copy.deepcopy(models)
