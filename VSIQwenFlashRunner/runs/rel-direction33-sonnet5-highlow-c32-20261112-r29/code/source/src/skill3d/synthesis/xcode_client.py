"""Native Responses adapter; credentials are loaded only at the send boundary."""
from __future__ import annotations

import hashlib
import os
from xcodeapi.client import XcodeClient, normalize_response
from xcodeapi.config import Profile
from xcodeapi.transport import JSONTransport, ResponsesStreamTransport, MessagesStreamTransport
from xcodeapi.errors import XcodeAPIError
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.client_errors import ServiceUnavailable
from skill3d.synthesis.request_audit import describe_request


class XcodeResponsesClient(BailianClient):
    provider = "xcodeapi"
    protocol = "responses"
    stream_transport = ResponsesStreamTransport

    def __init__(self, *, reasoning_effort="", **kwargs):
        super().__init__(**kwargs)
        if reasoning_effort not in {"", "low", "medium", "high"}:
            raise ValueError("unsupported native reasoning effort")
        self.reasoning_effort = reasoning_effort
        self.profile = Profile(self.protocol, self.base_url, self.model,
                               self.api_key_env, timeout_seconds=self.timeout_s)
        self.endpoint_hash = hashlib.sha256(self.profile.endpoint.encode()).hexdigest()
        self._native_builder = XcodeClient(self.profile)
        self.returned_effort = None

    def audit_identity(self, *, seed=None):
        # Responses has no seed. Framework sampling seed is recorded separately.
        return {**super().audit_identity(seed=None), "protocol": self.protocol}

    def build_payload(self, messages, max_tokens=2048, seed=None, temperature=None,
                      response_format=None, enable_thinking=False, top_p=None):
        if response_format is not None or top_p is not None:
            raise ValueError("unsupported Responses sampling parameter")
        effort = self.reasoning_effort or ("high" if enable_thinking else "low")
        return self._native_builder.prepare(
            messages, max_output_tokens=min(int(max_tokens), self.max_output_tokens),
            temperature=self.temperature if temperature is None else temperature,
            extra_body={"reasoning": {"effort": effort}}, stream=effort == "high")

    def validate_native(self, payload):
        self._native_builder._validate_payload(payload)
        required = {"model", "input", "temperature", "max_output_tokens", "reasoning",
                    "stream", "store"}
        if (not required <= set(payload) or set(payload) - required - {"instructions"}
                or payload["stream"] is not (payload["reasoning"].get("effort") == "high")
                or payload["store"] is not False
                or payload["temperature"] != self.temperature
                or type(payload["max_output_tokens"]) is not int
                or not 0 < payload["max_output_tokens"] <= self.max_output_tokens
                or payload["reasoning"] not in [
                    {"effort": value} for value in ("high", "low", "medium")]):
            raise ValueError("native payload differs from frozen settings")

    def chat(self, messages, **kwargs):
        return self.send_native(self.build_payload(messages, **kwargs))

    def payload_effort(self, payload):
        return payload["reasoning"]["effort"]

    def send_native(self, payload, *, api_key=None):
        self.validate_native(payload)
        self.last_usage = {}
        self.last_response_identity = {}
        self.last_request_identity = describe_request(payload, model_identity=self.audit_identity())
        self.last_generation = {
            "mode": "stream" if payload["stream"] else "nonstream",
            "usage_status": "unknown", "provider": self.provider,
            "endpoint_sha256": self.endpoint_hash, "request_identity": self.last_request_identity}
        token = api_key or os.environ.get(self.api_key_env, "").strip()
        if not token:
            raise ServiceUnavailable("Xcode credential unavailable")
        try:
            transport_type = self.stream_transport if payload["stream"] else JSONTransport
            reply = transport_type(token, "bearer", self.timeout_s).request(
                "POST", self.profile.endpoint, payload)
        except XcodeAPIError as error:
            self.last_generation["transport_error"] = {
                "kind": type(error).__name__,
                "http_status": getattr(error, "status", None),
            }
            raise ServiceUnavailable("Xcode HTTPS request unsuccessful") from None
        usage = reply.body.get("usage") or {}
        # Preserve known paid usage even when answer parsing subsequently fails.
        mapping = {"prompt_tokens": "input_tokens", "completion_tokens": "output_tokens",
                   "total_tokens": "total_tokens"}
        self.last_usage = {out: usage[name] for out, name in mapping.items()
                           if type(usage.get(name)) is int and usage[name] >= 0}
        for name, section, field in (
            ("reasoning_tokens", "output_tokens_details", "reasoning_tokens"),
            ("cached_input_tokens", "input_tokens_details", "cached_tokens")):
            value = (usage.get(section) or {}).get(field)
            if type(value) is int and value >= 0:
                self.last_usage[name] = value
        if self.protocol == "messages":
            for source, name in (("cache_read_input_tokens", "cached_input_tokens"),
                                 ("cache_creation_input_tokens", "cache_creation_input_tokens")):
                if type(usage.get(source)) is int and usage[source] >= 0:
                    self.last_usage[name] = usage[source]
            if "prompt_tokens" in self.last_usage:
                self.last_usage["prompt_tokens"] += sum(
                    self.last_usage.get(name, 0) for name in (
                        "cached_input_tokens", "cache_creation_input_tokens"))
            if "prompt_tokens" in self.last_usage and "completion_tokens" in self.last_usage:
                self.last_usage["total_tokens"] = (
                    self.last_usage["prompt_tokens"] + self.last_usage["completion_tokens"])
        self.returned_effort = (reply.body.get("reasoning") or {}).get("effort")
        self.last_response_identity = {
            "id": str(reply.body.get("id") or ""), "model": str(reply.body.get("model") or ""),
            "created": reply.body.get("created_at"), "backend_fingerprint": None,
            "model_reported": isinstance(reply.body.get("model"), str)}
        result = normalize_response(self.profile, reply)
        if result.returned_model != self.model or (
                self.protocol == "responses" and self.returned_effort != self.payload_effort(payload)):
            raise ValueError("native returned model or effort mismatch")
        self.last_reasoning_content = result.reasoning or None
        self.last_generation.update(
            usage_status="measured" if self.last_usage else "unknown",
            finish_reason=result.finish_reason or "unknown",
            response_identity=self.last_response_identity)
        return result.text


class XcodeMessagesClient(XcodeResponsesClient):
    protocol = "messages"
    stream_transport = MessagesStreamTransport

    def build_payload(self, messages, max_tokens=2048, seed=None, temperature=None,
                      response_format=None, enable_thinking=False, top_p=None):
        if response_format is not None or top_p is not None:
            raise ValueError("unsupported Messages sampling parameter")
        effort = self.reasoning_effort or ("high" if enable_thinking else "low")
        return self._native_builder.prepare(
            messages, max_output_tokens=min(int(max_tokens), self.max_output_tokens),
            temperature=self.temperature if temperature is None else temperature,
            extra_body={"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}},
            stream=True)

    def payload_effort(self, payload):
        return payload["output_config"]["effort"]

    def validate_native(self, payload):
        self._native_builder._validate_payload(payload)
        required = {"model", "messages", "temperature", "max_tokens", "thinking",
                    "output_config", "stream"}
        if (not required <= set(payload) or set(payload) - required - {"system"}
                or payload["stream"] is not True or payload["temperature"] != self.temperature
                or type(payload["max_tokens"]) is not int
                or not 0 < payload["max_tokens"] <= self.max_output_tokens
                or payload["thinking"] != {"type": "adaptive"}
                or payload["output_config"] not in [
                    {"effort": value} for value in ("high", "low", "medium")]):
            raise ValueError("native payload differs from frozen settings")


def get_xcode_client_class(model, *, remote=False):
    if remote:
        from skill3d.cluster.xcode_client import RemoteXcodeClient, RemoteXcodeMessagesClient
        return RemoteXcodeMessagesClient if model.startswith("claude-") else RemoteXcodeClient
    return XcodeMessagesClient if model.startswith("claude-") else XcodeResponsesClient
