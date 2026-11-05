"""Bailian-compatible public interface backed by a single Responses attempt."""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from xcodeapi.client import normalize_response
from xcodeapi.config import Profile
from xcodeapi.content import convert_messages
from xcodeapi.transport import HTTPResult, _NoRedirect

from .api_pool import resolve_key
from .traces import credential_free, digest, write


class APIError(RuntimeError):
    def __init__(self, code, http_status=None):
        super().__init__(code)
        self.code = code
        self.http_status = http_status


class ResponsesClient:
    provider = "xcode-responses"

    def __init__(self, api, pool, directory, operation, *, reasoning_effort="high",
                 max_output_tokens=8192):
        self.api = api
        self.pool = pool
        self.directory = Path(directory)
        self.operation = operation
        self.model = api["model"]
        self.base_url = api["base_url"].rstrip("/")
        self.timeout_s = api["timeout_seconds"]
        self.temperature = 0
        self.max_output_tokens = max_output_tokens
        self.model_revision = ""
        self.endpoint_hash = digest(self.base_url)
        self.reasoning_effort = reasoning_effort
        self.index = 0
        self.last_usage = {}
        self.last_generation = {}
        self.last_response_identity = {}
        self.last_request_identity = {}
        self.last_reasoning_content = None
        self.last_service_error = {}
        self.profile = Profile("responses", self.base_url, self.model,
                               "XCODE_GPT_API_KEY", timeout_seconds=self.timeout_s)

    @property
    def configured(self):
        return True

    def check_service(self):
        return True, "configured"

    def audit_identity(self, *, seed=None):
        return {"enabled": True, "provider": self.provider, "model": self.model,
                "model_revision": "", "endpoint_sha256": self.endpoint_hash, "api_id": self.api["id"],
                "seed": seed, "temperature": self.temperature, "max_tokens": self.max_output_tokens,
                "reasoning_effort": self.reasoning_effort, "cache_eligible": False}

    def build_payload(self, messages, max_tokens=2048, seed=None, temperature=None,
                      response_format=None, enable_thinking=False, top_p=None):
        converted, system = convert_messages(messages, "responses")
        payload = {"model": self.model, "input": converted, "stream": False, "store": False,
                   "max_output_tokens": min(int(max_tokens), self.max_output_tokens),
                   "temperature": self.temperature if temperature is None else float(temperature),
                   "reasoning": {"effort": self.reasoning_effort}}
        if system:
            payload["instructions"] = system
        if seed is not None:
            payload["seed"] = int(seed)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if response_format is not None:
            payload["text"] = {"format": dict(response_format)}
        return payload

    def chat(self, messages, **kwargs):
        payload = self.build_payload(messages, **kwargs)
        self.index += 1
        prefix = self.directory / f"{self.index:03d}"
        self.last_usage = {}
        self.last_reasoning_content = None
        self.last_response_identity = {}
        self.last_generation = {}
        self.last_service_error = {}
        self.last_request_identity = {"payload_sha256": digest(payload),
                                      "model_identity": self.audit_identity(seed=kwargs.get("seed"))}
        # Resolve credentials before writing the intent; they remain host-local.
        key = getattr(self, "_key", None) or resolve_key(self.api)
        credential_free(payload, key)
        write(str(prefix) + ".intent.json", {
            "schema": "gptexp-request/1", "api_id": self.api["id"], "operation": self.operation,
            "ordinal": self.index, "messages": messages, "original_parameters": kwargs,
            "native_payload": payload, "payload_sha256": digest(payload), "at": time.time(),
            "http_timeout_seconds": self.timeout_s, "automatic_retry": False,
            "reference_answer_visible": False}, exclusive=True)
        started = time.monotonic()
        raw_body = None
        admission = None
        try:
            with self.pool.request(self.api, self.operation, self.index, digest(payload)) as admission:
                write(str(prefix) + ".admission.json", {**admission, "at": time.time()}, exclusive=True)
                request = urllib.request.Request(
                    self.profile.endpoint,
                    data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                    headers={"Authorization": "Bearer " + key, "Accept": "application/json",
                             "Content-Type": "application/json"}, method="POST")
                opener = urllib.request.build_opener(_NoRedirect())
                sent = time.monotonic()
                try:
                    with opener.open(request, timeout=self.timeout_s) as response:
                        received_body = json.loads(response.read())
                        credential_free(received_body, key)
                        raw_body = received_body
                        request_id = response.headers.get("x-request-id")
                except urllib.error.HTTPError as exc:
                    admission["http_status"] = exc.code
                    # Retain the structured public API error if JSON; no headers.
                    try:
                        error_body = json.loads(exc.read())
                        credential_free(error_body, key)
                        raw_body = error_body
                    except (ValueError, OSError, http.client.HTTPException):
                        raw_body = None
                    finally:
                        exc.close()
                    raise APIError("http_error", exc.code) from None
                except (OSError, urllib.error.URLError, http.client.HTTPException):
                    raise APIError("transport_outcome_unknown") from None
                except (ValueError, UnicodeError):
                    raise APIError("invalid_response_json") from None
                credential_free(raw_body, key)
                result = normalize_response(
                    self.profile, HTTPResult(raw_body, time.monotonic() - sent, request_id))
                if result.returned_model != self.model:
                    raise APIError("actual_model_changed")
                if (raw_body.get("reasoning") or {}).get("effort") != self.reasoning_effort:
                    raise APIError("reported_reasoning_effort_changed")
                policy = self.pool.config.get("upstream_policy", {})
                if policy.get("require_reported_sampling_match") and (
                        "temperature" in raw_body and raw_body["temperature"] != payload["temperature"]
                        or "seed" in raw_body and raw_body["seed"] not in (None, payload.get("seed"))):
                    raise APIError("reported_sampling_changed")
                if policy.get("require_reported_output_limit_match") and (
                        "max_output_tokens" in raw_body
                        and raw_body["max_output_tokens"] != payload["max_output_tokens"]):
                    raise APIError("reported_output_limit_changed")
                if policy.get("require_reported_instructions_match") and (
                        raw_body.get("instructions", "") not in ("", None, payload.get("instructions", ""))):
                    raise APIError("reported_instructions_changed")
                self.last_usage = {
                    "prompt_tokens": result.usage.input_tokens,
                    "completion_tokens": result.usage.output_tokens,
                    "total_tokens": result.usage.total_tokens}
                self.last_usage = {k: v for k, v in self.last_usage.items() if v is not None}
                self.last_response_identity = {"model": result.returned_model, "model_reported": True}
                self.last_reasoning_content = result.reasoning or None
                finish_reason = {"completed": "stop", "max_output_tokens": "length"}.get(
                    result.finish_reason, result.finish_reason)
                self.last_generation = {"status": result.finish_reason, "finish_reason": finish_reason,
                                        "usage": self.last_usage, "reasoning_tokens": result.usage.reasoning_tokens}
                record = {"status": "received", "native_response": raw_body, "text": result.text,
                          "reasoning": result.reasoning, "reasoning_returned": bool(result.reasoning),
                          "usage": self.last_usage, "generation": self.last_generation,
                          "response_identity": self.last_response_identity,
                          "elapsed_seconds": time.monotonic() - started, "at": time.time()}
                write(str(prefix) + ".response.json", record, exclusive=True)
                return result.text
        except BaseException as exc:
            self.last_service_error = {"reason_code": getattr(exc, "code", "service_error"),
                                       "exception_type": type(exc).__name__,
                                       "http_status": getattr(exc, "http_status", None)}
            write(str(prefix) + ".response.json", {
                "status": "error", **self.last_service_error, "native_response": raw_body,
                "admitted": admission is not None, "elapsed_seconds": time.monotonic() - started,
                "automatic_retry": False, "reference_answer_visible": False}, exclusive=True)
            raise
