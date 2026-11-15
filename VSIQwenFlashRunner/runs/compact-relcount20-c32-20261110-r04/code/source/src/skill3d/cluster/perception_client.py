"""Authenticated transport for the unchanged Bailian logical client contract.

Workers need only HARNESS3D_PERCEPTION_ENDPOINT and HARNESS3D_PERCEPTION_TOKEN.
The endpoint is a credential-free gateway origin, not the upstream model URL.
One chat freezes its payload and UUID before any transport attempt; retries
never change either. A failed/indeterminate upstream call is not retried here.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import time
from typing import Optional
import urllib.error
import urllib.request
from urllib.parse import urlsplit
import uuid

from skill3d.synthesis.bailian_client import (
    BailianClient, DEFAULT_API_KEY_ENV, DEFAULT_BASE_URL, DEFAULT_MODEL,
)
from skill3d.synthesis.client_errors import ServiceUnavailable
from skill3d.synthesis.request_audit import describe_request, payload_bytes

RPC_SCHEMA = "harness3d.perception-rpc/1.0"
ENDPOINT_ENV = "HARNESS3D_PERCEPTION_ENDPOINT"
TOKEN_ENV = "HARNESS3D_PERCEPTION_TOKEN"
MAX_BODY_BYTES = 64 * 1024 * 1024


def bearer_token() -> str:
    value = os.environ.get(TOKEN_ENV, "")
    if not value or len(value) > 4096 or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", value):
        raise ValueError(f"{TOKEN_ENV} must contain a nonempty bearer token")
    return value


def gateway_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or "?" in value or "#" in value
                or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
                or parsed.port is not None and not 1 <= parsed.port <= 65535):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("invalid credential-free perception gateway origin") from None
    return value.rstrip("/")


def json_object(raw: bytes) -> dict:
    """Reject duplicate fields and nonfinite numbers, including overflow to inf."""
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite JSON number")
        return result

    def constant(_value):
        raise ValueError("nonfinite JSON number")

    result = json.loads(raw, object_pairs_hook=pairs, parse_float=number,
                        parse_constant=constant)
    if not isinstance(result, dict):
        raise ValueError("JSON object required")
    return result


def validate_result(result: dict, *, request_id: str, payload: dict,
                    audit_identity: dict) -> None:
    """Validate actual upstream evidence without inventing response identities."""
    expected_request = describe_request(payload, model_identity=audit_identity)
    required = {
        "schema_version", "request_id", "payload_sha256", "audit_identity", "content",
        "last_usage", "last_generation", "last_response_identity", "last_request_identity",
    }
    if not isinstance(result, dict) or set(result) != required:
        raise ValueError("invalid perception result fields")
    if (result["schema_version"] != RPC_SCHEMA or result["request_id"] != request_id
            or result["payload_sha256"] != hashlib.sha256(payload_bytes(payload)).hexdigest()
            or result["audit_identity"] != audit_identity
            or result["last_request_identity"] != expected_request
            or not isinstance(result["content"], str) or not result["content"]):
        raise ValueError("perception request evidence mismatch")
    usage = result["last_usage"]
    generation = result["last_generation"]
    response = result["last_response_identity"]
    if not all(isinstance(value, dict) for value in (usage, generation, response)):
        raise ValueError("invalid perception evidence types")
    if (set(generation) != {
            "mode", "usage_status", "provider", "endpoint_sha256", "request_identity",
            "finish_reason", "response_identity",
        }
            or generation["mode"] != "nonstream"
            or generation["usage_status"] != ("measured" if usage else "unknown")
            or generation["provider"] != audit_identity["provider"]
            or generation["endpoint_sha256"] != audit_identity["endpoint_sha256"]
            or generation["request_identity"] != expected_request
            or generation["response_identity"] != response
            or not isinstance(generation["finish_reason"], str)
            or set(response) != {"id", "model", "created", "backend_fingerprint"}
            or not isinstance(response["id"], str) or not isinstance(response["model"], str)):
        raise ValueError("perception generation evidence mismatch")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class RemoteBailianClient(BailianClient):
    """BailianClient options plus endpoint=None and transport_retries=2.

    ``endpoint=None`` resolves HARNESS3D_PERCEPTION_ENDPOINT at construction.
    ``configured`` depends on the RPC token, never DASHSCOPE_API_KEY. The inherited
    endpoint_hash/audit_identity always refer to the original Bailian HTTPS URL.
    ``opener`` retains the request/timeout callable contract for test injection.
    """

    def __init__(
        self, *, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_API_KEY_ENV, timeout_s: float = 180.0,
        temperature: float = 0.0, max_output_tokens: int = 2048,
        model_revision: str = "", opener=None, endpoint: Optional[str] = None,
        transport_retries: int = 2,
    ) -> None:
        super().__init__(
            model=model, base_url=base_url, api_key_env=api_key_env, timeout_s=timeout_s,
            temperature=temperature, max_output_tokens=max_output_tokens,
            model_revision=model_revision, opener=opener,
        )
        value = os.environ.get(ENDPOINT_ENV, "") if endpoint is None else endpoint
        self.gateway_endpoint = gateway_endpoint(value) if value else ""
        if type(transport_retries) is not int or not 0 <= transport_retries <= 8:
            raise ValueError("transport_retries must be between 0 and 8")
        if not math.isfinite(self.timeout_s):
            raise ValueError("timeout_s must be finite")
        self.transport_retries = transport_retries
        # Do not forward bearer credentials through redirects or ambient proxies.
        self._rpc_opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()).open

    @property
    def configured(self) -> bool:
        try:
            bearer_token()
        except ValueError:
            return False
        return bool(self.gateway_endpoint)

    def check_service(self) -> tuple[bool, str]:
        if not self.configured:
            return False, f"{ENDPOINT_ENV} and {TOKEN_ENV} are required"
        return True, (
            f"configured provider={self.provider} model={self.model} "
            f"endpoint_hash={self.endpoint_hash} transport=perception-rpc")

    def _post(self, raw: bytes, token: str) -> tuple[int, bytes]:
        request = urllib.request.Request(
            self.gateway_endpoint + "/chat", data=raw, method="POST",
            headers={"Authorization": "Bearer " + token,
                     "Content-Type": "application/json", "Connection": "close"},
        )
        try:
            response = self._rpc_opener(request, timeout=self.timeout_s)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            body = response.read(MAX_BODY_BYTES + 1)
            if len(body) > MAX_BODY_BYTES:
                raise RuntimeError("perception response exceeds body limit")
            length = response.headers.get("Content-Length")
            if length is not None and (not re.fullmatch(r"[0-9]{1,10}", length)
                                       or int(length) != len(body)):
                raise http.client.IncompleteRead(b"")
            return response.code, body

    def chat(
        self, messages: list[dict], max_tokens: int = 2048,
        seed: Optional[int] = None, temperature: Optional[float] = None,
        response_format: Optional[dict] = None, enable_thinking: bool = False,
        top_p: Optional[float] = None,
    ) -> str:
        # The JSON roundtrip severs caller aliases while preserving Bailian wire order.
        payload = json_object(payload_bytes(self.build_payload(
            messages, max_tokens=max_tokens, seed=seed, temperature=temperature,
            response_format=response_format, enable_thinking=enable_thinking, top_p=top_p,
        )))
        identity = self.audit_identity(seed=payload.get("seed"))
        request_id = str(uuid.uuid4())
        raw = payload_bytes({
            "schema_version": RPC_SCHEMA, "request_id": request_id,
            "payload": payload, "audit_identity": identity,
        })
        self.last_request_identity = describe_request(payload, model_identity=identity)
        self.last_usage = {}
        self.last_response_identity = {}
        self.last_generation = {
            "mode": "nonstream", "usage_status": "unknown", "provider": self.provider,
            "endpoint_sha256": self.endpoint_hash,
            "request_identity": self.last_request_identity,
        }
        if len(raw) > MAX_BODY_BYTES:
            raise ValueError("perception request exceeds body limit")
        if not self.configured:
            raise ServiceUnavailable(f"{ENDPOINT_ENV} and {TOKEN_ENV} are required")
        token = bearer_token()
        for attempt in range(self.transport_retries + 1):
            try:
                status, body = self._post(raw, token)
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                if attempt == self.transport_retries:
                    raise ServiceUnavailable("perception transport unavailable") from None
            else:
                try:
                    result = json_object(body)
                    if any(secret and secret in body.decode("utf-8")
                           for secret in (token, os.environ.get(self.api_key_env, "").strip())):
                        raise ValueError("unsafe response")
                    if status == 200:
                        validate_result(result, request_id=request_id, payload=payload,
                                        audit_identity=identity)
                        self.last_usage = result["last_usage"]
                        self.last_generation = result["last_generation"]
                        self.last_response_identity = result["last_response_identity"]
                        self.last_request_identity = result["last_request_identity"]
                        return result["content"]
                except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
                    raise RuntimeError("invalid perception response evidence") from None
                retryable = (
                    status in {429, 503} and result.get("retryable") is True
                    and result.get("error") in {"capacity", "pending"}
                )
                if not retryable or attempt == self.transport_retries:
                    # Never include an untrusted body, status reason, URL or exception text.
                    if status in {401, 403, 408, 409, 429} or status >= 500:
                        raise ServiceUnavailable(f"perception RPC HTTP {status}") from None
                    raise RuntimeError(f"perception RPC HTTP {status}") from None
            time.sleep(min(0.1 * (2 ** attempt), 1.0))
        raise ServiceUnavailable("perception transport unavailable")
