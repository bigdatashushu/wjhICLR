"""One Responses HTTP attempt; credentials stay in memory, errors use fixed codes."""
from __future__ import annotations

import http.client
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

from xcodeapi.client import normalize_response
from xcodeapi.config import Profile
from xcodeapi.transport import HTTPResult, _NoRedirect

from ...api_pool import resolve_key
from ...response_policy import CORRELATION, policy
from ...traces import credential_free, digest


class ServiceError(RuntimeError):
    def __init__(self, code, http_status=None):
        self.code = code
        self.http_status = http_status
        super().__init__(code)


def configuration(path, api_id, *, max_output_tokens=8192):
    """Read endpoint configuration, not a credential file; return public identity."""
    raw = Path(path).read_bytes()
    config = json.loads(raw)
    apis = [api for api in config["apis"] if api["id"] == api_id]
    if len(apis) != 1:
        raise ValueError("select_exactly_one_api")
    api = apis[0]
    if (config["model"] != "gpt-5.6-terra" or api["model"] != config["model"]
            or api["protocol"] != "responses" or not api.get("enabled", True)
            or not api.get("validated", False)):
        raise ValueError("requires_validated_terra_responses_api")
    if any(name in api for name in ("api_key", "key", "token", "password", "authorization")):
        raise ValueError("inline_credentials_not_supported")
    profile = Profile("responses", api["base_url"], api["model"], "XCODE_GPT_API_KEY",
                      timeout_seconds=api["timeout_seconds"])
    effort = config.get("reasoning_effort", "high")
    seed = config.get("seed", 137)
    temperature = config.get("temperature", 0)
    if (effort != "high" or type(seed) is not int or seed != 137
            or isinstance(temperature, bool) or temperature != 0
            or type(max_output_tokens) is not int or max_output_tokens < 1):
        raise ValueError("direct_project_inference_parameters_changed")
    metadata_policy = policy(config)
    inference = {
        "provider": "xcode-responses", "model": api["model"], "api_id": api_id,
        "endpoint_sha256": digest(profile.endpoint),
        "reasoning_effort": effort, "seed": seed, "temperature": temperature,
        "max_output_tokens": max_output_tokens, "timeout_seconds": profile.timeout_seconds,
        "response_metadata_policy": metadata_policy,
        "upstream_policy": {
            field: bool(config.get("upstream_policy", {}).get(field, False))
            for field in ("require_reported_sampling_match", "require_reported_output_limit_match",
                          "require_reported_instructions_match")
        },
    }
    return api, profile, inference, hashlib.sha256(raw).hexdigest()


def response_record(profile, native_body, inference, *, elapsed, request_id=None):
    """Normalize a returned result; raw provider bodies are never persisted."""
    try:
        result = normalize_response(profile, HTTPResult(native_body, elapsed, request_id))
    except Exception:
        raise ServiceError("invalid_provider_response") from None
    if result.tool_calls:
        raise ServiceError("unexpected_tool_call")
    reported = {
        "model": result.returned_model,
        "reasoning_effort": (native_body.get("reasoning") or {}).get("effort"),
        "temperature": native_body.get("temperature"),
        "seed": native_body.get("seed"),
        "max_output_tokens": native_body.get("max_output_tokens"),
    }
    expected = {
        "model": inference["model"], "reasoning_effort": inference["reasoning_effort"],
        "temperature": inference["temperature"], "seed": inference["seed"],
        "max_output_tokens": inference["max_output_tokens"],
    }
    match = {key: reported[key] == value for key, value in expected.items()}
    if inference["response_metadata_policy"] != CORRELATION:
        if not match["model"] or not match["reasoning_effort"]:
            raise ServiceError("reported_model_or_reasoning_changed")
        upstream = inference["upstream_policy"]
        if upstream["require_reported_sampling_match"] and (
                "temperature" in native_body and not match["temperature"]
                or "seed" in native_body and reported["seed"] not in (None, inference["seed"])):
            raise ServiceError("reported_sampling_changed")
        if (upstream["require_reported_output_limit_match"] and "max_output_tokens" in native_body
                and not match["max_output_tokens"]):
            raise ServiceError("reported_output_limit_changed")
        if (upstream["require_reported_instructions_match"]
                and native_body.get("instructions") not in ("", None)):
            raise ServiceError("reported_instructions_changed")
    # Keep only scalar metadata; provider strings are checked against credentials by caller.
    for key, value in reported.items():
        if value is not None and (not isinstance(value, (str, int, float))
                                  or isinstance(value, bool)
                                  or isinstance(value, float) and not math.isfinite(value)):
            raise ServiceError("invalid_reported_metadata")
    return {
        "text": result.text, "reasoning_summary": result.reasoning,
        "finish_reason": result.finish_reason, "usage": result.metadata(),
        "reported": reported, "reported_matches_requested": match,
        "response_metadata_policy": inference["response_metadata_policy"],
        "native_response_sha256": digest(native_body), "elapsed_seconds": elapsed,
        "provider_request_id_sha256": digest(request_id) if request_id else None,
        "tool_call_count": 0,
    }


class ResponsesClient:
    def __init__(self, api, profile, inference):
        self.api, self.profile, self.inference = api, profile, inference
        # Explicit execution is the only path that resolves credentials.
        self._key = resolve_key(api)

    def assert_public(self, value):
        credential_free(value, self._key)

    def call(self, body):
        self.assert_public(body)
        request = urllib.request.Request(
            self.profile.endpoint, data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self._key, "Accept": "application/json",
                     "Content-Type": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.build_opener(_NoRedirect()).open(
                    request, timeout=self.profile.timeout_seconds) as response:
                raw = response.read()
                request_id = response.headers.get("x-request-id")
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            raise ServiceError("http_error", code) from None
        except (OSError, urllib.error.URLError, http.client.HTTPException):
            raise ServiceError("transport_outcome_unknown") from None
        try:
            native = json.loads(raw)
            self.assert_public(native)
            self.assert_public(request_id)
            record = response_record(self.profile, native, self.inference,
                                     elapsed=time.monotonic() - started, request_id=request_id)
            self.assert_public(record)
            return record
        except ServiceError:
            raise
        except Exception:
            raise ServiceError("invalid_or_nonpublic_response") from None
