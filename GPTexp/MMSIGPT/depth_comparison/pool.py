"""Multiple credential slots, one request per attempt, no persisted credentials."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from xcodeapi.config import Profile
from xcodeapi.transport import _NoRedirect

from ...traces import credential_free, digest
from ..prototype.client import ServiceError, response_record

BASE_URL = "https://xcode.best/v1"
MODEL = "gpt-5.6-terra"


def read_keys(path):
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError("credential_file_requires_private_permissions")
    keys = []
    for line in path.read_text().splitlines():
        line = line.strip()
        assignment = re.fullmatch(
            r"(?:export\s+)?XCODE_GPT_API_KEY\s*=\s*(.*?)\s*", line)
        commented = re.fullmatch(r"#\s*(sk-[A-Za-z0-9_-]+)\s*", line)
        key = (assignment.group(1).strip("'\"") if assignment
               else commented.group(1) if commented else None)
        if key and key not in keys:
            keys.append(key)
    if not keys or any(any(c.isspace() for c in key) for key in keys):
        raise ValueError("missing_or_invalid_gpt_credentials")
    return {f"pool-{i + 1:02d}": key for i, key in enumerate(keys)}


def inference(api_id):
    return {
        "provider": "xcode-responses", "model": MODEL, "api_id": api_id,
        "endpoint_sha256": digest(BASE_URL + "/responses"), "reasoning_effort": "high",
        "seed": 137, "temperature": 0, "max_output_tokens": 8192,
        "timeout_seconds": 36000, "response_metadata_policy": "request_context",
        "upstream_policy": {
            "require_reported_sampling_match": False,
            "require_reported_output_limit_match": False,
            "require_reported_instructions_match": False,
        },
    }


def qualify(keys):
    def check(pair):
        api_id, key = pair
        request = urllib.request.Request(
            BASE_URL + "/models", headers={"Authorization": "Bearer " + key,
                                         "Accept": "application/json"})
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=25) as response:
                data = json.loads(response.read())
                credential_free(data, key)
                return {
                    "api_id": api_id, "http_status": response.status,
                    "terra_listed": any(row.get("id") == MODEL for row in data.get("data", [])),
                }
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            return {"api_id": api_id, "http_status": status, "terra_listed": False}
        except Exception:
            return {"api_id": api_id, "reason_code": "qualification_failed", "terra_listed": False}
    with ThreadPoolExecutor(max_workers=len(keys)) as pool:
        return list(pool.map(check, keys.items()))


class Client:
    def __init__(self, api_id, key):
        self._key = key
        self.inference = inference(api_id)
        self.profile = Profile("responses", BASE_URL, MODEL, "XCODE_GPT_API_KEY",
                               timeout_seconds=self.inference["timeout_seconds"])

    def assert_public(self, value):
        credential_free(value, self._key)

    def call(self, body):
        self.assert_public(body)
        request = urllib.request.Request(
            self.profile.endpoint, data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json",
                     "Accept": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.build_opener(_NoRedirect()).open(
                    request, timeout=self.profile.timeout_seconds) as response:
                native = json.loads(response.read())
                request_id = response.headers.get("x-request-id")
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise ServiceError("http_error", status) from None
        except (OSError, urllib.error.URLError, http.client.HTTPException):
            raise ServiceError("transport_outcome_unknown") from None
        except Exception:
            raise ServiceError("invalid_provider_response") from None
        try:
            self.assert_public(native)
            self.assert_public(request_id)
            record = response_record(
                self.profile, native, self.inference,
                elapsed=time.monotonic() - started, request_id=request_id)
            reported = record["reported"]
            if reported["model"] and reported["model"] != MODEL:
                raise ServiceError("reported_model_changed")
            if reported["reasoning_effort"] and reported["reasoning_effort"] != "high":
                raise ServiceError("reported_reasoning_changed")
            self.assert_public(record)
            return record
        except ServiceError:
            raise
        except Exception:
            raise ServiceError("nonpublic_or_invalid_provider_response") from None
