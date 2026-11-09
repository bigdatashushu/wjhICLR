"""Durable identical-payload retries with shared credential-slot admission."""
from __future__ import annotations

from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import fcntl
import http.client
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from .common import BASE_URL, MODEL, assert_public, digest, read, write
from .vendor.xcodeapi.client import normalize_response
from .vendor.xcodeapi.config import Profile
from .vendor.xcodeapi.content import convert_messages
from .vendor.xcodeapi.transport import HTTPResult, _NoRedirect


class APIError(RuntimeError):
    def __init__(self, code, status=None, retry_after=0):
        super().__init__(code)
        self.code, self.status, self.retry_after = code, status, retry_after


def backoff(attempt, config, retry_after=0):
    return max(float(retry_after), min(config["backoff_max_seconds"],
                                      config["backoff_min_seconds"] * 2 ** (attempt - 1)))


def retry_after(value):
    try:
        return max(0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return 0


class Slots:
    def __init__(self, root, keys):
        self.root, self.keys = Path(root), keys
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def acquire(self, preferred=0):
        ids = list(self.keys)
        while True:
            for offset in range(len(ids)):
                api = ids[(preferred + offset) % len(ids)]
                stream = (self.root / f"{api}.lock").open("a+b")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                state_path = self.root / f"{api}.json"
                state = read(state_path) if state_path.exists() else {}
                if state.get("cooldown_until", 0) > time.time():
                    stream.close()
                    continue
                try:
                    yield api, self.keys[api], state
                finally:
                    state["inflight"] = None
                    write(state_path, state)
                    stream.close()
                return
            time.sleep(0.5)


class Client:
    provider = "xcode-responses"
    model = MODEL
    base_url = BASE_URL
    model_revision = ""
    temperature = 0
    max_output_tokens = 8192
    configured = True

    def __init__(self, config, directory, slots, job_id, cap, *, send=None):
        self.config, self.directory, self.slots = config, Path(directory), slots
        self.job_id, self.cap, self.index = job_id, cap, 0
        self.timeout_s = config["timeout_seconds"]
        self.endpoint_hash = digest(BASE_URL + "/responses")
        self.profile = Profile("responses", BASE_URL, MODEL, "XCODE_GPT_API_KEY",
                               timeout_seconds=self.timeout_s)
        self.send = send or self._send
        self.replayed = 0
        self.last_usage = {}
        self.last_generation = {}
        self.last_response_identity = {}
        self.last_request_identity = {}
        self.last_reasoning_content = None
        self.last_service_error = {}
        self.last_response_json = None

    def check_service(self):
        return True, "configured"

    def audit_identity(self, *, seed=None):
        return {"enabled": True, "provider": self.provider, "model": MODEL,
                "endpoint_sha256": self.endpoint_hash, "seed": seed, "temperature": 0,
                "max_tokens": 8192, "reasoning_effort": "high", "cache_eligible": False,
                "response_metadata_policy": "request_context_with_reported_model_check"}

    def build_payload(self, messages, max_tokens=2048, seed=None, temperature=None,
                      response_format=None, enable_thinking=False, top_p=None):
        converted, system = convert_messages(messages, "responses")
        payload = {"model": MODEL, "input": converted, "stream": False, "store": False,
                   "max_output_tokens": min(int(max_tokens), 8192),
                   "temperature": 0 if temperature is None else float(temperature),
                   "reasoning": {"effort": "high"}}
        if system:
            payload["instructions"] = system
        if seed is not None:
            payload["seed"] = int(seed)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if response_format is not None:
            payload["text"] = {"format": dict(response_format)}
        return payload

    def _send(self, payload, key):
        request = urllib.request.Request(
            self.profile.endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "Accept": "application/json"}, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.build_opener(_NoRedirect()).open(
                    request, timeout=self.timeout_s) as response:
                raw = response.read()
                request_id = response.headers.get("x-request-id")
        except urllib.error.HTTPError as exc:
            code, delay = exc.code, retry_after(exc.headers.get("Retry-After"))
            exc.close()
            raise APIError("http_error", code, delay) from None
        except (OSError, urllib.error.URLError, http.client.HTTPException):
            raise APIError("transport_outcome_unknown") from None
        try:
            native = json.loads(raw)
            assert_public(native, self.slots.keys.values())
            assert_public(request_id, self.slots.keys.values())
            result = normalize_response(
                self.profile, HTTPResult(native, time.monotonic() - started, request_id))
        except Exception:
            raise APIError("invalid_provider_response") from None
        if result.returned_model and result.returned_model != MODEL:
            raise APIError("reported_model_mismatch")
        if result.tool_calls:
            raise APIError("unexpected_native_tool_call")
        if result.finish_reason != "completed":
            raise APIError("incomplete_provider_response")
        return {
            "text": result.text, "reasoning": result.reasoning, "native_response": native,
            "response_identity": {"model": result.returned_model,
                                  "model_reported": bool(result.returned_model),
                                  "requested_model": MODEL},
            "usage": {"prompt_tokens": result.usage.input_tokens,
                      "completion_tokens": result.usage.output_tokens,
                      "total_tokens": result.usage.total_tokens},
            "generation": {"status": result.finish_reason, "finish_reason": "stop",
                           "reasoning_tokens": result.usage.reasoning_tokens},
            "provider_request_id_sha256": digest(request_id) if request_id else None,
            "elapsed_seconds": time.monotonic() - started,
        }

    def _restore(self, record):
        self.last_usage = record["usage"]
        self.last_generation = {**record["generation"], "usage": self.last_usage}
        self.last_response_identity = record["response_identity"]
        self.last_reasoning_content = record.get("reasoning") or None
        self.last_response_json = record.get("native_response")
        self.last_service_error = {}
        return record["text"]

    def chat(self, messages, **kwargs):
        self.index += 1
        if self.index > self.cap:
            raise APIError("logical_request_budget_exhausted")
        self.last_usage = {}
        self.last_generation = {}
        self.last_response_identity = {}
        self.last_reasoning_content = None
        self.last_service_error = {}
        self.last_response_json = None
        payload = self.build_payload(messages, **kwargs)
        assert_public(payload, self.slots.keys.values())
        request_dir = self.directory / f"{self.index:03d}"
        request_dir.mkdir(parents=True, exist_ok=True)
        identity = digest(payload)
        self.last_request_identity = {"payload_sha256": identity,
                                      "model_identity": self.audit_identity(seed=kwargs.get("seed"))}
        intent = {"payload_sha256": identity, "native_payload": payload,
                  "messages": messages, "parameters": kwargs, "job_id": self.job_id,
                  "ordinal": self.index, "automatic_retry": True,
                  "reference_answer_visible": False}
        intent_path = request_dir / "intent.json"
        if intent_path.exists():
            if read(intent_path) != intent:
                raise APIError("resume_payload_identity_changed")
        else:
            write(intent_path, intent, once=True)
        response_path = request_dir / "success.json"
        if response_path.exists():
            record = read(response_path)
            if record["payload_sha256"] != identity:
                raise APIError("resume_response_identity_changed")
            self.replayed += 1
            return self._restore(record)
        attempts = sorted(request_dir.glob("attempt-*.intent.json"))
        count = len(attempts)
        last = request_dir / f"attempt-{count:02d}.json"
        if last.exists():
            time.sleep(max(0, read(last).get("retry_at", 0) - time.time()))
        while count < self.config["attempts_per_request"]:
            count += 1
            with self.slots.acquire((int(digest(self.job_id)[:8], 16) + count - 1)
                                    % len(self.slots.keys)) as (api, key, state):
                started = time.time()
                admission = {"api_id": api, "job_id": self.job_id, "ordinal": self.index,
                             "attempt": count, "payload_sha256": identity, "started_at": started,
                             "timeout_seconds": self.timeout_s}
                write(request_dir / f"attempt-{count:02d}.intent.json", admission, once=True)
                state["inflight"] = admission
                write(self.slots.root / f"{api}.json", state)
                try:
                    record = self.send(payload, key)
                    assert_public(record, self.slots.keys.values())
                    record.update(payload_sha256=identity, api_id=api, attempt=count,
                                  received_at=time.time(), status="received")
                    write(response_path, record, once=True)
                    write(request_dir / f"attempt-{count:02d}.json",
                          {**admission, "status": "received", "ended_at": time.time()}, once=True)
                    state.update(successes=state.get("successes", 0) + 1,
                                 last_error=None, cooldown_until=0)
                    return self._restore(record)
                except APIError as exc:
                    delay = backoff(count, self.config, exc.retry_after)
                    failure = {**admission, "status": "error", "code": exc.code,
                               "http_status": exc.status, "ended_at": time.time(),
                               "retry_at": time.time() + delay}
                    write(request_dir / f"attempt-{count:02d}.json", failure, once=True)
                    self.last_service_error = {"reason_code": exc.code, "http_status": exc.status}
                    state.update(errors=state.get("errors", 0) + 1, last_error=failure,
                                 cooldown_until=time.time() + delay)
                    if exc.code == "reported_model_mismatch":
                        raise
            if count < self.config["attempts_per_request"]:
                time.sleep(delay)
        raise APIError("transport_attempt_budget_exhausted")


def qualify(keys):
    """Credential readiness checks only; no inference or question sent."""
    rows = []
    for api, key in keys.items():
        request = urllib.request.Request(BASE_URL + "/models",
                    headers={"Authorization": "Bearer " + key, "Accept": "application/json"})
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=30) as response:
                native = json.loads(response.read())
                assert_public(native, keys.values())
                rows.append({"api_id": api, "http_status": response.status,
                             "model_listed": any(r.get("id") == MODEL
                                                 for r in native.get("data", []))})
        except urllib.error.HTTPError as exc:
            rows.append({"api_id": api, "http_status": exc.code, "model_listed": False})
            exc.close()
        except Exception:
            rows.append({"api_id": api, "code": "readiness_unavailable", "model_listed": False})
    return rows
