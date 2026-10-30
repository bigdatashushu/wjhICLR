"""Durable client.chat boundary evidence using fixtures only, never model IO."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import hashlib
import http.client
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import uuid

import pytest

from skill3d.cluster.perception_client import RPC_SCHEMA, TOKEN_ENV, RemoteBailianClient
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.client_errors import ServiceUnavailable
from skill3d.synthesis.request_audit import describe_request, payload_bytes

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "perception_gateway_telemetry_tests", ROOT / "scripts/cluster_perception_gateway.py")
gateway_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway_module)
CONFIG = {"model_revision": "fixture-revision", "timeout_s": 5.0}
TOKEN = "telemetry-fixture-token-not-a-real-credential"
KEY = "telemetry-fixture-key-not-a-real-credential"
PROMPT = "fixture-prompt-must-not-enter-audit"
CONTENT = "fixture-content-must-not-enter-audit"
ERROR_BODY = "fixture-exception-body-must-not-enter-audit"
USAGE = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
UntrustedError = type("fixture-private-exception-name", (Exception,), {})


def envelope():
    client = BailianClient(**CONFIG)
    payload = client.build_payload([{"role": "user", "content": PROMPT}])
    return payload_bytes({
        "schema_version": RPC_SCHEMA, "request_id": str(uuid.uuid4()),
        "payload": payload, "audit_identity": client.audit_identity(),
    })


class FixtureUpstream:
    def __init__(self):
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.expected = 1
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.lock = threading.Lock()
        self.usage = copy.deepcopy(USAGE)
        self.error = None
        self.content = CONTENT
        self.database = None

    def client(self, **options):
        fixture = self

        class Client(BailianClient):
            def chat(self, messages, **kwargs):
                # A separate connection proves the start receipt committed before chat.
                with sqlite3.connect(fixture.database) as db:
                    assert db.execute(
                        "SELECT COUNT(*) FROM upstream_audit WHERE state='active'"
                    ).fetchone()[0] > 0
                with fixture.lock:
                    fixture.calls += 1
                    fixture.active += 1
                    fixture.peak = max(fixture.peak, fixture.active)
                    if fixture.active == fixture.expected:
                        fixture.entered.set()
                try:
                    assert fixture.release.wait(timeout=10), "fixture release timed out"
                    self.last_usage = copy.deepcopy(fixture.usage)
                    if fixture.error is not None:
                        raise fixture.error
                    payload = self.build_payload(messages, **kwargs)
                    self.last_request_identity = describe_request(
                        payload, model_identity=self.audit_identity(seed=kwargs.get("seed")))
                    self.last_response_identity = {
                        "id": "fixture-response-not-for-audit",
                        "model": self.model, "created": 123, "backend_fingerprint": None,
                    }
                    self.last_generation = {
                        "mode": "nonstream",
                        "usage_status": "measured" if self.last_usage else "unknown",
                        "provider": self.provider, "endpoint_sha256": self.endpoint_hash,
                        "request_identity": self.last_request_identity,
                        "finish_reason": "stop", "response_identity": self.last_response_identity,
                    }
                    return fixture.content
                finally:
                    with fixture.lock:
                        fixture.active -= 1

        return Client(**options)


@pytest.fixture
def upstream(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.setenv("DASHSCOPE_API_KEY", KEY)

    def forbidden(*_args, **_kwargs):
        pytest.fail("real BailianClient.chat is forbidden in telemetry tests")

    monkeypatch.setattr(BailianClient, "chat", forbidden)
    fixture = FixtureUpstream()
    yield fixture
    fixture.release.set()


@contextmanager
def service(tmp_path, upstream, *, max_concurrency=8):
    root = tmp_path / "cache"
    upstream.database = root / "responses.sqlite3"
    gateway = gateway_module.PerceptionGateway(
        CONFIG, root, max_concurrency=max_concurrency, client_factory=upstream.client)
    try:
        yield gateway
    finally:
        upstream.release.set()
        gateway.close()


def receipts(gateway):
    with gateway._condition:
        cursor = gateway._db.execute("SELECT * FROM upstream_audit ORDER BY started_at_ns")
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row)) for row in cursor]


def counts(gateway, **expected):
    actual = gateway.health()["upstream"]
    assert actual["started"] == (
        actual["active"] + actual["succeeded"] + actual["failed"] + actual["unknown"])
    assert actual["usage_measured"] + actual["usage_unknown"] == (
        actual["succeeded"] + actual["failed"] + actual["unknown"])
    for name, value in expected.items():
        assert actual[name] == value, (name, actual)
    return actual


def assert_no_replay(gateway, raw):
    with pytest.raises(gateway_module.GatewayError) as error:
        gateway.chat(raw)
    assert error.value.status == 409
    assert error.value.code == "indeterminate"
    assert error.value.retryable is False


def test_durable_safe_receipt_at_chat_boundary_and_health_compatibility(tmp_path, upstream):
    raw = envelope()
    upstream.release.clear()
    with service(tmp_path, upstream) as gateway:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(gateway.chat, raw)
            try:
                assert upstream.entered.wait(timeout=5)
                counts(gateway, started=1, active=1, peak=1, succeeded=0, unknown=0)
                start = receipts(gateway)[0]
                assert start["state"] == "active"
                assert start["ended_at_ns"] is start["duration_ns"] is None
                assert start["success"] is start["error_type"] is None
                assert start["active_at_start"] == start["peak_at_start"] == 1
                assert start["token_usage"] == "{}"
            finally:
                upstream.release.set()
            status, reply = future.result(timeout=5)
        assert status == 200
        assert set(json.loads(reply)) == {
            "schema_version", "request_id", "payload_sha256", "audit_identity", "content",
            "last_usage", "last_generation", "last_response_identity", "last_request_identity",
        }
        counts(gateway, active=0, succeeded=1, failed=0, usage_measured=1, usage_unknown=0)
        end = receipts(gateway)[0]
        assert end["request_id"] == json.loads(raw)["request_id"]
        assert end["payload_sha256"] == hashlib.sha256(
            payload_bytes(json.loads(raw)["payload"])).hexdigest()
        assert end["state"] == "success" and end["success"] == 1
        assert end["ended_at_ns"] >= end["started_at_ns"] > 0
        assert end["duration_ns"] > 0
        assert end["active_at_end"] == 0 and end["peak_at_end"] == 1
        assert json.loads(end["token_usage"]) == USAGE
        health = gateway.health()
        assert health["schema_version"] == RPC_SCHEMA and health["status"] == "ready"
        assert health["max_concurrency"] == 8
        assert health["identity"] == BailianClient(**CONFIG).audit_identity()
        health["upstream"]["started"] = 999
        counts(gateway, started=1)
        safe = payload_bytes({"receipts": receipts(gateway), "health": gateway.health()})
        for forbidden in (PROMPT, CONTENT, TOKEN, KEY, ERROR_BODY, "fixture-response-not-for-audit"):
            assert forbidden.encode() not in safe
        assert gateway.root.stat().st_mode & 0o777 == 0o700
        assert upstream.database.stat().st_mode & 0o777 == 0o600


def test_32_rpc_invocations_are_only_eight_upstream_calls_with_capacity_and_cache(
        tmp_path, upstream, monkeypatch):
    upstream.expected = 8
    upstream.release.clear()
    unique = [envelope() for _ in range(8)]
    ninth = envelope()
    with service(tmp_path, upstream) as gateway:
        waiters = []
        duplicates_waiting = threading.Event()
        original_wait = gateway._condition.wait

        def record_wait(*args, **kwargs):
            waiters.append(1)
            if len(waiters) == 24:
                duplicates_waiting.set()
            return original_wait(*args, **kwargs)

        monkeypatch.setattr(gateway._condition, "wait", record_wait)
        with ThreadPoolExecutor(max_workers=32) as pool:
            owners = [pool.submit(gateway.chat, raw) for raw in unique]
            duplicates = []
            try:
                assert upstream.entered.wait(timeout=5)
                duplicates = [pool.submit(gateway.chat, unique[index % 8]) for index in range(24)]
                assert duplicates_waiting.wait(timeout=5)
                counts(gateway, started=8, active=8, peak=8, succeeded=0)
                assert upstream.calls == upstream.active == upstream.peak == 8
                assert len(receipts(gateway)) == 8
                with pytest.raises(gateway_module.GatewayError) as error:
                    gateway.chat(ninth)
                assert (error.value.status, error.value.code, error.value.retryable) == (
                    429, "capacity", True)
                conflict = json.loads(unique[0])
                conflict["payload"]["messages"][0]["content"] = "conflicting fixture"
                with pytest.raises(gateway_module.GatewayError) as error:
                    gateway.chat(payload_bytes(conflict))
                assert error.value.code == "request_id_conflict"
                counts(gateway, started=8)
            finally:
                upstream.release.set()
            replies = [future.result(timeout=5) for future in owners]
            assert all(reply[0] == 200 for reply in replies)
            for index, future in enumerate(duplicates):
                assert future.result(timeout=5) == replies[index % 8]
        for raw, reply in zip(unique, replies):
            assert gateway.chat(raw) == reply
        counts(gateway, started=8, active=0, succeeded=8, peak=8, failed=0, unknown=0)
        assert gateway.chat(ninth)[0] == 200
        counts(gateway, started=9, succeeded=9, peak=8)
        assert upstream.calls == 9
        assert max(row["active_at_start"] for row in receipts(gateway)) == 8


def test_lost_ack_transport_retry_reads_cache_without_new_upstream_count(tmp_path, upstream):
    with service(tmp_path, upstream) as gateway:
        sent = []

        class Reply(io.BytesIO):
            def __init__(self, status, raw):
                super().__init__(raw)
                self.code = status
                self.headers = {"Content-Length": str(len(raw))}

        def opener(request, timeout):
            sent.append(request.data)
            status, raw = gateway.chat(request.data)
            if len(sent) == 1:
                raise http.client.IncompleteRead(b"")
            return Reply(status, raw)

        remote = RemoteBailianClient(
            **CONFIG, endpoint="http://127.0.0.1:27035", opener=opener)
        assert remote.chat([{"role": "user", "content": PROMPT}]) == CONTENT
        assert len(sent) == 2 and sent[0] == sent[1]
        assert upstream.calls == len(receipts(gateway)) == 1
        counts(gateway, started=1, succeeded=1, active=0, peak=1, usage_measured=1)


def test_pending_transport_response_does_not_start_another_upstream(
        tmp_path, upstream, monkeypatch):
    raw = envelope()
    upstream.release.clear()
    with service(tmp_path, upstream) as gateway:
        with ThreadPoolExecutor(max_workers=1) as pool:
            owner = pool.submit(gateway.chat, raw)
            try:
                assert upstream.entered.wait(timeout=5)
                with monkeypatch.context() as patch:
                    patch.setattr(gateway._condition, "wait", lambda **_kwargs: False)
                    with pytest.raises(gateway_module.GatewayError) as error:
                        gateway.chat(raw)
                    assert (error.value.status, error.value.code, error.value.retryable) == (
                        503, "pending", True)
                counts(gateway, started=1, active=1, failed=0, unknown=0)
            finally:
                upstream.release.set()
            first = owner.result(timeout=5)
        assert gateway.chat(raw) == first and upstream.calls == 1
        counts(gateway, started=1, succeeded=1, active=0)


@pytest.mark.parametrize("stage", ["client_factory", "response_validation"])
def test_admission_and_postprocessing_are_not_upstream_active(
        tmp_path, upstream, monkeypatch, stage):
    entered, release = threading.Event(), threading.Event()
    with service(tmp_path, upstream, max_concurrency=1) as gateway:
        target, name = ((gateway, "_client_factory") if stage == "client_factory"
                        else (gateway_module, "validate_result"))
        original = getattr(target, name)

        def blocked(*args, **kwargs):
            entered.set()
            assert release.wait(timeout=10)
            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, blocked)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(gateway.chat, envelope())
            try:
                assert entered.wait(timeout=5)
                expected = 0 if stage == "client_factory" else 1
                counts(gateway, started=expected, succeeded=expected, active=0, peak=expected)
                assert upstream.calls == expected and upstream.active == 0
                with pytest.raises(gateway_module.GatewayError) as error:
                    gateway.chat(envelope())
                assert error.value.code == "capacity"
            finally:
                release.set()
            assert future.result(timeout=5)[0] == 200
        counts(gateway, started=1, succeeded=1, active=0)


@pytest.mark.parametrize("kind,error_type,status", [
    (ServiceUnavailable, "ServiceUnavailable", 503),
    (TimeoutError, "TimeoutError", 502), (OSError, "OSError", 502),
    (RuntimeError, "RuntimeError", 502), (ValueError, "ValueError", 502),
    (TypeError, "TypeError", 502), (UntrustedError, "Exception", 502),
])
def test_failed_chat_receipt_is_safe_cached_and_survives_restart(
        tmp_path, upstream, kind, error_type, status, capsys):
    upstream.error = kind(ERROR_BODY + TOKEN + KEY)
    upstream.usage = {}
    raw = envelope()
    with service(tmp_path, upstream) as gateway:
        first = gateway.chat(raw)
        assert first[0] == status and json.loads(first[1])["retryable"] is False
        assert gateway.chat(raw) == first
        counts(gateway, started=1, failed=1, succeeded=0, unknown=0, usage_unknown=1)
        receipt = receipts(gateway)[0]
        assert receipt["state"] == "error" and receipt["success"] == 0
        assert receipt["error_type"] == error_type
        safe = payload_bytes(receipt) + first[1] + payload_bytes(gateway.health())
        for forbidden in (PROMPT, CONTENT, ERROR_BODY, TOKEN, KEY, UntrustedError.__name__):
            assert forbidden.encode() not in safe
    with service(tmp_path, upstream) as gateway:
        assert gateway.chat(raw) == first
        counts(gateway, started=1, failed=1, active=0, peak=1, unknown=0, usage_unknown=1)
        assert upstream.calls == 1
    assert not capsys.readouterr().err


def test_factory_failure_has_no_upstream_receipt_or_count(tmp_path, upstream, monkeypatch):
    with service(tmp_path, upstream) as gateway:
        def unavailable(**_options):
            raise ServiceUnavailable(ERROR_BODY)

        monkeypatch.setattr(gateway, "_client_factory", unavailable)
        raw = envelope()
        first = gateway.chat(raw)
        assert first[0] == 503 and gateway.chat(raw) == first
        assert not receipts(gateway) and upstream.calls == 0
        counts(gateway, started=0, failed=0, unknown=0)


def test_failed_call_with_partial_measured_usage_is_not_usage_unknown(tmp_path, upstream):
    upstream.error = RuntimeError(ERROR_BODY)
    upstream.usage = {"prompt_tokens": 0, "private": ERROR_BODY}
    with service(tmp_path, upstream) as gateway:
        assert gateway.chat(envelope())[0] == 502
        counts(gateway, failed=1, succeeded=0, unknown=0, usage_measured=1, usage_unknown=0)
        assert json.loads(receipts(gateway)[0]["token_usage"]) == {"prompt_tokens": 0}


def test_successful_chat_but_invalid_rpc_evidence_is_not_upstream_failure(tmp_path, upstream):
    upstream.content = ""
    raw = envelope()
    with service(tmp_path, upstream) as gateway:
        first = gateway.chat(raw)
        assert first[0] == 502 and gateway.chat(raw) == first
        counts(gateway, started=1, succeeded=1, failed=0, unknown=0, usage_measured=1)
        assert receipts(gateway)[0]["success"] == 1
        assert upstream.calls == 1


@pytest.mark.parametrize("usage,safe", [
    ({}, {}),
    ({"private": ERROR_BODY}, {}),
    ({"prompt_tokens": True, "completion_tokens": -1, "total_tokens": KEY}, {}),
    ({"prompt_tokens": 12.0, "completion_tokens": 0, "total_tokens": 12,
      "private": ERROR_BODY}, {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12}),
    ({"prompt_tokens": 1.5, "completion_tokens": 2**53, "total_tokens": False}, {}),
])
def test_only_allowlisted_safe_numeric_usage_is_measured(tmp_path, upstream, usage, safe):
    upstream.usage = usage
    with service(tmp_path, upstream) as gateway:
        gateway.chat(envelope())
        receipt = receipts(gateway)[0]
        assert json.loads(receipt["token_usage"]) == safe
        assert receipt["usage_status"] == ("measured" if safe else "unknown")
        counts(gateway, succeeded=1, failed=0, unknown=0,
               usage_measured=int(bool(safe)), usage_unknown=int(not safe))
        assert ERROR_BODY.encode() not in payload_bytes(receipt)
        assert KEY.encode() not in payload_bytes(receipt)


@pytest.mark.parametrize("value", [
    None, [], True, KEY,
    {"prompt_tokens": float("nan"), "completion_tokens": float("inf"),
     "total_tokens": float("-inf")},
])
def test_invalid_usage_cannot_break_audit(value):
    assert gateway_module._safe_usage(value) == {}


@pytest.mark.parametrize("kind", [KeyboardInterrupt, SystemExit])
def test_interruption_is_unknown_not_failure_and_cannot_replay(tmp_path, upstream, kind):
    upstream.error = kind(ERROR_BODY)
    upstream.usage = {}
    raw = envelope()
    with service(tmp_path, upstream) as gateway:
        with pytest.raises(kind):
            gateway.chat(raw)
        counts(gateway, started=1, active=0, failed=0, succeeded=0, unknown=1, usage_unknown=1)
        receipt = receipts(gateway)[0]
        assert receipt["state"] == "unknown" and receipt["success"] is None
        assert receipt["error_type"] == "interrupted" and receipt["ended_at_ns"] > 0
        assert_no_replay(gateway, raw)
    with service(tmp_path, upstream) as gateway:
        assert_no_replay(gateway, raw)
        counts(gateway, unknown=1, active=0, failed=0)
        assert upstream.calls == 1


@pytest.mark.parametrize("stage", ["start", "end", "reply"])
def test_persistence_failures_never_replay_or_leak_capacity(
        tmp_path, upstream, stage, capsys):
    raw = envelope()
    with service(tmp_path, upstream) as gateway:
        operation = {
            "start": "INSERT ON upstream_audit",
            "end": "UPDATE ON upstream_audit",
            "reply": "UPDATE ON requests WHEN NEW.state='complete'",
        }[stage]
        with gateway._db:
            gateway._db.execute(
                f"CREATE TRIGGER fixture_failure BEFORE {operation} "
                f"BEGIN SELECT RAISE(ABORT,'{ERROR_BODY}'); END")
        expected_error = sqlite3.DatabaseError if stage == "reply" else gateway_module.GatewayError
        with pytest.raises(expected_error) as error:
            gateway.chat(raw)
        if stage != "reply":
            assert error.value.status == 503 and error.value.retryable is False
            assert ERROR_BODY not in str(error.value)
        started = int(stage != "start")
        counts(gateway, started=started, active=0, succeeded=int(stage == "reply"),
               failed=0, unknown=int(stage == "end"),
               audit_write_failures=int(stage != "reply"))
        assert upstream.calls == started
        assert_no_replay(gateway, raw)
        if stage == "end":
            assert receipts(gateway)[0]["state"] == "active"
            assert receipts(gateway)[0]["ended_at_ns"] is None
        with gateway._db:
            gateway._db.execute("DROP TRIGGER fixture_failure")
        assert gateway.chat(envelope())[0] == 200
    with service(tmp_path, upstream) as gateway:
        assert_no_replay(gateway, raw)
        counts(gateway, started=started + 1, active=0, unknown=int(stage == "end"),
               succeeded=1 + int(stage == "reply"), failed=0)
        assert upstream.calls == started + 1
        assert all(row["state"] != "active" for row in receipts(gateway))
    assert not capsys.readouterr().err


@pytest.mark.parametrize("stage", ["before_chat", "inside_chat"])
def test_process_crash_recovers_durable_unknown_without_inventing_end_or_replay(
        tmp_path, upstream, stage):
    raw = envelope()
    root = tmp_path / "cache"
    script = (
        "import os,sys\n"
        "from pathlib import Path\n"
        "from cluster_perception_gateway import PerceptionGateway\n"
        "class FixtureClient:\n"
        "    def __init__(self, **kwargs): pass\n"
        "    def chat(self, *args, **kwargs): os._exit(73)\n"
        f"gateway=PerceptionGateway({CONFIG!r},Path({str(root)!r}),client_factory=FixtureClient)\n"
    )
    if stage == "before_chat":
        script += (
            "original=gateway._start_upstream\n"
            "def crash(*args):\n"
            "    original(*args)\n"
            "    os._exit(73)\n"
            "gateway._start_upstream=crash\n"
        )
    script += "gateway.chat(sys.stdin.buffer.read())\n"
    process = subprocess.run(
        [sys.executable, "-c", script], input=raw, capture_output=True, timeout=10,
        env={"PATH": os.defpath, TOKEN_ENV: TOKEN,
             "PYTHONPATH": str(ROOT / "scripts") + os.pathsep + str(ROOT / "src")})
    assert process.returncode == 73 and not process.stdout and not process.stderr
    with sqlite3.connect(root / "responses.sqlite3") as db:
        assert db.execute("SELECT state FROM requests").fetchone() == ("pending",)
        assert db.execute("SELECT state,ended_at_ns FROM upstream_audit").fetchone() == (
            "active", None)
    with service(tmp_path, upstream) as gateway:
        counts(gateway, started=1, unknown=1, failed=0, succeeded=0, active=0,
               usage_unknown=1, peak=1)
        assert_no_replay(gateway, raw)
        receipt = receipts(gateway)[0]
        assert receipt["success"] is receipt["ended_at_ns"] is receipt["duration_ns"] is None
        assert upstream.calls == 0
        assert gateway.chat(envelope())[0] == 200
        counts(gateway, started=2, succeeded=1, unknown=1, active=0, peak=1)


def test_legacy_cache_migration_does_not_invent_upstream_history(tmp_path, upstream):
    raw = envelope()
    with service(tmp_path, upstream) as gateway:
        first = gateway.chat(raw)
        with gateway._db:
            gateway._db.execute("DROP TABLE upstream_audit")
    with service(tmp_path, upstream) as gateway:
        assert gateway.chat(raw) == first
        assert receipts(gateway) == []
        counts(gateway, started=0, succeeded=0, failed=0, unknown=0, peak=0)
        assert upstream.calls == 1


@pytest.mark.parametrize("cap", [0, 65, -1, True])
def test_upstream_cap_is_bounded_to_64(tmp_path, upstream, cap):
    with pytest.raises(ValueError, match="between 1 and 64"):
        gateway_module.PerceptionGateway(CONFIG, tmp_path / "cache", max_concurrency=cap)


@pytest.mark.parametrize("cap", [16, 32, 64])
def test_expanded_capacity_preserves_admission_and_audit(tmp_path, upstream, cap):
    upstream.expected = cap
    upstream.release.clear()
    unique = [envelope() for _ in range(cap)]
    with service(tmp_path, upstream, max_concurrency=cap) as gateway:
        with ThreadPoolExecutor(max_workers=cap) as pool:
            owners = [pool.submit(gateway.chat, raw) for raw in unique]
            try:
                assert upstream.entered.wait(timeout=10)
                counts(gateway, started=cap, active=cap, peak=cap)
                with pytest.raises(gateway_module.GatewayError) as error:
                    gateway.chat(envelope())
                assert (error.value.status, error.value.code) == (429, "capacity")
            finally:
                upstream.release.set()
            assert all(future.result(timeout=10)[0] == 200 for future in owners)
        counts(gateway, started=cap, active=0, succeeded=cap, failed=0, unknown=0)
        assert upstream.calls == cap
        assert len(receipts(gateway)) == cap
