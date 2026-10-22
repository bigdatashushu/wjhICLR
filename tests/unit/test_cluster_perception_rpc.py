"""Real loopback HTTP and Bailian chat evidence, never actual model calls."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import selectors
import socket
import sqlite3
import subprocess
import sys
import threading
import urllib.request
import uuid

import pytest

from skill3d.cluster.perception_client import (
    ENDPOINT_ENV, MAX_BODY_BYTES, RPC_SCHEMA, TOKEN_ENV, RemoteBailianClient,
)
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.client_errors import ServiceUnavailable
from skill3d.synthesis.request_audit import payload_bytes

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "perception_gateway_tests", ROOT / "scripts/cluster_perception_gateway.py")
gateway_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway_module)
TOKEN = "rpc-test-only-not-a-real-credential"
KEY = "upstream-test-only-not-a-real-credential"
CONFIG = {
    "online_model": "qwen3.8-flash", "online_model_revision": "frozen-test-revision",
    "online_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "online_timeout_s": 5.0, "temperature": 0.0, "max_tokens": 8192,
}
OPTIONS = gateway_module.client_options(CONFIG)
UPSTREAM_BODY = {
    "id": "actual-upstream-response", "model": "qwen3.8-flash-resolved",
    "created": 1234, "system_fingerprint": "actual-backend",
    "choices": [{"message": {"content": '{"objects":["chair"]}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
}


@contextmanager
def running_server(server):
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


class Upstream:
    """HTTP stub used by the real BailianClient opener instead of HTTPS/model IO."""

    def __init__(self):
        self.requests = []
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.all_entered = threading.Event()
        self.expected_active = 1
        self.body = copy.deepcopy(UPSTREAM_BODY)
        self.status = 200
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                with stub.lock:
                    stub.requests.append((raw, dict(self.headers), self.path))
                    if len(stub.requests) >= stub.expected_active:
                        stub.all_entered.set()
                stub.entered.set()
                if not stub.release.wait(timeout=10):
                    return
                body = payload_bytes(stub.body)
                self.send_response(stub.status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)

    @contextmanager
    def opener(self, request, timeout):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1],
                                                timeout=timeout)
        try:
            connection.request("POST", "/chat/completions", body=request.data,
                               headers=dict(request.header_items()))
            with connection.getresponse() as response:
                if response.status >= 400:
                    import io
                    import urllib.error
                    raise urllib.error.HTTPError(request.full_url, response.status, "stub",
                                                 {}, io.BytesIO(response.read()))
                yield response
        finally:
            connection.close()

    def client(self, **options):
        return BailianClient(**options, opener=self.opener)


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.setenv("DASHSCOPE_API_KEY", KEY)
    upstream = Upstream()
    with running_server(upstream.server):
        try:
            yield upstream
        finally:
            upstream.release.set()


@contextmanager
def service(tmp_path, stub, *, max_concurrency=8):
    gateway = gateway_module.PerceptionGateway(
        CONFIG, tmp_path / "cache", max_concurrency=max_concurrency,
        client_factory=stub.client)
    try:
        with running_server(gateway_module.make_server(gateway, "127.0.0.1", 0)) as endpoint:
            yield gateway, endpoint
    finally:
        gateway.close()


def exchange(endpoint, raw=None, *, token=TOKEN, route="/chat", headers=None):
    port = int(endpoint.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    supplied = {"Content-Type": "application/json"}
    if token is not None:
        supplied["Authorization"] = "Bearer " + token
    supplied.update(headers or {})
    try:
        connection.request("GET" if raw is None else "POST", route, body=raw, headers=supplied)
        with connection.getresponse() as response:
            return response.status, response.read()
    finally:
        connection.close()


def envelope(messages=None, *, request_id=None, **kwargs):
    client = BailianClient(**OPTIONS)
    payload = client.build_payload(messages or [{"role": "user", "content": "unique prompt"}],
                                   **kwargs)
    return {
        "schema_version": RPC_SCHEMA, "request_id": request_id or str(uuid.uuid4()),
        "payload": payload, "audit_identity": client.audit_identity(seed=payload.get("seed")),
    }


def test_remote_constructor_and_health_do_not_require_worker_upstream_key(tmp_path, stub, monkeypatch):
    with service(tmp_path, stub) as (_gateway, endpoint):
        monkeypatch.delenv("DASHSCOPE_API_KEY")
        monkeypatch.setenv(ENDPOINT_ENV, endpoint)
        remote = RemoteBailianClient(**OPTIONS)
        assert isinstance(remote, BailianClient)
        assert remote.configured
        assert remote.check_service()[0]
        assert remote.audit_identity(seed=137) == BailianClient(**OPTIONS).audit_identity(seed=137)
        assert remote.endpoint_hash == hashlib.sha256(OPTIONS["base_url"].encode()).hexdigest()
        status, raw = exchange(endpoint, token=None, route="/health")
        assert status == 200
        health = json.loads(raw)
        assert health["identity"] == remote.audit_identity()
        assert health["max_body_bytes"] == 64 * 1024 * 1024
        assert health["max_concurrency"] == 8
        assert not stub.requests
        monkeypatch.delenv(TOKEN_ENV)
        assert not remote.configured
        with pytest.raises(ServiceUnavailable):
            remote.chat([{"role": "user", "content": "hello"}])


def test_real_bailian_multimodal_body_and_all_upstream_evidence_unchanged(tmp_path, stub):
    jpeg_a, jpeg_b = b"\xff\xd8first-exact-jpeg\xff\xd9", b"\xff\xd8second-exact-jpeg\xff\xd9"
    messages = [
        {"role": "system", "content": "Frozen instruction"},
        {"role": "user", "content": [
            {"type": "text", "text": "First frame, then second"},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(jpeg_a).decode(),
                "detail": "high"}},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(jpeg_b).decode()}},
        ]},
    ]
    arguments = {"max_tokens": 3072, "seed": 137, "temperature": 0.0,
                 "response_format": {"type": "json_object"}, "enable_thinking": False,
                 "top_p": 0.95}
    expected = BailianClient(**OPTIONS).build_payload(messages, **arguments)
    with service(tmp_path, stub) as (gateway, endpoint):
        remote = RemoteBailianClient(**OPTIONS, endpoint=endpoint)
        assert remote.chat(messages, **arguments) == UPSTREAM_BODY["choices"][0]["message"]["content"]
        assert stub.requests[0][0] == payload_bytes(expected)
        assert stub.requests[0][1]["Authorization"] == "Bearer " + KEY
        assert stub.requests[0][2] == "/chat/completions"
        actual = BailianClient(**OPTIONS, opener=stub.opener)
        actual.chat(messages, **arguments)
        for field in ("last_usage", "last_generation", "last_response_identity", "last_request_identity"):
            assert getattr(remote, field) == getattr(actual, field)
        assert remote.last_response_identity["id"] == "actual-upstream-response"
        assert remote.last_response_identity["model"] == "qwen3.8-flash-resolved"
        images = remote.last_request_identity["images_in_order"]
        assert [image["jpeg_sha256"] for image in images] == [
            hashlib.sha256(jpeg_a).hexdigest(), hashlib.sha256(jpeg_b).hexdigest()]
        persisted = gateway._db.execute("SELECT reply FROM requests").fetchone()[0]
        assert KEY.encode() not in persisted and TOKEN.encode() not in persisted
        assert b"Frozen instruction" not in persisted
        assert base64.b64encode(jpeg_a) not in persisted


@pytest.mark.parametrize("partial_response", [False, True])
def test_ack_loss_retries_same_uuid_and_frozen_bytes_without_reexecution(
        tmp_path, stub, monkeypatch, partial_response):
    messages = [{"role": "user", "content": "original immutable input"}]
    gateway = gateway_module.PerceptionGateway(CONFIG, tmp_path / "cache", client_factory=stub.client)
    server = gateway_module.make_server(gateway, "127.0.0.1", 0)
    original_reply = server.RequestHandlerClass.reply
    original_chat = gateway.chat
    captured = []
    replies = []

    def record_chat(raw):
        captured.append(raw)
        messages[0]["content"] = "caller mutated this after first send"
        result = original_chat(raw)
        replies.append(result)
        return result

    def lose_first_ack(handler, status, raw):
        if status == 200 and len(captured) == 1:
            row = gateway._db.execute("SELECT state,reply FROM requests").fetchone()
            assert row == ("complete", raw)
            handler.close_connection = True
            if partial_response:
                handler.send_response(status)
                handler.send_header("Content-Length", str(len(raw)))
                handler.end_headers()
                handler.wfile.write(raw[:20])
                handler.wfile.flush()
            handler.connection.shutdown(socket.SHUT_RDWR)
            return
        original_reply(handler, status, raw)

    monkeypatch.setattr(gateway, "chat", record_chat)
    monkeypatch.setattr(server.RequestHandlerClass, "reply", lose_first_ack)
    try:
        with running_server(server) as endpoint:
            remote = RemoteBailianClient(**OPTIONS, endpoint=endpoint)
            assert remote.chat(messages) == UPSTREAM_BODY["choices"][0]["message"]["content"]
            assert len(captured) == 2 and captured[0] == captured[1]
            request_id = json.loads(captured[0])["request_id"]
            assert str(uuid.UUID(request_id)) == request_id
            assert len(stub.requests) == 1
            assert json.loads(stub.requests[0][0])["messages"][0]["content"] == "original immutable input"
            assert replies[0] == replies[1]
    finally:
        gateway.close()


def test_concurrent_duplicates_reuse_reply_and_conflicting_uuid_is_rejected(tmp_path, stub):
    stub.release.clear()
    raw = payload_bytes(envelope())
    changed = json.loads(raw)
    changed["payload"]["messages"][0]["content"] = "conflicting prompt"
    with service(tmp_path, stub) as (_gateway, endpoint):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(exchange, endpoint, raw) for _ in range(8)]
            try:
                assert stub.entered.wait(timeout=5)
                status, body = exchange(endpoint, payload_bytes(changed))
                assert status == 409 and json.loads(body)["error"] == "request_id_conflict"
                assert len(stub.requests) == 1
            finally:
                stub.release.set()
            replies = [future.result(timeout=5) for future in futures]
        assert all(reply == replies[0] for reply in replies)
        assert replies[0][0] == 200 and len(stub.requests) == 1
        assert exchange(endpoint, raw) == replies[0]


def test_completed_cache_survives_restart_and_pending_becomes_indeterminate(tmp_path, stub):
    complete = envelope()
    pending = envelope()
    with service(tmp_path, stub) as (gateway, endpoint):
        first = exchange(endpoint, payload_bytes(complete))
        assert first[0] == 200
        with gateway._db:
            gateway._db.execute(
                "INSERT INTO requests VALUES (?,?,?,'pending',NULL,NULL)",
                (pending["request_id"],
                 hashlib.sha256(payload_bytes(pending["payload"])).hexdigest(),
                 hashlib.sha256(payload_bytes(pending["audit_identity"])).hexdigest()))
    with service(tmp_path, stub) as (gateway, endpoint):
        assert exchange(endpoint, payload_bytes(complete)) == first
        status, raw = exchange(endpoint, payload_bytes(pending))
        assert status == 409
        assert json.loads(raw)["error"] == "indeterminate"
        assert json.loads(raw)["retryable"] is False
        assert len(stub.requests) == 1
        assert gateway._db.execute(
            "SELECT state FROM requests WHERE request_id=?", (pending["request_id"],)
        ).fetchone()[0] == "indeterminate"


def test_bounded_eight_upstream_calls_and_capacity_retry_is_transport_only(tmp_path, stub):
    stub.expected_active = 8
    stub.release.clear()
    ninth = payload_bytes(envelope())
    with service(tmp_path, stub) as (gateway, endpoint):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(exchange, endpoint, payload_bytes(envelope())) for _ in range(8)]
            try:
                assert stub.all_entered.wait(timeout=5)
                status, raw = exchange(endpoint, ninth)
                assert status == 429 and json.loads(raw)["retryable"] is True
                assert gateway._db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 8
            finally:
                stub.release.set()
            assert all(future.result(timeout=5)[0] == 200 for future in futures)
        assert exchange(endpoint, ninth)[0] == 200
        assert len(stub.requests) == 9


@pytest.mark.parametrize("change", [
    {"model": "different-model"}, {"max_tokens": 8193}, {"max_tokens": 0},
    {"temperature": 0.5}, {"api_key": "must-not-be-accepted"}, {"seed": True},
])
def test_frozen_model_budget_temperature_and_secret_fields_rejected(tmp_path, stub, change):
    request = envelope()
    request["payload"].update(change)
    with service(tmp_path, stub) as (_gateway, endpoint):
        status, _raw = exchange(endpoint, payload_bytes(request))
        assert status == 422
        assert not stub.requests


def test_auth_and_body_limit_are_enforced_before_read_or_upstream(tmp_path, stub):
    with service(tmp_path, stub) as (_gateway, endpoint):
        raw = payload_bytes(envelope())
        assert exchange(endpoint, raw, token=None)[0] == 401
        assert exchange(endpoint, raw, token="wrong")[0] == 401
        assert exchange(endpoint, b"", headers={"Content-Length": str(MAX_BODY_BYTES + 1)})[0] == 413
        assert exchange(endpoint, raw, headers={"Transfer-Encoding": "chunked"})[0] == 400
        assert exchange(endpoint, raw, headers={"Content-Type": "text/plain"})[0] == 415
        assert exchange(endpoint, b'{"schema_version":1,"schema_version":2}')[0] == 422
        assert not stub.requests


@pytest.mark.parametrize("field", [
    "payload_sha256", "audit_identity", "last_request_identity", "last_usage",
    "last_generation", "last_response_identity",
])
def test_remote_rejects_tampered_response_evidence(tmp_path, stub, monkeypatch, field):
    with service(tmp_path, stub) as (gateway, endpoint):
        original = gateway.chat

        def tampered(raw):
            status, body = original(raw)
            result = json.loads(body)
            if field == "payload_sha256":
                result[field] = "0" * 64
            elif field == "last_usage":
                result[field] = []
            elif field == "last_response_identity":
                result[field]["id"] = "not-actual-upstream"
            elif field == "last_generation":
                result[field]["endpoint_sha256"] = "0" * 64
            elif field == "audit_identity":
                result[field]["model_revision"] = "wrong"
            else:
                result[field]["messages_sha256"] = "0" * 64
            return status, payload_bytes(result)

        monkeypatch.setattr(gateway, "chat", tampered)
        remote = RemoteBailianClient(**OPTIONS, endpoint=endpoint)
        with pytest.raises(RuntimeError, match="invalid perception response evidence"):
            remote.chat([{"role": "user", "content": "hello"}])
        assert remote.last_usage == {}
        assert remote.last_response_identity == {}


def test_upstream_error_is_sanitized_persisted_and_not_retried(tmp_path, stub, capsys):
    stub.status = 500
    stub.body = {"error": {"message": f"private prompt and {KEY} and {TOKEN}"}}
    request = payload_bytes(envelope())
    with service(tmp_path, stub) as (_gateway, endpoint):
        first = exchange(endpoint, request)
        assert first[0] == 503
        assert json.loads(first[1])["error"] == "upstream_unavailable"
        assert KEY.encode() not in first[1] and TOKEN.encode() not in first[1]
        assert b"private prompt" not in first[1]
        assert exchange(endpoint, request) == first
        remote = RemoteBailianClient(**OPTIONS, endpoint=endpoint)
        with pytest.raises(ServiceUnavailable, match="HTTP 503") as error:
            remote.chat([{"role": "user", "content": "hello"}])
        assert KEY not in str(error.value) and TOKEN not in str(error.value)
        assert len(stub.requests) == 2
    assert "private prompt" not in capsys.readouterr().err


@pytest.mark.parametrize("field", ["content", "usage", "response_identity"])
def test_credential_bearing_upstream_reply_never_enters_cache_or_rpc(tmp_path, stub, field):
    if field == "content":
        stub.body["choices"][0]["message"]["content"] = "accidental echo " + KEY
    elif field == "usage":
        stub.body["usage"]["private"] = TOKEN
    else:
        stub.body["id"] = KEY
    request = payload_bytes(envelope())
    with service(tmp_path, stub) as (gateway, endpoint):
        status, raw = exchange(endpoint, request)
        assert status == 502 and json.loads(raw)["error"] == "upstream_invalid"
        assert KEY.encode() not in raw and TOKEN.encode() not in raw
        cached = gateway._db.execute("SELECT reply FROM requests").fetchone()[0]
        assert cached == raw
        assert exchange(endpoint, request) == (status, raw)
        assert len(stub.requests) == 1


def test_actual_service_process_crash_leaves_durable_indeterminate_without_reexecution(tmp_path, stub):
    root = tmp_path / "cache"
    script = (
        "import os,json; from pathlib import Path; "
        "from cluster_perception_gateway import PerceptionGateway,make_server; "
        f"gateway=PerceptionGateway({CONFIG!r},Path({str(root)!r})); "
        "gateway._invoke=lambda *args: os._exit(73); "
        "server=make_server(gateway,'127.0.0.1',0); "
        "print(server.server_address[1],flush=True); server.serve_forever()"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "scripts") + os.pathsep + str(ROOT / "src"),
           TOKEN_ENV: TOKEN, "DASHSCOPE_API_KEY": KEY}
    process = subprocess.Popen([sys.executable, "-c", script], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    request = envelope()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=10), "service did not become ready"
        endpoint = "http://127.0.0.1:" + process.stdout.readline().decode().strip()
        with pytest.raises(http.client.RemoteDisconnected):
            exchange(endpoint, payload_bytes(request))
        assert process.wait(timeout=5) == 73
        with sqlite3.connect(root / "responses.sqlite3") as db:
            assert db.execute("SELECT state FROM requests").fetchone()[0] == "pending"
        with service(tmp_path, stub) as (_gateway, endpoint):
            status, raw = exchange(endpoint, payload_bytes(request))
            assert status == 409 and json.loads(raw)["error"] == "indeterminate"
            assert not stub.requests
    finally:
        if process.poll() is None:
            process.kill()
        _out, errors = process.communicate(timeout=5)
        assert not errors


def test_public_config_shapes_and_ownership(tmp_path, stub):
    nested = {"online_model": {
        "model": CONFIG["online_model"], "model_revision": CONFIG["online_model_revision"],
        "base_url": CONFIG["online_base_url"], "timeout_s": CONFIG["online_timeout_s"],
        "max_tokens": 8192, "temperature": 0.0, "provider": "bailian", "enabled": True,
    }}
    assert gateway_module.client_options(nested) == OPTIONS
    assert gateway_module.client_options(OPTIONS) == OPTIONS
    for unsafe in (
        {"api_key": KEY}, {"authorization": TOKEN},
        {"online_model": {"api_key": KEY}},
        {"base_url": "https://user:password@host/v1"},
        {"base_url": "https://host/v1?key=value"},
        {"model": "a", "online_model": "b"},
        {"max_tokens": True}, {"temperature": float("nan")},
    ):
        with pytest.raises(ValueError):
            gateway_module.client_options(unsafe)
    gateway = gateway_module.PerceptionGateway(CONFIG, tmp_path / "cache", client_factory=stub.client)
    try:
        with pytest.raises(OSError):
            gateway_module.PerceptionGateway(CONFIG, tmp_path / "cache", client_factory=stub.client)
        assert (tmp_path / "cache").stat().st_mode & 0o777 == 0o700
        assert (tmp_path / "cache/responses.sqlite3").stat().st_mode & 0o777 == 0o600
    finally:
        gateway.close()


def test_cli_contract_help_has_frozen_defaults_without_model_calls():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/cluster_perception_gateway.py"), "--help"],
        capture_output=True, text=True, timeout=10, check=True)
    assert "--config" in result.stdout and "--root" in result.stdout
    assert "--max-concurrency" in result.stdout and "27035" in result.stdout
    assert not result.stderr
