"""Real HTTP transport regressions using stubs, never live detector inference."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from skill3d.infra import m5_identity as identity
from skill3d.segmentation import open_vocab_detector

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/cluster_detector_gateway.py"
SPEC = importlib.util.spec_from_file_location("detector_gateway_test", SCRIPT)
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)
TOKEN = "detector-unit-test-environment-only"
RAW = b'{ "image":"AA==", "text_prompt":"\\u0061.", "box_threshold":2.5e-1 }\n'
NATIVE_RAW = (b'{ "success":true, "detections":[{"bbox":[1.0,2e0,8,9],'
              b'"label":"chair","confidence":0.7,"id":0,"normalizedbox":[0.1,0.2,0.3,0.4]}],'
              b'"annotated_image":"AA==","shape":[12,12,3],"image_size":[12,12] }\n')
EMPTY_RAW = b'{ "success":true, "detections":[],"annotated_image":"AA==" }\n'
MODEL = {"model": "GroundingDINO_SwinB", "files": {"weights": {"sha256": "a" * 64}},
         "bert_model": "bert-base-uncased", "mode": "eval", "annotate_fallback": "0"}
PIN = identity.content_sha256(MODEL)


class NativeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = self.maximum = 0
        self.requests = []
        self.delay = 0.01
        self.cleanup_delay = 0
        self.status = self.health_status = 200
        self.response = NATIVE_RAW
        self.response_headers = []
        self.health = b'{ "success":true, "model":"GroundingDINO", "opaque":[1.00,2e0] }\n'
        self.started = threading.Event()
        self.release = None
        self.process = {"pid": 12345, "start_ticks": "678", "boot_id": "test-boot"}
        self.command = b"python\0/frozen/scripts/serve_groundingdino.py\0--port\0" + b"20025\0"
        self.listening = True
        self.cwd = None
        self.verifications = []


@contextmanager
def running_server(server):
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        assert not thread.is_alive()


def native_server(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def serve(self):
            with state.lock:
                state.active += 1
                state.maximum = max(state.maximum, state.active)
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                state.requests.append((self.command, self.path, body, dict(self.headers)))
                if self.command == "GET":
                    status, raw = state.health_status, state.health
                else:
                    state.started.set()
                    if state.release is not None:
                        state.release.wait(timeout=5)
                    time.sleep(state.delay)
                    status, raw = state.status, body if state.response is None else state.response
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                for name, value in state.response_headers:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(raw)
            except OSError:
                pass
            finally:
                # The bytes have reached the client, but the handler still owns native work.
                time.sleep(state.cleanup_delay)
                with state.lock:
                    state.active -= 1

        do_GET = serve
        do_POST = serve

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.request_queue_size = 64
    return server


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv(gateway.TOKEN_ENV, TOKEN)


@pytest.fixture
def native(tmp_path, monkeypatch):
    state = NativeState()
    root = tmp_path / "service"
    root.mkdir()
    state.cwd = str(root)
    asset = root / "checkpoint"
    asset.write_bytes(b"original weights")
    source = root / "inference.py"
    source.write_bytes(b"original source")
    receipt = tmp_path / "groundingdino_startup.json"
    original_read = Path.read_bytes
    original_link = identity.os.readlink
    monkeypatch.setattr(identity, "process_identity", lambda pid: dict(state.process))
    monkeypatch.setattr(identity, "_listens_on",
                        lambda pid, port: state.listening and port == state.port)
    monkeypatch.setattr(Path, "read_bytes", lambda path: (
        state.command if str(path) == "/proc/12345/cmdline" else original_read(path)))
    monkeypatch.setattr(identity.os, "readlink", lambda path, *args, **kwargs: (
        state.cwd if str(path) == "/proc/12345/cwd" else original_link(path, *args, **kwargs)))
    original_verifier = identity.detector_identity

    def verify(endpoint, path):
        state.verifications.append((endpoint, path))
        return original_verifier(endpoint, path)

    monkeypatch.setattr(identity, "detector_identity", verify)
    with running_server(native_server(state)) as endpoint:
        state.port = gateway.urlsplit(endpoint).port
        state.endpoint, state.receipt = endpoint, receipt
        state.asset, state.source = asset, source
        receipt.write_text(json.dumps({
            "process": state.process, "port": state.port, "status": "starting_service",
            "service_root": str(root), "generation_identity": MODEL,
            "identity_files": [{"path": str(path), "identity": identity.file_identity(path)}
                               for path in (asset, source)],
        }))
        yield state


@contextmanager
def central(native, tmp_path, *, pin=PIN, **kwargs):
    transport = gateway.DetectorGateway(native.endpoint, native.receipt, ROOT, pin,
                                        lock_dir=tmp_path / "locks", **kwargs)
    try:
        with running_server(gateway.make_server(transport, "127.0.0.1", 0)) as endpoint:
            yield transport, endpoint
    finally:
        transport.close()


@contextmanager
def local(endpoint, *, pin=PIN, **kwargs):
    transport = gateway.LoopbackProxy(endpoint, pin, **kwargs)
    try:
        with running_server(gateway.make_server(transport, "127.0.0.1", 0)) as address:
            yield transport, address
    finally:
        transport.close()


def call(endpoint, path, body=None, *, token=TOKEN, method=None, headers=None):
    parsed = gateway.urlsplit(endpoint)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    supplied = {"Content-Type": "application/json"}
    if token is not None:
        supplied["Authorization"] = "Bearer " + token
    supplied.update(headers or {})
    try:
        connection.request(method or ("POST" if body is not None else "GET"),
                           path, body=body, headers=supplied)
        response = connection.getresponse()
        return response.status, response.read(), dict(response.getheaders())
    finally:
        connection.close()


def posts(state):
    return [row for row in state.requests if row[0] == "POST"]


def wait_queue(transport, size):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with transport.condition:
            if len(transport.queue) == size:
                return
        time.sleep(0.005)
    pytest.fail("request did not enter the FIFO queue")


def test_public_identity_health_and_plain_proxy_preserve_bytes(native, tmp_path):
    with central(native, tmp_path) as (transport, endpoint):
        status, raw, headers = call(endpoint, "/identity", token=None)
        assert status == 200
        assert json.loads(raw) == {"enabled": True, "model": MODEL, "model_sha256": PIN}
        assert headers[gateway.MODEL_HEADER] == PIN
        assert TOKEN.encode() not in raw and str(native.receipt).encode() not in raw
        assert gateway.digest(MODEL) == PIN
        with local(endpoint) as (_, address):
            status, health, headers = call(address, "/health", token=None)
            assert (status, health) == (200, native.health)
            assert headers[gateway.MODEL_HEADER] == PIN
            before = len(native.verifications)
            status, raw, headers = call(address, "/infer", RAW, token=None)
            assert (status, raw) == (200, NATIVE_RAW)
            assert headers[gateway.MODEL_HEADER] == PIN
            assert len(native.verifications) == before + 1
            assert native.verifications[-1] == (transport.backend, native.receipt)
            assert all("Authorization" not in row[3] for row in native.requests)
            assert posts(native)[0][2] == RAW


def test_unchanged_open_vocab_detector_client_uses_plain_proxy(native, tmp_path):
    native.response = (b'{ "success":true, "detections":[{"label":"chair",'
                       b'"bbox":[1,2,8,9],"confidence":0.7}] }\n')
    with central(native, tmp_path) as (_, endpoint), local(endpoint) as (_, address):
        result = open_vocab_detector.detect(np.zeros((12, 12, 3), dtype=np.uint8), "chair.",
                                            endpoint=address, timeout_s=180, total_timeout_s=180,
                                            max_attempts=1)
        assert len(result) == 1
        assert result[0].label == "chair" and result[0].bbox_xyxy == (1, 2, 8, 9)
        assert not open_vocab_detector.detect.last_error
        request = json.loads(posts(native)[0][2])
        assert request["text_prompt"] == "chair." and request["box_threshold"] == 0.25
        assert "Authorization" not in posts(native)[0][3]


@pytest.mark.parametrize("token", [None, "", "incorrect"])
def test_auth_rejection_never_contacts_backend_or_verifier(native, tmp_path, token, capsys):
    with central(native, tmp_path) as (_, endpoint):
        before = (len(native.requests), len(native.verifications))
        status, raw, _ = call(endpoint, "/infer", RAW, token=token)
        assert status == 401 and json.loads(raw)["error"] == "unauthorized"
        assert (len(native.requests), len(native.verifications)) == before
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_eight_workers_share_one_backend_through_post_response_cleanup(native, tmp_path):
    native.delay = 0.015
    native.cleanup_delay = 0.025
    with central(native, tmp_path) as (_, endpoint), ExitStack() as stack:
        addresses = [stack.enter_context(local(endpoint))[1] for _ in range(2)]
        barrier = threading.Barrier(8)

        def worker(index):
            payload = RAW + b" " * index
            barrier.wait(timeout=5)
            assert call(addresses[index % 2], "/infer", payload, token=None)[:2] == (200, NATIVE_RAW)
            return payload

        with ThreadPoolExecutor(max_workers=8) as executor:
            payloads = list(executor.map(worker, range(8)))
        assert sorted(row[2] for row in posts(native)) == sorted(payloads)
        assert native.maximum == 1 and native.active == 0


def test_fifo_order_is_arrival_order_across_http_workers(native, tmp_path):
    native.release = threading.Event()
    native.cleanup_delay = 0.025
    with central(native, tmp_path) as (transport, endpoint):
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(call, endpoint, "/infer", b'{"id":0}')]
            try:
                assert native.started.wait(timeout=2)
                for index in range(1, 5):
                    futures.append(executor.submit(call, endpoint, "/infer",
                                                   ('{"id":%d}' % index).encode()))
                    wait_queue(transport, index + 1)
            finally:
                native.release.set()
            assert all(future.result(timeout=5)[0] == 200 for future in futures)
    assert [json.loads(row[2])["id"] for row in posts(native)] == list(range(5))
    assert native.maximum == 1


@pytest.mark.parametrize("change", ["pid", "start_ticks", "boot_id", "listener", "command", "cwd",
                                   "checkpoint", "source", "receipt", "receipt_model",
                                   "receipt_file_list"])
def test_process_receipt_or_source_tamper_blocks_every_dispatch(native, tmp_path, change):
    with central(native, tmp_path) as (_, endpoint):
        if change in {"pid", "start_ticks", "boot_id"}:
            native.process[change] = 54321 if change == "pid" else "changed"
        elif change == "listener":
            native.listening = False
        elif change == "command":
            native.command = b"python\0different_service.py\0"
        elif change == "cwd":
            native.cwd = str(tmp_path)
        elif change in {"checkpoint", "source"}:
            path = native.asset if change == "checkpoint" else native.source
            old_mtime = path.stat().st_mtime_ns
            path.write_bytes(b"x" * path.stat().st_size)
            identity.os.utime(path, ns=(old_mtime, old_mtime))
        elif change == "receipt":
            native.receipt.write_bytes(b"{invalid receipt")
        else:
            receipt = json.loads(native.receipt.read_text())
            if change == "receipt_model":
                receipt["generation_identity"]["mode"] = "train"
            else:
                receipt["identity_files"] = []
            native.receipt.write_text(json.dumps(receipt))
        for path, body in [("/infer", RAW), ("/identity", None), ("/health", None)]:
            status, raw, _ = call(endpoint, path, body)
            assert status == 503 and json.loads(raw)["error"] == "detector_backend_identity_mismatch"
            assert TOKEN.encode() not in raw
        assert not posts(native)


def test_identity_is_reverified_after_queue_admission(native, tmp_path):
    native.release = threading.Event()
    with central(native, tmp_path) as (transport, endpoint):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(call, endpoint, "/infer", RAW)
            assert native.started.wait(timeout=2)
            second = executor.submit(call, endpoint, "/infer", RAW + b" ")
            try:
                wait_queue(transport, 2)
                native.source.write_bytes(b"changed source")
            finally:
                native.release.set()
            assert first.result(timeout=5)[0] == 200
            assert second.result(timeout=5)[0] == 503
        assert len(posts(native)) == 1


def test_wrong_model_or_incomplete_identity_blocks_startup(native, tmp_path, monkeypatch):
    with pytest.raises(gateway.GatewayError, match="backend_identity_mismatch"):
        with central(native, tmp_path, pin="f" * 64):
            pass
    assert not native.requests
    for value in [None, {"enabled": False}, {"enabled": True, "model": {}},
                  {"enabled": 1, "model": MODEL}]:
        monkeypatch.setattr(identity, "detector_identity", lambda endpoint, receipt: value)
        with pytest.raises(gateway.GatewayError, match="backend_identity_mismatch"):
            with central(native, tmp_path):
                pass
    assert not native.requests


def test_proxy_model_pin_blocks_startup_and_each_inference(native, tmp_path, monkeypatch):
    with central(native, tmp_path) as (transport, endpoint):
        with pytest.raises(gateway.GatewayError, match="model_identity_mismatch"):
            with local(endpoint, pin="f" * 64):
                pass
        with local(endpoint) as (_, address):
            original = transport.forward

            def forged(method, path, body, deadline, content_type):
                reply = original(method, path, body, deadline, content_type)
                if path == "/infer":
                    return gateway.Reply(reply.status, reply.body,
                        [(name, "f" * 64 if name.lower() == gateway.MODEL_HEADER.lower() else value)
                         for name, value in reply.headers])
                return reply

            monkeypatch.setattr(transport, "forward", forged)
            status, raw, _ = call(address, "/infer", RAW, token=None)
            assert status == 502 and json.loads(raw)["error"] == "detector_model_identity_mismatch"
            assert len(posts(native)) == 1


@pytest.mark.parametrize("raw", [
    b"not json", b"[]", b"null", b"{}", b'{"enabled":false,"model":{},"model_sha256":"x"}',
    json.dumps({"enabled": True, "model": MODEL, "model_sha256": "f" * 64}).encode(),
    (json.dumps({"enabled": True, "model": MODEL, "model_sha256": PIN})[:-1]
     + ',"model":{}}').encode(),
    b'{"enabled":true,"model":{"number":NaN},"model_sha256":"' + PIN.encode() + b'"}',
])
def test_public_model_identity_validation_fails_closed(raw):
    with pytest.raises(gateway.GatewayError, match="model_identity_mismatch"):
        gateway.validate_identity(raw, PIN)


@pytest.mark.parametrize("status", [400, 404, 422, 429, 500, 503, 307])
def test_native_status_error_bytes_and_safe_headers_are_preserved(native, tmp_path, status):
    native.status = status
    native.response = b'{ "native_error":"unchanged", "numbers":[1.00,2e0] }\n'
    native.response_headers = [
        ("Retry-After", "2"), ("Location", "http://example.invalid"),
        ("Set-Cookie", "not-forwarded"), ("Authorization", "not-forwarded"),
        ("X-Secret", "not-forwarded"), (gateway.MODEL_HEADER, "untrusted-native-pin")]
    with central(native, tmp_path) as (transport, endpoint), local(endpoint) as (_, address):
        result, raw, headers = call(address, "/infer", RAW, token=None)
        assert (result, raw) == (status, native.response)
        assert headers["Retry-After"] == "2" and headers[gateway.MODEL_HEADER] == PIN
        assert not {"Location", "Set-Cookie", "Authorization", "X-Secret"} & headers.keys()
        assert not transport.quarantined and len(posts(native)) == 1


@pytest.mark.parametrize("path,method", [
    ("/infer?x=1", "POST"), ("/infer/", "POST"), ("/%69nfer", "POST"),
    ("/../infer", "POST"), ("//infer", "POST"), ("/health", "POST"),
    ("/identity", "POST"), ("/infer", "GET"), ("/unknown", "GET"),
])
def test_exact_routes_do_not_rewrite_or_escape(native, tmp_path, path, method):
    with central(native, tmp_path) as (_, endpoint):
        before = len(native.requests), len(native.verifications)
        assert call(endpoint, path, RAW if method == "POST" else None, method=method)[0] == 404
        assert (len(native.requests), len(native.verifications)) == before


def test_request_limits_and_opaque_native_json(native, tmp_path):
    with central(native, tmp_path, max_request_bytes=16) as (_, endpoint):
        assert call(endpoint, "/infer", RAW)[0] == 413
        assert not posts(native)
        assert call(endpoint, "/infer", b"{native handles}")[:2] == (200, NATIVE_RAW)
        assert posts(native)[0][2] == b"{native handles}"
        for headers, expected in [({"Content-Type": "text/plain"}, 415),
                                  ({"Transfer-Encoding": "chunked"}, 400),
                                  ({"Content-Length": "-1"}, 400)]:
            assert call(endpoint, "/infer", b"{}", headers=headers)[0] == expected
        assert call(endpoint, "/health", b"{}", method="GET")[0] == 400


def test_response_limit_quarantines_backend(native, tmp_path):
    with central(native, tmp_path, max_response_bytes=2048) as (transport, endpoint):
        native.response = b"x" * 2049
        status, raw, _ = call(endpoint, "/infer", RAW)
        assert status == 502 and json.loads(raw)["error"] == "detector_response_limit"
        assert transport.quarantined
        assert call(endpoint, "/infer", RAW)[0] == 503
        assert len(posts(native)) == 1


def test_fifo_capacity_and_queue_deadline_are_retryable(native, tmp_path):
    native.release = threading.Event()
    with central(native, tmp_path, max_pending=1, deadline_s=2) as (transport, endpoint):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(call, endpoint, "/infer", b'{"id":1}')
            assert native.started.wait(timeout=2)
            second = executor.submit(call, endpoint, "/infer", b'{"id":2}',
                                     headers={gateway.BUDGET_HEADER: "0.15"})
            try:
                wait_queue(transport, 2)
                status, raw, headers = call(endpoint, "/infer", b'{"id":3}')
                assert status == 429 and json.loads(raw)["retryable"] is True
                assert headers["Retry-After"] == "1"
                status, raw, headers = second.result(timeout=2)
                assert status == 429 and json.loads(raw)["error"] == "detector_queue_deadline"
                assert headers["Retry-After"] == "1"
            finally:
                native.release.set()
            assert first.result(timeout=3)[0] == 200
            assert call(endpoint, "/infer", b'{"id":4}')[0] == 200
            assert not transport.queue and not transport.quarantined
    assert [json.loads(row[2])["id"] for row in posts(native)] == [1, 4]
    assert native.maximum == 1


def test_timed_out_post_quarantines_without_overlap_even_after_health(native, tmp_path):
    native.delay = 0.4
    with central(native, tmp_path, deadline_s=0.15) as (transport, endpoint):
        began = time.monotonic()
        status, raw, _ = call(endpoint, "/infer", RAW)
        assert status in {502, 504} and time.monotonic() - began < 0.35
        assert json.loads(raw)["error"].startswith("detector_")
        assert transport.quarantined
        for path, body in [("/infer", RAW), ("/health", None), ("/identity", None)]:
            status, raw, _ = call(endpoint, path, body)
            assert status == 503 and json.loads(raw)["error"] == "detector_backend_quarantined"
        time.sleep(0.45)
        assert call(endpoint, "/infer", RAW)[0] == 503
        assert native.maximum == 1 and len(posts(native)) == 1


def test_exclusive_owner_covers_localhost_alias_and_other_processes(native, tmp_path):
    with central(native, tmp_path):
        with pytest.raises(ValueError, match="already owned"):
            gateway.DetectorGateway(native.endpoint.replace("127.0.0.1", "localhost"),
                                    native.receipt, ROOT, PIN, lock_dir=tmp_path / "locks")
        code = (
            "import sys; sys.path.insert(0,sys.argv[1]); "
            "from cluster_detector_gateway import DetectorGateway; "
            "DetectorGateway(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],lock_dir=sys.argv[6])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(ROOT / "scripts"), native.endpoint,
             str(native.receipt), str(ROOT), PIN, str(tmp_path / "locks")],
            capture_output=True, timeout=10)
        assert result.returncode != 0 and b"already owned" in result.stderr
        assert TOKEN.encode() not in result.stdout + result.stderr
    with central(native, tmp_path):
        pass


def test_connection_capacity_reply_has_correct_framing(native, tmp_path):
    transport = gateway.DetectorGateway(native.endpoint, native.receipt, ROOT, PIN,
                                        lock_dir=tmp_path / "locks")
    try:
        server = gateway.make_server(transport, "127.0.0.1", 0, max_connections=1)
        with running_server(server) as endpoint:
            with socket.create_connection(("127.0.0.1", gateway.urlsplit(endpoint).port)) as idle:
                idle.sendall(b"GET ")
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline and server.slots._value != 0:
                    time.sleep(0.005)
                status, raw, headers = call(endpoint, "/identity")
                assert status == 429 and json.loads(raw)["retryable"] is True
                assert int(headers["Content-Length"]) == len(raw) and headers["Retry-After"] == "1"
    finally:
        transport.close()


@pytest.mark.parametrize("kind", ["truncated", "extra", "trickled"])
def test_upstream_framing_and_total_deadline_are_bounded(kind):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            try:
                self.send_response(200)
                self.send_header("Content-Length", "10" if kind != "extra" else "2")
                self.end_headers()
                if kind == "trickled":
                    for _ in range(10):
                        self.wfile.write(b" ")
                        time.sleep(0.035)
                else:
                    self.wfile.write(b"{}extra" if kind == "extra" else b"{}")
            except OSError:
                pass

    with running_server(ThreadingHTTPServer(("127.0.0.1", 0), Handler)) as endpoint:
        started = time.monotonic()
        with pytest.raises(gateway.GatewayError):
            gateway._exchange(endpoint, "GET", "/health", None, started + 0.12, 1024)
        assert time.monotonic() - started < 0.3


@pytest.mark.parametrize("url", [
    "http://10.0.0.1:20025", "https://127.0.0.1:20025",
    "http://u:password@127.0.0.1:20025", "http://127.0.0.1:20025/infer",
    "http://127.0.0.1:20025?", "http://127.0.0.1",
])
def test_fixed_credential_free_loopback_backend(native, tmp_path, url):
    with pytest.raises(ValueError, match="transport URL"):
        gateway.DetectorGateway(url, native.receipt, ROOT, PIN, lock_dir=tmp_path / "locks")


def test_frozen_source_root_is_mandatory(native, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="source-root"):
        gateway.load_identity(tmp_path)
    monkeypatch.setattr(identity, "__file__", str(tmp_path / "wrong/m5_identity.py"))
    with pytest.raises(ValueError, match="different source-root"):
        gateway.load_identity(ROOT)


def test_default_source_uses_launcher_package_and_large_identity_is_supported(native, tmp_path):
    assert gateway.load_identity() is identity
    receipt = json.loads(native.receipt.read_text())
    model = {**MODEL, "frozen_source_manifest": "x" * (334 * 1024)}
    receipt["generation_identity"] = model
    native.receipt.write_text(json.dumps(receipt))
    pin = identity.content_sha256(model)
    transport = gateway.DetectorGateway(native.endpoint, native.receipt,
                                        expected_model_sha256=pin, lock_dir=tmp_path / "locks")
    try:
        with running_server(gateway.make_server(transport, "127.0.0.1", 0)) as endpoint:
            with local(endpoint, pin=pin) as (_, address):
                status, raw, _ = call(address, "/identity", token=None)
                assert status == 200 and len(raw) > 334 * 1024
                assert json.loads(raw)["model_sha256"] == pin
                assert call(address, "/infer", RAW, token=None)[:2] == (200, NATIVE_RAW)
    finally:
        transport.close()


def test_cold_startup_hashing_has_no_inference_deadline(native, tmp_path, monkeypatch):
    verifier = identity.detector_identity

    def cold_verify(endpoint, receipt):
        time.sleep(0.06)
        return verifier(endpoint, receipt)

    monkeypatch.setattr(identity, "detector_identity", cold_verify)
    with central(native, tmp_path, deadline_s=0.03):
        pass
    assert len(native.requests) == 1 and not posts(native)


def test_environment_only_credentials_loopback_binding_and_finite_limits(native, tmp_path, monkeypatch):
    monkeypatch.delenv(gateway.TOKEN_ENV)
    monkeypatch.setenv("HARNESS3D_REVIEWER_TOKEN", "not-a-detector-token")
    with pytest.raises(ValueError, match=gateway.TOKEN_ENV):
        gateway.DetectorGateway(native.endpoint, native.receipt, ROOT, PIN)
    with pytest.raises(ValueError, match=gateway.TOKEN_ENV):
        gateway.LoopbackProxy("http://service:27037", PIN)
    monkeypatch.setenv(gateway.TOKEN_ENV, TOKEN)
    proxy = gateway.LoopbackProxy("http://service:27037", PIN)
    with pytest.raises(ValueError, match="bind to loopback"):
        gateway.make_server(proxy, "0.0.0.0", 0)
    for value in [0, 180, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="deadline_s"):
            gateway.LoopbackProxy("http://service:27037", PIN, deadline_s=value)
    for url in ["http://token@service:27037", "http://service:27037?token=secret"]:
        with pytest.raises(ValueError, match="transport URL"):
            gateway.LoopbackProxy(url, PIN)
    for value in ["", "A" * 64, "a" * 63]:
        with pytest.raises(ValueError, match="SHA256"):
            gateway.LoopbackProxy("http://service:27037", value)
    assert proxy.deadline_s == 175
    transport = gateway.DetectorGateway(native.endpoint, native.receipt, ROOT, PIN,
                                        lock_dir=tmp_path / "locks")
    try:
        assert transport.deadline_s == 170
        assert transport.max_request_bytes == 32 * 1024 * 1024
        assert transport.max_response_bytes == 8 * 1024 * 1024
    finally:
        transport.close()


def test_cli_localproxy_is_stdlib_only_and_ready_output_is_public(native, tmp_path):
    with central(native, tmp_path) as (_, endpoint):
        process = subprocess.Popen(
            [sys.executable, "-S", str(SCRIPT), "localproxy", "--gateway", endpoint,
             "--expected-model-sha256", PIN, "--port", "0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            assert select.select([process.stdout], [], [], 10)[0], "CLI never became ready"
            ready_line = process.stdout.readline()
            ready = json.loads(ready_line)
            assert ready["status"] == "ready" and ready["model_sha256"] == PIN
            assert TOKEN.encode() not in ready_line
            address = f"http://127.0.0.1:{ready['port']}"
            assert call(address, "/infer", RAW, token=None)[:2] == (200, NATIVE_RAW)
        finally:
            process.send_signal(signal.SIGINT)
            output, error = process.communicate(timeout=10)
            assert process.returncode == 0
            assert TOKEN.encode() not in output + error and not error


def test_cli_startup_errors_do_not_log_credentials(native, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(gateway.TOKEN_ENV)
    with pytest.raises(SystemExit) as caught:
        gateway.main(["serve", "--backend", native.endpoint, "--receipt", str(native.receipt),
                      "--source-root", str(ROOT), "--expected-model-sha256", PIN,
                      "--port", "0", "--lock-dir", str(tmp_path / "locks")])
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "detector startup failed" in captured.err
    assert TOKEN not in captured.err and str(native.receipt) not in captured.err


def test_cli_serve_accepts_launcher_pythonpath_without_source_root(native, tmp_path, monkeypatch, capsys):
    make_server = gateway.make_server

    def ready_server(transport, host, port, **kwargs):
        server = make_server(transport, host, port, **kwargs)

        def stop(**kwargs):
            raise KeyboardInterrupt

        server.serve_forever = stop
        return server

    monkeypatch.setattr(gateway, "make_server", ready_server)
    gateway.main(["serve", "--backend", native.endpoint, "--receipt", str(native.receipt),
                  "--expected-model-sha256", PIN, "--host", "127.0.0.1", "--port", "0",
                  "--lock-dir", str(tmp_path / "locks")])
    captured = capsys.readouterr()
    ready = json.loads(captured.out)
    assert ready["status"] == "ready" and ready["model_sha256"] == PIN
    assert not captured.err and TOKEN not in captured.out
    assert len(native.requests) == 1 and not posts(native)


@pytest.fixture
def replicas(tmp_path, monkeypatch):
    """Independent HTTP natives with real file identity checks and per-process receipts."""
    states = []

    def verify(endpoint, receipt):
        state = next(item for item in states if item.endpoint == endpoint)
        state.verifications.append((endpoint, receipt))
        value = json.loads(Path(receipt).read_bytes())
        if (value["process"] != state.process or not state.listening
                or value["port"] != state.port
                or any(identity.file_identity(row["path"]) != row["identity"]
                       for row in value["identity_files"])):
            return None
        return {"enabled": True, "model": value["generation_identity"]}

    monkeypatch.setattr(identity, "detector_identity", verify)
    with ExitStack() as stack:
        for index in range(3):
            state = NativeState()
            state.process["pid"] += index
            state.endpoint = stack.enter_context(running_server(native_server(state)))
            state.port = gateway.urlsplit(state.endpoint).port
            state.asset = tmp_path / f"weights-{index}"
            state.asset.write_bytes(b"original weights")
            state.receipt = tmp_path / f"receipt-{index}.json"
            state.receipt.write_text(json.dumps({
                "process": state.process, "port": state.port, "status": "starting_service",
                "generation_identity": MODEL,
                "identity_files": [{"path": str(state.asset),
                                    "identity": identity.file_identity(state.asset)}],
            }))
            states.append(state)
        yield states


@contextmanager
def pooled(states, tmp_path, *, cache=True, pin=PIN, **kwargs):
    transport = gateway.DetectorPool(
        [{"backend": state.endpoint, "receipt": str(state.receipt)} for state in states],
        ROOT, pin, lock_dir=tmp_path / "locks",
        cache_dir=tmp_path / "cache" if cache else None, **kwargs)
    try:
        with running_server(gateway.make_server(transport, "127.0.0.1", 0)) as endpoint:
            yield transport, endpoint
    finally:
        transport.close()


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    pytest.fail("expected concurrent state was not reached")


@pytest.mark.parametrize("response", [NATIVE_RAW, EMPTY_RAW])
def test_pool_persistent_exact_cache_and_provenance(replicas, tmp_path, response):
    replicas[0].response = response
    first_source = identity.file_identity(replicas[0].receipt)["sha256"]
    request_sha = hashlib.sha256(RAW).hexdigest()
    response_sha = hashlib.sha256(response).hexdigest()
    with pooled(replicas, tmp_path) as (transport, endpoint), local(endpoint) as (_, address):
        before = len(replicas[0].verifications)
        miss = call(address, "/infer", RAW, token=None)
        hit = call(address, "/infer", RAW, token=None)
        assert miss[:2] == hit[:2] == (200, response)
        assert len(replicas[0].verifications) == before + 2
        assert len(posts(replicas[0])) == 1
        for reply, outcome in [(miss, "miss"), (hit, "hit")]:
            headers = reply[2]
            assert headers[gateway.CACHE_HEADER] == outcome
            assert headers[gateway.REQUEST_HEADER] == request_sha
            assert headers[gateway.RESPONSE_HEADER] == response_sha
            assert headers[gateway.SOURCE_HEADER] == headers[gateway.SERVING_HEADER] == first_source
            assert headers[gateway.MODEL_HEADER] == PIN
            assert all(float(headers[header]) >= 0 for header in gateway.TIMING_HEADERS.values())
        assert float(hit[2][gateway.TIMING_HEADERS["native_infer_ms"]]) == 0
        assert transport.metrics.snapshot()["model_calls"] == 1
        assert transport.metrics.snapshot()["logical_requests"] == 2
    # New pool owner may serve persisted bytes on another replica, retaining the original source.
    with pooled(list(reversed(replicas)), tmp_path) as (transport, endpoint):
        status, raw, headers = call(endpoint, "/infer", RAW)
        assert (status, raw) == (200, response)
        assert headers[gateway.CACHE_HEADER] == "hit"
        assert headers[gateway.SOURCE_HEADER] == first_source
        assert headers[gateway.SERVING_HEADER] == identity.file_identity(replicas[2].receipt)["sha256"]
        assert transport.metrics.snapshot()["model_calls"] == 0
        assert sum(len(posts(state)) for state in replicas) == 1


def test_exact_key_separates_body_content_type_parameters_and_model(replicas, tmp_path):
    with pooled(replicas, tmp_path) as (_, endpoint):
        for body, content_type in [
            (RAW, "application/json"), (RAW + b" ", "application/json"),
            (RAW, "application/json; charset=utf-8"),
            (RAW.replace(b"2.5e-1", b"0.5"), "application/json"),
            (RAW.replace(b"\\u0061.", b"chair."), "application/json"),
        ]:
            result = call(endpoint, "/infer", body, headers={"Content-Type": content_type})
            assert result[0] == 200 and result[2][gateway.CACHE_HEADER] == "miss"
        assert call(endpoint, "/infer", RAW)[2][gateway.CACHE_HEADER] == "hit"
    assert sum(len(posts(state)) for state in replicas) == 5
    changed = {**MODEL, "mode": "different-frozen-source"}
    changed_pin = identity.content_sha256(changed)
    for state in replicas:
        receipt = json.loads(state.receipt.read_bytes())
        receipt["generation_identity"] = changed
        state.receipt.write_text(json.dumps(receipt))
    with pooled(replicas, tmp_path, pin=changed_pin) as (_, endpoint):
        assert call(endpoint, "/infer", RAW)[2][gateway.CACHE_HEADER] == "miss"
    assert sum(len(posts(state)) for state in replicas) == 6


@pytest.mark.parametrize("damage", ["bytes", "metadata", "schema"])
def test_corrupt_cache_entry_evicted_not_returned(replicas, tmp_path, damage):
    with pooled(replicas, tmp_path) as (transport, endpoint):
        assert call(endpoint, "/infer", RAW)[0] == 200
        with transport.cache.lock, transport.cache.connection:
            if damage == "bytes":
                transport.cache.connection.execute("UPDATE entries SET response=?", (b"garbled",))
            elif damage == "metadata":
                transport.cache.connection.execute("UPDATE entries SET metadata='not-json'")
            else:
                metadata = json.loads(transport.cache.connection.execute(
                    "SELECT metadata FROM entries").fetchone()[0])
                bad_raw = b'{"success":true,"detections":[{"label":"missing bbox"}]}'
                metadata["response_sha"] = hashlib.sha256(bad_raw).hexdigest()
                transport.cache.connection.execute("UPDATE entries SET metadata=?,response=?",
                                                   (json.dumps(metadata), bad_raw))
        result = call(endpoint, "/infer", RAW)
        assert result[:2] == (200, NATIVE_RAW)
        assert result[2][gateway.CACHE_HEADER] == "miss"
        assert len(posts(replicas[0])) == 2
        assert not transport.backends[0].quarantined


def test_corrupt_database_fails_startup_and_releases_all_backend_owners(replicas, tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()
    (directory / f"{gateway.CACHE_VERSION}.sqlite3").write_bytes(b"corrupt database")
    with pytest.raises(ValueError, match="cache unavailable"):
        with pooled(replicas, tmp_path):
            pass
    with pooled(replicas, tmp_path, cache=False):
        pass


def test_cache_lru_entry_and_byte_bounds(replicas, tmp_path):
    with pooled(replicas, tmp_path, cache_max_entries=2, cache_max_bytes=2000) as (transport, endpoint):
        for body in [RAW, RAW + b" ", RAW, RAW + b"  "]:
            assert call(endpoint, "/infer", body)[0] == 200
        count, size = transport.cache.connection.execute(
            "SELECT count(*),sum(size) FROM entries").fetchone()
        assert count == 2 and size <= 2000
        assert call(endpoint, "/infer", RAW)[2][gateway.CACHE_HEADER] == "hit"
        assert call(endpoint, "/infer", RAW + b" ")[2][gateway.CACHE_HEADER] == "miss"
        assert transport.cache.connection.execute("PRAGMA auto_vacuum").fetchone() == (1,)
    with pooled(replicas, tmp_path, cache_max_bytes=1) as (transport, endpoint):
        for _ in range(2):
            assert call(endpoint, "/infer", RAW)[2][gateway.CACHE_HEADER] == "miss"
        assert transport.cache.connection.execute("SELECT count(*) FROM entries").fetchone() == (0,)


@pytest.mark.parametrize("change", ["receipt", "model", "process", "asset"])
def test_identity_mutation_on_cache_hit_is_rejected_and_replica_quarantined(replicas, tmp_path, change):
    with pooled(replicas, tmp_path) as (transport, endpoint):
        assert call(endpoint, "/infer", RAW)[0] == 200
        state = replicas[0]
        if change == "receipt":
            state.receipt.write_bytes(state.receipt.read_bytes() + b" ")
        elif change == "model":
            value = json.loads(state.receipt.read_bytes())
            value["generation_identity"]["mode"] = "train"
            state.receipt.write_text(json.dumps(value))
        elif change == "process":
            state.process["start_ticks"] = "changed"
        else:
            state.asset.write_bytes(b"changed weights")
        result = call(endpoint, "/infer", RAW)
        assert result[0] == 503
        assert json.loads(result[1])["error"] == "detector_backend_identity_mismatch"
        assert transport.backends[0].quarantined and len(posts(state)) == 1
        # A separate logical request may use a healthy replica; the rejected request was not retried.
        result = call(endpoint, "/infer", RAW)
        assert result[0] == 200 and result[2][gateway.CACHE_HEADER] == "hit"
        assert sum(len(posts(item)) for item in replicas) == 1


def test_many_workers_singleflight_each_identity_check_and_metrics(replicas, tmp_path):
    replicas[0].release = threading.Event()
    with pooled(replicas, tmp_path, max_pending=32) as (transport, endpoint):
        before = sum(len(state.verifications) for state in replicas)
        with ThreadPoolExecutor(max_workers=24) as executor:
            first = executor.submit(call, endpoint, "/infer", RAW)
            assert replicas[0].started.wait(2)
            rest = [executor.submit(call, endpoint, "/infer", RAW) for _ in range(23)]
            try:
                wait_until(lambda: transport.waiters == 23)
            finally:
                replicas[0].release.set()
            replies = [future.result(timeout=5) for future in [first, *rest]]
        assert all(reply[:2] == (200, NATIVE_RAW) for reply in replies)
        assert [reply[2][gateway.CACHE_HEADER] for reply in replies].count("shared") == 23
        assert sum(len(posts(state)) for state in replicas) == 1
        assert sum(len(state.verifications) for state in replicas) == before + 24
        metrics = transport.metrics.snapshot()
        assert (metrics["logical_requests"], metrics["model_calls"], metrics["cache_hits"],
                metrics["shared_requests"], metrics["errors"]) == (24, 1, 23, 23, 0)
        assert not transport.flights and not transport.waiters


@pytest.mark.parametrize("status", [200, 400, 429, 500])
def test_singleflight_shares_failure_but_never_caches_errors(replicas, tmp_path, status):
    state = replicas[0]
    state.release = threading.Event()
    state.status, state.response = status, b'{"success":false,"error":"native failed"}'
    with pooled(replicas[:1], tmp_path) as (transport, endpoint):
        with ThreadPoolExecutor(max_workers=8) as executor:
            first = executor.submit(call, endpoint, "/infer", RAW)
            assert state.started.wait(2)
            rest = [executor.submit(call, endpoint, "/infer", RAW) for _ in range(7)]
            try:
                wait_until(lambda: transport.waiters == 7)
            finally:
                state.release.set()
            assert all(future.result(timeout=5)[:2] == (status, state.response)
                       for future in [first, *rest])
        assert len(posts(state)) == 1 and not transport.backends[0].quarantined
        assert transport.cache.connection.execute("SELECT count(*) FROM entries").fetchone() == (0,)
        assert transport.metrics.snapshot()["cache_hits"] == 0
        assert transport.metrics.snapshot()["errors"] == 8
        state.status, state.response = 200, NATIVE_RAW
        assert call(endpoint, "/infer", RAW)[2][gateway.CACHE_HEADER] == "miss"
        assert len(posts(state)) == 2


def test_native_invalid_boxes_failure_preserves_error_without_quarantining_model(replicas, tmp_path):
    state = replicas[0]
    state.response = b'{"success":false,"detections":[],"message":"no valid boxes"}'
    with pooled(replicas[:1], tmp_path) as (transport, endpoint):
        assert call(endpoint, "/infer", RAW)[:2] == (200, state.response)
        assert call(endpoint, "/infer", RAW)[:2] == (200, state.response)
        assert len(posts(state)) == 2
        assert not transport.backends[0].quarantined
        assert transport.cache.connection.execute("SELECT count(*) FROM entries").fetchone() == (0,)
        assert transport.metrics.snapshot()["cache_hits"] == 0
        assert transport.metrics.snapshot()["errors"] == 2


@pytest.mark.parametrize("raw", [
    b"not-json", b"[]", b'{"success":false,"detections":[]}',
    b'{"success":1,"detections":[]}', b'{"success":true}',
    b'{"success":true,"detections":{},"detections":[]}',
    b'{"success":true,"detections":[{}]}',
    b'{"success":true,"detections":[{"bbox":[1,2,3],"label":"x","confidence":0.5}]}',
    b'{"success":true,"detections":[{"bbox":[1,2,3,4],"label":2,"confidence":0.5}]}',
    b'{"success":true,"detections":[{"bbox":[1,2,3,4],"label":"","confidence":0.5}]}',
    b'{"success":true,"detections":[{"bbox":[1,2,3,4],"label":"x","confidence":true}]}',
    b'{"success":true,"detections":[{"bbox":[1,2,3,4],"label":"x","confidence":1.5}]}',
    b'{"success":true,"detections":[{"bbox":[1,2,3,4],"label":"x","confidence":NaN}]}',
    b'{"success":true,"detections":[{"bbox":[1,2,3,4],"label":"x","confidence":1e999}]}',
    b'{"success":true,"detections":[{"bbox":[9,2,3,4],"label":"x","confidence":0.5}]}',
])
def test_bad_success_schema_quarantines_without_retry_or_cache(replicas, tmp_path, raw):
    replicas[0].response = raw
    with pooled(replicas, tmp_path) as (transport, endpoint):
        result = call(endpoint, "/infer", RAW)
        assert result[0] == 502 and json.loads(result[1])["error"] == "detector_native_schema"
        assert transport.backends[0].quarantined
        assert sum(len(posts(state)) for state in replicas) == 1
        assert transport.cache.connection.execute("SELECT count(*) FROM entries").fetchone() == (0,)
        assert call(endpoint, "/infer", RAW)[2][gateway.CACHE_HEADER] == "miss"
        assert [len(posts(state)) for state in replicas] == [1, 1, 0]


def test_ambiguous_post_quarantine_shared_failure_independent_other_replica(replicas, tmp_path):
    replicas[0].delay = 0.3
    with pooled(replicas, tmp_path, deadline_s=2) as (transport, endpoint):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(call, endpoint, "/infer", RAW,
                                    headers={gateway.BUDGET_HEADER: "0.12"})
            assert replicas[0].started.wait(2)
            second = executor.submit(call, endpoint, "/infer", RAW)
            wait_until(lambda: transport.waiters == 1)
            replies = [first.result(timeout=3), second.result(timeout=3)]
        assert all(reply[0] in {502, 504} for reply in replies)
        assert replies[0][:2] == replies[1][:2]
        assert transport.backends[0].quarantined
        assert [len(posts(state)) for state in replicas] == [1, 0, 0]
        assert call(endpoint, "/infer", RAW)[0] == 200
        assert [len(posts(state)) for state in replicas] == [1, 1, 0]
        time.sleep(0.35)
        assert transport.backends[0].quarantined and replicas[0].maximum == 1


def test_pool_uncached_default_executes_identical_requests_each_time(replicas, tmp_path):
    with pooled(replicas, tmp_path, cache=False) as (transport, endpoint):
        with ThreadPoolExecutor(max_workers=12) as executor:
            results = list(executor.map(lambda _: call(endpoint, "/infer", RAW), range(12)))
        assert all(reply[0] == 200 and reply[2][gateway.CACHE_HEADER] == "disabled"
                   for reply in results)
        assert sum(len(posts(state)) for state in replicas) == 12
        assert transport.metrics.snapshot()["model_calls"] == 12
        assert all(state.maximum == 1 for state in replicas)


def test_true_parallel_native_fifo_and_no_per_replica_overlap_through_eof(replicas, tmp_path):
    for state in replicas:
        state.release = threading.Event()
        state.cleanup_delay = 0.03
    with pooled(replicas, tmp_path, cache=False) as (transport, endpoint):
        with ThreadPoolExecutor(max_workers=7) as executor:
            first = []
            for index, state in enumerate(replicas):
                first.append(executor.submit(call, endpoint, "/infer", ('{"id":%d}' % index).encode()))
                assert state.started.wait(2)
            assert all(state.active == 1 for state in replicas)
            rest = []
            for index in range(3, 7):
                rest.append(executor.submit(call, endpoint, "/infer", ('{"id":%d}' % index).encode()))
                wait_queue(transport, index - 2)
            order = []
            original = gateway._exchange
            # Record dispatch order centrally; response completion order need not be FIFO.
            def observed(endpoint, method, path, body, *args, **kwargs):
                if method == "POST":
                    order.append(json.loads(body)["id"])
                return original(endpoint, method, path, body, *args, **kwargs)

            try:
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(gateway, "_exchange", observed)
                    # Only one free replica makes FIFO reservation/dispatch observable.
                    replicas[0].release.set()
                    assert first[0].result(timeout=5)[0] == 200
                    assert all(future.result(timeout=5)[0] == 200 for future in rest)
            finally:
                for state in replicas:
                    state.release.set()
            assert order == [3, 4, 5, 6]
            assert all(future.result(timeout=5)[0] == 200 for future in first)
        assert not transport.queue and not transport.busy
        assert all(state.maximum == 1 and state.active == 0 for state in replicas)


def test_pool_serializes_identity_checks_but_keeps_three_native_exchanges(
        replicas, tmp_path, monkeypatch):
    verifier = identity.detector_identity
    verify_started, verify_release = threading.Event(), threading.Event()
    counts = {"active": 0, "maximum": 0, "calls": 0}
    lock = threading.Lock()

    def blocked_verifier(*args):
        with lock:
            counts["active"] += 1
            counts["calls"] += 1
            counts["maximum"] = max(counts["maximum"], counts["active"])
        try:
            verify_started.set()
            assert verify_release.wait(3)
            return verifier(*args)
        finally:
            with lock:
                counts["active"] -= 1

    for state in replicas:
        state.release = threading.Event()
    with pooled(replicas, tmp_path, cache=False) as (transport, endpoint):
        monkeypatch.setattr(identity, "detector_identity", blocked_verifier)
        with ThreadPoolExecutor(max_workers=3) as executor:
            requests = [executor.submit(call, endpoint, "/infer",
                                       ('{"id":%d}' % index).encode())
                        for index in range(3)]
            try:
                assert verify_started.wait(2)
                wait_until(lambda: len(transport.busy) == 3)
                assert counts["active"] == counts["calls"] == 1
                verify_release.set()
                for state in replicas:
                    assert state.started.wait(2)
                assert all(state.active == 1 for state in replicas)
            finally:
                verify_release.set()
                for state in replicas:
                    state.release.set()
            assert all(request.result(timeout=5)[0] == 200 for request in requests)
        assert counts == {"active": 0, "maximum": 1, "calls": 3}


def test_pool_identity_mutex_wait_is_bounded_without_dispatch_or_quarantine(
        replicas, tmp_path):
    with pooled(replicas, tmp_path, cache=False) as (transport, endpoint):
        before = sum(len(state.verifications) for state in replicas)
        with transport.identity_lock:
            result = call(endpoint, "/infer", RAW,
                          headers={gateway.BUDGET_HEADER: "0.10"})
        assert result[0] == 429
        assert json.loads(result[1])["error"] == "detector_queue_deadline"
        assert sum(len(state.verifications) for state in replicas) == before
        assert not any(posts(state) for state in replicas)
        assert not any(backend.quarantined for backend in transport.backends)
        assert not transport.busy and not transport.queue
        metrics = transport.metrics.snapshot()
        assert metrics["queue_wait_ms"] >= 50 and metrics.get("identity_ms", 0) == 0
        assert call(endpoint, "/infer", RAW)[0] == 200


def test_pool_pending_bounds_include_singleflight_waiters_and_deadlines(replicas, tmp_path):
    state = replicas[0]
    state.release = threading.Event()
    with pooled(replicas[:1], tmp_path, max_pending=1, deadline_s=2) as (transport, endpoint):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(call, endpoint, "/infer", RAW)
            assert state.started.wait(2)
            second = executor.submit(call, endpoint, "/infer", RAW,
                                     headers={gateway.BUDGET_HEADER: "0.15"})
            try:
                wait_until(lambda: transport.waiters == 1)
                for payload in [RAW, RAW + b" "]:
                    status, raw, headers = call(endpoint, "/infer", payload)
                    assert status == 429 and json.loads(raw)["error"] == "detector_capacity"
                    assert headers["Retry-After"] == "1"
                assert not first.done()
                result = second.result(timeout=2)
                assert result[0] == 429 and json.loads(result[1])["error"] == "detector_queue_deadline"
                assert not transport.waiters
            finally:
                state.release.set()
            assert first.result(timeout=3)[0] == 200
        assert len(posts(state)) == 1 and not transport.backends[0].quarantined
        assert call(endpoint, "/infer", RAW)[2][gateway.CACHE_HEADER] == "hit"


def test_pool_zero_pending_allows_only_free_slots_not_shared_waiters(replicas, tmp_path):
    replicas[0].release = threading.Event()
    with pooled(replicas[:1], tmp_path, max_pending=0) as (_, endpoint):
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(call, endpoint, "/infer", RAW)
            assert replicas[0].started.wait(2)
            try:
                assert call(endpoint, "/infer", RAW)[0] == 429
                assert call(endpoint, "/infer", RAW + b" ")[0] == 429
            finally:
                replicas[0].release.set()
            assert first.result(timeout=3)[0] == 200


def test_pool_ownership_config_failures_and_independent_process_requirement(replicas, tmp_path):
    rows = [{"backend": state.endpoint, "receipt": str(state.receipt)} for state in replicas]
    with central(replicas[1], tmp_path):
        with pytest.raises(ValueError, match="already owned"):
            gateway.DetectorPool(rows, ROOT, PIN, lock_dir=tmp_path / "locks")
        # Partial construction must release the earlier free replica's lock.
        with central(replicas[0], tmp_path):
            pass
    for invalid in [[], rows + [rows[0]], [{"backend": rows[0]["backend"]}],
                    [rows[0], {**rows[1], "receipt": rows[0]["receipt"]}]]:
        with pytest.raises(ValueError):
            gateway.DetectorPool(invalid, ROOT, PIN, lock_dir=tmp_path / "locks")
    replicas[1].process = dict(replicas[0].process)
    value = json.loads(replicas[1].receipt.read_bytes())
    value["process"] = replicas[1].process
    replicas[1].receipt.write_text(json.dumps(value))
    with pytest.raises(gateway.GatewayError, match="pool_process_identity"):
        with pooled(replicas, tmp_path, cache=False):
            pass
    with central(replicas[0], tmp_path):
        pass


def test_cache_exclusive_owner_across_processes_and_release(replicas, tmp_path):
    with pooled(replicas, tmp_path) as (transport, _):
        with pytest.raises(ValueError, match="already owned"):
            gateway.ExactResponseCache(tmp_path / "cache")
        code = ("import sys;sys.path.insert(0,sys.argv[1]);"
                "from cluster_detector_gateway import ExactResponseCache;"
                "ExactResponseCache(sys.argv[2])")
        result = subprocess.run([sys.executable, "-S", "-c", code, str(ROOT / "scripts"),
                                 str(tmp_path / "cache")], capture_output=True, timeout=10)
        assert result.returncode != 0 and b"already owned" in result.stderr
        assert TOKEN.encode() not in result.stdout + result.stderr
        assert transport.cache.connection.execute("PRAGMA quick_check").fetchone() == ("ok",)
    cache = gateway.ExactResponseCache(tmp_path / "cache")
    cache.close()


def test_cache_hit_still_requires_auth_and_telemetry_never_logs_payloads(replicas, tmp_path, capsys):
    with pooled(replicas, tmp_path, log_metrics=True) as (transport, endpoint):
        assert call(endpoint, "/infer", RAW)[0] == 200
        before = len(replicas[0].verifications)
        assert call(endpoint, "/infer", RAW, token=None)[0] == 401
        assert len(replicas[0].verifications) == before
        assert call(endpoint, "/infer", RAW)[0] == 200
        metrics = transport.metrics.snapshot()
        assert metrics["logical_requests"] == 2 and metrics["model_calls"] == 1
    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.out.splitlines()]
    assert len(records) == 2 and not captured.err
    assert sum(row.get("cache_hits", 0) for row in records) == 1
    assert all(row["event"] == "detector_request" for row in records)
    assert [row["model_calls"] for row in records] == [1, 0]
    assert records[1]["native_infer_ms"] == 0 and records[1]["cache_outcome"] == "hit"
    assert records[1]["response_sha256"] == hashlib.sha256(NATIVE_RAW).hexdigest()
    assert all(value not in captured.out for value in (
        TOKEN, "AA==", "annotated_image", "text_prompt", str(replicas[0].receipt)))


def test_cli_pool_config_limits_ready_and_cleanup(replicas, tmp_path, monkeypatch, capsys):
    config = tmp_path / "pool.json"
    config.write_text(json.dumps({"backends": [
        {"backend": state.endpoint, "receipt": str(state.receipt)} for state in replicas]}))
    make_server = gateway.make_server
    observed = []

    def ready_server(transport, host, port, **kwargs):
        observed.append(transport)
        server = make_server(transport, host, port, **kwargs)
        def stop(**kwargs):
            raise KeyboardInterrupt
        server.serve_forever = stop
        return server

    monkeypatch.setattr(gateway, "make_server", ready_server)
    gateway.main(["pool", "--backends-config", str(config), "--expected-model-sha256", PIN,
                  "--cache-dir", str(tmp_path / "cache"), "--cache-max-entries", "5",
                  "--cache-max-bytes", "2147483648", "--max-pending", "8",
                  "--lock-dir", str(tmp_path / "locks"), "--host", "127.0.0.1", "--port", "0"])
    ready = json.loads(capsys.readouterr().out)
    assert ready["mode"] == "pool" and ready["status"] == "ready" and ready["model_sha256"] == PIN
    assert observed[0].max_pending == 8 and observed[0].cache is None
    assert all(backend._ownership is None for backend in observed[0].backends)
    monkeypatch.setattr(gateway, "make_server", make_server)
    with pooled(replicas, tmp_path):
        pass
