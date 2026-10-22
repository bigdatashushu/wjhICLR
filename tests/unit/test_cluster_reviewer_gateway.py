"""Real threaded native transport checks; no GPU, deployment, or remote access."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import copy
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from skill3d.infra.m5_identity import content_sha256
from skill3d.segmentation.intern_binary_review import InternBinaryClient
from skill3d.segmentation.intern_decision_client import InternDecisionClient
from skill3d.segmentation.intern_review_pool import InternReviewPool

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/cluster_reviewer_gateway.py"
SPEC = importlib.util.spec_from_file_location("reviewer_gateway_test", SCRIPT)
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)
TOKEN = "unit-test-environment-only"
RAW = b'{ "images" : ["data:image/jpeg;base64,AA=="], "n":1e0, "text":"\\u0061" }\n'


def model_identity(index):
    return {
        "model": "internlm/Intern-Decision-4B", "model_revision": "a" * 40,
        "backend": "native-hf-model-package", "dtype": "bfloat16",
        "implementation_sha256": "b" * 64, "service_sha256": "c" * 64,
        "files": {"inference.py": {"sha256": "b" * 64}},
        "runtime": {"torch": "pinned"}, "processor": {"pinned": True},
        "limits": {"concurrency": 1, "hard_inference_timeout_s": 120},
        "calibration": {"temperature": 1.99241824},
        "calibration_source": "DEFAULT_TEMPERATURE",
        "readout": "official compile_row",
        "device": {"name": "GPU", "visible_devices": str(index)},
    }


class NativeState:
    def __init__(self, index, totals):
        self.identity = model_identity(index)
        self.claimed_sha = None
        self.totals = totals
        self.active = self.maximum = 0
        self.requests = []
        self.delay = 0.02
        self.status = 200
        self.response = None
        self.response_headers = []
        self.health_status = 200
        self.health_override = None
        self.port = None
        self.started = threading.Event()
        self.release = None
        self.cleanup_delay = 0

    def health(self):
        if self.health_override is not None:
            return self.health_override
        return json.dumps({
            "identity": self.identity,
            "identity_sha256": self.claimed_sha or content_sha256(self.identity),
            "pid": 12345, "port": self.port, "host": "127.0.0.1", "cold_start_s": 1.2,
            "counters": {"requests": 7}, "opaque": ["keep", {"all": "fields"}],
        }, indent=1, ensure_ascii=False).encode() + b"\n"


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

        def respond(self, status, raw, headers=()):
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                for name, value in headers:
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(raw)
            except OSError:
                pass

        def serve(self):
            with state.totals["lock"]:
                state.active += 1
                state.maximum = max(state.maximum, state.active)
                state.totals["active"] += 1
                state.totals["maximum"] = max(state.totals["maximum"], state.totals["active"])
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                state.requests.append((self.command, self.path, body, dict(self.headers)))
                if self.command == "GET":
                    self.respond(state.health_status, state.health())
                else:
                    state.started.set()
                    if state.release is not None:
                        assert state.release.wait(timeout=5)
                    time.sleep(state.delay)
                    self.respond(state.status, body if state.response is None else state.response,
                                 state.response_headers)
            finally:
                time.sleep(state.cleanup_delay)
                with state.totals["lock"]:
                    state.active -= 1
                    state.totals["active"] -= 1

        do_GET = serve
        do_POST = serve

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.request_queue_size = 64
    state.port = server.server_address[1]
    return server


@pytest.fixture
def replicas():
    totals = {"lock": threading.Lock(), "active": 0, "maximum": 0}
    with ExitStack() as stack:
        states = [NativeState(index, totals) for index in range(4)]
        backends = []
        for state in states:
            endpoint = stack.enter_context(running_server(native_server(state)))
            backends.append({"endpoint": endpoint,
                             "expected_identity_sha256": content_sha256(state.identity)})
        yield states, backends, totals


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    monkeypatch.setenv(gateway.TOKEN_ENV, TOKEN)


@contextmanager
def central(backends, tmp_path, **kwargs):
    transport = gateway.ReviewerGateway(backends, lock_dir=tmp_path / "locks", **kwargs)
    try:
        server = gateway.make_server(transport, "127.0.0.1", 0)
        with running_server(server) as endpoint:
            yield transport, endpoint
    finally:
        transport.close()


@contextmanager
def local(endpoint, index, pin, **kwargs):
    transport = gateway.LoopbackProxy(endpoint, index, pin, **kwargs)
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


def test_eight_workers_four_replicas_globally_serialized_and_byte_exact(replicas, tmp_path):
    states, backends, totals = replicas
    with central(backends, tmp_path) as (_, endpoint), ExitStack() as stack:
        proxies = [stack.enter_context(local(endpoint, index, row["expected_identity_sha256"]))[1]
                   for index, row in enumerate(backends)]
        for state in states:
            state.delay = 0.04
            state.cleanup_delay = 0.005
        barrier = threading.Barrier(32)

        def worker(worker_id, index):
            payload = RAW + b" " * (worker_id + index)
            barrier.wait(timeout=10)
            status, body, _ = call(proxies[index], "/review", payload, token=None)
            assert status == 200
            assert body == payload
            return payload

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(worker, worker_id, index)
                       for worker_id in range(8) for index in range(4)]
            for future in futures:
                future.result(timeout=20)
        for index, state in enumerate(states):
            assert state.maximum == 1
            posts = [row for row in state.requests if row[0] == "POST"]
            assert len(posts) == 8 and all(row[1] == "/review" for row in posts)
            assert all("Authorization" not in row[3] for row in state.requests)
            expected = sorted(RAW + b" " * (worker_id + index) for worker_id in range(8))
            assert sorted(row[2] for row in posts) == expected
        assert 2 <= totals["maximum"] <= 4


def test_health_opaque_and_unchanged_native_clients_use_proxy(replicas, tmp_path):
    states, backends, _ = replicas
    with central(backends, tmp_path) as (transport, endpoint), ExitStack() as stack:
        addresses = []
        for index, row in enumerate(backends):
            _, address = stack.enter_context(local(endpoint, index, row["expected_identity_sha256"]))
            addresses.append(address)
            status, raw, _ = call(address, "/health", token=None)
            assert status == 200 and raw == states[index].health()
            status, raw, _ = call(address, "/decisions", RAW, token=None)
            assert status == 200 and raw == RAW
            client = InternDecisionClient(address, expected_identity_sha256=row["expected_identity_sha256"])
            assert client.identity() == states[index].identity
            binary = InternBinaryClient(address, expected_identity_sha256=row["expected_identity_sha256"])
            assert binary.identity() == states[index].identity
            assert binary._call("/decisions", {"arbitrary": ["opaque", index]}) == {
                "arbitrary": ["opaque", index]}
        pool = InternReviewPool([InternDecisionClient(
            address, expected_identity_sha256=row["expected_identity_sha256"])
            for address, row in zip(addresses, backends)])
        assert pool.identity()["method"] == {
            key: value for key, value in states[0].identity.items() if key != "device"}
        assert transport.semantic_identity_sha256 == content_sha256(pool.identity()["method"])


@pytest.mark.parametrize("token", [None, "", "incorrect"])
def test_auth_rejection_never_contacts_native(replicas, tmp_path, token, capsys):
    states, backends, _ = replicas
    with central(backends, tmp_path) as (_, endpoint):
        counts = [len(state.requests) for state in states]
        for path, body in [("/replicas/0/health", None), ("/replicas/0/review", RAW)]:
            status, raw, _ = call(endpoint, path, body, token=token)
            assert status == 401 and json.loads(raw)["error"] == "unauthorized"
        assert [len(state.requests) for state in states] == counts
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("status", [400, 404, 422, 429, 500, 503, 307])
def test_upstream_errors_are_byte_exact_with_safe_headers(replicas, tmp_path, status):
    states, backends, _ = replicas
    with central(backends, tmp_path) as (_, endpoint):
        with local(endpoint, 2, backends[2]["expected_identity_sha256"]) as (_, address):
            state = states[2]
            state.status = status
            state.response = b'{ "native_error": "opaque", "numbers":[1.00,2e0] }\n'
            state.response_headers = [
                ("Retry-After", "2"), ("Location", "http://example.invalid/steal"),
                ("Set-Cookie", "do-not-forward"), ("X-Secret", "do-not-forward"),
                ("WWW-Authenticate", "do-not-forward")]
            result, raw, headers = call(address, "/review", RAW, token=None)
            assert result == status and raw == state.response
            assert headers["Retry-After"] == "2"
            assert not {"Location", "Set-Cookie", "X-Secret", "WWW-Authenticate"} & headers.keys()
            assert len([row for row in state.requests if row[0] == "POST"]) == 1


@pytest.mark.parametrize("change", ["device", "model", "service_sha256", "limits", "claimed_sha",
                                   "invalid_json", "missing_identity", "duplicate"])
def test_changed_health_fail_closed_before_post(replicas, tmp_path, change):
    states, backends, _ = replicas
    with central(backends, tmp_path) as (_, endpoint):
        state = states[1]
        if change == "device":
            state.identity["device"]["visible_devices"] = "9"
        elif change == "model":
            state.identity["model"] = "changed"
        elif change == "service_sha256":
            state.identity["service_sha256"] = "e" * 64
        elif change == "limits":
            state.identity["limits"]["concurrency"] = 2
        elif change == "claimed_sha":
            state.claimed_sha = "f" * 64
        elif change == "invalid_json":
            state.health_override = b"{bad"
        elif change == "missing_identity":
            state.health_override = b'{"pid":123}'
        elif change == "duplicate":
            state.health_override = state.health().rstrip()[:-1] + b',"identity":{}}'
        for body, path in [(None, "/replicas/1/health"), (RAW, "/replicas/1/decisions")]:
            status, raw, _ = call(endpoint, path, body)
            assert status == 502 and json.loads(raw)["error"] == "reviewer_health_identity_mismatch"
        assert not [row for row in state.requests if row[0] == "POST"]


def test_shared_identity_excludes_only_device_and_can_be_explicitly_pinned(replicas, tmp_path):
    states, backends, _ = replicas
    semantic = content_sha256({key: value for key, value in states[0].identity.items() if key != "device"})
    with central(backends, tmp_path, semantic_identity_sha256=semantic) as (transport, _):
        assert transport.semantic_identity_sha256 == semantic
    altered = copy.deepcopy(states[3].identity)
    altered["runtime"]["torch"] = "different"
    states[3].identity = altered
    backends[3]["expected_identity_sha256"] = content_sha256(altered)
    with pytest.raises(gateway.GatewayError, match="identity_mismatch"):
        with central(backends, tmp_path):
            pass
    states[3].identity = model_identity(3)
    backends[3]["expected_identity_sha256"] = content_sha256(states[3].identity)
    with pytest.raises(gateway.GatewayError, match="identity_mismatch"):
        with central(backends, tmp_path, semantic_identity_sha256="f" * 64):
            pass


def test_proxy_pin_mismatch_blocks_startup_and_requests(replicas, tmp_path):
    states, backends, _ = replicas
    with central(backends, tmp_path) as (_, endpoint):
        with pytest.raises(gateway.GatewayError, match="identity_mismatch"):
            with local(endpoint, 0, backends[1]["expected_identity_sha256"]):
                pass
        with local(endpoint, 0, backends[0]["expected_identity_sha256"]) as (proxy, address):
            proxy.expected_identity_sha256 = "f" * 64
            status, raw, _ = call(address, "/decisions", RAW, token=None)
            assert status == 502 and json.loads(raw)["error"] == "reviewer_health_identity_mismatch"
            assert not [row for row in states[0].requests if row[0] == "POST"]


@pytest.mark.parametrize("path,method", [
    ("/health", "GET"), ("/replicas/4/health", "GET"), ("/replicas/00/health", "GET"),
    ("/replicas/0/health?x=1", "GET"), ("/replicas/0/../review", "POST"),
    ("/replicas/0/%2e%2e/review", "POST"), ("/replicas/0//review", "POST"),
    ("/replicas/0/review", "GET"), ("/replicas/0/http://host/review", "POST"),
])
def test_explicit_route_rejects_rewriting_and_escape(replicas, tmp_path, path, method):
    states, backends, _ = replicas
    with central(backends, tmp_path) as (_, endpoint):
        counts = [len(state.requests) for state in states]
        status, _, _ = call(endpoint, path, RAW if method == "POST" else None)
        assert status == 404
        assert counts == [len(state.requests) for state in states]


def test_request_and_response_limits_and_opaque_invalid_json(replicas, tmp_path):
    states, backends, _ = replicas
    with central(backends, tmp_path, max_request_bytes=16) as (_, endpoint):
        status, raw, _ = call(endpoint, "/replicas/0/review", RAW)
        assert status == 413 and json.loads(raw)["error"] == "reviewer_request_limit"
        assert not [row for row in states[0].requests if row[0] == "POST"]
        status, raw, _ = call(endpoint, "/replicas/0/review", b"{native handles}")
        assert status == 200 and raw == b"{native handles}"
        for headers, expected in [({"Content-Type": "text/plain"}, 415),
                                  ({"Transfer-Encoding": "chunked"}, 400)]:
            status, _, _ = call(endpoint, "/replicas/0/review", b"{}", headers=headers)
            assert status == expected
    with central(backends, tmp_path, max_response_bytes=2048) as (transport, endpoint):
        states[0].response = b"x" * 2049
        status, raw, _ = call(endpoint, "/replicas/0/review", b"{}")
        assert status == 502 and json.loads(raw)["error"] == "reviewer_response_limit"
        assert transport.replicas[0].quarantined


def test_fifo_queue_capacity_and_deadline_are_retryable(replicas, tmp_path):
    states, backends, _ = replicas
    states[0].release = threading.Event()
    with central(backends, tmp_path, max_pending=1, deadline_s=2) as (transport, endpoint):
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(call, endpoint, "/replicas/0/review", b'{"id":1}')
            assert states[0].started.wait(timeout=2)
            second = executor.submit(call, endpoint, "/replicas/0/review", b'{"id":2}',
                                     headers={"X-Reviewer-Budget-S": "0.15"})
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                with transport.replicas[0].condition:
                    if len(transport.replicas[0].queue) == 2:
                        break
                time.sleep(0.005)
            status, raw, headers = call(endpoint, "/replicas/0/review", b'{"id":3}')
            assert status == 429 and json.loads(raw)["retryable"] is True
            assert headers["Retry-After"] == "1"
            status, raw, headers = second.result(timeout=2)
            assert status == 429 and json.loads(raw)["retryable"] is True
            assert headers["Retry-After"] == "1"
            states[0].release.set()
            assert first.result(timeout=2)[0] == 200
            assert call(endpoint, "/replicas/0/review", b'{"id":4}')[0] == 200
    assert [row[2] for row in states[0].requests if row[0] == "POST"] == [b'{"id":1}', b'{"id":4}']
    assert states[0].maximum == 1


def test_timed_out_post_quarantines_without_backend_overlap(replicas, tmp_path):
    states, backends, _ = replicas
    states[0].delay = 0.4
    with central(backends, tmp_path, deadline_s=0.15) as (transport, endpoint):
        began = time.monotonic()
        status, raw, _ = call(endpoint, "/replicas/0/review", b"{}")
        assert status in {502, 504}
        assert time.monotonic() - began < 0.35
        assert json.loads(raw)["error"].startswith("reviewer_")
        assert transport.replicas[0].quarantined
        status, raw, _ = call(endpoint, "/replicas/0/review", b"{}")
        assert status == 503 and json.loads(raw)["error"] == "reviewer_replica_quarantined"
        assert call(endpoint, "/replicas/0/health")[0] == 503
        # Finishing native work does not automatically clear the quarantine.
        time.sleep(0.45)
        assert call(endpoint, "/replicas/0/health")[0] == 503
        assert states[0].maximum == 1
        assert len([row for row in states[0].requests if row[0] == "POST"]) == 1


def test_connection_capacity_reply_has_correct_framing(replicas, tmp_path):
    _, backends, _ = replicas
    transport = gateway.ReviewerGateway(backends, lock_dir=tmp_path / "locks")
    try:
        with running_server(gateway.make_server(
                transport, "127.0.0.1", 0, max_connections=1)) as endpoint:
            assert call(endpoint, "/replicas/0/health")[0] == 200
            with socket.create_connection(("127.0.0.1", gateway.urlsplit(endpoint).port)) as idle:
                idle.sendall(b"GET ")
                time.sleep(0.03)
                status, raw, headers = call(endpoint, "/replicas/0/health")
                assert status == 429 and json.loads(raw)["retryable"] is True
                assert int(headers["Content-Length"]) == len(raw)
                assert headers["Retry-After"] == "1"
    finally:
        transport.close()


def test_reply_strips_connection_nominated_headers_and_rejects_incomplete_body(replicas, tmp_path):
    reply = gateway.Reply(422, b"{}", [
        ("Connection", "Retry-After, X-Secret"), ("Retry-After", "1"),
        ("Content-Type", "application/json"), ("X-Secret", "no")])
    assert reply.headers == (("Content-Type", "application/json"),)
    states, backends, _ = replicas
    with central(backends, tmp_path) as (_, endpoint):
        # Simulate a native handler truncating a declared response.
        original_exchange = gateway._exchange

        class TruncatedHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "10")
                self.end_headers()
                self.wfile.write(b"{}")

        with running_server(ThreadingHTTPServer(("127.0.0.1", 0), TruncatedHandler)) as truncated:
            with pytest.raises(gateway.GatewayError, match="incomplete_response"):
                original_exchange(truncated, "GET", "/health", None,
                                  time.monotonic() + 1, 1024)
        assert call(endpoint, "/replicas/0/health")[1] == states[0].health()


def test_exclusive_gateway_ownership_across_processes(replicas, tmp_path):
    _, backends, _ = replicas
    with central(backends, tmp_path):
        code = (
            "import sys; sys.path.insert(0,sys.argv[1]); "
            "from cluster_reviewer_gateway import ReviewerGateway; import json; "
            "ReviewerGateway(json.loads(sys.argv[2]),lock_dir=sys.argv[3])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(ROOT / "scripts"), json.dumps(backends),
             str(tmp_path / "locks")], capture_output=True, timeout=5)
        assert result.returncode != 0 and b"already owned" in result.stderr
        assert TOKEN.encode() not in result.stdout + result.stderr
    with central(backends, tmp_path):
        pass


@pytest.mark.parametrize("url", ["http://10.0.0.1:27145", "https://127.0.0.1:27145",
                                 "http://u:password@127.0.0.1:27145",
                                 "http://127.0.0.1:27145/review", "http://127.0.0.1:27145?"])
def test_backend_targets_are_fixed_credential_free_loopback(replicas, tmp_path, url):
    _, backends, _ = replicas
    backends[0]["endpoint"] = url
    with pytest.raises(ValueError, match="transport URL"):
        gateway.ReviewerGateway(backends, lock_dir=tmp_path / "locks")


def test_environment_credentials_only_and_loopback_binding(replicas, tmp_path, monkeypatch):
    _, backends, _ = replicas
    monkeypatch.delenv(gateway.TOKEN_ENV)
    with pytest.raises(ValueError, match=gateway.TOKEN_ENV):
        gateway.ReviewerGateway(backends, lock_dir=tmp_path / "locks")
    with pytest.raises(ValueError, match=gateway.TOKEN_ENV):
        gateway.LoopbackProxy("http://service:27034", 0, "a" * 64)
    monkeypatch.setenv(gateway.TOKEN_ENV, TOKEN)
    proxy = gateway.LoopbackProxy("http://service:27034", 0, "a" * 64)
    with pytest.raises(ValueError, match="bind to loopback"):
        gateway.make_server(proxy, "0.0.0.0", 0)
    for deadline in [0, 180, float("nan")]:
        with pytest.raises(ValueError, match="deadline_s"):
            gateway.LoopbackProxy("http://service:27034", 0, "a" * 64, deadline_s=deadline)
    for url in ["http://token@service:27034", "http://service:27034?token=secret"]:
        with pytest.raises(ValueError, match="transport URL"):
            gateway.LoopbackProxy(url, 0, "a" * 64)


def test_cli_serve_and_localproxy_ready_and_no_credentials_in_output(replicas, tmp_path):
    _, backends, _ = replicas
    config = tmp_path / "backends.json"
    config.write_text(json.dumps({"backends": backends}))
    processes = []

    def start(arguments):
        process = subprocess.Popen([sys.executable, str(SCRIPT), *arguments],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append(process)
        ready = json.loads(process.stdout.readline())
        assert ready["status"] == "ready"
        assert TOKEN not in json.dumps(ready)
        return ready

    try:
        ready = start(["serve", "--backends-config", str(config), "--host", "127.0.0.1",
                       "--port", "0", "--lock-dir", str(tmp_path / "locks")])
        endpoint = f"http://127.0.0.1:{ready['port']}"
        proxy = start(["localproxy", "--gateway", endpoint, "--replica", "3", "--port", "0",
                       "--expected-identity-sha256", backends[3]["expected_identity_sha256"]])
        assert call(f"http://127.0.0.1:{proxy['port']}", "/review", RAW, token=None)[:2] == (200, RAW)
    finally:
        for process in reversed(processes):
            process.terminate()
            output, error = process.communicate(timeout=5)
            assert TOKEN.encode() not in output + error
            assert not error
