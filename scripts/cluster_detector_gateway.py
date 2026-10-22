#!/usr/bin/env python3
"""FIFO transport for the already-running, owned GroundingDINO service.

serve --backend http://127.0.0.1:20025 --receipt /existing/groundingdino_startup.json
      [--source-root /frozen/repository] --expected-model-sha256 SHA --port 27037
localproxy --gateway http://SERVICE:27037 --expected-model-sha256 SHA --port 20027

Both modes read HARNESS3D_DETECTOR_TOKEN only from the environment. POST /infer
requires this bearer at the central service; the loopback-only proxy injects it
for unchanged native clients. GET /health preserves native bytes, with the public
pin in X-Detector-Model-Sha256. GET /identity publishes the verified model and its
canonical model_sha256, without receipt metadata.

Only serve lazily imports the frozen m5_identity implementation. Its unchanged
detector_identity verifies the receipt, process lifetime, listener, command line,
working directory and model/source files at startup and before every dispatch.
The startup receipt itself is also pinned. No model is loaded or restarted.
Without --source-root, the launcher's frozen PYTHONPATH selects m5_identity.
No fabricated local process receipt is provided for the proxy: native generation
identity remains incomplete, while the separately frozen method pins the model.

One central owner and FIFO queue cover all workers, including backend cleanup
through peer EOF. An ambiguous POST failure quarantines the backend until an
operator verifies/restarts it and restarts this gateway; health cannot prove idle.
All clients must use this gateway; direct native access bypasses serialization.
Use native detector timeout_s=180 (total_timeout_s=180) above the service/proxy
170s/175s deadlines. No transport retries or detection JSON rewriting occur here.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit

TOKEN_ENV = "HARNESS3D_DETECTOR_TOKEN"
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
BUDGET_HEADER = "X-Detector-Budget-S"
MODEL_HEADER = "X-Detector-Model-Sha256"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("model pin must be a lower-case SHA256")
    return value


def _url(value, *, backend=False):
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if (parsed.scheme not in ({"http"} if backend else {"http", "https"})
                or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment or "?" in value or "#" in value
                or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
                or backend and (parsed.hostname not in LOOPBACK or port is None)
                or port is not None and not 1 <= port <= 65535):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("invalid credential-free transport URL") from None
    host = "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname
    authority = f"[{host}]" if ":" in host else host
    return f"{parsed.scheme}://{authority}:{port or (443 if parsed.scheme == 'https' else 80)}"


def _token():
    value = os.environ.get(TOKEN_ENV, "")
    if not value or len(value) > 4096 or not re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", value):
        raise ValueError(f"{TOKEN_ENV} must contain a nonempty bearer token")
    return value


def load_identity(source_root=None):
    """Import exactly the requested frozen package, never the detector's package."""
    if source_root is None:
        return importlib.import_module("skill3d.infra.m5_identity")
    root = Path(source_root).expanduser().resolve()
    source = root / "src"
    expected = source / "skill3d/infra/m5_identity.py"
    if not expected.is_file():
        raise ValueError("source-root must contain src/skill3d/infra/m5_identity.py")
    sys.path.insert(0, str(source))
    module = importlib.import_module("skill3d.infra.m5_identity")
    if Path(module.__file__).resolve() != expected.resolve():
        raise ValueError("m5_identity was imported from a different source-root")
    return module


class GatewayError(Exception):
    def __init__(self, status, code, *, retryable=False):
        super().__init__(code)
        self.status, self.code, self.retryable = status, code, retryable


class Reply:
    def __init__(self, status, body, headers=()):
        self.status, self.body = status, body
        headers = tuple(headers)
        hop_headers = {part.strip().lower() for name, value in headers
                       if name.lower() == "connection" for part in value.split(",")}
        self.headers = tuple((name, value) for name, value in headers
                             if name.lower() in {
                                 "content-type", "retry-after", "content-encoding",
                                 MODEL_HEADER.lower()}
                             and name.lower() not in hop_headers
                             and len(value) <= 4096 and "\r" not in value and "\n" not in value)


class _Response(http.client.HTTPResponse):
    """Keep prefetched bytes available until the peer EOF check completes."""
    def _close_conn(self):
        if self.fp is not None:
            self._retained_stream = self.fp
            self.fp = None

    def close(self):
        try:
            super().close()
        finally:
            stream = getattr(self, "_retained_stream", None)
            if stream is not None:
                stream.close()
                self._retained_stream = None


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GatewayError(504, "detector_deadline")
    return remaining


@contextmanager
def _socket_deadline(sock, deadline):
    """Bound trickled bytes as well as socket idle time."""
    def abort():
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    timer = threading.Timer(_remaining(deadline), abort)
    timer.daemon = True
    timer.start()
    try:
        sock.settimeout(_remaining(deadline))
        yield
    finally:
        timer.cancel()


def _exchange(endpoint, method, path, body, deadline, max_response_bytes, *,
              token=None, content_type="application/json"):
    parsed = urlsplit(endpoint)
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = cls(parsed.hostname, parsed.port, timeout=_remaining(deadline))
    try:
        # No redirects, proxy environment variables, or upstream credential propagation.
        connection.connect()
        upstream_socket = connection.sock
        with _socket_deadline(upstream_socket, deadline):
            headers = {"Content-Type": content_type, "Connection": "close"}
            if token is not None:
                headers["Authorization"] = "Bearer " + token
                headers[BUDGET_HEADER] = str(_remaining(deadline))
            connection.request(method, path, body=body, headers=headers)
            # Retain socket ownership even for HTTP/1.0 responses.
            with _Response(upstream_socket, method=method) as response:
                response.begin()
                upstream_stream = response.fp
                raw = response.read(max_response_bytes + 1)
                if len(raw) > max_response_bytes:
                    raise GatewayError(502, "detector_response_limit")
                lengths = response.headers.get_all("Content-Length", [])
                if (len(lengths) > 1
                        or lengths and (not re.fullmatch(r"[0-9]{1,10}", lengths[0])
                                        or int(lengths[0]) != len(raw))):
                    raise GatewayError(502, "detector_incomplete_response")
                # Content-Length alone can precede native post-response cleanup.
                if upstream_stream.read(1):
                    raise GatewayError(502, "detector_extra_response_bytes")
                _remaining(deadline)
                return Reply(response.status, raw, response.getheaders())
    except (TimeoutError, socket.timeout):
        raise GatewayError(504, "detector_upstream_timeout") from None
    except (OSError, http.client.HTTPException):
        _remaining(deadline)
        raise GatewayError(502, "detector_upstream_transport") from None
    finally:
        connection.close()


def validate_identity(raw, expected_model_sha256):
    """Validate the public model pin without changing response bytes."""
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate identity field")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
        if (value["enabled"] is not True or not isinstance(value["model"], dict)
                or not value["model"] or value["model_sha256"] != digest(value["model"])
                or value["model_sha256"] != expected_model_sha256):
            raise ValueError
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        raise GatewayError(502, "detector_model_identity_mismatch") from None


def _deadline_setting(value, ceiling):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= ceiling:
        raise ValueError(f"deadline_s must be finite, positive, and <={ceiling}")
    return float(value)


def _byte_limit(value, ceiling):
    if type(value) is not int or not 0 < value <= ceiling:
        raise ValueError("invalid byte limit")
    return value


class DetectorGateway:
    """Single service-host owner with a FIFO admission lock for all native traffic."""
    authenticated = True

    def __init__(self, backend, receipt, source_root=None, expected_model_sha256=None, *,
                 deadline_s=170, max_pending=16, max_request_bytes=MAX_REQUEST_BYTES,
                 max_response_bytes=MAX_RESPONSE_BYTES, lock_dir=None):
        self.token = _token()
        self.backend = _url(backend, backend=True)
        self.receipt = Path(receipt).expanduser().resolve()
        self.expected_model_sha256 = _sha(expected_model_sha256)
        self.deadline_s = _deadline_setting(deadline_s, 170)
        self.max_request_bytes = _byte_limit(max_request_bytes, MAX_REQUEST_BYTES)
        self.max_response_bytes = _byte_limit(max_response_bytes, MAX_RESPONSE_BYTES)
        if type(max_pending) is not int or not 0 <= max_pending <= 64:
            raise ValueError("max_pending must be between 0 and 64")
        self.max_pending = max_pending
        self.condition = threading.Condition()
        self.queue = deque()
        self.quarantined = False
        self._ownership = None
        self._receipt_identity = None
        self._identity_module = load_identity(source_root)
        directory = (Path(lock_dir) if lock_dir is not None else
                     Path(tempfile.gettempdir()) / f"harness3d-detector-{os.getuid()}")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            # Localhost/IPv4/IPv6 aliases must not create separate backend owners.
            path = directory / f"port-{urlsplit(self.backend).port}.lock"
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            self._ownership = os.fdopen(fd, "rb+")
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("invalid ownership lock")
            fcntl.flock(self._ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.close()
            raise ValueError("backend already owned or its lock is unavailable") from None

    def close(self):
        if self._ownership is not None:
            self._ownership.close()
            self._ownership = None

    @contextmanager
    def slot(self, deadline):
        ticket = object()
        with self.condition:
            if len(self.queue) >= self.max_pending + 1:
                raise GatewayError(429, "detector_capacity", retryable=True)
            self.queue.append(ticket)
            try:
                while self.queue[0] is not ticket:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self.condition.wait(timeout=remaining):
                        raise GatewayError(429, "detector_queue_deadline", retryable=True)
                if time.monotonic() >= deadline:
                    raise GatewayError(429, "detector_queue_deadline", retryable=True)
            except BaseException:
                self.queue.remove(ticket)
                self.condition.notify_all()
                raise
        try:
            yield
        finally:
            with self.condition:
                self.queue.popleft()
                self.condition.notify_all()

    def _verify_identity(self):
        if self.quarantined:
            raise GatewayError(503, "detector_backend_quarantined")
        try:
            receipt_identity = self._identity_module.file_identity(self.receipt)
            if self._receipt_identity is not None and receipt_identity != self._receipt_identity:
                raise ValueError
            value = self._identity_module.detector_identity(self.backend, self.receipt)
            if (not isinstance(value, dict) or value.get("enabled") is not True
                    or not isinstance(value.get("model"), dict) or not value["model"]
                    or digest(value["model"]) != self.expected_model_sha256
                    or self._identity_module.file_identity(self.receipt) != receipt_identity):
                raise ValueError
            self._receipt_identity = receipt_identity
            return {**value, "model_sha256": self.expected_model_sha256}
        except Exception:
            # Never expose receipt content, asset paths, or raw verifier exceptions.
            raise GatewayError(503, "detector_backend_identity_mismatch") from None

    def preflight(self):
        deadline = time.monotonic() + self.deadline_s
        with self.slot(deadline):
            # Cold asset hashing happens before bind, outside the HTTP request budget.
            self._verify_identity()
            health = _exchange(self.backend, "GET", "/health", None,
                               time.monotonic() + min(10, self.deadline_s), self.max_response_bytes)
            if health.status != 200:
                raise GatewayError(502, "detector_health_unavailable")

    def forward(self, method, path, body, deadline, content_type):
        _route(method, path)
        with self.slot(deadline):
            value = self._verify_identity()
            _remaining(deadline)
            if path == "/identity":
                raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, allow_nan=False).encode("utf-8")
                if len(raw) > self.max_response_bytes:
                    raise GatewayError(502, "detector_response_limit")
                return Reply(200, raw, [("Content-Type", "application/json"),
                                        (MODEL_HEADER, self.expected_model_sha256)])
            try:
                reply = _exchange(self.backend, method, path, body, deadline,
                                  self.max_response_bytes, content_type=content_type)
            except GatewayError:
                if method == "POST":
                    # A lost response does not prove the existing process stopped work.
                    self.quarantined = True
                raise
            headers = [(name, value) for name, value in reply.headers
                       if name.lower() != MODEL_HEADER.lower()]
            return Reply(reply.status, reply.body,
                         [*headers, (MODEL_HEADER, self.expected_model_sha256)])


class LoopbackProxy:
    """Plain native interface with credential injection and per-inference model pinning."""
    authenticated = False

    def __init__(self, gateway, expected_model_sha256, *, deadline_s=175,
                 max_request_bytes=MAX_REQUEST_BYTES, max_response_bytes=MAX_RESPONSE_BYTES):
        self.token = _token()
        self.gateway = _url(gateway)
        self.expected_model_sha256 = _sha(expected_model_sha256)
        self.deadline_s = _deadline_setting(deadline_s, 175)
        self.max_request_bytes = _byte_limit(max_request_bytes, MAX_REQUEST_BYTES)
        self.max_response_bytes = _byte_limit(max_response_bytes, MAX_RESPONSE_BYTES)

    def close(self):
        pass

    def _identity(self, deadline):
        reply = _exchange(self.gateway, "GET", "/identity", None, deadline,
                          self.max_response_bytes, token=self.token)
        if reply.status != 200:
            raise GatewayError(502, "detector_gateway_identity_unavailable")
        validate_identity(reply.body, self.expected_model_sha256)
        return reply

    def preflight(self):
        reply = self.forward("GET", "/health", None, time.monotonic() + self.deadline_s,
                             "application/json")
        if reply.status != 200:
            raise GatewayError(502, "detector_gateway_health_unavailable")

    def forward(self, method, path, body, deadline, content_type):
        _route(method, path)
        identity = self._identity(deadline)
        if path == "/identity":
            return identity
        reply = _exchange(self.gateway, method, path, body, deadline,
                          self.max_response_bytes, token=self.token, content_type=content_type)
        if reply.status == 200:
            pins = [value for name, value in reply.headers if name.lower() == MODEL_HEADER.lower()]
            if pins != [self.expected_model_sha256]:
                raise GatewayError(502, "detector_model_identity_mismatch")
        return reply


def _route(method, path):
    if (method, path) not in {("GET", "/health"), ("GET", "/identity"), ("POST", "/infer")}:
        raise GatewayError(404, "unknown_route")


def make_server(transport, host, port, *, max_connections=128):
    """Preflight before bind; shutdown/close this server before closing transport."""
    if not transport.authenticated and host not in LOOPBACK:
        raise ValueError("localproxy must bind to loopback")
    if type(max_connections) is not int or not 1 <= max_connections <= 256:
        raise ValueError("invalid connection limit")
    transport.preflight()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(min(30, transport.deadline_s))

        def log_message(self, *args):
            pass

        def reply(self, value):
            self.close_connection = True
            try:
                self.send_response(value.status)
                for name, content in value.headers:
                    self.send_header(name, content)
                self.send_header("Content-Length", str(len(value.body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(value.body)
            except OSError:
                pass

        def fail(self, error):
            headers = [("Content-Type", "application/json")]
            if error.retryable:
                headers.append(("Retry-After", "1"))
            raw = json.dumps({"error": error.code, "retryable": error.retryable}).encode()
            self.reply(Reply(error.status, raw, headers))

        def dispatch(self):
            deadline = time.monotonic() + transport.deadline_s
            if transport.authenticated and self.command == "POST":
                values = self.headers.get_all("Authorization", [])
                supplied = values[0].encode("utf-8") if len(values) == 1 else b""
                if not hmac.compare_digest(supplied, ("Bearer " + transport.token).encode()):
                    return self.fail(GatewayError(401, "unauthorized"))
            try:
                # BaseHTTPRequestHandler normalizes leading "//"; reject the raw target.
                _route(self.command, self.requestline.split()[1])
                _route(self.command, self.path)
                budgets = self.headers.get_all(BUDGET_HEADER, [])
                if budgets and transport.authenticated:
                    if len(budgets) != 1:
                        raise ValueError
                    deadline = min(deadline, time.monotonic() +
                                   _deadline_setting(float(budgets[0]), 175))
                if self.headers.get_all("Transfer-Encoding"):
                    raise GatewayError(400, "unsupported_transfer_encoding")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) > 1 or lengths and not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
                    raise GatewayError(400, "invalid_content_length")
                size = int(lengths[0]) if lengths else 0
                if self.command == "POST" and not lengths:
                    raise GatewayError(411, "content_length_required")
                if self.command == "POST" and not 0 < size <= transport.max_request_bytes:
                    raise GatewayError(413, "detector_request_limit")
                if self.command == "GET" and size:
                    raise GatewayError(400, "unexpected_get_body")
                types = self.headers.get_all("Content-Type", [])
                content_type = types[0] if types else "application/json"
                if (len(types) > 1
                        or content_type.split(";", 1)[0].strip().lower() != "application/json"
                        or "\r" in content_type or "\n" in content_type):
                    raise GatewayError(415, "json_content_type_required")
                body = None
                if self.command == "POST":
                    with _socket_deadline(self.connection, min(deadline, time.monotonic() + 30)):
                        body = self.rfile.read(size)
                    if len(body) != size:
                        raise GatewayError(400, "incomplete_request_body")
                self.reply(transport.forward(self.command, self.path, body, deadline, content_type))
            except GatewayError as error:
                self.fail(error)
            except (ValueError, OverflowError):
                self.fail(GatewayError(400, "invalid_request"))
            except (OSError, http.client.HTTPException):
                self.fail(GatewayError(408, "request_read_timeout"))
            except Exception:
                self.fail(GatewayError(503, "detector_service_unavailable"))

        do_GET = dispatch
        do_POST = dispatch

    class Server(ThreadingHTTPServer):
        daemon_threads = False
        block_on_close = True
        request_queue_size = 64
        address_family = socket.AF_INET6 if host == "::1" else socket.AF_INET

        def __init__(self):
            self.slots = threading.BoundedSemaphore(max_connections)
            super().__init__((host, port), Handler)

        def process_request(self, request, address):
            if not self.slots.acquire(blocking=False):
                try:
                    request.settimeout(1)
                    raw = b'{"error":"detector_capacity","retryable":true}'
                    request.sendall(b"HTTP/1.0 429 Too Many Requests\r\n"
                                    b"Content-Type: application/json\r\nRetry-After: 1\r\n"
                                    b"Connection: close\r\nContent-Length: "
                                    + str(len(raw)).encode() + b"\r\n\r\n" + raw)
                except OSError:
                    pass
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, address)
            except BaseException:
                self.slots.release()
                raise

        def process_request_thread(self, request, address):
            try:
                super().process_request_thread(request, address)
            finally:
                self.slots.release()

        def handle_error(self, request, client_address):
            pass

    return Server()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--backend", required=True)
    serve.add_argument("--receipt", type=Path, required=True)
    serve.add_argument("--source-root", type=Path,
                       help="optional frozen repository; defaults to launcher PYTHONPATH")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=27037)
    serve.add_argument("--max-pending", type=int, default=16)
    serve.add_argument("--lock-dir", type=Path)
    proxy = subparsers.add_parser("localproxy")
    proxy.add_argument("--gateway", required=True)
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int, default=20027)
    for command, deadline in ((serve, 170), (proxy, 175)):
        command.add_argument("--expected-model-sha256", required=True)
        command.add_argument("--deadline-s", type=float, default=deadline)
        command.add_argument("--max-request-bytes", type=int, default=MAX_REQUEST_BYTES)
        command.add_argument("--max-response-bytes", type=int, default=MAX_RESPONSE_BYTES)
        command.add_argument("--max-connections", type=int, default=128)
    args = parser.parse_args(argv)
    transport = server = None
    try:
        limits = {"deadline_s": args.deadline_s, "max_request_bytes": args.max_request_bytes,
                  "max_response_bytes": args.max_response_bytes}
        if args.mode == "serve":
            transport = DetectorGateway(args.backend, args.receipt, args.source_root,
                args.expected_model_sha256, max_pending=args.max_pending,
                lock_dir=args.lock_dir, **limits)
        else:
            transport = LoopbackProxy(args.gateway, args.expected_model_sha256, **limits)
        server = make_server(transport, args.host, args.port, max_connections=args.max_connections)
        print(json.dumps({"status": "ready", "mode": args.mode, "host": args.host,
                          "port": server.server_address[1],
                          "model_sha256": transport.expected_model_sha256}), flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    except Exception:
        parser.exit(1, "detector startup failed; check environment, frozen source, receipt, pin, and backend\n")
    finally:
        if server is not None:
            server.server_close()
        if transport is not None:
            transport.close()


if __name__ == "__main__":
    main()
