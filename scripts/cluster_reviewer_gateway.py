#!/usr/bin/env python3
"""Stdlib native-reviewer transport, with no project/model imports.

serve --backends-config FILE --host HOST --port 27034
localproxy --gateway http://SERVICE:27034 --replica N --port 27145+N
           --expected-identity-sha256 SHA

Both modes require HARNESS3D_REVIEWER_TOKEN in the environment, never in JSON,
arguments, receipts, or logs. FILE contains four ordered ``backends`` rows with
``endpoint`` and ``expected_identity_sha256``; optional
``semantic_identity_sha256`` pins the canonical identity minus ONLY ``device``.

Deploy exactly one central gateway on the service host. Lifetime advisory locks
also prevent another local gateway from owning the same backend. All workers
must use it; direct native access bypasses its FIFO admission queues. Native
clients should use timeout_s=180, above the 170s gateway/175s proxy deadlines.
Capacity errors are retryable transport failures, not semantic decisions.
After an ambiguous POST failure the replica is quarantined: verify/restart the
native service before restarting the gateway. Health alone cannot prove idle.
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
import json
import math
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
from urllib.parse import urlsplit

TOKEN_ENV = "HARNESS3D_REVIEWER_TOKEN"
BACKEND_PORTS = (27145, 27146, 27147, 27148)
MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
SHA256 = re.compile(r"[0-9a-f]{64}")
NATIVE_PATH = re.compile(r"/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("identity pin must be a lower-case SHA256")
    return value


def _url(value, *, backend=False):
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if (parsed.scheme not in ({"http"} if backend else {"http", "https"})
                or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment
                or "?" in value or "#" in value
                or any(ord(char) <= 32 or ord(char) >= 127 for char in value)
                or backend and parsed.hostname not in LOOPBACK
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
                             if name.lower() in {"content-type", "retry-after", "content-encoding"}
                             and name.lower() not in hop_headers
                             and len(value) <= 4096 and "\r" not in value and "\n" not in value)


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GatewayError(504, "reviewer_deadline")
    return remaining


@contextmanager
def _socket_deadline(sock, deadline):
    """Bound even a peer that trickles bytes below its socket idle timeout."""
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


def _exchange(endpoint, method, path, body, deadline, max_response_bytes, *, token=None,
              content_type="application/json"):
    parsed = urlsplit(endpoint)
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = cls(parsed.hostname, parsed.port, timeout=_remaining(deadline))
    try:
        # http.client neither follows redirects nor consults HTTP(S)_PROXY.
        connection.connect()
        upstream_socket = connection.sock
        with _socket_deadline(upstream_socket, deadline):
            headers = {"Content-Type": content_type, "Connection": "close"}
            if token is not None:
                headers["Authorization"] = "Bearer " + token
                headers["X-Reviewer-Budget-S"] = str(_remaining(deadline))
            connection.request(method, path, body=body, headers=headers)
            # Keep socket ownership: getresponse() closes HTTP/1.0 sockets at body end.
            with http.client.HTTPResponse(upstream_socket, method=method) as response:
                response.begin()
                raw = response.read(max_response_bytes + 1)
                if len(raw) > max_response_bytes:
                    raise GatewayError(502, "reviewer_response_limit")
                declared = response.getheader("Content-Length")
                if declared is not None and (not declared.isdecimal() or int(declared) != len(raw)):
                    raise GatewayError(502, "reviewer_incomplete_response")
                # Native handlers release their inference lock after writing the body.
                # Peer EOF proves handler cleanup completed, unlike Content-Length alone.
                if upstream_socket.recv(1):
                    raise GatewayError(502, "reviewer_extra_response_bytes")
                _remaining(deadline)
                return Reply(response.status, raw, response.getheaders())
    except (TimeoutError, socket.timeout):
        raise GatewayError(504, "reviewer_upstream_timeout") from None
    except (OSError, http.client.HTTPException):
        raise GatewayError(502, "reviewer_upstream_transport") from None
    finally:
        connection.close()


def validate_health(raw, expected_identity_sha256, semantic_identity_sha256=None):
    """Validate without changing the opaque native health bytes."""
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate health field")
            result[key] = value
        return result

    try:
        health = json.loads(raw, object_pairs_hook=pairs)
        identity = health["identity"]
        if (not isinstance(identity, dict) or not isinstance(identity.get("device"), dict)
                or health["identity_sha256"] != digest(identity)
                or health["identity_sha256"] != expected_identity_sha256):
            raise ValueError
        semantic = digest({key: value for key, value in identity.items() if key != "device"})
        if semantic_identity_sha256 is not None and semantic != semantic_identity_sha256:
            raise ValueError
        return semantic
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError):
        raise GatewayError(502, "reviewer_health_identity_mismatch") from None


class _Replica:
    def __init__(self, endpoint, expected_identity_sha256, max_pending):
        self.endpoint, self.expected_identity_sha256 = endpoint, expected_identity_sha256
        self.condition = threading.Condition()
        self.queue = deque()
        self.max_pending = max_pending
        self.quarantined = False

    @contextmanager
    def slot(self, deadline):
        ticket = object()
        with self.condition:
            if len(self.queue) >= self.max_pending + 1:
                raise GatewayError(429, "reviewer_capacity", retryable=True)
            self.queue.append(ticket)
            try:
                while self.queue[0] is not ticket:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self.condition.wait(timeout=remaining):
                        raise GatewayError(429, "reviewer_queue_deadline", retryable=True)
                if time.monotonic() >= deadline:
                    raise GatewayError(429, "reviewer_queue_deadline", retryable=True)
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


class ReviewerGateway:
    """One service-host owner; FIFO per-replica queues cover all HTTP workers."""
    authenticated = True

    def __init__(self, backends, *, semantic_identity_sha256=None, deadline_s=170,
                 max_pending=16, max_request_bytes=MAX_REQUEST_BYTES,
                 max_response_bytes=MAX_RESPONSE_BYTES, lock_dir=None):
        self.token = _token()
        self.deadline_s = _deadline_setting(deadline_s)
        self.max_request_bytes = _byte_limit(max_request_bytes, MAX_REQUEST_BYTES)
        self.max_response_bytes = _byte_limit(max_response_bytes, MAX_RESPONSE_BYTES)
        if type(max_pending) is not int or not 0 <= max_pending <= 64:
            raise ValueError("max_pending must be between 0 and 64")
        if not isinstance(backends, list) or len(backends) != 4:
            raise ValueError("exactly four ordered backends are required")
        self.replicas = []
        for row in backends:
            if not isinstance(row, dict) or set(row) != {"endpoint", "expected_identity_sha256"}:
                raise ValueError("backend requires only endpoint and expected_identity_sha256")
            self.replicas.append(_Replica(_url(row["endpoint"], backend=True),
                                          _sha(row["expected_identity_sha256"]), max_pending))
        if len({row.endpoint for row in self.replicas}) != 4:
            raise ValueError("backend endpoints must be distinct")
        self.semantic_identity_sha256 = (
            None if semantic_identity_sha256 is None else _sha(semantic_identity_sha256))
        directory = Path(lock_dir) if lock_dir is not None else (
            Path(tempfile.gettempdir()) / f"harness3d-reviewer-{os.getuid()}")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._ownership = []
        try:
            for replica in sorted(self.replicas, key=lambda row: row.endpoint):
                path = directory / (hashlib.sha256(replica.endpoint.encode()).hexdigest() + ".lock")
                fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                stream = os.fdopen(fd, "rb+")
                self._ownership.append(stream)
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.close()
            raise ValueError("a backend is already owned or its lock is unavailable") from None

    def close(self):
        for stream in self._ownership:
            stream.close()
        self._ownership.clear()

    def preflight(self):
        for replica in self.replicas:
            reply = _exchange(replica.endpoint, "GET", "/health", None,
                              time.monotonic() + min(10, self.deadline_s), self.max_response_bytes)
            if reply.status != 200:
                raise GatewayError(502, "reviewer_health_unavailable")
            semantic = validate_health(reply.body, replica.expected_identity_sha256,
                                       self.semantic_identity_sha256)
            if self.semantic_identity_sha256 is None:
                self.semantic_identity_sha256 = semantic

    def forward(self, method, path, body, deadline, content_type):
        match = re.fullmatch(r"/replicas/([0-3])(/.*)", path)
        if not match:
            raise GatewayError(404, "unknown_route")
        native_path = match[2]
        _route(method, native_path)
        replica = self.replicas[int(match[1])]
        with replica.slot(deadline):
            if replica.quarantined:
                raise GatewayError(503, "reviewer_replica_quarantined")
            health = _exchange(replica.endpoint, "GET", "/health", None,
                               min(deadline, time.monotonic() + 10), self.max_response_bytes)
            if health.status != 200:
                return health
            validate_health(health.body, replica.expected_identity_sha256,
                            self.semantic_identity_sha256)
            if method == "GET":
                return health
            try:
                return _exchange(replica.endpoint, method, native_path, body, deadline,
                                 self.max_response_bytes, content_type=content_type)
            except GatewayError:
                # A connection failure is NOT evidence that native inference stopped.
                replica.quarantined = True
                raise


class LoopbackProxy:
    """Credential-injecting local endpoint for the unchanged loopback-only clients."""
    authenticated = False

    def __init__(self, gateway, replica, expected_identity_sha256, *,
                 semantic_identity_sha256=None, deadline_s=175,
                 max_request_bytes=MAX_REQUEST_BYTES, max_response_bytes=MAX_RESPONSE_BYTES):
        self.token = _token()
        self.gateway = _url(gateway)
        if type(replica) is not int or not 0 <= replica <= 3:
            raise ValueError("replica must be between 0 and 3")
        self.replica = replica
        self.expected_identity_sha256 = _sha(expected_identity_sha256)
        self.semantic_identity_sha256 = (
            None if semantic_identity_sha256 is None else _sha(semantic_identity_sha256))
        self.deadline_s = _deadline_setting(deadline_s)
        self.max_request_bytes = _byte_limit(max_request_bytes, MAX_REQUEST_BYTES)
        self.max_response_bytes = _byte_limit(max_response_bytes, MAX_RESPONSE_BYTES)

    def close(self):
        pass

    def preflight(self):
        reply = self.forward("GET", "/health", None, time.monotonic() + self.deadline_s,
                             "application/json")
        if reply.status != 200:
            raise GatewayError(502, "reviewer_gateway_health_unavailable")

    def forward(self, method, path, body, deadline, content_type):
        _route(method, path)
        if method == "POST":
            health = self.forward("GET", "/health", None, deadline, content_type)
            if health.status != 200:
                return health
        reply = _exchange(self.gateway, method, f"/replicas/{self.replica}" + path, body,
                          deadline, self.max_response_bytes, token=self.token,
                          content_type=content_type)
        if method == "GET" and reply.status == 200:
            validate_health(reply.body, self.expected_identity_sha256,
                            self.semantic_identity_sha256)
        return reply


def _route(method, path):
    if (not NATIVE_PATH.fullmatch(path)
            or method == "GET" and path != "/health"
            or method not in {"GET", "POST"}):
        raise GatewayError(404, "unknown_route")


def _deadline_setting(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= 175:
        raise ValueError("deadline_s must be finite, positive, and <=175")
    return float(value)


def _byte_limit(value, ceiling):
    if type(value) is not int or not 0 < value <= ceiling:
        raise ValueError("invalid byte limit")
    return value


def make_server(transport, host, port, *, max_connections=128):
    """Preflight before binding. Caller must shutdown, server_close, then close transport."""
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
            self.reply(Reply(error.status, json.dumps({
                "error": error.code, "retryable": error.retryable}).encode(), headers))

        def dispatch(self):
            deadline = time.monotonic() + transport.deadline_s
            if transport.authenticated:
                values = self.headers.get_all("Authorization", [])
                supplied = values[0].encode("utf-8") if len(values) == 1 else b""
                if not hmac.compare_digest(supplied, ("Bearer " + transport.token).encode()):
                    return self.fail(GatewayError(401, "unauthorized"))
            try:
                budget = self.headers.get("X-Reviewer-Budget-S")
                if budget is not None and transport.authenticated:
                    deadline = min(deadline, time.monotonic() + _deadline_setting(float(budget)))
                if self.headers.get_all("Transfer-Encoding"):
                    raise GatewayError(400, "unsupported_transfer_encoding")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) > 1 or lengths and not re.fullmatch(r"[0-9]+", lengths[0]):
                    raise GatewayError(400, "invalid_content_length")
                size = int(lengths[0]) if lengths else 0
                if self.command == "POST" and not lengths:
                    raise GatewayError(411, "content_length_required")
                if self.command == "POST" and not 0 < size <= transport.max_request_bytes:
                    raise GatewayError(413, "reviewer_request_limit")
                if self.command == "GET" and size:
                    raise GatewayError(400, "unexpected_get_body")
                content_type = self.headers.get("Content-Type", "application/json")
                if (content_type.split(";", 1)[0].strip().lower() != "application/json"
                        or "\r" in content_type or "\n" in content_type):
                    raise GatewayError(415, "json_content_type_required")
                body = None
                if self.command == "POST":
                    with _socket_deadline(self.connection, min(deadline, time.monotonic() + 30)):
                        body = self.rfile.read(size)
                    if len(body) != size:
                        raise GatewayError(400, "incomplete_request_body")
                value = transport.forward(self.command, self.path, body, deadline, content_type)
                self.reply(value)
            except GatewayError as error:
                self.fail(error)
            except (ValueError, OverflowError):
                self.fail(GatewayError(400, "invalid_request"))
            except (OSError, http.client.HTTPException):
                self.fail(GatewayError(408, "request_read_timeout"))

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
                    raw = b'{"error":"reviewer_capacity","retryable":true}'
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
            pass  # Never emit tracebacks containing request metadata or credentials.

    return Server()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--backends-config", type=Path, required=True)
    serve.add_argument("--host", required=True)
    serve.add_argument("--port", type=int, default=27034)
    serve.add_argument("--max-pending", type=int, default=16)
    serve.add_argument("--lock-dir", type=Path)
    proxy = subparsers.add_parser("localproxy")
    proxy.add_argument("--gateway", required=True)
    proxy.add_argument("--replica", type=int, choices=range(4), required=True)
    proxy.add_argument("--expected-identity-sha256", required=True)
    proxy.add_argument("--semantic-identity-sha256")
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int)
    for command, deadline in ((serve, 170), (proxy, 175)):
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
            config = json.loads(args.backends_config.read_text())
            if not isinstance(config, dict) or set(config) - {"backends", "semantic_identity_sha256"}:
                raise ValueError("invalid backends config fields")
            transport = ReviewerGateway(config["backends"], max_pending=args.max_pending,
                semantic_identity_sha256=config.get("semantic_identity_sha256"),
                lock_dir=args.lock_dir, **limits)
        else:
            transport = LoopbackProxy(args.gateway, args.replica, args.expected_identity_sha256,
                semantic_identity_sha256=args.semantic_identity_sha256, **limits)
        port = args.port if args.port is not None else BACKEND_PORTS[args.replica]
        server = make_server(transport, args.host, port, max_connections=args.max_connections)
        print(json.dumps({"status": "ready", "mode": args.mode, "host": args.host,
                          "port": server.server_address[1],
                          "semantic_identity_sha256": transport.semantic_identity_sha256}), flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    except (ValueError, KeyError, OSError, GatewayError):
        parser.exit(1, "reviewer startup failed; check environment, config, pins, and backend readiness\n")
    finally:
        if server is not None:
            server.server_close()
        if transport is not None:
            transport.close()


if __name__ == "__main__":
    main()
