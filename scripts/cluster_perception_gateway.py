#!/usr/bin/env python3
"""Service-host Bailian RPC gateway. No worker ever needs the upstream key.

CLI: --config PUBLIC.json --root LOCAL_CACHE [--host 0.0.0.0] [--port 27035]
     [--max-concurrency 8]

The authorized launcher loads its private credential file into the environment.
This process reads DASHSCOPE_API_KEY (or the configured api_key_env) only through
BailianClient. HARNESS3D_PERCEPTION_TOKEN authenticates POST /chat. Neither secret
belongs in JSON, command arguments, receipts, or logs.

Config accepts BailianClient public kwargs (without opener), public
OnlineRunConfig kwargs (online_model, online_base_url, online_model_revision,
online_api_key_env, online_timeout_s, temperature, max_tokens), or an
online_model settings mapping. Unrelated public run settings are not forwarded.
GET /health returns the frozen Bailian identity without contacting the model.

One active owner of a private, persistent LOCAL filesystem cache is required.
SQLite FULL transactions persist pending BEFORE inference and exact reply bytes
BEFORE ACK. Restart converts pending to indeterminate, never blindly reexecuting.
Responses have no expiration: retain the cache for all worker transport retries.

upstream_audit in the same private SQLite database contains only safe receipts
at the client.chat boundary, not episode or RPC concurrency. Health adds lifetime
upstream counters; pre-instrumentation cache entries are not counted retroactively.
An unfinished receipt becomes unknown on restart, with no invented end time.
The durable start precedes client.chat: a crash in that gap is also unknown,
not proof of a wire request or provider-side execution.
"""

from __future__ import annotations

import argparse
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
import sqlite3
import stat
import sys
import threading
import time
import uuid

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill3d.cluster.perception_client import (
    MAX_BODY_BYTES, RPC_SCHEMA, bearer_token, json_object, validate_result,
)
from skill3d.cluster.queue import _local_database_path
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.client_errors import ServiceUnavailable
from skill3d.synthesis.request_audit import payload_bytes


class GatewayError(Exception):
    def __init__(self, status: int, code: str, *, retryable: bool = False):
        super().__init__(code)
        self.status, self.code, self.retryable = status, code, retryable


def _error_reply(error: GatewayError) -> tuple[int, bytes]:
    return error.status, payload_bytes({
        "schema_version": RPC_SCHEMA, "error": error.code, "retryable": error.retryable,
    })


def _safe_usage(value) -> dict:
    if type(value) is not dict:
        return {}
    usage = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        number = value.get(name)
        if (type(number) in (int, float) and 0 <= number <= 2**53 - 1
                and int(number) == number):
            usage[name] = int(number)
    return usage


def _safe_error_type(error: BaseException) -> str:
    # Class names on arbitrary exceptions, as well as their bodies, are untrusted.
    for kind in (ServiceUnavailable, TimeoutError, OSError, ValueError, TypeError, RuntimeError):
        if isinstance(error, kind):
            return kind.__name__
    return "Exception" if isinstance(error, Exception) else "interrupted"


def _public_config(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if (not isinstance(key, str)
                    or any(word in key.lower() for word in (
                        "password", "secret", "authorization", "credential"))
                    or "token" in key.lower() and key not in {
                        "max_tokens", "max_output_tokens"}
                    or "api_key" in key.lower() and key not in {
                        "api_key_env", "online_api_key_env"}):
                raise ValueError("config must contain only public settings")
            _public_config(item)
    elif isinstance(value, list):
        for item in value:
            _public_config(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("config numbers must be finite")


def client_options(config: dict) -> dict:
    """Extract a frozen operating point without importing the online/GPU runner."""
    if not isinstance(config, dict):
        raise ValueError("public config must be a JSON object")
    _public_config(config)
    if isinstance(config.get("online_model"), dict):
        config = config["online_model"]
    aliases = {
        "model": ("model", "online_model"),
        "base_url": ("base_url", "online_base_url"),
        "model_revision": ("model_revision", "online_model_revision"),
        "api_key_env": ("api_key_env", "online_api_key_env"),
        "timeout_s": ("timeout_s", "online_timeout_s"),
        "temperature": ("temperature",),
        "max_output_tokens": ("max_output_tokens", "max_tokens"),
    }
    for name in ("enabled", "online_model_enabled"):
        if name in config and config[name] is not True:
            raise ValueError("perception service requires the enabled online model")
    for name in ("provider", "online_model_provider"):
        if name in config and config[name] != "bailian":
            raise ValueError("perception service requires Bailian")
    if "opener" in config:
        raise ValueError("opener is not public model configuration")
    options = {}
    for target, names in aliases.items():
        supplied = [config[name] for name in names if name in config]
        if len(supplied) > 1 and supplied[0] != supplied[1]:
            raise ValueError("conflicting public model settings")
        if supplied:
            options[target] = supplied[0]
    for name in ("model", "base_url", "api_key_env", "model_revision"):
        if name in options and not isinstance(options[name], str):
            raise ValueError("invalid public model string")
    if "api_key_env" in options and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", options["api_key_env"]):
        raise ValueError("invalid key environment name")
    budget = options.get("max_output_tokens", 2048)
    temperature = options.get("temperature", 0.0)
    timeout = options.get("timeout_s", 180.0)
    if (type(budget) is not int or budget <= 0
            or type(temperature) not in (int, float) or not math.isfinite(temperature)
            or not 0 <= temperature < 2
            or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("invalid frozen model budget, temperature or timeout")
    # Bailian validates HTTPS and /v1; additionally forbid credential-bearing URLs.
    from urllib.parse import urlsplit
    url = options.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    parsed = urlsplit(url)
    if (not parsed.hostname or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or "?" in url or "#" in url
            or any(ord(char) <= 32 for char in url)):
        raise ValueError("invalid credential-free upstream URL")
    client = BailianClient(**options)
    return {name: getattr(client, name) for name in aliases}


class PerceptionGateway:
    """Frozen config and durable deduplication; each invocation has its own client."""

    def __init__(self, config: dict, root: Path, *, max_concurrency: int = 8,
                 client_factory=BailianClient):
        if type(max_concurrency) is not int or not 1 <= max_concurrency <= 8:
            raise ValueError("max_concurrency must be between 1 and 8")
        self.token = bearer_token()
        self._options = client_options(config)
        self._client_factory = client_factory
        self._identity = BailianClient(**self._options).audit_identity()
        self.max_concurrency = max_concurrency
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._condition = threading.Condition()
        self._lock_file = None
        self._db = None
        root = Path(root).expanduser()
        if root.is_symlink():
            raise ValueError("cache root must not be a symlink")
        self.root = _local_database_path(root / "responses.sqlite3").parent
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_uid != os.getuid():
            raise ValueError("cache root must belong to the service user")
        self.root.chmod(0o700)
        try:
            fd = os.open(self.root / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise ValueError("invalid cache owner lock")
            self._lock_file = os.fdopen(fd, "rb+")
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            database = self.root / "responses.sqlite3"
            if database.is_symlink():
                raise ValueError("cache database must not be a symlink")
            fd = os.open(database, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise ValueError("invalid cache database")
            os.close(fd)
            database.chmod(0o600)
            self._db = sqlite3.connect(database, timeout=30, check_same_thread=False)
            self._db.execute("PRAGMA journal_mode=DELETE")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA fullfsync=ON")
            with self._db:
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS requests ("
                    "request_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL, "
                    "identity_sha256 TEXT NOT NULL, state TEXT NOT NULL, "
                    "http_status INTEGER, reply BLOB)")
                self._db.execute(
                    "CREATE TABLE IF NOT EXISTS upstream_audit ("
                    "request_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL, "
                    "started_at_ns INTEGER NOT NULL, ended_at_ns INTEGER, duration_ns INTEGER, "
                    "active_at_start INTEGER NOT NULL, peak_at_start INTEGER NOT NULL, "
                    "active_at_end INTEGER, peak_at_end INTEGER, "
                    "state TEXT NOT NULL, success INTEGER, error_type TEXT, "
                    "usage_status TEXT NOT NULL, token_usage TEXT NOT NULL)")
                self._db.execute("UPDATE requests SET state='indeterminate' WHERE state='pending'")
                self._db.execute(
                    "UPDATE upstream_audit SET state='unknown' WHERE state='active'")
            self._upstream = {
                "started": 0, "active": 0, "peak": 0, "succeeded": 0, "failed": 0,
                "unknown": 0, "usage_measured": 0, "usage_unknown": 0,
                "audit_write_failures": 0,
            }
            for state, usage_status, count in self._db.execute(
                    "SELECT state, usage_status, COUNT(*) FROM upstream_audit "
                    "GROUP BY state, usage_status"):
                self._upstream["started"] += count
                self._upstream[{"success": "succeeded", "error": "failed",
                                "unknown": "unknown"}[state]] += count
                self._upstream["usage_" + usage_status] += count
            self._upstream["peak"] = self._db.execute(
                "SELECT COALESCE(MAX(peak_at_start),0) FROM upstream_audit").fetchone()[0]
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None

    def health(self) -> dict:
        """Calls partition into active/succeeded/failed/unknown; usage excludes active.

        Call counters and peak survive restart; audit_write_failures is process-local.
        Success means client.chat returned, not that subsequent RPC validation passed.
        """
        with self._condition:
            return {
                "schema_version": RPC_SCHEMA, "status": "ready",
                "identity": dict(self._identity), "max_concurrency": self.max_concurrency,
                "max_body_bytes": MAX_BODY_BYTES, "upstream": dict(self._upstream),
            }

    def _start_upstream(self, request_id, payload_hash):
        with self._condition:
            active = self._upstream["active"] + 1
            peak = max(self._upstream["peak"], active)
            try:
                with self._db:
                    self._db.execute(
                        "INSERT INTO upstream_audit "
                        "(request_id,payload_sha256,started_at_ns,active_at_start,"
                        "peak_at_start,state,usage_status,token_usage) "
                        "VALUES (?,?,?,?,?,'active','unknown','{}')",
                        (request_id, payload_hash, time.time_ns(), active, peak))
            except Exception:
                self._upstream["audit_write_failures"] += 1
                raise GatewayError(503, "service_unavailable") from None
            self._upstream["started"] += 1
            self._upstream["active"] = active
            self._upstream["peak"] = peak

    def _finish_upstream(self, request_id, client, *, success, error_type,
                         ended_at_ns, duration_ns):
        try:
            usage = _safe_usage(client.last_usage)
        except Exception:
            usage = {}
        state = "unknown" if success is None else "success" if success else "error"
        usage_status = "measured" if usage else "unknown"
        with self._condition:
            self._upstream["active"] -= 1
            try:
                with self._db:
                    self._db.execute(
                        "UPDATE upstream_audit SET ended_at_ns=?,duration_ns=?,"
                        "active_at_end=?,peak_at_end=?,state=?,success=?,error_type=?,"
                        "usage_status=?,token_usage=? WHERE request_id=?",
                        (ended_at_ns, duration_ns, self._upstream["active"],
                         self._upstream["peak"], state, success, error_type,
                         usage_status, payload_bytes(usage).decode("ascii"), request_id))
            except Exception:
                # A failed end receipt leaves only durable start evidence. Never replay.
                self._upstream["unknown"] += 1
                self._upstream["usage_unknown"] += 1
                self._upstream["audit_write_failures"] += 1
                raise GatewayError(503, "service_unavailable") from None
            self._upstream[{"success": "succeeded", "error": "failed",
                            "unknown": "unknown"}[state]] += 1
            self._upstream["usage_" + usage_status] += 1

    def _validate(self, raw: bytes):
        if not 0 < len(raw) <= MAX_BODY_BYTES:
            raise GatewayError(413, "body_limit")
        try:
            envelope = json_object(raw)
            if (set(envelope) != {"schema_version", "request_id", "payload", "audit_identity"}
                    or envelope["schema_version"] != RPC_SCHEMA):
                raise ValueError
            request_id = envelope["request_id"]
            if not isinstance(request_id, str) or str(uuid.UUID(request_id)) != request_id:
                raise ValueError
            payload = envelope["payload"]
            if not isinstance(payload, dict):
                raise ValueError
            required = {"model", "messages", "temperature", "max_tokens", "enable_thinking"}
            if (not required <= set(payload)
                    or set(payload) - required - {"seed", "response_format", "top_p"}
                    or payload["model"] != self._options["model"]
                    or type(payload["max_tokens"]) is not int
                    or not 0 < payload["max_tokens"] <= self._options["max_output_tokens"]
                    or type(payload["temperature"]) not in (int, float)
                    or payload["temperature"] != self._options["temperature"]
                    or type(payload["enable_thinking"]) is not bool
                    or not isinstance(payload["messages"], list)
                    or any(not isinstance(message, dict) for message in payload["messages"])
                    or "seed" in payload and type(payload["seed"]) is not int
                    or "response_format" in payload and not isinstance(payload["response_format"], dict)
                    or "top_p" in payload and (type(payload["top_p"]) not in (int, float)
                                              or not 0 < payload["top_p"] <= 1)):
                raise ValueError
            identity = {**self._identity, "seed": payload.get("seed")}
            if envelope["audit_identity"] != identity:
                raise ValueError
            # Require the frozen payload to survive Bailian's own construction unchanged.
            rebuilt = BailianClient(**self._options).build_payload(
                payload["messages"], **{key: value for key, value in payload.items()
                                         if key not in {"messages", "model"}})
            if payload_bytes(rebuilt) != payload_bytes(payload):
                raise ValueError
            return request_id, payload, identity
        except (KeyError, ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
            raise GatewayError(422, "frozen_request_mismatch") from None

    def chat(self, raw: bytes) -> tuple[int, bytes]:
        request_id, payload, identity = self._validate(raw)
        payload_hash = hashlib.sha256(payload_bytes(payload)).hexdigest()
        identity_hash = hashlib.sha256(payload_bytes(identity)).hexdigest()
        deadline = time.monotonic() + min(self._options["timeout_s"] + 5, 300)
        with self._condition:
            while True:
                row = self._db.execute(
                    "SELECT payload_sha256, identity_sha256, state, http_status, reply "
                    "FROM requests WHERE request_id=?", (request_id,)).fetchone()
                if row is None:
                    if not self._slots.acquire(blocking=False):
                        raise GatewayError(429, "capacity", retryable=True)
                    try:
                        with self._db:
                            self._db.execute("INSERT INTO requests VALUES (?,?,?,'pending',NULL,NULL)",
                                             (request_id, payload_hash, identity_hash))
                    except BaseException:
                        self._slots.release()
                        raise
                    break
                if row[0] != payload_hash or row[1] != identity_hash:
                    raise GatewayError(409, "request_id_conflict")
                if row[2] == "complete":
                    return row[3], bytes(row[4])
                if row[2] == "indeterminate":
                    raise GatewayError(409, "indeterminate")
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._condition.wait(timeout=remaining):
                    raise GatewayError(503, "pending", retryable=True)
        try:
            status, reply = self._invoke(request_id, payload, identity, payload_hash)
            with self._condition:
                with self._db:
                    self._db.execute(
                        "UPDATE requests SET state='complete', http_status=?, reply=? "
                        "WHERE request_id=?", (status, reply, request_id))
                self._condition.notify_all()
            return status, reply
        except BaseException:
            with self._condition:
                # No evidence of a committed reply: do not call the model again.
                with self._db:
                    self._db.execute(
                        "UPDATE requests SET state='indeterminate' "
                        "WHERE request_id=? AND state='pending'", (request_id,))
                self._condition.notify_all()
            raise
        finally:
            self._slots.release()

    def _invoke(self, request_id, payload, identity, payload_hash):
        try:
            client = self._client_factory(**self._options)
            arguments = {key: value for key, value in payload.items()
                         if key not in {"messages", "model"}}
        except Exception as error:
            return self._invoke_error(error)
        self._start_upstream(request_id, payload_hash)
        started_ns = time.monotonic_ns()
        try:
            content = client.chat(payload["messages"], **arguments)
        except BaseException as error:
            self._finish_upstream(
                request_id, client, success=False if isinstance(error, Exception) else None,
                error_type=_safe_error_type(error), ended_at_ns=time.time_ns(),
                duration_ns=time.monotonic_ns() - started_ns)
            if isinstance(error, Exception):
                return self._invoke_error(error)
            raise
        else:
            self._finish_upstream(
                request_id, client, success=True, error_type=None,
                ended_at_ns=time.time_ns(), duration_ns=time.monotonic_ns() - started_ns)
        try:
            result = {
                "schema_version": RPC_SCHEMA, "request_id": request_id,
                "payload_sha256": payload_hash, "audit_identity": client.audit_identity(
                    seed=payload.get("seed")),
                "content": content, "last_usage": client.last_usage,
                "last_generation": client.last_generation,
                "last_response_identity": client.last_response_identity,
                "last_request_identity": client.last_request_identity,
            }
            validate_result(result, request_id=request_id, payload=payload, audit_identity=identity)
            reply = payload_bytes(result)
            if len(reply) > MAX_BODY_BYTES:
                raise ValueError("response limit")
            # Credentials are never valid model evidence or model output.
            secrets = (self.token, os.environ.get(self._options["api_key_env"], "").strip())
            if any(secret and secret in reply.decode("utf-8") for secret in secrets):
                raise ValueError("unsafe response")
            return 200, reply
        except Exception as error:
            return self._invoke_error(error)

    @staticmethod
    def _invoke_error(error):
        # The original client can include raw upstream bodies in exceptions.
        # Only a fixed code and HTTP status cross this service boundary.
        if isinstance(error, ServiceUnavailable):
            return _error_reply(GatewayError(503, "upstream_unavailable"))
        return _error_reply(GatewayError(502, "upstream_invalid"))


def make_server(gateway: PerceptionGateway, host="0.0.0.0", port=27035,
                *, max_connections=64):
    """Bind without model calls. Shutdown/close the server before closing gateway."""
    if type(max_connections) is not int or not 1 <= max_connections <= 128:
        raise ValueError("max_connections must be between 1 and 128")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *_args):
            pass

        def reply(self, status, raw):
            self.close_connection = True
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                if status == 429:
                    self.send_header("Retry-After", "1")
                self.end_headers()
                self.wfile.write(raw)
            except OSError:
                pass  # Result is already durable; lost ACKs replay the same bytes.

        def dispatch(self):
            if self.command == "GET" and self.path == "/health":
                self.reply(200, payload_bytes(gateway.health()))
                return
            values = self.headers.get_all("Authorization", [])
            supplied = values[0].encode("utf-8") if len(values) == 1 else b""
            if not hmac.compare_digest(supplied, ("Bearer " + gateway.token).encode()):
                self.reply(*_error_reply(GatewayError(401, "unauthorized")))
                return
            try:
                if self.command != "POST" or self.path != "/chat":
                    raise GatewayError(404, "unknown_route")
                if self.headers.get_all("Transfer-Encoding"):
                    raise GatewayError(400, "unsupported_transfer_encoding")
                lengths = self.headers.get_all("Content-Length", [])
                if not lengths:
                    raise GatewayError(411, "content_length_required")
                if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
                    raise GatewayError(400, "invalid_content_length")
                size = int(lengths[0])
                if not 0 < size <= MAX_BODY_BYTES:
                    raise GatewayError(413, "body_limit")
                types = self.headers.get_all("Content-Type", [])
                if (len(types) != 1
                        or types[0].split(";", 1)[0].strip().lower() != "application/json"):
                    raise GatewayError(415, "json_content_type_required")

                def abort():
                    try:
                        self.connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

                timer = threading.Timer(30, abort)
                timer.daemon = True
                timer.start()
                try:
                    raw = self.rfile.read(size)
                finally:
                    timer.cancel()
                if len(raw) != size:
                    raise GatewayError(400, "incomplete_body")
                self.reply(*gateway.chat(raw))
            except GatewayError as error:
                self.reply(*_error_reply(error))
            except (OSError, http.client.HTTPException):
                self.reply(*_error_reply(GatewayError(408, "request_transport")))
            except Exception:
                self.reply(*_error_reply(GatewayError(503, "service_unavailable")))

        do_GET = dispatch
        do_POST = dispatch

    class Server(ThreadingHTTPServer):
        daemon_threads = False
        block_on_close = True
        request_queue_size = 64
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

        def __init__(self):
            self.slots = threading.BoundedSemaphore(max_connections)
            super().__init__((host, port), Handler)

        def process_request(self, request, address):
            if not self.slots.acquire(blocking=False):
                status, raw = _error_reply(GatewayError(429, "capacity", retryable=True))
                try:
                    request.settimeout(1)
                    request.sendall(
                        f"HTTP/1.0 {status} Too Many Requests\r\n"
                        "Content-Type: application/json\r\nConnection: close\r\n"
                        f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=27035)
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    gateway = server = None
    try:
        config = json_object(args.config.read_bytes())
        options = client_options(config)
        if not BailianClient(**options).configured:
            raise ValueError("authorized upstream environment is required")
        gateway = PerceptionGateway(config, args.root, max_concurrency=args.max_concurrency)
        server = make_server(gateway, args.host, args.port)
        print(json.dumps(gateway.health()), flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    except Exception:
        parser.exit(1, "perception startup failed; check public config, authorized environment, and local cache\n")
    finally:
        if server is not None:
            server.server_close()
        if gateway is not None:
            gateway.close()


if __name__ == "__main__":
    main()
