"""Authenticated forwarding of exact evaluation payloads and original upstream bytes."""
from contextlib import contextmanager
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path[:0] = [str(BASE), str(ROOT / "eval_provider_runtime"), str(ROOT / "code/src")]
from transport_auth import load_transport_token
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_audit import payload_bytes

SCHEMA = "harness3d.audited-evaluation-rpc/1"
TOKEN = load_transport_token()
OPTIONS = dict(model="qwen3.8-flash", temperature=0, max_output_tokens=8192,
               timeout_s=1200, api_key_env="H3D_AUDITED_EVALUATION_KEY")
assert BailianClient(**OPTIONS).endpoint_hash == "5891aed827c4e67b2d7c0c73ea819327ce3f2b6ef72213cd79669486a26b1ead"
assert os.environ.get(OPTIONS["api_key_env"])
CACHE = ROOT / "audited_eval_gateway"
assert CACHE.is_dir()
LIMIT = threading.BoundedSemaphore(4)
LOCK = threading.Lock()
KEY_LOCKS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temp = path.with_name("." + path.name + "-" + str(threading.get_ident()))
    temp.write_bytes(payload_bytes(value))
    temp.replace(path)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def auth(self):
        return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + TOKEN)

    def respond(self, code, raw):
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != "/authenticated-health" or not self.auth():
            return self.respond(403, b'{"status":"unauthorized"}')
        self.respond(200, payload_bytes({
            "status": "ready", "schema_version": SCHEMA, "model": OPTIONS["model"],
            "endpoint_sha256": BailianClient(**OPTIONS).endpoint_hash,
            "source_sha256": sha(__file__), "max_inflight": 4,
            "upstream_timeout_s": OPTIONS["timeout_s"], "mmsi_timeout_extension": True,
            "cache_entries": len(list(CACHE.glob("*.intent.json"))),
        }))

    def do_POST(self):
        if self.path != "/chat" or not self.auth():
            return self.respond(403, b'{"status":"unauthorized"}')
        try:
            length = int(self.headers["Content-Length"])
            assert 0 < length <= 64 * 1024**2
            envelope = json.loads(self.rfile.read(length))
            assert set(envelope) == {"schema_version", "request_id", "payload", "audit_identity"}
            assert envelope["schema_version"] == SCHEMA
            request_id = envelope["request_id"]
            assert isinstance(request_id, str) and re.fullmatch(r"[0-9a-f]{64}", request_id)
            identity = envelope["audit_identity"]
            assert set(identity) == {"run_identity", "operation_id"}
            assert all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in identity.values())
            assert identity["run_identity"] == "92a55aecff8c3e79090ef2b4fdfd02132185e55bca006479de332eafc04edbb2"
            payload = envelope["payload"]
            assert {"model", "messages", "temperature", "max_tokens", "enable_thinking", "seed"} <= set(payload)
            assert not set(payload) - {"model", "messages", "temperature", "max_tokens",
                                      "enable_thinking", "seed", "response_format", "top_p"}
            assert payload["model"] == OPTIONS["model"] and payload["max_tokens"] == 8192
            assert payload["seed"] == 137 and payload["temperature"] == 0
            assert payload["enable_thinking"] is True
            client = BailianClient(**OPTIONS)
            kwargs = {k: v for k, v in payload.items() if k not in {"model", "messages"}}
            assert payload_bytes(client.build_payload(payload["messages"], **kwargs)) == payload_bytes(payload)
            digest = hashlib.sha256(payload_bytes(payload)).hexdigest()
        except BaseException:
            return self.respond(422, b'{"status":"invalid_frozen_evaluation_request"}')
        with LOCK:
            key_lock = KEY_LOCKS.setdefault(request_id, threading.Lock())
        with key_lock, LIMIT:
            intent = CACHE / (request_id + ".intent.json")
            response = CACHE / (request_id + ".response.json")
            body_path = CACHE / (request_id + ".body.json")
            if intent.exists():
                previous = json.loads(intent.read_text())
                if previous["payload_sha256"] != digest or previous["audit_identity"] != identity:
                    return self.respond(409, b'{"status":"request_identity_conflict"}')
                if response.exists():
                    record = json.loads(response.read_text())
                    if record["status"] == "received":
                        assert sha(body_path) == record["body_sha256"]
                        return self.respond(200, body_path.read_bytes())
                return self.respond(409, b'{"status":"previous_request_not_retriable"}')
            save(intent, {"at": time.time(), "request_id": request_id,
                          "upstream_timeout_s": OPTIONS["timeout_s"],
                          "payload_sha256": digest, "audit_identity": identity,
                          "parameters": {k: v for k, v in payload.items() if k != "messages"}})
            original_opener = client._opener
            raw_response = []

            @contextmanager
            def capture(request, timeout):
                import io
                with original_opener(request, timeout=timeout) as upstream:
                    raw = upstream.read(64 * 1024**2 + 1)
                assert len(raw) <= 64 * 1024**2
                assert TOKEN.encode() not in raw
                assert os.environ[OPTIONS["api_key_env"]].encode() not in raw
                raw_response.append(raw)
                body_path.write_bytes(raw)
                yield io.BytesIO(raw)

            client._opener = capture
            started = time.monotonic()
            try:
                client.chat(payload["messages"], **kwargs)
                assert client.last_response_identity.get("model") == OPTIONS["model"]
                assert client.last_response_identity.get("model_reported") is True
                assert len(raw_response) == 1
                body_path.write_bytes(raw_response[0])
                save(response, {"status": "received", "at": time.time(),
                                "seconds": time.monotonic() - started,
                                "body_sha256": sha(body_path), "usage": client.last_usage,
                                "request_identity": client.last_request_identity,
                                "response_identity": client.last_response_identity,
                                "finish_reason": client.last_generation.get("finish_reason")})
            except BaseException as error:
                save(response, {"status": "failed", "at": time.time(),
                                "error_type": type(error).__name__, "automatic_retry": False,
                                "service_error": getattr(client, "last_service_error", {}),
                                "usage": client.last_usage,
                                "body_sha256": sha(body_path) if body_path.exists() else None})
                return self.respond(502, b'{"status":"upstream_evaluation_failed"}')
            self.respond(200, raw_response[0])


server = ThreadingHTTPServer(("0.0.0.0", 27440), Handler)
proc = Path("/proc") / str(os.getpid())
fields = (proc / "stat").read_text().split(") ", 1)[1].split()
save(ROOT / "audited_eval_gateway_timeout06_startup.json", {
    "pid": os.getpid(), "start_ticks": fields[19], "command_sha256": sha(proc / "cmdline"),
    "source_sha256": sha(__file__), "at": time.time(), "port": 27440,
    "upstream_timeout_s": OPTIONS["timeout_s"],
    "schema_version": SCHEMA, "max_inflight": 4, "automatic_retry": False})
server.serve_forever()
