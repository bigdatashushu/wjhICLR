"""Authenticated, single-attempt forwarding for native tool and direct requests."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "gateway_config.json").read_text())
sys.path[:0] = [CONFIG["auth_module_root"], CONFIG["source_root"]]
from transport_auth import load_transport_token
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_audit import payload_bytes

TOKEN = load_transport_token()
SCHEMA = "cv3d-native-evaluation-rpc/1"
CACHE = ROOT / "requests"
CACHE.mkdir(exist_ok=True)
MAX_INFLIGHT = CONFIG.get("max_inflight", 4)
LIMIT = threading.BoundedSemaphore(MAX_INFLIGHT)
LOCK = threading.Lock()
KEY_LOCKS = {}
CLIENT = BailianClient(
    model="qwen3.8-flash", max_output_tokens=8192, timeout_s=600,
    api_key_env=CONFIG["api_key_env"])
assert CLIENT.configured


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, row):
    path = Path(path)
    temporary = path.with_name("." + path.name + str(threading.get_ident()))
    temporary.write_bytes(payload_bytes(row))
    temporary.replace(path)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def auth(self):
        return hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + TOKEN)

    def respond(self, code, raw):
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != "/authenticated-health" or not self.auth():
            return self.respond(403, b'{"status":"unauthorized"}')
        self.respond(200, payload_bytes({
            "status": "ready", "schema_version": SCHEMA, "model": CLIENT.model,
            "endpoint_sha256": CLIENT.endpoint_hash, "source_sha256": sha(__file__),
            "max_inflight": MAX_INFLIGHT, "native_tools": True,
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
            assert re.fullmatch(r"[0-9a-f]{64}", request_id)
            identity = envelope["audit_identity"]
            assert set(identity) == {"run_identity", "operation_id"}
            assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in identity.values())
            payload = envelope["payload"]
            assert set(payload) <= {
                "model", "messages", "temperature", "max_tokens", "enable_thinking",
                "seed", "tools", "tool_choice"}
            assert payload["model"] == "qwen3.8-flash"
            assert payload["max_tokens"] == 8192 and payload["seed"] == 137
            assert payload["temperature"] == 0 and payload["enable_thinking"] is True
            assert isinstance(payload["messages"], list)
            if "tools" in payload:
                assert payload["tool_choice"] == "auto"
                assert {t["function"]["name"] for t in payload["tools"]} <= {
                    "point_depth", "pair_point_distance", "view_candidates", "view_depth_summary"}
            else:
                assert "tool_choice" not in payload
            wire = payload_bytes(payload)
            digest = hashlib.sha256(wire).hexdigest()
        except BaseException:
            return self.respond(422, b'{"status":"invalid_native_evaluation_request"}')
        with LOCK:
            key_lock = KEY_LOCKS.setdefault(request_id, threading.Lock())
        with key_lock, LIMIT:
            intent = CACHE / (request_id + ".intent.json")
            record = CACHE / (request_id + ".response.json")
            body_path = CACHE / (request_id + ".body.json")
            # No replay or upstream retry is performed on any repeated request ID.
            if intent.exists():
                return self.respond(409, b'{"status":"request_already_attempted"}')
            save(intent, {
                "at": time.time(), "payload_sha256": digest, "audit_identity": identity,
                "request_id": request_id, "has_tools": bool(payload.get("tools")),
            })
            started = time.monotonic()
            key = os.environ[CONFIG["api_key_env"]]
            request = urllib.request.Request(
                CLIENT.base_url + "/chat/completions", data=wire,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                method="POST")
            try:
                with CLIENT._opener(request, timeout=600) as upstream:
                    raw = upstream.read(64 * 1024**2 + 1)
                assert len(raw) <= 64 * 1024**2
                assert TOKEN.encode() not in raw and key.encode() not in raw
                body = json.loads(raw)
                assert body["model"] == "qwen3.8-flash"
                assert len(body["choices"]) == 1
                message = body["choices"][0]["message"]
                assert message.get("role") == "assistant"
                assert message.get("content") or message.get("tool_calls")
                body_path.write_bytes(raw)
                save(record, {
                    "status": "received", "at": time.time(),
                    "body_sha256": sha(body_path), "model": body["model"],
                    "seconds": time.monotonic() - started, "usage": body.get("usage"),
                })
            except BaseException as exc:
                save(record, {
                    "status": "failed", "at": time.time(),
                    "seconds": time.monotonic() - started,
                    "error_type": type(exc).__name__,
                    "http_status": getattr(exc, "code", None), "automatic_retry": False,
                })
                return self.respond(502, b'{"status":"upstream_native_evaluation_failed"}')
            return self.respond(200, raw)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", CONFIG["port"]), Handler)
    process = Path("/proc") / str(os.getpid())
    save(ROOT / "startup.json", {
        "pid": os.getpid(),
        "start_ticks": (process / "stat").read_text().split(") ", 1)[1].split()[19],
        "command_sha256": sha(process / "cmdline"), "source_sha256": sha(__file__),
        "at": time.time(), "port": CONFIG["port"], "max_inflight": MAX_INFLIGHT,
    })
    server.serve_forever()
