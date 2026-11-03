"""Authenticated, run-scoped receipt reader. Never reads response bodies."""
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from transport_auth import load_transport_token

TOKEN = load_transport_token()
RUN = "92a55aecff8c3e79090ef2b4fdfd02132185e55bca006479de332eafc04edbb2"
SCHEMA = "harness3d.timeout-receipt-reader/1"
GATEWAY_SHA = "61e7166812e170dd6e2e3446e611c0e7f3a976cf2ee99a5c686c79f06bea09d1"
CACHE = ROOT / "audited_eval_gateway"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    assert sha(ROOT / "audited_eval_gateway_timeout06.py") == GATEWAY_SHA
    start = json.loads((ROOT / "audited_eval_gateway_timeout06_startup.json").read_text())
    assert start["source_sha256"] == GATEWAY_SHA and start["port"] == 27440
    proc = Path("/proc") / str(start["pid"])
    fields = (proc / "stat").read_text().split(") ", 1)[1].split()
    assert fields[19] == str(start["start_ticks"]) and fields[0] not in {"Z", "X"}
    assert sha(proc / "cmdline") == start["command_sha256"]
    return {"schema_version": SCHEMA, "reader_source_sha256": sha(__file__),
            "gateway_source_sha256": GATEWAY_SHA, "gateway_process_verified": True,
            "run_identity": RUN, "model_requests": 0, "response_bodies_read": False}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, code, value):
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        assert TOKEN.encode() not in raw
        self.send_response(code)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + TOKEN):
            return self.respond(403, {"status": "unauthorized"})
        try:
            base = identity()
            if self.path == "/authenticated-health":
                return self.respond(200, {"status": "ready", **base})
            match = re.fullmatch(r"/request-receipt/([0-9a-f]{64})", self.path)
            if not match:
                return self.respond(404, {"status": "not_found"})
            request_id = match[1]
            intent_path = CACHE / (request_id + ".intent.json")
            response_path = CACHE / (request_id + ".response.json")
            if not intent_path.exists() or not response_path.exists():
                return self.respond(404, {"status": "receipt_not_terminal"})
            intent = json.loads(intent_path.read_text())
            assert intent["request_id"] == request_id
            assert intent["audit_identity"]["run_identity"] == RUN
            response = json.loads(response_path.read_text())
            clean = {key: response.get(key) for key in (
                "status", "at", "error_type", "automatic_retry", "service_error", "body_sha256")}
            self.respond(200, {"status": "receipt_available", **base, "request_id": request_id,
                              "intent": intent, "response": clean,
                              "intent_sha256": sha(intent_path), "response_sha256": sha(response_path)})
        except Exception:
            self.respond(503, {"status": "receipt_verification_unavailable"})


identity()
server = ThreadingHTTPServer(("0.0.0.0", 27442), Handler)
proc = Path("/proc") / str(os.getpid())
fields = (proc / "stat").read_text().split(") ", 1)[1].split()
with (ROOT / "timeout_receipt_reader07_startup.json").open("x") as stream:
    json.dump({"pid": os.getpid(), "start_ticks": fields[19], "command_sha256": sha(proc / "cmdline"),
               "source_sha256": sha(__file__), "at": time.time(), "port": 27442,
               "model_requests": 0, "response_bodies_read": False}, stream)
server.serve_forever()
