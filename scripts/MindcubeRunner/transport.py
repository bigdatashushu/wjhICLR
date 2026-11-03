"""Single-attempt adapter for the existing authenticated evaluation gateway."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error

from .common import sha

SCHEMA = "cv3d-native-evaluation-rpc/1"
AUTH_SHA = "ba590ff1909c6ec3d162aa5009d966278a9ca6324e922077c0145b9056ee23dd"
GATEWAY_SOURCE_SHA = "19e0377251c1ed95379e6b65d828091b79926c37ca2cb3af6802f74c053d5c60"
ENDPOINT_SHA = "5891aed827c4e67b2d7c0c73ea819327ce3f2b6ef72213cd79669486a26b1ead"


def validate_gateway(gateway):
    parsed = urllib.parse.urlsplit(gateway)
    existing_private_gateway = (
        parsed.hostname == "10.10.0.198" and parsed.port == 27441
        and parsed.path in ("", "/"))
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or (parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1")
                and not existing_private_gateway)):
        raise ValueError("authenticated_gateway_requires_tls_or_loopback")


def auth_loader(root):
    path = Path(root).resolve() / "transport_auth.py"
    if sha(path) != AUTH_SHA:
        raise ValueError("transport_auth_changed")
    spec = importlib.util.spec_from_file_location("_mindcube_transport_auth", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_transport_token


def health(gateway, loader):
    validate_gateway(gateway)
    secret = loader()
    request = urllib.request.Request(gateway.rstrip("/") + "/authenticated-health",
                                     headers={"Authorization": "Bearer " + secret})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=10) as response:
        raw = response.read()
    if secret.encode() in raw:
        raise ValueError("credential_in_health")
    value = json.loads(raw)
    if (value.get("schema_version") != SCHEMA or value.get("model") != "qwen3.8-flash"
            or value.get("source_sha256") != GATEWAY_SOURCE_SHA
            or value.get("endpoint_sha256") != ENDPOINT_SHA):
        raise ValueError("gateway_identity_changed")
    return {key: value[key] for key in
            ("schema_version", "model", "source_sha256", "endpoint_sha256")}


class ProxyOpener:
    def __init__(self, gateway, loader, operation_id, run_identity):
        validate_gateway(gateway)
        self.gateway, self.loader = gateway.rstrip("/"), loader
        self.operation_id, self.run_identity, self.count = operation_id, run_identity, 0
        self.trace = None

    def __call__(self, request, timeout):
        self.count += 1
        secret = self.loader()
        if self.trace:
            self.trace.secrets = (*self.trace.secrets, secret)
            self.trace.body("client_payload", request.data)
        value = {"schema_version": SCHEMA, "request_id": hashlib.sha256(
            f"{self.run_identity}:{self.operation_id}:{self.count}".encode()).hexdigest(),
            "audit_identity": {"run_identity": self.run_identity, "operation_id": self.operation_id},
            "payload": json.loads(request.data)}
        envelope = json.dumps(value, allow_nan=False).encode()
        if self.trace:
            self.trace.body("gateway_envelope", envelope)
        forward = urllib.request.Request(
            self.gateway + "/chat", data=envelope,
            headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
            method="POST")
        if self.trace:
            self.trace.event("request_sent", {"request_id": value["request_id"],
                                             "automatic_retry": False})
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                    forward, timeout=timeout) as response:
                raw = response.read()
                http_status = getattr(response, "status", 200)
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read()
                if self.trace:
                    self.trace.body("response_body", raw, http_status=exc.code, complete=True)
            except BaseException as read_error:
                if self.trace and not self.trace.failed:
                    self.trace.event("response_unavailable", {
                        "error_type": type(read_error).__name__, "http_status": exc.code,
                        "partial_bytes_available": False})
                raise
            finally:
                exc.close()
            raise
        except BaseException as exc:
            if self.trace and not self.trace.failed:
                self.trace.event("response_unavailable", {
                    "error_type": type(exc).__name__, "partial_bytes_available": False})
            raise
        if secret.encode() in raw:
            if self.trace:
                self.trace.event("redacted_response", {"reason": "credential_in_response",
                                                      "original_body_persisted": False})
            raise ValueError("credential_in_response")
        if self.trace:
            self.trace.body("response_body", raw, http_status=http_status, complete=True)
        return io.BytesIO(raw)
