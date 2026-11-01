"""Authenticated raw payload forwarding through an existing cluster credential."""
from __future__ import annotations

import hashlib
import io
import json
import urllib.request

SCHEMA = "cv3d-native-evaluation-rpc/1"


class ProxyOpener:
    def __init__(self, *, gateway, secret_loader, operation_id, run_identity):
        self.gateway = gateway.rstrip("/")
        self.secret_loader = secret_loader
        self.operation_id = operation_id
        self.run_identity = run_identity
        self.count = 0

    def __call__(self, request, timeout):
        self.count += 1
        secret = self.secret_loader()
        envelope = {
            "schema_version": SCHEMA,
            "request_id": hashlib.sha256(
                f"{self.run_identity}:{self.operation_id}:{self.count}".encode()).hexdigest(),
            "audit_identity": {
                "run_identity": self.run_identity, "operation_id": self.operation_id},
            "payload": json.loads(request.data),
        }
        forward = urllib.request.Request(
            self.gateway + "/chat",
            data=json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
            method="POST")
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                forward, timeout=timeout) as response:
            raw = response.read()
        if secret.encode() in raw:
            raise ValueError("credential_in_response")
        return io.BytesIO(raw)


def health(gateway, secret_loader):
    secret = secret_loader()
    request = urllib.request.Request(
        gateway.rstrip("/") + "/authenticated-health",
        headers={"Authorization": "Bearer " + secret})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request, timeout=10) as response:
        raw = response.read()
    if secret.encode() in raw:
        raise ValueError("credential_in_health")
    value = json.loads(raw)
    if value.get("schema_version") != SCHEMA or value.get("model") != "qwen3.8-flash":
        raise ValueError("gateway_identity_mismatch")
    return value
