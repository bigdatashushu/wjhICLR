"""Single-attempt forwarding through the already running authenticated gateway."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import urllib.request

from skill3d.synthesis.request_governor import RequestBudgetExhausted
from .common import digest

SCHEMA = "cv3d-native-evaluation-rpc/1"
AUTH_SHA = "ba590ff1909c6ec3d162aa5009d966278a9ca6324e922077c0145b9056ee23dd"
GATEWAY_SOURCE_SHA = "19e0377251c1ed95379e6b65d828091b79926c37ca2cb3af6802f74c053d5c60"
ENDPOINT_SHA = "5891aed827c4e67b2d7c0c73ea819327ce3f2b6ef72213cd79669486a26b1ead"


class ProxyOpener:
    def __init__(self, gateway, loader, operation_id, run_identity):
        self.gateway, self.loader = gateway.rstrip("/"), loader
        self.operation_id, self.run_identity = operation_id, run_identity
        self.count = 0

    def __call__(self, request, timeout):
        self.count += 1
        secret = self.loader()
        value = {
            "schema_version": SCHEMA,
            "request_id": hashlib.sha256(
                f"{self.run_identity}:{self.operation_id}:{self.count}".encode()).hexdigest(),
            "audit_identity": {"run_identity": self.run_identity, "operation_id": self.operation_id},
            "payload": json.loads(request.data)}
        forward = urllib.request.Request(
            self.gateway + "/chat", data=json.dumps(value, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json"},
            method="POST")
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                forward, timeout=timeout) as response:
            raw = response.read()
        if secret.encode() in raw:
            raise ValueError("credential_in_response")
        return io.BytesIO(raw)


def health(gateway, loader):
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
    return value


class DualBudget:
    """Config-derived local cap and existing common cap20000, without retries."""
    def __init__(self, local, shared, stop_file, campaign, arm, sample_id, run_identity, *,
                 reserved_cap=None, aggregate_policy=None):
        self.local, self.shared, self.stop = local, shared, Path(stop_file)
        self.context = {"variant_id": arm, "slot_key": sample_id,
                        "campaign_id": campaign, "run_identity": run_identity}
        self.reserved_cap = reserved_cap
        self.aggregate_policy = aggregate_policy

    @contextmanager
    def request(self, category="solver"):
        if category not in ("solver", "gate"):
            raise ValueError("omni_request_category_invalid")
        if self.stop.exists():
            raise RequestBudgetExhausted("run_admission_stopped")
        from .budget import admission, audit
        accounting = audit(self.shared.root.parents[1])
        if self.aggregate_policy is None and accounting["used"] >= accounting["global_cap"]:
            raise RequestBudgetExhausted("aggregate_global_budget_exhausted")
        if self.reserved_cap is not None:
            remaining = max(0, self.reserved_cap - accounting["shared_campaign_counts"].get(
                self.context["campaign_id"], 0))
            admission(accounting, remaining, self.aggregate_policy)
        with self.local.request(category, self.context) as local:
            if self.stop.exists():
                raise RequestBudgetExhausted("run_admission_stopped")
            with self.shared.request(category, self.context) as shared:
                try:
                    yield local
                finally:
                    shared["usage"] = local["usage"]
