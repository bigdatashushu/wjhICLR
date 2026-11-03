"""Apply only the authorized runtime wait extension to frozen MMSI clients."""
from copy import deepcopy
import hashlib
import io
import json
import urllib.request

import pilot
import provider
import timeout_transport06

UPSTREAM_TIMEOUT_S = 1200
CLIENT_TIMEOUT_S = 1230
GATEWAY_SOURCE_SHA = "61e7166812e170dd6e2e3446e611c0e7f3a976cf2ee99a5c686c79f06bea09d1"
FrozenTraceClient = pilot.TraceClient


class ExtendedTraceClient(FrozenTraceClient):
    def __init__(self, job):
        assert job["model"]["timeout_s"] == 600
        super().__init__(job)
        self.timeout_s = float(CLIENT_TIMEOUT_S)


def apply():
    provider.ProxyOpener = timeout_transport06.ProxyOpener
    pilot.TraceClient = ExtendedTraceClient


def zero_request_check(job):
    """Exercise the real transport with an in-memory response and no network."""
    before = pilot.sha_value(job)
    captured = []
    response_body = b'{"synthetic_transport_check":true}'

    def fake_forward(request, timeout):
        captured.append((request.full_url, json.loads(request.data), timeout))
        return io.BytesIO(response_body)

    old_proxy = provider.ProxyOpener
    try:
        provider.ProxyOpener = lambda operation, run, schema: timeout_transport06.ProxyOpener(
            operation, run, schema, secret="offline_transport_test", transport=fake_forward)
        frozen = FrozenTraceClient(deepcopy(job))
        extended = ExtendedTraceClient(deepcopy(job))
        assert frozen.timeout_s == 600 and extended.timeout_s == CLIENT_TIMEOUT_S
        messages = [{"role": "user", "content": "wait configuration check"}]
        kwargs = dict(max_tokens=8192, temperature=0, seed=137, enable_thinking=True)
        payload = frozen.build_payload(messages, **kwargs)
        assert extended.build_payload(messages, **kwargs) == payload
        request = urllib.request.Request(
            "https://offline.invalid/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode())
        with extended._opener(request, timeout=extended.timeout_s) as response:
            assert response.read() == response_body
        assert len(captured) == 1
        url, envelope, timeout = captured[0]
        assert url == timeout_transport06.GATEWAY + "/chat" and timeout == CLIENT_TIMEOUT_S
        assert envelope["payload"] == payload
        assert envelope["audit_identity"] == {
            "run_identity": job["run_identity"], "operation_id": job["operation_id"]}
        assert envelope["request_id"] == hashlib.sha256(
            (job["run_identity"] + ":" + job["operation_id"] + ":1").encode()).hexdigest()
        assert pilot.sha_value(job) == before
    finally:
        provider.ProxyOpener = old_proxy
    return {
        "status": "passed", "model_requests": 0, "network_requests": 0,
        "upstream_timeout_s": UPSTREAM_TIMEOUT_S, "client_timeout_s": CLIENT_TIMEOUT_S,
        "payload_and_request_identity_unchanged": True, "response_bytes_unchanged": True,
        "job_identity_unchanged": True,
    }
