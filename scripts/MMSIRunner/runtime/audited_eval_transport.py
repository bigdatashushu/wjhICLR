"""Forward frozen payloads once and retain the original OpenAI response bytes."""
import hashlib
import io
import json
import urllib.request
from provider import token

SCHEMA = "harness3d.audited-evaluation-rpc/1"
GATEWAY = "http://10.10.0.198:27439"


def health():
    secret = token()
    request = urllib.request.Request(GATEWAY + "/authenticated-health",
                                    headers={"Authorization": "Bearer " + secret})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=10) as response:
        raw = response.read()
    assert secret.encode() not in raw
    result = json.loads(raw)
    assert result["schema_version"] == SCHEMA
    assert result["model"] == "qwen3.8-flash"
    assert result["endpoint_sha256"] == "5891aed827c4e67b2d7c0c73ea819327ce3f2b6ef72213cd79669486a26b1ead"
    return result


class ProxyOpener:
    def __init__(self, operation_id, run_identity, schema, secret=None, transport=None):
        self.operation_id = operation_id
        self.run_identity = run_identity
        self._secret = secret if secret is not None else token()
        self._transport = transport or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        self.count = 0

    def __call__(self, request, timeout):
        self.count += 1
        payload = json.loads(request.data)
        request_id = hashlib.sha256(
            (self.run_identity + ":" + self.operation_id + ":" + str(self.count)).encode()).hexdigest()
        envelope = {"schema_version": SCHEMA, "request_id": request_id, "payload": payload,
                    "audit_identity": {"run_identity": self.run_identity, "operation_id": self.operation_id}}
        forward = urllib.request.Request(
            GATEWAY + "/chat", data=json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode(),
            headers={"Authorization": "Bearer " + self._secret, "Content-Type": "application/json"})
        with self._transport(forward, timeout=timeout) as response:
            raw = response.read()
        assert self._secret.encode() not in raw
        return io.BytesIO(raw)
