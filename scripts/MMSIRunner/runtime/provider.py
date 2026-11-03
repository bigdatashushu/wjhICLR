"""One-call authenticated local proxy bridge. Secrets remain in memory."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import urllib.request

LAB = Path("/home/wangy/wangjh/harness3d/four-datasets-20261031")
PLACEHOLDER_ENV = "H3D_MMSI_PROXY_PLACEHOLDER"
def token():
    sys.path.insert(0, str(LAB))
    from transport_auth import load_transport_token
    return load_transport_token()
def health():
    secret = token()
    request = urllib.request.Request("http://127.0.0.1:27431/authenticated-health",
        headers={"Authorization":"Bearer "+secret})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=10) as r:
        raw = r.read()
    if secret.encode() in raw:
        raise ValueError("credential_in_health")
    return json.loads(raw)
class ProxyOpener:
    def __init__(self, operation_id, run_identity, schema, secret=None, transport=None):
        self.operation_id=operation_id; self.run_identity=run_identity; self.schema=schema
        self._secret=secret if secret is not None else token()
        self._transport=transport or urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        self.count=0
    def __call__(self, request, timeout):
        self.count+=1
        payload=json.loads(request.data)
        envelope={"schema_version":self.schema,
            "request_id":hashlib.sha256((self.run_identity+":"+self.operation_id+":"+str(self.count)).encode()).hexdigest(),
            "payload":payload,"audit_identity":{"run_identity":self.run_identity,"operation_id":self.operation_id}}
        proxy=urllib.request.Request("http://127.0.0.1:27431/chat",
            data=json.dumps(envelope,ensure_ascii=False,allow_nan=False).encode(),
            headers={"Authorization":"Bearer "+self._secret,"Content-Type":"application/json"},method="POST")
        with self._transport(proxy,timeout=timeout) as response:
            raw=response.read()
        if self._secret.encode() in raw:
            raise ValueError("credential_in_response")
        return io.BytesIO(raw)
def configure_placeholder():
    # This literal only satisfies BailianClient's configuration check.
    # The bridge replaces the authorization header with in-memory proxy auth.
    os.environ[PLACEHOLDER_ENV]="local-proxy-configuration"
