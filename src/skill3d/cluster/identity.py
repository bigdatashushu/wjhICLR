"""Canonical, immutable identities for coordinator jobs. No model imports."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlsplit


class QueueError(ValueError):
    """Base class for rejected queue operations."""


class IdentityMismatch(QueueError):
    pass


class UnsafePayload(QueueError):
    pass


_FORBIDDEN_KEYS = {
    "apikey", "accesskey", "secretkey", "accesskeyid", "awsaccesskeyid",
    "awssecretaccesskey", "password", "passwd", "secret", "clientsecret",
    "credential", "credentials", "authorization", "authentication",
    "accesstoken", "refreshtoken", "authtoken", "bearertoken", "token",
    "cookie", "cookies", "privatekey", "referenceanswer", "referenceanswers",
    "groundtruth", "groundtruthanswer", "groundtruthanswers", "gtanswer",
    "gtanswers", "answerkey", "goldanswer", "goldanswers", "expectedanswer",
    "expectedanswers", "xamzsignature", "xamzcredential", "signature",
}
_SECRET_TEXT = re.compile(
    r"(?i)\bbearer\s+\S+|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|\b(?:api[_-]?key|password|access[_-]?token|client[_-]?secret)\s*[:=]\s*\S+"
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _key(value: str) -> str:
    return "".join(c for c in value.lower() if c.isalnum())


def safe_text(value: str, *, field: str = "text", limit: int = 4096) -> str:
    """Reject obvious secrets; callers must still supply sanitized metadata."""
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise UnsafePayload(f"{field} must be nonempty text of at most {limit} characters")
    if any(ord(c) < 32 for c in value):
        raise UnsafePayload(f"{field} must not contain control characters")
    if _SECRET_TEXT.search(value):
        raise UnsafePayload(f"{field} contains forbidden credential text")
    if "://" in value:
        try:
            uri = urlsplit(value)
            if uri.username is not None or uri.password is not None:
                raise UnsafePayload(f"{field} contains URL credentials")
            if any(_key(k) in _FORBIDDEN_KEYS for k, _ in parse_qsl(uri.query)):
                raise UnsafePayload(f"{field} contains a credential or answer URL parameter")
        except ValueError as exc:
            if isinstance(exc, UnsafePayload):
                raise
            raise UnsafePayload(f"{field} contains an invalid URL") from exc
    return value


def _check_json(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise UnsafePayload("JSON nesting exceeds 64 levels")
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise UnsafePayload("nonfinite JSON number")
        return
    if isinstance(value, str):
        if value:
            safe_text(value, limit=1_000_000)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise UnsafePayload("JSON object keys must be strings")
            if _key(k) in _FORBIDDEN_KEYS:
                raise UnsafePayload("credential/reference-answer fields are forbidden")
            if k:
                safe_text(k, field="JSON key", limit=256)
            _check_json(v, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _check_json(item, depth + 1)
        return
    raise UnsafePayload(f"not a JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    _check_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def check_digest(value: str, field: str) -> str:
    if not isinstance(value, str) or _HEX.fullmatch(value) is None:
        raise IdentityMismatch(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class FrozenIdentity:
    """Manifest and method configuration snapshots, never caller-owned mappings."""

    manifest_json: str
    method_config_json: str
    manifest_hash: str
    method_config_hash: str
    identity_hash: str

    def __post_init__(self) -> None:
        try:
            manifest = json.loads(self.manifest_json)
            method = json.loads(self.method_config_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise IdentityMismatch("identity contains invalid JSON") from exc
        if not isinstance(manifest, dict) or not isinstance(method, dict):
            raise IdentityMismatch("manifest and method configuration must be objects")
        if canonical_json(manifest) != self.manifest_json:
            raise IdentityMismatch("manifest is not canonical")
        if canonical_json(method) != self.method_config_json:
            raise IdentityMismatch("method configuration is not canonical")
        expected_manifest = digest_json(manifest)
        expected_method = digest_json(method)
        if self.manifest_hash != expected_manifest or self.method_config_hash != expected_method:
            raise IdentityMismatch("manifest/method configuration hash mismatch")
        if self.identity_hash != self._combined(expected_manifest, expected_method):
            raise IdentityMismatch("combined identity hash mismatch")

    @staticmethod
    def _combined(manifest_hash: str, method_hash: str) -> str:
        return digest_json({"schema": "skill3d.cluster.identity.v1",
                            "manifest_hash": manifest_hash,
                            "method_config_hash": method_hash})

    @classmethod
    def freeze(cls, manifest: Mapping[str, Any], method_config: Mapping[str, Any],
               *, expected_manifest_hash: str | None = None,
               expected_method_config_hash: str | None = None) -> FrozenIdentity:
        if not isinstance(manifest, Mapping) or not isinstance(method_config, Mapping):
            raise IdentityMismatch("manifest and method configuration must be objects")
        manifest_json = canonical_json(dict(manifest))
        method_json = canonical_json(dict(method_config))
        mh = hashlib.sha256(manifest_json.encode("ascii")).hexdigest()
        ch = hashlib.sha256(method_json.encode("ascii")).hexdigest()
        if expected_manifest_hash is not None and expected_manifest_hash != mh:
            raise IdentityMismatch("expected manifest hash mismatch")
        if expected_method_config_hash is not None and expected_method_config_hash != ch:
            raise IdentityMismatch("expected method configuration hash mismatch")
        return cls(manifest_json, method_json, mh, ch, cls._combined(mh, ch))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "skill3d.cluster.identity.v1",
            "manifest": json.loads(self.manifest_json),
            "method_config": json.loads(self.method_config_json),
            "manifest_hash": self.manifest_hash,
            "method_config_hash": self.method_config_hash,
            "identity_hash": self.identity_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FrozenIdentity:
        required = {"schema", "manifest", "method_config", "manifest_hash",
                    "method_config_hash", "identity_hash"}
        if (not isinstance(value, Mapping) or set(value) != required or
                value["schema"] != "skill3d.cluster.identity.v1"):
            raise IdentityMismatch("invalid frozen identity envelope")
        if not isinstance(value["manifest"], dict) or not isinstance(value["method_config"], dict):
            raise IdentityMismatch("identity snapshots must be JSON objects")
        return cls(canonical_json(value["manifest"]), canonical_json(value["method_config"]),
                   value["manifest_hash"], value["method_config_hash"], value["identity_hash"])
