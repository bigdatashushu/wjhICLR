"""Validate coordinator-local staging files before the SQLite completion commit."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .identity import QueueError, canonical_json, check_digest, digest_json, safe_text


class PublicationError(QueueError):
    pass


class PublicationConflict(PublicationError):
    pass


@dataclass(frozen=True)
class ArtifactDigest:
    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        safe_text(self.name, field="artifact name", limit=512)
        path = PurePosixPath(self.name)
        if (path.is_absolute() or "\\" in self.name or
                any(part in ("", ".", "..") for part in self.name.split("/"))):
            raise PublicationError("artifact names must be normalized relative POSIX paths")
        check_digest(self.sha256, "artifact sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise PublicationError("artifact size_bytes must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class PublicationReceipt:
    job_key: str
    completed_key: str
    identity_hash: str
    stage: str
    attempt: int
    validator: str
    artifacts: tuple[ArtifactDigest, ...]

    def __post_init__(self) -> None:
        for field in ("job_key", "completed_key", "stage", "validator"):
            safe_text(getattr(self, field), field=field, limit=512)
        check_digest(self.identity_hash, "identity_hash")
        if type(self.attempt) is not int or self.attempt < 1:
            raise PublicationError("receipt attempt must be a positive integer")
        if not isinstance(self.artifacts, tuple) or not self.artifacts:
            raise PublicationError("a completed job must publish at least one artifact")
        if not all(isinstance(item, ArtifactDigest) for item in self.artifacts):
            raise PublicationError("artifacts must be ArtifactDigest values")
        names = [item.name for item in self.artifacts]
        if len(set(names)) != len(names):
            raise PublicationError("duplicate artifact names")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "skill3d.cluster.publication.v1", "job_key": self.job_key,
            "completed_key": self.completed_key, "identity_hash": self.identity_hash,
            "stage": self.stage, "attempt": self.attempt,
            "validation": {"status": "validated", "validator": self.validator},
            "artifacts": [a.to_dict() for a in sorted(self.artifacts, key=lambda a: a.name)],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PublicationReceipt:
        fields = {"schema", "job_key", "completed_key", "identity_hash", "stage",
                  "attempt", "validation", "artifacts"}
        if set(data) != fields or data["schema"] != "skill3d.cluster.publication.v1":
            raise PublicationError("invalid publication receipt envelope")
        validation = data["validation"]
        if (not isinstance(validation, dict) or set(validation) != {"status", "validator"}
                or validation["status"] != "validated"):
            raise PublicationError("receipt does not attest validated output")
        if not isinstance(data["artifacts"], list):
            raise PublicationError("artifacts must be a list")
        artifacts = []
        for item in data["artifacts"]:
            if not isinstance(item, dict) or set(item) != {"name", "sha256", "size_bytes"}:
                raise PublicationError("invalid artifact digest")
            artifacts.append(ArtifactDigest(**item))
        return cls(data["job_key"], data["completed_key"], data["identity_hash"],
                   data["stage"], data["attempt"], validation["validator"], tuple(artifacts))


@dataclass(frozen=True)
class InspectedPublication:
    receipt: PublicationReceipt
    receipt_hash: str

    @property
    def fingerprint(self) -> str:
        return digest_json({"receipt_hash": self.receipt_hash,
                            "receipt": self.receipt.to_dict()})


def hash_file(path: str | Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise PublicationError("artifact must be a regular file")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise PublicationError("file changed during hashing")
    return digest.hexdigest(), size


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PublicationError("duplicate receipt JSON key")
        result[key] = value
    return result


def inspect_receipt(path: str | Path, *, expected_receipt_hash: str) -> InspectedPublication:
    check_digest(expected_receipt_hash, "expected_receipt_hash")
    try:
        with Path(path).open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise PublicationError("receipt exceeds 1 MiB")
        receipt_hash = hashlib.sha256(raw).hexdigest()
        if receipt_hash != expected_receipt_hash:
            raise PublicationError("receipt content hash mismatch")
        data = json.loads(raw, object_pairs_hook=_unique_object)
        canonical_json(data)
        if not isinstance(data, dict):
            raise PublicationError("receipt must be a JSON object")
        receipt = PublicationReceipt.from_dict(data)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicationError("cannot read a valid receipt") from exc
    return InspectedPublication(receipt, receipt_hash)


def validate_artifacts(publication: InspectedPublication, artifact_root: str | Path) -> None:
    """The root must be a trusted, immutable coordinator-local publication directory."""
    try:
        root = Path(artifact_root).resolve(strict=True)
    except OSError as exc:
        raise PublicationError("cannot read publication root") from exc
    for artifact in publication.receipt.artifacts:
        try:
            path = root.joinpath(*PurePosixPath(artifact.name).parts).resolve(strict=True)
            if not path.is_relative_to(root):
                raise PublicationError("artifact escapes publication root")
            digest, size = hash_file(path)
        except OSError as exc:
            raise PublicationError("cannot read published artifact") from exc
        if (digest, size) != (artifact.sha256, artifact.size_bytes):
            raise PublicationError(f"artifact content mismatch: {artifact.name}")
