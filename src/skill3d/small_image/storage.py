"""Content-verified, pickle-free persistence of still-image prepared evidence."""
from __future__ import annotations

import base64
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .precompute import PreparedSmallImage, _prepared, validate_prepared_content

SCHEMA = "small-image-prepared-storage/1"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_prepared(prepared: PreparedSmallImage, directory: str | Path) -> dict:
    """Publish complete JSON + NPZ atomically; existing evidence is never replaced."""
    validate_prepared_content(prepared)
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        existing = load_prepared(directory)
        if existing.artifact_sha256 != prepared.artifact_sha256:
            raise ValueError("existing prepared evidence differs")
        return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    arrays, identities = {}, {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError("object arrays are forbidden")
            key = identities.get(id(value))
            if key is None:
                key = f"a{len(arrays)}"
                identities[id(value)] = key
                arrays[key] = value
            return ["array", key]
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, Mapping):
            return ["mapping", [[encode(k), encode(v)] for k, v in value.items()]]
        if isinstance(value, tuple):
            return ["tuple", [encode(item) for item in value]]
        if isinstance(value, list):
            return ["list", [encode(item) for item in value]]
        if isinstance(value, bytes):
            return ["bytes", base64.b64encode(value).decode("ascii")]
        if isinstance(value, Path):
            return ["path", str(value)]
        if isinstance(value, float):
            return ["float", value.hex()]
        if value is None or isinstance(value, (bool, int, str)):
            return ["scalar", value]
        raise TypeError("unsupported prepared storage value")

    fields = {name: getattr(prepared, name)
              for name in PreparedSmallImage.__dataclass_fields__
              if name != "artifact_sha256"}
    tree = encode(fields)
    temporary = Path(tempfile.mkdtemp(prefix=".prepared-", dir=directory.parent))
    try:
        np.savez_compressed(temporary / "arrays.npz", **arrays)
        metadata = {
            "schema": SCHEMA, "artifact_sha256": prepared.artifact_sha256,
            "input_sha256": prepared.problem["input_sha256"],
            "cache_key": prepared.cache_key, "fields": tree,
            "arrays_sha256": _sha(temporary / "arrays.npz"),
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=True, allow_nan=False,
                       separators=(",", ":")) + "\n", encoding="utf-8")
        # Validate serialized bytes before publishing the directory.
        load_prepared(temporary, problem=prepared.problem)
        temporary.rename(directory)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_prepared(directory: str | Path, *, problem=None) -> PreparedSmallImage:
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if (set(metadata) != {"schema", "artifact_sha256", "input_sha256", "cache_key",
                         "fields", "arrays_sha256"} or metadata["schema"] != SCHEMA):
        raise ValueError("invalid prepared storage schema")
    if _sha(directory / "arrays.npz") != metadata["arrays_sha256"]:
        raise ValueError("prepared array file digest mismatch")
    referenced = set()
    with np.load(directory / "arrays.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("object arrays are forbidden")

    def decode(row):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("invalid prepared storage node")
        kind, value = row
        if kind == "array":
            referenced.add(value)
            return arrays[value]
        if kind == "mapping":
            result = {}
            for key, item in value:
                key = decode(key)
                if key in result:
                    raise ValueError("duplicate prepared mapping key")
                result[key] = decode(item)
            return result
        if kind in ("tuple", "list"):
            decoded = [decode(item) for item in value]
            return tuple(decoded) if kind == "tuple" else decoded
        if kind == "bytes":
            return base64.b64decode(value, validate=True)
        if kind == "path":
            return Path(value)
        if kind == "float":
            return float.fromhex(value)
        if kind == "scalar" and (value is None or isinstance(value, (bool, int, str))):
            return value
        raise ValueError("unknown prepared storage node")

    fields = decode(metadata["fields"])
    expected = set(PreparedSmallImage.__dataclass_fields__) - {"artifact_sha256"}
    if set(fields) != expected or referenced != set(arrays):
        raise ValueError("prepared storage field or array mismatch")
    prepared = _prepared(**fields)
    if (prepared.artifact_sha256 != metadata["artifact_sha256"]
            or prepared.cache_key != metadata["cache_key"]
            or prepared.problem["input_sha256"] != metadata["input_sha256"]):
        raise ValueError("prepared content digest mismatch")
    validate_prepared_content(prepared)
    if problem is not None:
        public = problem if isinstance(problem, Mapping) else problem.public_dict()
        if dict(prepared.problem) != dict(public):
            raise ValueError("prepared public input identity mismatch")
    return prepared
