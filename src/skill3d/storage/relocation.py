"""Explicit relocation receipts for byte-preserving reconstruction copies.

``scene-cache-relocation-v1`` contains a complete ``logical_input_identity``
(the existing evaluation input-identity schema), an optional ``binding_identity``
anchor, a canonical absolute ``destination_root``, and ``artifact`` /
``dependencies`` records. Each record binds ``source_ref``, ``source_path``,
``identity`` (SHA256 and size), and a root-relative ``target_ref``.
``receipt_sha256`` hashes the canonical receipt body, not an authentication
signature. Consumers must pin the old input hash and, for bindings, the immutable
manifest identity. Rehashing a receipt cannot authorize different input bytes.

Only artifact/dependency locations are relocated. Episode, source, video and
frame metadata stay frozen. No original path needs to exist on the destination
host, and no artifact or prepared manifest is rewritten. Relative source refs
must be unambiguous canonical paths without traversal; v1 rejects unsafe legacy
refs rather than guessing their meaning.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np

from skill3d.infra.m5_identity import canonical_bytes, content_sha256, file_identity
from skill3d.schemas import FrameSet, ReconstructionArtifact

RELOCATION_SCHEMA_VERSION = "scene-cache-relocation-v1"
ARTIFACT_REFS = (
    "c2w_list", "intrinsics", "depth_maps", "point_map", "point_conf",
    "depth_conf", "track_list", "per_frame_scale_ref",
    "metric_depth_maps", "metric_valid_masks",
)
_INPUT_FIELDS = {
    "episode", "source", "video_path", "pixels", "artifact_path",
    "artifact_json_sha256", "artifact_id", "artifact_version", "files", "sha256",
}
_RECEIPT_FIELDS = {
    "schema_version", "logical_input_identity", "binding_identity",
    "destination_root", "artifact", "dependencies", "receipt_sha256",
}
_RECORD_FIELDS = {"source_ref", "source_path", "identity", "target_ref"}


class RelocationError(ValueError):
    """A relocation lacks a trusted identity or complete, intact local files."""


def _json_copy(value: Mapping) -> dict:
    try:
        result = json.loads(canonical_bytes(dict(value)))
    except (TypeError, ValueError) as exc:
        raise RelocationError("relocation data is not canonical JSON") from exc
    return result


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RelocationError("relocation identity is not SHA256")
    return value


def _ref(value: object, *, absolute: bool = False) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise RelocationError(f"unsafe relocation reference: {value!r}")
    path = PurePosixPath(value)
    if (
        str(path) != value
        or value == "/"
        or value.startswith("//")
        or any(part in (".", "..") or ":" in part or part.startswith("~")
               for part in value.split("/") if part)
        or (absolute and not path.is_absolute())
    ):
        raise RelocationError(f"unsafe relocation reference: {value!r}")
    return value


def _no_symlinks(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise RelocationError(f"symlink is forbidden in relocation: {candidate}")


def _root(value: str | Path) -> Path:
    try:
        path = Path(value).expanduser().absolute()
        _no_symlinks(path)
        if not path.is_dir() or path == path.parent:
            raise RelocationError(f"relocation root is not a directory: {path}")
        return path.resolve()
    except RelocationError:
        raise
    except (OSError, ValueError) as exc:
        raise RelocationError(f"invalid relocation root: {value}") from exc


def _target(root: Path, value: object) -> Path:
    ref = _ref(value)
    if PurePosixPath(ref).is_absolute():
        raise RelocationError(f"target reference must be root-relative: {ref}")
    path = root / ref
    try:
        _no_symlinks(path)
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise RelocationError(f"missing or escaping relocation target: {ref}")
        return resolved
    except RelocationError:
        raise
    except (OSError, ValueError) as exc:
        raise RelocationError(f"invalid relocation target: {ref}") from exc


def _identity(path: Path) -> dict:
    try:
        return file_identity(path)
    except (OSError, ValueError) as exc:
        raise RelocationError(f"relocation file cannot be hashed: {path}") from exc


def _check_file_identity(value: object) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {"sha256", "size"}
        or type(value["size"]) is not int
        or value["size"] < 0
    ):
        raise RelocationError("relocation file identity requires SHA256 and size")
    _sha256(value["sha256"])
    return value


def _logical_identity(value: Mapping, expected_sha256: str) -> dict:
    logical = _json_copy(value)
    if set(logical) != _INPUT_FIELDS:
        raise RelocationError("relocation requires a full scene input identity")
    claimed = _sha256(logical["sha256"])
    body = {name: child for name, child in logical.items() if name != "sha256"}
    if claimed != _sha256(expected_sha256) or content_sha256(body) != claimed:
        raise RelocationError("logical scene input identity does not replay")
    _ref(logical["artifact_path"], absolute=True)
    _sha256(logical["artifact_json_sha256"])
    if (
        not isinstance(logical["episode"], dict)
        or not isinstance(logical["pixels"], list)
        or not logical["pixels"]
        or not isinstance(logical["files"], dict)
        or not isinstance(logical["artifact_id"], str)
        or not logical["artifact_id"]
        or not isinstance(logical["artifact_version"], str)
        or not logical["artifact_version"]
        or not isinstance(logical["source"], str)
        or logical["video_path"] is not None
        and not isinstance(logical["video_path"], str)
    ):
        raise RelocationError("incomplete logical scene input identity")
    for row in logical["pixels"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"shape", "dtype", "sha256"}
            or not isinstance(row["shape"], list)
            or len(row["shape"]) != 3
            or any(type(size) is not int or size <= 0 for size in row["shape"])
            or row["shape"][2] != 3
            or not isinstance(row["dtype"], str)
            or not row["dtype"]
        ):
            raise RelocationError("invalid logical RGB pixel identity")
        _sha256(row["sha256"])
    for row in logical["files"].values():
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise RelocationError("invalid logical dependency identity")
        _ref(row["path"], absolute=True)
        _sha256(row["sha256"])
    return logical


def _artifact(path: Path, logical: Mapping) -> ReconstructionArtifact:
    try:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != logical["artifact_json_sha256"]:
            raise RelocationError("relocated artifact bytes changed")
        art = ReconstructionArtifact.model_validate_json(data)
    except RelocationError:
        raise
    except (OSError, ValueError) as exc:
        raise RelocationError(f"invalid relocated artifact: {path}: {exc}") from exc
    if (
        art.artifact_id != logical["artifact_id"]
        or art.artifact_version != logical["artifact_version"]
    ):
        raise RelocationError("relocated artifact logical identity changed")
    return art


def artifact_dependency_refs(artifact: ReconstructionArtifact) -> dict[str, str]:
    """Enumerate every declared scene reference, including confidence/metric refs."""
    refs = {name: getattr(artifact, name) for name in ARTIFACT_REFS}
    refs.update({
        f"confidence.{name}": ref
        for name, ref in artifact.confidence.model_dump().items()
    })
    return {name: _ref(ref) for name, ref in refs.items() if ref}


def _source_records(artifact: ReconstructionArtifact, logical: Mapping) -> dict:
    refs = artifact_dependency_refs(artifact)
    if set(refs) != set(logical["files"]):
        raise RelocationError("logical identity has missing or extra dependencies")
    for name, ref in refs.items():
        source = logical["files"][name]["path"]
        if (
            PurePosixPath(ref).is_absolute() and ref != source
            or not PurePosixPath(ref).is_absolute() and not source.endswith("/" + ref)
        ):
            raise RelocationError(f"logical dependency reference changed: {name}")
    return refs


def read_relocation_receipt(receipt: Mapping | str | Path) -> dict:
    """Read a JSON receipt without accepting duplicate object keys."""
    if isinstance(receipt, Mapping):
        return _json_copy(receipt)

    def unique_pairs(pairs):
        value = {}
        for name, child in pairs:
            if name in value:
                raise RelocationError(f"duplicate relocation receipt field: {name}")
            value[name] = child
        return value

    try:
        value = json.loads(
            Path(receipt).read_text(encoding="utf-8"),
            object_pairs_hook=unique_pairs,
        )
        if not isinstance(value, dict):
            raise RelocationError("relocation receipt must be an object")
        return _json_copy(value)
    except (OSError, TypeError, ValueError) as exc:
        raise RelocationError(f"relocation receipt is unreadable: {receipt}") from exc


def create_relocation_receipt(
    *,
    logical_input_identity: Mapping,
    destination_root: str | Path,
    artifact_ref: str,
    dependency_refs: Mapping[str, str],
    binding_identity: Mapping | None = None,
) -> dict:
    """Describe an already copied scene; all destination bytes are checked now.

    ``dependency_refs`` maps every ``artifact_dependency_refs`` name to an
    explicit root-relative destination. A full saved input identity is required;
    a hash-only legacy reference cannot safely recover dependency identities.
    Sources may already be gone. The new bytes must match the old anchored hashes.
    Persist the returned JSON separately from immutable cache entries.
    """
    logical = _logical_identity(
        logical_input_identity, logical_input_identity.get("sha256", "")
    )
    root = _root(destination_root)
    path = _target(root, artifact_ref)
    art = _artifact(path, logical)
    refs = _source_records(art, logical)
    if set(dependency_refs) != set(refs):
        raise RelocationError("relocation has missing or extra dependencies")
    dependencies = {}
    for name, source_ref in refs.items():
        target_ref = dependency_refs[name]
        identity = _identity(_target(root, target_ref))
        if identity["sha256"] != logical["files"][name]["sha256"]:
            raise RelocationError(f"relocation dependency integrity changed: {name}")
        dependencies[name] = {
            "source_ref": source_ref,
            "source_path": logical["files"][name]["path"],
            "identity": identity,
            "target_ref": target_ref,
        }
    body = {
        "schema_version": RELOCATION_SCHEMA_VERSION,
        "logical_input_identity": logical,
        "binding_identity": dict(binding_identity) if binding_identity is not None else None,
        "destination_root": str(root),
        "artifact": {
            "source_ref": logical["artifact_path"],
            "source_path": logical["artifact_path"],
            "identity": _identity(path),
            "target_ref": artifact_ref,
        },
        "dependencies": dependencies,
    }
    receipt = {**body, "receipt_sha256": content_sha256(body)}
    validate_relocation_receipt(
        receipt,
        expected_input_sha256=logical["sha256"],
        expected_binding_identity=binding_identity,
    )
    return receipt


class ValidatedRelocation:
    """A checked scene resolver; resolving a ref rechecks its local file bytes."""

    def __init__(self, receipt: dict) -> None:
        self._receipt = deepcopy(receipt)
        self._root = Path(self._receipt["destination_root"])
        self._records = {
            "artifact": self._receipt["artifact"], **self._receipt["dependencies"],
        }
        self._aliases = {}
        for name, record in self._records.items():
            for alias in (record["source_ref"], record["source_path"]):
                previous = self._aliases.get(alias)
                if previous is not None and (
                    self._records[previous]["target_ref"] != record["target_ref"]
                    or self._records[previous]["identity"] != record["identity"]
                ):
                    raise RelocationError(f"ambiguous relocation reference: {alias}")
                self._aliases[alias] = name

    @property
    def logical_input_identity(self) -> dict:
        return deepcopy(self._receipt["logical_input_identity"])

    def _path(self, name: str) -> Path:
        record = self._records[name]
        _no_symlinks(self._root)
        path = _target(self._root, record["target_ref"])
        if _identity(path) != record["identity"]:
            raise RelocationError(f"relocation dependency integrity changed: {name}")
        return path

    @property
    def artifact_path(self) -> Path:
        return self._path("artifact")

    @property
    def dependency_paths(self) -> dict[str, Path]:
        return {name: self._path(name) for name in self._receipt["dependencies"]}

    def resolve(self, reference: str) -> Path:
        """Resolve only an exact declared old ref/path, never a prefix rewrite."""
        ref = _ref(reference)
        if ref not in self._aliases:
            raise RelocationError(f"undeclared relocation reference: {ref}")
        return self._path(self._aliases[ref])

    def resolved_artifact(self) -> ReconstructionArtifact:
        """Return a runtime model with verified local refs; no JSON is changed."""
        art = _artifact(self.artifact_path, self._receipt["logical_input_identity"])
        refs = artifact_dependency_refs(art)
        updates = {
            name: str(self.resolve(ref))
            for name, ref in refs.items() if name in ARTIFACT_REFS
        }
        updates["confidence"] = art.confidence.model_copy(update={
            name.removeprefix("confidence."): str(self.resolve(ref))
            for name, ref in refs.items() if name.startswith("confidence.")
        })
        return art.model_copy(update=updates)

    def validate_input_identity(self, identity: Mapping) -> dict:
        """Reject hash-only assertions and any changed frozen input field."""
        logical = self._receipt["logical_input_identity"]
        actual = _logical_identity(identity, logical["sha256"])
        if actual != logical:
            raise RelocationError("relocated scene input identity changed")
        for name in self._records:
            self._path(name)
        return actual

    def recompute_scene_input_identity(
        self,
        *,
        episode: Mapping,
        source: object,
        video_path: str | None,
        pixels: Sequence[np.ndarray],
    ) -> dict:
        """Recompute the old logical identity from actual local scene/pixel bytes.

        Pass frozen ``item.episode.model_dump(mode="json")``, ``item.source``,
        ``item.video_path`` and actual ``item.pixels``. Only the explicitly
        receipted cache paths are represented by their old logical locations.
        Metadata, pixel order/content, all geometry and metric evidence are
        checked; no source-host files are opened.
        """
        from skill3d.adapters.frame_set import frame_set_hash, with_pixel_identity

        logical = self._receipt["logical_input_identity"]
        if (
            dict(episode) != logical["episode"]
            or source != logical["source"]
            or video_path != logical["video_path"]
        ):
            raise RelocationError("relocated scene input metadata changed")
        try:
            fs = FrameSet.model_validate(episode["frame_set"])
            art = self.resolved_artifact()
            if (
                not episode.get("qa_id")
                or not episode.get("dataset")
                or not episode.get("scene_name")
                or fs.frame_set_hash != frame_set_hash(fs.frame_ids)
                or art.scene_name != episode["scene_name"]
                or art.frame_set_hash != fs.frame_set_hash
            ):
                raise RelocationError("relocated scene/frame identity mismatch")
            slots = [fs.frame_ids.index(fid) for fid in fs.readable_frame_ids]
            for name, expected in (
                ("frame_ids", fs.readable_frame_ids),
                ("source_frame_indices", [fs.source_frame_indices[i] for i in slots]),
                ("timestamps", [fs.timestamps[i] for i in slots]),
            ):
                if list(getattr(art, name)) != list(expected):
                    raise RelocationError(f"relocated artifact {name} mismatch")
            for name, expected in (
                ("dataset_id", episode["dataset"]),
                ("scene_name", episode["scene_name"]),
                ("episode_id", episode["qa_id"]),
            ):
                actual = getattr(fs, name)
                if actual and actual != expected:
                    raise RelocationError(f"relocated FrameSet {name} mismatch")
            if len(pixels) != len(slots) or any(
                p.ndim != 3 or p.shape[2] != 3 or p.size == 0 for p in pixels
            ):
                raise RelocationError("relocated RGB pixel count/shape mismatch")
            actual_frames = with_pixel_identity(fs, pixels)
            paths = self.dependency_paths
            for name in ("depth_maps", "c2w_list", "point_map", "depth_conf"):
                if name in paths:
                    array = np.load(paths[name], mmap_mode="r", allow_pickle=False)
                    if array.ndim == 0 or array.shape[0] != len(slots):
                        raise RelocationError(f"relocated {name} frame dimension mismatch")
            if art.metric_model not in (None, "none"):
                from skill3d.reconstruction.metric_evidence import validated_receipt

                if actual_frames.cache_identity() != art.metric_input_identity:
                    raise RelocationError("relocated metric input pixels mismatch")
                validated_receipt(art)
            body = {
                "episode": dict(episode),
                "source": source,
                "video_path": video_path,
                "pixels": [
                    {"shape": list(p.shape), "dtype": str(p.dtype),
                     "sha256": hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()}
                    for p in pixels
                ],
                "artifact_path": logical["artifact_path"],
                "artifact_json_sha256": _identity(self.artifact_path)["sha256"],
                "artifact_id": art.artifact_id,
                "artifact_version": art.artifact_version,
                "files": {
                    name: {"path": logical["files"][name]["path"],
                           "sha256": _identity(path)["sha256"]}
                    for name, path in sorted(paths.items())
                },
            }
            return self.validate_input_identity({**body, "sha256": content_sha256(body)})
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RelocationError(f"cannot recompute relocated scene input: {exc}") from exc


def validate_relocation_receipt(
    receipt: Mapping | str | Path,
    *,
    expected_input_sha256: str,
    expected_binding_identity: Mapping | None = None,
    expected_artifact_identity: Mapping | None = None,
    expected_source_artifact: str | None = None,
    destination_root: str | Path | None = None,
) -> ValidatedRelocation:
    """Validate schema, trusted anchors, complete ref closure and every local byte.

    ``destination_root`` optionally pins the worker-approved physical root.
    Returns a generic resolver usable by geometry loaders. A receipt digest alone
    is never sufficient: ``expected_input_sha256`` comes from a trusted caller.
    """
    data = read_relocation_receipt(receipt)
    if set(data) != _RECEIPT_FIELDS or data["schema_version"] != RELOCATION_SCHEMA_VERSION:
        raise RelocationError("unsupported relocation receipt schema")
    body = {name: child for name, child in data.items() if name != "receipt_sha256"}
    if _sha256(data["receipt_sha256"]) != content_sha256(body):
        raise RelocationError("relocation receipt SHA256 does not replay")
    logical = _logical_identity(data["logical_input_identity"], expected_input_sha256)
    if data["binding_identity"] is not None and not isinstance(
        data["binding_identity"], dict
    ):
        raise RelocationError("relocation binding identity must be an object or null")
    if expected_binding_identity is not None and data["binding_identity"] != dict(
        expected_binding_identity
    ):
        raise RelocationError("relocation binding/manifest identity changed")
    root = _root(_ref(data["destination_root"], absolute=True))
    if str(root) != data["destination_root"] or (
        destination_root is not None and root != _root(destination_root)
    ):
        raise RelocationError("relocation destination root changed")
    record = data["artifact"]
    if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
        raise RelocationError("invalid relocation artifact record")
    if (
        record["source_ref"] != logical["artifact_path"]
        or record["source_path"] != logical["artifact_path"]
        or expected_source_artifact is not None
        and record["source_path"] != expected_source_artifact
    ):
        raise RelocationError("relocation source artifact identity changed")
    identity = _check_file_identity(record["identity"])
    if identity["sha256"] != logical["artifact_json_sha256"] or (
        expected_artifact_identity is not None and identity != dict(expected_artifact_identity)
    ):
        raise RelocationError("relocation artifact identity changed")
    path = _target(root, record["target_ref"])
    if _identity(path) != identity:
        raise RelocationError("relocation artifact integrity changed")
    refs = _source_records(_artifact(path, logical), logical)
    dependencies = data["dependencies"]
    if not isinstance(dependencies, dict) or set(dependencies) != set(refs):
        raise RelocationError("relocation has missing or extra dependencies")
    for name, ref in refs.items():
        record = dependencies[name]
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise RelocationError(f"invalid relocation dependency record: {name}")
        identity = _check_file_identity(record["identity"])
        if (
            record["source_ref"] != ref
            or record["source_path"] != logical["files"][name]["path"]
            or identity["sha256"] != logical["files"][name]["sha256"]
        ):
            raise RelocationError(f"relocation logical dependency changed: {name}")
        if _identity(_target(root, record["target_ref"])) != identity:
            raise RelocationError(f"relocation dependency integrity changed: {name}")
    return ValidatedRelocation(data)


__all__ = [
    "RELOCATION_SCHEMA_VERSION", "RelocationError", "ValidatedRelocation",
    "artifact_dependency_refs", "create_relocation_receipt",
    "read_relocation_receipt", "validate_relocation_receipt",
]
