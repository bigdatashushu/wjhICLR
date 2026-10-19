"""Immutable content-addressed storage for prepared question bindings."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Mapping

import numpy as np

from skill3d.infra.m5_identity import file_identity, source_identity
from skill3d.online.runner import OnlineRunConfig, PreparedObjectBinding
from skill3d.schemas import ObjectRecord
from skill3d.segmentation.question_binding import (
    M5_BINDING_VERSION,
    binding_parameters,
)

STORE_SCHEMA_VERSION = "prepared-question-binding-store-v1"
REQUEST_SCHEMA_VERSION = "prepared-question-binding-request-v1"
OBJECT_REFS = (
    "mask_per_frame",
    "pointcloud_world",
    "pointconf_world",
    "centroid_ref",
)


class PreparedBindingStoreError(RuntimeError):
    """A prepared binding cannot be published or replayed immutably."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _safe_ref(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise PreparedBindingStoreError(
            f"prepared binding reference escapes its immutable entry: {value}"
        )
    return path


def prepared_binding_request_identity(
    *,
    input_identity: Mapping,
    cfg: OnlineRunConfig,
    scene_audit_qa_ids: list[str] | tuple[str, ...],
) -> dict:
    """Bind a reusable M5 result to the exact question, model and audit group."""
    endpoint_sha256 = hashlib.sha256(
        str(cfg.online_base_url).encode("utf-8")
    ).hexdigest()
    parameters = binding_parameters(
        semantic_min_pixels=cfg.semantic_min_pixels,
        semantic_max_pixels=cfg.semantic_max_pixels,
        semantic_relation_batch_size=cfg.semantic_relation_batch_size,
        semantic_review_config=cfg.semantic_review_config,
    )
    code_root = Path(__file__).resolve().parents[1]
    value = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "input_sha256": str(input_identity.get("sha256") or ""),
        "qa_id": str((input_identity.get("episode") or {}).get("qa_id") or ""),
        "scene_name": str(
            (input_identity.get("episode") or {}).get("scene_name") or ""
        ),
        "scene_audit_qa_ids": sorted({
            str(qa_id) for qa_id in scene_audit_qa_ids if str(qa_id)
        }),
        "seed": int(cfg.seed),
        "model": {
            "provider": str(cfg.online_model_provider),
            "model_id": str(cfg.online_model),
            "model_revision": str(cfg.online_model_revision),
            "model_weights_sha256": str(cfg.model_weights_sha256),
            "endpoint_sha256": endpoint_sha256,
        },
        "binding_version": M5_BINDING_VERSION,
        "parameters": parameters,
        "implementation_sources": source_identity(code_root),
        "store_source": file_identity(Path(__file__)),
    }
    if (
        not value["input_sha256"]
        or not value["qa_id"]
        or not value["scene_name"]
        or value["qa_id"] not in value["scene_audit_qa_ids"]
    ):
        raise PreparedBindingStoreError(
            "prepared binding request lacks input, question, scene, or audit identity"
        )
    return {**value, "sha256": _sha256(value)}


class PreparedBindingStore:
    """Publish and replay binding-only payloads without copying scene geometry."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.entries = self.root / "entries"
        self.staging = self.root / ".staging"

    @staticmethod
    def key_for(request_identity: Mapping) -> str:
        payload = dict(request_identity)
        claimed = str(payload.pop("sha256", "") or "")
        actual = _sha256(payload)
        if claimed and claimed != actual:
            raise PreparedBindingStoreError(
                "prepared binding request sha256 does not replay"
            )
        return actual

    def entry_path(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise PreparedBindingStoreError("prepared binding key is not sha256")
        return self.entries / key[:2] / key

    @staticmethod
    def _copy_payload_file(
        source: str | Path,
        payload_root: Path,
        copied: dict[tuple[int, str], str],
    ) -> str:
        path = Path(source).resolve()
        if not path.is_file():
            raise PreparedBindingStoreError(
                f"prepared binding payload is missing: {path}"
            )
        identity = file_identity(path)
        dedup_key = (int(identity["size"]), str(identity["sha256"]))
        existing = copied.get(dedup_key)
        if existing is not None:
            return existing
        suffix = path.suffix if path.suffix else ".bin"
        relative = (
            Path("payload")
            / "files"
            / f"{identity['sha256']}-{identity['size']}{suffix}"
        )
        target = payload_root.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied[dedup_key] = relative.as_posix()
        return relative.as_posix()

    @staticmethod
    def _pack_arrays(value, entry: Path, copied: dict, path: str = "stats"):
        if isinstance(value, np.ndarray):
            digest = hashlib.sha256(
                np.ascontiguousarray(value).tobytes()
            ).hexdigest()
            relative = Path("payload") / "arrays" / f"{digest}.npy"
            target = entry / relative
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, value)
            return {"__array_ref__": relative.as_posix()}
        if isinstance(value, dict):
            return {
                str(key): PreparedBindingStore._pack_arrays(
                    child,
                    entry,
                    copied,
                    f"{path}.{key}",
                )
                for key, child in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [
                PreparedBindingStore._pack_arrays(
                    child,
                    entry,
                    copied,
                    f"{path}.{index}",
                )
                for index, child in enumerate(value)
            ]
        return value

    @staticmethod
    def _payload_identity(entry: Path) -> dict:
        files = {}
        payload = entry / "payload"
        if payload.is_dir():
            for path in sorted(payload.rglob("*")):
                if path.is_symlink():
                    raise PreparedBindingStoreError(
                        f"symlink is forbidden in prepared binding entry: {path}"
                    )
                if path.is_file():
                    files[path.relative_to(entry).as_posix()] = file_identity(path)
        return {"files": files, "sha256": _sha256(files)}

    def publish(
        self,
        *,
        request_identity: Mapping,
        scene_artifact: str | Path,
        scene_input_identity: Mapping,
        binding: PreparedObjectBinding,
    ) -> dict:
        key = self.key_for(request_identity)
        target = self.entry_path(key)
        self.staging.mkdir(parents=True, exist_ok=True)
        temporary = self.staging / f"{key}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            copied: dict[tuple[int, str], str] = {}
            rows = []
            for obj in binding.objects:
                row = obj.model_dump(mode="json")
                for name in OBJECT_REFS:
                    if row.get(name):
                        row[name] = self._copy_payload_file(
                            row[name],
                            temporary / "payload",
                            copied,
                        )
                rows.append(row)
            binding_payload = {
                "objects": rows,
                "stats": self._pack_arrays(
                    binding.stats,
                    temporary,
                    copied,
                ),
                "notes": list(binding.notes),
                "materialized": bool(binding.materialized),
                "prepared_measurements": self._pack_arrays(
                    binding.prepared_measurements,
                    temporary,
                    copied,
                    "prepared_measurements",
                ),
            }
            _write_json(temporary / "payload" / "binding.json", binding_payload)
            scene_path = Path(scene_artifact).expanduser().resolve()
            if not scene_path.is_file():
                raise PreparedBindingStoreError(
                    f"scene artifact is missing: {scene_path}"
                )
            input_sha256 = str(scene_input_identity.get("sha256") or "")
            if input_sha256 != request_identity.get("input_sha256"):
                raise PreparedBindingStoreError(
                    "scene input identity differs from binding request"
                )
            manifest_body = {
                "schema_version": STORE_SCHEMA_VERSION,
                "key": key,
                "request_identity": dict(request_identity),
                "scene_artifact_ref": str(scene_path),
                "scene_artifact_json": file_identity(scene_path),
                "scene_input_identity_sha256": input_sha256,
                "binding_generation_sha256": str(
                    (binding.stats.get("generation") or {}).get("sha256") or ""
                ),
                "payload": self._payload_identity(temporary),
            }
            manifest = {
                **manifest_body,
                "manifest_sha256": _sha256(manifest_body),
            }
            _write_json(temporary / "manifest.json", manifest)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(temporary, target)
            except OSError:
                if not target.is_dir():
                    raise
                existing = self.manifest(key)
                if (
                    existing.get("manifest_sha256")
                    != manifest["manifest_sha256"]
                ):
                    raise PreparedBindingStoreError(
                        f"immutable prepared binding differs for key {key}"
                    )
            return self.reference(key)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def manifest(self, key: str) -> dict:
        path = self.entry_path(key) / "manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise PreparedBindingStoreError(
                f"prepared binding manifest is unreadable: {path}"
            ) from exc
        body = {
            name: value
            for name, value in manifest.items()
            if name != "manifest_sha256"
        }
        if (
            manifest.get("schema_version") != STORE_SCHEMA_VERSION
            or manifest.get("key") != key
            or manifest.get("manifest_sha256") != _sha256(body)
        ):
            raise PreparedBindingStoreError(
                f"prepared binding manifest identity mismatch: {path}"
            )
        return manifest

    def reference(self, key: str) -> dict:
        manifest = self.manifest(key)
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "store_root": str(self.root),
            "key": key,
            "request_identity": manifest["request_identity"],
            "scene_artifact_ref": manifest["scene_artifact_ref"],
            "scene_input_identity_sha256": manifest[
                "scene_input_identity_sha256"
            ],
            "manifest_sha256": manifest["manifest_sha256"],
        }

    def load(
        self,
        key: str,
        *,
        expected_request_identity: Mapping,
        scene_artifact: str | Path,
        scene_input_identity: Mapping,
    ) -> PreparedObjectBinding:
        expected_key = self.key_for(expected_request_identity)
        if key != expected_key:
            raise PreparedBindingStoreError(
                "prepared binding key differs from expected request identity"
            )
        manifest = self.manifest(key)
        if manifest.get("request_identity") != dict(expected_request_identity):
            raise PreparedBindingStoreError(
                "prepared binding request identity changed"
            )
        scene_path = Path(scene_artifact).expanduser().resolve()
        if (
            str(scene_path) != manifest.get("scene_artifact_ref")
            or file_identity(scene_path) != manifest.get("scene_artifact_json")
            or scene_input_identity.get("sha256")
            != manifest.get("scene_input_identity_sha256")
        ):
            raise PreparedBindingStoreError(
                "prepared binding scene artifact identity changed"
            )
        entry = self.entry_path(key)
        if self._payload_identity(entry) != manifest.get("payload"):
            raise PreparedBindingStoreError(
                "prepared binding payload identity changed"
            )
        try:
            data = json.loads(
                (entry / "payload" / "binding.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError, TypeError) as exc:
            raise PreparedBindingStoreError(
                "prepared binding payload is unreadable"
            ) from exc

        def unpack(value):
            if isinstance(value, dict):
                if set(value) == {"__array_ref__"}:
                    return np.load(
                        _safe_ref(entry, value["__array_ref__"]),
                        allow_pickle=False,
                    )
                return {key: unpack(child) for key, child in value.items()}
            if isinstance(value, list):
                return [unpack(child) for child in value]
            return value

        unpacked = unpack(data)
        objects = []
        for raw in unpacked["objects"]:
            row = dict(raw)
            for name in OBJECT_REFS:
                if row.get(name):
                    row[name] = str(_safe_ref(entry, row[name]))
            objects.append(ObjectRecord.model_validate(row))
        allowed = {field.name for field in fields(PreparedObjectBinding)}
        if set(unpacked) != allowed:
            raise PreparedBindingStoreError(
                "prepared binding payload fields changed"
            )
        unpacked["objects"] = objects
        return PreparedObjectBinding(**unpacked)


def load_prepared_binding_reference(
    reference: Mapping,
    *,
    scene_input_identity: Mapping,
) -> PreparedObjectBinding:
    """Replay a serialized store reference with full request and scene checks."""
    store = PreparedBindingStore(reference["store_root"])
    manifest = store.manifest(str(reference["key"]))
    if reference.get("manifest_sha256") != manifest.get("manifest_sha256"):
        raise PreparedBindingStoreError(
            "prepared binding reference manifest changed"
        )
    return store.load(
        str(reference["key"]),
        expected_request_identity=reference["request_identity"],
        scene_artifact=reference["scene_artifact_ref"],
        scene_input_identity=scene_input_identity,
    )


__all__ = [
    "PreparedBindingStore",
    "PreparedBindingStoreError",
    "STORE_SCHEMA_VERSION",
    "load_prepared_binding_reference",
    "prepared_binding_request_identity",
]
