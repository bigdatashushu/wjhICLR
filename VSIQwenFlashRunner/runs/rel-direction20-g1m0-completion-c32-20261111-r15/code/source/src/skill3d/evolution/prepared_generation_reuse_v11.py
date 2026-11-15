"""Byte-proven M5 generation reuse, independent of destination binding identity.

Package schema ``v11-generation-reuse-package/1.0``:
``bindings[qa_id][base|metric]`` contains ``source_reference`` (raw store
reference), ``source_input_identity`` (full logical input), and
``source_migration_receipt`` (raw receipt), plus root-relative
``store_root_ref``, ``scene_artifact_ref`` and ``dependency_refs``. The last
mapping covers every declared old geometry/confidence/metric dependency.
Copied entries and artifact JSON must remain byte-identical. Receipts are
created anew for input_root, never written into an immutable source entry.
"""

from __future__ import annotations

import copy
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evaluation.skill_ablation_v11 import input_identity
from skill3d.infra.m5_identity import (
    array_identity, content_sha256,
)
from skill3d.legacy.readers import load_artifact_v5
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig, PreparedObjectBinding
from skill3d.schemas import FrameSet
from skill3d.segmentation.binding_validation import validate_binding
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore, PreparedBindingStoreError,
    load_prepared_binding_reference, prepared_binding_request_identity,
)
from skill3d.segmentation.question_binding import (
    question_input, validate_prepared_binding,
)
from skill3d.storage.relocation import (
    RelocationError, read_relocation_receipt,
)

PACKAGE_SCHEMA_VERSION = "v11-generation-reuse-package/1.0"
DERIVATION_SCHEMA_VERSION = "v11-generation-reuse-derivation/1.0"
_REQUEST_DIFFERENCES = {"input_sha256", "scene_audit_qa_ids", "sha256"}
_RECORD_FIELDS = {
    "source_reference", "source_input_identity", "source_migration_receipt",
    "store_root_ref", "scene_artifact_ref", "dependency_refs",
}


def _check_digest(value: Mapping, field: str, label: str) -> None:
    if not isinstance(value, Mapping) or value.get(field) != content_sha256({
        key: child for key, child in value.items() if key != field
    }):
        raise PreparedBindingStoreError(f"{label} identity does not replay")


def _old_item(item: EpisodeItem, identity: Mapping) -> EpisodeItem | None:
    """Only historical split and nonsemantic frame refs may be relocated."""
    old = identity["episode"]
    new = item.episode.model_dump(mode="json")
    if old.get("ground_truth") != "" or new.get("ground_truth") != "":
        raise PreparedBindingStoreError("generation reuse requires public questions")
    if ({k: v for k, v in old.items() if k not in {"split", "frame_set"}}
            != {k: v for k, v in new.items() if k not in {"split", "frame_set"}}):
        return None
    old_fs, new_fs = old.get("frame_set"), new.get("frame_set")
    if not isinstance(old_fs, dict) or not isinstance(new_fs, dict):
        return None
    location_fields = {"frame_refs", "pixel_sha256"}
    if ({k: v for k, v in old_fs.items() if k not in location_fields}
            != {k: v for k, v in new_fs.items() if k not in location_fields}):
        return None
    if item.source != identity["source"] or item.video_path != identity["video_path"]:
        return None
    result = copy.copy(item)
    result.episode = item.episode.model_copy(update={
        "split": old["split"], "frame_set": FrameSet.model_validate(old_fs),
    }, deep=True)
    return result


def _m5_arrays(art) -> dict:
    return {
        name: runner._load_npy(getattr(art, ref))
        for name, ref in (
            ("depth", "depth_maps"), ("c2w", "c2w_list"),
            ("intrinsics", "intrinsics"), ("point_conf", "depth_conf"),
            ("depth_conf", "depth_conf"), ("point_map", "point_map"),
        )
    }


def _scene_identity(art) -> str:
    return str(art.reconstruction_input_identity or art.frame_set_hash or art.artifact_id)


def reuse_verified_binding(
    binding: PreparedObjectBinding,
    old_identity: Mapping,
    old_request: Mapping,
    old_manifest: Mapping,
    new_identity: Mapping,
    new_request: Mapping,
    item: EpisodeItem,
    art,
    cfg: OnlineRunConfig,
    proof: Mapping,
) -> PreparedObjectBinding | None:
    """Derive a binding from a store-loaded source and recomputed old input.

    ``proof`` supplies ``source_artifact`` (resolved, quality-prepared runtime
    model) and ``source_input_recomputed`` (actual full identity). Other JSON
    proof fields are recorded. Valid but incompatible evidence returns None;
    corrupt identity/hash anchors raise, never becoming perception outcomes.
    """
    for value, field, label in (
        (old_identity, "sha256", "old input"),
        (new_identity, "sha256", "new input"),
        (old_manifest, "manifest_sha256", "source manifest"),
    ):
        _check_digest(value, field, label)
    old_key = PreparedBindingStore.key_for(old_request)
    PreparedBindingStore.key_for(new_request)
    if (old_manifest.get("key") != old_key
            or old_manifest.get("request_identity") != dict(old_request)
            or old_manifest.get("scene_input_identity_sha256") != old_identity["sha256"]
            or old_manifest.get("scene_artifact_ref") != old_identity["artifact_path"]
            or (old_manifest.get("scene_artifact_json") or {}).get("sha256")
            != old_identity["artifact_json_sha256"]
            or old_request.get("input_sha256") != old_identity["sha256"]
            or new_request.get("input_sha256") != new_identity["sha256"]
            or proof.get("source_input_recomputed") != dict(old_identity)):
        raise PreparedBindingStoreError("generation reuse source/request/input anchor changed")
    payload = old_manifest.get("payload") or {}
    if payload.get("sha256") != content_sha256(payload.get("files")):
        raise PreparedBindingStoreError("generation reuse payload anchor changed")
    generation = binding.stats.get("generation") or {}
    generation_sha = str(generation.get("sha256") or "")
    if generation_sha != old_manifest.get("binding_generation_sha256"):
        raise PreparedBindingStoreError("source manifest/payload generation changed")
    if generation and generation_sha != content_sha256(generation.get("identity") or {}):
        raise PreparedBindingStoreError("source generation content hash changed")
    expected = prepared_binding_request_identity(
        input_identity=new_identity, cfg=cfg,
        scene_audit_qa_ids=new_request["scene_audit_qa_ids"],
    )
    if dict(new_request) != expected:
        raise PreparedBindingStoreError("generation reuse destination request changed")
    if ({k: v for k, v in old_request.items() if k not in _REQUEST_DIFFERENCES}
            != {k: v for k, v in new_request.items() if k not in _REQUEST_DIFFERENCES}):
        return None
    if cfg.mode != "real" or _old_item(item, old_identity) is None:
        return None
    if new_identity["episode"] != item.episode.model_dump(mode="json"):
        raise PreparedBindingStoreError("generation reuse destination episode changed")
    if old_identity["pixels"] != new_identity["pixels"]:
        return None
    old_art = proof.get("source_artifact")
    if old_art is None:
        return None
    arrays = _m5_arrays(art)
    geometry = {name: array_identity(value) for name, value in arrays.items()}
    if geometry != {name: array_identity(value) for name, value in _m5_arrays(old_art).items()}:
        return None
    usable = bool(art.quality and art.quality.main_gate_passed)
    if usable != bool(old_art.quality and old_art.quality.main_gate_passed):
        return None
    if binding.stats.get("binding_error") or binding.stats.get("status") == "parse_unavailable":
        return None
    parse = binding.stats.get("object_parse")
    if parse is not None and parse.get("status") != "parsed":
        return None
    status = binding.stats.get("status")
    no_generation_outcome = (
        not binding.objects
        and ((status == "m4_unavailable" and not usable and not binding.materialized)
             or (status == "objects_not_required" and usable and binding.materialized
                 and runner.canonical_task(item.episode.question_type) == "room_size_estimation"))
    )
    if not generation and not no_generation_outcome:
        return None
    derived = copy.deepcopy(binding)
    derived.prepared_measurements = {}
    if generation:
        identity = generation.get("identity") or {}
        model = identity.get("vlm") or {}
        pins = {
            "enabled": True, "provider": str(cfg.online_model_provider),
            "model": str(cfg.online_model), "model_revision": str(cfg.online_model_revision),
            "endpoint_sha256": new_request["model"]["endpoint_sha256"],
            "seed": int(cfg.seed), "temperature": float(cfg.temperature),
            "max_tokens": int(cfg.max_tokens),
        }
        if any(model.get(name) != value for name, value in pins.items()):
            return None
        if (identity.get("scene") != art.scene_name
                or identity.get("frame_set_hash") != art.frame_set_hash
                or identity.get("grid_transform") is not None
                or identity.get("frame_provenance") != runner._episode_frame_provenance(
                    item.episode, item.pixels)):
            return None
        derived.objects = [
            obj.model_copy(update={"binding_validation": validate_binding(
                obj, depth_maps=arrays["depth"], n_frames=len(item.pixels),
                generation_sha256=generation_sha,
            )}) for obj in derived.objects
        ]
        try:
            validate_prepared_binding(
                derived, frames=item.pixels, **arrays,
                inputs=question_input(item.episode.question,
                    runner.canonical_task(item.episode.question_type), item.episode.options),
                scene_identity=_scene_identity(art), seed=int(cfg.seed),
                model_weights_sha256="", parameters=new_request["parameters"],
            )
        except ValueError:
            return None
    if ((status == "m4_unavailable" and not no_generation_outcome)
            or (status == "objects_not_required" and not no_generation_outcome)):
        return None
    derived.stats.update(
        cache_hit=False, actual_model_requests=0, actual_detector_requests=0,
        actual_sam2_tracks=0,
        generation_reuse={
            "schema_version": DERIVATION_SCHEMA_VERSION,
            "source_key": old_key,
            "source_request_identity": dict(old_request),
            "source_input_sha256": old_identity["sha256"],
            "source_manifest_sha256": old_manifest["manifest_sha256"],
            "source_payload_sha256": payload["sha256"],
            "source_generation_sha256": generation_sha,
            "destination_request_sha256": new_request["sha256"],
            "destination_input_sha256": new_identity["sha256"],
            "hash_proof": {
                "geometry": geometry,
                "frames_sha256": content_sha256(new_identity["pixels"]),
                "question_sha256": content_sha256(question_input(
                    item.episode.question, runner.canonical_task(item.episode.question_type),
                    item.episode.options)),
                "implementation_sources_sha256": content_sha256(
                    new_request["implementation_sources"]),
                "generation_content_sha256": generation_sha,
                "geometry_usable": usable,
            },
            "source_proof": {
                key: value for key, value in proof.items()
                if key not in {"source_artifact", "source_input_recomputed"}
            },
            "new_model_requests": 0, "new_detector_requests": 0,
            "new_sam2_tracks": 0, "generation_resealed": False,
        },
    )
    return derived


def _destination(item, path, identity):
    if input_identity(item, path) != identity:
        raise PreparedBindingStoreError("generation reuse destination input changed")
    return runner._prepare_m4_quality(
        load_artifact_v5(path), item.pixels, persist_quality=False)


def make_local_generation_reuse_factory(
    items: Sequence[EpisodeItem],
    cfg: OnlineRunConfig,
    source_preparation: Mapping,
    source_identities: Mapping[str, Mapping],
    source_artifact_path: str | Path | Mapping[str, str | Path],
):
    """Reuse current base sealed references for metric QA without migration."""
    indexed = {str(item.episode.qa_id): item for item in items}
    references = source_preparation.get("bindings", {})

    def reuse(item, artifact_path, new_identity, new_request):
        qa_id = str(item.episode.qa_id)
        if qa_id not in indexed or qa_id not in references or qa_id not in source_identities:
            return None
        old_identity = source_identities[qa_id]
        _check_digest(old_identity, "sha256", "old input")
        path = (source_artifact_path[qa_id] if isinstance(source_artifact_path, Mapping)
                else source_artifact_path)
        reference = references[qa_id]
        store = PreparedBindingStore(reference["store_root"])
        if store.reference(reference["key"]) != dict(reference):
            raise PreparedBindingStoreError("local generation reuse source reference changed")
        binding = load_prepared_binding_reference(reference, scene_input_identity=old_identity)
        if Path(path).expanduser().resolve() != Path(reference["scene_artifact_ref"]):
            raise PreparedBindingStoreError("local generation reuse source artifact path changed")
        prior = _old_item(item, old_identity)
        if prior is None or old_identity["pixels"] != new_identity["pixels"]:
            return None
        actual = input_identity(prior, path)
        if actual != old_identity:
            raise PreparedBindingStoreError("local generation reuse source input changed")
        old_art = runner._prepare_m4_quality(
            load_artifact_v5(path), item.pixels, persist_quality=False)
        return reuse_verified_binding(
            binding, old_identity, reference["request_identity"],
            store.manifest(reference["key"]), new_identity, new_request, item,
            _destination(item, artifact_path, new_identity), cfg,
            {"source_artifact": old_art, "source_input_recomputed": actual,
             "kind": "local-sealed-binding"},
        )

    return reuse


def _package_path(root: Path, ref: str, *, directory=False) -> Path:
    relative = PurePosixPath(ref)
    if (not ref or relative.is_absolute() or relative.as_posix() != ref
            or ".." in relative.parts or "\\" in ref):
        raise PreparedBindingStoreError("unsafe generation reuse package reference")
    path = root / ref
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)):
        raise PreparedBindingStoreError("generation reuse package symlink is forbidden")
    if not path.resolve().is_relative_to(root) or not (
        path.is_dir() if directory else path.is_file()
    ):
        raise PreparedBindingStoreError("generation reuse package target missing or escaping")
    return path


def make_generation_reuse_factory(
    scene_items: Sequence[EpisodeItem],
    cfg: OnlineRunConfig,
    input_root: str | Path,
    package_manifest: Mapping,
):
    """Load exported old entries through newly validated local migration receipts."""
    if (set(package_manifest) != {"schema_version", "bindings"}
            or package_manifest["schema_version"] != PACKAGE_SCHEMA_VERSION):
        raise PreparedBindingStoreError("unsupported generation reuse package schema")
    root = Path(input_root).expanduser().resolve()
    indexed = {str(item.episode.qa_id): item for item in scene_items}

    def reuse(item, artifact_path, new_identity, new_request):
        qa_id = str(item.episode.qa_id)
        if qa_id not in indexed:
            return None
        art = _destination(item, artifact_path, new_identity)
        variant = "metric" if art.metric_model not in (None, "none") else "base"
        record = package_manifest["bindings"].get(qa_id, {}).get(variant)
        if record is None:
            return None
        if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
            raise PreparedBindingStoreError("generation reuse package record changed")
        store = PreparedBindingStore(_package_path(root, record["store_root_ref"], directory=True))
        old_identity = record["source_input_identity"]
        _check_digest(old_identity, "sha256", "old input")
        reference = record["source_reference"]
        key = reference["key"]
        local_reference = {**reference, "store_root": str(store.root)}
        if store.reference(key) != local_reference:
            raise PreparedBindingStoreError("exported generation reuse reference changed")
        _package_path(root, record["scene_artifact_ref"])
        try:
            fresh = store.create_migration_receipt(
                key, scene_input_identity=old_identity, destination_root=root,
                scene_artifact_ref=record["scene_artifact_ref"],
                dependency_refs=record["dependency_refs"],
            )
            original = read_relocation_receipt(record["source_migration_receipt"])
            _check_digest(original, "receipt_sha256", "old migration receipt")
            # Verify the raw old receipt with only sanctioned locations relocated.
            relocated = copy.deepcopy(original)
            relocated["destination_root"] = fresh["destination_root"]
            relocated["artifact"]["target_ref"] = fresh["artifact"]["target_ref"]
            for name, dependency in relocated["dependencies"].items():
                dependency["target_ref"] = fresh["dependencies"][name]["target_ref"]
            relocated["receipt_sha256"] = content_sha256({
                k: v for k, v in relocated.items() if k != "receipt_sha256"
            })
            if relocated != fresh:
                raise PreparedBindingStoreError("old migration receipt anchors changed")
            resolver = store.validate_migration_receipt(key, fresh, destination_root=root)
            prior = _old_item(item, old_identity)
            if prior is None or old_identity["pixels"] != new_identity["pixels"]:
                return None
            actual = resolver.recompute_scene_input_identity(
                episode=prior.episode.model_dump(mode="json"),
                source=prior.source, video_path=prior.video_path, pixels=prior.pixels,
            )
            binding = load_prepared_binding_reference(
                local_reference, scene_input_identity=actual,
                migration_receipt=fresh, migration_destination_root=root,
            )
            old_art = runner._prepare_m4_quality(
                resolver.resolved_artifact(), item.pixels, persist_quality=False)
        except (RelocationError, KeyError, TypeError) as exc:
            raise PreparedBindingStoreError(f"generation reuse migration rejected: {exc}") from exc
        return reuse_verified_binding(
            binding, old_identity, reference["request_identity"], store.manifest(key),
            new_identity, new_request, item, art, cfg,
            {"source_artifact": old_art, "source_input_recomputed": actual,
             "kind": "exported-binding",
             "source_receipt_sha256": original["receipt_sha256"],
             "destination_receipt_sha256": fresh["receipt_sha256"]},
        )

    return reuse


__all__ = [
    "PACKAGE_SCHEMA_VERSION", "DERIVATION_SCHEMA_VERSION",
    "reuse_verified_binding", "make_generation_reuse_factory",
    "make_local_generation_reuse_factory",
]
