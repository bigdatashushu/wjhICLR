"""Real byte-copy, identity replay and adversarial relocation contracts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.adapters.frame_set import build_frame_set, with_pixel_identity
from skill3d.evaluation.skill_ablation_v11 import input_identity
from skill3d.infra.m5_identity import content_sha256, file_identity
from skill3d.online.runner import PreparedObjectBinding
from skill3d.schemas import ObjectRecord, ReconstructionArtifact
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore,
    PreparedBindingStoreError,
    load_prepared_binding_reference,
)
from skill3d.storage.relocation import (
    RelocationError,
    artifact_dependency_refs,
    create_relocation_receipt,
    validate_relocation_receipt,
)


def _signed(body):
    return {**body, "sha256": content_sha256(body)}


def _rehash_receipt(receipt):
    receipt["receipt_sha256"] = content_sha256({
        name: value for name, value in receipt.items() if name != "receipt_sha256"
    })
    return receipt


def _scene_copy(root: Path, *, relative=False, depth_value=1.0, metric=False):
    source = root / "source-scene"
    source.mkdir(parents=True)
    frames = [np.full((3, 4, 3), i, dtype=np.uint8) for i in range(2)]
    fs = build_frame_set(
        2, n_frames=2, episode_id="qa-1", dataset_id="dataset-1",
        scene_name="scene-1", video_id="video-1",
    )
    episode = {
        "qa_id": "qa-1", "dataset": "dataset-1", "scene_name": "scene-1",
        "question": "Which chair is closer?", "question_type": "object_rel_distance",
        "options": ["A", "B"], "frame_set": fs.model_dump(mode="json"),
    }
    arrays = {
        "depth_maps": np.full((2, 3, 4), depth_value, dtype=np.float32),
        "c2w_list": np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        "point_map": np.arange(72, dtype=np.float32).reshape(2, 3, 4, 3),
        "point_conf": np.full((2, 3, 4), 0.75, dtype=np.float32),
        "depth_conf": np.full((2, 3, 4), 0.25, dtype=np.float32),
        "metric_depth_maps": np.full((2, 3, 4), 2.0, dtype=np.float32),
        "metric_valid_masks": np.ones((2, 3, 4), dtype=bool),
        "coverage": np.array([3, 4], dtype=np.int32),
    }
    paths = {}
    for name, array in arrays.items():
        path = source / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        paths[name] = path
    for name in ("track_list", "per_frame_scale_ref"):
        path = source / f"{name}.json"
        path.write_text(json.dumps({"reference": name}), encoding="utf-8")
        paths[name] = path
    refs = {
        name: path.name if relative else str(path)
        for name, path in paths.items() if name != "coverage"
    }
    art = ReconstructionArtifact(
        artifact_id="geometry-1", artifact_version="geometry-v1",
        scene_name="scene-1", frame_ids=fs.readable_frame_ids,
        frame_set_hash=fs.frame_set_hash, source_frame_indices=fs.source_frame_indices,
        timestamps=fs.timestamps, **refs,
        confidence={
            "per_point_confidence": refs["point_conf"],
            "coverage_count_per_frame": (
                paths["coverage"].name if relative else str(paths["coverage"])
            ),
        },
    )
    if metric:
        from skill3d.reconstruction import metric_evidence, metric_fusion

        frame_identity = with_pixel_identity(fs, frames).cache_identity()
        request = {
            "input_identity": frame_identity,
            "sources": {
                name: file_identity(paths[name])["sha256"]
                for name in metric_evidence.GEOMETRY_REFS if name in paths
            },
            "frame_ids": fs.readable_frame_ids, "frame_set_hash": fs.frame_set_hash,
            "scene_name": art.scene_name, "source_frame_indices": fs.source_frame_indices,
            "timestamps": fs.timestamps,
        }
        fusion = metric_fusion.fuse_metric_scale(
            arrays["metric_depth_maps"], arrays["depth_maps"],
            arrays["metric_valid_masks"], model="moge2", min_valid_pixels=1,
        )
        rec = metric_fusion.per_frame_receipt(fusion)
        rec.update(
            request=request, request_sha256=metric_evidence.digest(request),
            evidence_sha256={
                name: file_identity(paths[name])["sha256"]
                for name in ("metric_depth_maps", "metric_valid_masks")
            },
        )
        paths["per_frame_scale_ref"].write_text(json.dumps(rec), encoding="utf-8")
        art = art.model_copy(update={
            "reconstruction_input_identity": frame_identity,
            "metric_input_identity": frame_identity,
            "metric_request_sha256": rec["request_sha256"],
            "metric_receipt_sha256": file_identity(paths["per_frame_scale_ref"])["sha256"],
            "metric_model": fusion.model, "metric_fusion_version": fusion.version,
            "scale_fusion_status": fusion.status, "metric_scale": fusion.metric_scale,
            "scale_self_consistency": fusion.scale_self_consistency,
        })
    scene = source / "artifact.json"
    scene.write_text(art.model_dump_json(indent=2) + "\n", encoding="utf-8")
    ep = SimpleNamespace(
        **{name: episode[name] for name in ("qa_id", "dataset", "scene_name")},
        frame_set=fs, model_dump=lambda **kwargs: deepcopy(episode),
    )
    item = SimpleNamespace(
        episode=ep, source="frozen-video-source", video_path=None, pixels=frames,
    )
    logical = input_identity(item, scene)
    request = _signed({
        "schema_version": "test-request-v1", "input_sha256": logical["sha256"],
        "qa_id": "qa-1", "scene_name": "scene-1",
    })
    binding_root = root / "binding-source"
    binding_root.mkdir()
    mask, points = binding_root / "mask.npy", binding_root / "points.npy"
    np.save(mask, np.ones((2, 3, 4), dtype=np.uint8))
    np.save(points, np.arange(18, dtype=np.float32).reshape(6, 3))
    binding = PreparedObjectBinding(
        objects=[ObjectRecord(
            obj_id="obj-1", category_name="chair", grounding_status="question_driven",
            mask_per_frame=str(mask), pointcloud_world=str(points),
        )],
        stats={"generation": {"sha256": "a" * 64}, "array": np.arange(3)},
        notes=["byte-preserving migration"], materialized=True,
        prepared_measurements={"example": np.eye(2)},
    )
    old_store = PreparedBindingStore(root / "source-store")
    reference = old_store.publish(
        request_identity=request, scene_artifact=scene,
        scene_input_identity=logical, binding=binding,
    )
    destination = root / "destination"
    shutil.copytree(source, destination / "scene")
    shutil.copytree(old_store.root, destination / "store")
    store = PreparedBindingStore(destination / "store")
    dependency_refs = {
        name: "scene/" + Path(row["path"]).name
        for name, row in logical["files"].items()
    }
    return SimpleNamespace(
        source=source, old_store=old_store, store=store, destination=destination,
        scene=scene, art=art, item=item, episode=episode, logical=logical,
        request=request, reference=reference, key=reference["key"],
        dependency_refs=dependency_refs,
    )


def _receipt(case):
    return case.store.create_migration_receipt(
        case.key, scene_input_identity=case.logical,
        destination_root=case.destination, scene_artifact_ref="scene/artifact.json",
        dependency_refs=case.dependency_refs,
    )


def _load(case, receipt, **kwargs):
    return case.store.load(
        case.key, expected_request_identity=case.request,
        scene_input_identity=kwargs.pop("scene_input_identity", case.logical),
        migration_receipt=receipt, **kwargs,
    )


@pytest.fixture
def relocated(tmp_path):
    case = _scene_copy(tmp_path)
    case.receipt = _receipt(case)
    return case


@pytest.mark.parametrize("relative", [False, True])
def test_byte_preserving_migration_replays_after_original_host_is_gone(tmp_path, relative):
    case = _scene_copy(tmp_path, relative=relative)
    old_manifest = (case.old_store.entry_path(case.key) / "manifest.json").read_bytes()
    old_files = {
        p.relative_to(case.old_store.entry_path(case.key)).as_posix(): file_identity(p)
        for p in case.old_store.entry_path(case.key).rglob("*") if p.is_file()
    }
    shutil.rmtree(case.source)
    shutil.rmtree(case.old_store.root)
    receipt = _receipt(case)
    receipt_path = case.destination / "migration.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    resolver = case.store.validate_migration_receipt(
        case.key, receipt_path, destination_root=case.destination,
    )
    actual = resolver.recompute_scene_input_identity(
        episode=case.episode, source=case.item.source,
        video_path=case.item.video_path, pixels=case.item.pixels,
    )
    assert actual == case.logical
    assert len(resolver.dependency_paths) == len(artifact_dependency_refs(case.art))
    assert resolver.resolve(case.art.depth_maps) == case.destination / "scene/depth_maps.npy"
    runtime_art = resolver.resolved_artifact()
    assert runtime_art.confidence.per_point_confidence == runtime_art.point_conf
    assert np.array_equal(np.load(runtime_art.point_map), np.load(resolver.resolve(case.art.point_map)))
    loaded = _load(case, receipt_path, scene_input_identity=actual)
    assert loaded.materialized and loaded.notes == ["byte-preserving migration"]
    assert np.array_equal(loaded.stats["array"], np.arange(3))
    assert np.array_equal(loaded.prepared_measurements["example"], np.eye(2))
    assert np.load(loaded.objects[0].pointcloud_world).shape == (6, 3)
    new_manifest = case.store.entry_path(case.key) / "manifest.json"
    assert new_manifest.read_bytes() == old_manifest
    assert {
        p.relative_to(case.store.entry_path(case.key)).as_posix(): file_identity(p)
        for p in case.store.entry_path(case.key).rglob("*") if p.is_file()
    } == old_files
    assert receipt["binding_identity"]["key"] == case.key
    assert receipt["binding_identity"]["manifest_json"] == file_identity(new_manifest)
    reference = case.store.reference(case.key)
    assert reference["scene_artifact_ref"] == str(case.scene)
    assert load_prepared_binding_reference(
        reference, scene_input_identity=actual, migration_receipt=receipt_path,
        migration_destination_root=case.destination,
    ).notes == loaded.notes


def test_legacy_load_is_unchanged_and_copy_requires_explicit_receipt(relocated):
    case = relocated
    legacy = case.old_store.load(
        case.key, expected_request_identity=case.request,
        scene_artifact=case.scene, scene_input_identity={"sha256": case.logical["sha256"]},
    )
    assert legacy.notes == _load(case, case.receipt).notes
    assert load_prepared_binding_reference(
        case.reference, scene_input_identity=case.logical,
    ).notes == legacy.notes
    with pytest.raises(PreparedBindingStoreError, match="scene artifact identity changed"):
        case.store.load(
            case.key, expected_request_identity=case.request,
            scene_artifact=case.destination / "scene/artifact.json",
            scene_input_identity=case.logical,
        )


@pytest.mark.parametrize("name", [
    "depth_maps", "c2w_list", "intrinsics", "point_map", "point_conf", "depth_conf",
    "track_list", "per_frame_scale_ref", "metric_depth_maps", "metric_valid_masks",
    "confidence.coverage_count_per_frame",
])
def test_all_dependency_bytes_are_verified_even_when_size_does_not_change(relocated, name):
    case = relocated
    path = case.destination / case.dependency_refs[name]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    assert len(data) == case.receipt["dependencies"][name]["identity"]["size"]
    with pytest.raises(PreparedBindingStoreError, match="dependency integrity changed"):
        _load(case, case.receipt)
    with pytest.raises(PreparedBindingStoreError, match="dependency integrity changed"):
        _receipt(case)


def test_changed_artifact_bytes_are_rejected_without_rewriting_manifest(relocated):
    case = relocated
    path = case.destination / "scene/artifact.json"
    before = (case.store.entry_path(case.key) / "manifest.json").read_bytes()
    path.write_bytes(path.read_bytes().replace(b"geometry-v1", b"geometry-v2"))
    with pytest.raises(PreparedBindingStoreError, match="artifact integrity changed"):
        _load(case, case.receipt)
    assert (case.store.entry_path(case.key) / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("operation", ["omit", "extra"])
def test_rehashed_receipt_cannot_omit_or_add_dependencies(relocated, operation):
    receipt = deepcopy(relocated.receipt)
    if operation == "omit":
        del receipt["dependencies"]["confidence.per_point_confidence"]
    else:
        receipt["dependencies"]["undeclared"] = deepcopy(receipt["dependencies"]["point_map"])
    _rehash_receipt(receipt)
    with pytest.raises(PreparedBindingStoreError, match="missing or extra dependencies"):
        _load(relocated, receipt)
    refs = dict(relocated.dependency_refs)
    refs.pop("confidence.coverage_count_per_frame")
    with pytest.raises(PreparedBindingStoreError, match="missing or extra dependencies"):
        relocated.store.create_migration_receipt(
            relocated.key, scene_input_identity=relocated.logical,
            destination_root=relocated.destination, scene_artifact_ref="scene/artifact.json",
            dependency_refs=refs,
        )


def test_missing_dependency_file_is_rejected(relocated):
    (relocated.destination / relocated.dependency_refs["depth_conf"]).unlink()
    with pytest.raises(PreparedBindingStoreError, match="missing or escaping"):
        _load(relocated, relocated.receipt)


@pytest.mark.parametrize("field", ["target_ref", "size", "source_ref", "source_path", "sha256"])
def test_receipt_tamper_and_rehashed_logical_or_size_tamper_are_rejected(relocated, field):
    receipt = deepcopy(relocated.receipt)
    row = receipt["dependencies"]["depth_maps"]
    if field == "size":
        row["identity"]["size"] += 1
    elif field == "sha256":
        row["identity"]["sha256"] = "f" * 64
    else:
        row[field] = "scene/other.npy" if field == "target_ref" else "/other/depth.npy"
    with pytest.raises(PreparedBindingStoreError, match="receipt SHA256"):
        _load(relocated, receipt)
    _rehash_receipt(receipt)
    with pytest.raises(PreparedBindingStoreError, match="migration rejected"):
        _load(relocated, receipt)


def test_rehashed_receipt_and_new_hash_cannot_attach_old_binding_to_new_geometry(relocated):
    case = relocated
    receipt = deepcopy(case.receipt)
    path = case.destination / case.dependency_refs["point_map"]
    array = np.load(path)
    array += 100
    np.save(path, array)
    receipt["dependencies"]["point_map"]["identity"] = file_identity(path)
    _rehash_receipt(receipt)
    with pytest.raises(PreparedBindingStoreError, match="logical dependency changed"):
        _load(case, receipt)
    logical = receipt["logical_input_identity"]
    logical["files"]["point_map"]["sha256"] = file_identity(path)["sha256"]
    logical["sha256"] = content_sha256({
        name: child for name, child in logical.items() if name != "sha256"
    })
    _rehash_receipt(receipt)
    with pytest.raises(PreparedBindingStoreError, match="input identity does not replay"):
        _load(case, receipt, scene_input_identity=logical)


def test_receipt_from_a_different_scene_cannot_attach_to_binding(tmp_path):
    first = _scene_copy(tmp_path / "one")
    second = _scene_copy(tmp_path / "two", depth_value=5.0)
    receipt = _receipt(second)
    with pytest.raises(PreparedBindingStoreError, match="input identity does not replay"):
        _load(first, receipt)
    with pytest.raises(PreparedBindingStoreError, match="input identity does not replay"):
        first.store.create_migration_receipt(
            first.key, scene_input_identity=second.logical,
            destination_root=second.destination, scene_artifact_ref="scene/artifact.json",
            dependency_refs=second.dependency_refs,
        )


@pytest.mark.parametrize("change", ["qa_id", "source", "pixel", "hash_only"])
def test_migration_rejects_wrong_input_even_with_claimed_old_hash(relocated, change):
    identity = deepcopy(relocated.logical)
    if change == "qa_id":
        identity["episode"]["qa_id"] = "another-question"
    elif change == "source":
        identity["source"] = "another-video"
    elif change == "pixel":
        identity["pixels"][0]["sha256"] = "e" * 64
    else:
        identity = {"sha256": identity["sha256"]}
    with pytest.raises(PreparedBindingStoreError, match="input identity"):
        _load(relocated, relocated.receipt, scene_input_identity=identity)


@pytest.mark.parametrize("change", ["pixels", "source", "video", "question", "frame_order"])
def test_local_identity_recomputation_checks_actual_input_not_only_metadata(relocated, change):
    case = relocated
    resolver = case.store.validate_migration_receipt(case.key, case.receipt)
    args = {
        "episode": deepcopy(case.episode), "source": case.item.source,
        "video_path": None, "pixels": [p.copy() for p in case.item.pixels],
    }
    if change == "pixels":
        args["pixels"][0][0, 0, 0] ^= 1
    elif change == "source":
        args["source"] = "different-source"
    elif change == "video":
        args["video_path"] = "different-video"
    elif change == "question":
        args["episode"]["question"] = "different question"
    else:
        args["pixels"].reverse()
    with pytest.raises(RelocationError, match="input"):
        resolver.recompute_scene_input_identity(**args)


@pytest.mark.parametrize("field", ["key", "manifest_sha256", "manifest_json"])
def test_receipt_is_bound_to_original_manifest_and_key(relocated, field):
    receipt = deepcopy(relocated.receipt)
    if field == "manifest_json":
        receipt["binding_identity"][field]["size"] += 1
    else:
        receipt["binding_identity"][field] = "d" * 64
    _rehash_receipt(receipt)
    with pytest.raises(PreparedBindingStoreError, match="binding/manifest identity changed"):
        _load(relocated, receipt)


def test_binding_payload_is_still_verified_after_scene_migration(relocated):
    entry = relocated.store.entry_path(relocated.key)
    payload = next((entry / "payload").rglob("*.npy"))
    payload.write_bytes(b"not original")
    with pytest.raises(PreparedBindingStoreError, match="payload identity changed"):
        _load(relocated, relocated.receipt)
    with pytest.raises(PreparedBindingStoreError, match="payload identity changed"):
        relocated.store.validate_migration_receipt(relocated.key, relocated.receipt)


@pytest.mark.parametrize("ref", [
    "../scene/depth_maps.npy", "/tmp/depth_maps.npy", "scene/../depth_maps.npy",
    "scene//depth_maps.npy", "./scene/depth_maps.npy", "file://scene/depth_maps.npy",
    "scene\\depth_maps.npy", "scene/depth_maps.npy\n",
])
def test_unsafe_receipt_paths_are_rejected_even_after_rehash(relocated, ref):
    receipt = deepcopy(relocated.receipt)
    receipt["dependencies"]["depth_maps"]["target_ref"] = ref
    _rehash_receipt(receipt)
    with pytest.raises(PreparedBindingStoreError, match="reference"):
        _load(relocated, receipt)


@pytest.mark.parametrize("outside", [False, True])
def test_symlink_dependency_is_rejected_even_when_bytes_match(relocated, outside):
    case = relocated
    path = case.destination / case.dependency_refs["depth_maps"]
    target = (
        case.destination.parent / "external-depth.npy"
        if outside else case.destination / "same-bytes.npy"
    )
    shutil.copyfile(path, target)
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(PreparedBindingStoreError, match="symlink"):
        _load(case, case.receipt)


def test_symlink_parent_and_changed_root_are_rejected(relocated):
    case = relocated
    scene = case.destination / "scene"
    original = case.destination / "real-scene"
    scene.rename(original)
    scene.symlink_to(original, target_is_directory=True)
    with pytest.raises(PreparedBindingStoreError, match="symlink"):
        _load(case, case.receipt)
    scene.unlink()
    original.rename(scene)
    other_root = case.destination / "other"
    other_root.mkdir()
    with pytest.raises(PreparedBindingStoreError, match="destination root changed"):
        _load(case, case.receipt, migration_destination_root=other_root)


def test_resolver_rechecks_bytes_and_ref_membership_after_validation(relocated):
    case = relocated
    resolver = case.store.validate_migration_receipt(case.key, case.receipt)
    with pytest.raises(RelocationError, match="undeclared"):
        resolver.resolve(str(case.source / "not-declared.npy"))
    path = resolver.resolve(case.art.depth_maps)
    path.write_bytes(b"changed after validation")
    with pytest.raises(RelocationError, match="integrity changed"):
        resolver.resolve(case.art.depth_maps)


def test_ambiguous_alias_destinations_are_not_guessed(relocated):
    receipt = deepcopy(relocated.receipt)
    duplicate = relocated.destination / "duplicate.npy"
    shutil.copyfile(relocated.destination / "scene/point_conf.npy", duplicate)
    receipt["dependencies"]["confidence.per_point_confidence"]["target_ref"] = duplicate.name
    _rehash_receipt(receipt)
    with pytest.raises(PreparedBindingStoreError, match="ambiguous relocation reference"):
        _load(relocated, receipt)


def test_explicit_wrong_artifact_and_changed_reference_are_rejected(relocated):
    case = relocated
    other = case.destination / "other-artifact.json"
    shutil.copyfile(case.destination / "scene/artifact.json", other)
    with pytest.raises(PreparedBindingStoreError, match="not the receipted artifact"):
        _load(case, case.receipt, scene_artifact=other)
    reference = case.store.reference(case.key)
    reference["scene_artifact_ref"] = str(other)
    with pytest.raises(PreparedBindingStoreError, match="reference scene identity changed"):
        load_prepared_binding_reference(
            reference, scene_input_identity=case.logical, migration_receipt=case.receipt,
        )


def test_generic_receipt_resolver_works_without_prepared_binding_anchor(relocated):
    case = relocated
    receipt = create_relocation_receipt(
        logical_input_identity=case.logical, destination_root=case.destination,
        artifact_ref="scene/artifact.json", dependency_refs=case.dependency_refs,
    )
    assert receipt["binding_identity"] is None
    resolver = validate_relocation_receipt(
        receipt, expected_input_sha256=case.logical["sha256"],
        destination_root=case.destination,
    )
    assert np.load(resolver.resolve(case.art.point_map)).shape == (2, 3, 4, 3)
    with pytest.raises(PreparedBindingStoreError, match="binding/manifest"):
        _load(case, receipt)


def test_duplicate_json_fields_and_unsupported_schema_are_rejected(relocated):
    receipt = deepcopy(relocated.receipt)
    receipt["schema_version"] = "unknown-v2"
    _rehash_receipt(receipt)
    with pytest.raises(PreparedBindingStoreError, match="unsupported"):
        _load(relocated, receipt)
    path = relocated.destination / "duplicate-receipt.json"
    text = json.dumps(relocated.receipt)
    path.write_text(
        '{"receipt_sha256":"' + relocated.receipt["receipt_sha256"] + '",' + text[1:],
        encoding="utf-8",
    )
    with pytest.raises(PreparedBindingStoreError, match="receipt is unreadable"):
        _load(relocated, path)


def test_metric_scene_recomputes_and_preserves_metric_evidence_after_copy(tmp_path):
    from skill3d.reconstruction.metric_evidence import validated_receipt

    case = _scene_copy(tmp_path, metric=True)
    original = validated_receipt(case.art)
    receipt = _receipt(case)
    shutil.rmtree(case.source)
    resolver = case.store.validate_migration_receipt(case.key, receipt)
    actual = resolver.recompute_scene_input_identity(
        episode=case.episode, source=case.item.source,
        video_path=case.item.video_path, pixels=case.item.pixels,
    )
    assert actual == case.logical
    assert validated_receipt(resolver.resolved_artifact()) == original
    assert _load(case, receipt, scene_input_identity=actual).materialized


def test_full_logical_identity_is_required_to_generate_a_receipt(relocated):
    with pytest.raises(PreparedBindingStoreError, match="full scene input identity"):
        relocated.store.create_migration_receipt(
            relocated.key, scene_input_identity={"sha256": relocated.logical["sha256"]},
            destination_root=relocated.destination, scene_artifact_ref="scene/artifact.json",
            dependency_refs=relocated.dependency_refs,
        )


def test_wrong_expected_request_is_rejected_before_migration(relocated):
    request = dict(relocated.request)
    request.pop("sha256")
    request["qa_id"] = "different-question"
    with pytest.raises(PreparedBindingStoreError, match="key differs"):
        relocated.store.load(
            relocated.key, expected_request_identity=_signed(request),
            scene_input_identity=relocated.logical, migration_receipt=relocated.receipt,
        )


@pytest.mark.parametrize("size", [True, -1, "12", None])
def test_file_size_schema_is_strict(relocated, size):
    receipt = deepcopy(relocated.receipt)
    receipt["dependencies"]["depth_maps"]["identity"]["size"] = size
    _rehash_receipt(receipt)
    with pytest.raises(PreparedBindingStoreError, match="requires SHA256 and size"):
        _load(relocated, receipt)


@pytest.mark.parametrize("field", ["manifest.json", "payload"])
def test_immutable_entry_symlinks_are_rejected_in_migration(relocated, field):
    path = relocated.store.entry_path(relocated.key) / field
    target = path.with_name("original-" + path.name)
    path.rename(target)
    path.symlink_to(target, target_is_directory=field == "payload")
    with pytest.raises(PreparedBindingStoreError, match="symlink"):
        _load(relocated, relocated.receipt)


def test_manifest_byte_tamper_is_detected_even_if_logical_manifest_hash_replays(relocated):
    path = relocated.store.entry_path(relocated.key) / "manifest.json"
    before = relocated.store.manifest(relocated.key)
    path.write_text(json.dumps(before, indent=4), encoding="utf-8")
    assert relocated.store.manifest(relocated.key) == before
    with pytest.raises(PreparedBindingStoreError, match="binding/manifest identity changed"):
        _load(relocated, relocated.receipt)


def test_partial_logical_dependency_inventory_cannot_authorize_missing_refs(relocated):
    logical = deepcopy(relocated.logical)
    del logical["files"]["confidence.coverage_count_per_frame"]
    logical["sha256"] = content_sha256({
        name: child for name, child in logical.items() if name != "sha256"
    })
    with pytest.raises(RelocationError, match="missing or extra dependencies"):
        create_relocation_receipt(
            logical_input_identity=logical, destination_root=relocated.destination,
            artifact_ref="scene/artifact.json", dependency_refs=relocated.dependency_refs,
        )


@pytest.mark.parametrize("change", ["shape", "dtype", "missing_sha256"])
def test_pixel_identity_schema_is_strict(relocated, change):
    logical = deepcopy(relocated.logical)
    row = logical["pixels"][0]
    if change == "shape":
        row["shape"] = [3, 4]
    elif change == "dtype":
        row["dtype"] = None
    else:
        row.pop("sha256")
    logical["sha256"] = content_sha256({
        name: child for name, child in logical.items() if name != "sha256"
    })
    with pytest.raises(RelocationError, match="RGB pixel identity"):
        create_relocation_receipt(
            logical_input_identity=logical, destination_root=relocated.destination,
            artifact_ref="scene/artifact.json", dependency_refs=relocated.dependency_refs,
        )
