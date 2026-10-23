"""Label-free transport with real frozen PNGs, copied stores and sealed hashes."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.cluster import scene_compute
from skill3d.evaluation.published_fitset_v11 import (
    PUBLISHED_INPUT_SCHEMA,
    PublishedFitSetInputProvider,
)
from skill3d.infra.m5_identity import content_sha256, file_identity
from skill3d.online.runner import PreparedObjectBinding
from skill3d.schemas import ObjectRecord
from skill3d.segmentation.prepared_binding_store import PreparedBindingStore

from test_cache_relocation import _receipt, _rehash_receipt, _signed
from test_cluster_scene_compute import _frozen, _write


def _published(tmp_path, *, n_questions=2):
    frozen = _frozen(tmp_path, n_questions=n_questions)
    frozen.manifest["split"] = "induction"
    for item, question in zip(frozen.items, frozen.manifest["questions"]):
        item.episode.split = "induction"
        question["episode"]["split"] = "induction"
        logical = question["logical_input_identity"]
        logical["episode"]["split"] = "induction"
        logical["sha256"] = content_sha256(
            {name: value for name, value in logical.items() if name != "sha256"})
    _write(frozen.root / "scene.json", frozen.manifest)

    # Restore the source just long enough to publish, then remove it. Both the
    # frozen input and publication must replay without any source-host path.
    source_artifact = Path(frozen.manifest["artifact_source_path"])
    source = source_artifact.parent
    source.mkdir()
    for original, target in frozen.manifest["files"].items():
        if Path(original).parent == source:
            shutil.copyfile(frozen.root / target, original)
    masks = tmp_path / "source-masks"
    masks.mkdir()
    np.save(masks / "mask.npy", np.ones((32, 3, 4), dtype=np.uint8))
    np.save(masks / "points.npy", np.arange(18, dtype=np.float32).reshape(6, 3))
    binding = PreparedObjectBinding(
        objects=[ObjectRecord(
            obj_id="chair-one", category_name="chair",
            grounding_status="question_driven",
            mask_per_frame=str(masks / "mask.npy"),
            pointcloud_world=str(masks / "points.npy"),
        )],
        stats={"array": np.arange(3)}, notes=["sealed M5 bytes"],
        materialized=True, prepared_measurements={"sample": np.eye(2)},
    )
    old_store = PreparedBindingStore(tmp_path / "old-store")
    original_refs, old_manifests = {}, {}
    for item, question in zip(frozen.items, frozen.manifest["questions"]):
        qa_id, logical = item.episode.qa_id, question["logical_input_identity"]
        reference = old_store.publish(
            request_identity=_signed({
                "schema_version": "test-request-v1", "input_sha256": logical["sha256"],
                "qa_id": qa_id, "scene_name": item.episode.scene_name,
            }),
            scene_artifact=source_artifact, scene_input_identity=logical,
            binding=binding,
        )
        original_refs[qa_id] = reference
        old_manifests[qa_id] = (
            old_store.entry_path(reference["key"]) / "manifest.json").read_bytes()
    destination = tmp_path / "publication"
    shutil.copytree(source, destination / "scene")
    shutil.copytree(old_store.root, destination / "store")
    store = PreparedBindingStore(destination / "store")
    migrations, catalog = {}, {}
    validation_path = tmp_path / "audited-validation.json"
    for item, question in zip(frozen.items, frozen.manifest["questions"]):
        qa_id, logical = item.episode.qa_id, question["logical_input_identity"]
        reference = {**original_refs[qa_id], "store_root": str(store.root)}
        migrations[qa_id] = _receipt(SimpleNamespace(
            store=store, key=reference["key"], logical=logical,
            destination=destination,
            dependency_refs={
                name: "scene/" + Path(row["path"]).name
                for name, row in logical["files"].items()
            },
        ))
        identity_path = destination / "identities" / f"{qa_id}.json"
        _write(identity_path, logical)
        catalog[qa_id] = {
            "qa_id": qa_id, "scene_id": item.episode.scene_name,
            "split": "induction", "variant": "base", "reference": reference,
            "scene_artifact": str(destination / "scene/artifact.json"),
            "scene_input_identity_file": str(identity_path),
            "scene_input_identity_file_sha256": file_identity(identity_path)["sha256"],
            "migration_receipt_file": str(validation_path),
            "publication_root": str(destination),
            "publication_receipt_sha256": "c" * 64,
            "outcome": {
                "status": "prepared", "materialized": True,
                "n_objects": 1, "binding_error": None,
            },
        }
    validation = {"variants": {"base": {"migration_receipts": migrations}}}
    _write(validation_path, validation)
    for entry in catalog.values():
        entry["migration_receipt_file_sha256"] = file_identity(validation_path)["sha256"]
    catalog_path = tmp_path / "original-catalog.json"
    _write(catalog_path, {"schema": "harness3d-published-binding-catalog/1",
                          "entries": catalog})
    manifest = {
        "schema": PUBLISHED_INPUT_SCHEMA,
        "source_catalog_sha256": file_identity(catalog_path)["sha256"],
        "selection_policy": {"split": "induction", "variant": "base", "limit": 8},
        "entries": {
            qa_id: {**deepcopy(entry), "input_root": str(frozen.root),
                    "frozen_scene_sha256": file_identity(frozen.root / "scene.json")["sha256"]}
            for qa_id, entry in catalog.items()
        },
    }
    manifest_path = tmp_path / "published-inputs.json"
    _write(manifest_path, manifest)
    variants = {
        qa_id: {
            "variant_id": f"published/{qa_id}", "qa_id": qa_id,
            "scene_id": "scene-one", "split": "induction",
            "question_type": "object_rel_distance",
            "artifact_ref": entry["reference"]["scene_artifact_ref"],
        }
        for qa_id, entry in catalog.items()
    }
    shutil.rmtree(source)
    shutil.rmtree(old_store.root)
    shutil.rmtree(masks)
    return SimpleNamespace(
        frozen=frozen, store=store, catalog=catalog, manifest=manifest,
        manifest_path=manifest_path, validation=validation,
        validation_path=validation_path, variants=variants,
        variant=variants["qa-00"], old_manifests=old_manifests,
    )


@pytest.fixture
def published(tmp_path):
    return _published(tmp_path)


def _rewrite_manifest(case):
    _write(case.manifest_path, case.manifest)


def _rewrite_validation(case):
    _write(case.validation_path, case.validation)
    for entry in case.manifest["entries"].values():
        entry["migration_receipt_file_sha256"] = file_identity(case.validation_path)["sha256"]
    _rewrite_manifest(case)


def _assert_safe(value):
    if isinstance(value, dict):
        assert not set(value) & {
            "ground_truth", "answer", "reference_answer", "score", "scores",
            "outcome", "outcomes", "labels",
        }
        for child in value.values():
            _assert_safe(child)
    elif isinstance(value, list):
        for child in value:
            _assert_safe(child)


def test_replays_original_identity_and_binding_without_source_host(published, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("transport must not run models, datasets or M5 preparation")

    from skill3d.adapters import episode_source
    from skill3d.online import runner

    monkeypatch.setattr(episode_source, "_load_from_sources", forbidden)
    monkeypatch.setattr(runner, "_bind_objects_best_effort", forbidden)
    monkeypatch.setattr(runner, "run_episode", forbidden)
    provider = PublishedFitSetInputProvider(published.manifest_path)
    for qa_id, variant in published.variants.items():
        item = provider(variant)
        assert item.episode.qa_id == qa_id and item.episode.ground_truth == ""
        assert len(item.pixels) == len(item.episode.frames) == 32
        receipt = provider.receipt(variant, item)
        result = provider.prepare_published(variant, item)
        assert set(result) == {
            "artifact_path", "artifact", "identity", "reference", "prepared",
            "artifact_relocation", "transport_receipt",
        }
        entry = published.catalog[qa_id]
        assert result["reference"] == entry["reference"]
        assert result["identity"] == published.frozen.manifest["questions"][
            int(qa_id[-2:])]["logical_input_identity"]
        assert result["identity"]["artifact_path"] == variant["artifact_ref"]
        assert not Path(variant["artifact_ref"]).exists()
        assert result["artifact_path"] == Path(entry["scene_artifact"])
        resolver = result["artifact_relocation"]
        assert result["artifact"].depth_maps == str(
            resolver.resolve(result["identity"]["files"]["depth_maps"]["path"]))
        assert result["prepared"].notes == ["sealed M5 bytes"]
        assert np.load(result["prepared"].objects[0].mask_per_frame).shape == (32, 3, 4)
        assert np.array_equal(result["prepared"].stats["array"], np.arange(3))
        manifest = published.store.entry_path(entry["reference"]["key"]) / "manifest.json"
        assert manifest.read_bytes() == published.old_manifests[qa_id]
        assert receipt == result["transport_receipt"] == json.loads(json.dumps(receipt))
        assert receipt["catalog_entry_sha256"] == content_sha256(entry)
        assert receipt["source_catalog_sha256"] == published.manifest["source_catalog_sha256"]
        assert receipt["scene_input_identity_sha256"] == result["identity"]["sha256"]
        assert receipt["frame_identity"] == item.episode.frame_set.cache_identity()
        assert receipt["pixel_sha256"] == item.episode.frame_set.pixel_sha256
        assert receipt["reference_answer_visible"] is False
        assert receipt["receipt_sha256"] == content_sha256(
            {k: v for k, v in receipt.items() if k != "receipt_sha256"})
        _assert_safe(receipt)
    assert provider.input_contract_identity["selection_policy"] == (
        published.manifest["selection_policy"])
    _assert_safe(provider.input_contract_identity)
    _assert_safe(provider.entries)
    # Public projections are isolated; diagnostics remain in the anchored file.
    public = provider.entries
    public["qa-00"]["reference"]["key"] = "changed"
    identity = provider.input_contract_identity
    identity["source_catalog_sha256"] = "0" * 64
    assert provider.entries["qa-00"]["reference"] == published.catalog["qa-00"]["reference"]
    assert "outcome" in published.manifest["entries"]["qa-00"]
    assert provider.input_contract_identity["source_catalog_sha256"] != "0" * 64


def test_uses_normal_scene_loader_and_clears_cache_when_pickled(published, monkeypatch):
    calls = []
    load = scene_compute._load_scene

    def spy(root, *, question_cap):
        calls.append((root, question_cap))
        return load(root, question_cap=question_cap)

    monkeypatch.setattr(scene_compute, "_load_scene", spy)
    provider = PublishedFitSetInputProvider(published.manifest_path)
    first = provider(published.variant)
    first.episode.question = "caller mutation"
    first.pixels[0][:] = 255
    second = provider(published.variants["qa-01"])
    assert second.episode.question != first.episode.question
    assert int(second.pixels[0][0, 0, 0]) == 0
    assert calls == [(published.frozen.root, 8)]
    restored = pickle.loads(pickle.dumps(provider))
    assert not restored.cache
    assert len(provider.cache) == 1
    assert restored(published.variant).episode.ground_truth == ""
    assert len(calls) == 2 and len(restored.cache) == 1


def test_scene_cache_is_bounded(tmp_path):
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, second = _published(first_root), _published(second_root)
    # Cache keys depend on extracted roots, not only the scene name.
    provider = PublishedFitSetInputProvider(first.manifest_path, max_scenes=1)
    provider._scene(first.manifest["entries"]["qa-00"])
    provider._scene(second.manifest["entries"]["qa-00"])
    assert len(provider.cache) == 1
    assert next(iter(provider.cache))[0] == str(second.frozen.root)


def test_complete_scene_question_cap_is_enforced(tmp_path):
    case = _published(tmp_path, n_questions=9)
    provider = PublishedFitSetInputProvider(case.manifest_path)
    with pytest.raises(ValueError, match="question cap 8"):
        provider(case.variant)


def test_selected_qa_must_exist_in_frozen_scene(published):
    entry = published.manifest["entries"].pop("qa-00")
    entry["qa_id"] = "missing-qa"
    published.manifest["entries"]["missing-qa"] = entry
    _rewrite_manifest(published)
    provider = PublishedFitSetInputProvider(published.manifest_path)
    with pytest.raises(ValueError, match="QA is missing"):
        provider({**published.variant, "qa_id": "missing-qa"})


@pytest.mark.parametrize("max_scenes", [0, -1, True, 1.5])
def test_invalid_cache_limit_rejected(published, max_scenes):
    with pytest.raises(ValueError, match="cache size"):
        PublishedFitSetInputProvider(published.manifest_path, max_scenes=max_scenes)


@pytest.mark.parametrize("mutation", [
    "schema", "missing_catalog_hash", "missing_policy", "key", "missing_field",
    "extra_field", "split", "variant", "score", "outcome_score", "label",
    "selection_score",
])
def test_manifest_and_public_catalog_fail_closed(published, mutation):
    manifest = published.manifest
    entry = manifest["entries"]["qa-00"]
    if mutation == "schema":
        manifest["schema"] = "future-schema"
    elif mutation == "missing_catalog_hash":
        del manifest["source_catalog_sha256"]
    elif mutation == "missing_policy":
        del manifest["selection_policy"]
    elif mutation == "key":
        entry["qa_id"] = "wrong"
    elif mutation == "missing_field":
        del entry["publication_receipt_sha256"]
    elif mutation == "extra_field":
        entry["unknown"] = "unreviewed"
    elif mutation == "split":
        entry["split"] = "outer_holdout"
    elif mutation == "variant":
        entry["variant"] = "metric"
    elif mutation == "score":
        entry["score"] = 0.99
    elif mutation == "outcome_score":
        entry["outcome"]["score"] = 0.99
    elif mutation == "label":
        entry["reference"]["request_identity"]["reference_answer"] = "hidden"
    elif mutation == "selection_score":
        manifest["selection_policy"]["scores"] = [1.0]
    _rewrite_manifest(published)
    with pytest.raises(ValueError):
        PublishedFitSetInputProvider(published.manifest_path)


@pytest.mark.parametrize("field,value", [
    ("qa_id", "missing-qa"), ("scene_id", "different-scene"),
    ("split", "inner_validation"), ("question_type", "object_counting"),
    ("artifact_ref", "local"),
])
def test_panel_identity_and_original_artifact_ref_required(published, field, value):
    provider = PublishedFitSetInputProvider(published.manifest_path)
    variant = {**published.variant, field: value}
    if value == "local":
        variant[field] = published.catalog["qa-00"]["scene_artifact"]
    with pytest.raises(ValueError):
        provider(variant)


@pytest.mark.parametrize("operation", ["call", "receipt", "prepare"])
def test_manifest_file_mutation_is_rejected_after_initialization(published, operation):
    provider = PublishedFitSetInputProvider(published.manifest_path)
    item = provider(published.variant)
    published.manifest["selection_policy"]["limit"] = 1
    _rewrite_manifest(published)
    with pytest.raises(ValueError, match="manifest integrity"):
        if operation == "call":
            provider(published.variant)
        elif operation == "receipt":
            provider.receipt(published.variant, item)
        else:
            provider.prepare_published(published.variant, item)


@pytest.mark.parametrize("operation", ["receipt", "prepare"])
@pytest.mark.parametrize("mutation", [
    "question", "options", "task", "dataset", "scene", "split", "ground_truth",
    "source", "video_path", "frame_source", "frame_timestamp", "frame_id",
    "frame_pixel_hash", "frame_shape", "frame_count", "pixel", "pixel_order",
    "pixel_dtype", "pixel_shape", "missing_pixel", "input_error", "synthetic_geometry",
])
def test_actual_episode_frames_and_pixels_cannot_be_substituted(
    published, operation, mutation,
):
    provider = PublishedFitSetInputProvider(published.manifest_path)
    item = provider(published.variant)
    ep, fs = item.episode, item.episode.frame_set
    if mutation in {"question", "dataset", "split", "ground_truth"}:
        setattr(ep, mutation, "different")
    elif mutation == "options":
        ep.options = ["changed", "B"]
    elif mutation == "task":
        ep.question_type = "object_counting"
    elif mutation == "scene":
        ep.scene_name = "different"
    elif mutation in {"source", "video_path"}:
        setattr(item, mutation, "different")
    elif mutation == "frame_source":
        fs.source_frame_indices[0] += 1
    elif mutation == "frame_timestamp":
        fs.timestamps[0] += 1
    elif mutation == "frame_id":
        fs.frame_ids[0] += 1
    elif mutation == "frame_pixel_hash":
        fs.pixel_sha256[0] = "0" * 64
    elif mutation == "frame_shape":
        ep.frames[0].width += 1
    elif mutation == "frame_count":
        ep.frames.pop()
    elif mutation == "pixel":
        item.pixels[0][0, 0, 0] += 1
    elif mutation == "pixel_order":
        item.pixels[0], item.pixels[1] = item.pixels[1], item.pixels[0]
    elif mutation == "pixel_dtype":
        item.pixels[0] = item.pixels[0].astype(np.float32)
    elif mutation == "pixel_shape":
        item.pixels[0] = item.pixels[0].transpose(1, 0, 2)
    elif mutation == "missing_pixel":
        item.pixels.pop()
    elif mutation == "input_error":
        item.input_error = SimpleNamespace(reason="unavailable")
    elif mutation == "synthetic_geometry":
        item.geometry = SimpleNamespace()
    with pytest.raises(ValueError):
        if operation == "receipt":
            provider.receipt(published.variant, item)
        else:
            provider.prepare_published(published.variant, item)


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("mutation", ["scene_json", "png", "geometry", "missing", "symlink"])
def test_frozen_archive_tampering_rejected_even_with_warm_cache(published, warm, mutation):
    provider = PublishedFitSetInputProvider(published.manifest_path)
    if warm:
        provider(published.variant)
    root, manifest = published.frozen.root, published.frozen.manifest
    path = root / manifest["files"][manifest["frames"][0]["source_path"]]
    if mutation == "scene_json":
        (root / "scene.json").write_bytes(b"{}")
    elif mutation == "png":
        path.write_bytes(b"changed PNG")
    elif mutation == "geometry":
        path = root / manifest["files"][next(
            p for p in manifest["files"] if p.endswith("point_map.npy"))]
        np.save(path, np.zeros((32, 3, 4, 3), dtype=np.float32))
    elif mutation == "missing":
        path.unlink()
    elif mutation == "symlink":
        other = root / "same-bytes.png"
        shutil.copyfile(path, other)
        path.unlink()
        path.symlink_to(other)
    with pytest.raises((ValueError, OSError)):
        provider(published.variant)


@pytest.mark.parametrize("field", ["answer", "reference_answer", "ground_truth", "labels"])
def test_labelled_frozen_public_episode_rejected_even_with_reanchored_scene(published, field):
    published.frozen.manifest["questions"][0]["episode"][field] = "hidden-label"
    _write(published.frozen.root / "scene.json", published.frozen.manifest)
    for entry in published.manifest["entries"].values():
        entry["frozen_scene_sha256"] = file_identity(
            published.frozen.root / "scene.json")["sha256"]
    _rewrite_manifest(published)
    provider = PublishedFitSetInputProvider(published.manifest_path)
    with pytest.raises(ValueError, match="label"):
        provider(published.variant)


@pytest.mark.parametrize("operation", ["receipt", "prepare"])
@pytest.mark.parametrize("file_field", ["scene_input_identity_file", "migration_receipt_file"])
def test_catalog_identity_and_migration_file_hashes_are_required(
    published, operation, file_field,
):
    provider = PublishedFitSetInputProvider(published.manifest_path)
    item = provider(published.variant)
    Path(published.catalog["qa-00"][file_field]).write_bytes(b"{}")
    with pytest.raises(ValueError, match="file integrity"):
        if operation == "receipt":
            provider.receipt(published.variant, item)
        else:
            provider.prepare_published(published.variant, item)


@pytest.mark.parametrize("mutation", ["hash_only", "metadata", "ground_truth"])
def test_logical_identity_cannot_be_forged_by_reanchoring_its_file(published, mutation):
    entry = published.manifest["entries"]["qa-00"]
    path = Path(entry["scene_input_identity_file"])
    identity = json.loads(path.read_bytes())
    if mutation == "hash_only":
        identity = {"sha256": identity["sha256"]}
    else:
        identity["episode"]["question" if mutation == "metadata" else "ground_truth"] = (
            "substitution")
        identity["sha256"] = content_sha256(
            {k: v for k, v in identity.items() if k != "sha256"})
    _write(path, identity)
    entry["scene_input_identity_file_sha256"] = file_identity(path)["sha256"]
    _rewrite_manifest(published)
    provider = PublishedFitSetInputProvider(published.manifest_path)
    with pytest.raises(ValueError):
        provider.prepare_published(published.variant, provider(published.variant))


@pytest.mark.parametrize("mutation", [
    "artifact", "depth", "point_map", "confidence", "binding_payload", "store_manifest",
])
def test_copied_dependency_and_store_bytes_checked_before_receipt(published, mutation):
    provider = PublishedFitSetInputProvider(published.manifest_path)
    item = provider(published.variant)
    entry = published.catalog["qa-00"]
    if mutation == "store_manifest":
        path = published.store.entry_path(entry["reference"]["key"]) / "manifest.json"
        value = json.loads(path.read_bytes())
        value["scene_artifact_ref"] = "changed"
        _write(path, value)
    elif mutation == "binding_payload":
        path = published.store.entry_path(entry["reference"]["key"]) / "payload/binding.json"
        path.write_bytes(b"{}")
    else:
        names = {"artifact": "artifact.json", "depth": "depth_maps.npy",
                 "point_map": "point_map.npy", "confidence": "point_conf.npy"}
        path = Path(entry["scene_artifact"]).parent / names[mutation]
        path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        provider.receipt(published.variant, item)


@pytest.mark.parametrize("mutation", [
    "binding_key", "binding_manifest", "source_artifact", "artifact_path", "destination",
    "missing_dependency", "dependency_hash", "receipt_hash", "missing_qa",
])
def test_forged_migration_receipt_rejected_after_catalog_file_reanchoring(published, mutation):
    migration = published.validation["variants"]["base"]["migration_receipts"]["qa-00"]
    if mutation == "binding_key":
        migration["binding_identity"]["key"] = "0" * 64
    elif mutation == "binding_manifest":
        migration["binding_identity"]["manifest_json"]["sha256"] = "0" * 64
    elif mutation == "source_artifact":
        migration["artifact"]["source_ref"] = "/substitute/artifact.json"
    elif mutation == "artifact_path":
        migration["artifact"]["target_ref"] = "../scene/artifact.json"
    elif mutation == "destination":
        migration["destination_root"] = str(published.frozen.root)
    elif mutation == "missing_dependency":
        del migration["dependencies"]["point_map"]
    elif mutation == "dependency_hash":
        migration["dependencies"]["point_map"]["identity"]["sha256"] = "0" * 64
    elif mutation == "missing_qa":
        del published.validation["variants"]["base"]["migration_receipts"]["qa-00"]
    if mutation == "receipt_hash":
        migration["receipt_sha256"] = "0" * 64
    else:
        _rehash_receipt(migration)
    _rewrite_validation(published)
    provider = PublishedFitSetInputProvider(published.manifest_path)
    with pytest.raises(ValueError):
        provider.prepare_published(published.variant, provider(published.variant))


@pytest.mark.parametrize("field", [
    "manifest_sha256", "scene_artifact_ref", "scene_input_identity_sha256",
    "request_identity", "key", "schema_version",
])
def test_original_reference_fields_cannot_be_substituted(published, field):
    entry = published.manifest["entries"]["qa-00"]
    reference = entry["reference"]
    if field == "request_identity":
        reference[field]["qa_id"] = "different"
    elif field == "scene_artifact_ref":
        reference[field] = entry["scene_artifact"]
        published.variant["artifact_ref"] = reference[field]
    else:
        reference[field] = "0" * 64
    _rewrite_manifest(published)
    provider = PublishedFitSetInputProvider(published.manifest_path)
    with pytest.raises(ValueError):
        provider.prepare_published(published.variant, provider(published.variant))


def test_validated_resolver_rechecks_dependency_bytes_after_preparation(published):
    provider = PublishedFitSetInputProvider(published.manifest_path)
    result = provider.prepare_published(published.variant, provider(published.variant))
    original_depth = result["identity"]["files"]["depth_maps"]["path"]
    local_depth = result["artifact_relocation"].resolve(original_depth)
    local_depth.write_bytes(b"late substitution")
    with pytest.raises(ValueError, match="integrity"):
        result["artifact_relocation"].resolve(original_depth)


def test_prepare_recomputes_actual_item_and_uses_original_identity_for_normal_loader(
    published, monkeypatch,
):
    from skill3d.segmentation import prepared_binding_store as bindings
    from skill3d.storage.relocation import ValidatedRelocation

    provider = PublishedFitSetInputProvider(published.manifest_path)
    item = provider(published.variant)
    entry = published.catalog["qa-00"]
    logical = json.loads(Path(entry["scene_input_identity_file"]).read_bytes())
    calls = []
    validate = PreparedBindingStore.validate_migration_receipt
    recompute = ValidatedRelocation.recompute_scene_input_identity
    load = bindings.load_prepared_binding_reference

    def validate_spy(store, key, receipt, **kwargs):
        assert key == entry["reference"]["key"]
        assert store.reference(key) == entry["reference"]
        assert kwargs["scene_input_identity"] == logical
        assert kwargs["scene_artifact"] == entry["scene_artifact"]
        assert kwargs["destination_root"] == entry["publication_root"]
        calls.append("validate")
        return validate(store, key, receipt, **kwargs)

    def recompute_spy(resolver, **kwargs):
        assert kwargs["episode"] == item.episode.model_dump(mode="json")
        assert kwargs["pixels"] is item.pixels
        assert kwargs["source"] == item.source and kwargs["video_path"] == item.video_path
        calls.append("recompute")
        return recompute(resolver, **kwargs)

    def load_spy(reference, **kwargs):
        assert reference == entry["reference"]
        assert kwargs["scene_input_identity"] == logical
        assert kwargs["scene_artifact"] == entry["scene_artifact"]
        assert kwargs["migration_destination_root"] == entry["publication_root"]
        calls.append("load")
        return load(reference, **kwargs)

    monkeypatch.setattr(PreparedBindingStore, "validate_migration_receipt", validate_spy)
    monkeypatch.setattr(ValidatedRelocation, "recompute_scene_input_identity", recompute_spy)
    monkeypatch.setattr(bindings, "load_prepared_binding_reference", load_spy)
    result = provider.prepare_published(published.variant, item)
    assert calls == ["validate", "recompute", "load", "validate"]
    assert result["identity"] == logical


def test_import_is_transport_only_without_inference_or_array_dependencies():
    source_root = Path(__file__).resolve().parents[2] / "src"
    code = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(source_root)!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'vggt', 'moge', 'sam2', 'numpy', 'PIL'}}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from skill3d.evaluation.published_fitset_v11 import PublishedFitSetInputProvider
assert callable(PublishedFitSetInputProvider)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True,
                   text=True, timeout=15)
