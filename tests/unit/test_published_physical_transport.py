"""Synthetic physical copies only: no private data, services, or model calls."""

from copy import deepcopy
import json
from pathlib import Path
import pickle
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.evaluation.derived_published_fitset_v11 import DerivedPublishedFitSetInputProvider
from skill3d.cluster import scene_compute
from skill3d.evaluation.published_fitset_v11 import (
    FULLQA_INPUT_SCHEMA, PublishedFitSetInputError, PublishedFitSetInputProvider,
)
from skill3d.evolution import prepared_binding_derivation_v11 as derivation
from skill3d.infra.m5_identity import array_identity, content_sha256, file_identity
from skill3d.online.runner import OnlineRunConfig
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore, PreparedBindingStoreError, load_prepared_binding_reference,
    prepared_binding_request_identity,
)
from skill3d.segmentation.question_binding import M5_BINDING_VERSION, question_input
from skill3d.storage.published_transport import (
    PUBLISHED_TRANSPORT_SCHEMA, PublishedTransportLocator,
)
from skill3d.storage.relocation import RelocationError

from test_cluster_scene_compute import _write
from test_published_fitset_v11 import _published


def _never(*args, **kwargs):
    raise AssertionError("relocation must not derive, run models, or load a dataset")


def _fullqa(case):
    catalog = case.manifest_path.parent / "full-catalog.json"
    _write(catalog, {
        "schema": "harness3d-published-binding-catalog/1",
        "variants": {"base": case.catalog},
    })
    case.manifest.update(
        schema=FULLQA_INPUT_SCHEMA, source_catalog_file=str(catalog),
        source_catalog_sha256=file_identity(catalog)["sha256"],
        scene_scopes={"scene-one": {
            "qa_ids": sorted(case.catalog), "split": "induction",
            "input_root": str(case.frozen.root),
            "frozen_scene_sha256": file_identity(case.frozen.root / "scene.json")["sha256"],
        }})
    _write(case.manifest_path, case.manifest)


def _copy(case, extra_roots=()):
    parent = case.manifest_path.parent
    local = parent / "hot"
    local.mkdir()
    descriptor = {"schema_version": PUBLISHED_TRANSPORT_SCHEMA, "files": {}, "roots": {}}
    publication = Path(case.catalog["qa-00"]["publication_root"])
    roots = [case.frozen.root, publication, *extra_roots]
    for index, source in enumerate(roots):
        target = local / f"root-{index}"
        shutil.copytree(source, target)
        members = {
            path.relative_to(source).as_posix(): file_identity(path)
            for path in source.rglob("*") if path.is_file()}
        descriptor["roots"][str(source)] = {
            "path": str(target), "members": members,
            "members_sha256": content_sha256(members)}
    store_target = Path(descriptor["roots"][str(publication)]["path"]) / "store"
    store_members = {
        path.relative_to(case.store.root).as_posix(): file_identity(path)
        for path in case.store.root.rglob("*") if path.is_file()}
    descriptor["roots"][str(case.store.root)] = {
        "path": str(store_target), "members": store_members,
        "members_sha256": content_sha256(store_members)}
    files = {case.validation_path}
    files.update(Path(row["scene_input_identity_file"]) for row in case.catalog.values())
    if case.manifest.get("source_catalog_file"):
        files.add(Path(case.manifest["source_catalog_file"]))
    for index, source in enumerate(sorted(files)):
        target = local / f"file-{index}.json"
        shutil.copyfile(source, target)
        descriptor["files"][str(source)] = {
            "path": str(target), "identity": file_identity(source)}
    manifest_path = local / "published_inputs.json"
    shutil.copyfile(case.manifest_path, manifest_path)
    descriptor["files"][str(case.manifest_path)] = {
        "path": str(manifest_path), "identity": file_identity(case.manifest_path)}
    for source in roots:
        shutil.rmtree(source)
    for source in files:
        if source.exists():
            source.unlink()
    case.manifest_path.unlink()
    return SimpleNamespace(
        descriptor=descriptor, manifest_path=manifest_path, local=local,
        publication=publication, store_root=store_target)


def _provider(case, copied):
    return PublishedFitSetInputProvider(
        copied.manifest_path, transport_locator=copied.descriptor)


@pytest.mark.parametrize("fullqa", [False, True])
def test_unchanged_publication_replays_at_new_roots_with_regular_loader(
    tmp_path, monkeypatch, fullqa,
):
    case = _published(tmp_path)
    if fullqa:
        _fullqa(case)
    original_manifest = case.manifest_path.read_bytes()
    copied = _copy(case)
    from skill3d.adapters import episode_source
    from skill3d.online import runner

    monkeypatch.setattr(episode_source, "_load_from_sources", _never)
    monkeypatch.setattr(runner, "_bind_objects_best_effort", _never)
    provider = _provider(case, copied)
    for qa, variant in case.variants.items():
        item = provider(variant)
        result = provider.prepare_published(variant, item)
        original = case.catalog[qa]
        assert result["reference"] == original["reference"]
        assert result["identity"] == case.frozen.manifest["questions"][
            int(qa[-2:])]["logical_input_identity"]
        assert result["artifact_path"].is_relative_to(copied.local)
        assert Path(result["prepared"].objects[0].mask_per_frame).is_relative_to(copied.store_root)
        receipt = result["transport_receipt"]
        assert receipt["migration_receipt_sha256"] == case.validation["variants"]["base"][
            "migration_receipts"][qa]["receipt_sha256"]
        assert receipt["physical_transport"]["migration_receipt_sha256"] != (
            receipt["migration_receipt_sha256"])
        assert receipt["catalog_entry_sha256"] == content_sha256(original)
        assert receipt["input_manifest"]["sha256"] == file_identity(copied.manifest_path)["sha256"]
        store = PreparedBindingStore(copied.store_root)
        assert (store.entry_path(original["reference"]["key"]) / "manifest.json").read_bytes() == (
            case.old_manifests[qa])
    assert copied.manifest_path.read_bytes() == original_manifest
    clone = pickle.loads(pickle.dumps(provider))
    assert clone.prepare_published(case.variant, clone(case.variant))["reference"] == (
        case.catalog["qa-00"]["reference"])


def test_declared_file_relocations_can_rename_geometry_members(tmp_path):
    case = _published(tmp_path)
    copied = _copy(case)
    root_row = copied.descriptor["roots"][str(copied.publication)]
    root = Path(root_row["path"])
    migration = case.validation["variants"]["base"]["migration_receipts"]["qa-00"]
    moved = set()
    for index, record in enumerate((migration["artifact"], *migration["dependencies"].values())):
        if record["target_ref"] in moved:
            continue
        moved.add(record["target_ref"])
        old = root / record["target_ref"]
        renamed = root / "renamed" / f"{index}{old.suffix}"
        renamed.parent.mkdir(exist_ok=True)
        old.rename(renamed)
        copied.descriptor["files"][str(copied.publication / record["target_ref"])] = {
            "path": str(renamed), "identity": record["identity"]}
        root_row["members"].pop(record["target_ref"])
        root_row["members"][renamed.relative_to(root).as_posix()] = record["identity"]
    root_row["members_sha256"] = content_sha256(root_row["members"])
    provider = _provider(case, copied)
    result = provider.prepare_published(case.variant, provider(case.variant))
    assert result["artifact_path"].parent.name == "renamed"
    assert result["identity"]["artifact_path"] == case.variant["artifact_ref"]


def test_original_manifest_locator_can_be_exactly_pinned_to_local_copy(tmp_path):
    case = _published(tmp_path)
    copied = _copy(case)
    provider = PublishedFitSetInputProvider(
        case.manifest_path, transport_locator=PublishedTransportLocator(copied.descriptor))
    assert provider.manifest_path == copied.manifest_path
    assert provider.prepare_published(case.variant, provider(case.variant))["reference"] == (
        case.catalog["qa-00"]["reference"])


@pytest.mark.parametrize("failure", [
    "missing_root", "missing_file", "hash_only", "path_escape", "changed_member",
    "extra_member", "symlink", "prefix_only",
])
def test_unapproved_or_changed_physical_copies_reject(tmp_path, failure):
    case = _published(tmp_path)
    copied = _copy(case)
    descriptor = copied.descriptor
    row = descriptor["roots"][str(case.frozen.root)]
    if failure == "missing_root":
        del descriptor["roots"][str(case.frozen.root)]
    elif failure == "missing_file":
        del descriptor["files"][str(case.validation_path)]
    elif failure == "hash_only":
        row.pop("members")
    elif failure == "path_escape":
        identity = next(iter(row["members"].values()))
        row["members"]["../escaped.npy"] = identity
        row["members_sha256"] = content_sha256(row["members"])
    elif failure == "changed_member":
        (Path(row["path"]) / next(iter(row["members"]))).write_bytes(b"changed")
    elif failure == "extra_member":
        (Path(row["path"]) / "undeclared.bin").write_bytes(b"extra")
    elif failure == "symlink":
        path = Path(row["path"]) / next(iter(row["members"]))
        path.unlink()
        path.symlink_to(copied.manifest_path)
    elif failure == "prefix_only":
        descriptor["roots"][str(case.frozen.root.parent)] = descriptor["roots"].pop(
            str(case.frozen.root))
    with pytest.raises((PublishedFitSetInputError, RelocationError, OSError, ValueError)):
        provider = _provider(case, copied)
        provider.prepare_published(case.variant, provider(case.variant))


def test_mutated_actual_pixels_reject(tmp_path):
    case = _published(tmp_path)
    copied = _copy(case)
    provider = _provider(case, copied)
    item = provider(case.variant)
    item.pixels[0][0, 0, 0] ^= 1
    with pytest.raises(PublishedFitSetInputError, match="pixel identity changed"):
        provider.prepare_published(case.variant, item)


@pytest.mark.parametrize("field", [
    "schema_version", "key", "request_identity", "scene_artifact_ref",
    "scene_input_identity_sha256", "manifest_sha256",
])
def test_store_override_validates_original_reference_without_mutation(tmp_path, field):
    case = _published(tmp_path)
    copied = _copy(case)
    provider = _provider(case, copied)
    transport = provider._transport(case.variant, provider(case.variant))
    reference = deepcopy(case.catalog["qa-00"]["reference"])
    original = deepcopy(reference)
    if field == "request_identity":
        reference[field]["qa_id"] = "different"
    else:
        reference[field] = "a" * 64 if field.endswith("sha256") or field == "key" else "changed"
    with pytest.raises(PreparedBindingStoreError):
        load_prepared_binding_reference(
            reference, scene_input_identity=transport["identity"],
            physical_store_root=copied.store_root, migration_receipt=transport["migration"],
            migration_destination_root=transport["destination_root"])
    assert original == case.catalog["qa-00"]["reference"]


def test_store_override_rejects_hash_only_identity_and_no_migration(tmp_path):
    case = _published(tmp_path)
    copied = _copy(case)
    provider = _provider(case, copied)
    transport = provider._transport(case.variant, provider(case.variant))
    for migration in (None, transport["migration"]):
        with pytest.raises(PreparedBindingStoreError):
            load_prepared_binding_reference(
                case.catalog["qa-00"]["reference"],
                scene_input_identity={"sha256": transport["identity"]["sha256"]},
                physical_store_root=copied.store_root, migration_receipt=migration)


def test_relocated_derived_provider_requires_explicit_existing_replay(tmp_path):
    case = _published(tmp_path)
    _fullqa(case)
    copied = _copy(case)
    with pytest.raises(PublishedFitSetInputError, match="approved existing derived replay"):
        DerivedPublishedFitSetInputProvider(
            copied.manifest_path, derivation_root=tmp_path / "new-derived",
            binding_cfg=OnlineRunConfig(mode="real"), transport_locator=copied.descriptor)


def _derived_case(tmp_path):
    case = _published(tmp_path)
    case.frozen.artifact = scene_compute._load_scene(case.frozen.root, question_cap=8).artifact
    cfg = OnlineRunConfig(mode="real")
    old_bindings = {}
    for qa, original in case.catalog.items():
        identity = case.frozen.manifest["questions"][int(qa[-2:])]["logical_input_identity"]
        old_bindings[qa] = load_prepared_binding_reference(
            original["reference"], scene_input_identity=identity,
            migration_receipt=case.validation["variants"]["base"]["migration_receipts"][qa])
    anchors, inputs, requests, sources, migrations = {}, {}, {}, {}, {}
    geometry = {}
    first = case.frozen.manifest["questions"][0]["logical_input_identity"]
    geometry_names = {
        "depth": "depth_maps", "c2w": "c2w_list", "intrinsics": "intrinsics",
        "point_conf": "point_conf", "depth_conf": "depth_conf", "point_map": "point_map",
    }
    for name, ref in geometry_names.items():
        path = case.frozen.root / case.frozen.manifest["files"][first["files"][ref]["path"]]
        geometry[name] = np.load(path, allow_pickle=False)
    for qa, item in zip(sorted(case.catalog), case.frozen.items):
        identity = case.frozen.manifest["questions"][int(qa[-2:])]["logical_input_identity"]
        inputs[qa] = identity
        request = prepared_binding_request_identity(
            input_identity=identity, cfg=cfg, scene_audit_qa_ids=sorted(case.catalog))
        requests[qa] = request
        old_request = {**deepcopy(request), "binding_version": "synthetic-frozen-v3",
                       "implementation_sources": {"frozen_v3_source_sha256": "f" * 64},
                       "store_source": {"size": 123, "sha256": "e" * 64}}
        old_request["sha256"] = content_sha256({k: v for k, v in old_request.items() if k != "sha256"})
        binding = old_bindings[qa]
        gen_identity = {
            "geometry": {k: array_identity(v) for k, v in geometry.items()},
            "frames": [array_identity(p) for p in item.pixels],
            "question": question_input(item.episode.question, "object_rel_distance", item.episode.options),
            "reconstruction_identity": str(case.frozen.artifact.reconstruction_input_identity),
        }
        generation = {"identity": gen_identity, "sha256": content_sha256(gen_identity)}
        binding.stats["generation"] = generation
        original = case.catalog[qa]
        old_receipt = case.validation["variants"]["base"]["migration_receipts"][qa]
        reference = case.store.publish(
            request_identity=old_request, scene_artifact=original["scene_artifact"],
            scene_input_identity=identity, binding=binding, scene_migration_receipt={
                **old_receipt, "binding_identity": None,
                "receipt_sha256": content_sha256({
                    k: (None if k == "binding_identity" else v)
                    for k, v in old_receipt.items() if k != "receipt_sha256"})})
        migration = case.store.create_migration_receipt(
            reference["key"], scene_input_identity=identity,
            destination_root=original["publication_root"],
            scene_artifact_ref=old_receipt["artifact"]["target_ref"],
            dependency_refs={k: v["target_ref"] for k, v in old_receipt["dependencies"].items()})
        original["reference"] = reference
        case.manifest["entries"][qa]["reference"] = deepcopy(reference)
        sources[qa], migrations[qa] = binding, migration
        manifest = case.store.manifest(reference["key"])
        anchors[qa] = {
            "reference": deepcopy(reference), "input_sha256": identity["sha256"],
            "request_identity": deepcopy(old_request), "manifest_sha256": manifest["manifest_sha256"],
            "payload_sha256": manifest["payload"]["sha256"], "generation_sha256": generation["sha256"],
            "generation_stats_sha256": content_sha256(generation),
            "binding_content_sha256": content_sha256(derivation._content(binding)),
            "relocation_sha256": content_sha256(migration),
        }
    case.validation["variants"]["base"]["migration_receipts"] = migrations
    _write(case.validation_path, case.validation)
    for qa in case.catalog:
        digest = file_identity(case.validation_path)["sha256"]
        case.catalog[qa]["migration_receipt_file_sha256"] = digest
        case.manifest["entries"][qa]["migration_receipt_file_sha256"] = digest
    root = tmp_path / "original-derived"
    store = PreparedBindingStore(root / "scene-one")
    destinations = {
        qa: {"input_sha256": inputs[qa]["sha256"], "request_identity": requests[qa]}
        for qa in sources}
    references, derived_migrations, seals = {}, {}, {}
    for qa, source in sources.items():
        binding = deepcopy(source)
        binding.objects = []
        binding.prepared_measurements = {}
        binding.stats["version"] = M5_BINDING_VERSION
        receipt = {
            "schema_version": derivation.DERIVATION_SCHEMA_VERSION,
            "source_commit": derivation.LEGACY_SOURCE_COMMIT, "semantic_policy": "category-only",
            "source": anchors[qa], "scene_sources": anchors,
            "destination": destinations[qa], "scene_destinations": destinations,
            "implementation": {"original_implementation_sha256": "d" * 64},
            "scene_audit_qa_ids": sorted(sources),
            "output_binding_sha256": content_sha256(derivation._content(binding)),
            "new_model_requests": 0, "new_detector_requests": 0, "new_sam2_tracks": 0,
            "generation_resealed": False, "quality_assessed": False,
        }
        receipt["sha256"] = content_sha256(receipt)
        binding.stats["binding_derivation"] = receipt
        seals[qa] = deepcopy(receipt)
        original = case.catalog[qa]
        references[qa] = store.publish(
            request_identity=requests[qa], scene_artifact=original["scene_artifact"],
            scene_input_identity=inputs[qa], binding=binding, scene_migration_receipt=migrations[qa])
        derived_migrations[qa] = store.create_migration_receipt(
            references[qa]["key"], scene_input_identity=inputs[qa],
            destination_root=original["publication_root"],
            scene_artifact_ref=migrations[qa]["artifact"]["target_ref"],
            dependency_refs={k: v["target_ref"] for k, v in migrations[qa]["dependencies"].items()})
    _fullqa(case)
    approved = {"schema_version": "published-derived-replay-v1", "scenes": {"scene-one": {
        "references": references, "migration_receipts": derived_migrations}}}
    copied = _copy(case, extra_roots=[store.root])
    return case, copied, approved, seals, root, cfg, geometry


@pytest.mark.parametrize("receipt_files", [False, True])
def test_derived_replay_retains_original_v3_hashes_without_rederiving(
    tmp_path, monkeypatch, receipt_files,
):
    case, copied, approved, seals, root, cfg, geometry = _derived_case(tmp_path)
    if receipt_files:
        receipts = approved["scenes"]["scene-one"]["migration_receipts"]
        for qa, receipt in receipts.items():
            old_path = tmp_path / f"original-derived-receipt-{qa}.json"
            physical = copied.local / f"derived-receipt-{qa}.json"
            _write(physical, receipt)
            copied.descriptor["files"][str(old_path)] = {
                "path": str(physical), "identity": file_identity(physical)}
            receipts[qa] = str(old_path)
    monkeypatch.setattr(derivation, "derive_prepared_binding_scene_v11", _never)
    monkeypatch.setattr(
        "skill3d.evaluation.derived_published_fitset_v11.publish_derived_binding_scene_v11", _never)
    provider = DerivedPublishedFitSetInputProvider(
        copied.manifest_path, derivation_root=root, binding_cfg=cfg,
        transport_locator=copied.descriptor, derived_replay=approved)
    item = provider(case.variant)
    result = provider.prepare_published(case.variant, item)
    assert result["prepared"].stats["binding_derivation"] == seals["qa-00"]
    assert result["derived_reference"] == approved["scenes"]["scene-one"]["references"]["qa-00"]
    assert result["reference"] == case.catalog["qa-00"]["reference"]
    assert result["reference"]["request_identity"]["store_source"]["sha256"] == "e" * 64
    context = result["binding_derivation_context"]
    assert derivation._check_cache(context)
    derivation.validate_derived_prepared_binding(
        result["prepared"], context=context, frames=item.pixels, **geometry,
        inputs=question_input(item.episode.question, "object_rel_distance", item.episode.options),
        scene_identity=str(case.frozen.artifact.reconstruction_input_identity),
        parameters=context.target_requests["qa-00"]["parameters"])
    spawned = pickle.loads(pickle.dumps(provider.for_spawned_episode(case.variant)))
    assert spawned.prepare_published(case.variant, spawned(case.variant))["prepared"].stats[
        "binding_derivation"] == seals["qa-00"]


def test_derived_replay_rejects_changed_initial_source_reference(tmp_path):
    case, copied, approved, _, root, cfg, _ = _derived_case(tmp_path)
    manifest = json.loads(copied.manifest_path.read_text())
    manifest["entries"]["qa-00"]["reference"]["store_root"] = str(tmp_path / "different-source")
    # Change the fullQA catalog too: equality alone is not a provenance seal.
    catalog_row = copied.descriptor["files"][manifest["source_catalog_file"]]
    catalog = json.loads(Path(catalog_row["path"]).read_text())
    catalog["variants"]["base"]["qa-00"]["reference"] = manifest["entries"]["qa-00"]["reference"]
    _write(Path(catalog_row["path"]), catalog)
    catalog_row["identity"] = file_identity(catalog_row["path"])
    manifest["source_catalog_sha256"] = catalog_row["identity"]["sha256"]
    _write(copied.manifest_path, manifest)
    copied.descriptor["roots"][str(tmp_path / "different-source")] = (
        copied.descriptor["roots"][str(case.store.root)])
    provider = DerivedPublishedFitSetInputProvider(
        copied.manifest_path, derivation_root=root, binding_cfg=cfg,
        transport_locator=copied.descriptor, derived_replay=approved)
    with pytest.raises(PublishedFitSetInputError, match="provenance changed"):
        provider.prepare_published(case.variant, provider(case.variant))
