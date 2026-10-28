"""Scene-only verified spawn with synthetic pinned copies and no model calls."""

from copy import deepcopy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.cluster import scene_compute
from skill3d.evaluation.derived_published_fitset_v11 import DerivedPublishedFitSetInputProvider
from skill3d.evaluation.published_fitset_v11 import (
    PublishedFitSetInputError, PublishedFitSetInputProvider,
)
from skill3d.evolution import prepared_binding_derivation_v11 as derivation
from skill3d.infra import m5_identity
from skill3d.infra.m5_identity import content_sha256, file_identity
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore, PreparedBindingStoreError,
)
from skill3d.storage.published_transport import (
    PUBLISHED_TRANSPORT_SCHEMA, PublishedTransportLocator,
)
from skill3d.storage.relocation import RelocationError
from skill3d.storage.hot_cache import IntegrityError, StorageError
from skill3d.storage.scene_runtime import SceneHotInputProvider, SceneHotRuntime

from test_cluster_scene_compute import _write
from test_published_fitset_v11 import _assert_safe, _published
from test_published_physical_transport import _copy, _derived_case, _fullqa, _never


def _root_row(path):
    members = {p.relative_to(path).as_posix(): file_identity(p)
               for p in path.rglob("*") if p.is_file()}
    return {"path": str(path), "members": members,
            "members_sha256": content_sha256(members)}


def _add_excluded_scene(case, copied, approved=None, derivation_root=None):
    """Add inert catalog/transport decoys that must never reach this child."""
    marker = "excluded-scene-metadata"
    manifest = json.loads(copied.manifest_path.read_bytes())
    excluded = deepcopy(manifest["entries"]["qa-00"])
    excluded.update(qa_id="qa-excluded", scene_id=marker)
    for name in ("input_root", "publication_root"):
        source = str(case.manifest_path.parent / marker / name)
        target = copied.local / marker / name
        target.mkdir(parents=True)
        (target / "excluded.bin").write_bytes(b"synthetic excluded bytes")
        copied.descriptor["roots"][source] = _root_row(target)
        excluded[name] = source
    excluded["scene_artifact"] = str(Path(excluded["publication_root"]) / "artifact.json")
    excluded["reference"]["store_root"] = excluded["publication_root"]
    for name in ("scene_input_identity_file", "migration_receipt_file"):
        source = str(case.manifest_path.parent / marker / (name + ".json"))
        target = copied.local / marker / (name + ".json")
        _write(target, {"synthetic_scope": marker})
        copied.descriptor["files"][source] = {
            "path": str(target), "identity": file_identity(target),
        }
        excluded[name] = source
        excluded[name + "_sha256"] = file_identity(target)["sha256"]
    manifest["entries"]["qa-excluded"] = excluded
    manifest["scene_scopes"][marker] = {
        "qa_ids": ["qa-excluded"], "split": excluded["split"],
        "input_root": excluded["input_root"],
        "frozen_scene_sha256": excluded["frozen_scene_sha256"],
    }
    catalog_source = manifest["source_catalog_file"]
    catalog_row = copied.descriptor["files"][catalog_source]
    catalog_path = Path(catalog_row["path"])
    catalog = json.loads(catalog_path.read_bytes())
    catalog["variants"]["base"]["qa-excluded"] = {
        k: v for k, v in excluded.items() if k not in {"input_root", "frozen_scene_sha256"}
    }
    _write(catalog_path, catalog)
    catalog_row["identity"] = file_identity(catalog_path)
    manifest["source_catalog_sha256"] = catalog_row["identity"]["sha256"]
    _write(copied.manifest_path, manifest)
    copied.descriptor["files"][str(case.manifest_path)]["identity"] = (
        file_identity(copied.manifest_path))
    if approved is not None:
        reference = deepcopy(approved["scenes"]["scene-one"]["references"]["qa-00"])
        reference["store_root"] = str(derivation_root / marker)
        approved["scenes"][marker] = {
            "references": {"qa-excluded": reference},
            "migration_receipts": {"qa-excluded": {"synthetic_scope": marker}},
        }
    # The shared publication root also contains bytes outside the selected
    # migration/store-entry trees. No full-root subset may retain these rows.
    root = copied.descriptor["roots"][str(copied.publication)]
    (Path(root["path"]) / "excluded-shared.bin").write_bytes(b"synthetic shared decoy")
    root.update(_root_row(Path(root["path"])))
    return marker, {
        **case.variant, "qa_id": "qa-excluded", "scene_id": marker,
        "artifact_ref": excluded["reference"]["scene_artifact_ref"],
    }


def _rename_geometry(case, copied):
    root_row = copied.descriptor["roots"][str(copied.publication)]
    root = Path(root_row["path"])
    records = {}
    for migration in case.validation["variants"]["base"]["migration_receipts"].values():
        for record in (migration["artifact"], *migration["dependencies"].values()):
            records[record["target_ref"]] = record
    for index, (name, record) in enumerate(records.items()):
        old = root / name
        target = root / "renamed" / f"member-{index}{old.suffix}"
        target.parent.mkdir(exist_ok=True)
        old.rename(target)
        copied.descriptor["files"][str(copied.publication / name)] = {
            "path": str(target), "identity": record["identity"],
        }
    root_row.update(_root_row(root))


def _real_wrapper(tmp_path, provider, case, copied, marker, approved=None):
    # Input bytes are already installed by the synthetic publication fixture;
    # empty shard members let the real constructor check its local-only setup.
    closure = {
        "schema_version": "h3d-scene-closure-v1",
        "original_published_manifest": str(case.manifest_path),
        "descriptor": copied.descriptor,
        "shards": {
            name: {"key": "scope-" + name, "members": {}, "bytes": 0,
                   "allocation_bytes": 65536}
            for name in ("global", "state", "scene-one", marker)
        },
    }
    if approved is not None:
        closure["derived_replay"] = approved
    path = tmp_path / "synthetic-closure.json"
    _write(path, closure)
    runtime = SceneHotRuntime(
        closure_path=path, cache_root=tmp_path / "runtime-cache",
        endpoint="http://unused.invalid", token_file=tmp_path / "no-token",
        closure_sha256=file_identity(path)["sha256"])
    return SceneHotInputProvider(provider, runtime)


@pytest.mark.parametrize("original_manifest", [False, True])
def test_verified_base_spawn_retains_whole_scene_and_global_anchors(
    tmp_path, monkeypatch, original_manifest,
):
    case = _published(tmp_path)
    _fullqa(case)
    copied = _copy(case)
    marker, excluded_variant = _add_excluded_scene(case, copied)
    _rename_geometry(case, copied)
    provider = PublishedFitSetInputProvider(
        case.manifest_path if original_manifest else copied.manifest_path,
        transport_locator=copied.descriptor)
    wrapper = _real_wrapper(tmp_path, provider, case, copied, marker)
    originals = {
        qa: wrapper.prepare_published(variant, wrapper(variant))
        for qa, variant in case.variants.items()
    }
    contract = wrapper.input_contract_identity
    provider_contract = provider.input_contract_identity
    entries = deepcopy(provider._entries)
    scopes = deepcopy(provider.scene_scopes)
    descriptor_hash = provider.transport_locator.descriptor_sha256
    monkeypatch.setattr(scene_compute, "_load_scene", _never)
    monkeypatch.setattr(m5_identity, "file_identity", _never)
    monkeypatch.setattr(m5_identity, "content_sha256", _never)
    monkeypatch.setattr("skill3d.storage.published_transport.content_sha256", _never)
    monkeypatch.setattr(SceneHotRuntime, "_request", _never)
    # A full-catalog deepcopy during spawn would visit this inert decoy.
    class NoCopy(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("spawn copied an excluded catalog entry")
    provider._entries["qa-excluded"] = NoCopy(provider._entries["qa-excluded"])
    payload = pickle.dumps(wrapper.for_spawned_episode(case.variant))
    assert marker.encode() not in payload
    assert b"excluded-shared.bin" not in payload
    spawned = pickle.loads(payload)
    child = spawned.provider
    assert isinstance(spawned.runtime, SceneHotRuntime)
    assert set(spawned.runtime.closure["shards"]) == {"global", "state", "scene-one"}
    assert "descriptor" not in spawned.runtime.closure
    assert spawned.runtime.scenes == ("scene-one",)
    assert spawned.runtime.closure_sha256 == wrapper.runtime.closure_sha256
    assert spawned.runtime.locator is child.transport_locator
    assert spawned.input_contract_identity == contract
    assert set(child._entries) == set(case.catalog)
    assert child._entries == {qa: entries[qa] for qa in case.catalog}
    assert child.scene_scopes == {"scene-one": scopes["scene-one"]}
    assert child.input_contract_identity == provider_contract
    assert child.transport_locator.descriptor_sha256 == descriptor_hash
    assert child.transport_locator._descriptor is None
    assert str(case.manifest_path) in child.transport_locator._explicit_file_sources
    assert case.manifest["source_catalog_file"] in child.transport_locator._verified_files
    assert spawned.logical_path(copied.manifest_path) == case.manifest_path
    assert spawned.logical_file(
        case.manifest_path, provider_contract["manifest"]["sha256"]) == copied.manifest_path
    frozen_source = str(Path(entries["qa-00"]["input_root"]) / "scene.json")
    frozen_path = spawned.logical_file(frozen_source, entries["qa-00"]["frozen_scene_sha256"])
    # Root-derived aliases must not expand logical_path's original files-only API.
    for candidate in (wrapper, spawned):
        with pytest.raises(IntegrityError, match="unique declared logical origin"):
            candidate.logical_path(frozen_path)
    assert not child.cache and not child._publication_files and not child._scene_checks
    assert not child._scene_transport_scopes
    # Repeated parent spawns reuse the selected tables, not full descriptor work.
    assert provider.for_spawned_episode(case.variant).transport_locator is (
        provider._scene_transport_scopes["scene-one"])
    assert set(provider._entries) == set(entries)
    assert provider.scene_scopes == scopes
    with pytest.raises(PublishedFitSetInputError, match="catalog entry is missing"):
        spawned(excluded_variant)
    with pytest.raises(StorageError):
        with spawned.episode_lease(excluded_variant):
            pytest.fail("excluded scene obtained a runtime lease")
    with pytest.raises(StorageError):
        spawned.runtime.prefetch(marker)
    with pytest.raises(RelocationError):
        child.transport_locator.verified_root(entries["qa-excluded"]["input_root"])
    with pytest.raises(RelocationError):
        child.transport_locator.file(entries["qa-excluded"]["scene_input_identity_file"])
    # Hashing is forbidden only for spawn; receipts retain their original seals.
    monkeypatch.undo()
    for qa, variant in case.variants.items():
        item = spawned(variant)
        assert item.episode.ground_truth == ""
        result = spawned.prepare_published(variant, item)
        original = originals[qa]
        assert result["reference"] == original["reference"]
        assert result["identity"] == original["identity"]
        assert result["transport_receipt"] == original["transport_receipt"]
        assert set(result["prepared"].stats) == set(original["prepared"].stats)
        np.testing.assert_array_equal(
            result["prepared"].stats["array"], original["prepared"].stats["array"])
        assert result["artifact_path"] == original["artifact_path"]
        assert result["artifact_path"].parent.name == "renamed"
        _assert_safe(result["transport_receipt"])
    _assert_safe(child.entries)
    _assert_safe(child.input_contract_identity)
    leaked = child(case.variant)
    leaked.episode.ground_truth = "synthetic forbidden answer"
    with pytest.raises(PublishedFitSetInputError, match="public episode"):
        child.prepare_published(case.variant, leaked)
    assert child(case.variant).episode.ground_truth == ""


@pytest.mark.parametrize("receipt_files", [False, True])
def test_verified_derived_spawn_keeps_current_scene_seals_and_receipt_files(
    tmp_path, monkeypatch, receipt_files,
):
    case, copied, approved, seals, root, cfg, _ = _derived_case(tmp_path)
    if receipt_files:
        receipts = approved["scenes"]["scene-one"]["migration_receipts"]
        for qa, receipt in receipts.items():
            source = str(tmp_path / f"original-derived-receipt-{qa}.json")
            physical = copied.local / f"derived-receipt-{qa}.json"
            _write(physical, receipt)
            copied.descriptor["files"][source] = {
                "path": str(physical), "identity": file_identity(physical),
            }
            receipts[qa] = source
    marker, excluded = _add_excluded_scene(case, copied, approved, root)
    _rename_geometry(case, copied)
    provider = DerivedPublishedFitSetInputProvider(
        copied.manifest_path, derivation_root=root, binding_cfg=cfg,
        transport_locator=copied.descriptor, derived_replay=approved)
    wrapper = _real_wrapper(tmp_path, provider, case, copied, marker, approved)
    originals = {
        qa: wrapper.prepare_published(variant, wrapper(variant))
        for qa, variant in case.variants.items()
    }
    contract = wrapper.input_contract_identity
    # Verified mode must never transfer or validate a strict derivation context.
    provider._derived_scenes = {"scene-one": ("not-a-strict-context",),
                                marker: ("excluded-context",)}
    monkeypatch.setattr(
        "skill3d.evaluation.derived_published_fitset_v11._check_cache", _never)
    monkeypatch.setattr(PreparedBindingStore, "validate_reference", _never)
    monkeypatch.setattr(PreparedBindingStore, "validate_migration_receipt", _never)
    monkeypatch.setattr(derivation, "derive_prepared_binding_scene_v11", _never)
    monkeypatch.setattr(
        "skill3d.evaluation.derived_published_fitset_v11.publish_derived_binding_scene_v11", _never)
    monkeypatch.setattr(SceneHotRuntime, "_request", _never)
    payload = pickle.dumps(wrapper.for_spawned_episode(case.variant))
    assert marker.encode() not in payload
    assert b"excluded-context" not in payload
    assert b"excluded-shared.bin" not in payload
    spawned = pickle.loads(payload)
    child = spawned.provider
    assert child._approved_derived_replay == {
        "schema_version": approved["schema_version"],
        "scenes": {"scene-one": approved["scenes"]["scene-one"]},
    }
    assert not child._derived_scenes and child._spawn_replay_scene is None
    assert isinstance(spawned.runtime, SceneHotRuntime)
    assert set(spawned.runtime.closure["shards"]) == {"global", "state", "scene-one"}
    assert spawned.runtime.closure["derived_replay"] == child._approved_derived_replay
    assert spawned.runtime.closure_sha256 == wrapper.runtime.closure_sha256
    assert spawned.runtime.locator is child.transport_locator
    assert spawned.input_contract_identity == contract
    assert child.transport_locator.descriptor_sha256 == provider.transport_locator.descriptor_sha256
    if receipt_files:
        for source in approved["scenes"]["scene-one"]["migration_receipts"].values():
            assert source in child.transport_locator._explicit_file_sources
            assert child.transport_locator.verified_file(source).is_file()
    with pytest.raises(PublishedFitSetInputError, match="catalog entry is missing"):
        spawned(excluded)
    with pytest.raises(StorageError):
        with spawned.episode_lease(excluded):
            pytest.fail("excluded scene obtained a runtime lease")
    for qa, variant in case.variants.items():
        result = spawned.prepare_published(variant, spawned(variant))
        original = originals[qa]
        assert result["reference"] == case.catalog[qa]["reference"]
        assert result["derived_reference"] == original["derived_reference"]
        assert result["derived_migration_receipt"] == original["derived_migration_receipt"]
        assert result["transport_receipt"] == original["transport_receipt"]
        assert result["identity"] == original["identity"]
        assert result["prepared"].stats["generation"] == original["prepared"].stats["generation"]
        assert result["prepared"].stats["binding_derivation"] == seals[qa]
        _assert_safe(result["transport_receipt"])
    respawned = pickle.loads(pickle.dumps(spawned.for_spawned_episode(case.variant)))
    assert respawned.input_contract_identity == contract
    assert set(respawned.provider._entries) == set(case.catalog)
    assert respawned.prepare_published(case.variant, respawned(case.variant))[
        "transport_receipt"] == originals["qa-00"]["transport_receipt"]


def test_locator_scope_preserves_nested_aliases_overrides_and_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "copy"
    current = root / "current"
    current.mkdir(parents=True)
    (current / "renamed.json").write_bytes(b"synthetic current bytes")
    (root / "excluded.bin").write_bytes(b"synthetic excluded bytes")
    source = "/synthetic/all"
    logical_file = source + "/current/original.json"
    identity = file_identity(current / "renamed.json")
    descriptor = {
        "schema_version": PUBLISHED_TRANSPORT_SCHEMA,
        "files": {logical_file: {"path": str(current / "renamed.json"), "identity": identity}},
        "roots": {source: _root_row(root), source + "/current": _root_row(current)},
    }
    locator = PublishedTransportLocator(descriptor)
    anchor = locator.descriptor_sha256
    monkeypatch.setattr("skill3d.storage.published_transport.content_sha256", _never)
    scoped = locator.for_sources(
        files=[logical_file], roots=[source, source + "/current"],
        trees=[source + "/current"])
    scoped = pickle.loads(pickle.dumps(scoped))
    assert scoped.descriptor_sha256 == anchor
    assert scoped._descriptor is None
    assert scoped._explicit_file_sources == {logical_file}
    assert set(locator.sources_for_file(current / "renamed.json")) == {
        logical_file, source + "/current/renamed.json",
    }
    assert locator.explicit_sources_for_file(current / "renamed.json") == (logical_file,)
    assert scoped.explicit_sources_for_file(current / "renamed.json") == (logical_file,)
    assert scoped.explicit_sources_for_file(root / "excluded.bin") == ()
    assert scoped.verified_member(source, "current/original.json", identity) == (
        current / "renamed.json")
    assert scoped.verified_member(source + "/current", "original.json", identity) == (
        current / "renamed.json")
    assert scoped.file(logical_file, identity["sha256"]) == current / "renamed.json"
    assert locator._descriptor == descriptor
    assert locator.root(source) == root
    assert locator.member(source, "current/original.json", identity) == current / "renamed.json"
    with pytest.raises(RelocationError, match="complete root closure"):
        scoped.root(source)
    with pytest.raises(RelocationError):
        scoped.verified_file(source + "/excluded.bin")
    with pytest.raises(RelocationError):
        scoped.verified_member(source, "excluded.bin", file_identity(root / "excluded.bin"))
    with pytest.raises(RelocationError):
        scoped.verified_root(source + "/current/undeclared")
    with pytest.raises(RelocationError):
        scoped.for_sources(trees=[source + "/excluded"])
    with pytest.raises(RelocationError):
        scoped.for_sources(files=[source + "/excluded.bin"])
    with pytest.raises(RelocationError):
        scoped.for_sources(roots=[source + "/undeclared"])
    with pytest.raises(RelocationError):
        scoped.file(str(root / "excluded.bin"))
    (root / "unapproved.bin").write_bytes(b"synthetic extra bytes")
    with pytest.raises(RelocationError, match="closure changed"):
        locator.root(source)


def test_strict_derived_spawn_keeps_full_inputs_and_validated_context(tmp_path, monkeypatch):
    case, copied, approved, seals, root, cfg, _ = _derived_case(tmp_path)
    provider = DerivedPublishedFitSetInputProvider(
        copied.manifest_path, derivation_root=root, binding_cfg=cfg,
        transport_locator=copied.descriptor, derived_replay=approved)
    original = provider.prepare_published(case.variant, provider(case.variant))
    context = original["binding_derivation_context"]
    assert derivation._check_cache(context)
    provider._entries["excluded"] = {"synthetic_scope": "must-retain-in-strict-inputs"}
    provider._derived_scenes["excluded"] = ("must-not-transfer",)
    clone = pickle.loads(pickle.dumps(provider.for_spawned_episode(case.variant)))
    assert clone._entries == provider._entries
    assert clone.scene_scopes == provider.scene_scopes
    assert clone._approved_derived_replay == approved
    assert clone.transport_locator._descriptor == copied.descriptor
    assert not clone.transport_locator._scoped
    assert set(clone._derived_scenes) == {"scene-one"}
    assert clone._spawn_replay_scene == "scene-one"
    assert derivation._check_cache(clone._derived_scenes["scene-one"][0])
    monkeypatch.setattr(DerivedPublishedFitSetInputProvider, "_replay_derived_scene", _never)
    result = clone.prepare_published(case.variant, clone(case.variant))
    assert result["prepared"].stats["binding_derivation"] == seals["qa-00"]
    assert result["transport_receipt"] == original["transport_receipt"]
    clone._derived_scenes["scene-one"][0].cfg.max_tokens += 1
    with pytest.raises(PreparedBindingStoreError, match="cached derivation context"):
        clone.for_spawned_episode(case.variant)


def test_original_explicit_alias_ambiguity_remains_rejected_after_scope_and_pickle(tmp_path):
    physical = tmp_path / "manifest.json"
    physical.write_bytes(b"synthetic manifest")
    row = {"path": str(physical), "identity": file_identity(physical)}
    locator = PublishedTransportLocator({
        "schema_version": PUBLISHED_TRANSPORT_SCHEMA,
        "files": {"/synthetic/current/manifest.json": row,
                  "/synthetic/excluded/manifest.json": row},
        "roots": {},
    })
    scoped = pickle.loads(pickle.dumps(locator.for_sources(
        files=["/synthetic/current/manifest.json"])))
    assert scoped.explicit_sources_for_file(physical) == (
        "/synthetic/current/manifest.json",)
    assert scoped.has_ambiguous_explicit_source(physical)
    assert "/synthetic/excluded/manifest.json" not in scoped._verified_files
    for candidate in (locator, scoped):
        wrapper = SceneHotInputProvider.__new__(SceneHotInputProvider)
        wrapper.runtime = SimpleNamespace(locator=candidate)
        with pytest.raises(IntegrityError, match="unique declared logical origin"):
            wrapper.logical_path(physical)
