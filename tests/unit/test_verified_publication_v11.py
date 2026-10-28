"""Reuse producer-verified outputs without secondary audits or model requests."""

from copy import deepcopy
import pickle

import pytest

from skill3d.cluster import scene_compute
from skill3d.evaluation.derived_published_fitset_v11 import DerivedPublishedFitSetInputProvider
from skill3d.evaluation.verified_publication_v11 import ProducerVerifiedBindingContext
from skill3d.infra import m5_identity
from skill3d.segmentation.prepared_binding_store import PreparedBindingStore
from skill3d.segmentation.question_binding import question_input, validate_prepared_binding
from skill3d.storage.relocation import ValidatedRelocation

from test_published_physical_transport import _derived_case, _never


def test_verified_replay_decodes_only_the_selected_binding(tmp_path, monkeypatch):
    case, copied, approved, seals, root, cfg, _ = _derived_case(tmp_path)
    provider = DerivedPublishedFitSetInputProvider(
        copied.manifest_path, derivation_root=root, binding_cfg=cfg,
        transport_locator=copied.descriptor, derived_replay=approved)
    provider.reuse_verified_publications()
    # Production fixtures above already validated their publications. Everything
    # below must decode and resolve them without rechecking files or arrays.
    monkeypatch.setattr(scene_compute, "_load_scene", _never)
    monkeypatch.setattr(m5_identity, "file_identity", _never)
    monkeypatch.setattr(m5_identity, "array_identity", _never)
    monkeypatch.setattr(PreparedBindingStore, "validate_reference", _never)
    monkeypatch.setattr(PreparedBindingStore, "_payload_identity", _never)
    monkeypatch.setattr(PreparedBindingStore, "validate_migration_receipt", _never)
    monkeypatch.setattr(ValidatedRelocation, "recompute_scene_input_identity", _never)
    monkeypatch.setattr(provider, "_replay_derived_scene", _never)
    loaded = []
    load = PreparedBindingStore.load_verified

    def spy(store, key):
        loaded.append(key)
        return load(store, key)

    monkeypatch.setattr(PreparedBindingStore, "load_verified", spy)
    item = provider(case.variant)
    result = provider.prepare_published(case.variant, item)
    assert loaded == [approved["scenes"]["scene-one"]["references"]["qa-00"]["key"]]
    assert result["reference"] == case.catalog["qa-00"]["reference"]
    assert result["prepared"].stats["binding_derivation"] == seals["qa-00"]
    assert result["derived_reference"] == approved["scenes"]["scene-one"]["references"]["qa-00"]
    assert not provider._derived_scenes
    context = result["binding_derivation_context"]
    assert isinstance(context, ProducerVerifiedBindingContext)
    kwargs = dict(
        frames=item.pixels, depth=None, c2w=None, intrinsics=None,
        point_conf=None, depth_conf=None, point_map=None,
        inputs=question_input(item.episode.question, "object_rel_distance", item.episode.options),
        scene_identity=str(case.frozen.artifact.reconstruction_input_identity),
        parameters=context.request["parameters"], seed=context.request["seed"],
        binding_derivation_context=context)
    validate_prepared_binding(result["prepared"], **kwargs)
    with pytest.raises(ValueError, match="decoded object"):
        validate_prepared_binding(deepcopy(result["prepared"]), **kwargs)
    with pytest.raises(ValueError, match="question/scene"):
        validate_prepared_binding(result["prepared"], **{**kwargs, "inputs": {"changed": True}})
    assert len(provider.frozen_frame_bytes(case.variant)) == 32
    first_pngs = provider.frozen_frame_bytes(case.variant)
    assert first_pngs is provider.frozen_frame_bytes(case.variant)
    # Spawn keeps the trusted descriptor/mode, with no all-scene derivation.
    clone = pickle.loads(pickle.dumps(provider.for_spawned_episode(case.variant)))
    again = clone.prepare_published(case.variant, clone(case.variant))
    assert again["prepared"].stats["binding_derivation"] == seals["qa-00"]
    assert again["derived_reference"] == result["derived_reference"]


def test_verified_replay_preserves_public_episode_isolation(tmp_path):
    case, copied, approved, _, root, cfg, _ = _derived_case(tmp_path)
    provider = DerivedPublishedFitSetInputProvider(
        copied.manifest_path, derivation_root=root, binding_cfg=cfg,
        transport_locator=copied.descriptor, derived_replay=approved)
    provider.reuse_verified_publications()
    item = provider(case.variant)
    item.pixels[0][:] = 255
    assert provider(case.variant).pixels[0][0, 0, 0] == 0
    item.episode.ground_truth = "hidden"
    with pytest.raises(ValueError, match="public episode"):
        provider.prepare_published(case.variant, item)
    with pytest.raises(ValueError, match="panel/catalog"):
        provider({**case.variant, "source_split": "inner_validation"})


def test_offline_frames_reuse_original_pngs_in_one_admitted_section(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from PIL import Image
    from skill3d.evaluation.fitset_v11 import SystemVariantRunner, _variant_key
    from skill3d.storage import hot_cache as hot
    from skill3d.storage.framework_state_io import FrameworkStateIO
    from skill3d.storage.hot_cache import HotCache
    from skill3d.storage.state_workspace import StateWorkspace
    from test_atomic_workspace_io import free_disk

    case, copied, approved, _, root, cfg, _ = _derived_case(tmp_path)
    provider = DerivedPublishedFitSetInputProvider(
        copied.manifest_path, derivation_root=root, binding_cfg=cfg,
        transport_locator=copied.descriptor, derived_replay=approved)
    provider.reuse_verified_publications()
    item = provider(case.variant)
    cache = HotCache(tmp_path / "state-cache", quota_bytes=64 * 1024**2, disk_usage=free_disk)
    cache.populate("state", lambda _: None, kind="workspace",
                   workspace_bytes=16 * 1024**2, allocation_bytes=16 * 1024**2)
    StateWorkspace.adopt(
        cache, "state", baseline_anchor="1" * 64, baseline_members={},
        mutable_roots=("accumulation",), workspace_bytes=16 * 1024**2,
        allocation_bytes=16 * 1024**2)
    framework = FrameworkStateIO(StateWorkspace(cache, "state", bounded=True), job="synthetic")
    runner = SystemVariantRunner.__new__(SystemVariantRunner)
    runner.root = framework.root / "accumulation"
    runner.framework_state_io = framework
    runner.published_inputs = provider
    runner.logical_path = lambda path: str(path)
    runner._write_create_once = framework.write_json_once
    # Successful offline freezing must use the bytes supplied by production;
    # reencoding an image or scanning the workspace would fail this check.
    monkeypatch.setattr(Image.Image, "save", _never)
    monkeypatch.setattr(hot, "_tree_bytes", _never)
    runner._freeze_offline_case_input(case.variant, item)
    directory = runner.root / "variants" / _variant_key(case.variant["variant_id"])
    frozen = json.loads((directory / "offline_input.json").read_bytes())
    assert len(frozen["image_refs"]) == 32
    for index, record in enumerate(frozen["image_refs"]):
        assert Path(record["path"]).read_bytes() == provider.frozen_frame_bytes(case.variant)[index]
    assert not {"ground_truth", "answer", "labels"}.intersection(frozen)
