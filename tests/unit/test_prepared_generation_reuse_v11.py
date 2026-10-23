"""Strict generation derivation and preparation, with real stores and no models."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.adapters.episode_source import EpisodeItem, _episode_from_row
from skill3d.adapters.frame_set import build_frame_set, with_pixel_identity
from skill3d.evaluation.skill_ablation_v11 import input_identity
from skill3d.evolution import prepared_bindings_v11 as prep
from skill3d.evolution.prepared_generation_reuse_v11 import (
    PACKAGE_SCHEMA_VERSION, make_generation_reuse_factory,
    make_local_generation_reuse_factory, reuse_verified_binding,
)
from skill3d.infra.m5_identity import content_sha256, file_identity, generation_identity
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig, PreparedObjectBinding
from skill3d.schemas import ObjectRecord, ReconstructionArtifact
from skill3d.segmentation.binding_validation import validate_binding
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore, PreparedBindingStoreError, load_prepared_binding_reference,
    prepared_binding_request_identity,
)
from skill3d.segmentation.question_binding import M5_BINDING_VERSION
from skill3d.storage.relocation import artifact_dependency_refs


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def _never(*_args, **_kwargs):
    raise AssertionError("unexpected model/client operation")


def _quality(passed=True):
    return {
        "warp_inlier_ratio": 1, "warp_photometric_inlier_ratio": 1,
        "cloud_overlap_ratio": 1, "point_depth_relative_p95": 0,
        "cloud_neighbor_tail_ratio": 0, "main_gate_passed": passed,
        "g1_blur_ok": 1, "g2_brightness": 1, "g3_motion_blur": 0,
        "g4_frame_count": 32, "g6_depth_var_coeff": 0,
        "g7_dynamic_ratio": 0, "g9_tracker_consistency": 1,
        "g10_baseline_quality": 1, "overall_quality": 1,
    }


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_prepare_m4_quality", lambda art, *_a, **_k: art)
    monkeypatch.setattr(runner, "_bind_objects_best_effort", _never)
    monkeypatch.setattr(runner, "_make_online_client", _never)
    pixels = [np.full((3, 4, 3), i, dtype=np.uint8) for i in range(32)]
    items = []
    for qa_id in ("qa-1", "qa-2"):
        fs = with_pixel_identity(build_frame_set(
            32, episode_id=qa_id, dataset_id="scannet", video_id="scene",
            scene_name="scene", fps=24,
        ), pixels)
        ep = _episode_from_row({
            "qa_id": qa_id, "scene_name": "scene", "dataset": "scannet",
            "question": "Which chair is nearer?", "question_type": "object_rel_distance",
            "options": ["A", "B"], "ground_truth": "",
        }, "induction", pixels, 32, frame_set=fs)
        items.append(EpisodeItem(ep, pixels, source="vsi_bench", video_path="/old/video.mp4"))
    root = tmp_path / "source"
    root.mkdir()
    arrays = {
        "depth_maps": np.ones((32, 3, 4), dtype=np.float32),
        "c2w_list": np.repeat(np.eye(4, dtype=np.float32)[None], 32, axis=0),
        "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 32, axis=0),
        "point_conf": np.ones((32, 3, 4), dtype=np.float32),
        "depth_conf": np.ones((32, 3, 4), dtype=np.float32),
        "point_map": np.ones((32, 3, 4, 3), dtype=np.float32),
    }
    refs = {}
    for name, value in arrays.items():
        path = root / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        refs[name] = str(path)
    art = ReconstructionArtifact(
        artifact_id="scene-vggt", artifact_version="base", scene_name="scene",
        frame_ids=items[0].episode.frame_set.frame_ids,
        frame_set_hash=items[0].episode.frame_set.frame_set_hash,
        source_frame_indices=items[0].episode.frame_set.source_frame_indices,
        timestamps=items[0].episode.frame_set.timestamps,
        reconstruction_input_identity=items[0].episode.frame_set.cache_identity(),
        confidence={"per_point_confidence": refs["depth_conf"]},
        quality=_quality(), quality_status="computed", **refs,
    )
    path = root / "artifact.json"
    _write(path, art.model_dump(mode="json"))
    cfg = OnlineRunConfig(semantic_review_config=None, online_model_revision="pinned-v1")
    identities, references = {}, {}
    store = PreparedBindingStore(root / "store")
    for item in items:
        qa_id = item.episode.qa_id
        identity = input_identity(item, path)
        request = prepared_binding_request_identity(
            input_identity=identity, cfg=cfg, scene_audit_qa_ids=[qa_id],
        )
        generation = generation_identity(
            scene_name=art.scene_name, frame_set_hash=art.frame_set_hash, frames=pixels,
            depth_maps=arrays["depth_maps"], c2w_list=arrays["c2w_list"],
            intrinsics=arrays["intrinsics"], point_conf=arrays["depth_conf"],
            depth_conf=arrays["depth_conf"], point_map=arrays["point_map"],
            frame_provenance=runner._episode_frame_provenance(item.episode, pixels),
            model={
                "enabled": True, "provider": cfg.online_model_provider,
                "model": cfg.online_model, "model_revision": cfg.online_model_revision,
                "endpoint_sha256": request["model"]["endpoint_sha256"],
                "seed": cfg.seed, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens,
                "cache_eligible": False,
            },
            detector={"enabled": True}, sam2={"checkpoint": "test-only"},
            parameters=request["parameters"], question=item.episode.question,
            question_type=runner.canonical_task(item.episode.question_type),
            options=item.episode.options, reconstruction_identity=art.reconstruction_input_identity,
        )
        mask, points = root / (qa_id + "-mask.npy"), root / (qa_id + "-points.npy")
        np.save(mask, np.ones((32, 3, 4), dtype=np.uint8))
        np.save(points, np.ones((4, 3), dtype=np.float32))
        obj = ObjectRecord(
            obj_id=qa_id + "-chair", category_name="chair", grounding_status="question_driven",
            mask_per_frame=str(mask), pointcloud_world=str(points),
            centroid_world=[1, 1, 1], bbox=[1, 1, 1, 1, 1, 1],
            visible_frames=list(range(32)), mask_support_ratio=1,
            track_id="track-" + qa_id, candidate_ids=["candidate-" + qa_id],
            question_roles=[{"role": "candidate"}], binding_provenance=[{
                "semantic_relation": "match", "semantic_visual_request_sha256": "v" * 64,
                "semantic_relation_request_sha256": "r" * 64,
            }],
        )
        obj.binding_validation = validate_binding(
            obj, depth_maps=arrays["depth_maps"], n_frames=32,
            generation_sha256=generation["sha256"],
        )
        assert obj.binding_validation.status == "validated"
        binding = PreparedObjectBinding(
            [obj], {"version": M5_BINDING_VERSION, "generation": generation,
                    "status": "prepared", "actual_model_requests": 3}, ["original generation"],
            True, {"plane_fit": {"old_metric_scale": 7}},
        )
        identities[qa_id] = identity
        references[qa_id] = store.publish(
            request_identity=request, scene_artifact=path, scene_input_identity=identity,
            binding=binding,
        )
    destination = tmp_path / "destination"
    destination.mkdir()
    dest_refs = {}
    for name, ref in refs.items():
        target = destination / Path(ref).name
        shutil.copyfile(ref, target)
        dest_refs[name] = str(target)
    new_art = art.model_copy(update={
        **dest_refs, "artifact_version": "destination",
        "confidence": art.confidence.model_copy(update={
            "per_point_confidence": dest_refs["depth_conf"],
        }),
    })
    new_path = destination / "artifact.json"
    _write(new_path, new_art.model_dump(mode="json"))
    new_items = [
        EpisodeItem(item.episode.model_copy(update={"split": "inner_validation"}, deep=True),
                    item.pixels, source=item.source, video_path=item.video_path)
        for item in items
    ]
    return SimpleNamespace(
        root=root, path=path, art=art, items=items, cfg=cfg, store=store,
        identities=identities, preparation={"bindings": references},
        new_art=new_art, new_path=new_path, new_items=new_items, arrays=arrays,
    )


def _local(case):
    return make_local_generation_reuse_factory(
        case.new_items, case.cfg, case.preparation, case.identities, case.path)


def _derive(case, factory=None, item=None, cfg=None):
    item = item or case.new_items[0]
    cfg = cfg or case.cfg
    identity = input_identity(item, case.new_path)
    request = prepared_binding_request_identity(
        input_identity=identity, cfg=cfg, scene_audit_qa_ids=["qa-1", "qa-2"],
    )
    return (factory or _local(case))(item, case.new_path, identity, request)


def _prepare(case, root, **kwargs):
    return prep.prepare_shared_bindings_v11(
        case.new_items, artifact_paths={i.episode.qa_id: case.new_path for i in case.new_items},
        base_cfg=case.cfg, store_root=root, min_free_gib=0,
        client_factory=_never, **kwargs,
    )


def _source_binding(case, qa_id="qa-1"):
    reference = case.preparation["bindings"][qa_id]
    return load_prepared_binding_reference(
        reference, scene_input_identity=case.identities[qa_id])


def _replace_source(case, binding, qa_id="qa-1"):
    reference = case.preparation["bindings"][qa_id]
    shutil.rmtree(case.store.entry_path(reference["key"]))
    case.preparation["bindings"][qa_id] = case.store.publish(
        request_identity=reference["request_identity"], scene_artifact=case.path,
        scene_input_identity=case.identities[qa_id], binding=binding,
    )


def _metric(case):
    from skill3d.reconstruction import metric_evidence as me, metric_fusion as mf

    art = case.new_art
    metric = np.full_like(case.arrays["depth_maps"], np.nan)
    valid = np.zeros_like(metric, dtype=bool)
    refs = {}
    for name, value in (("metric_depth_maps", metric), ("metric_valid_masks", valid)):
        path = case.new_path.parent / (name + ".npy")
        np.save(path, value)
        refs[name] = str(path)
    fusion = mf.fuse_metric_scale(metric, case.arrays["depth_maps"], valid, model="moge2")
    request = {
        "input_identity": case.new_items[0].episode.frame_set.cache_identity(),
        "sources": {name: file_identity(getattr(art, name))["sha256"]
                    for name in me.GEOMETRY_REFS if getattr(art, name)},
        "frame_ids": art.frame_ids, "frame_set_hash": art.frame_set_hash,
        "scene_name": art.scene_name, "source_frame_indices": art.source_frame_indices,
        "timestamps": art.timestamps,
    }
    receipt = mf.per_frame_receipt(fusion)
    receipt.update(request=request, request_sha256=me.digest(request),
                   evidence_sha256={name: file_identity(ref)["sha256"] for name, ref in refs.items()})
    receipt_path = case.new_path.parent / "scale_receipt.json"
    _write(receipt_path, receipt)
    case.new_art = art.model_copy(update={
        **refs, "artifact_version": "metric", "per_frame_scale_ref": str(receipt_path),
        "metric_receipt_sha256": file_identity(receipt_path)["sha256"],
        "metric_input_identity": request["input_identity"],
        "metric_request_sha256": me.digest(request), "metric_model": fusion.model,
        "metric_fusion_version": fusion.version, "metric_scale": fusion.metric_scale,
        "scale_self_consistency": fusion.scale_self_consistency, "scale_fusion_status": fusion.status,
    })
    _write(case.new_path, case.new_art.model_dump(mode="json"))


def test_local_metric_reuses_every_qa_preserves_generation_and_audits(case, tmp_path, capsys):
    _metric(case)
    before = {q: _source_binding(case, q).stats["generation"] for q in case.identities}
    audits = []
    original = prep.audit_semantic_mask_conflicts

    def audit(bindings, **kwargs):
        audits.append([q for q, _ in bindings])
        return original(bindings, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(prep, "audit_semantic_mask_conflicts", audit)
        result = _prepare(case, tmp_path / "metric-store", reuse_factory=_local(case))
    assert result["generation_reused_qa_ids"] == ["qa-1", "qa-2"]
    assert result["cache_hit_qa_ids"] == []
    assert audits == [["qa-1", "qa-2"]]
    for item in case.new_items:
        q = item.episode.qa_id
        reference = result["bindings"][q]
        binding = load_prepared_binding_reference(
            reference, scene_input_identity=input_identity(item, case.new_path))
        assert binding.stats["generation"] == before[q]
        assert binding.stats["generation"]["cache_eligible"] is False
        assert binding.prepared_measurements == {}
        assert binding.objects[0].binding_validation.status == "validated"
        assert binding.stats["actual_model_requests"] == 0
        assert reference["request_identity"]["scene_audit_qa_ids"] == ["qa-1", "qa-2"]
        assert reference["key"] != case.preparation["bindings"][q]["key"]
        assert binding.stats["generation_reuse"]["source_key"] == case.preparation["bindings"][q]["key"]
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["stage"] for line in lines] == [
        "questionbindingstarted", "questionbindingfinished",
        "questionbindingstarted", "questionbindingfinished",
    ]
    assert all(line["generation_source"] == "reused" and line["elapsed_s"] >= 0
               for line in lines if line["stage"] == "questionbindingfinished")


@pytest.mark.parametrize("change", [
    "depth", "intrinsics", "confidence", "point_map", "gate", "question",
    "pixels", "source_indices", "timestamps", "parameters", "budget", "seed", "model",
])
def test_unproven_changes_fall_back_without_client_creation(case, change):
    item = case.new_items[0]
    if change in {"depth", "intrinsics", "confidence", "point_map"}:
        name = {"depth": "depth_maps", "confidence": "depth_conf"}.get(change, change)
        value = np.load(getattr(case.new_art, name))
        value.flat[0] += 1
        np.save(getattr(case.new_art, name), value)
    elif change == "gate":
        case.new_art = case.new_art.model_copy(update={
            "quality": case.new_art.quality.model_copy(update={"main_gate_passed": False}),
        })
        _write(case.new_path, case.new_art.model_dump(mode="json"))
    elif change == "question":
        item.episode = item.episode.model_copy(update={"question": "A different question?"})
    elif change == "pixels":
        item.pixels = [p.copy() for p in item.pixels]
        item.pixels[0][0, 0, 0] += 1
        item.episode = item.episode.model_copy(update={
            "frame_set": item.episode.frame_set.model_copy(update={"pixel_sha256": []}),
        })
    elif change in {"source_indices", "timestamps"}:
        field = "source_frame_indices" if change == "source_indices" else "timestamps"
        values = list(getattr(item.episode.frame_set, field))
        values[0] += 0.1 if field == "timestamps" else 1
        item.episode = item.episode.model_copy(update={
            "frame_set": item.episode.frame_set.model_copy(update={field: values}),
        })
        case.new_art = case.new_art.model_copy(update={field: values})
        _write(case.new_path, case.new_art.model_dump(mode="json"))
    else:
        field = {"parameters": "semantic_relation_batch_size", "budget": "max_tokens",
                 "model": "online_model"}.get(change, change)
        value = "different-model" if change == "model" else getattr(case.cfg, field) + 1
        setattr(case.cfg, field, value)
    assert _derive(case) is None


def _export(case, root):
    package = {"schema_version": PACKAGE_SCHEMA_VERSION, "bindings": {}}
    export_root = root.parent / "old-export-root"
    export_root.mkdir()
    artifact_ref = "reuse/base/geometry/artifact.json"
    target = root / artifact_ref
    target.parent.mkdir(parents=True)
    shutil.copyfile(case.path, target)
    dependency_refs = {}
    for name, source in artifact_dependency_refs(case.art).items():
        ref = "reuse/base/geometry/" + Path(source).name
        dependency_refs[name] = ref
        shutil.copyfile(source, root / ref)
    shutil.copytree(root / "reuse/base/geometry", export_root / "geometry")
    shutil.copytree(case.store.root, root / "reuse/base/store")
    for qa_id, reference in case.preparation["bindings"].items():
        original = case.store.create_migration_receipt(
            reference["key"], scene_input_identity=case.identities[qa_id],
            destination_root=export_root, scene_artifact_ref="geometry/artifact.json",
            dependency_refs={name: "geometry/" + Path(ref).name
                             for name, ref in dependency_refs.items()},
        )
        package["bindings"][qa_id] = {"base": {
            "source_reference": copy.deepcopy(reference),
            "source_input_identity": copy.deepcopy(case.identities[qa_id]),
            "source_migration_receipt": original,
            "store_root_ref": "reuse/base/store", "scene_artifact_ref": artifact_ref,
            "dependency_refs": dependency_refs,
        }}
    shutil.rmtree(export_root)
    shutil.rmtree(case.root)
    return package


def test_export_relocation_recomputes_old_logical_identity_without_original_paths(case, tmp_path):
    input_root = tmp_path / "input-root"
    package = _export(case, input_root)
    factory = make_generation_reuse_factory(case.new_items, case.cfg, input_root, package)
    result = _prepare(case, tmp_path / "export-store", reuse_factory=factory)
    assert result["generation_reused_qa_ids"] == ["qa-1", "qa-2"]
    binding = load_prepared_binding_reference(
        result["bindings"]["qa-1"],
        scene_input_identity=input_identity(case.new_items[0], case.new_path))
    assert binding.stats["generation_reuse"]["source_proof"]["kind"] == "exported-binding"
    assert not case.root.exists()


@pytest.mark.parametrize("tamper", ["input", "reference", "receipt", "artifact", "dependency", "payload"])
def test_export_tamper_is_strict_not_perception_fallback(case, tmp_path, tamper):
    root = tmp_path / "input-root"
    package = _export(case, root)
    record = package["bindings"]["qa-1"]["base"]
    if tamper == "input":
        record["source_input_identity"]["episode"]["question"] = "tampered"
    elif tamper == "reference":
        record["source_reference"]["manifest_sha256"] = "0" * 64
    elif tamper == "receipt":
        record["source_migration_receipt"]["binding_identity"]["manifest_sha256"] = "0" * 64
        receipt = record["source_migration_receipt"]
        receipt["receipt_sha256"] = content_sha256({
            k: v for k, v in receipt.items() if k != "receipt_sha256"
        })
    elif tamper in {"artifact", "dependency"}:
        ref = (record["scene_artifact_ref"] if tamper == "artifact"
               else record["dependency_refs"]["depth_maps"])
        (root / ref).write_bytes(b"tampered")
    else:
        store = PreparedBindingStore(root / record["store_root_ref"])
        path = store.entry_path(record["source_reference"]["key"]) / "payload/binding.json"
        path.write_bytes(b"tampered")
    factory = make_generation_reuse_factory(case.new_items, case.cfg, root, package)
    with pytest.raises(PreparedBindingStoreError):
        _derive(case, factory)


def test_export_metric_requires_separate_old_variant(case, tmp_path):
    root = tmp_path / "input-root"
    package = _export(case, root)
    _metric(case)
    factory = make_generation_reuse_factory(case.new_items, case.cfg, root, package)
    assert _derive(case, factory) is None


@pytest.mark.parametrize("status", ["parse_unavailable", "binding_unavailable"])
def test_parse_or_unanchored_failure_is_not_reused(case, status):
    binding = _source_binding(case)
    binding.objects = []
    binding.materialized = False
    binding.stats.update(status=status)
    if status == "parse_unavailable":
        binding.stats["object_parse"] = {"status": "unavailable"}
    else:
        binding.stats["binding_error"] = "propagation unavailable"
    _replace_source(case, binding)
    assert _derive(case) is None


@pytest.mark.parametrize("status,passed,task,materialized", [
    ("m4_unavailable", False, "object_rel_distance", False),
    ("objects_not_required", True, "room_size_estimation", True),
])
def test_no_generation_outcomes_require_real_gate_and_task(case, status, passed, task, materialized):
    case.art = case.art.model_copy(update={"quality": case.art.quality.model_copy(
        update={"main_gate_passed": passed})})
    case.new_art = case.new_art.model_copy(update={"quality": case.new_art.quality.model_copy(
        update={"main_gate_passed": passed})})
    _write(case.path, case.art.model_dump(mode="json"))
    _write(case.new_path, case.new_art.model_dump(mode="json"))
    for items in (case.items, case.new_items):
        items[0].episode = items[0].episode.model_copy(update={"question_type": task})
    identity = input_identity(case.items[0], case.path)
    request = prepared_binding_request_identity(
        input_identity=identity, cfg=case.cfg, scene_audit_qa_ids=["qa-1"])
    case.identities["qa-1"] = identity
    case.preparation["bindings"]["qa-1"] = case.store.publish(
        request_identity=request, scene_artifact=case.path, scene_input_identity=identity,
        binding=PreparedObjectBinding([], {"status": status}, [], materialized),
    )
    derived = _derive(case)
    assert derived is not None and not derived.stats.get("generation")
    assert derived.stats["status"] == status and derived.materialized == materialized
    case.new_art = case.new_art.model_copy(update={"quality": case.new_art.quality.model_copy(
        update={"main_gate_passed": not passed})})
    _write(case.new_path, case.new_art.model_dump(mode="json"))
    assert _derive(case) is None


def test_common_helper_rejects_generation_hash_tamper(case):
    item = case.new_items[0]
    binding = _source_binding(case)
    binding.stats["generation"]["identity"]["question"]["question"] = "tampered"
    old = case.preparation["bindings"]["qa-1"]
    identity = input_identity(item, case.new_path)
    request = prepared_binding_request_identity(
        input_identity=identity, cfg=case.cfg, scene_audit_qa_ids=["qa-1", "qa-2"])
    with pytest.raises(PreparedBindingStoreError, match="generation content"):
        reuse_verified_binding(
            binding, case.identities["qa-1"], old["request_identity"],
            case.store.manifest(old["key"]), identity, request, item, case.new_art, case.cfg,
            {"source_artifact": case.art, "source_input_recomputed": case.identities["qa-1"]},
        )


def _reject_outage(stats):
    if stats.get("binding_error") == "HTTPError: HTTP Error 503":
        raise RuntimeError("infrastructure outage")


@pytest.mark.parametrize("branch", ["new", "reused", "draft", "sealed"])
def test_outcome_validation_prevents_outage_publication_on_every_branch(
        case, tmp_path, monkeypatch, branch):
    root = tmp_path / "new-store"
    outage = PreparedObjectBinding([], {"binding_error": "HTTPError: HTTP Error 503"}, [], False)
    if branch == "new":
        monkeypatch.setattr(runner, "_bind_objects_best_effort", lambda *_a, **_k: (
            outage.objects, outage.stats, outage.notes, outage.materialized))
    if branch in {"draft", "sealed"}:
        store = PreparedBindingStore(root / ".drafts" if branch == "draft" else root)
        for item in case.new_items:
            identity = input_identity(item, case.new_path)
            request = prepared_binding_request_identity(
                input_identity=identity, cfg=case.cfg, scene_audit_qa_ids=["qa-1", "qa-2"])
            store.publish(request_identity=request, scene_artifact=case.new_path,
                          scene_input_identity=identity, binding=outage)
    reuse = (lambda *_args: copy.deepcopy(outage)) if branch == "reused" else None
    with pytest.raises(RuntimeError, match="infrastructure outage"):
        prep.prepare_shared_bindings_v11(
            case.new_items, artifact_paths={i.episode.qa_id: case.new_path for i in case.new_items},
            base_cfg=case.cfg, store_root=root, min_free_gib=0,
            client_factory=(lambda _q: None) if branch == "new" else _never,
            reuse_factory=reuse, outcome_validator=_reject_outage,
        )
    if branch in {"new", "reused"}:
        assert not list(root.rglob("manifest.json"))
    elif branch == "draft":
        assert not (root / "entries").exists()


def test_reused_draft_resume_and_sealed_cache_remain_distinct(case, tmp_path, monkeypatch, capsys):
    root = tmp_path / "resume-store"
    original = prep.audit_semantic_mask_conflicts

    def fail_audit(*_a, **_k):
        raise RuntimeError("interrupted audit")

    monkeypatch.setattr(prep, "audit_semantic_mask_conflicts", fail_audit)
    with pytest.raises(RuntimeError, match="interrupted audit"):
        _prepare(case, root, reuse_factory=_local(case))
    assert len(list((root / ".drafts").rglob("manifest.json"))) == 2
    monkeypatch.setattr(prep, "audit_semantic_mask_conflicts", original)
    resumed = _prepare(case, root, reuse_factory=_never)
    assert resumed["generation_reused_qa_ids"] == ["qa-1", "qa-2"]
    assert resumed["cache_hit_qa_ids"] == []
    sealed = _prepare(case, root, reuse_factory=_never)
    assert sealed["cache_hit_qa_ids"] == ["qa-1", "qa-2"]
    assert sealed["generation_reused_qa_ids"] == []
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    sources = [line["generation_source"] for line in lines
               if line["stage"] == "questionbindingfinished"]
    assert sources == ["reused", "reused", "draft", "draft", "sealed", "sealed"]


def test_outcome_revalidated_after_scene_audit_before_any_seal(case, tmp_path):
    def validator(stats):
        if "semantic_conflict_audits" in stats:
            raise RuntimeError("audit outcome rejected")

    root = tmp_path / "audit-rejection"
    with pytest.raises(RuntimeError, match="audit outcome rejected"):
        _prepare(case, root, reuse_factory=_local(case), outcome_validator=validator)
    assert not (root / "entries").exists()


def test_reuse_preserves_flags_until_audit_clears_only_audit_managed_flags(case, tmp_path):
    binding = _source_binding(case)
    original_generation = copy.deepcopy(binding.stats["generation"])
    flags = [
        {"code": "semantic_label_mask_conflict", "contract_version": "semantic-mask-conflict-v2"},
        {"code": "manual_binding_block"},
    ]
    binding.objects[0] = binding.objects[0].model_copy(update={"binding_flags": flags})
    reference = case.preparation["bindings"]["qa-1"]
    flagged_store = PreparedBindingStore(tmp_path / "flagged-source")
    case.preparation["bindings"]["qa-1"] = flagged_store.publish(
        request_identity=reference["request_identity"], scene_artifact=case.path,
        scene_input_identity=case.identities["qa-1"], binding=binding,
    )
    derived = _derive(case)
    assert derived.objects[0].binding_flags == flags
    assert derived.objects[0].binding_validation.status == "unavailable"
    result = _prepare(case, tmp_path / "flagged-destination", reuse_factory=_local(case))
    sealed = load_prepared_binding_reference(
        result["bindings"]["qa-1"],
        scene_input_identity=input_identity(case.new_items[0], case.new_path),
    )
    assert sealed.objects[0].binding_flags == [{"code": "manual_binding_block"}]
    assert sealed.objects[0].binding_validation.status == "unavailable"
    assert sealed.stats["generation"] == original_generation


def test_completed_empty_object_outcome_reuses_real_generation(case):
    binding = _source_binding(case)
    binding.objects = []
    binding.stats["object_parse"] = {"status": "parsed"}
    _replace_source(case, binding)
    derived = _derive(case)
    assert derived is not None and derived.materialized and not derived.objects
    assert derived.stats["generation"] == binding.stats["generation"]
    assert derived.stats["actual_model_requests"] == 0


def test_disk_reserve_and_reuse_miss_keep_existing_preparation_behavior(case, tmp_path, monkeypatch):
    with pytest.raises(prep.V11BindingPreparationError, match="runtime reserve"):
        prep.prepare_shared_bindings_v11(
            case.new_items, artifact_paths={i.episode.qa_id: case.new_path for i in case.new_items},
            base_cfg=case.cfg, store_root=tmp_path / "reserve", min_free_gib=1e12,
            reuse_factory=_never, client_factory=_never,
        )
    calls = []

    def bind(_art, _pixels, _scene, episode, _cfg, llm, **_kwargs):
        assert llm == "client"
        calls.append(episode.qa_id)
        return [], {"status": "m4_unavailable"}, [], False

    monkeypatch.setattr(runner, "_bind_objects_best_effort", bind)
    result = prep.prepare_shared_bindings_v11(
        case.new_items, artifact_paths={i.episode.qa_id: case.new_path for i in case.new_items},
        base_cfg=case.cfg, store_root=tmp_path / "miss", min_free_gib=0,
        reuse_factory=lambda *_args: None, client_factory=lambda _q: "client",
    )
    assert calls == ["qa-1", "qa-2"]
    assert result["generation_reused_qa_ids"] == result["cache_hit_qa_ids"] == []
