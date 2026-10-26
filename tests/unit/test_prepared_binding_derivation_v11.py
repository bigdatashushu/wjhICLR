"""Real immutable stores with synthetic category evidence; no models/network."""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.adapters.episode_source import EpisodeItem, _episode_from_row
from skill3d.adapters.frame_set import build_frame_set, with_pixel_identity
from skill3d.evaluation.skill_ablation_v11 import input_identity
from skill3d.evolution import prepared_binding_derivation_v11 as derivation
from skill3d.infra.m5_identity import content_sha256, generation_identity
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig, PreparedObjectBinding
from skill3d.schemas import ObjectRecord, ReconstructionArtifact
from skill3d.segmentation.intern_binary_review import (
    PROTOCOL_SHA256, RENDERING, admission_identity,
)
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore, PreparedBindingStoreError, load_prepared_binding_reference,
    prepared_binding_request_identity,
)
from skill3d.segmentation.question_binding import (
    M5_BINDING_VERSION, question_input, validate_prepared_binding,
)
from skill3d.storage.relocation import artifact_dependency_refs

# Recorded git 3588bc8 SOURCE_REFS, not an injectable runtime allowlist.
_LEGACY_SOURCES = {
    "coords.py": (9513, "caa0103e3bb8c53e8b1997cff43ea35d6eb9f21be19cbb86b7e05c657b15b7d9"),
    "infra/m5_identity.py": (16468, "d54b21f5ed37819a68a944b9f39e606a94e4270824695a8a314fc81917e2bf4a"),
    "infra/version_lock.py": (5325, "4b5ed7d294f0717569bfb86055531b66e1af6a5788f0be6119becd747c78f963"),
    "reconstruction/image_grid.py": (3224, "960b592c8f36267bd2bdf0af73a70a29a9c27d66d42cd596d5f9cb36c970edc3"),
    "reconstruction_gate/confidence_map.py": (2931, "24aff275b9455ba9f5dc2a929af5707c905e3464f76841acba2aff05b88707b6"),
    "schemas/reconstruction.py": (22442, "7090d71322ec60304c71433edc358482ffa05206c39de4b1cb5d644d35048453"),
    "segmentation/binding_validation.py": (8650, "d99629ccf100edc4646bdb7b0ed4bb81b7968f9c8a36f0809ed192de9e9610a9"),
    "segmentation/groundingdino_startup.py": (4812, "13dc513182317a991a5d0910aa135c0d1a71336f447dcdcbb32a33a08a89dc80"),
    "segmentation/intern_binary_review.py": (19942, "b5781b16f6fa412835eac9ff65c49dbe01e7656805743a6a15970c2fd84f072c"),
    "segmentation/intern_decision_client.py": (11947, "b2c7efa451d2b3606ec78112035ad5eb34e9ae0800839861f12fc8aaac240a0a"),
    "segmentation/intern_decision_protocol.py": (4848, "e55a2744b485d2622f7edf257ca7b6f834c4bd932e20ed467769889de97f2a2e"),
    "segmentation/intern_review_pool.py": (3918, "5660c3f99b4199747beccb86ca3c881f3a21e799bf1012bb203837ae758cf58e"),
    "segmentation/mask_semantic_review.py": (12637, "dc1cbc79c5cb92853508d331ab1ba9a921e413b80880c9014f9b1f063944860f"),
    "segmentation/open_vocab_detector.py": (10716, "8dd943a15ce7d4fc0fa5f05e41e8e8030ab12d35fc151f9a52eeefc443099e7d"),
    "segmentation/question_binding.py": (52991, "8d163a595a0bb584b3a907c4fd51d8e6fc8b3b74962a9a335c04ce8bfb5c5bf6"),
    "segmentation/sam2_tracker.py": (40757, "fcfbdc4f7fed3e64d7da6fa79313d90e5eebca8962fc0006a229ce1cb62ca3a3"),
    "synthesis/bailian_client.py": (9311, "00f8dee9450a844d9b17f09122bf429f481d1e9222ce511273a129c5830ab143"),
    "synthesis/vllm_client.py": None,
    "tools/category_match.py": (10190, "76032ba2a8975f9c6d2b209aedd2233528b665de4d5f1794630b15faf6112de7"),
}


def _legacy_sources():
    return {name: {"size": value[0], "sha256": value[1]} if value else None
            for name, value in _LEGACY_SOURCES.items()}


def _seal(value, field="sha256"):
    value[field] = content_sha256({k: v for k, v in value.items() if k != field})
    return value


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def _never(*_args, **_kwargs):
    raise AssertionError("unexpected model/network/solver operation")


def _answer(yes, calibration):
    return {
        "model": "Intern-Decision-4B", "backend": "hf", "calibration": calibration,
        "answers": {"target_match": {
            "type": "noul", "decision": "yes", "noul": yes, "confidence": yes,
            "probabilities": {"yes": yes, "no": 1 - yes},
        }},
    }


def _evidence(cid, category, model):
    request_sha = content_sha256({"synthetic_binary_request": cid})
    admission = admission_identity("category-only")
    identity = {
        "candidate_id": cid, "frame_slot": 0, "bbox": [0, 0, 4, 3],
        "model": model, "prompt_version": "P1", "target_category": category,
        "rendering": RENDERING, "protocol_sha256": PROTOCOL_SHA256,
        "images": [{"sha256": content_sha256(["image", i]), "size": 20} for i in range(3)],
        "admission": admission, "yes_threshold": .77, "quality_manifest_sha256": "",
    }
    body = {
        "request_sha256": request_sha, "identity_sha256": content_sha256(model),
        "result": _answer(.85, model["calibration"]),
        "uncalibrated_result": _answer(.9, {"temperature": 1}),
        "vision_evidence": {
            "image_count": 3, "image_token_count": 100, "pixel_values_shape": [3, 4, 4],
            "image_grid_thw": [[1, 2, 2]] * 3, "decision_logit_positions": [42],
        },
    }
    binary = {
        "p_yes": .85, "p_no": 1 - .85, "raw_p_yes": .9, "raw_p_no": 1 - .9,
        "decision": "yes", "target_category": category, "prompt_version": "P1",
        "protocol_sha256": PROTOCOL_SHA256, "threshold": .77, "semantic_pass": True,
        "admission_policy": "category-only",
    }
    visual = {
        "candidate_id": cid, "status": "observed", "provider": "intern-decision-binary-local",
        "request_identity": identity, "request_sha256": request_sha,
        "raw_response": body, "model_identity_sha256": content_sha256(model),
        "observation": {}, "observation_sha256": content_sha256({}),
        "admission": admission, "binary_review": binary,
        "binary_review_sha256": content_sha256(binary), "frozen_quality": {},
    }
    row = {
        "candidate_id": cid, "category": category, "frame_slot": 0, "bbox": [0, 0, 4, 3],
        "semantic_provider": visual["provider"], "semantic_relation": "match",
        "semantic_relation_request_sha256": request_sha,
        "semantic_binary_response_sha256": content_sha256(body),
        **{saved: copy.deepcopy(visual[original])
           for saved, original in derivation._VISUAL_FIELDS.items()},
    }
    receipt = {
        "candidate_id": cid, "category": category, "frame_slot": 0, "bbox": [0, 0, 4, 3],
        "semantic_visual": visual,
        "semantic_relation": {"status": "reviewed", "relation": "match",
                              "request_sha256": request_sha},
    }
    return row, receipt


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_bind_objects_best_effort", _never)
    monkeypatch.setattr(runner, "_make_online_client", _never)
    monkeypatch.setattr(runner, "_prepare_m4_quality", _never)
    from skill3d.segmentation import intern_binary_review, open_vocab_detector, question_binding
    monkeypatch.setattr(intern_binary_review.InternBinaryClient, "_call", _never)
    monkeypatch.setattr(open_vocab_detector, "detect", _never)
    monkeypatch.setattr(question_binding, "inventory_generation_identity", _never)
    root = tmp_path / "source"
    root.mkdir()
    pixels = [np.full((3, 4, 3), i, dtype=np.uint8) for i in range(32)]
    items = []
    for qa, category in (("qa-1", "door"), ("qa-2", "chair")):
        fs = with_pixel_identity(build_frame_set(
            32, episode_id=qa, dataset_id="scannet", video_id="scene", scene_name="scene",
            fps=24), pixels)
        ep = _episode_from_row({
            "qa_id": qa, "scene_name": "scene", "dataset": "scannet",
            "question": f"How many {category}?", "question_type": "object_counting",
            "options": ["A. one", "B. two"], "ground_truth": "",
        }, "induction", pixels, 32, frame_set=fs)
        items.append(EpisodeItem(ep, pixels, source="vsi_bench", video_path="/old/video.mp4"))
    values = {
        "depth_maps": np.ones((32, 3, 4), np.float32),
        "c2w_list": np.repeat(np.eye(4, dtype=np.float32)[None], 32, axis=0),
        "intrinsics": np.repeat(np.eye(3, dtype=np.float32)[None], 32, axis=0),
        "depth_conf": np.ones((32, 3, 4), np.float32),
        "point_map": np.ones((32, 3, 4, 3), np.float32),
    }
    refs = {}
    for name, value in values.items():
        path = root / (name + ".npy")
        np.save(path, value, allow_pickle=False)
        refs[name] = str(path)
    refs["point_conf"] = refs["depth_conf"]
    art = ReconstructionArtifact(
        artifact_id="synthetic-vggt", artifact_version="base", scene_name="scene",
        frame_ids=items[0].episode.frame_set.frame_ids,
        frame_set_hash=items[0].episode.frame_set.frame_set_hash,
        source_frame_indices=items[0].episode.frame_set.source_frame_indices,
        timestamps=items[0].episode.frame_set.timestamps,
        reconstruction_input_identity=items[0].episode.frame_set.cache_identity(),
        confidence={"per_point_confidence": refs["depth_conf"]}, **refs,
    )
    artifact_path = root / "artifact.json"
    _write(artifact_path, art.model_dump(mode="json"))
    semantic_model = {"model": "synthetic-category-fixture", "calibration": {"temperature": 2},
                      "weights_sha256": content_sha256("test-only-weights")}
    cfg = OnlineRunConfig(seed=0, semantic_review_config={
        "provider": "intern-decision", "mode": "binary", "prompt_version": "P1",
        "admission_policy": "category-only", "yes_threshold": .77,
        "endpoints": [{"endpoint": "http://127.0.0.1:27345",
                       "expected_identity_sha256": content_sha256(semantic_model)}],
    })
    store = PreparedBindingStore(root / "store")
    sources = {}
    for item in items:
        qa = item.episode.qa_id
        category = "door" if qa == "qa-1" else "chair"
        identity = input_identity(item, artifact_path)
        request = prepared_binding_request_identity(
            input_identity=identity, cfg=cfg, scene_audit_qa_ids=["qa-1", "qa-2"])
        request["binding_version"] = derivation.LEGACY_BINDING_VERSION
        request["implementation_sources"] = _legacy_sources()
        request["store_source"] = copy.deepcopy(derivation.LEGACY_STORE_SOURCE)
        request["parameters"].pop("binding_validation_version")
        request["parameters"]["semantic_visual_version"] = "m5-mask-observation-qwen38-v1"
        _seal(request)
        generation = generation_identity(
            scene_name="scene", frame_set_hash=art.frame_set_hash, frames=pixels,
            depth_maps=values["depth_maps"], c2w_list=values["c2w_list"],
            intrinsics=values["intrinsics"], point_conf=values["depth_conf"],
            depth_conf=values["depth_conf"], point_map=values["point_map"],
            frame_provenance=runner._episode_frame_provenance(item.episode, pixels),
            model={
                "enabled": True, "provider": cfg.online_model_provider,
                "model": cfg.online_model, "model_revision": cfg.online_model_revision,
                "endpoint_sha256": request["model"]["endpoint_sha256"], "seed": 0,
                "temperature": 0, "max_tokens": cfg.max_tokens, "cache_eligible": False,
            },
            detector=None, sam2={"synthetic": True}, sources=_legacy_sources(),
            parameters=request["parameters"], question=item.episode.question,
            question_type=item.episode.question_type, options=item.episode.options,
            reconstruction_identity=art.reconstruction_input_identity,
        )
        cid = "candidate-" + qa
        row, receipt = _evidence(cid, category, semantic_model)
        mask, points = root / (qa + "-mask.npy"), root / (qa + "-points.npy")
        np.save(mask, np.ones((32, 3, 4), np.uint8))
        np.save(points, np.ones((4, 3), np.float32))
        obj = ObjectRecord(
            obj_id=qa + "-object", category_name=category, source_label=category,
            grounding_status="question_driven", mask_per_frame=str(mask),
            pointcloud_world=str(points), centroid_world=[1, 1, 1], bbox=[1] * 6,
            visible_frames=list(range(32)), mask_support_ratio=1,
            track_id="track-" + qa, candidate_ids=[cid],
            question_roles=[{"role": "target", "source_text": category}],
            binding_provenance=[row],
            binding_validation={
                "version": "m5-binding-validation-v2-semantic-review", "status": "validated",
                "generation_sha256": generation["sha256"],
            },
        )
        binding = PreparedObjectBinding(
            [obj], {
                "version": derivation.LEGACY_BINDING_VERSION, "generation": generation,
                "status": "prepared", "object_parse": {"status": "parsed"},
                "candidate_receipts": [receipt], "actual_model_requests": 9,
                "actual_detector_requests": 32, "actual_sam2_tracks": 2,
                "semantic_conflict_audits": [{"old_group": "must not survive"}],
                "grounding": {"requirements": [{"name": category, "scan_complete": True}]},
            }, ["original notes"], True, {"old_measurement": 7})
        reference = store.publish(
            request_identity=request, scene_artifact=artifact_path,
            scene_input_identity=identity, binding=binding)
        sources[qa] = {
            "reference": reference, "identity": identity, "artifact_path": artifact_path,
            "prepared": load_prepared_binding_reference(reference, scene_input_identity=identity),
            "artifact": art,
        }
    new_items = [EpisodeItem(item.episode.model_copy(update={"split": "inner_validation"}),
                            pixels, source=item.source, video_path=item.video_path)
                 for item in items]
    identities = {item.episode.qa_id: input_identity(item, artifact_path) for item in new_items}
    requests = {
        qa: prepared_binding_request_identity(
            input_identity=identity, cfg=cfg, scene_audit_qa_ids=["qa-1", "qa-2"])
        for qa, identity in identities.items()
    }
    context = derivation.PreparedBindingDerivationContext(
        sources=sources, items=new_items, target_inputs=identities, target_requests=requests,
        target_artifact_paths={qa: artifact_path for qa in identities}, cfg=cfg)
    return SimpleNamespace(context=context, store=store, art=art, path=artifact_path,
                           arrays=values, old_items=items, root=root)


def _validate(case, binding, *, context=True, **overrides):
    item = case.context.items[0]
    values = {
        "frames": item.pixels, "depth": case.arrays["depth_maps"],
        "c2w": case.arrays["c2w_list"], "intrinsics": case.arrays["intrinsics"],
        "point_conf": case.arrays["depth_conf"], "depth_conf": case.arrays["depth_conf"],
        "point_map": case.arrays["point_map"],
        "inputs": question_input(item.episode.question, item.episode.question_type,
                                 item.episode.options),
        "scene_identity": case.art.reconstruction_input_identity, "seed": 0,
        "parameters": case.context.target_requests["qa-1"]["parameters"],
    }
    if context:
        values["binding_derivation_context"] = case.context
    values.update(overrides)
    return validate_prepared_binding(binding, **values)


def _replace_source(case, mutate, *, request_mutate=None):
    source = case.context.sources["qa-1"]
    binding = copy.deepcopy(source["prepared"])
    mutate(binding)
    request = copy.deepcopy(source["reference"]["request_identity"])
    if request_mutate:
        request_mutate(request)
        _seal(request)
    store = PreparedBindingStore(case.root / "replacement-store")
    reference = store.publish(
        request_identity=request, scene_artifact=case.path,
        scene_input_identity=source["identity"], binding=binding)
    source.update(
        reference=reference,
        prepared=load_prepared_binding_reference(reference, scene_input_identity=source["identity"]))


def test_whitelist_is_full_exact_reviewed_snapshot():
    assert content_sha256(_legacy_sources()) == derivation.LEGACY_SOURCES_SHA256


def test_scene_derivation_preserves_generation_and_reaudits(case, monkeypatch):
    before = copy.deepcopy(case.context.sources)
    calls = []
    audit = derivation.audit_semantic_mask_conflicts

    def track(group, **kwargs):
        calls.append([qa for qa, _ in group])
        return audit(group, **kwargs)

    monkeypatch.setattr(derivation, "audit_semantic_mask_conflicts", track)
    bindings = derivation.derive_prepared_binding_scene_v11(case.context)
    assert calls == [["qa-1", "qa-2"]]
    for qa, binding in bindings.items():
        original = before[qa]["prepared"]
        assert binding is not original
        assert binding.stats["generation"] == original.stats["generation"]
        assert binding.stats["generation"]["cache_eligible"] is False
        assert binding.stats["actual_model_requests"] == 9
        assert binding.stats["actual_detector_requests"] == 32
        assert binding.stats["actual_sam2_tracks"] == 2
        assert binding.prepared_measurements == {}
        assert binding.stats["version"] == M5_BINDING_VERSION
        assert binding.notes == original.notes
        assert binding.stats["candidate_receipts"] == original.stats["candidate_receipts"]
        assert binding.objects[0].binding_provenance[0] == {
            **original.objects[0].binding_provenance[0], "semantic_policy": "category-only"}
        assert binding.objects[0].binding_flags[0]["code"] == "semantic_label_mask_conflict"
        assert binding.objects[0].binding_validation.status == "unavailable"
        assert "binding_semantics" in binding.objects[0].binding_validation.files
        assert binding.stats["grounding"]["requirements"][0]["usable_object_ids"] == []
        receipt = binding.stats["binding_derivation"]
        assert receipt["generation_resealed"] is False
        assert receipt["quality_assessed"] is False
        assert receipt["new_model_requests"] == receipt["new_detector_requests"] == 0
        assert receipt["source"]["reference"] == before[qa]["reference"]
        assert receipt["source"]["payload_sha256"] == case.store.manifest(
            before[qa]["reference"]["key"])["payload"]["sha256"]
    assert case.context.sources["qa-1"]["prepared"].stats == before["qa-1"]["prepared"].stats
    _validate(case, bindings["qa-1"])


def test_publish_reload_and_consume_is_path_independent(case, tmp_path):
    references = derivation.publish_derived_binding_scene_v11(
        case.context, store_root=tmp_path / "derived-store")
    reference = references["qa-1"]
    assert reference["request_identity"] == case.context.target_requests["qa-1"]
    assert reference["key"] != case.context.sources["qa-1"]["reference"]["key"]
    binding = load_prepared_binding_reference(
        reference, scene_input_identity=case.context.target_inputs["qa-1"])
    assert Path(binding.objects[0].mask_per_frame).is_relative_to(tmp_path / "derived-store")
    _validate(case, binding)
    with pytest.raises(PreparedBindingStoreError, match="must be new"):
        derivation.publish_derived_binding_scene_v11(case.context, store_root=case.store.root)


def test_non_audit_flags_and_suspects_are_not_promoted(case):
    def mutate(binding):
        binding.objects[0].binding_flags = [
            {"code": "manual_block"},
            {"code": "semantic_label_mask_conflict", "contract_version": "semantic-mask-conflict-v2"},
        ]
        binding.objects[0].duplicate_suspect = True
    _replace_source(case, mutate)
    ctx = replace(case.context, items=case.context.items[:1],
                  sources={"qa-1": case.context.sources["qa-1"]},
                  target_inputs={"qa-1": case.context.target_inputs["qa-1"]},
                  target_requests={"qa-1": prepared_binding_request_identity(
                      input_identity=case.context.target_inputs["qa-1"], cfg=case.context.cfg,
                      scene_audit_qa_ids=["qa-1"])},
                  target_artifact_paths={"qa-1": case.path})
    binding = derivation.derive_prepared_binding_scene_v11(ctx)["qa-1"]
    assert binding.objects[0].binding_flags == [{"code": "manual_block"}]
    assert binding.objects[0].duplicate_suspect is True
    assert binding.objects[0].binding_validation.status == "unavailable"
    assert binding.stats["semantic_conflict_audits"] == []


@pytest.mark.parametrize("field", [
    "semantic_binary_response", "semantic_binary_response_sha256", "semantic_request_identity",
    "semantic_model_identity_sha256", "semantic_admission", "semantic_binary_review",
    "semantic_binary_review_sha256", "semantic_observation_sha256",
    "semantic_visual_request_sha256", "semantic_relation_request_sha256",
])
def test_missing_provenance_fails_closed_even_if_old_validation_passed(case, field):
    _replace_source(case, lambda binding: binding.objects[0].binding_provenance[0].pop(field))
    with pytest.raises(PreparedBindingStoreError):
        derivation.derive_prepared_binding_scene_v11(case.context)


@pytest.mark.parametrize("field", ["raw_response", "request_identity", "model_identity_sha256", "admission"])
def test_missing_exact_original_candidate_receipt_fails_closed(case, field):
    _replace_source(case, lambda binding: binding.stats["candidate_receipts"][0]["semantic_visual"].pop(field))
    with pytest.raises(PreparedBindingStoreError, match="missing or changed"):
        derivation.derive_prepared_binding_scene_v11(case.context)


def test_no_statistics_only_category_admission(case):
    _replace_source(case, lambda binding: binding.stats.pop("candidate_receipts"))
    with pytest.raises(PreparedBindingStoreError, match="original semantic receipt"):
        derivation.derive_prepared_binding_scene_v11(case.context)


@pytest.mark.parametrize("tamper", ["policy", "quality", "model", "probability", "binary_hash"])
def test_semantic_tampering_is_rejected_even_after_payload_reseal(case, tamper):
    def mutate(binding):
        row = binding.objects[0].binding_provenance[0]
        if tamper == "policy":
            row["semantic_policy"] = "frozen-quality"
        elif tamper == "quality":
            row["semantic_observation"] = {"visibility": "clear"}
        elif tamper == "model":
            row["semantic_request_identity"]["model"]["weights_sha256"] = "0" * 64
        elif tamper == "probability":
            row["semantic_binary_response"]["result"]["answers"]["target_match"]["probabilities"]["yes"] = .99
        else:
            row["semantic_binary_review_sha256"] = "0" * 64
    _replace_source(case, mutate)
    with pytest.raises(PreparedBindingStoreError):
        derivation.derive_prepared_binding_scene_v11(case.context)


def test_different_source_snapshot_cannot_be_whitelisted_by_resealing(case):
    def mutate(binding):
        identity = binding.stats["generation"]["identity"]
        identity["sources"]["segmentation/question_binding.py"]["sha256"] = "0" * 64
        binding.stats["generation"]["sha256"] = content_sha256(identity)

    def request_mutate(request):
        request["implementation_sources"]["segmentation/question_binding.py"]["sha256"] = "0" * 64
    _replace_source(case, mutate, request_mutate=request_mutate)
    with pytest.raises(PreparedBindingStoreError, match="exact reviewed"):
        derivation.derive_prepared_binding_scene_v11(case.context)


@pytest.mark.parametrize("tamper", [
    "parameters", "seed", "model", "audit_group", "geometry", "pixels", "question",
])
def test_target_changes_abort_instead_of_perception_fallback(case, tamper):
    ctx = case.context
    if tamper in {"parameters", "seed", "model"}:
        cfg = copy.deepcopy(ctx.cfg)
        if tamper == "parameters":
            cfg.semantic_relation_batch_size += 1
        elif tamper == "seed":
            cfg.seed = 137
        else:
            cfg.online_model = "different-model"
        ctx = replace(ctx, cfg=cfg, target_requests={
            qa: prepared_binding_request_identity(
                input_identity=identity, cfg=cfg, scene_audit_qa_ids=["qa-1", "qa-2"])
            for qa, identity in ctx.target_inputs.items()})
    elif tamper == "audit_group":
        ctx.target_requests["qa-1"]["scene_audit_qa_ids"] = ["qa-1"]
        _seal(ctx.target_requests["qa-1"])
    elif tamper == "geometry":
        np.save(case.art.depth_maps, np.full((32, 3, 4), 2, np.float32))
    elif tamper == "pixels":
        ctx.items[0].pixels[0][0, 0, 0] += 1
    else:
        ctx.items[0].episode = ctx.items[0].episode.model_copy(update={"question": "changed question"})
    with pytest.raises(PreparedBindingStoreError):
        derivation.derive_prepared_binding_scene_v11(ctx)


@pytest.mark.parametrize("tamper", ["reference", "payload", "generation", "provider"])
def test_store_and_provider_anchor_changes_are_rejected(case, tamper):
    source = case.context.sources["qa-1"]
    if tamper == "reference":
        source["reference"]["manifest_sha256"] = "0" * 64
    elif tamper == "payload":
        path = case.store.entry_path(source["reference"]["key"]) / "payload/binding.json"
        path.write_bytes(b"tampered")
    elif tamper == "generation":
        _replace_source(case, lambda binding: binding.stats["generation"].update(sha256="0" * 64))
    else:
        source["prepared"].stats["actual_model_requests"] = 0
    with pytest.raises(PreparedBindingStoreError):
        derivation.derive_prepared_binding_scene_v11(case.context)


@pytest.mark.parametrize("tamper", [
    "output", "policy", "receipt", "source_seal", "source_reference", "implementation",
    "generation_stats", "measurement", "mask_file",
])
def test_output_and_derivation_forgery_rejected_after_self_reseal(case, tamper):
    binding = derivation.derive_prepared_binding_scene_v11(case.context)["qa-1"]
    receipt = binding.stats["binding_derivation"]
    if tamper == "output":
        binding.objects[0].source_label = "invented"
    elif tamper == "policy":
        binding.objects[0].binding_provenance[0]["semantic_policy"] = "frozen-quality"
    elif tamper == "receipt":
        receipt["schema_version"] = "unsupported"
    elif tamper == "source_seal":
        receipt["source"]["manifest_sha256"] = "0" * 64
    elif tamper == "source_reference":
        receipt["source"]["reference"]["store_root"] = "/invented"
    elif tamper == "implementation":
        receipt["implementation"]["validator"]["sha256"] = "0" * 64
    elif tamper == "generation_stats":
        binding.stats["generation"]["cache_eligible"] = True
    elif tamper == "measurement":
        binding.prepared_measurements = {"old_metric": 7}
    else:
        np.save(binding.objects[0].mask_per_frame, np.zeros((32, 3, 4), np.uint8))
    if tamper != "mask_file":
        receipt["output_binding_sha256"] = content_sha256(derivation._content(binding))
    _seal(receipt)
    with pytest.raises(PreparedBindingStoreError):
        _validate(case, binding)


def test_explicit_opt_in_and_legacy_swap_cannot_bypass_native_validation(case):
    old = copy.deepcopy(case.context.sources["qa-1"]["prepared"])
    with pytest.raises(ValueError, match="legacy version"):
        _validate(case, old, context=False)
    old.stats["version"] = M5_BINDING_VERSION
    with pytest.raises(ValueError, match="parameters"):
        _validate(case, old, context=False)
    old.stats["generation"]["identity"]["parameters"] = case.context.target_requests["qa-1"]["parameters"]
    old.stats["generation"]["sha256"] = content_sha256(old.stats["generation"]["identity"])
    with pytest.raises(ValueError, match="implementation"):
        _validate(case, old, context=False)
    derived = derivation.derive_prepared_binding_scene_v11(case.context)["qa-1"]
    with pytest.raises(PreparedBindingStoreError, match="trusted derivation context"):
        _validate(case, derived, context=False)
    with pytest.raises(ValueError, match="source compatibility"):
        _validate(case, derived, source_compatibility={"fake_allowlist": True})
    with pytest.raises(PreparedBindingStoreError, match="exact current parameters"):
        _validate(case, derived, parameters=None)


def test_incomplete_context_group_is_rejected(case):
    ctx = replace(case.context, sources={"qa-1": case.context.sources["qa-1"]})
    with pytest.raises(PreparedBindingStoreError, match="group is incomplete"):
        derivation.derive_prepared_binding_scene_v11(ctx)


def test_completed_empty_object_outcome_preserves_sealed_generation(case):
    _replace_source(case, lambda binding: binding.objects.clear())
    binding = derivation.derive_prepared_binding_scene_v11(case.context)["qa-1"]
    assert not binding.objects and binding.materialized
    assert binding.stats["generation"] == case.context.sources["qa-1"]["prepared"].stats["generation"]
    _validate(case, binding)


def test_unavailable_parse_preserves_empty_source_without_new_requests(case):
    def mutate(binding):
        binding.objects.clear()
        binding.materialized = False
        binding.stats.update(
            status="parse_unavailable",
            object_parse={"status": "unavailable", "objects": []},
            candidate_receipts=[], detection_receipts=[])
        binding.stats.pop("grounding")

    _replace_source(case, mutate)
    before = copy.deepcopy(case.context.sources["qa-1"]["prepared"])
    binding = derivation.derive_prepared_binding_scene_v11(case.context)["qa-1"]
    assert not binding.objects and not binding.materialized
    assert binding.stats["status"] == "parse_unavailable"
    assert binding.stats["object_parse"] == before.stats["object_parse"]
    assert binding.stats["generation"] == before.stats["generation"]
    assert binding.stats["actual_model_requests"] == before.stats["actual_model_requests"]
    assert binding.stats["binding_derivation"]["new_model_requests"] == 0
    assert binding.stats["binding_derivation"]["quality_assessed"] is False
    _validate(case, binding)


def test_unavailable_parse_cannot_admit_existing_object_evidence(case):
    _replace_source(case, lambda binding: binding.stats.update(
        status="parse_unavailable", object_parse={"status": "unavailable", "objects": []}))
    with pytest.raises(PreparedBindingStoreError, match="contains object evidence"):
        derivation.derive_prepared_binding_scene_v11(case.context)


def test_provider_style_relocation_does_not_need_old_server_paths(case, tmp_path):
    root = tmp_path / "published"
    root.mkdir()
    shutil.copytree(case.store.root, root / "store")
    shutil.copyfile(case.path, root / "artifact.json")
    dependencies = {}
    for name, ref in artifact_dependency_refs(case.art).items():
        target = Path(ref).name
        dependencies[name] = target
        shutil.copyfile(ref, root / target)
    store = PreparedBindingStore(root / "store")
    sources = {}
    resolvers = {}
    for qa, old in case.context.sources.items():
        reference = {**old["reference"], "store_root": str(store.root)}
        receipt = store.create_migration_receipt(
            reference["key"], scene_input_identity=old["identity"],
            destination_root=root, scene_artifact_ref="artifact.json", dependency_refs=dependencies)
        receipt_path = root / (qa + "-migration.json")
        _write(receipt_path, receipt)
        resolver = store.validate_migration_receipt(reference["key"], receipt)
        sources[qa] = {
            "reference": reference, "identity": old["identity"],
            "prepared": load_prepared_binding_reference(
                reference, scene_input_identity=old["identity"], migration_receipt=receipt),
            "artifact_path": resolver.artifact_path, "artifact": resolver.resolved_artifact(),
            "artifact_relocation": resolver,
            "transport_receipt": {"migration_receipt_file": str(receipt_path)},
        }
        resolvers[qa] = resolver
    shutil.rmtree(case.root)
    ctx = replace(
        case.context, sources=sources, items=case.old_items,
        target_inputs={qa: source["identity"] for qa, source in sources.items()},
        target_requests={
            qa: prepared_binding_request_identity(
                input_identity=source["identity"], cfg=case.context.cfg,
                scene_audit_qa_ids=["qa-1", "qa-2"])
            for qa, source in sources.items()},
        target_artifact_paths={qa: root / "artifact.json" for qa in sources},
        target_relocations=resolvers)
    bindings = derivation.derive_prepared_binding_scene_v11(ctx)
    assert bindings["qa-1"].stats["binding_derivation"]["source"]["relocation_sha256"]
    assert not case.root.exists()
    references = derivation.publish_derived_binding_scene_v11(ctx, store_root=root / "derived")
    resolver = resolvers["qa-1"]
    derived_store = PreparedBindingStore(root / "derived")
    # The new store needs its own explicit relocation receipt for the old logical input.
    fresh = derived_store.create_migration_receipt(
        references["qa-1"]["key"], scene_input_identity=sources["qa-1"]["identity"],
        destination_root=root, scene_artifact_ref="artifact.json", dependency_refs=dependencies)
    binding = load_prepared_binding_reference(
        references["qa-1"], scene_input_identity=sources["qa-1"]["identity"],
        migration_receipt=fresh, scene_artifact=resolver.artifact_path)
    case.context = ctx
    _validate(case, binding)


def test_geometry_is_shared_readonly_mmap_once_per_unique_file(case, monkeypatch):
    original = np.load
    original_arrays = derivation._arrays
    geometry_paths = {str(getattr(case.art, name)) for name in case.arrays}
    reads = []
    in_geometry_loader = False

    def arrays(art, cache):
        nonlocal in_geometry_loader
        in_geometry_loader = True
        try:
            return original_arrays(art, cache)
        finally:
            in_geometry_loader = False

    def track(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if str(path) in geometry_paths:
            if in_geometry_loader:
                reads.append(str(path))
            assert kwargs["allow_pickle"] is False
            assert kwargs["mmap_mode"] == "r"
            assert isinstance(result, np.memmap) and not result.flags.writeable
        return result

    monkeypatch.setattr(np, "load", track)
    monkeypatch.setattr(derivation, "_arrays", arrays)
    derivation.derive_prepared_binding_scene_v11(case.context)
    assert sorted(reads) == sorted(geometry_paths)


def test_consumer_cache_does_not_rederive_or_retain_scene_arrays(case, monkeypatch):
    binding = derivation.derive_prepared_binding_scene_v11(case.context)["qa-1"]
    monkeypatch.setattr(derivation, "_derive", _never)
    _validate(case, binding)
    _validate(case, binding)
    cache = case.context._replay_cache
    assert set(cache) == {"fingerprint", "closure", "expected"}
    assert set(cache["expected"]) == {"qa-1", "qa-2"}
    # Only JSON receipts/stat tuples remain, not objects, RGB or geometry.
    json.dumps(cache, allow_nan=False)


def test_cache_checks_ctime_even_when_size_and_mtime_restored(case):
    binding = derivation.derive_prepared_binding_scene_v11(case.context)["qa-1"]
    source = case.context.sources["qa-2"]
    path = case.store.entry_path(source["reference"]["key"]) / "payload/binding.json"
    stat = path.stat()
    original = path.read_bytes()
    path.write_bytes(b"!" + original[1:])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(PreparedBindingStoreError, match="cached sealed"):
        _validate(case, binding)


@pytest.mark.parametrize("tamper", ["other_input", "cfg_budget", "current_implementation"])
def test_cached_context_and_implementation_changes_fail_closed(case, monkeypatch, tamper):
    binding = derivation.derive_prepared_binding_scene_v11(case.context)["qa-1"]
    if tamper == "other_input":
        case.context.target_inputs["qa-2"]["episode"]["question"] = "changed"
    elif tamper == "cfg_budget":
        case.context.cfg.max_tokens += 1
    else:
        implementation = derivation._implementation()
        implementation["validator"]["sha256"] = "0" * 64
        monkeypatch.setattr(derivation, "_implementation", lambda: implementation)
    with pytest.raises(PreparedBindingStoreError, match="cached derivation context"):
        _validate(case, binding)


def test_cached_consumption_revalidates_destination_payload_files(case, tmp_path, monkeypatch):
    references = derivation.publish_derived_binding_scene_v11(
        case.context, store_root=tmp_path / "derived")
    binding = load_prepared_binding_reference(
        references["qa-1"], scene_input_identity=case.context.target_inputs["qa-1"])
    monkeypatch.setattr(derivation, "_derive", _never)
    np.save(binding.objects[0].pointcloud_world, np.full((4, 3), 2, np.float32))
    with pytest.raises(PreparedBindingStoreError, match="current validation receipt"):
        _validate(case, binding)


def test_fully_resealed_receipts_cannot_invent_quality_or_drop_vision_evidence(case):
    def mutate(binding):
        row = binding.objects[0].binding_provenance[0]
        visual = binding.stats["candidate_receipts"][0]["semantic_visual"]
        body = row["semantic_binary_response"]
        body["vision_evidence"].pop("image_grid_thw")
        row["semantic_binary_response_sha256"] = content_sha256(body)
        visual["raw_response"] = copy.deepcopy(body)
        observation = {"visibility": "clear", "mask_purity": "clean", "mask_completeness": "whole"}
        row["semantic_observation"] = observation
        row["semantic_observation_sha256"] = content_sha256(observation)
        visual["observation"] = copy.deepcopy(observation)
        visual["observation_sha256"] = content_sha256(observation)
    _replace_source(case, mutate)
    with pytest.raises(PreparedBindingStoreError, match="binary evidence|image_grid_thw"):
        derivation.derive_prepared_binding_scene_v11(case.context)


def test_changed_legacy_store_source_and_forged_current_source_are_rejected(case):
    _replace_source(
        case, lambda binding: None,
        request_mutate=lambda request: request["store_source"].update(sha256="0" * 64))
    with pytest.raises(PreparedBindingStoreError, match="exact reviewed"):
        derivation.derive_prepared_binding_scene_v11(case.context)


def test_native_source_reseal_alone_does_not_upgrade_original_receipts(case):
    from skill3d.infra.m5_identity import source_identity
    old = copy.deepcopy(case.context.sources["qa-1"]["prepared"])
    old.stats["version"] = M5_BINDING_VERSION
    identity = old.stats["generation"]["identity"]
    identity["parameters"] = case.context.target_requests["qa-1"]["parameters"]
    identity["sources"] = source_identity(Path(derivation.__file__).resolve().parents[1])
    old.stats["generation"]["sha256"] = content_sha256(identity)
    with pytest.raises(ValueError, match="binding changed"):
        _validate(case, old, context=False)


def test_relocated_publish_rejects_dependency_forgery_without_fallback(case, tmp_path):
    root = tmp_path / "relocated"
    root.mkdir()
    shutil.copyfile(case.path, root / "artifact.json")
    dependencies = {}
    for name, path in artifact_dependency_refs(case.art).items():
        ref = Path(path).name
        dependencies[name] = ref
        shutil.copyfile(path, root / ref)
    source = case.context.sources["qa-1"]
    receipt = case.store.create_migration_receipt(
        source["reference"]["key"], scene_input_identity=source["identity"],
        destination_root=root, scene_artifact_ref="artifact.json", dependency_refs=dependencies)
    receipt["dependencies"]["depth_maps"]["identity"]["sha256"] = "0" * 64
    _seal(receipt, "receipt_sha256")
    store = PreparedBindingStore(root / "new-store")
    request = prepared_binding_request_identity(
        input_identity=source["identity"], cfg=case.context.cfg,
        scene_audit_qa_ids=["qa-1", "qa-2"])
    with pytest.raises(PreparedBindingStoreError, match="publication relocation rejected"):
        store.publish(
            request_identity=request, scene_artifact=root / "artifact.json",
            scene_input_identity=source["identity"], binding=source["prepared"],
            scene_migration_receipt=receipt)
    assert not list(store.root.rglob("manifest.json"))
