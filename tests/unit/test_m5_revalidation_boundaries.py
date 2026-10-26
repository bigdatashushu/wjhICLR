"""Synthetic M5 receipts must survive every independent consumption boundary."""

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from test_question_binding import harness
from skill3d.infra.m5_identity import content_sha256
from skill3d.schemas.reconstruction import SceneState
from skill3d.segmentation import binding_validation as bv, question_binding as qb
from skill3d.tools.scene_handle import SceneHandle


def assert_rejected(obj, grounding):
    assert bv.validate_binding(obj, check_receipt=True).status == "unavailable"
    assert not bv.binding_is_usable(obj)
    coverage = qb.refresh_coverage(grounding, [obj])
    assert coverage["n_new"] == 0
    assert coverage["status"] == "unavailable"
    handle = SceneHandle(
        SceneState(artifact_ref="synthetic"), objects=[obj],
        objects_materialized=True, require_question_binding=True,
        grounding_coverage=grounding)
    assert not handle.binding_usable(obj.obj_id)


@pytest.mark.parametrize("field,value", [
    ("mask_purity", "mixed"), ("mask_purity", "background"),
    ("mask_completeness", "fragment"), ("mask_completeness", "unknown"),
    ("visibility", "insufficient"), ("observed_object", "unknown"),
    ("observed_object", "other"),
])
@pytest.mark.parametrize("rehash_observation", [False, True])
def test_saved_qwen_quality_is_rechecked(harness, field, value, rehash_observation):
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    obj = objects[0]
    rows = copy.deepcopy(obj.binding_provenance)
    rows[-1]["semantic_observation"][field] = value
    if rehash_observation:
        rows[-1]["semantic_observation_sha256"] = content_sha256(
            rows[-1]["semantic_observation"])
    changed = obj.model_copy(update={"binding_provenance": rows})
    assert bv.validate_binding(changed).status == "unavailable"
    assert_rejected(changed, stats["grounding"])


@pytest.mark.parametrize("visibility,completeness", [
    ("clear", "whole"), ("partial", "whole"),
    ("clear", "major_part"), ("partial", "major_part"),
])
def test_qwen_quality_thresholds_are_unchanged(harness, visibility, completeness):
    objects, _, _ = qb.bind_question_objects(**harness.args)
    obj = objects[0]
    rows = copy.deepcopy(obj.binding_provenance)
    for row in rows:
        row["semantic_observation"].update(
            visibility=visibility, mask_completeness=completeness)
        row["semantic_observation_sha256"] = content_sha256(row["semantic_observation"])
    changed = obj.model_copy(update={"binding_provenance": rows})
    verdict = bv.validate_binding(changed, generation_sha256="new-synthetic-generation")
    assert verdict.status == "validated"
    changed = changed.model_copy(update={"binding_validation": verdict})
    assert bv.binding_is_usable(changed)


@pytest.mark.parametrize("mutation", [
    "alternate_object", "observed_object", "extra_observation",
    "candidate_id", "frame_slot", "bbox", "relation_request",
    "roles", "track_id", "obj_id", "source_label", "provider", "policy", "grounding_status",
    "model_identity", "known_provider",
])
def test_receipt_seals_complete_semantic_identity(harness, mutation):
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    obj = objects[0]
    rows = copy.deepcopy(obj.binding_provenance)
    update = {"binding_provenance": rows}
    if mutation in {"alternate_object", "observed_object", "extra_observation"}:
        rows[0]["semantic_observation"][mutation] = "cup"
        rows[0]["semantic_observation_sha256"] = content_sha256(
            rows[0]["semantic_observation"])
    elif mutation == "candidate_id":
        rows[0]["candidate_id"] = "different-candidate"
        update["candidate_ids"] = [row["candidate_id"] for row in rows]
        identity = rows[0]["semantic_request_identity"]
        identity["candidate_id"] = rows[0]["candidate_id"]
        rows[0]["semantic_visual_request_sha256"] = content_sha256(identity)
    elif mutation == "frame_slot":
        rows[0]["frame_slot"] = 1
    elif mutation == "bbox":
        rows[0]["bbox"] = [0, 0, 1, 1]
    elif mutation == "relation_request":
        rows[0]["semantic_relation_request_sha256"] = "d" * 64
    elif mutation == "model_identity":
        identity = rows[0]["semantic_request_identity"]
        identity["model"] = {"model": "different-model"}
        rows[0]["semantic_model_identity_sha256"] = content_sha256(identity["model"])
        rows[0]["semantic_visual_request_sha256"] = content_sha256(identity)
    elif mutation == "known_provider":
        from skill3d.segmentation.intern_decision_protocol import PROTOCOL_SHA256
        rows[0]["semantic_provider"] = "intern-decision-local"
        rows[0]["semantic_request_identity"]["protocol_sha256"] = PROTOCOL_SHA256
    elif mutation == "roles":
        update["question_roles"] = [{"category": "chair", "role": "target_a",
                                    "source_text": "chairs"}]
    elif mutation == "grounding_status":
        update["grounding_status"] = "base_list"
    elif mutation in {"track_id", "obj_id", "source_label"}:
        update[mutation] = "different-identity"
    elif mutation == "provider":
        rows[0]["semantic_provider"] = "unregistered-reviewer"
    else:
        rows[0]["semantic_policy"] = "unregistered-policy"
    assert_rejected(obj.model_copy(update=update), stats["grounding"])


@pytest.mark.parametrize("provider,policy", [
    ("unknown", "visual-quality-and-relation"),
    ("", "visual-quality-and-relation"),
    ("bailian", "category-only"),
    ("bailian", "frozen-quality"),
    ("bailian", "unknown"),
    ("intern-decision-local", "category-only"),
    ("intern-decision-binary-local", "visual-quality-and-relation"),
    ("intern-decision-binary-local", "unknown"),
    (["bailian"], "visual-quality-and-relation"),
    ("bailian", ["visual-quality-and-relation"]),
])
def test_unknown_or_incompatible_review_routes_fail_closed(harness, provider, policy):
    objects, _, _ = qb.bind_question_objects(**harness.args)
    rows = copy.deepcopy(objects[0].binding_provenance)
    rows[0].update(semantic_provider=provider, semantic_policy=policy)
    changed = objects[0].model_copy(update={"binding_provenance": rows})
    verdict = bv.validate_binding(changed, generation_sha256="new-generation")
    assert verdict.status == "unavailable"
    assert not verdict.checks["semantic_policy"]


@pytest.mark.parametrize("missing", [
    "semantic_observation_sha256", "semantic_policy", "semantic_provider",
])
def test_new_acceptance_cannot_backfill_missing_semantic_evidence(harness, missing):
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    rows = copy.deepcopy(objects[0].binding_provenance)
    rows[0].pop(missing)
    changed = objects[0].model_copy(update={"binding_provenance": rows})
    before = changed.model_dump()
    assert bv.validate_binding(changed).status == "unavailable"
    assert_rejected(changed, stats["grounding"])
    assert changed.model_dump() == before


def test_legacy_receipt_and_missing_semantic_seal_are_not_upgraded(harness):
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    obj = objects[0]
    for receipt in [
        obj.binding_validation.model_copy(update={
            "version": "m5-binding-validation-v2-semantic-review"}),
        obj.binding_validation.model_copy(update={"files": {
            key: value for key, value in obj.binding_validation.files.items()
            if key != bv.SEMANTIC_IDENTITY_KEY}}),
    ]:
        changed = obj.model_copy(update={"binding_validation": receipt})
        before = changed.model_dump()
        assert_rejected(changed, stats["grounding"])
        assert changed.model_dump() == before


@pytest.mark.parametrize("mutation", ["corrupt", "same_size_mask", "replace_points"])
def test_file_mutations_remain_rejected_after_warm_revalidation(harness, mutation):
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    obj = objects[0]
    assert bv.binding_is_usable(obj)
    assert qb.refresh_coverage(stats["grounding"], objects)["n_new"] == 1
    field = "pointcloud_world" if mutation == "replace_points" else "mask_per_frame"
    path = Path(getattr(obj, field))
    stamp = path.stat()
    if mutation == "corrupt":
        path.write_bytes(b"not a numpy file")
    elif mutation == "same_size_mask":
        mask = np.load(path)
        mask[0, 1, 1] = 0
        mask[0, 0, 0] = 1
        np.save(path, mask)
        assert path.stat().st_size == stamp.st_size
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert bv.validate_binding(obj).status == "validated"
    else:
        points = np.load(path)
        replacement = path.with_name("replacement.npy")
        np.save(replacement, points[::-1])
        assert replacement.stat().st_size == stamp.st_size
        os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        replacement.replace(path)
        assert bv.validate_binding(obj).status == "validated"
    assert_rejected(obj, stats["grounding"])


@pytest.fixture
def category_binding(harness, monkeypatch):
    from skill3d.segmentation import intern_decision_client as local
    from skill3d.segmentation.intern_binary_review import InternBinaryClient, default_mask_review
    client = InternBinaryClient("http://127.0.0.1:18445", prompt_version="P1",
                               yes_threshold=.77, admission_policy="category-only")
    client._identity = {
        "model": "internlm/Intern-Decision-4B",
        "calibration": {"temperature": 1.99241824}}

    def answer(probability):
        return {
            "model": "Intern-Decision-4B", "backend": "hf",
            "answers": {"target_match": {
                "type": "noul", "probabilities": {"no": 1-probability, "yes": probability},
                "decision": "yes", "confidence": probability, "noul": probability}}}

    def call(path, body):
        return {
            "request_sha256": body["request_sha256"],
            "identity_sha256": content_sha256(client._identity),
            "result": {**answer(.77), "calibration": client._identity["calibration"]},
            "uncalibrated_result": {**answer(.9), "calibration": {"temperature": 1}},
            "vision_evidence": {
                "image_count": 3, "image_token_count": 44, "pixel_values_shape": [1, 1],
                "image_grid_thw": [[1, 2, 2]] * 3, "decision_logit_positions": [100]}}

    monkeypatch.setattr(client, "_call", call)
    monkeypatch.setattr(local, "make_review_client", lambda config: client)
    return qb.bind_question_objects(
        **harness.args, semantic_review_config=default_mask_review())


def test_category_only_empty_observation_remains_legal_and_sealed(category_binding):
    objects, stats, _ = category_binding
    obj = objects[0]
    assert obj.binding_validation.status == "validated"
    assert bv.binding_is_usable(obj)
    assert all(row["semantic_observation"] == {} for row in obj.binding_provenance)
    assert not stats["semantic_review"]["quality_assessed"]
    assert obj.binding_validation.files[bv.SEMANTIC_IDENTITY_KEY]["sha256"]
    rows = copy.deepcopy(obj.binding_provenance)
    rows[0]["semantic_observation"]["visibility"] = "clear"
    rows[0]["semantic_observation_sha256"] = content_sha256(rows[0]["semantic_observation"])
    assert_rejected(obj.model_copy(update={"binding_provenance": rows}), stats["grounding"])


@pytest.mark.parametrize("policy", ["unknown", "frozen-quality", "visual-quality-and-relation", None])
def test_category_only_cannot_be_relabelled_as_another_policy(category_binding, policy):
    objects, stats, _ = category_binding
    rows = copy.deepcopy(objects[0].binding_provenance)
    rows[0]["semantic_policy"] = policy
    changed = objects[0].model_copy(update={"binding_provenance": rows})
    assert bv.validate_binding(changed).status == "unavailable"
    assert_rejected(changed, stats["grounding"])


def test_qwen_visual_receipt_generates_observation_and_model_hashes(harness):
    _, stats, _ = qb.bind_question_objects(**harness.args)
    for row in stats["candidate_receipts"]:
        if not row.get("review_selected"):
            continue
        visual = row["semantic_visual"]
        assert visual["observation_sha256"] == content_sha256(visual["observation"])
        assert visual["model_identity_sha256"] == content_sha256(
            visual["request_identity"]["model"])
        assert visual["provider"] == "bailian"
        assert visual["policy"] == "visual-quality-and-relation"
        saved = json.loads(next(
            Path(harness.args["out_dir"]).glob(
                f"question_bindings/*/*/semantic_reviews/visual/{row['candidate_id']}.json")
        ).read_text())
        assert saved["observation_sha256"] == visual["observation_sha256"]


def test_prepared_validation_rejects_old_contract_without_mutating_it(harness, monkeypatch):
    from types import SimpleNamespace
    objects, stats, notes = qb.bind_question_objects(**harness.args)
    binding = SimpleNamespace(objects=objects, stats=stats, notes=notes)
    args = harness.args
    validation = dict(
        frames=args["frames"], depth=args["depth_maps"], c2w=args["c2w_list"],
        intrinsics=args["intrinsics"], point_conf=args["point_conf"],
        depth_conf=args["depth_conf"], point_map=None,
        inputs=qb.question_input(args["question"], args["question_type"], []),
        scene_identity=args["scene_identity"])
    monkeypatch.setattr(qb, "source_identity",
                        lambda root: stats["generation"]["identity"]["sources"])
    qb.validate_prepared_binding(binding, **validation)
    stats["version"] = "m5-question-binding-v3-grouped-review"
    before = copy.deepcopy(stats)
    receipts = [obj.model_dump() for obj in objects]
    with pytest.raises(ValueError, match="legacy version"):
        qb.validate_prepared_binding(binding, **validation,
                                     source_compatibility="r10-frozen-binding-transport-review/1.0")
    assert stats == before
    assert [obj.model_dump() for obj in objects] == receipts
