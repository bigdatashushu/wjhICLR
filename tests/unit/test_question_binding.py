"""M5 contracts exercised with real arrays, disk receipts and bounded model/detector fakes."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.infra.m5_identity import generation_identity
from skill3d.schemas.reconstruction import ObjectRecord
from skill3d.segmentation import question_binding as qb, sam2_tracker as st
from skill3d.segmentation.binding_validation import validate_binding
from skill3d.synthesis.request_context import current_request_phase
from skill3d.tools.scene_handle import SceneHandle


class TextModel:
    def __init__(self, rows, replies=None):
        self.rows, self.calls = rows, []
        self.replies = list(replies or [])
        self.phase_calls = {}

    def chat(self, messages, **kwargs):
        assert kwargs["seed"] == 137 and kwargs["temperature"] == 0
        phase = current_request_phase()
        self.phase_calls.setdefault(phase, []).append((copy.deepcopy(messages), kwargs))
        self.calls.append((phase, copy.deepcopy(messages), kwargs))
        if phase == "m5_mask_visual":
            return json.dumps({
                "observed_object": self.rows[0]["category"] if self.rows else "unknown",
                "alternate_object": "none", "visibility": "clear",
                "mask_purity": "clean", "mask_completeness": "whole",
            })
        if phase == "m5_mask_relation":
            pairs = json.loads(messages[0]["content"].split("Input pairs:\n", 1)[1])
            return json.dumps({"results": [
                {"id": row["id"], "relation": "match"} for row in pairs]})
        assert all(isinstance(row["content"], str) for row in messages)
        return self.replies.pop(0) if self.replies else json.dumps({"objects": self.rows})


@pytest.fixture
def harness(tmp_path, monkeypatch):
    frames = [np.full((8, 10, 3), i, np.uint8) for i in range(32)]
    depth = np.full((32, 8, 10), 4.0)
    c2w = np.repeat(np.eye(4)[None], 32, axis=0)
    intr = np.eye(3)
    conf = np.arange(depth.size, dtype=float).reshape(depth.shape)
    calls, segment_calls = [], []

    def identity(**kw):
        return generation_identity(
            scene_name=kw["scene_name"], frame_set_hash=kw["frame_set_hash"],
            frames=kw["frames"], depth_maps=kw["depth_maps"], c2w_list=kw["c2w_list"],
            intrinsics=kw["intrinsics"], point_conf=kw["point_conf"],
            depth_conf=kw["depth_conf"], point_map=kw["point_map"],
            question=kw["question"], question_type=kw["question_type"],
            options=kw["options"], reconstruction_identity=kw["reconstruction_identity"],
            model={"weights": "a" * 64, "seed": kw["seed"]},
            detector={"weights": "b" * 64}, sam2={"weights": "c" * 64},
            parameters=kw["parameters"], sources={"fixture": {"sha256": "d" * 64}})

    def detect(frame, prompt):
        calls.append((int(frame[0, 0, 0]), prompt))
        return [qb.ovd.Detection(prompt[:-1], (1, 1, 3, 3), .8)]

    def segment(frame, boxes, predictor):
        segment_calls.append((int(frame[0, 0, 0]), len(boxes)))
        result = []
        for box in boxes:
            mask = np.zeros(frame.shape[:2], bool)
            x0, y0, x1, y1 = map(int, box)
            mask[y0:y1, x0:x1] = True
            result.append(mask)
        return result

    monkeypatch.setattr(qb, "inventory_generation_identity", identity)
    monkeypatch.setattr(qb.ovd, "detect", detect)
    monkeypatch.setattr(st, "segment_frame_boxes", segment)
    monkeypatch.setattr(st, "build_image_predictor", lambda predictor=None: object())
    monkeypatch.setattr(st, "dynamic_mask_from_rigidity", lambda *args: (
        np.zeros(depth.shape, bool), []))
    args = dict(frames=frames, question="How many chairs are visible?",
                question_type="object_counting", options=[], depth_maps=depth,
                c2w_list=c2w, intrinsics=intr, point_conf=conf, depth_conf=conf,
                out_dir=tmp_path, seed=137, scene_identity="frozen",
                vlm_client=TextModel([
                    {"category": "chair", "role": "target", "source_text": "chairs"}]))
    return SimpleNamespace(args=args, calls=calls, segment_calls=segment_calls,
                           detect=detect, segment=segment)


def test_exhaustive_scan_deduplicates_observations_and_recomputes_saved_facts(harness):
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    assert harness.calls == [(i, "chair.") for i in range(32)]
    assert harness.segment_calls == [(i, 1) for i in range(32)]
    assert len(objects) == 1 and len(objects[0].candidate_ids) == 32
    obj = objects[0]
    assert obj.grounding_status == "question_driven" and not obj.duplicate_suspect
    assert obj.binding_validation.status == "validated"
    assert stats["track_consensus"]["track_fragmentation_ratio"] == 0
    assert stats["track_consensus"]["n_dedup_absorbed"] == 31
    masks = np.load(obj.mask_per_frame)
    pts, conf = np.load(obj.pointcloud_world), np.load(obj.pointconf_world)
    assert obj.visible_frames == np.flatnonzero(masks.any(axis=(1, 2))).tolist()
    np.testing.assert_allclose(obj.centroid_world, pts.mean(axis=0))
    np.testing.assert_allclose(obj.bbox, np.r_[pts.min(axis=0), pts.max(axis=0)])
    assert len(pts) == len(conf)
    assert len(harness.args["vlm_client"].phase_calls["m5_object_parse"]) == 1
    parse_messages, parse_kwargs = (
        harness.args["vlm_client"].phase_calls["m5_object_parse"][0])
    assert "json" in json.dumps(parse_messages).lower()
    assert parse_kwargs["response_format"] == {"type": "json_object"}


def test_late_detection_still_scans_every_frame(harness, monkeypatch):
    def detect(frame, prompt):
        result = harness.detect(frame, prompt)
        return result if int(frame[0, 0, 0]) == 27 else []
    monkeypatch.setattr(qb.ovd, "detect", detect)
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    assert len(objects) == 1 and len(harness.calls) == 32
    assert stats["grounding"]["all_present"]
    assert objects[0].binding_provenance[0]["frame_slot"] == 27


def test_all_instances_and_nms_suppression_have_candidate_receipts(harness, monkeypatch):
    def detect(frame, prompt):
        return harness.detect(frame, prompt) + [
            qb.ovd.Detection("chair", (1, 1, 3, 3), .7),
            qb.ovd.Detection("chair", (6, 1, 8, 3), .9)]
    monkeypatch.setattr(qb.ovd, "detect", detect)
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    assert len(objects) == 2 and all(len(o.candidate_ids) == 32 for o in objects)
    assert harness.segment_calls == [(i, 2) for i in range(32)]
    for scan in stats["detection_receipts"]:
        assert len(scan["candidates"]) == 3
        assert sum(bool(row["suppressed_by"]) for row in scan["candidates"]) == 1
        assert all(row["request_sha256"] and row["score"] for row in scan["candidates"])


@pytest.mark.parametrize("task", list(qb.ROLES)[:-1])
def test_all_object_tasks_use_the_same_complete_scan(harness, task):
    roles = sorted(qb.ROLES[task])
    question = "Locate " + ", ".join("object" + str(i) for i in range(len(roles)))
    rows = [{"category": "chair", "role": role, "source_text": "object" + str(i)}
            for i, role in enumerate(roles)]
    harness.args.update(question=question, question_type=task, vlm_client=TextModel(rows))
    qb.bind_question_objects(**harness.args)
    assert harness.calls == [(i, "chair.") for i in range(32)]


@pytest.mark.parametrize("replies", [
    ["not JSON", '{"objects":[]}'],
    ['{"objects":[{"category":"chair","role":"target","source_text":"invented"}]}'] * 2,
    ['{"objects":[{"category":"robot","role":"target","source_text":"chairs"}]}'] * 2,
])
def test_invalid_parse_has_only_one_repair_and_no_guessing(harness, replies):
    model = TextModel([], replies)
    harness.args["vlm_client"] = model
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    assert objects == [] and stats["status"] == "parse_unavailable"
    assert len(model.calls) == 2 and harness.calls == []
    assert all(row["raw_response"] for row in stats["object_parse"]["requests"])
    assert list(Path(harness.args["out_dir"]).glob("object_parses/*.json"))


def test_format_repair_and_source_roles(harness):
    model = TextModel(harness.args["vlm_client"].rows, ["```json\n{}\n```"])
    harness.args["vlm_client"] = model
    _, stats, _ = qb.bind_question_objects(**harness.args)
    assert (stats["object_parse"]["status"] == "parsed"
            and len(model.phase_calls["m5_object_parse"]) == 2)


def test_option_coverage_and_limit_are_validated():
    inputs = qb.question_input("Closest to the desk?", "object_rel_distance",
                               ["A. chair", "B. cup"])
    rows = [{"category": "desk", "role": "reference", "source_text": "desk"},
            {"category": "chair", "role": "candidate", "source_text": "chair"}]
    with pytest.raises(ValueError, match="option"):
        qb.validate_parse(json.dumps({"objects": rows}), inputs)
    rows.append({"category": "cup", "role": "candidate", "source_text": "cup"})
    assert len(qb.validate_parse(json.dumps({"objects": rows}), inputs)) == 3
    with pytest.raises(ValueError, match="16"):
        qb.validate_parse(json.dumps({"objects": rows * 6}), inputs)


def test_failed_detector_frame_does_not_stop_the_scan(harness, monkeypatch):
    def detect(frame, prompt):
        result = harness.detect(frame, prompt)
        if int(frame[0, 0, 0]) == 3:
            qb.ovd.detect.last_error = "TimeoutError: final failure"
            return []
        return result
    monkeypatch.setattr(qb.ovd, "detect", detect)
    _, stats, _ = qb.bind_question_objects(**harness.args)
    assert len(harness.calls) == 32 and stats["detector_fault"]
    assert not stats["grounding"]["all_present"]
    assert stats["grounding"]["requirements"][0]["scan_errors"][0]["frame_slot"] == 3


def test_failed_track_or_empty_detection_is_unavailable_not_absence(harness, monkeypatch):
    monkeypatch.setattr(st, "segment_frame_boxes", lambda *args, **kwargs: [
        np.zeros((8, 10), dtype=bool) for _ in args[1]])
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    assert objects == [] and stats["grounding"]["unresolved_names"] == ["chair"]
    assert len(stats["candidate_receipts"]) == 32
    assert stats["track_consensus"]["track_fragmentation_ratio"] == 1


def test_semantic_mismatch_is_audited_rejection_not_transport_failure(
        harness, monkeypatch):
    def reject(rows, client, **kwargs):
        return ({row["candidate_id"]: {
            "status": "reviewed", "relation": "mismatch",
            "request_sha256": "r" * 64,
        } for row in rows}, {"actual_requests": 1, "n_batches": 1})

    monkeypatch.setattr(qb.msr, "compare_categories", reject)
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    assert objects == []
    assert all(row["status"] == "semantic_rejected"
               for row in stats["candidate_receipts"])
    requirement = stats["grounding"]["requirements"][0]
    assert len(requirement["semantic_rejections"]) == 32
    assert requirement["binding_failures"] == []
    assert stats["track_consensus"]["n_semantic_rejected"] == 32


def test_cached_binding_is_same_question_only_and_corruption_rebuilds(harness):
    objects, first, _ = qb.bind_question_objects(**harness.args)
    cached, second, _ = qb.bind_question_objects(**harness.args)
    assert second["cache_hit"] and len(harness.calls) == 32
    assert cached[0].model_dump() == objects[0].model_dump()
    assert first["generation"] == second["generation"]
    np.save(objects[0].pointcloud_world, np.ones((128, 3)))
    _, third, _ = qb.bind_question_objects(**harness.args)
    assert not third["cache_hit"] and len(harness.calls) == 64
    harness.args["question"] += " exactly"
    _, fourth, _ = qb.bind_question_objects(**harness.args)
    assert not fourth["cache_hit"] and len(harness.calls) == 96
    assert third["generation"]["sha256"] != fourth["generation"]["sha256"]


@pytest.mark.parametrize("field", ["visible_frames", "centroid_world", "bbox",
                                   "pointconf_world", "mask_per_frame"])
def test_saved_binding_validation_rejects_inconsistent_facts(harness, field):
    objects, _, _ = qb.bind_question_objects(**harness.args)
    obj = objects[0]
    if field == "visible_frames":
        obj = obj.model_copy(update={field: [0]})
    elif field in ("centroid_world", "bbox"):
        obj = obj.model_copy(update={field: [0.] * len(getattr(obj, field))})
    else:
        np.save(getattr(obj, field), np.zeros((1,)))
    assert validate_binding(obj, check_receipt=True).status == "unavailable"


def test_room_task_and_m4_failure_do_not_request_objects(harness):
    harness.args.update(question_type="room_size_estimation")
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    assert objects == [] and stats["status"] == "objects_not_required"
    assert not harness.calls and not harness.args["vlm_client"].calls
    harness.args.update(question_type="object_counting", geometry_usable=False)
    _, stats, _ = qb.bind_question_objects(**harness.args)
    assert stats["status"] == "m4_unavailable" and not stats["materialized"]
    assert not harness.calls and not harness.args["vlm_client"].calls


def test_new_handle_rejects_old_sources_even_when_materialized(harness):
    from skill3d.schemas.reconstruction import SceneState
    from skill3d.tools.geometry_tools import count_objects, exists_in_scene
    from skill3d.tools.contract import ConfidenceGateError
    handle = SceneHandle(SceneState(artifact_ref="s"), objects=[
        ObjectRecord(obj_id="old", category_name="chair", track_id="old")],
        objects_materialized=True, require_question_binding=True)
    assert not handle.binding_usable("old")
    with pytest.raises(ConfidenceGateError):
        count_objects(handle, "chair")
    with pytest.raises(ConfidenceGateError):
        exists_in_scene(handle, "chair")


def test_m11_rejects_tampered_bindings_and_zero_without_usable_instances(harness):
    from skill3d.schemas.reconstruction import SceneState
    from skill3d.verifier.geometry_oracle import check_binding_integrity
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    handle = SceneHandle(SceneState(artifact_ref="s"), objects=objects,
                         objects_materialized=True, require_question_binding=True,
                         grounding_coverage=stats["grounding"])
    result = SimpleNamespace(
        status="ok", error=None, tool="count_objects",
        args={"category_name": "chair"}, value='{"count":1}',
        object_dependencies=[objects[0].obj_id])
    trace = SimpleNamespace(results=[result])
    assert check_binding_integrity(trace, handle)
    np.save(objects[0].pointcloud_world, np.zeros((128, 3)))
    assert not check_binding_integrity(trace, handle)
    empty = SceneHandle(SceneState(artifact_ref="s"), objects=[],
                        objects_materialized=True, require_question_binding=True)
    result.value = '{"count":0}'
    result.object_dependencies = []
    assert not check_binding_integrity(trace, empty)


def test_incomplete_generation_identity_never_reuses_persistent_results(harness, monkeypatch):
    original = qb.inventory_generation_identity
    def incomplete(**kwargs):
        generation = original(**kwargs)
        generation.update(cache_eligible=False, missing=["sam2"])
        return generation
    monkeypatch.setattr(qb, "inventory_generation_identity", incomplete)
    _, first, _ = qb.bind_question_objects(**harness.args)
    _, second, _ = qb.bind_question_objects(**harness.args)
    assert not first["cache_hit"] and not second["cache_hit"]
    assert len(harness.calls) == 64
    assert len(harness.args["vlm_client"].phase_calls["m5_object_parse"]) == 2


def test_cache_rejects_legacy_version_and_escaped_references(harness):
    objects, stats, _ = qb.bind_question_objects(**harness.args)
    cache = (Path(harness.args["out_dir"]) / "question_bindings"
             / stats["generation"]["sha256"] / "binding.json")
    data = json.loads(cache.read_text())
    data["version"] = "m5-inventory-10-frozen-inputs"
    cache.write_text(json.dumps(data))
    _, rebuilt, _ = qb.bind_question_objects(**harness.args)
    assert not rebuilt["cache_hit"] and len(harness.calls) == 64
    payload = qb._load_envelope(cache, rebuilt["generation"]["sha256"])
    payload["objects"][0]["mask_per_frame"] = str(Path(objects[0].mask_per_frame).resolve())
    # Absolute refs to an old publication are not relative cache-owned references.
    payload["objects"][0]["pointcloud_world"] = "../../outside.npy"
    qb._save_envelope(cache, rebuilt["generation"]["sha256"], payload)
    _, rebuilt_again, _ = qb.bind_question_objects(**harness.args)
    assert not rebuilt_again["cache_hit"] and len(harness.calls) == 96

@pytest.mark.parametrize('label,purity,complete,visibility,accepted',[
    ('chair','clean','whole','clear',True),('chair','mixed','whole','clear',False),
    ('chair','clean','fragment','clear',False),('chair','unknown','whole','clear',False),
    ('chair','clean','whole','insufficient',False),('unknown','clean','whole','clear',False),
])
def test_experimental_provider_runs_m5_and_gates_geometry(harness,monkeypatch,label,purity,complete,visibility,accepted):
    from skill3d.segmentation import intern_decision_client as local
    class Visual:
        def observe_candidate(self,candidate,*args,**kwargs):
            return {'version':'experimental','provider':'test-experimental','status':'observed',
                'request_sent':True,'request_sha256':'a'*64,'observation':{
                'observed_object':label,'mask_purity':purity,'mask_completeness':complete,
                'visibility':visibility}}
    monkeypatch.setattr(local,'make_review_client',lambda config:Visual())
    objects,stats,_=qb.bind_question_objects(**harness.args,
        semantic_review_config={'provider':'intern-decision','relation':'frozen-rules'})
    assert bool(objects)==accepted
    assert stats['generation']['cache_eligible'] is False
    assert 'experimental_semantic_review' in stats['generation']['identity']['parameters']
    assert list(harness.args['vlm_client'].phase_calls)==['m5_object_parse']
    if accepted:
        assert all(o.binding_validation.status=='validated' for o in objects)


def test_intern_binding_receipt_rejects_changed_observation(harness):
    from skill3d.infra.m5_identity import content_sha256
    from skill3d.segmentation.intern_decision_protocol import PROTOCOL_SHA256
    objects, _, _ = qb.bind_question_objects(**harness.args)
    obj = objects[0]
    observation = {"observed_object": "chair", "visibility": "clear",
                   "mask_purity": "clean", "mask_completeness": "whole"}
    model = {"model": "internlm/Intern-Decision-4B", "revision": "frozen"}
    provenance = [{**r, "semantic_provider": "intern-decision-local",
        "semantic_observation": observation,
        "semantic_observation_sha256": content_sha256(observation),
        "semantic_request_identity": {"model": model, "protocol_sha256": PROTOCOL_SHA256},
        "semantic_model_identity_sha256": content_sha256(model)} for r in obj.binding_provenance]
    good = obj.model_copy(update={"binding_provenance": provenance})
    assert validate_binding(good).status == "validated"
    changed = copy.deepcopy(provenance)
    changed[0]["semantic_observation"]["mask_completeness"] = "fragment"
    bad = good.model_copy(update={"binding_provenance": changed})
    assert validate_binding(bad).status == "unavailable"
