"""Opt-in 32+1 delivery, frozen provenance, and request-time integrity without services."""

import base64
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig
from skill3d.schemas import SceneState
from skill3d.schemas.evidence import EvidenceProfile
from skill3d.schemas.image_ledger import ImageRound, ProducedImage, SceneOverviewImage
from skill3d.synthesis.eval_visual_fallback import build_visual_request, request_visual_answer
from skill3d.synthesis.prompt_builder import SCENE_OVERVIEW_GUIDANCE, build_image_messages
from skill3d.synthesis.request_context import current_request_phase
from skill3d.tools.image_ledger import ImageLedger
from skill3d.tools.scene_handle import SceneHandle

from test_tool_contract_recovery import _write_v6_artifact


def overview(tmp_path, scene_id="scene-test"):
    pixels = np.full((480, 640, 3), (31, 127, 211), dtype=np.uint8)
    path = (tmp_path / "overview.png").resolve()
    Image.fromarray(pixels).save(path)
    rgb_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
    return {
        "schema": "harness3d-scene-overview/1",
        "scene_id": scene_id,
        "image_id": f"overview-{scene_id}-{rgb_sha[:12]}",
        "image_path": str(path),
        "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rgb_sha256": rgb_sha,
        "scene_artifact_sha256": "a" * 64,
        "frozen_scene_sha256": "b" * 64,
        "render_spec_sha256": "c" * 64,
        "world_frame_status": "degraded",
        "resolution_wh": [640, 480],
        "selected_azimuth_degrees": 45.0,
        "elevation_degrees": 60.0,
        "render_only_cutaway": True,
    }, pixels


def make_ledger(*, max_images=32, max_derived=8):
    ledger = ImageLedger(max_images=max_images, max_derived_images=max_derived)
    ledger.set_frames(
        list(range(32)), [np.full((24, 32, 3), i * 7, np.uint8) for i in range(32)],
        source_frame_indices=[i * 100 for i in range(32)])
    ledger.ensure_frame_records()
    return ledger


def crops(ledger, count):
    return [
        ledger.produce(
            ledger.frames[31 - i % 32][:12, :12], kind="crop",
            produced_by="inspect_frames", source_frame_id=31 - i % 32,
            box_xyxy=[0, 0, 12, 12], round_index=1).image_id
        for i in range(count)
    ]


def image_parts(messages):
    return [part for message in messages if isinstance(message["content"], list)
            for part in message["content"] if part["type"] == "image_url"]


def assert_actual_pack(ledger, plan, messages):
    actual = image_parts(messages)
    expected = image_parts(build_image_messages(
        "", [ledger.pixels_of(i) for i in plan.original_image_ids],
        max_images=33, extra=[(i, ledger.pixels_of(i)) for i in plan.derived_image_ids]))
    assert len(actual) == plan.n_images == 33
    assert actual == expected
    hashes = [hashlib.sha256(base64.b64decode(
        item["image_url"]["url"].split(",", 1)[1])).hexdigest() for item in actual]
    assert plan.jpeg_sha256 == hashes
    assert plan.request_sha256 == hashlib.sha256(json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(
        actual[0]["image_url"]["url"].split(",", 1)[1]))))
    assert decoded.shape == (480, 640, 3)
    assert np.max(np.abs(decoded.astype(int) - ledger.pixels_of(plan.image_ids[0]))) <= 2


def test_typed_flat_provenance_preserves_frames_and_offline_refs(tmp_path):
    descriptor, pixels = overview(tmp_path)
    ledger = make_ledger(max_images=33)
    before = [record.model_dump() for record in ledger.all_images()]
    rec = ledger.register_scene_overview(descriptor, scene_id="scene-test")
    assert isinstance(rec, SceneOverviewImage)
    row = rec.model_dump()
    assert all(row[key] == value for key, value in descriptor.items() if key != "image_path")
    assert "source_provenance" not in row and "image_path" not in row
    assert row["source_frame_id"] is row["source_frame_index"] is row["box_xyxy"] is None
    assert rec.observed_rounds == rec.delivered_rounds == []
    assert not ledger.pixels_of(rec.image_id).flags.writeable
    np.testing.assert_array_equal(ledger.pixels_of(rec.image_id), pixels)
    assert [r.model_dump() for r in ledger.all_images() if r.kind == "frame"] == before
    assert ledger.frame_ids == list(range(32))
    assert ledger.stats()["n_derived_produced"] == 0
    assert ledger.stats()["n_scene_overview_produced"] == 1
    assert SceneOverviewImage.model_validate(row).model_dump() == row
    assert ledger.register_scene_overview(descriptor) is rec
    assert ledger.pending_derived_images() == []

    from skill3d.evolution.case_records_v11 import extract_case_image_refs
    from skill3d.storage.offline_frames import overview_ref

    file_ref = overview_ref(tmp_path, descriptor)
    refs = extract_case_image_refs(
        {"image_ledger": ledger.to_dict()}, image_refs=[file_ref])
    recorded = next(ref for ref in refs if ref.get("kind") == "scene_overview")
    assert recorded["rgb_sha256"] == descriptor["rgb_sha256"]
    assert refs[-1]["path"] == file_ref["path"]
    assert refs[-1]["image_path"] == descriptor["image_path"]


@pytest.mark.parametrize("quota", [0, 8, 32])
@pytest.mark.parametrize("capacity", [33, 40])
def test_overview_reserves_one_slot_without_using_crop_quota(tmp_path, quota, capacity):
    descriptor, _ = overview(tmp_path)
    ledger = make_ledger(max_images=capacity, max_derived=quota)
    legacy = make_ledger(max_derived=quota)
    ledger.register_scene_overview(descriptor)
    crop_ids = crops(ledger, 34)
    assert crops(legacy, 34) == crop_ids  # Overview does not change tool image IDs.
    initial = ledger.plan_round(
        round_index=1, trigger="initial", derived_image_ids=crop_ids,
        original_frame_ids=ledger.frame_ids)
    assert initial.image_ids == [descriptor["image_id"]] + legacy.frame_image_ids(legacy.frame_ids)
    assert not initial.omitted_originals
    ledger.mark_round_sent(1)
    ledger.mark_delivered(1, initial.image_ids)
    ledger.mark_round_observed(1)

    plan = ledger.plan_round(
        round_index=2, trigger="observation", derived_image_ids=crop_ids,
        original_frame_ids=ledger.frame_ids)
    old_plan = legacy.plan_round(
        round_index=2, trigger="observation", derived_image_ids=crop_ids,
        original_frame_ids=legacy.frame_ids)
    assert plan.image_ids == [descriptor["image_id"]] + old_plan.image_ids
    assert plan.derived_image_ids == [descriptor["image_id"]] + old_plan.derived_image_ids
    assert plan.omitted_derived == old_plan.omitted_derived
    assert plan.omitted_originals == old_plan.omitted_originals
    assert plan.scene_overview_image_ids == [descriptor["image_id"]]
    messages = build_image_messages(
        "", [ledger.pixels_of(i) for i in plan.original_image_ids],
        extra=[(i, ledger.pixels_of(i)) for i in plan.derived_image_ids], max_images=33)
    ledger.bind_request(2, messages)
    assert_actual_pack(ledger, plan, messages)


class Client:
    def __init__(self, replies, ledger=None):
        self.replies = list(replies)
        self.ledger = ledger
        self.calls = []
        self.contexts = []
        self.last_usage = {"prompt_tokens": 123}

    def set_request_context(self, context):
        self.contexts.append(copy.deepcopy(context))

    def chat(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), current_request_phase()))
        if self.ledger is not None:
            plan = self.ledger.rounds[-1]
            assert plan.delivered and not plan.observed
            for image_id in plan.image_ids:
                rec = self.ledger.get(image_id)
                assert plan.round_index in rec.delivered_rounds
                assert plan.round_index not in rec.observed_rounds
                assert plan.request_id in rec.sent_in_requests
                assert plan.request_id not in rec.response_completed_in_requests
                assert rec.present_in_current_request
            assert_actual_pack(self.ledger, plan, messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def setup_synthesis(tmp_path):
    item = load_synthetic_items(
        "inner_validation", question_types=["room_size_estimation"],
        n_frames=32, frame_size=(24, 32), seed=0)[0]
    descriptor, _ = overview(tmp_path, item.episode.scene_name)
    scene = SceneState(
        artifact_ref="test", scene_route="fallback_2d_only", objects=[],
        evidence_profile=EvidenceProfile(geometry_3d="unavailable", metric_scale="unavailable"))
    handle = SceneHandle(scene)
    handle._set_frames(list(range(32)), item.pixels)
    cfg = OnlineRunConfig(mode="real", scene_overview=descriptor, max_images=33)
    runner._configure_image_ledger(handle, cfg, item.episode)
    return item, scene, handle, cfg


def test_all_m8_phases_deliver_actual_overview_and_update_current_round(tmp_path):
    item, scene, handle, cfg = setup_synthesis(tmp_path)
    ledger = handle._ledger
    scene_before = scene.model_dump()
    frame_set_before = item.episode.frame_set.model_dump()
    final = json.dumps({"action": "answer", "value": 3, "basis": "visual_estimate",
                        "used_result_ids": [], "derivation": None})
    client = Client(["ReturnAnswer(3)"] * 3 + [final], ledger)
    crop_ids = []
    for index, trigger in enumerate(["initial", "recovery", "observation", "finalize"], 1):
        if index == 2:
            crop_ids = crops(ledger, 10)
        result = runner._synthesize(
            item.episode, scene, handle, [], cfg, client, pixels=item.pixels,
            observation_image_ids=crop_ids, round_index=index, trigger=trigger,
            prior_program="x = 1" if index == 2 else None,
            traceback_feedback="repair" if index == 2 else None)
        assert result.program is not None
        plan = ledger.rounds[-1]
        rec = ledger.get(cfg.scene_overview["image_id"])
        assert rec.delivered_rounds == rec.observed_rounds == list(range(1, index + 1))
        assert plan.observed and result.n_images == 33
        assert SCENE_OVERVIEW_GUIDANCE in result.prompt
        assert f"1. [Scene overview] image_id={rec.image_id}" in result.prompt
        assert cfg.scene_overview["image_path"] not in result.prompt
        context = client.contexts[-1]["images_in_order"][0]
        assert context["kind"] == "scene_overview"
        assert context["png_sha256"] == cfg.scene_overview["png_sha256"]
        assert "frame_slot" not in context and "crop_box" not in context
        assert ImageRound.model_validate(plan.model_dump()).model_dump() == plan.model_dump()
        for image in ledger.all_images():
            assert image.present_in_current_request == (image.image_id in plan.image_ids)
            model = SceneOverviewImage if image.kind == "scene_overview" else ProducedImage
            model.model_validate(image.model_dump())
    assert scene.model_dump() == scene_before
    assert item.episode.frame_set.model_dump() == frame_set_before
    _, baseline, _ = build_visual_request(
        question=item.episode.question, options=[], question_type=item.episode.question_type,
        frames=item.pixels, max_images=32)
    assert image_parts(client.calls[0][0])[1:] == image_parts(baseline)


def test_failed_response_never_marks_overview_observed(tmp_path):
    item, scene, handle, cfg = setup_synthesis(tmp_path)
    client = Client([TimeoutError("offline")], handle._ledger)
    result = runner._synthesize(
        item.episode, scene, handle, [], cfg, client, pixels=item.pixels)
    assert result.source == "online_model_service_error"
    rec = handle._ledger.get(cfg.scene_overview["image_id"])
    assert rec.delivered_rounds == [1] and rec.observed_rounds == []
    assert rec.sent_in_requests == ["image-request-1"]
    assert rec.response_completed_in_requests == []
    assert "Request failed" in rec.unobserved_reason


@pytest.mark.parametrize("reply", ["3", TimeoutError("offline")])
def test_fallback_observation_requires_completed_response(tmp_path, reply):
    item, _, handle, cfg = setup_synthesis(tmp_path)
    ledger = handle._ledger
    client = Client([reply], ledger)
    payload, audit = request_visual_answer(
        question=item.episode.question, options=[], question_type=item.episode.question_type,
        ledger=ledger, client=client, round_index=1, max_images=33, max_tokens=100, seed=0)
    completed = not isinstance(reply, Exception)
    assert audit["request_count"] == 1 and audit["accepted"] == completed
    assert (payload is not None) == completed
    rec = ledger.get(cfg.scene_overview["image_id"])
    assert rec.delivered_rounds == [1]
    assert rec.observed_rounds == ([1] if completed else [])
    assert ledger.rounds[0].observed == completed


@pytest.mark.parametrize("damage", ["missing", "changed"])
@pytest.mark.parametrize("entrypoint", ["episode", "synthesis", "fallback"])
def test_invalid_png_makes_zero_requests_even_after_decode_cache(tmp_path, damage, entrypoint):
    item, scene, handle, cfg = setup_synthesis(tmp_path)
    path = Path(cfg.scene_overview["image_path"])
    if damage == "missing":
        path.unlink()
    else:
        Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(path)
    client = Client([])
    if entrypoint == "fallback":
        payload, audit = request_visual_answer(
            question=item.episode.question, options=[], question_type=item.episode.question_type,
            ledger=handle._ledger, client=client, round_index=1,
            max_images=33, max_tokens=100, seed=0)
        assert payload is None and audit["request_count"] == 0
        assert audit["skipped_reason"] == "invalid_visual_input"
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            if entrypoint == "episode":
                runner.run_episode(item.episode, item.pixels, cfg, llm=client)
            else:
                runner._synthesize(
                    item.episode, scene, handle, [], cfg, client, pixels=item.pixels)
    assert client.calls == []
    assert handle._ledger.rounds == []


def test_scene_mismatch_and_insufficient_capacity_fail_closed(tmp_path):
    descriptor, _ = overview(tmp_path)
    with pytest.raises(ValueError, match="max_images"):
        OnlineRunConfig(scene_overview=descriptor)
    with pytest.raises(ValueError, match="max_images"):
        make_ledger().register_scene_overview(descriptor)
    with pytest.raises(ValueError, match="another scene"):
        make_ledger(max_images=33).register_scene_overview(descriptor, scene_id="other")


def test_disabled_models_and_direct32_request_keep_legacy_values(tmp_path):
    descriptor, _ = overview(tmp_path)
    ledger = make_ledger()
    assert OnlineRunConfig().scene_overview is None
    assert OnlineRunConfig().max_images == 32
    plan = ledger.plan_round(
        round_index=1, trigger="initial", derived_image_ids=[], original_frame_ids=ledger.frame_ids)
    assert plan.layout == "originals_only" and plan.n_images == 32
    assert "scene_overview_image_ids" not in plan.model_dump()
    assert "scene_overview" not in json.dumps(ledger.to_dict())
    prompt, messages, _ = build_visual_request(
        question="How many?", options=[], question_type="object_counting",
        frames=list(ledger.frames.values()), max_images=32)
    assert len(image_parts(messages)) == 32
    assert "overview" not in prompt and SCENE_OVERVIEW_GUIDANCE not in prompt
    assert descriptor["image_id"] not in json.dumps(messages)


def test_legacy_schema_dumps_have_exactly_the_previous_fields_and_defaults():
    expected_image = {
        "image_id": "frame-0", "kind": "frame", "produced_by": "", "result_id": "",
        "produced_round": 0, "source_frame_id": None, "source_frame_index": None,
        "box_xyxy": None, "source_hw": [], "sent_hw": [], "scale": None,
        "transform_version": "inspect_frames_v1", "content_sha256": "sha256:test",
        "delivered_rounds": [], "observed_rounds": [], "sent_in_requests": [],
        "response_completed_in_requests": [], "present_in_current_request": False,
        "unobserved_reason": "", "n_pixels_sent": 0, "token_estimate": 0,
        "token_estimate_source": "qwen_vl_patch_28",
    }
    expected_round = {
        "round_index": 0, "request_id": "", "request_sha256": "", "jpeg_sha256": [],
        "layout": "originals_only", "trigger": "", "image_ids": [],
        "original_image_ids": [], "derived_image_ids": [], "n_images": 0, "max_images": 0,
        "omitted_originals": [], "omitted_derived": [], "n_pixels_total": 0,
        "token_estimate_total": 0, "prompt_tokens": None, "delivered": False, "observed": False,
    }
    image = ProducedImage(image_id="frame-0", kind="frame", content_sha256="sha256:test")
    for mode in ("python", "json"):
        assert image.model_dump(mode=mode) == expected_image
        assert ImageRound().model_dump(mode=mode) == expected_round
    assert json.loads(image.model_dump_json()) == expected_image
    assert json.loads(ImageRound().model_dump_json()) == expected_round


def test_overview_schema_rejects_fake_frame_and_inconsistent_current_manifest(tmp_path):
    descriptor, _ = overview(tmp_path)
    ledger = make_ledger(max_images=33)
    rec = ledger.register_scene_overview(descriptor)
    row = rec.model_dump()
    for update in ({"source_frame_id": 0}, {"source_frame_index": 0},
                   {"box_xyxy": [0, 0, 32, 24]}, {"content_sha256": "wrong"}):
        with pytest.raises(ValueError):
            SceneOverviewImage.model_validate({**row, **update})
    with pytest.raises(ValueError):
        ProducedImage(image_id=rec.image_id, kind="scene_overview", content_sha256="sha256:x")
    plan = ledger.plan_round(
        round_index=1, trigger="initial", derived_image_ids=[], original_frame_ids=ledger.frame_ids)
    for update in ({"scene_overview_image_ids": []}, {"layout": "originals_only"},
                   {"image_ids": list(reversed(plan.image_ids))},
                   {"omitted_derived": [rec.image_id]}, {"max_images": 32}):
        with pytest.raises(ValueError):
            ImageRound.model_validate({**plan.model_dump(), **update})


def test_runner_repairs_finalizes_and_falls_back_with_overview(tmp_path):
    item = load_synthetic_items(
        "inner_validation", question_types=["room_size_estimation"],
        frame_size=(120, 160), n_frames=32, seed=0)[0]
    descriptor, _ = overview(tmp_path, item.episode.scene_name)
    artifact = _write_v6_artifact(tmp_path, with_poses=True)
    cfg = OnlineRunConfig(
        mode="real", reuse_artifact=artifact, scene_overview=descriptor, max_images=33,
        eval_visual_fallback=True, max_solver_rounds=3, finalization_rounds=1)
    client = Client([
        'img = inspect_frames([0])\nYieldObservations([img["result_id"]], "inspect")',
        "x = 1",
        json.dumps({"action": "answer", "value": -1, "basis": "visual_estimate",
                    "used_result_ids": [], "derivation": None}),
        "12.5",
    ])
    outcome = runner.run_episode(item.episode, item.pixels, cfg, llm=client)
    assert outcome.answer == "12.5" and outcome.eval_visual_fallback["accepted"]
    assert [phase for _, phase in client.calls] == [
        "program", "program", "finalization", "eval_visual_fallback"]
    first = image_parts(client.calls[0][0])
    assert first == image_parts(client.calls[-1][0])
    for messages, _ in client.calls:
        assert len(image_parts(messages)) == 33
        assert image_parts(messages)[0] == first[0]
        assert SCENE_OVERVIEW_GUIDANCE in messages[0]["content"][0]["text"]
    rounds = outcome.image_ledger["rounds"]
    assert all(row["scene_overview_image_ids"] == [descriptor["image_id"]] for row in rounds)
    assert len(rounds[0]["original_image_ids"]) == 32
    assert rounds[0]["omitted_originals"] == []
    assert len(rounds[1]["original_image_ids"]) == 31
    fallback = outcome.eval_visual_fallback["image_round"]
    assert fallback["n_images"] == 33 and fallback["omitted_originals"] == []
    assert fallback["derived_image_ids"] == [descriptor["image_id"]]
    assert len(fallback["jpeg_sha256"]) == 33 and fallback["request_sha256"]
    overview_row = next(row for row in outcome.image_ledger["images"]
                        if row["kind"] == "scene_overview")
    assert overview_row["observed_rounds"] == [1, 2, 3, 4]
    assert overview_row["delivered_rounds"] == [1, 2, 3, 4]
    assert overview_row["present_in_current_request"]
    assert outcome.used_result_ids == []
    assert outcome.answer_basis == "visual_estimate"
