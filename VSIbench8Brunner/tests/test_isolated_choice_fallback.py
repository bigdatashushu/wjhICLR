"""Offline regressions through the real solver, sandbox, verifier and trace writer."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online import runner, synthetic
from skill3d.online.config import load_config
from skill3d.online.recovery import ValidatedObservation, observation_feedback
from skill3d.online.runner import OnlineRunConfig
from skill3d.reconstruction.metric_fusion import write_per_frame_receipt
from skill3d.reconstruction_gate.quality_metrics import compute_quality
from skill3d.schemas import ConfidenceMap, ReconstructionArtifact
from skill3d.synthesis.client_errors import ServiceUnavailable
from skill3d.synthesis.eval_visual_fallback import (
    ISOLATED_CHOICE_PROMPT_VERSION, build_visual_request, request_visual_answer,
)
from skill3d.synthesis.request_context import current_request_phase
from skill3d.tools.image_ledger import ImageLedger
from skill3d.trace.store import TraceStore

ROOT = Path(__file__).resolve().parents[1]


class Client:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.last_usage = {"prompt_tokens": 123}

    def chat(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), current_request_phase(), kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    item = load_synthetic_items(
        "inner_validation", question_types=["obj_appearance_order"],
        frame_size=(120, 160), n_frames=32, seed=0)[0]
    item.episode.options = ["A. chair", "B. lamp"]
    item.episode.ground_truth = "A"
    geometry, frames = item.geometry, item.episode.frame_set
    root = tmp_path_factory.mktemp("isolated-choice")
    paths = {}
    for name, array in (
        ("depth", geometry.depth_maps), ("pm", geometry.point_map),
        ("intrinsics", geometry.intrinsics), ("conf", geometry.depth_conf),
        ("poses", geometry.c2w),
    ):
        paths[name] = str(root / f"{name}.npy")
        np.save(paths[name], array)
    quality = compute_quality(
        None, frames=item.pixels, depth_maps=geometry.depth_maps,
        c2w_list=geometry.c2w, intrinsics=geometry.intrinsics,
        point_map=geometry.point_map, depth_conf=geometry.depth_conf)
    assert quality.main_gate_passed
    fusion = synthetic._synthetic_metric_fusion(geometry)
    scale = write_per_frame_receipt(fusion, root / "scale.json")
    artifact = ReconstructionArtifact(
        artifact_id="isolated-choice-artifact", artifact_version="v1",
        scene_name=item.episode.scene_name, frame_ids=frames.frame_ids,
        source_frame_indices=frames.source_frame_indices,
        timestamps=frames.timestamps, frame_set_hash=frames.frame_set_hash,
        c2w_list=paths["poses"], intrinsics=paths["intrinsics"],
        depth_maps=paths["depth"], point_map=paths["pm"], point_conf="",
        depth_conf=paths["conf"], world_up=geometry.world_up,
        handedness=geometry.handedness, world_frame_status=geometry.world_frame_status,
        metric_scale=float(fusion.metric_scale),
        scale_self_consistency=fusion.scale_self_consistency,
        per_frame_scale_ref=str(scale), metric_model="none",
        metric_fusion_version=fusion.version, scale_fusion_status="success",
        quality_status="computed", quality=quality,
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""),
    )
    path = root / "artifact.json"
    path.write_text(artifact.model_dump_json(indent=2))
    return item, str(path)


@pytest.fixture(autouse=True)
def offline_binding(monkeypatch):
    # M5 perception has no bearing on routing; never contact a detector or audit service.
    monkeypatch.setattr(runner, "_bind_objects_best_effort",
                        lambda *args, **kwargs: ([], {}, [], False))


def visual_program(answer):
    return (f"ReturnAnswer(AnswerPayload(value={answer!r}, unit='option', "
            "basis='visual_estimate', used_result_ids=[], derivation=None))")


def run_case(source, tmp_path, replies, *, reference="A", split="inner_validation",
             role=None, isolated=True, legacy=True, overview=None,
             question_type="obj_appearance_order"):
    item, artifact = source
    episode = item.episode.model_copy(deep=True, update={
        "ground_truth": reference, "split": split, "question_type": question_type})
    cfg = OnlineRunConfig(
        mode="real", reuse_artifact=artifact, seed=7,
        max_solver_rounds=2, finalization_rounds=1,
        eval_visual_fallback=legacy, isolated_choice_visual_fallback=isolated,
        evaluation_execution_role=role, scene_overview=overview,
        max_images=33 if overview else 32, online_enable_thinking=True,
        allow_final_test=split == "final_test", trace_dir=str(tmp_path / "traces"))
    client = Client(replies)
    outcome = runner.run_episode(
        episode, item.pixels, cfg, llm=client, trace_store=TraceStore(cfg.trace_dir))
    return outcome, client


@pytest.mark.parametrize("reference", ["A", "B", ""])
@pytest.mark.parametrize("task", [
    "obj_appearance_order", "object_rel_direction_hard",
    "object_rel_distance", "route_planning",
])
def test_accepted_visual_answer_is_replaced_without_reference_selection(
        source, tmp_path, reference, task):
    out, client = run_case(source, tmp_path, [visual_program("A"), "B"],
                           reference=reference, question_type=task)
    assert out.answer == out.predicted == "B"
    assert out.answer_basis == "visual_estimate" and out.answer_source == "direct_vlm_routed"
    assert out.legal_answer and out.submission_status == "accepted"
    assert out.agent_rounds == len(out.rounds) == 1
    assert [call[1] for call in client.calls] == ["program", "eval_visual_fallback"]
    assert client.calls[-1][2] == {
        "max_tokens": 8192, "seed": 7, "enable_thinking": True}
    audit = out.eval_visual_fallback
    assert audit["trigger"] == "accepted_visual_estimate"
    assert audit["isolated_choice"] and audit["request_count"] == 1 and audit["accepted"]
    assert audit["prompt_version"] == ISOLATED_CHOICE_PROMPT_VERSION
    assert audit["input_policy"] == "32_original_rgb_only"
    original = audit["original_result"]
    assert original["answer"] == "A" and original["submission_status"] == "accepted"
    assert original["answer_payload"]["basis"] == original["answer_basis"] == "visual_estimate"
    assert original["geometry_verify"]["passed"]
    assert out.episode_trace.eval_visual_fallback == audit
    assert out.episode_trace.budget["eval_visual_fallback_max_requests"] == 1
    assert out.verify.check_status["answer_contract"] == "pass" and out.receipts_ok
    messages = client.calls[-1][0]
    assert len(messages) == 1
    assert messages[0]["content"][1:] == client.calls[0][0][0]["content"][1:]
    for forbidden in ("Still-valid", "ReturnAnswer", "Scene summary", "previous program"):
        assert forbidden not in messages[0]["content"][0]["text"]
    if reference:
        assert original["score"] == float(reference == "A")
        assert out.score == float(reference == "B")
    else:
        assert out.score is original["score"] is None
        assert out.scoring_status == "reference_unavailable"


def test_reference_changes_do_not_change_fresh_request(source, tmp_path):
    calls = []
    for ref in ("A", "B", ""):
        _, client = run_case(source, tmp_path / (ref or "unlabelled"),
                             [visual_program("A"), "B"], reference=ref)
        calls.append(client.calls[-1])
    assert calls[0] == calls[1] == calls[2]


@pytest.mark.parametrize("basis", ["tool_derived", "mixed"])
def test_verified_tool_and_mixed_answers_keep_original_route(source, tmp_path, basis):
    # The allowed frame tool keeps a real result ID and a replayable option mapping.
    program = (
        "r = inspect_frames([0])\n"
        f"ReturnAnswer(AnswerPayload(value='A', unit='option', basis={basis!r}, "
        "used_result_ids=[r['result_id']], derivation={"
        "'op': 'option_map', 'input_result_ids': [r['result_id']], "
        "'parameters': {'field': 'n_images', 'mapping': {'1': 'A'}}}))")
    out, client = run_case(source, tmp_path, [program], reference="B")
    assert out.answer == "A" and out.score == 0
    assert out.verify.passed and out.answer_basis == basis
    assert out.attribution.ledger.verified_used_result_ids
    assert len(client.calls) == 1 and not out.eval_visual_fallback


@pytest.mark.parametrize(("split", "role", "enabled"), [
    ("induction", None, True),
    ("inner_validation", "learning", True),
    ("inner_validation", None, False),
])
def test_learning_and_disabled_policy_keep_accepted_answer(
        source, tmp_path, split, role, enabled):
    out, client = run_case(source, tmp_path, [visual_program("A")],
                           split=split, role=role, isolated=enabled)
    assert out.answer == "A" and len(client.calls) == 1
    assert not out.eval_visual_fallback


def test_validation_role_keeps_source_split_and_uses_shared_single_allowance(source, tmp_path):
    out, client = run_case(source, tmp_path, [visual_program("A"), "B"],
                           split="induction", role="validation", legacy=False)
    assert out.answer == "B" and len(client.calls) == 2
    assert out.episode_trace.budget["eval_visual_fallback_max_requests"] == 1
    start = next(receipt for receipt in out.receipts if receipt.event == "episode_start")
    expected = {"qa_id": source[0].episode.qa_id, "mode": "real",
                "baseline": "C1_tools_program", "split": "induction"}
    assert start.payload_digest == hashlib.sha256(
        json.dumps(expected, sort_keys=True, default=str).encode()).hexdigest()


@pytest.mark.parametrize("reply", ["answer B", "ReturnAnswer('B')", "Z",
                                  ServiceUnavailable("offline")])
def test_failed_replacement_never_restores_original_and_never_retries(
        source, tmp_path, reply):
    out, client = run_case(source, tmp_path, [visual_program("A"), reply])
    assert len(client.calls) == 2 and out.answer is None and out.predicted is None
    assert out.answer_payload is None and out.attribution is None and out.answer_basis == ""
    assert not out.legal_answer and out.submission_status == "rejected"
    assert out.eval_visual_fallback["original_result"]["answer"] == "A"
    assert not out.eval_visual_fallback["accepted"]
    assert out.used_result_ids == [] and out.verify is None
    assert out.receipts_ok
    if isinstance(reply, Exception):
        assert out.final_state == "unavailable" and out.score is None
        assert out.failure_code == "service_unavailable"
    else:
        assert out.final_state == "run_error" and out.score == 0
        assert out.failure_code == "isolated_visual_answer_rejected"


def test_initial_service_failure_never_triggers_extra_request(source, tmp_path):
    out, client = run_case(source, tmp_path, [ServiceUnavailable("offline")])
    assert out.final_state == "unavailable" and len(client.calls) == 1
    assert not out.eval_visual_fallback


def test_answer_slot_and_user_variables_are_cleared_before_fresh_call(
        source, tmp_path, monkeypatch):
    held = {}
    solve = runner._solve_program
    chat = Client.chat
    def capture_kernel(*args, **kwargs):
        result = solve(*args, **kwargs)
        held["kernel"] = result[3]
        return result
    def check_chat(client, messages, **kwargs):
        if current_request_phase() == "eval_visual_fallback":
            kernel = held["kernel"]
            assert kernel.answer_slot.answer is kernel.answer_slot.payload is None
            assert not kernel.answer_slot.given
            assert "history_marker" not in kernel._ns
        return chat(client, messages, **kwargs)
    monkeypatch.setattr(runner, "_solve_program", capture_kernel)
    monkeypatch.setattr(Client, "chat", check_chat)
    out, client = run_case(
        source, tmp_path, ["history_marker = 'old choice'\n" + visual_program("A"), "B"])
    assert out.answer == "B" and len(client.calls) == 2


def test_skipped_replacement_does_not_resurrect_accepted_answer(
        source, tmp_path, monkeypatch):
    def missing_frame(**kwargs):
        kwargs["ledger"].frames.pop(31)
        return request_visual_answer(**kwargs)
    monkeypatch.setattr(runner, "request_visual_answer", missing_frame)
    out, client = run_case(source, tmp_path, [visual_program("A")])
    assert len(client.calls) == 1 and out.answer is None
    assert out.final_state == "run_error" and out.submission_status == "rejected"
    assert out.eval_visual_fallback["request_count"] == 0
    assert out.eval_visual_fallback["original_result"]["answer"] == "A"
    assert out.terminal_failure["termination_reason"] == "requires_32_readable_frozen_frames"


def overview_descriptor(root, scene):
    pixels = np.full((480, 640, 3), (31, 127, 211), dtype=np.uint8)
    path = root / "overview.png"
    Image.fromarray(pixels).save(path)
    rgb_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
    return {
        "schema": "harness3d-scene-overview/1", "scene_id": scene,
        "image_id": f"overview-{scene}-{rgb_sha[:12]}", "image_path": str(path),
        "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rgb_sha256": rgb_sha,
        "scene_artifact_sha256": "a" * 64, "frozen_scene_sha256": "b" * 64,
        "render_spec_sha256": "c" * 64, "world_frame_status": "degraded",
        "resolution_wh": [640, 480], "selected_azimuth_degrees": 45.0,
        "elevation_degrees": 60.0, "render_only_cutaway": True,
    }


def test_overview_and_yielded_crop_do_not_enter_fresh_request(source, tmp_path):
    descriptor = overview_descriptor(tmp_path, source[0].episode.scene_name)
    out, client = run_case(source, tmp_path, [
        'img = inspect_frames([0])\nYieldObservations([img["result_id"]], "inspect")',
        json.dumps({"action": "answer", "value": "A", "basis": "visual_estimate",
                    "used_result_ids": [], "derivation": None}),
        "B"], overview=descriptor)
    assert len(client.calls) == 3 and out.answer == "B"
    assert "crop_box" in client.calls[1][0][0]["content"][0]["text"]
    clean = client.calls[-1][0][0]["content"]
    assert clean[1:] == client.calls[0][0][0]["content"][2:]
    assert len(clean) == 33
    plan = out.eval_visual_fallback["image_round"]
    assert plan["n_images"] == 32 and plan["layout"] == "originals_only"
    assert plan["derived_image_ids"] == [] and plan.get("scene_overview_image_ids", []) == []
    assert len(plan["jpeg_sha256"]) == 32
    assert plan["request_sha256"] == out.eval_visual_fallback["request_sha256"]
    assert plan["delivered"] and plan["observed"]
    assert out.attribution.ledger.verified_used_result_ids == []
    assert out.attribution.ledger.ignored_result_ids


def test_failed_solver_uses_only_one_shared_visual_request(source, tmp_path):
    out, client = run_case(source, tmp_path, ["x = 1", "x = 1", "B"])
    assert out.answer == "B" and len(client.calls) == 3
    assert out.agent_rounds == 2
    assert out.eval_visual_fallback["trigger"] == "solver_run_error"
    assert out.eval_visual_fallback["isolated_choice"]
    assert out.episode_trace.budget["eval_visual_fallback_max_requests"] == 1


def test_missing_frozen_frame_skips_without_a_request(tmp_path):
    ledger = ImageLedger()
    ledger.set_frames(range(31), [np.zeros((24, 32, 3), np.uint8)] * 31)
    client = Client([])
    answer, audit = request_visual_answer(
        question="Which?", options=["A. chair", "B. lamp"],
        question_type="obj_appearance_order", ledger=ledger, client=client,
        round_index=1, max_images=32, max_tokens=8192, seed=7, isolated_choice=True)
    assert answer is None and not client.calls and audit["request_count"] == 0
    assert audit["skipped_reason"] == "requires_32_readable_frozen_frames"


@pytest.mark.parametrize(("status", "winner"), [
    ("incomplete_candidates", None), ("partial", None), ("ok", None), ("tied", "chair")])
def test_indeterminate_rank_warning_preserves_raw_local_values(status, winner):
    payload = {"status": status, "closest_category": winner,
               "categories_without_detection": ["plant"],
               "ranking": [{"category": "monitor", "distance": 0.42}]}
    original = copy.deepcopy(payload)
    feedback = observation_feedback([
        ValidatedObservation("rank-1", "relative_distance_rank", payload)])
    assert "Ranking scope: indeterminate" in feedback
    assert "do not select the nearest measurable candidate" in feedback
    assert '"distance": 0.42' in feedback and status in feedback
    assert payload == original
    complete = ValidatedObservation(
        "rank-2", "relative_distance_rank", {"status": "ok", "closest_category": "chair"})
    assert "Ranking scope: indeterminate" not in complete.summary()


def test_yaml_and_fitset_entry_enable_policy_and_reject_truthy_strings(tmp_path):
    raw = load_config(ROOT / "runner.yaml")
    spec = importlib.util.spec_from_file_location(
        "_choice_test_fitset", ROOT / "scripts/run_fitset_v11.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = module._base_config(raw)
    assert cfg.isolated_choice_visual_fallback and cfg.max_solver_rounds == 6
    path = tmp_path / "invalid.yaml"
    path.write_text('isolated_choice_visual_fallback: "false"\n')
    with pytest.raises(ValueError, match="isolated_choice_visual_fallback must be boolean"):
        load_config(path)
    with pytest.raises(ValueError, match="isolated_choice_visual_fallback must be boolean"):
        OnlineRunConfig(isolated_choice_visual_fallback="false")


def test_old_pilot_runtime_is_rejected_before_service_preflight(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_choice_test_pilot", ROOT / "scripts/pilot40.py")
    pilot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pilot
    spec.loader.exec_module(pilot)
    cfg = OnlineRunConfig(
        isolated_choice_visual_fallback=True, evaluation_execution_role="validation")
    monkeypatch.setattr(pilot, "base_config", lambda: (cfg, "snapshot"))
    monkeypatch.setattr(pilot, "validate_panel", lambda *args: None)
    def forbid_service_setup(*args, **kwargs):
        raise AssertionError("Old policy must be rejected before provider or service setup")
    monkeypatch.setattr(pilot, "PublishedFitSetInputProvider", forbid_service_setup)
    sampling = tmp_path / "sampling.json"
    sampling.write_text("{}")
    (tmp_path / "runtime_identity.json").write_text(json.dumps({
        "schema": "vsi8b-pilot-runtime/1", "solver_enable_thinking": True}))
    args = SimpleNamespace(run_dir=tmp_path, sampling=sampling, expected_questions=40)
    with pytest.raises(ValueError, match="pilot solver policy changed"):
        pilot.initialize(args)


@pytest.mark.parametrize("mutation", [None, "rounds", "timeout"])
def test_public_startup_check_records_only_declared_deltas(tmp_path, monkeypatch, mutation):
    spec = importlib.util.spec_from_file_location("_choice_test_entry", ROOT / "run.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    # Full public shape, no geometry or labels: every split covers all 164 scenes.
    rows = [{"qa_id": str(i), "scene_name": f"scene-{i % 164}"} for i in range(1000)]
    for name, subset in (("questions", rows), ("train", rows[:300]), ("test", rows[300:])):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in subset))
    (tmp_path / "split.json").write_text(json.dumps({
        "questions_sha256": entry.file_hash(tmp_path / "questions.jsonl"),
        "files": {role: {"sha256": entry.file_hash(tmp_path / f"{role}.jsonl")}
                  for role in ("train", "test")},
    }))
    (tmp_path / "published_inputs.json").write_text(json.dumps({
        "entries": {row["qa_id"]: {"input_root": str(tmp_path)} for row in rows},
        "scene_scopes": {"all": {"qa_ids": [row["qa_id"] for row in rows]}},
    }))
    current = load_config(ROOT / "runner.yaml")
    original = copy.deepcopy(current)
    original["online_model"]["mask_review"].pop("target_description")
    original["online_model"]["enable_thinking"] = False
    original["online_model"]["timeout_s"] = 180.0
    original["isolated_choice_visual_fallback"] = False
    if mutation == "rounds":
        current["max_solver_rounds"] = 7
    elif mutation == "timeout":
        current["online_model"]["timeout_s"] = 901.0
    import skill3d.online.config as config_module
    monkeypatch.setattr(config_module, "load_config", lambda path: copy.deepcopy(
        original if Path(path).name == "online.yaml" else current))
    if mutation:
        with pytest.raises(ValueError, match="runner"):
            entry.check_inputs(tmp_path, ROOT / "runner.yaml")
    else:
        receipt = entry.check_inputs(tmp_path, ROOT / "runner.yaml")
        assert receipt["model_requests"] == 0
        assert receipt["questions"] == 1000
        assert receipt["isolated_choice_visual_fallback"]
        assert receipt["isolated_choice_prompt_version"] == ISOLATED_CHOICE_PROMPT_VERSION
        assert receipt["eval_visual_fallback_max_requests"] == 1
        assert receipt["solver_timeout_s"] == 900.0
        assert receipt["original_solver_timeout_s"] == 180.0


def test_direct_baseline_prompt_stays_legacy():
    frames = [np.zeros((24, 32, 3), np.uint8)] * 32
    default = build_visual_request(
        question="Which?", options=["A. chair", "B. lamp"],
        question_type="obj_appearance_order", frames=frames, max_images=32)
    legacy = build_visual_request(
        question="Which?", options=["A. chair", "B. lamp"],
        question_type="obj_appearance_order", frames=frames, max_images=32,
        isolated_choice=False)
    assert default == legacy and "observer position" not in default[0]
