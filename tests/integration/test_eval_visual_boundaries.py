"""Behavioral boundaries of the evaluation supplement, including real runner wiring."""

import copy
import json

import numpy as np
import pytest

from skill3d.adapters.episode_source import load_synthetic_items
from skill3d.online.runner import OnlineRunConfig, PreparedObjectBinding, run_episode
from skill3d.synthesis.eval_visual_fallback import parse_visual_answer, request_visual_answer
from skill3d.synthesis.request_context import current_request_phase
from skill3d.tools.image_ledger import ImageLedger
from skill3d.trace.store import TraceStore

from test_tool_contract_recovery import _write_v6_artifact


class PhaseClient:
    def __init__(self, program="import os\nReturnAnswer(1)\n", answer="1"):
        self.program, self.answer = program, answer
        self.calls = []
        self.last_usage = {"prompt_tokens": 123, "completion_tokens": 1}

    def chat(self, messages, **kwargs):
        phase = current_request_phase()
        self.calls.append((phase, copy.deepcopy(messages), kwargs))
        response = self.answer if phase == "eval_visual_fallback" else self.program
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(scope="module")
def visual_scene(tmp_path_factory):
    item = load_synthetic_items(
        "inner_validation", question_types=["room_size_estimation"],
        n_frames=32, frame_size=(120, 160), seed=0)[0]
    return item, _write_v6_artifact(tmp_path_factory.mktemp("visual-boundary"), with_poses=True)


def run_case(visual_scene, tmp_path, client, *, split="inner_validation",
             enabled=True, allow_final_test=False, max_regen=3):
    item, artifact = visual_scene
    episode = item.episode.model_copy(deep=True)
    episode.split = split
    episode.ground_truth = "1001"  # Must never be visible in either model request.
    cfg = OnlineRunConfig(
        mode="real", reuse_artifact=artifact, max_solver_rounds=2,
        finalization_rounds=1, max_regen=max_regen, max_images=32, seed=17,
        eval_visual_fallback=enabled, allow_final_test=allow_final_test,
        trace_dir=str(tmp_path / "trace"))
    out = run_episode(
        episode, item.pixels, cfg, llm=client,
        prepared_binding=PreparedObjectBinding(),
        trace_store=TraceStore(cfg.trace_dir))
    return out


@pytest.mark.parametrize(("split", "enabled", "allow_final", "expected"), [
    ("induction", True, False, False),
    ("inner_validation", False, False, False),
    ("inner_validation", True, False, True),
    ("outer_holdout", True, False, True),
    ("final_test", True, False, False),
    ("final_test", True, True, True),
])
def test_only_authorized_evaluation_splits_receive_one_supplement(
        visual_scene, tmp_path, split, enabled, allow_final, expected):
    client = PhaseClient()
    out = run_case(visual_scene, tmp_path, client, split=split,
                   enabled=enabled, allow_final_test=allow_final)
    visual = [c for c in client.calls if c[0] == "eval_visual_fallback"]
    assert len(visual) == int(expected)
    if expected:
        assert [c[0] for c in client.calls] == ["program", "program", "eval_visual_fallback"]
        assert out.answer == "1" and out.answer_basis == "visual_estimate"
        assert out.eval_visual_fallback["original_result"]["final_state"] == "run_error"
        assert out.eval_visual_fallback["original_result"]["terminal_failure"]["stage"] == "M9"
        assert out.failure_code is None
        assert out.attribution.ledger.verified_used_result_ids == []
        messages = visual[0][1]
        assert len(messages) == 1 and messages[0]["role"] == "user"
        parts = messages[0]["content"]
        assert sum(p["type"] == "image_url" for p in parts) == 32
        text = "\n".join(p["text"] for p in parts if p["type"] == "text")
        assert all(s not in text for s in ("ReturnAnswer", "1001", "traceback", "SKILL", "import os"))
        assert visual[0][2] == {"max_tokens": 4096, "seed": 17}
        saved = json.loads((tmp_path / "trace/episode_trace.jsonl").read_text().splitlines()[0])
        assert saved["eval_visual_fallback"]["accepted"]
        assert saved["answer_basis"] == "visual_estimate"
        assert out.receipts_ok
    else:
        assert not out.eval_visual_fallback


def test_accepted_but_wrong_answer_does_not_trigger_fallback(visual_scene, tmp_path):
    client = PhaseClient(program="ReturnAnswer(1)\n", answer="1001")
    out = run_case(visual_scene, tmp_path, client)
    assert out.final_state == "answer" and out.mra_value == 0.0
    assert [c[0] for c in client.calls] == ["program"]
    assert not out.eval_visual_fallback


def test_successful_supplement_keeps_original_flags_only_in_snapshot(
        visual_scene, tmp_path):
    out = run_case(
        visual_scene, tmp_path, PhaseClient(), max_regen=1)
    assert out.final_state == "answer" and out.answer == "1"
    assert "unanswerable" not in out.answer_flags
    assert "finalization" in out.eval_visual_fallback["original_result"]["answer_flags"]
    assert "eval_visual_fallback" in out.answer_flags


@pytest.mark.parametrize("response", ["NaN", "-1", "1 or 2", "ReturnAnswer(1)"])
def test_invalid_visual_answer_preserves_original_failure_without_retry(
        visual_scene, tmp_path, response):
    out = run_case(visual_scene, tmp_path, client := PhaseClient(answer=response))
    assert out.final_state == "run_error" and out.answer is None
    assert out.eval_visual_fallback["request_count"] == 1
    assert not out.eval_visual_fallback["accepted"]
    assert out.terminal_failure == out.eval_visual_fallback["original_result"]["terminal_failure"]
    assert len(client.calls) == 3


def test_visual_service_error_remains_unavailable(visual_scene, tmp_path):
    out = run_case(
        visual_scene, tmp_path, client := PhaseClient(answer=TimeoutError("visual offline")))
    assert out.final_state == "unavailable"
    assert out.eval_visual_fallback["service_error"]
    assert out.failure_code == "service_unavailable"
    assert out.terminal_failure["stage"] == "EVAL_VISUAL_FALLBACK"
    assert out.answer is None and out.mra_value is None
    assert len(client.calls) == 3


def test_program_service_error_does_not_start_an_extra_request(visual_scene, tmp_path):
    out = run_case(
        visual_scene, tmp_path, client := PhaseClient(program=TimeoutError("solver offline")))
    assert out.final_state == "unavailable"
    assert len(client.calls) == 1
    assert not out.eval_visual_fallback


@pytest.mark.parametrize(("question_type", "text"), [
    ("object_counting", "1.5"), ("object_counting", "true"),
    ("object_counting", "-1"), ("object_abs_distance", "1e999"),
    ("object_size_estimation", "NaN"), ("room_size_estimation", "12 m2"),
    ("object_rel_direction", "C"), ("object_rel_distance", "A or B"),
])
def test_visual_contract_rejects_invalid_answer_domains(question_type, text):
    with pytest.raises(ValueError):
        parse_visual_answer(text, question_type=question_type, options=["chair", "table"])


@pytest.mark.parametrize("n_frames,max_images", [(31, 32), (33, 40), (32, 31)])
def test_visual_request_does_not_pad_or_resample(n_frames, max_images):
    ledger = ImageLedger(max_images=max_images)
    ledger.set_frames(
        range(n_frames), [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n_frames)])
    client = PhaseClient()
    payload, audit = request_visual_answer(
        question="How many?", options=[], question_type="object_counting",
        ledger=ledger, client=client, round_index=3,
        max_images=max_images, max_tokens=128, seed=17)
    assert payload is None and audit["request_count"] == 0
    assert not client.calls and audit["skipped_reason"]


def test_visual_request_resends_all_originals_without_derived_images():
    ledger = ImageLedger(max_images=32)
    frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(32)]
    ledger.set_frames(range(32), frames)
    ledger.ensure_frame_records()
    crop = ledger.produce(
        frames[0], kind="crop", produced_by="inspect_frames", round_index=1,
        source_frame_id=0, box_xyxy=[0, 0, 4, 4])
    prior = ledger.plan_round(
        round_index=1, trigger="observation", derived_image_ids=[crop.image_id],
        original_frame_ids=ledger.frame_ids)
    ledger.mark_round_sent(1)
    ledger.mark_delivered(1, prior.image_ids)
    ledger.mark_round_observed(1)
    assert prior.omitted_originals
    payload, audit = request_visual_answer(
        question="How many?", options=[], question_type="object_counting",
        ledger=ledger, client=PhaseClient(), round_index=2,
        max_images=32, max_tokens=128, seed=17)
    assert payload is not None and audit["accepted"]
    latest = audit["image_round"]
    assert latest["n_images"] == 32 and latest["observed"]
    assert not latest["derived_image_ids"] and not latest["omitted_originals"]
    assert latest["original_image_ids"] == [f"frame-{i}" for i in range(32)]
    assert all(2 in ledger.get(f"frame-{i}").observed_rounds for i in range(32))
    assert ledger.get(crop.image_id).observed_rounds == [1]
