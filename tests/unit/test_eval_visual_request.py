import itertools
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from skill3d.synthesis.eval_visual_fallback import (
    build_visual_prompt, parse_visual_answer, request_visual_answer,
)
from skill3d.synthesis.request_context import current_request_phase, request_phase
from skill3d.synthesis.vllm_client import VLLMClient
from skill3d.tools.image_ledger import ImageLedger


@pytest.mark.parametrize(("qtype", "text", "unit"), [
    ("object_counting", "3", "count"),
    ("object_abs_distance", "1.25", "m"),
    ("object_size_estimation", "20", "cm"),
    ("room_size_estimation", "32.5", "m2"),
    ("object_rel_distance", "B", "option"),
    ("object_rel_direction", "A", "option"),
    ("route_planning", "A", "option"),
    ("obj_appearance_order", "B", "option"),
])
def test_visual_answer_contract(qtype, text, unit):
    payload, replay = parse_visual_answer(text, question_type=qtype, options=["left", "right"])
    assert payload.value == text and payload.unit == unit
    assert payload.basis == "visual_estimate"
    assert payload.used_result_ids == [] and payload.derivation is None
    assert replay["passed"] and not replay["performed"]


@pytest.mark.parametrize(("qtype", "text"), [
    ("object_rel_distance", "C"), ("object_rel_distance", "A or B"),
    ("object_rel_distance", "The answer is A."),
    ("object_rel_distance", "```python\nReturnAnswer('A')\n```"),
    ("room_size_estimation", "-2"), ("room_size_estimation", "NaN"),
    ("room_size_estimation", "inf"), ("room_size_estimation", "1e999"),
    ("room_size_estimation", "12 m2"), ("room_size_estimation", ""),
    ("object_counting", "1.5"), ("object_counting", "True"),
])
def test_invalid_visual_output_is_not_salvaged(qtype, text):
    with pytest.raises(ValueError):
        parse_visual_answer(text, question_type=qtype, options=["chair", "table"])


@pytest.mark.parametrize(("n_frames", "max_images", "reason"), [
    (0, 32, "requires_32_readable_frozen_frames"),
    (31, 32, "requires_32_readable_frozen_frames"),
    (33, 32, "requires_32_readable_frozen_frames"),
    (32, 31, "image_limit_below_32"),
])
def test_visual_request_never_pads_or_truncates(n_frames, max_images, reason):
    ledger = ImageLedger(max_images=max_images)
    ledger.set_frames(list(range(n_frames)), [np.zeros((4, 4, 3), np.uint8)] * n_frames)
    client = Mock()
    payload, audit = request_visual_answer(
        question="How many?", options=[], question_type="object_counting",
        ledger=ledger, client=client, round_index=3, max_images=max_images,
        max_tokens=100, seed=7)
    assert payload is None and audit["skipped_reason"] == reason
    assert audit["request_count"] == 0
    client.chat.assert_not_called()
    assert ledger.rounds == []


def test_output_templates_use_only_question_options_and_units():
    for task, phrase in [
        ("object_counting", "integer count"), ("object_abs_distance", "meters"),
        ("object_size_estimation", "centimeters"),
        ("room_size_estimation", "square meters"),
    ]:
        assert phrase in build_visual_prompt("query", [], task)
    prompt = build_visual_prompt("query", ["A. chair", "B. table"], "object_rel_distance")
    assert "A. chair\nB. table" in prompt
    assert "AnswerPayload" not in prompt and "Skill" not in prompt


def test_single_visual_request_disables_sdk_retries_and_restores_phase():
    sdk = Mock()
    one_shot = sdk.with_options.return_value
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="3"))], usage=None)
    sdk.chat.completions.create.return_value = response
    one_shot.chat.completions.create.return_value = response
    client = VLLMClient.__new__(VLLMClient)
    client._clients, client._rr, client.model = [sdk], itertools.cycle([0]), "fixture"
    messages = [{"role": "user", "content": "query"}]
    with request_phase("program"):
        assert client.chat(messages, seed=7) == "3"
        sdk.with_options.assert_not_called()
        with request_phase("eval_visual_fallback"):
            assert client.chat(messages, seed=7) == "3"
        assert current_request_phase() == "program"
    assert current_request_phase() == "perception"
    sdk.with_options.assert_called_once_with(max_retries=0)
    assert sdk.chat.completions.create.call_count == 1
    assert one_shot.chat.completions.create.call_count == 1
