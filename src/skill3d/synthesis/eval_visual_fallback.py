"""One additional evaluation request with only frozen frames and the question.

No episode, Skill, scene, program, feedback or reference answer enters this API.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

from skill3d.schemas import AnswerPayload, CANONICAL_UNIT_BY_QUESTION_TYPE
from skill3d.synthesis.prompt_builder import build_image_messages
from skill3d.synthesis.request_context import request_phase
from skill3d.verifier.derivation import replay_derivation

EVAL_VISUAL_PROMPT_VERSION = "eval-visual-v1"
EVAL_VISUAL_PHASE = "eval_visual_fallback"
EVAL_VISUAL_SPLITS = frozenset({"inner_validation", "outer_holdout", "final_test"})
N_FRAMES = 32


def build_visual_prompt(question: str, options: Sequence[str], question_type: str) -> str:
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE[question_type]
    text = "Answer the question using the 32 video frames in chronological order.\n"
    text += f"Question: {question}\n"
    if unit == "option":
        if not 1 <= len(options) <= 4:
            raise ValueError("option question requires 1–4 options")
        # Keep option wording exactly as supplied, including existing A./B. labels.
        text += "Options:\n" + "\n".join(options) + "\n"
        text += f"Output only one option letter ({', '.join('ABCD'[:len(options)])})."
    else:
        template = {
            "count": "Output only a non-negative integer count.",
            "m": "Output only a non-negative number in meters.",
            "cm": "Output only a non-negative number in centimeters.",
            "m2": "Output only a non-negative number in square meters.",
        }[unit]
        text += template
    return text + " Do not include an explanation, units, or code."


def parse_visual_answer(text: str, *, question_type: str,
                        options: Sequence[str]) -> tuple[AnswerPayload, dict]:
    """Accept a single answer, never extract a guess from prose or executable code."""
    value = text.strip()
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE[question_type]
    if unit == "option":
        if value not in list("ABCD"[:len(options)]):
            raise ValueError("response must be one valid option letter")
    elif not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
                          value):
        raise ValueError("response must be a single number")
    payload = AnswerPayload(value=value, unit=unit, basis="visual_estimate")
    replay = replay_derivation(payload, [], question_type=question_type, options=options)
    if not replay.passed:
        raise ValueError("; ".join(issue.reason for issue in replay.issues))
    return payload, replay.model_dump(mode="json")


def request_visual_answer(*, question: str, options: Sequence[str], question_type: str,
                          ledger, client, round_index: int, max_images: int,
                          max_tokens: int, seed: int) -> tuple[AnswerPayload | None, dict]:
    audit = {
        "prompt_version": EVAL_VISUAL_PROMPT_VERSION, "phase": EVAL_VISUAL_PHASE,
        "request_count": 0, "accepted": False, "round_index": round_index,
        "response_text": "", "service_error": False,
    }
    if ledger is None or len(ledger.frame_ids) != N_FRAMES:
        return None, {**audit, "skipped_reason": "requires_32_readable_frozen_frames"}
    if max_images < N_FRAMES or ledger.max_images < N_FRAMES:
        return None, {**audit, "skipped_reason": "image_limit_below_32"}
    if client is None:
        return None, {**audit, "skipped_reason": "client_unavailable"}
    try:
        prompt = build_visual_prompt(question, options, question_type)
        messages = build_image_messages(
            prompt, [ledger.frame_pixels(fid) for fid in ledger.frame_ids],
            max_images=max_images)
    except (KeyError, ValueError) as exc:
        return None, {**audit, "skipped_reason": "invalid_visual_input", "error": str(exc)}
    ledger.ensure_frame_records()
    plan = ledger.plan_round(
        round_index=round_index, trigger=EVAL_VISUAL_PHASE,
        derived_image_ids=[], original_frame_ids=ledger.frame_ids)
    audit.update(
        prompt=prompt, n_images=N_FRAMES, seed=seed, max_tokens=max_tokens,
        request_sha256=hashlib.sha256(json.dumps(
            messages, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest())
    ledger.mark_round_sent(round_index)
    ledger.mark_delivered(round_index, plan.image_ids)
    audit["request_count"] = 1
    try:
        with request_phase(EVAL_VISUAL_PHASE):
            text = client.chat(messages, max_tokens=max_tokens, seed=seed)
    except Exception as exc:  # A single failed request is terminal; no retry.
        audit.update(service_error=True, error=f"{type(exc).__name__}: {exc}")
        ledger.mark_round_failed(round_index, audit["error"])
    else:
        audit["response_text"] = text
        audit["usage"] = dict(getattr(client, "last_usage", None) or {})
        ledger.mark_round_observed(round_index,
                                   prompt_tokens=audit["usage"].get("prompt_tokens"))
        try:
            payload, replay = parse_visual_answer(
                text, question_type=question_type, options=options)
        except (ValueError, TypeError, AttributeError) as exc:
            audit["error"] = f"invalid_answer: {exc}"
        else:
            audit.update(accepted=True, answer=payload.model_dump(mode="json"),
                         derivation_replay=replay, image_round=plan.model_dump(mode="json"))
            return payload, audit
    audit["image_round"] = plan.model_dump(mode="json")
    return None, audit
