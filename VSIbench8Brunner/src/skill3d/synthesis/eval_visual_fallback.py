"""One additional evaluation request with frozen images and the question.

The isolated choice policy uses only originals; the legacy policy may add an overview.
No episode, Skill, program, feedback or reference answer enters this API.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Sequence

from skill3d.schemas import AnswerPayload, CANONICAL_UNIT_BY_QUESTION_TYPE
from skill3d.synthesis.prompt_builder import build_image_messages, scene_overview_manifest_text
from skill3d.synthesis.request_context import request_phase
from skill3d.verifier.derivation import replay_derivation

EVAL_VISUAL_PROMPT_VERSION = "eval-visual-v1"
ISOLATED_CHOICE_PROMPT_VERSION = "eval-visual-choice-isolated-v2"
EVAL_VISUAL_PHASE = "eval_visual_fallback"
EVAL_VISUAL_SPLITS = frozenset({"inner_validation", "outer_holdout", "final_test"})
N_FRAMES = 32


def build_visual_prompt(question: str, options: Sequence[str], question_type: str,
                        *, isolated_choice: bool = False) -> str:
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE[question_type]
    if isolated_choice and unit != "option":
        raise ValueError("isolated choice policy requires an option question")
    text = "Answer the question using the 32 video frames in chronological order.\n"
    if isolated_choice:
        text += (
            "Use the observer position and facing direction specified in the question "
            "for spatial directions. Video frame order describes the camera recording; "
            "it is not itself a sequence of navigation actions. An object being outside "
            "one frame does not establish that it is behind the question's observer. "
            "For a nearest-object question, compare all listed candidates to the same "
            "reference object across the available views.\n")
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


def build_visual_request(
    *,
    question: str,
    options: Sequence[str],
    question_type: str,
    frames=None,
    encoded_frames: Sequence[dict] | None = None,
    max_images: int,
    isolated_choice: bool = False,
) -> tuple[str, list[dict], str]:
    """Build the frozen eval-visual request without sending it."""
    prompt = build_visual_prompt(question, options, question_type,
                                 isolated_choice=isolated_choice)
    if (frames is None) == (encoded_frames is None):
        raise ValueError("provide exactly one of frames or encoded_frames")
    if encoded_frames is None:
        encoded_frames = encode_visual_frames(frames, max_images=max_images)
    images = list(encoded_frames)
    if not images or len(images) > max_images:
        raise ValueError("encoded visual frame count is outside max_images")
    if isolated_choice and len(images) != N_FRAMES:
        raise ValueError("isolated choice requires exactly 32 original frames")
    if any(not isinstance(item, dict) or item.get("type") != "image_url"
           for item in images):
        raise ValueError("encoded visual frames must be image_url content items")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt}, *images]}]
    messages_sha256 = hashlib.sha256(json.dumps(
        messages, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()
    return prompt, messages, messages_sha256


def encode_visual_frames(frames, *, max_images: int) -> tuple[dict, ...]:
    """Encode one frozen scene once so same-scene questions reuse JPEG bytes."""
    messages = build_image_messages("", frames, max_images=max_images)
    return tuple(messages[0]["content"][1:])


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
                          max_tokens: int, seed: int,
                          enable_thinking: bool = False,
                          isolated_choice: bool = False,
                          trigger: str = "solver_run_error") -> tuple[AnswerPayload | None, dict]:
    audit = {
        "prompt_version": (ISOLATED_CHOICE_PROMPT_VERSION if isolated_choice
                           else EVAL_VISUAL_PROMPT_VERSION),
        "phase": EVAL_VISUAL_PHASE, "trigger": trigger,
        "isolated_choice": isolated_choice,
        "input_policy": ("32_original_rgb_only" if isolated_choice
                         else "frozen_frames_with_optional_overview"),
        "request_count": 0, "accepted": False, "round_index": round_index,
        "response_text": "", "service_error": False, "enable_thinking": enable_thinking,
    }
    if ledger is None or len(ledger.frame_ids) != N_FRAMES:
        return None, {**audit, "skipped_reason": "requires_32_readable_frozen_frames"}
    if max_images < N_FRAMES or ledger.max_images < N_FRAMES:
        return None, {**audit, "skipped_reason": "image_limit_below_32"}
    overview_id = None if isolated_choice else ledger.scene_overview_image_id
    if overview_id is not None and min(max_images, ledger.max_images) < N_FRAMES + 1:
        return None, {**audit, "skipped_reason": "image_limit_below_33"}
    if client is None:
        return None, {**audit, "skipped_reason": "client_unavailable"}
    try:
        ledger.ensure_frame_records()
        plan = ledger.plan_round(
            round_index=round_index, trigger=EVAL_VISUAL_PHASE,
            derived_image_ids=[], original_frame_ids=ledger.frame_ids,
            include_scene_overview=not isolated_choice)
        if overview_id is None:
            prompt, messages, messages_sha256 = build_visual_request(
                question=question, options=options, question_type=question_type,
                frames=[ledger.frame_pixels(fid) for fid in ledger.frame_ids],
                max_images=N_FRAMES if isolated_choice else max_images,
                isolated_choice=isolated_choice)
        else:
            prompt = build_visual_prompt(question, options, question_type)
            prompt += "\nImage 1: " + scene_overview_manifest_text(ledger.get(overview_id))
            prompt += "\nImages 2-33: the 32 original video frames in chronological order."
            messages = build_image_messages(
                prompt, [ledger.pixels_of(i) for i in plan.original_image_ids],
                extra=[(i, ledger.pixels_of(i)) for i in plan.derived_image_ids],
                max_images=max_images)
        ledger.bind_request(round_index, messages)
        messages_sha256 = plan.request_sha256
    except (KeyError, ValueError, OSError) as exc:
        return None, {**audit, "skipped_reason": "invalid_visual_input", "error": str(exc)}
    audit.update(
        prompt=prompt, n_images=plan.n_images, seed=seed, max_tokens=max_tokens,
        request_sha256=messages_sha256)
    ledger.mark_round_sent(round_index)
    ledger.mark_delivered(round_index, plan.image_ids)
    audit["request_count"] = 1
    try:
        with request_phase(EVAL_VISUAL_PHASE):
            text = client.chat(
                messages, max_tokens=max_tokens, seed=seed,
                **({"enable_thinking": True} if enable_thinking else {}))
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
