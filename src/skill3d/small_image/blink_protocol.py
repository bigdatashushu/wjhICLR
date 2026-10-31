"""BLINK request/answer protocol shared by the system and direct baseline."""
from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re

from PIL import Image

VERSION = "blink-original-thinking/1"
ANSWER_PARSER = "blink-explicit-answer/1"
DIRECT_SUFFIX = (
    '\n\nUse the two supplied images in their original order. '
    'Give a concise explanation based on their visual differences, then select '
    'one option. Return only a valid JSON object with the fields '
    '"reasoning" (a brief explanation) and "answer" (one option letter, A or B).'
)


def sha_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def original_messages(problem, text):
    """Keep original image bytes, MIME type and order; revalidate the input identity."""
    if replace(problem).input_sha256 != problem.input_sha256:
        raise ValueError("original_input_changed")
    content = [{"type": "text", "text": text}]
    for index, path in enumerate(problem.image_paths, 1):
        raw = Path(path).read_bytes()
        with Image.open(BytesIO(raw)) as image:
            mime = Image.MIME.get(image.format)
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("unsupported_original_image_format")
        content.extend([
            {"type": "text", "text": f"Image {index}:"},
            {"type": "image_url", "image_url": {
                "url": f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")}},
        ])
    return [{"role": "user", "content": content}]


def image_identities(problem):
    records = []
    for index, path in enumerate(problem.image_paths):
        raw = Path(path).read_bytes()
        with Image.open(BytesIO(raw)) as image:
            records.append({"view_index": index, "sha256": hashlib.sha256(raw).hexdigest(),
                            "width": image.width, "height": image.height,
                            "mime_type": Image.MIME.get(image.format)})
    return records


def benchmark_prompt(problem):
    # The legacy loader split this exact official prompt before "(A) left".
    if problem.benchmark != "blink" or problem.options != ("left", "right"):
        raise ValueError("unexpected_blink_options")
    return problem.question + "\n(A) left\n(B) right"


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite_json")
    return json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)


def explicit_answer(text, options):
    """Parse only explicit answers, without reading labels or executing model text."""
    from .runner import parse_small_image_answer
    if not isinstance(text, str) or not text.strip() or len(text) > 128000:
        raise ValueError("missing_or_oversized_answer")
    clean = text.strip()

    def object_answer(value):
        if (not isinstance(value, dict) or "answer" not in value
                or set(value) - {"answer", "reasoning"}):
            raise ValueError("invalid_answer_object")
        return parse_small_image_answer(value["answer"], options)

    if clean.startswith("{"):
        try:
            return object_answer(strict_json(clean)), "json"
        except json.JSONDecodeError:
            pass
    try:
        return parse_small_image_answer(clean, options), "option"
    except ValueError:
        pass
    blocks = list(re.finditer(r"```(?:json)?\s*([\s\S]*?)```", clean, re.IGNORECASE))
    if blocks and blocks[-1].end() == len(clean):
        return object_answer(strict_json(blocks[-1][1].strip())), "final_fenced_json"
    answers = []
    for match in re.finditer(r"(?m)(?<!\S)\{", clean):
        try:
            answers.append(object_answer(strict_json(clean[match.start():])))
        except (ValueError, TypeError):
            continue
    if len(answers) == 1:
        return answers[0], "final_json_after_prose"
    last = re.sub(r"[*`]", "", clean.splitlines()[-1]).strip()
    last = re.sub(r"^(?:final\s+)?answer\s*:\s*", "", last, flags=re.IGNORECASE)
    try:
        return parse_small_image_answer(last, options), "final_option_line"
    except ValueError:
        pass
    labels = set()
    for statement in re.finditer(
            r"^(?:therefore,\s*)?(?:the\s+)?(?:final\s+)?answer\s*(?::|is)\s*(.+?)\s*$",
            clean, re.IGNORECASE | re.MULTILINE):
        value = statement[1].strip().rstrip(".!").strip("*` ")
        try:
            labels.add(parse_small_image_answer(value, options))
        except ValueError:
            continue
    if len(labels) == 1:
        return labels.pop(), "explicit_answer_statement"
    raise ValueError("unparsed_explicit_answer")


def response_record(client, text):
    """Full model output stays in the episode record, outside the budget ledger."""
    reason = getattr(client, "last_reasoning_content", None)
    record = {
        "model_response": text, "reasoning_content": reason,
        "usage": dict(getattr(client, "last_usage", {}) or {}),
        "response_identity": dict(getattr(client, "last_response_identity", {}) or {}),
        "request_identity": dict(getattr(client, "last_request_identity", {}) or {}),
        "finish_reason": (getattr(client, "last_generation", {}) or {}).get("finish_reason"),
    }
    secret = os.environ.get(getattr(client, "api_key_env", ""), "")
    if secret and secret in json.dumps(record, ensure_ascii=False):
        raise ValueError("credential_in_response")
    return record
