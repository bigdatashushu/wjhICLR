"""Versioned official prompt, deterministic answer extraction and Responses body."""
from __future__ import annotations

import base64
import re

from xcodeapi.content import convert_messages

PROMPT_VERSION = "mmsi-official-direct-table7-v1"
PARSER_VERSION = "single-option-letter-v1"
DIRECT_INSTRUCTION = (
    "Answer with the option's letter from the given choices directly. "
    "Enclose the option's letter within ``."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def prompt(problem):
    # VLMEvalKit MMSIBench.build_prompt: question, original choices, instruction.
    choices = "".join(f"{LETTERS[i]}: {option}, " for i, option in enumerate(problem["options"]))
    return problem["question"] + "\nOptions: " + choices + "\n" + DIRECT_INSTRUCTION


def messages(problem, images):
    """Images precede text, without resizing, annotations or image reordering."""
    content = [
        {"type": "image_url", "image_url": {
            "url": f"data:{receipt['mime_type']};base64," + base64.b64encode(raw).decode("ascii")}}
        for receipt, raw in images
    ]
    content.append({"type": "text", "text": prompt(problem)})
    return [{"role": "user", "content": content}]


def payload(messages_value, inference):
    converted, system = convert_messages(messages_value, "responses")
    if system:
        raise ValueError("direct_has_no_system_prompt")
    return {
        "model": inference["model"], "input": converted, "stream": False, "store": False,
        "max_output_tokens": inference["max_output_tokens"],
        "temperature": inference["temperature"], "seed": inference["seed"],
        "reasoning": {"effort": inference["reasoning_effort"]},
    }


def parse_answer(text, option_count):
    """Accept only one letter, optionally wrapped or explicitly labelled."""
    if not isinstance(text, str):
        return {"answer": None, "parse_status": "invalid_format"}
    value = text.strip()
    # No free-text guessing, option-text matching, JSON default or judge call.
    patterns = (
        r"([A-Z])[.!]?",
        r"`{1,2}([A-Z])`{1,2}[.!]?",
        r"\(([A-Z])\)[.!]?",
        r"(?:Answer|Option|The answer is|The correct answer is)\s*:?\s*"
        r"(?:`{1,2}([A-Z])`{1,2}|([A-Z]))[.!]?",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, value, flags=re.IGNORECASE)
        if match:
            letter = next(group for group in match.groups() if group is not None).upper()
            if letter in LETTERS[:option_count]:
                return {"answer": letter, "parse_status": "legal"}
            return {"answer": None, "parse_status": "out_of_range"}
    return {"answer": None, "parse_status": "invalid_format"}
