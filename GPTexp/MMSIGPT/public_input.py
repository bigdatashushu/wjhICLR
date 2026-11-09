"""The same label-free input contract for model prompts and solve(ctx)."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

INTERFACE_VERSION = "mmsigpt-interface/3"
PROMPT_VERSION = "mmsigpt-prompt/4"
PUBLIC_CONTEXT_FIELDS = frozenset({"scene", "tools", "frames", "input", "options", "question"})


def public_input(problem, frames):
    from skill3d.small_image.runner import parse_options

    return {
        "benchmark": problem.benchmark,
        "task_type": problem.task_type,
        "question": problem.question,
        "options": [{"label": label, "text": text}
                    for label, text in parse_options(problem.options)],
        "views": [{"view_index": index, "width": frame.shape[1], "height": frame.shape[0]}
                  for index, frame in enumerate(frames)],
    }


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class MMSIContext:
    scene: object
    tools: object
    frames: tuple
    input: object
    options: tuple
    question: str
