"""Bounded text annotations for a separate RGB+depth image-input variant."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

VERSION = "mmsi-rgb-depth-note/1"
MAX_VIEWS = 6
FIXED_TOKEN_LIMIT = 80
TOTAL_TOKEN_LIMIT = 160
FIXED_WORD_LIMIT = 40

# The renderer must implement the legend before these annotations are used.
DEPTH_NOTE = (
    "Photo/image references mean RGBs only. Depthi matches RGBi pixels/size "
    "(WxH); x right, y down. Predicted depth: dark near, light far; magenta invalid. "
    "Per-view scaling: no meters or cross-view depth comparison. "
    "Directions come from RGBs/question."
)

COMPACT_DEPTH_NOTE = (
    "Photo/image references mean RGBs only. Depthi matches RGBi "
    "(WxH; x right, y down). Predicted depth: dark near, light far; magenta invalid. "
    "Per-view scaling; no meters/cross-view depth comparison. Orient from RGBs/question."
)


@dataclass(frozen=True)
class DepthAnnotation:
    """Original RGB order, then depth order; labels immediately precede images."""

    rgb_labels: tuple[str, ...]
    depth_labels: tuple[str, ...]
    note: str
    fixed_tokens: int
    total_tokens: int
    compacted: bool

    @property
    def all_text(self):
        # Count all supplemental text, including labels and dimensions.
        return "\n".join((*self.rgb_labels, *self.depth_labels, self.note))


def build_annotation(
    sizes: Sequence[tuple[int, int]], token_counter: Callable[[str], int]
) -> DepthAnnotation:
    """Count, shorten if needed, then reject anything exceeding either bound.

    The caller must supply an explicitly identified tokenizer. No word/token
    approximation, dimension truncation or model request is performed here.
    """
    if not 1 <= len(sizes) <= MAX_VIEWS:
        raise ValueError("depth_annotation_requires_one_to_six_views")
    for size in sizes:
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)):
            raise ValueError("depth_annotation_requires_positive_original_sizes")
    rgb_labels = tuple(f"RGB{i} {w}x{h}" for i, (w, h) in enumerate(sizes, 1))
    depth_labels = tuple(f"Depth{i}" for i in range(1, len(sizes) + 1))
    for note in (DEPTH_NOTE, COMPACT_DEPTH_NOTE):
        text = "\n".join((*rgb_labels, *depth_labels, note))
        fixed_tokens, total_tokens = token_counter(note), token_counter(text)
        if any(type(count) is not int or count < 1 for count in (fixed_tokens, total_tokens)):
            raise ValueError("invalid_annotation_token_counter")
        if (fixed_tokens <= FIXED_TOKEN_LIMIT and total_tokens <= TOTAL_TOKEN_LIMIT
                and len(note.split()) <= FIXED_WORD_LIMIT):
            return DepthAnnotation(
                rgb_labels, depth_labels, note, fixed_tokens, total_tokens,
                compacted=note != DEPTH_NOTE,
            )
    raise ValueError("depth_annotation_exceeds_token_budget")
