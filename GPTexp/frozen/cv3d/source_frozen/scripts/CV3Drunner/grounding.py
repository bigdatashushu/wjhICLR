"""Deterministic rectangle-edge extraction from the public original RGB.

Natural color pixels alone do not define an annotation. Four supported sides
and corners must form one unique rectangle for each required question color.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import re

import numpy as np
from scipy.ndimage import binary_closing

VERSION = "cv3d-annotation-rectangles/1"
COLORS = ("red", "blue", "green")
MAX_BANDS = 128
MAX_PAIRS = 4096


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False,
        separators=(",", ":")).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Box:
    box_id: str
    view_index: int
    color: str
    xyxy: tuple[int, int, int, int]
    stroke_px: int
    side_support: tuple[float, ...]

    @property
    def safe_xyxy(self):
        # Radius 2 sampling, rounded pixel, and one-pixel separation from stroke.
        margin = self.stroke_px + 3
        x0, y0, x1, y1 = self.xyxy
        return (x0 + margin, y0 + margin, x1 - margin, y1 - margin)

    def public(self):
        return {**asdict(self), "safe_xyxy": self.safe_xyxy,
                "surface_validated": False}


def _mask(rgb, color):
    r, g, b = np.moveaxis(np.asarray(rgb, dtype=np.float32), -1, 0)
    if color == "red":
        return (r >= 150) & (r >= g * 1.45 + 15) & (r >= b * 1.45 + 15)
    if color == "blue":
        return (b >= 130) & (b >= r * 1.4 + 15) & (b >= g * 1.3 + 15)
    return (g >= 95) & (g >= r * 1.4 + 15) & (g >= b * 1.3 + 15)


def _bands(mask, minimum):
    """Merge neighboring long line segments, never global color extrema."""
    joined = binary_closing(mask, structure=np.ones((1, 3), dtype=bool))
    groups = []
    for row, line in enumerate(joined):
        edges = np.diff(np.r_[False, line, False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if stop - start < minimum:
                continue
            for group in reversed(groups):
                if (row - group[1] <= 3 and abs(int(start) - group[2]) <= 4
                        and abs(int(stop - 1) - group[3]) <= 4):
                    group[1] = row
                    group[2] = min(group[2], int(start))
                    group[3] = max(group[3], int(stop - 1))
                    break
            else:
                groups.append([row, row, int(start), int(stop - 1)])
            if len(groups) > MAX_BANDS:
                return None
    return groups


def _rectangles(rgb, color):
    mask = _mask(rgb, color)
    h, w = mask.shape
    horizontal = _bands(mask, max(12, int(w * .015)))
    vertical = _bands(mask.T, max(12, int(h * .015)))
    if horizontal is None or vertical is None:
        return [], "candidate_limit"
    if len(horizontal) * len(horizontal) > MAX_PAIRS:
        return [], "candidate_limit"
    found = []
    for i, top in enumerate(horizontal):
        for bottom in horizontal[i + 1:]:
            if (bottom[0] - top[1] < 12 or abs(top[2] - bottom[2]) > 4
                    or abs(top[3] - bottom[3]) > 4):
                continue
            x0, x1 = min(top[2], bottom[2]), max(top[3], bottom[3])
            y0, y1 = top[0], bottom[1]
            left = [v for v in vertical if abs(v[0] - x0) <= 4
                    and abs(v[2] - y0) <= 4 and abs(v[3] - y1) <= 4]
            right = [v for v in vertical if abs(v[1] - x1) <= 4
                     and abs(v[2] - y0) <= 4 and abs(v[3] - y1) <= 4]
            if not left or not right:
                continue
            stroke = max(top[1] - top[0] + 1, bottom[1] - bottom[0] + 1,
                         max(v[1] - v[0] + 1 for v in left + right))
            if stroke > 8 or min(x1 - x0, y1 - y0) <= 2 * (stroke + 3):
                continue
            support = (
                mask[max(0, y0 - 2):top[1] + 3, x0:x1 + 1].any(axis=0).mean(),
                mask[max(0, bottom[0] - 2):min(h, y1 + 3), x0:x1 + 1].any(axis=0).mean(),
                mask[y0:y1 + 1, max(0, x0 - 2):min(w, x0 + stroke + 2)].any(axis=1).mean(),
                mask[y0:y1 + 1, max(0, x1 - stroke - 1):min(w, x1 + 3)].any(axis=1).mean(),
            )
            corners = [mask[max(0, y - 3):min(h, y + 4),
                            max(0, x - 3):min(w, x + 4)].any()
                       for x in (x0, x1) for y in (y0, y1)]
            if min(support) < .86 or not all(corners):
                continue
            rect = (x0, y0, x1, y1)
            if any(max(abs(a - b) for a, b in zip(rect, old.xyxy)) <= 5 for old in found):
                continue
            found.append(Box(f"v0:{color}", 0, color, rect, stroke,
                             tuple(round(float(s), 6) for s in support)))
    return sorted(found, key=lambda box: box.xyxy), None


class AnnotationCatalog:
    def __init__(self, rgb, question, task):
        if task not in {"Depth", "Distance"}:
            raise ValueError("CV3D annotation task required")
        self.required_colors = tuple(dict.fromkeys(
            re.findall(r"\b(red|blue|green)\b", question.lower())))
        self.boxes = {}
        self.status = {}
        self.candidates = {}
        for color in COLORS:
            found, error = _rectangles(rgb, color)
            self.candidates[color] = [box.public() for box in found]
            self.status[color] = error or (
                "unique" if len(found) == 1 else "missing" if not found else "ambiguous")
            if len(found) == 1:
                self.boxes[found[0].box_id] = found[0]
        expected = 2 if task == "Depth" else 3
        self.measurements_available = (
            len(self.required_colors) == expected
            and all(self.status[color] == "unique" for color in self.required_colors))
        self.identity = digest(self.public())

    def public(self):
        return {
            "schema": VERSION, "view_index": 0,
            "boxes": [box.public() for _, box in sorted(self.boxes.items())],
            "color_status": self.status,
            "question_bindings": [
                {"color": color, "box_id": f"v0:{color}", "question_order": i,
                 "role": f"question_color:{color}"}
                for i, color in enumerate(self.required_colors)],
            "measurements_available": self.measurements_available,
            "definition": "unique closed colored rectangle edges in original RGB",
            "surface_validated": False,
        }
