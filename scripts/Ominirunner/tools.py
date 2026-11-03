"""Immutable current-question candidate view and unbounded call accounting."""
from __future__ import annotations

import re
from types import SimpleNamespace

from skill3d.small_image.tools import build_registry as shared_registry
from skill3d.tools.contract import DomainValueError
from .sampling import VERSION as SAMPLING_VERSION, EDGE_NOTE, point_depth, pair_point_distance

VERSION = "omni-question-scoped-tools/4"
ALLOWED = {"view_candidates", "view_depth_summary", "point_depth", "pair_point_distance"}
MEASUREMENTS = {"point_depth", "pair_point_distance"}
SYNONYMS = {
    "sofa": ("couch",), "couch": ("sofa",), "armchair": ("chair",),
    "chair": ("armchair",), "tv": ("television",), "television": ("tv",),
    "cabinet": ("cupboard",), "cupboard": ("cabinet",),
}


def empty_counts():
    return {"calls": 0, "measurements": 0, "attempted": 0, "rejected": 0,
            "attempted_measurements": 0, "rejected_measurements": 0}


def phase_for(config, round_index):
    return "final" if round_index == config.max_program_rounds else "program"


def words(text):
    result = []
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and not token.endswith(("ss", "us")) and len(token) > 3:
            token = token[:-1]
        result.append(token)
    return result


def relevant(category, question):
    phrase = " " + " ".join(words(question)) + " "
    variants = (category, *SYNONYMS.get(category.lower(), ()))
    return any(" " + " ".join(words(v)) + " " in phrase for v in variants)


def build_registry(cache, question, config):
    candidates = [dict(c) for c in cache.candidates if relevant(c["category"], question.question)]
    view = SimpleNamespace(
        frames=cache.frames, candidates=candidates, geometries=cache.geometries,
        provenance={"scope": VERSION, "input_sha256": question.input_sha256})
    registry, scene = shared_registry(view, question)
    scene.question_type = "omni_native_scalar"
    for name, fn, description, fields in (
        ("point_depth", point_depth,
         "Raw camera Z at the rounded/clamped pixel and each valid 5x5 pixel, null if invalid. "
         "depth_samples has origin_xy and row-major values; no averaging. " + EDGE_NOTE,
         {"view_index", "pixel_xy", "depth", "depth_samples", "n_valid", "unit"}),
        ("pair_point_distance", pair_point_distance,
         "Same-view chord from raw 3D points at the rounded/clamped endpoint pixels; "
         "null if either center is invalid. Both endpoint depth matrices are returned. "
         "Call again for different endpoints. " + EDGE_NOTE,
         {"view_index", "a_xy", "b_xy", "distance", "a_depth_samples", "b_depth_samples", "unit"}),
    ):
        tool = registry.get(name)
        tool.fn = fn
        tool.spec = tool.spec.model_copy(update={
            "description": description, "args_schema_ref": SAMPLING_VERSION,
            "returns_schema_ref": SAMPLING_VERSION})
        registry.public_fields[name] = frozenset(fields)
    registry.decision_version = VERSION
    original = registry.call_tool
    counters = empty_counts()
    def counted(name, args, *rest, **kwargs):
        if name not in ALLOWED:
            raise DomainValueError(name, "omni_tool_scope_denied")
        measurement = int(name in MEASUREMENTS)
        counters["attempted"] += 1
        counters["attempted_measurements"] += measurement
        counters["calls"] += 1
        counters["measurements"] += measurement
        return original(name, args, *rest, **kwargs)

    registry.call_tool = counted
    return registry, scene, counters
