"""Immutable current-question candidate view and local call limits."""
from __future__ import annotations

import re
from types import SimpleNamespace

from skill3d.small_image.tools import build_registry as shared_registry
from skill3d.tools.contract import DomainValueError

from .common import digest

VERSION = "omni-question-scoped-tools/2"
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


def remaining(config, counters):
    return {"tool_calls": max(0, config.max_tool_calls - counters["calls"]),
            "measurements": max(0, config.max_measurements - counters["measurements"])}


def phase_for(config, round_index, counters):
    return "final" if (round_index == config.max_program_rounds
                       or counters["calls"] >= config.max_tool_calls
                       or counters.get("rejected", 0)) else "program"


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
    original = registry.call_tool
    counters = empty_counts()
    registry.quota_denials = []
    registry.on_quota_exhausted = None

    def limited(name, args, *rest, **kwargs):
        if name not in ALLOWED:
            raise DomainValueError(name, "omni_tool_scope_denied")
        measurement = int(name in MEASUREMENTS)
        counters["attempted"] += 1
        counters["attempted_measurements"] += measurement
        reason = ("tool_calls_exhausted" if counters["calls"] >= config.max_tool_calls else
                  "measurements_exhausted" if measurement and
                  counters["measurements"] >= config.max_measurements else "")
        if reason:
            counters["rejected"] += 1
            counters["rejected_measurements"] += measurement
            registry.quota_denials.append({
                "schema": VERSION, "tool": name, "args": dict(args),
                "argument_digest": digest(args), "admitted": False, "reason": reason,
                "remaining": remaining(config, counters)})
            # This host callback terminates the cell before the underlying call.
            # It is distinct from tool contract/domain/confidence failures.
            if registry.on_quota_exhausted:
                registry.on_quota_exhausted(reason)
            raise RuntimeError("quota_callback_missing")
        counters["calls"] += 1
        counters["measurements"] += measurement
        return original(name, args, *rest, **kwargs)

    registry.call_tool = limited
    return registry, scene, counters
