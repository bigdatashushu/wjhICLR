"""Question-scoped evidence with cumulative statistics and unlimited call counts."""
from __future__ import annotations

import re
from types import SimpleNamespace

from skill3d.small_image.tools import build_registry as shared_registry
from skill3d.tools.contract import DomainValueError
from .sampling import install_sampling_checks

ALLOWED = frozenset({"view_candidates", "view_depth_summary", "point_depth", "pair_point_distance"})
MEASUREMENTS = frozenset({"point_depth", "pair_point_distance"})
SYNONYMS = {"sofa": ("couch",), "couch": ("sofa",), "tv": ("television",),
            "television": ("tv",), "chair": ("armchair",), "armchair": ("chair",)}


def words(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t[:-3] + "y" if t.endswith("ies") and len(t) > 4 else
            t[:-1] if t.endswith("s") and not t.endswith(("ss", "us")) and len(t) > 3 else t
            for t in tokens]


def relevant(category, stem):
    text = " " + " ".join(words(stem)) + " "
    return any(" " + " ".join(words(v)) + " " in text
               for v in (category, *SYNONYMS.get(category.lower(), ())))


def empty_counts():
    return {"calls": 0, "measurements": 0}


def phase_for(config, index):
    return "final" if index == config.max_program_rounds else "program"


def build_registry(cache, question, config):
    matching_texts = (question.question, *getattr(question, "option_texts", ()))
    candidates = [dict(c) for c in cache.candidates
                  if c.get("category_confirmed") is True
                  and any(relevant(c["category"], text) for text in matching_texts)]
    # Re-number after filtering so hidden pooled candidates have no public IDs.
    for index, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"v{candidate['view_index']}:c{index}"
    view = SimpleNamespace(frames=cache.frames, candidates=candidates,
                           geometries=cache.geometries, provenance=cache.provenance)
    registry, scene = shared_registry(view, question)
    # Instance-only pruning also removes the forbidden tool from kernel globals.
    registry._tools = {name: registry.get(name) for name in ALLOWED}
    registry._model_tools = ALLOWED
    registry.public_fields = {name: registry.public_fields[name] for name in ALLOWED}
    install_sampling_checks(registry)
    scene.question_type = "mindcube_option"
    original = registry.call_tool
    counters = empty_counts()
    def counted(name, args, *rest, **kwargs):
        if name not in ALLOWED:
            raise DomainValueError(name, "mindcube_tool_scope_denied")
        counters["calls"] += 1
        counters["measurements"] += int(name in MEASUREMENTS)
        return original(name, args, *rest, **kwargs)

    registry.call_tool = counted
    return registry, scene, counters
