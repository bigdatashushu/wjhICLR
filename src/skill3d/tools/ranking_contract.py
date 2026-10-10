"""Shared, deterministic acceptance contract for relative-distance rankings."""

from __future__ import annotations

import math
from typing import Any, Sequence

from .category_match import matches, option_category

RANKING_CONTRACT_VERSION = "relative-distance-v11.2"
UNCERTAIN_DISTANCE_FLAGS = frozenset({"degraded", "point_contamination_suspect"})


def _distance(value: Any) -> bool:
    try:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value >= 0)
    except OverflowError:
        return False


def ranking_problems(value: Any, options: Sequence[str] | None = None) -> list[str]:
    """Validate completeness and uniqueness from details, never just status=ok."""
    if not isinstance(value, dict):
        return ["ranking must be an object"]
    if value.get("contract_version") != RANKING_CONTRACT_VERSION:
        return ["ranking contract missing or outdated"]
    if value.get("status") != "ok":
        return [f"ranking is not decidable: {value.get('status')}"]
    categories = value.get("requested_categories")
    if (not isinstance(categories, list) or len(categories) < 2
            or not all(isinstance(c, str) and c.strip() for c in categories)):
        return ["at least two candidate categories are required"]
    if any(matches(a, b) for i, a in enumerate(categories) for b in categories[i + 1:]):
        return ["candidate categories are not distinct"]
    if options and (
        len(options) != len(categories)
        or any(sum(matches(option_category(o), c) for c in categories) != 1 for o in options)
        or any(sum(matches(option_category(o), c) for o in options) != 1 for c in categories)
    ):
        return ["candidate categories do not cover the episode options exactly"]
    rows, details, distances = (value.get("ranking"), value.get("candidates"),
                                value.get("per_candidate"))
    if (not isinstance(rows, list) or not isinstance(details, list)
            or len(rows) != len(categories) or len(details) != len(categories)
            or not isinstance(distances, dict) or set(distances) != set(categories)):
        return ["ranking, candidates and per_candidate must cover every category"]
    if (any(not isinstance(r, dict) or not isinstance(r.get("category"), str)
            for r in rows + details)
            or sorted(str(r.get("category")) for r in rows) != sorted(categories)
            or sorted(str(r.get("category")) for r in details) != sorted(categories)):
        return ["candidate details contain duplicate or missing categories"]
    ref = value.get("reference")
    if (not isinstance(ref, dict) or not isinstance(ref.get("obj_id"), str)
            or not ref["obj_id"] or not isinstance(ref.get("track_id"), (str, type(None)))):
        return ["reference identity is missing"]
    owners: dict[tuple[str, str], str] = {("object", ref["obj_id"]): "reference"}
    if ref.get("track_id"):
        owners[("track", ref["track_id"])] = "reference"
    for detail in details:
        category = detail["category"]
        instances = detail.get("instances")
        if (detail.get("status") != "ok" or not isinstance(instances, list)
                or not instances or detail.get("n_instances") != len(instances)
                or not all(isinstance(i, dict) for i in instances)):
            return ["candidate instance coverage is incomplete"]
        ids = [i.get("obj_id") for i in instances]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            return ["candidate instance identities are missing or repeated"]
        for instance in instances:
            flags = instance.get("degradation_flags")
            if (instance.get("status") != "ok"
                    or not isinstance(instance.get("track_id"), (str, type(None)))
                    or not _distance(instance.get("distance_normalized"))
                    or not isinstance(flags, list)
                    or not all(isinstance(f, str) for f in flags)
                    or UNCERTAIN_DISTANCE_FLAGS.intersection(flags)
                    or not _distance(instance.get("n_valid_points"))
                    or instance["n_valid_points"] <= 0):
                return ["candidate has uncertain or invalid geometry"]
            identities = [("object", instance["obj_id"])]
            if instance.get("track_id"):
                identities.append(("track", instance["track_id"]))
            for identity in identities:
                if identity in owners and owners[identity] != category:
                    return ["reference or cross-category instance identity collision"]
                owners[identity] = category
        best = min(i["distance_normalized"] for i in instances)
        row = next(r for r in rows if r["category"] == category)
        if (not _distance(distances[category]) or distances[category] != best
                or row.get("distance_normalized") != best
                or not any(i["obj_id"] == row.get("obj_id")
                           and i["distance_normalized"] == best for i in instances)):
            return ["category minimum disagrees with the measured instances"]
    numbers = [r["distance_normalized"] for r in rows]
    if numbers != sorted(numbers) or numbers[0] == numbers[1]:
        return ["ranking is unsorted or has no unique minimum"]
    if (value.get("closest_category") != rows[0]["category"]
            or value.get("margin_normalized") != numbers[1] - numbers[0]
            or value.get("tied_categories") != []
            or value.get("categories_without_detection") != []
            or value.get("categories_with_invalid_geometry") != []):
        return ["ranking summary disagrees with the complete unique minimum"]
    return []
