"""Candidate lineage and explicit RGB entity scope, without mutating old caches."""
from __future__ import annotations

from itertools import combinations
import math
import re

from .common import digest

VERSION = "omni-candidate-entity-scope/4"


class ScopeError(ValueError):
    """Preserve the rejection code while locating the public entity field."""
    def __init__(self, code, field, **details):
        super().__init__(code)
        self.field, self.details = field, details


def iou(a, b):
    area = lambda r: max(0, r[2]-r[0]) * max(0, r[3]-r[1])
    overlap = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    total = area(a) + area(b) - overlap
    return overlap/total if total else 0.0


def candidate_facts(handle, view_index=-1, category=""):
    rows = []
    for c in handle.candidates:
        if (view_index >= 0 and c["view_index"] != view_index) or (
                category and c["category"] != category):
            continue
        row = {k: c[k] for k in ("candidate_id", "view_index", "category", "box_xyxy",
               "category_reviewed", "category_confirmed", "review_status",
               "mask_quality_assessed", "geometry_usable") if k in c}
        review = c.get("review") or {}
        row["lineage"] = {
            "cache_candidate_id": c["candidate_id"],
            "category_review_receipt_sha256": digest(review) if review else None,
            "review_identity": review.get("identity") if isinstance(review, dict) else None,
            "nms": c.get("nms", {"status": "not_recorded_in_cache"}),
            "instance_identity": "not_established_by_category_review"}
        rows.append(row)
    overlaps = [{"candidate_ids": [a["candidate_id"], b["candidate_id"]],
                 "iou": iou(a["box_xyxy"], b["box_xyxy"]),
                 "relationship": "unresolved"}
                for a, b in combinations(rows, 2)
                if a["category"] == b["category"] and a["view_index"] == b["view_index"]
                and iou(a["box_xyxy"], b["box_xyxy"]) > 0]
    return {"candidates": rows, "count": len(rows), "exhaustive": False,
            "coverage": "detections only; not exhaustive", "overlap_groups": overlaps,
            "lineage_version": VERSION,
            "instance_rule": "Overlap/category confirmation alone cannot merge RGB entities."}


def validate_scope(scope, observations, visible_ids, answer=None):
    field = "evidence.object_scope"
    fields = {"include", "exclude", "entities"}
    if not isinstance(scope, dict):
        raise ScopeError("object_scope_invalid", field, expected_type="object",
                         actual_type=type(scope).__name__,
                         repair="Counts require object_scope={include,exclude,entities} even with zero tool calls; enumerate RGB objects or requested subfeatures such as holes.")
    if set(scope) != fields:
        raise ScopeError("object_scope_invalid", field, missing_fields=sorted(fields-set(scope)),
                         unexpected_field_count=len(set(scope)-fields))
    for name in ("include", "exclude"):
        value = scope[name]
        if not isinstance(value, str) or len(value) > 200 or name == "include" and not value.strip():
            raise ScopeError("object_scope_invalid", field + "." + name,
                             expected_type="string", actual_type=type(value).__name__,
                             max_chars=200, actual_chars=len(value) if isinstance(value, str) else None)
    if not isinstance(scope["entities"], list) or len(scope["entities"]) > 128:
        raise ScopeError("object_scope_invalid", field + ".entities", expected_type="list",
                         max_entities=128)
    known = {c["candidate_id"]: c for r in observations
             if r["result_id"] in visible_ids and r["source_tool"] == "view_candidates"
             for c in r["data"]["candidates"][:8]}
    used, entities = set(), set()
    counts = {}
    for index, entity in enumerate(scope["entities"]):
        field = f"evidence.object_scope.entities[{index}]"
        fields = {"entity_id", "category", "candidate_ids", "relationship", "basis", "included"}
        if not isinstance(entity, dict):
            raise ScopeError("object_scope_invalid", field, expected_type="object",
                             actual_type=type(entity).__name__)
        if set(entity) != fields:
            raise ScopeError("object_scope_invalid", field, missing_fields=sorted(fields-set(entity)),
                             unexpected_field_count=len(set(entity)-fields))
        for name in ("entity_id", "category", "basis"):
            value = entity[name]
            if not isinstance(value, str) or not value.strip() or len(value) > 200:
                raise ScopeError("object_scope_invalid", field + "." + name,
                                 expected_type="nonempty string", actual_type=type(value).__name__,
                                 max_chars=200, actual_chars=len(value) if isinstance(value, str) else None)
        if type(entity["included"]) is not bool:
            raise ScopeError("object_scope_invalid", field + ".included", expected_type="bool",
                             actual_type=type(entity["included"]).__name__)
        if (not isinstance(entity["candidate_ids"], list)
                or any(not isinstance(k, str) for k in entity["candidate_ids"])):
            raise ScopeError("object_scope_invalid", field + ".candidate_ids", expected_type="list of strings")
        if entity["relationship"] not in ("single", "same_object", "part_whole", "rgb_only"):
            raise ScopeError("object_scope_invalid", field + ".relationship",
                             actual=entity["relationship"],
                             allowed=["single", "same_object", "part_whole", "rgb_only"])
        ids = entity["candidate_ids"]
        if (entity["entity_id"] in entities or len(set(ids)) != len(ids)
                or any(k not in known or k in used for k in ids)):
            raise ScopeError("object_scope_identity_invalid", field,
                             repair="Use distinct entity IDs and each delivered candidate ID once; unseen candidates cannot be referenced.")
        relation = entity["relationship"]
        if not ids and relation != "rgb_only":
            raise ScopeError(
                "object_scope_merge_requires_rgb", field + ".relationship",
                candidate_count=0, actual_relationship=relation, expected_relationship="rgb_only",
                repair='For candidate_ids=[] set relationship="rgb_only"; do not put rgb_only in basis or add a separate key.')
        if len(ids) == 1 and relation != "single":
            raise ScopeError(
                "object_scope_merge_requires_rgb", field + ".relationship",
                candidate_count=1, actual_relationship=relation, expected_relationship="single",
                repair='For one candidate ID set relationship="single".')
        if len(ids) > 1 and relation not in ("same_object", "part_whole"):
            raise ScopeError(
                "object_scope_merge_requires_rgb", field + ".relationship",
                candidate_count=len(ids), actual_relationship=relation,
                expected_relationships=["same_object", "part_whole"],
                repair="Combining candidate IDs requires same_object or part_whole with a specific RGB identity basis.")
        if len(ids) > 1 and not re.search(r"\brgb\b", entity["basis"], re.I):
            raise ScopeError(
                "object_scope_merge_requires_rgb", field + ".basis",
                candidate_count=len(ids),
                repair="Describe the specific RGB evidence for the same_object or part_whole relation; overlap alone is insufficient.")
        entities.add(entity["entity_id"])
        used.update(ids)
        if entity["included"]:
            counts[entity["category"]] = counts.get(entity["category"], 0) + 1
    if type(answer) in (int, float) and answer != sum(counts.values()):
        raise ScopeError("object_scope_count_mismatch", "answer", submitted=answer,
                         included_entities=sum(counts.values()),
                         repair="The count must equal included entities. Reinspect RGB and update the entity list when new evidence changes the count; the old list is not fixed.")
    return {"counts": counts, "candidate_ids": sorted(used),
            "included_entity_ids": [e["entity_id"] for e in scope["entities"] if e["included"]],
            "semantic_verification": "same_model_rgb_entity_declaration"}


def validate_count_groups(groups, scope, answer, operands=()):
    """An ordered ratio over the already identity-checked entity scope."""
    field = "evidence.count_groups"
    fields = {"numerator", "denominator"}
    if not isinstance(groups, dict) or set(groups) != fields:
        raise ScopeError("object_scope_ratio_invalid", field,
                         repair="Use count_groups={numerator:[entity IDs],denominator:[entity IDs]} in the question's ratio order.")
    included = {e["entity_id"] for e in scope["entities"] if e["included"]}
    categories = {e["entity_id"]: " ".join(e["category"].casefold().replace("_", " ").split())
                  for e in scope["entities"] if e["included"]}
    all_used = set()
    for role in ("numerator", "denominator"):
        ids = groups[role]
        if (not isinstance(ids, list) or any(not isinstance(k, str) for k in ids)
                or len(set(ids)) != len(ids) or set(ids) - included or set(ids) & all_used):
            raise ScopeError("object_scope_ratio_invalid", field + "." + role,
                             repair="Use included entity IDs once, in disjoint numerator/denominator groups; keep excluded entities outside both.")
        all_used.update(ids)
        expected = next((o["target"] for o in operands if o["role"] == role), None)
        if expected:
            expected = " ".join(expected.casefold().replace("_", " ").split())
            # A noun's singular/plural spelling is grammatical number, not an
            # object synonym or an inferred visual identity.
            singular = expected[:-1] if expected.endswith("s") else expected
            if any(categories[k] not in (expected, singular)
                   and not expected.startswith(categories[k] + " ") for k in ids):
                raise ScopeError("object_scope_ratio_invalid", field + "." + role,
                                 expected_category=expected,
                                 repair="Group the declared RGB entities by the requested numerator/denominator categories; preserve the question's order.")
    if all_used != included or not groups["denominator"]:
        raise ScopeError("object_scope_ratio_invalid", field,
                         repair="Group every included entity exactly once and provide a nonzero denominator.")
    value = len(groups["numerator"]) / len(groups["denominator"])
    if type(answer) not in (int, float) or not math.isclose(
            answer, value, rel_tol=1e-9, abs_tol=1e-9):
        raise ScopeError("object_scope_ratio_mismatch", "answer", submitted=answer,
                         numerator_count=len(groups["numerator"]),
                         denominator_count=len(groups["denominator"]), calculated=value,
                         repair="Compute the ordered numerator count / denominator count in Python; the ratio is not the total entity count.")
    return {"numerator_count": len(groups["numerator"]),
            "denominator_count": len(groups["denominator"]), "calculated_value": value}
