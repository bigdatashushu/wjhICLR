"""Explicit same-model surface judgments over previously delivered evidence."""
from __future__ import annotations

import ast
import json

from .tools import MEASUREMENTS

VERSION = "omni-rgb-surface-review/3"
DECISIONS = ("usable", "resample", "uncertain")


def points(row):
    data = row["data"]
    return [data["pixel_xy"]] if row["source_tool"] == "point_depth" else [
        data["a_xy"], data["b_xy"]]


def pending(observations, reviews):
    return [r["result_id"] for r in observations
            if r["source_tool"] in MEASUREMENTS and r["result_id"] not in reviews]


def priority_ids(observations, reviews, fresh=(), yielded=()):
    """Deliver selected/recent new samples before the older unreviewed backlog."""
    eligible = set(pending(observations, reviews))
    return list(dict.fromkeys(key for key in [*yielded, *reversed(fresh)] if key in eligible))


def state(reviews):
    return [dict(row) for row in reviews.values()]


def program_reviews(program):
    tree = ast.parse(program)
    body = tree.body[0].body
    assignments = [n for n in ast.walk(tree)
                   if isinstance(n, (ast.Assign, ast.AnnAssign))
                   and any(isinstance(t, ast.Name) and t.id == "measurement_reviews"
                           for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    if not assignments:
        return []
    node = assignments[0]
    if len(assignments) != 1 or body[0] is not node or not isinstance(node, ast.Assign):
        raise ValueError("reviews_must_be_first_literal_assignment")
    if len(node.targets) != 1:
        raise ValueError("reviews_must_be_first_literal_assignment")
    try:
        return ast.literal_eval(node.value)
    except (ValueError, TypeError):
        raise ValueError("reviews_must_be_literal") from None


def validate(rows, observations, eligible_ids=None):
    """Accept records independently; diagnostics identify the rejected field."""
    if not isinstance(rows, list):
        return {}, [{"code": "review_list_required", "field": "measurement_reviews"}]
    known = {r["result_id"]: r for r in observations if r["source_tool"] in MEASUREMENTS}
    all_ids = {r["result_id"] for r in observations}
    eligible = set(known) if eligible_ids is None else set(eligible_ids)
    result, errors, seen = {}, [], set()
    for index, row in enumerate(rows):
        def reject(code, field, **details):
            errors.append({"index": index, "code": code, "field": field, **details})
        try:
            if not isinstance(row, dict) or set(row) != {"result_id", "decision"}:
                reject("review_fields_required", "record")
                continue
            key = row["result_id"]
            if not isinstance(key, str) or key not in known:
                reject("review_not_measurement" if isinstance(key, str) and key in all_ids
                       else "review_reference_unknown", "result_id")
                continue
            if key in seen:
                reject("review_duplicate", "result_id")
                continue
            seen.add(key)
            if key not in eligible:
                reject("review_not_delivered", "result_id")
                continue
            if row["decision"] not in DECISIONS:
                reject("review_decision_invalid", "decision")
                continue
            result[key] = {"result_id": key, "decision": row["decision"]}
        except (TypeError, ValueError):
            reject("review_value_invalid", "record")
    return result, errors


def validate_program(program, observations, eligible_ids):
    try:
        return validate(program_reviews(program), observations, eligible_ids)
    except (ValueError, TypeError) as exc:
        return {}, [{"code": str(exc), "field": "measurement_reviews",
                     "repair": "Use the first literal measurement_reviews assignment."}]


def final_payload(text):
    raw = text.strip()
    if raw.startswith("```"):
        import re
        match = re.fullmatch(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", raw)
        if not match:
            raise ValueError("final_json_required")
        raw = match[1]

    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value

    value = json.loads(raw, object_pairs_hook=pairs)
    if (not isinstance(value, dict) or "answer" not in value
            or set(value) - {"answer", "measurement_reviews", "evidence"}):
        raise ValueError("final_fields_invalid")
    return value["answer"], value.get("measurement_reviews", []), value.get("evidence")
