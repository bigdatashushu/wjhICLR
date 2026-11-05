"""Explicit same-model surface judgments over previously delivered evidence."""
from __future__ import annotations

import ast
import json

from .tools import MEASUREMENTS
from .coordinates import EXTENT_ROLES, LEGACY_EXTENT_ROLES, extent_requirement, surface_name

VERSION = "omni-rgb-surface-review/6"
DECISIONS = ("usable", "resample", "uncertain")
ALIASES = {"wrong": "resample", "wrong target": "resample"}
VISIBILITY = ("visible", "cropped", "occluded", "unclear")
SCOPE_COMPLETENESS = ("complete", "incomplete", "unclear")


def valid_issues(value):
    return (isinstance(value, list) and len(value) <= 16
            and all(isinstance(issue, str) and issue.strip() and len(issue) <= 200
                    for issue in value))


def surface_review(submitted, expected, measurement, decision, reject):
    """Normalize physical identity separately from declared extremum roles."""
    required = {"target", "decision", "visibility"}
    allowed = required | {"surface", "observed_surface", "sample_xy", "extent_role",
                          "extent_basis", "scope_completeness", "issues"}
    if (not isinstance(submitted, dict) or not required <= set(submitted)
            or not {"surface", "observed_surface"} & set(submitted)
            or set(submitted) - allowed):
        reject("review_surface_fields_invalid", "",
               expected_fields=sorted(required | {"observed_surface"}),
               repair="Use the actual target, observed_surface (legacy surface also accepted), "
                      "decision, visibility; extent_role/extent_basis, scope_completeness, "
                      "issues and point_depth sample_xy are optional.")
        return None
    for name in ("target", "surface", "observed_surface", "extent_basis"):
        if name in submitted and (not isinstance(submitted[name], str)
                                  or not submitted[name].strip() or len(submitted[name]) > 80):
            reject("review_surface_fields_invalid", "." + name,
                   repair="Use a nonempty name of at most 80 characters.")
            return None
    observed = surface_name(submitted.get("observed_surface", submitted.get("surface")))
    if "surface" in submitted and surface_name(submitted["surface"]) != observed:
        reject("review_surface_target_mismatch", ".surface",
               actual=surface_name(submitted["surface"]), expected=observed,
               repair="surface is the legacy alias of observed_surface. Both must name "
                      "the same physical surface; put endpoint roles in extent_role.")
        return None
    surface = {**submitted, "observed_surface": observed, "surface": observed}
    raw_decision = surface["decision"]
    surface["decision"] = ALIASES.get(raw_decision, raw_decision) if isinstance(raw_decision, str) else None
    if surface["decision"] not in DECISIONS:
        reject("review_surface_fields_invalid", ".decision",
               actual=raw_decision if isinstance(raw_decision, str) else None,
               allowed=list(DECISIONS), repair="Use usable, resample, or uncertain for each surface.")
        return None
    if surface["visibility"] not in VISIBILITY:
        reject("review_surface_fields_invalid", ".visibility",
               actual=surface["visibility"] if isinstance(surface["visibility"], str) else None,
               allowed=list(VISIBILITY),
               repair="Use visible, cropped, occluded, or unclear. edge/partial do not establish full visible extrema.")
        return None
    if "extent_role" in surface and surface["extent_role"] not in EXTENT_ROLES:
        reject("review_surface_fields_invalid", ".extent_role", allowed=list(EXTENT_ROLES),
               repair="Declare the endpoint role separately from its physical surface.")
        return None
    if "scope_completeness" in surface and surface["scope_completeness"] not in SCOPE_COMPLETENESS:
        reject("review_surface_fields_invalid", ".scope_completeness",
               allowed=list(SCOPE_COMPLETENESS),
               repair="Use complete, incomplete, or unclear for the target extent's scope.")
        return None
    if "issues" in surface and not valid_issues(surface["issues"]):
        reject("review_surface_fields_invalid", ".issues",
               repair="Use a list of at most 16 unresolved visual doubts, each a nonempty string <=200 characters.")
        return None
    if "sample_xy" in surface and (measurement["source_tool"] != "point_depth"
            or not isinstance(surface["sample_xy"], list) or len(surface["sample_xy"]) != 2
            or any(type(v) is not int for v in surface["sample_xy"])):
        reject("review_surface_fields_invalid", ".sample_xy",
               repair="Only point_depth accepts sample_xy=[original integer x, original integer y].")
        return None
    if decision == "usable" and surface["decision"] != "usable":
        reject("review_surface_target_mismatch", ".decision",
               actual=surface["decision"], expected="usable",
               repair="A usable measurement needs usable surfaces; otherwise mark it resample or uncertain.")
        return None
    role, basis = extent_requirement(surface)
    if role is not None:
        surface["extent_role"] = role
    if surface["decision"] == "usable":
        for name, expected_value in (("visibility", "visible"), ("scope_completeness", "complete")):
            if name in surface and surface[name] != expected_value:
                reject("review_surface_not_usable", "." + name,
                       actual=surface[name], expected=expected_value,
                       repair="Do not mark cropped, occluded, unclear or incomplete evidence usable; "
                              "resample or mark uncertain.")
                return None
        if surface.get("issues"):
            reject("review_surface_not_usable", ".issues", actual=surface["issues"], expected=[],
                   repair="Unresolved visual doubts prevent usable. Resolve them with new evidence "
                          "or mark this surface resample/uncertain; do not merely remove the doubts.")
            return None
        if expected and surface["target"] != expected["target"]:
            reject("review_surface_target_mismatch", ".target",
                   actual=surface["target"], expected=expected["target"],
                   repair="Report the actual object first. A wrong sampled object requires resample/uncertain.")
            return None
        expected_role, _ = extent_requirement(expected)
        intended_surface = surface_name(expected.get("surface"))
        legacy_role = ("extent_role" not in expected and intended_surface in LEGACY_EXTENT_ROLES)
        changed_surface = bool(expected and observed != intended_surface)
        # Legacy canonical names may describe a role; new intents describe physical surfaces.
        if changed_surface and not (legacy_role and "extent_role" in submitted):
            name = "observed_surface" if "observed_surface" in submitted else "surface"
            reject("review_surface_target_mismatch", "." + name,
                   actual=observed, expected=intended_surface,
                   repair="Keep the actual physical surface. Only a legacy canonical endpoint may "
                          "be migrated with an explicit matching extent_role, extent_basis and "
                          "complete scope; otherwise resample or mark uncertain.")
            return None
        if expected_role is not None and role != expected_role:
            reject("review_surface_role_mismatch", ".extent_role", actual=role, expected=expected_role,
                   repair="Check the required endpoint role in point/a/b order. Do not relabel the "
                          "physical surface to satisfy it; resample or mark uncertain if unconfirmed.")
            return None
        needs_scope = ("extent_role" in expected or changed_surface
                       or "extent_role" in submitted and observed not in LEGACY_EXTENT_ROLES)
        if needs_scope:
            if basis is None:
                reject("review_surface_role_mismatch", ".extent_basis",
                       repair="An explicit physical-surface/role declaration needs a nonempty extent_basis.")
                return None
            if surface.get("scope_completeness") != "complete":
                reject("review_surface_not_usable", ".scope_completeness",
                       actual=surface.get("scope_completeness"), expected="complete",
                       repair="Confirm the complete target extent for this role, or mark resample/uncertain.")
                return None
    if raw_decision != surface["decision"]:
        surface.update(raw_decision=raw_decision, normalization="wrong_target_to_resample")
    raw_surface = submitted.get("surface", submitted.get("observed_surface"))
    if raw_surface != observed:
        surface["raw_surface"] = raw_surface
    return surface


def points(row):
    data = row["data"]
    return [data["pixel_xy"]] if row["source_tool"] == "point_depth" else [
        data["a_xy"], data["b_xy"]]


def pending(observations, reviews):
    superseded = {r["data"].get("replaces_result_id") for r in observations
                  if r["source_tool"] in MEASUREMENTS}
    return [r["result_id"] for r in observations
            if r["source_tool"] in MEASUREMENTS
            and r["result_id"] not in superseded
            and reviews.get(r["result_id"], {}).get("decision") != "usable"]


def apply_updates(reviews, accepted, observations):
    """Keep replaced measurements excluded, even if a later review reapproves them."""
    for key, row in accepted.items():
        old = reviews.get(key, {})
        if (old.get("decision") == row["decision"] == "usable"
                and "observed_surfaces" not in row and "observed_surfaces" in old):
            row = {**row, "observed_surfaces": [dict(s) for s in old["observed_surfaces"]]}
        reviews[key] = row
    for row in observations:
        old = row["data"].get("replaces_result_id") if row["source_tool"] in MEASUREMENTS else None
        if old:
            reviews[old] = {**reviews.get(old, {}), "result_id": old, "decision": "resample",
                            "exclusion": "replaced", "replaced_by_result_id": row["result_id"]}


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
            identity = {"measurement_result_id": row["result_id"]} if (
                isinstance(row, dict) and isinstance(row.get("result_id"), str)
                and row["result_id"] in known) else {}
            errors.append({"index": index, "code": code, "field": field, **identity, **details})
        try:
            if (not isinstance(row, dict) or not {"result_id", "decision"} <= set(row)
                    or set(row) - {"result_id", "decision", "observed_surfaces", "issues"}):
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
            raw = row["decision"]
            decision = ALIASES.get(raw, raw) if isinstance(raw, str) else None
            if decision not in DECISIONS:
                reject("review_decision_invalid", "decision", actual=raw if isinstance(raw, str) else None,
                       allowed=list(DECISIONS), repair="Use usable, resample, or uncertain.")
                continue
            if "issues" in row and not valid_issues(row["issues"]):
                reject("review_fields_required", "issues",
                       repair="Use a list of at most 16 unresolved visual doubts, each a nonempty string <=200 characters.")
                continue
            if decision == "usable" and row.get("issues"):
                reject("review_not_usable", "issues", actual=row["issues"], expected=[],
                       repair="Unresolved visual doubts prevent usable; resolve them with evidence "
                              "or mark the measurement resample/uncertain.")
                continue
            surfaces = row.get("observed_surfaces")
            expected = known[key]["data"].get("intended_targets")
            if (decision == "usable" and surfaces is None
                    and known[key]["data"].get("replaces_result_id")):
                reject("review_replacement_surfaces_required", "observed_surfaces",
                       replaces_result_id=known[key]["data"]["replaces_result_id"],
                       repair="Review this delivered replacement's own visible surfaces in point/a/b order. "
                              "It cannot inherit the replaced measurement's usable decision.")
                continue
            if surfaces is not None:
                count = len(points(known[key]))
                if not isinstance(surfaces, list) or len(surfaces) != count:
                    reject("review_surface_fields_invalid", "observed_surfaces", expected_count=count,
                           repair="Use one surface for point_depth and two surfaces in a/b order for a pair.")
                    continue
                normalized = []
                for point_index, submitted in enumerate(surfaces):
                    field = f"observed_surfaces[{point_index}]"
                    expected_point = expected[point_index] if expected else {}
                    required_role, required_basis = extent_requirement(expected_point)

                    def reject_surface(code, suffix, **details):
                        actual_surface = submitted.get("observed_surface", submitted.get("surface")) if (
                            isinstance(submitted, dict)) else None
                        reject(code, field + suffix, point_index=point_index,
                               expected_target=expected_point.get("target"),
                               intended_surface=surface_name(expected_point.get("surface")),
                               observed_surface=surface_name(actual_surface),
                               required_role=required_role, required_basis=required_basis, **details)

                    surface = surface_review(submitted, expected_point, known[key], decision, reject_surface)
                    if surface is None:
                        break
                    normalized.append(surface)
                if len(normalized) != count:
                    continue
                surfaces = normalized
            result[key] = {"result_id": key, "decision": decision}
            if decision != raw:
                result[key].update(raw_decision=raw, normalization="wrong_target_to_resample")
            if surfaces is not None:
                result[key]["observed_surfaces"] = surfaces
            if "issues" in row:
                result[key]["issues"] = list(row["issues"])
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
