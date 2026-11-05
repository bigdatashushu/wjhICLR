"""Bounded arithmetic over declared, delivered and usable measurement values."""
from __future__ import annotations

import ast
import math
import operator
import re

from .native import scalar
from .tools import MEASUREMENTS
from .sampling import depth_value
from .answer_contract import canonical, normalize_object_name, LENGTH_UNIT_METERS
from .instances import ScopeError, validate_scope, validate_count_groups
from .provenance import VERSION as ORIGIN_VERSION, declared_ids
from .task_evidence import match_operands, validate_expression

VERSION = "omni-answer-evidence/10"
REL_TOL = ABS_TOL = 1e-9
DIMENSIONS = {
    "height": ("object_vertical", {"top", "base"}),
    "width": ("object_horizontal", {"left", "right"}),
    "length": ("object_length", {"front", "back"}),
    "diameter": ("object_diameter", {"rim_a", "rim_b"}),
}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv}


REPAIRS = {
    "answer_option_required": "Select a public option ID (O1, O2, ...) in answer.",
    "answer_option_invalid": "Select one of the public option IDs or an exact allowed label; no added articles.",
    "tool_comparison_invalid": "Use bindings, left, right, operator and answers for tool_comparison. Operators: <, <=, >, >=, ==, !=, three_way.",
    "tool_comparison_answer_mismatch": "Choose the public option mapped to the host-computed comparison result.",
    "comparison_unit_mismatch": "Both sides must have identical units and view scale; metric thresholds require a public scale anchor.",
    "task_quantity_mismatch": "Bindings must represent the requested quantity. Camera Z cannot establish walking distance, collision or object-to-object distance.",
    "task_operand_mismatch": "Match each binding to a public operand's quantity and exact sampled object identity; operand_index may select an ordered role. For object distances sample the candidate and the fixed origin/reference.",
    "task_operand_missing": "Provide usable bindings for every requested numerator/denominator or comparison operand.",
    "task_expression_mismatch": "Use the public operand order: numerator / denominator, or the stated left/right comparison and stacked-object multiplier.",
    "task_unit_mismatch": "The expression unit must match the public task unit. Use a scale in that unit, converting the public span first; bare relative depth is not a physical length.",
    "scale_anchor_invalid": "Use scale={name,value,unit,anchor_binding} with a public span and matching complete-dimension anchor. Units: m/ft/in/yd/mi; convert to the requested task unit before setting value. Include the scale variable in the expression.",
    "measurement_surface_required": "Declare targets at sampling. Put observed_surfaces inside each measurement_reviews record, in point/a/b order, with target/surface/decision/visibility; never inside evidence.",
    "object_scope_invalid": "For counts provide object_scope={include,exclude,entities}; list RGB entities, candidate IDs, relationship, basis and included.",
    "object_scope_identity_invalid": "Use delivered candidate IDs once and distinct RGB entity IDs; do not count a candidate twice.",
    "object_scope_merge_requires_rgb": "Only an explicit same_object or part_whole judgment with RGB basis can combine candidates. Box overlap alone is insufficient.",
    "object_scope_count_mismatch": "The numeric count must equal included RGB entities under the declared scope.",
    "object_scope_ratio_invalid": "Provide count_groups with ordered numerator/denominator lists of included entity IDs.",
    "object_scope_ratio_mismatch": "Compute the numerator entity count divided by the denominator count.",
    "answer_provenance_required": "Declare provenance={source:rgb_only,referenced_result_ids:[],abandoned_result_ids:[prior IDs],abandonment_reason:reason} for an independent RGB fallback, or source=tool_informed with referenced IDs and checked bindings.",
    "answer_provenance_invalid": "Use source=rgb_only/tool_informed, referenced_result_ids, abandoned_result_ids and a nonempty abandonment_reason when abandoning prior inputs. Explicit tool-derived answers require checked expression/bindings.",
    "answer_provenance_conflict": "An explicitly tool-derived answer cannot be declared rgb_only. Use checked bindings and calculation/tool_estimate, or recompute an independent RGB estimate and explain abandoned dependencies.",
    "answer_evidence_required": "Provide evidence: visual_estimate with a short basis, or calculation with expression and bindings.",
    "answer_evidence_fields": "Use visual_estimate with basis and task scope/assumptions, calculation with expression+bindings, or tool_comparison with left+right+operator+answers+bindings; scale is optional. observed_surfaces belongs in measurement_reviews, not evidence.",
    "visual_basis_required_max200": "Use a nonempty visual_estimate basis of at most 200 characters; only object_scope and assumptions may accompany kind+basis.",
    "calculation_bindings_required": "Provide 1 to 16 named measurement bindings.",
    "calculation_binding_fields": "Each binding requires result_id and quantity; optional fields depend on the measurement type.",
    "calculation_measurement_unknown": "Use a successful measurement ID from visible TOOL FACTS.",
    "calculation_measurement_not_delivered": "Use only currently visible TOOL FACTS; new measurements need a subsequent review request.",
    "calculation_measurement_not_usable": "Review the target in RGB first; resample or uncertain measurements cannot support a calculation.",
    "calculation_measurement_replaced": "Use the replacement measurement after its own RGB review; replaced values remain excluded.",
    "point_is_camera_z": "point_depth only supports quantity=camera_z; it cannot be declared as height, width, or length.",
    "calculation_sample_xy_invalid": "sample_xy must be an exact [x,y] pair of native integers from the returned matrix.",
    "calculation_sample_not_delivered": "Choose sample_xy inside this measurement's returned depth_samples matrix.",
    "calculation_sample_invalid": "The selected depth is invalid; select an exact valid target pixel or resample.",
    "distance_quantity_invalid": "For a pair use chord, height, width, or length; chord must omit geometry.",
    "complete_dimension_not_declared": "Full dimensions require complete same-object extrema and the matching axis/anchors; partial or unclear extents cannot be declared complete.",
    "calculation_endpoint_invalid": "The pair has no valid center-endpoint distance; remeasure the intended endpoints.",
    "expression_length": "Use a nonempty expression of at most 200 characters.",
    "expression_syntax": "Use a valid arithmetic expression with numbers, binding names, parentheses, + - * /.",
    "expression_size": "Simplify the arithmetic expression to at most 64 AST nodes.",
    "expression_bindings_unused": "Use every declared binding in the expression and at least one measurement.",
    "expression_node_forbidden": "Only numbers, declared binding names, parentheses, unary +/- and + - * / are allowed.",
    "expression_arithmetic": "Avoid division by zero and overflow; check the bound measurement values.",
    "calculation_numeric_required": "A calculation requires a native int or float answer. For tool-supported text/category answers use tool_comparison.",
    "calculation_number_out_of_range": "Use finite numeric values with magnitude at most 1e100.",
    "answer_calculation_mismatch": "Compute answer in Python from the original bound values using the declared expression; do not round before ReturnAnswer. Final JSON must retain enough precision for the stated tolerances.",
    "native_finite_scalar_required": "answer must be a finite native int/float or the requested text; booleans, arrays, null and nonfinite numbers are invalid.",
    "duplicate_json_key": "Use each JSON key exactly once.",
    "final_answer_fields": "Output one JSON object with answer, measurement_reviews and evidence; omit other fields.",
    "final_answer_invalid": "Output valid final JSON with a native scalar answer and the required evidence.",
    "answer_evidence_invalid": "Check the evidence object and its native JSON field types.",
}


class EvidenceError(ValueError):
    """Only bounded, public arithmetic/contract diagnostics, never reference labels."""
    def __init__(self, code, field, *, repair=None, **details):
        super().__init__(code)
        self.code = code
        self.record = {"code": code, "field": field, **details,
                       "validation_status": {"rejected_field": field,
                                             "other_fields": "not_certified",
                                             "independent_visual_geometry": False},
                       "repair": repair or REPAIRS[code]}


def fail(code, field, **details):
    raise EvidenceError(code, field, **details)


def feedback(error):
    if isinstance(error, EvidenceError):
        return [dict(error.record)]
    code = str(error)
    if code not in REPAIRS:
        code = "answer_evidence_invalid"
    return [{"code": code, "repair": REPAIRS[code]}]


def field_details(value, required, allowed):
    """Bound malformed field-name diagnostics, including non-string keys."""
    names = set(value)
    extras = sorted({k if isinstance(k, str) and re.fullmatch(r"[a-z_]{1,32}", k)
                     else "<invalid_field>" for k in names - allowed})
    return {"missing_fields": sorted(required - names),
            "unexpected_fields": extras[:8],
            "unexpected_field_count": len(names - allowed)}


def public_span_matches(span, value, unit):
    """Allow unit conversion without relaxing same-unit value matching."""
    public_unit = span.get("unit", "m")
    if public_unit not in LENGTH_UNIT_METERS:
        return False
    if public_unit == unit:
        return span["value"] == value
    return math.isclose(span["value"] * LENGTH_UNIT_METERS[public_unit],
                        value * LENGTH_UNIT_METERS[unit], rel_tol=1e-12, abs_tol=0.0)


def dependency_ids(evidence, observations, previous=()):
    """Retain known calculation inputs, including inputs from a failed answer."""
    known = {r["result_id"] for r in observations if r["source_tool"] in MEASUREMENTS}
    bindings = evidence.get("bindings") if isinstance(evidence, dict) else None
    keys = []
    if isinstance(evidence, dict) and evidence.get("kind") in ("calculation", "tool_comparison", "tool_estimate") and isinstance(bindings, dict):
        # Canonical order survives JSON persistence, including dynamic Python
        # declarations that cannot be reconstructed through literal AST reads.
        names = sorted(k for k in bindings if isinstance(k, str))[:16]
        keys = [bindings[k].get("result_id") for k in names if isinstance(bindings[k], dict)]
    return list(dict.fromkeys(k for k in [*keys, *declared_ids(evidence), *previous] if isinstance(k, str) and k in known))


def origin_record(evidence, known, previous, explicit):
    """Validate explicit declarations; never infer origin from numeric equality."""
    origin = evidence.get("provenance")
    visual = evidence.get("kind") == "visual_estimate"
    previous, explicit = set(previous) & set(known), set(explicit) & set(known)
    if origin is None:
        if visual and (previous or explicit):
            fail("answer_provenance_required", "evidence.provenance",
                 prior_dependency_ids=sorted(previous), explicit_program_dependency_ids=sorted(explicit))
        return {"version": ORIGIN_VERSION,
                "source": "unclassified" if visual and known else "rgb_only" if visual else "tool_informed",
                "referenced_result_ids": [], "origin_independently_verified": False}
    fields = {"source", "referenced_result_ids", "abandoned_result_ids", "abandonment_reason"}
    if not isinstance(origin, dict) or set(origin) != fields:
        fail("answer_provenance_invalid", "evidence.provenance",
             **(field_details(origin, fields, fields) if isinstance(origin, dict) else {}))
    if origin["source"] not in ("rgb_only", "tool_informed"):
        fail("answer_provenance_invalid", "evidence.provenance.source")
    for name in ("referenced_result_ids", "abandoned_result_ids"):
        ids = origin[name]
        if (not isinstance(ids, list) or len(ids) > 128 or any(not isinstance(k, str) for k in ids)
                or len(set(ids)) != len(ids) or set(ids) - set(known)):
            fail("answer_provenance_invalid", "evidence.provenance." + name)
    refs, abandoned = set(origin["referenced_result_ids"]), set(origin["abandoned_result_ids"])
    reason = origin["abandonment_reason"]
    if (not isinstance(reason, str) or len(reason) > 200 or abandoned and not reason.strip()
            or refs & abandoned):
        fail("answer_provenance_invalid", "evidence.provenance.abandonment_reason")
    if visual:
        if origin["source"] != "rgb_only" or refs or explicit:
            fail("answer_provenance_conflict", "evidence.provenance",
                 explicit_program_dependency_ids=sorted(explicit))
        if previous - abandoned:
            fail("answer_provenance_required", "evidence.provenance.abandoned_result_ids",
                 missing_abandoned_result_ids=sorted(previous - abandoned))
    elif origin["source"] != "tool_informed" or not refs or explicit - refs:
        fail("answer_provenance_conflict", "evidence.provenance",
             explicit_program_dependency_ids=sorted(explicit))
    return {**origin, "version": ORIGIN_VERSION, "origin_independently_verified": False}


def finite(value, field="evidence.expression"):
    if type(value) not in (int, float):
        fail("calculation_numeric_required", field, actual_type=type(value).__name__)
    try:
        if not math.isfinite(value) or abs(value) > 1e100:
            fail("calculation_number_out_of_range", field)
    except OverflowError:
        fail("calculation_number_out_of_range", field)
    return value


def expression_value(expression, bindings, *, require_all=True, return_used=False):
    if not isinstance(expression, str) or not 0 < len(expression) <= 200:
        fail("expression_length", "evidence.expression", max_chars=200,
             actual_chars=len(expression) if isinstance(expression, str) else None)
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        fail("expression_syntax", "evidence.expression")
    if sum(1 for _ in ast.walk(tree)) > 64:
        fail("expression_size", "evidence.expression", max_nodes=64)
    used = set()

    def visit(node):
        if isinstance(node, ast.Constant):
            return finite(node.value)
        if isinstance(node, ast.Name) and node.id in bindings:
            used.add(node.id)
            return finite(bindings[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return finite(visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1))
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            try:
                return finite(OPS[type(node.op)](visit(node.left), visit(node.right)))
            except (ZeroDivisionError, OverflowError):
                fail("expression_arithmetic", "evidence.expression")
        fail("expression_node_forbidden", "evidence.expression", node_type=type(node).__name__)

    value = visit(tree.body)
    if require_all and (used != set(bindings) or not used):
        fail("expression_bindings_unused", "evidence.expression",
             unused_bindings=sorted(set(bindings) - used))
    return (value, used) if return_used else value


def expression_unit(expression, units):
    """Dimension algebra on the same already validated arithmetic AST."""
    def combine(a, b, sign=1):
        out = dict(a)
        for key, value in b.items():
            out[key] = out.get(key, 0) + sign*value
        return {k: v for k, v in out.items() if v}
    def visit(node):
        if isinstance(node, ast.Constant):
            return {}
        if isinstance(node, ast.Name):
            return units[node.id]
        if isinstance(node, ast.UnaryOp):
            return visit(node.operand)
        a, b = visit(node.left), visit(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if a != b:
                fail("comparison_unit_mismatch", "evidence.expression")
            return a
        return combine(a, b, -1 if isinstance(node.op, ast.Div) else 1)
    return visit(ast.parse(expression, mode="eval").body)


def validate(answer, evidence, observations, reviews, visible_ids, task=None, *,
             visible_result_ids=None, previous_dependency_ids=(), program_dependency_ids=(),
             measurement_reviewer=None):
    """Check declaration consistency, never independent visual geometry.

    visible_ids gates measurements; visible_result_ids gates candidate scope.
    """
    if measurement_reviewer is not None:
        from .jev import Policy, checked_bindings
        Policy.parse(measurement_reviewer)
        evidence = checked_bindings(evidence, observations, reviews)
    try:
        scalar(answer)
    except ValueError:
        fail("native_finite_scalar_required", "answer", actual_type=type(answer).__name__)
    if task is not None:
        try:
            answer = canonical(answer, task)
        except ValueError as exc:
            fail(str(exc), "answer", allowed_answers=task["allowed_answers"])
    known = {r["result_id"]: r for r in observations if r["source_tool"] in MEASUREMENTS}
    if evidence is None:
        if known or task and task["quantity"] in ("count", "volume", "volume_ratio"):
            fail("answer_evidence_required", "evidence")
        return {"kind": "visual_estimate", "basis": "original RGB", "verified_arithmetic": False}
    if not isinstance(evidence, dict):
        fail("answer_evidence_fields", "evidence", expected_type="object")
    origin = origin_record(evidence, known, previous_dependency_ids, program_dependency_ids)
    volume_task = task and task["quantity"] in ("volume", "volume_ratio")
    if volume_task:
        assumptions = evidence.get("assumptions")
        fields = {"object_scope", "volume_model"}
        repair = ('Volume assumptions must contain exactly object_scope and volume_model, '
                  'both nonempty strings of at most 200 characters. '
                  'Example: {"object_scope":"one sofa and one chair, excluding footstool",'
                  '"volume_model":"bounding cuboids for both objects"}. '
                  'The count object_scope entity dictionary does not belong here.')
        if not isinstance(assumptions, dict):
            fail("object_scope_invalid", "evidence.assumptions",
                 expected_type="object", actual_type=type(assumptions).__name__, repair=repair)
        if set(assumptions) != fields:
            fail("object_scope_invalid", "evidence.assumptions",
                 **field_details(assumptions, fields, fields), repair=repair)
        for name in sorted(fields):
            value = assumptions[name]
            if not isinstance(value, str) or not value.strip() or len(value) > 200:
                fail("object_scope_invalid", f"evidence.assumptions.{name}",
                     expected_type="nonempty string", actual_type=type(value).__name__,
                     max_chars=200, actual_chars=len(value) if isinstance(value, str) else None,
                     repair=repair)
    if evidence.get("kind") == "visual_estimate":
        basis = evidence.get("basis")
        if (not {"kind", "basis"} <= set(evidence)
                or set(evidence) - {"kind", "basis", "object_scope", "assumptions", "count_groups", "provenance"} or not isinstance(basis, str)
                or not basis.strip() or len(basis) > 200):
            fail("visual_basis_required_max200", "evidence.basis", max_chars=200,
                 actual_chars=len(basis) if isinstance(basis, str) else None,
                 actual_type=type(basis).__name__, expected_type="nonempty string",
                 repair="Write a nonempty basis <=200 characters (aim <=120). "
                        "In Python use assert 0 < len(evidence['basis']) <= 200 before submitting. "
                        "Shorten the reasoning yourself; do not silently truncate it.",
                     **field_details(evidence, {"kind", "basis"}, {
                         "kind", "basis", "object_scope", "assumptions", "count_groups", "provenance"}))
        checked_scope = None
        count_task = task and task["quantity"] == "count" and task["unit"] == "count"
        if "object_scope" in evidence or count_task or task and task["quantity"] == "count_ratio":
            try:
                checked_scope = validate_scope(evidence.get("object_scope"), observations,
                                               visible_ids if visible_result_ids is None
                                               else visible_result_ids,
                                               answer if count_task else None)
                if task and task["quantity"] == "count_ratio":
                    checked_scope["ratio"] = validate_count_groups(
                        evidence.get("count_groups"), evidence["object_scope"], answer, task.get("operands", ()))
            except ValueError as exc:
                if isinstance(exc, ScopeError):
                    fail(str(exc), exc.field, **exc.details)
                fail(str(exc), "evidence.object_scope")
        elif "count_groups" in evidence:
            fail("object_scope_ratio_invalid", "evidence.count_groups")
        result = {**evidence, "verified_arithmetic": False, "origin_validation": origin}
        if checked_scope:
            result["scope_validation"] = checked_scope
        return result
    comparison = evidence.get("kind") == "tool_comparison"
    required = ({"kind", "left", "right", "operator", "answers", "bindings"} if comparison
                else {"kind", "expression", "bindings"})
    estimate = evidence.get("kind") == "tool_estimate"
    allowed = required | {"scale", "provenance"} | ({"basis"} if estimate else set()) | ({"assumptions"} if volume_task else set())
    if (not required <= set(evidence) or set(evidence) - allowed
            or evidence.get("kind") not in ("calculation", "tool_comparison", "tool_estimate")):
        fail("answer_evidence_fields", "evidence",
             **field_details(evidence, required, allowed))
    if estimate and (not isinstance(evidence.get("basis"), str)
                     or not evidence["basis"].strip() or len(evidence["basis"]) > 200):
        fail("visual_basis_required_max200", "evidence.basis")
    bindings = evidence["bindings"]
    if not isinstance(bindings, dict) or not 1 <= len(bindings) <= 16:
        fail("calculation_bindings_required", "evidence.bindings",
             min_bindings=1, max_bindings=16)
    values, ids, units = {}, [], {}
    if any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,15}", name)
           for name in bindings):
        fail("calculation_binding_fields", "evidence.bindings",
             repair="Binding names must match [a-z][a-z0-9_]{0,15}.")
    # Stable error selection across sorted on-disk JSON and in-memory mappings.
    for name in sorted(bindings):
        binding = bindings[name]
        field = f"evidence.bindings.{name}"
        if not isinstance(binding, dict):
            fail("calculation_binding_fields", field, expected_type="object")
        required, allowed = {"result_id", "quantity"}, {"result_id", "quantity", "geometry", "sample_xy", "reference", "operand_index"}
        if not required <= set(binding) or set(binding) - allowed:
            fail("calculation_binding_fields", field, **field_details(binding, required, allowed))
        key, quantity = binding["result_id"], binding["quantity"]
        if not isinstance(key, str) or key not in known:
            fail("calculation_measurement_unknown", field + ".result_id")
        if key not in visible_ids:
            fail("calculation_measurement_not_delivered", field + ".result_id")
        if any(r["data"].get("replaces_result_id") == key for r in known.values()):
            fail("calculation_measurement_replaced", field + ".result_id")
        review = reviews.get(key)
        if not review or review["decision"] != "usable":
            fail("calculation_measurement_not_usable", field + ".result_id",
                 expected_review="usable", actual_review=review["decision"] if review else "unreviewed")
        row = known[key]
        # Establish the measurement's intrinsic quantity before its task role.
        # An illegal 'distance' must not be reported as an actual chord.
        if row["source_tool"] == "point_depth" and quantity != "camera_z":
            fail("point_is_camera_z", field + ".quantity", actual=quantity,
                 actual_type=type(quantity).__name__, allowed_quantities=["camera_z"],
                 measurement_result_id=key, source_tool=row["source_tool"])
        if row["source_tool"] != "point_depth" and quantity not in ("chord", *DIMENSIONS):
            fail("distance_quantity_invalid", field + ".quantity", actual=quantity,
                 actual_type=type(quantity).__name__, allowed_quantities=["chord", *DIMENSIONS],
                 measurement_result_id=key, source_tool=row["source_tool"])
        if task and task["quantity"] != "visual_judgment":
            surfaces, targets = review.get("observed_surfaces"), row["data"].get("intended_targets")
            if not surfaces or not targets:
                fail("measurement_surface_required", field + ".result_id",
                     measurement_result_id=key,
                     review_field="measurement_reviews[].observed_surfaces",
                     sampling_field="targets")
            if row["source_tool"] == "point_depth" and binding.get("sample_xy", row["data"]["pixel_xy"]) != (
                    surfaces[0].get("sample_xy", row["data"]["pixel_xy"])):
                fail("measurement_surface_required", field + ".sample_xy",
                     repair="Review the exact selected sample_xy; an alternative pixel cannot inherit a center-point review.")
            asked = task["quantity"]
            allowed = {
                "height": set(DIMENSIONS), "width": set(DIMENSIONS), "length": set(DIMENSIONS),
                "diameter": set(DIMENSIONS), "dimension_ratio": set(DIMENSIONS),
                "dimension_comparison": set(DIMENSIONS),
                "height_ratio": {"height"}, "width_ratio": {"width"}, "length_ratio": {"length"},
                "volume_ratio": set(DIMENSIONS), "camera_order": {"camera_z"},
                "object_distance_order": {"chord"}, "object_distance": {"chord", *DIMENSIONS},
            }.get(asked, set())
            operands = task.get("operands", [])
            declared_quantities = {o["quantity"] for o in operands}
            if operands and asked not in ("volume", "volume_ratio"):
                allowed = {("chord" if q == "object_distance" else q)
                           for q in declared_quantities if q in (*DIMENSIONS, "camera_z", "object_distance")}
            if quantity not in allowed:
                fail("task_quantity_mismatch", field + ".quantity", actual=quantity,
                     requested_quantity=asked, allowed_quantities=sorted(allowed),
                     measurement_result_id=key)
            if asked == "object_distance_order" and task["reference"] is None:
                fail("task_quantity_mismatch", field + ".reference",
                     repair="The public reference object could not be resolved; use an RGB estimate rather than an unbound object-distance comparison.")
            if task["reference"] is not None:
                if binding.get("reference") != task["reference"]:
                    fail("task_quantity_mismatch", field + ".reference", expected=task["reference"],
                         actual=binding.get("reference"),
                         repair="Declare the resolved common reference or fixed origin. The pair endpoints must connect that object to one candidate; an entire 'or' phrase is not an object.")
                if asked == "object_distance_order" and not task.get("distance_relation") and targets[1]["target"] != task["reference"]:
                    fail("task_quantity_mismatch", field + ".reference", expected=task["reference"])
        unit = row["data"].get("unit", "per_view_relative")
        view = row["data"].get("view_index", 0)
        units[name] = {f"relative_view_{view}": 1} if unit == "per_view_relative" else {unit: 1}
        if row["source_tool"] == "point_depth":
            if "geometry" in binding:
                fail("calculation_binding_fields", field + ".geometry",
                     repair="Remove geometry from point_depth bindings; camera_z is not a full object dimension.")
            if "sample_xy" in binding and binding["sample_xy"] is None:
                fail("calculation_sample_xy_invalid", field + ".sample_xy")
            try:
                values[name] = depth_value(row["data"], binding.get("sample_xy"))
            except ValueError as exc:
                fail(str(exc), field + (".sample_xy" if "sample_xy" in binding else ""),
                     selection="sample_xy" if "sample_xy" in binding else "raw_center")
        else:
            if "sample_xy" in binding:
                fail("calculation_binding_fields", field + ".sample_xy",
                     repair="Pair distance uses its measured center endpoints; remove sample_xy or remeasure different endpoints.")
            if quantity in DIMENSIONS:
                geometry = binding.get("geometry")
                axis, anchors = DIMENSIONS[quantity]
                geometry_field = field + ".geometry"
                geometry_fields = {"coverage", "axis", "anchors", "objects"}
                if not isinstance(geometry, dict):
                    fail("complete_dimension_not_declared", geometry_field,
                         expected_fields=sorted(geometry_fields))
                if set(geometry) != geometry_fields:
                    fail("complete_dimension_not_declared", geometry_field,
                         **field_details(geometry, geometry_fields, geometry_fields))
                if geometry["coverage"] != "complete":
                    fail("complete_dimension_not_declared", geometry_field + ".coverage",
                         expected="complete")
                if geometry["axis"] != axis:
                    fail("complete_dimension_not_declared", geometry_field + ".axis", expected=axis)
                if (not isinstance(geometry["anchors"], list) or len(geometry["anchors"]) != 2
                        or any(not isinstance(a, str) for a in geometry["anchors"])
                        or set(geometry["anchors"]) != anchors):
                    fail("complete_dimension_not_declared", geometry_field + ".anchors",
                         expected=sorted(anchors))
                if (not isinstance(geometry["objects"], list) or len(geometry["objects"]) != 2
                        or any(not isinstance(o, str) or not o.strip() or len(o) > 80
                               for o in geometry["objects"])
                        or geometry["objects"][0] != geometry["objects"][1]):
                    fail("complete_dimension_not_declared", geometry_field + ".objects",
                         repair="Declare two matching nonempty object names (each <=80 characters), only if both full extrema are on that same object.")
                if task and task["quantity"] != "visual_judgment":
                    for index, (surface, obj, anchor) in enumerate(zip(
                            review["observed_surfaces"], geometry["objects"], geometry["anchors"])):
                        expected_surface = {"target": obj, "extent_role": anchor, "visibility": "visible"}
                        actual_surface = {**surface, "extent_role": surface.get("extent_role", surface.get("surface"))}
                        mismatch = next((k for k, v in expected_surface.items()
                                         if (normalize_object_name(actual_surface.get(k, "")) != normalize_object_name(v)
                                             if k == "target" else actual_surface.get(k) != v)), None)
                        if mismatch:
                            fail("complete_dimension_not_declared", geometry_field,
                                 measurement_result_id=key, point_index=index,
                                 review_field=f"measurement_reviews[].observed_surfaces[{index}].{mismatch}",
                                 actual=actual_surface.get(mismatch), expected=expected_surface[mismatch],
                                 intended_surface=row["data"]["intended_targets"][index]["surface"],
                                 observed_surface=surface.get("observed_surface", surface.get("surface")),
                                 required_extent_role=anchor,
                                 repair="Keep the actual sampled/observed surface. Declare a separate extent_role "
                                        "with an RGB extent_basis only if it is a complete extremum on the same object; "
                                        "otherwise resample or mark uncertain. Do not repair this by renaming surfaces.")
                        if surface.get("issues"):
                            fail("complete_dimension_not_declared", geometry_field,
                                 measurement_result_id=key, point_index=index,
                                 repair="Resolve the recorded RGB hit/extent issues before declaring a complete dimension.")
            elif "geometry" in binding:
                fail("calculation_binding_fields", field + ".geometry",
                     quantity="chord",
                     repair="chord forbids geometry: remove geometry if this is only a chord. Use height/width/length plus matching geometry only for visually confirmed complete same-object extrema.")
            values[name] = row["data"]["distance"]
            if values[name] is None:
                fail("calculation_endpoint_invalid", field + ".result_id")
        ids.append(key)
    if evidence.get("provenance") is not None and set(origin["referenced_result_ids"]) != set(ids):
        fail("answer_provenance_conflict", "evidence.provenance.referenced_result_ids",
             expected_result_ids=sorted(set(ids)),
             repair="List exactly the measurement IDs used by the checked bindings; abandoned values must not be in the expression.")
    if evidence.get("provenance") is None:
        origin["referenced_result_ids"] = list(dict.fromkeys(ids))
    scale = evidence.get("scale")
    if scale is not None:
        if (task is None or not isinstance(scale, dict)
                or set(scale) != {"name", "value", "unit", "anchor_binding"}
                or not isinstance(scale["name"], str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,15}", scale["name"])
                or scale["name"] in values or not isinstance(scale["unit"], str)
                or scale["unit"] not in LENGTH_UNIT_METERS
                or type(scale["value"]) not in (int, float)
                or not isinstance(scale["anchor_binding"], str)
                or scale["anchor_binding"] not in bindings):
            fail("scale_anchor_invalid", "evidence.scale")
        anchor = bindings[scale["anchor_binding"]]
        targets = known[anchor["result_id"]]["data"].get("intended_targets")
        if not targets or anchor["quantity"] not in DIMENSIONS:
            fail("scale_anchor_invalid", "evidence.scale")
        target = normalize_object_name(targets[0]["target"])
        if not target:
            fail("scale_anchor_invalid", "evidence.scale")
        scale_value = finite(scale["value"])
        if not any(public_span_matches(p, scale_value, scale["unit"])
                   and p["quantity"] == anchor["quantity"]
                   and re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)",
                                 normalize_object_name(p["source"]))
                   for p in task["public_metric_values"]):
            fail("scale_anchor_invalid", "evidence.scale", public_metric_values=task["public_metric_values"],
                 measurement_result_id=anchor["result_id"], sampled_target=targets[0]["target"],
                 anchor_quantity=anchor["quantity"],
                 repair="Use the exact public anchor name and its stated dimension/unit. "
                        "Names only normalize case, spaces and underscores; correct targets and resample "
                        "when the sampled identity is wrong. Do not substitute an unverified synonym.")
        values[scale["name"]] = scale_value
        units[scale["name"]] = {scale["unit"]: 1}
    operand_groups = None
    if task and task.get("operands") and task["quantity"] not in ("volume", "volume_ratio"):
        operand_groups = match_operands(task, bindings, observations, scale, fail)
    if comparison:
        op, answers = evidence["operator"], evidence["answers"]
        labels = {"less", "equal", "greater"} if op == "three_way" else {"true", "false"}
        if (op not in ("<", "<=", ">", ">=", "==", "!=", "three_way")
                or not isinstance(answers, dict) or set(answers) != labels):
            fail("tool_comparison_invalid", "evidence")
        left, used_left = expression_value(evidence["left"], values, require_all=False, return_used=True)
        right, used_right = expression_value(evidence["right"], values, require_all=False, return_used=True)
        if used_left | used_right != set(values):
            fail("expression_bindings_unused", "evidence")
        if expression_unit(evidence["left"], units) != expression_unit(evidence["right"], units):
            fail("comparison_unit_mismatch", "evidence")
        if operand_groups is not None:
            validate_expression(task, evidence, operand_groups, fail)
        equal = math.isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        truth = {"<": left < right, "<=": left <= right, ">": left > right, ">=": left >= right,
                 "==": equal, "!=": not equal}
        key = ("equal" if equal else "less" if left < right else "greater") if op == "three_way" else (
            "true" if truth[op] else "false")
        try:
            expected = canonical(answers[key], task) if task else scalar(answers[key])
            if task:
                for label in answers.values():
                    canonical(label, task)
        except ValueError:
            fail("answer_option_invalid", "evidence.answers")
        if answer != expected:
            fail("tool_comparison_answer_mismatch", "answer", expected=expected, relation=key,
                 left_value=left, right_value=right)
        return {**evidence, "result_ids": ids, "comparison_result": key, "origin_validation": origin,
                "left_value": left, "right_value": right, "verified_arithmetic": True,
                "semantic_verification": "same_model_declaration",
                "validation_layers": {"arithmetic": True, "task_unit_contract": task is not None,
                                      "surface_identity": "same_model_declaration",
                                      "independent_visual_geometry": False}}
    calculated = expression_value(evidence["expression"], values)
    if operand_groups is not None:
        validate_expression(task, evidence, operand_groups, fail)
    if task:
        if task["quantity"] in DIMENSIONS and not any(
                b["quantity"] == task["quantity"] for b in bindings.values()):
            fail("task_quantity_mismatch", "evidence.bindings", requested_quantity=task["quantity"])
        if task["quantity"] == "object_distance" and not any(
                b["quantity"] == "chord" for b in bindings.values()):
            fail("task_quantity_mismatch", "evidence.bindings", requested_quantity="object_distance")
        expected_unit = {} if task["unit"] == "ratio" else {task["unit"]: 1}
        if task["unit"] in (*LENGTH_UNIT_METERS, "ratio", "count") and expression_unit(
                evidence["expression"], units) != expected_unit:
            fail("task_unit_mismatch", "evidence.expression", expected=task["unit"])
    finite(answer, "answer")
    if not math.isclose(calculated, answer, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        fail("answer_calculation_mismatch", "answer", submitted=answer, calculated=calculated,
             abs_error=abs(calculated-answer), rel_tol=REL_TOL, abs_tol=ABS_TOL,
             allowed_error=max(ABS_TOL, REL_TOL*max(abs(calculated), abs(answer))))
    return {**evidence, "result_ids": ids, "calculated_value": calculated, "origin_validation": origin,
            "verified_arithmetic": True, "semantic_verification": "same_model_declaration",
            "validation_layers": {"arithmetic": True, "task_unit_contract": task is not None,
                                  "surface_identity": "same_model_declaration",
                                  "independent_visual_geometry": False}}
