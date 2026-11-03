"""Bounded arithmetic over declared, delivered and usable measurement values."""
from __future__ import annotations

import ast
import math
import operator
import re

from .native import scalar
from .tools import MEASUREMENTS
from .sampling import depth_value

VERSION = "omni-answer-evidence/4"
REL_TOL = ABS_TOL = 1e-9
DIMENSIONS = {
    "height": ("object_vertical", {"top", "base"}),
    "width": ("object_horizontal", {"left", "right"}),
    "length": ("object_length", {"front", "back"}),
}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv}


REPAIRS = {
    "answer_evidence_required": "Provide evidence: visual_estimate with a short basis, or calculation with expression and bindings.",
    "answer_evidence_fields": "Use exactly kind+basis for visual_estimate, or kind+expression+bindings for calculation.",
    "visual_basis_required_max200": "Use a nonempty visual_estimate basis of at most 200 characters; omit extra evidence fields.",
    "calculation_bindings_required": "Provide 1 to 16 named measurement bindings.",
    "calculation_binding_fields": "Each binding requires result_id and quantity; optional fields depend on the measurement type.",
    "calculation_measurement_unknown": "Use a successful measurement ID from visible TOOL FACTS.",
    "calculation_measurement_not_delivered": "Use only currently visible TOOL FACTS; new measurements need a subsequent review request.",
    "calculation_measurement_not_usable": "Review the target in RGB first; resample or uncertain measurements cannot support a calculation.",
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
    "calculation_numeric_required": "A calculation requires a native int or float answer. For a category/text answer use visual_estimate with a short comparison basis.",
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


def dependency_ids(evidence, observations, previous=()):
    """Retain known calculation inputs, including inputs from a failed answer."""
    known = {r["result_id"] for r in observations if r["source_tool"] in MEASUREMENTS}
    bindings = evidence.get("bindings") if isinstance(evidence, dict) else None
    keys = []
    if isinstance(evidence, dict) and evidence.get("kind") == "calculation" and isinstance(bindings, dict):
        keys = [b.get("result_id") for b in list(bindings.values())[:16] if isinstance(b, dict)]
    return list(dict.fromkeys(k for k in [*keys, *previous] if isinstance(k, str) and k in known))


def finite(value, field="evidence.expression"):
    if type(value) not in (int, float):
        fail("calculation_numeric_required", field, actual_type=type(value).__name__)
    try:
        if not math.isfinite(value) or abs(value) > 1e100:
            fail("calculation_number_out_of_range", field)
    except OverflowError:
        fail("calculation_number_out_of_range", field)
    return value


def expression_value(expression, bindings):
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
    if used != set(bindings) or not used:
        fail("expression_bindings_unused", "evidence.expression",
             unused_bindings=sorted(set(bindings) - used))
    return value


def validate(answer, evidence, observations, reviews, visible_ids):
    """Certifies provenance/arithmetic and declaration consistency, not semantics."""
    try:
        scalar(answer)
    except ValueError:
        fail("native_finite_scalar_required", "answer", actual_type=type(answer).__name__)
    known = {r["result_id"]: r for r in observations if r["source_tool"] in MEASUREMENTS}
    if evidence is None:
        if known:
            fail("answer_evidence_required", "evidence")
        return {"kind": "visual_estimate", "basis": "original RGB", "verified_arithmetic": False}
    if not isinstance(evidence, dict):
        fail("answer_evidence_fields", "evidence", expected_type="object")
    if evidence.get("kind") == "visual_estimate":
        basis = evidence.get("basis")
        if (set(evidence) != {"kind", "basis"} or not isinstance(basis, str)
                or not basis.strip() or len(basis) > 200):
            fail("visual_basis_required_max200", "evidence.basis", max_chars=200,
                 actual_chars=len(basis) if isinstance(basis, str) else None,
                 **field_details(evidence, {"kind", "basis"}, {"kind", "basis"}))
        return {**evidence, "verified_arithmetic": False}
    if set(evidence) != {"kind", "expression", "bindings"} or evidence["kind"] != "calculation":
        fail("answer_evidence_fields", "evidence",
             **field_details(evidence, {"kind", "expression", "bindings"},
                             {"kind", "expression", "bindings"}))
    bindings = evidence["bindings"]
    if not isinstance(bindings, dict) or not 1 <= len(bindings) <= 16:
        fail("calculation_bindings_required", "evidence.bindings",
             min_bindings=1, max_bindings=16)
    values, ids = {}, []
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
        required, allowed = {"result_id", "quantity"}, {"result_id", "quantity", "geometry", "sample_xy"}
        if not required <= set(binding) or set(binding) - allowed:
            fail("calculation_binding_fields", field, **field_details(binding, required, allowed))
        key, quantity = binding["result_id"], binding["quantity"]
        if not isinstance(key, str) or key not in known:
            fail("calculation_measurement_unknown", field + ".result_id")
        if key not in visible_ids:
            fail("calculation_measurement_not_delivered", field + ".result_id")
        review = reviews.get(key)
        if not review or review["decision"] != "usable":
            fail("calculation_measurement_not_usable", field + ".result_id",
                 expected_review="usable", actual_review=review["decision"] if review else "unreviewed")
        row = known[key]
        if row["source_tool"] == "point_depth":
            if quantity != "camera_z":
                fail("point_is_camera_z", field + ".quantity", expected="camera_z")
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
            if quantity not in ("chord", *DIMENSIONS):
                fail("distance_quantity_invalid", field + ".quantity",
                     allowed_quantities=["chord", *DIMENSIONS])
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
            elif "geometry" in binding:
                fail("calculation_binding_fields", field + ".geometry",
                     quantity="chord",
                     repair="chord forbids geometry: remove geometry if this is only a chord. Use height/width/length plus matching geometry only for visually confirmed complete same-object extrema.")
            values[name] = row["data"]["distance"]
            if values[name] is None:
                fail("calculation_endpoint_invalid", field + ".result_id")
        ids.append(key)
    calculated = expression_value(evidence["expression"], values)
    finite(answer, "answer")
    if not math.isclose(calculated, answer, rel_tol=REL_TOL, abs_tol=ABS_TOL):
        fail("answer_calculation_mismatch", "answer", submitted=answer, calculated=calculated,
             abs_error=abs(calculated-answer), rel_tol=REL_TOL, abs_tol=ABS_TOL,
             allowed_error=max(ABS_TOL, REL_TOL*max(abs(calculated), abs(answer))))
    return {**evidence, "result_ids": ids, "calculated_value": calculated,
            "verified_arithmetic": True, "semantic_verification": "same_model_declaration"}
