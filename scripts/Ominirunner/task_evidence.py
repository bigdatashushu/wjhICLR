"""Ordered task operands and bounded symbolic checks for public arithmetic."""
from __future__ import annotations

import ast
from fractions import Fraction

from .answer_contract import normalize_object_name


def same_target(actual, expected):
    return expected is None or normalize_object_name(actual) == normalize_object_name(expected)


def operand_matches(operand, binding, targets, relation=None):
    quantity = "chord" if operand["quantity"] == "object_distance" else operand["quantity"]
    if binding["quantity"] != quantity or not targets:
        return False
    if relation:
        anchor = relation.get("reference", relation.get("origin"))
        pair = [t["target"] for t in targets]
        return (len(pair) == 2 and
                (same_target(pair[0], operand["target"]) and same_target(pair[1], anchor)
                 or same_target(pair[1], operand["target"]) and same_target(pair[0], anchor)))
    return all(same_target(t["target"], operand["target"]) for t in targets)


def match_operands(task, bindings, observations, scale, fail):
    """Map bindings to public roles; a distance anchor is not another measurement."""
    operands, relation = task.get("operands", []), task.get("distance_relation")
    rows = {r["result_id"]: r for r in observations}
    groups = {}
    selected_scale = scale.get("anchor_binding") if isinstance(scale, dict) else None
    for name, binding in bindings.items():
        targets = rows[binding["result_id"]]["data"].get("intended_targets", [])
        matches = []
        for index, operand in enumerate(operands):
            if relation and operand["role"] in ("origin", "reference"):
                continue
            if operand["role"] == "scale_anchor" and selected_scale is not None and name != selected_scale:
                continue
            if operand_matches(operand, binding, targets,
                               relation if relation and task["quantity"] == "object_distance_order" else None):
                matches.append(index)
        declared = binding.get("operand_index")
        if declared is not None:
            if type(declared) is not int or declared not in matches:
                fail("task_operand_mismatch", f"evidence.bindings.{name}.operand_index",
                     matching_operand_indices=matches, actual=declared)
            matches = [declared]
        if not matches:
            fail("task_operand_mismatch", f"evidence.bindings.{name}",
                 measurement_result_id=binding["result_id"], actual_quantity=binding["quantity"],
                 sampled_targets=[t["target"] for t in targets], public_operands=operands)
        for index in matches:
            groups.setdefault(index, []).append(name)
    for index, operand in enumerate(operands):
        if relation and operand["role"] in ("origin", "reference"):
            continue
        if operand["role"] == "scale_anchor":
            continue
        if index not in groups:
            fail("task_operand_missing", "evidence.bindings",
                 operand_index=index, operand=operand)
    return groups


def role_expression(operands, groups, role):
    names = [name for index, operand in enumerate(operands) if operand["role"] == role
             for name in groups.get(index, ())]
    return "(" + "+".join(dict.fromkeys(names)) + ")" if names else None


def equivalent(expression, expected):
    """Exact rational-polynomial equivalence over a small validated AST.

    Bounds apply to intermediate polynomials too. No numerical coincidences or
    finite probing is used to certify the order of task operands.
    """
    def add(a, b, sign=1):
        out = dict(a)
        for key, coefficient in b.items():
            out[key] = out.get(key, 0) + sign * coefficient
            if not out[key]:
                del out[key]
        if len(out) > 256:
            raise ValueError("symbolic expression too large")
        return out

    def multiply(a, b):
        out = {}
        for ka, va in a.items():
            for kb, vb in b.items():
                key = tuple(sorted((*ka, *kb)))
                out[key] = out.get(key, 0) + va * vb
                if len(out) > 256 or len(key) > 64:
                    raise ValueError("symbolic expression too large")
        return {k: v for k, v in out.items() if v}

    one = {(): Fraction(1)}

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return {(): Fraction(str(node.value))}, one
        if isinstance(node, ast.Name):
            return {(node.id,): Fraction(1)}, one
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            a, b = visit(node.operand)
            return ({k: -v for k, v in a.items()} if isinstance(node.op, ast.USub) else a), b
        if not isinstance(node, ast.BinOp):
            raise ValueError("unsupported symbolic expression")
        a, b = visit(node.left)
        c, d = visit(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return add(multiply(a, d), multiply(c, b), -1 if isinstance(node.op, ast.Sub) else 1), multiply(b, d)
        if isinstance(node.op, ast.Mult):
            return multiply(a, c), multiply(b, d)
        if isinstance(node.op, ast.Div):
            return multiply(a, d), multiply(b, c)
        raise ValueError("unsupported symbolic expression")

    a, b = visit(ast.parse(expression, mode="eval").body)
    c, d = visit(ast.parse(expected, mode="eval").body)
    return multiply(a, d) == multiply(c, b)


def validate_expression(task, evidence, groups, fail):
    operands = task.get("operands", [])
    if task.get("operation") == "ratio":
        numerator = role_expression(operands, groups, "numerator")
        denominator = role_expression(operands, groups, "denominator")
        expected = numerator + "/" + denominator
        if not equivalent(evidence["expression"], expected):
            fail("task_expression_mismatch", "evidence.expression", expected_expression=expected)
    elif task.get("operation") == "difference" and evidence.get("scale"):
        target = role_expression(operands, groups, "target")
        subtrahend = role_expression(operands, groups, "subtrahend")
        scale = evidence["scale"]
        expected = scale["name"] + "*(" + target + "-" + subtrahend + ")/" + scale["anchor_binding"]
        if not equivalent(evidence["expression"], expected):
            fail("task_expression_mismatch", "evidence.expression", expected_expression=expected)
    elif task.get("operation") == "measure" and evidence.get("scale"):
        target = role_expression(operands, groups, "target")
        scale = evidence["scale"]
        expected = scale["name"] + "*" + target + "/" + scale["anchor_binding"]
        if not equivalent(evidence["expression"], expected):
            fail("task_expression_mismatch", "evidence.expression", expected_expression=expected)
    elif (task.get("operation") == "comparison"
          and any(o["role"] == "left" for o in operands)
          and all(o["quantity"] in ("height", "width", "length", "diameter")
                  for o in operands if o["role"] in ("left", "right"))):
        left, right = (role_expression(operands, groups, role) for role in ("left", "right"))
        comparison = task["comparison"]
        expected_right = str(comparison.get("factor", 1)) + "*" + right
        if (evidence.get("operator") != comparison["operator"]
                or not equivalent(evidence["left"], left)
                or not equivalent(evidence["right"], expected_right)):
            fail("task_expression_mismatch", "evidence",
                 expected_left=left, expected_right=expected_right,
                 expected_operator=comparison["operator"])
