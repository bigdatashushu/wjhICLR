"""Declared origins and narrowly provable program dependencies, without tainting numbers."""
from __future__ import annotations

import ast

VERSION = "omni-answer-origin/1"


def program_inputs(program, observations, fresh_ids=()):
    """Trace straight-line assignments from tool dictionaries to ReturnAnswer.

    Literal numbers, comments, matching answers and prose are never dependency
    proof. Branches, loops, helper calls and dynamic Python are deliberately not
    certified by this analysis; the origin declaration and host audit remain.
    """
    result = {"version": VERSION, "result_ids": [], "coverage": "straight_line_only",
              "dynamic_dependencies_verified": False}
    if not program:
        return result
    try:
        tree = ast.parse(program)
        body = tree.body[0].body
    except (SyntaxError, IndexError, AttributeError):
        return result
    known = {r["result_id"]: r for r in observations}
    fresh = [known[k] for k in fresh_ids if k in known]
    cursor, names = 0, {}

    def inputs(node):
        if isinstance(node, ast.Name):
            return names.get(node.id, set())
        return set().union(*(inputs(child) for child in ast.iter_child_nodes(node)))

    for node in body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = node.value
            source = None
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and isinstance(value.func.value, ast.Attribute)
                    and isinstance(value.func.value.value, ast.Name)
                    and value.func.value.value.id == "ctx" and value.func.value.attr == "tools"):
                source = value.func.attr
            if source:
                if cursor >= len(fresh) or fresh[cursor]["source_tool"] != source:
                    # Nonmeasurement calls cannot be aligned to a numeric receipt.
                    result["coverage"] = "partial_straight_line"
                    break
                names[node.targets[0].id] = {fresh[cursor]["result_id"]}
                cursor += 1
            else:
                names[node.targets[0].id] = inputs(value)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name)
              and node.value.func.id == "ReturnAnswer" and node.value.args):
            result["result_ids"] = sorted(inputs(node.value.args[0]))
            break
        elif isinstance(node, (ast.Assert, ast.Pass)):
            continue
        else:
            result["coverage"] = "partial_straight_line"
            break
    return result


def declared_ids(evidence):
    origin = evidence.get("provenance") if isinstance(evidence, dict) else None
    values = origin.get("referenced_result_ids") if isinstance(origin, dict) else None
    return [k for k in values if isinstance(k, str)] if isinstance(values, list) else []
