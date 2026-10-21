"""Action validation, separate from the syntax/security whitelist.

Only definitely absent controls are rejected statically. A control referenced
through a helper or conditional is left to runtime; this is not a proof of flow.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from skill3d.sandbox.ast_guard import normalize_program_source

ARGUMENT_ERROR_REASONS = frozenset({
    "invalid_tool_arguments", "invalid_point_argument", "invalid_candidate_categories",
    "direction_arguments_invalid",
})


@dataclass(frozen=True)
class ActionCheck:
    ok: bool
    error_code: str = ""
    reason: str = ""


def check_action(source: str, *, finalizing: bool = False,
                 tool_names: set[str] | None = None) -> ActionCheck:
    tree = ast.parse(normalize_program_source(source))
    controls = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        and node.id in {"ReturnAnswer", "YieldObservations"}
    }
    # Attribute access is allowed only as a conservative possible action here;
    # the AST guard and kernel still decide whether the interface exists.
    controls.update(node.attr for node in ast.walk(tree)
                    if isinstance(node, ast.Attribute)
                    and node.attr in {"ReturnAnswer", "YieldObservations"})
    if finalizing:
        called = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                  for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, (ast.Name, ast.Attribute))}
        if called & (set(tool_names or ()) | {"YieldObservations"}):
            return ActionCheck(False, "finalization_tool_call_disallowed",
                               "Finalization cannot call evidence tools or YieldObservations.")
        if "ReturnAnswer" not in controls:
            return ActionCheck(False, "finalization_missing_submission",
                               "No submission action exists; an answer variable is not submitted.")
    elif not controls:
        return ActionCheck(False, "program_missing_action",
                           "No YieldObservations or ReturnAnswer action exists. "
                           "Assignment and print do not submit; call a control interface.")
    return ActionCheck(True)
