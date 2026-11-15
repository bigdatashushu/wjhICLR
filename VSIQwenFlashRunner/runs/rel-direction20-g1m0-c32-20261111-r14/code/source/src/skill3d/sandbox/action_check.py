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
    "crop_frame_count_mismatch", "invalid_crop_box", "frame_slot_out_of_range",
})

OBSERVATION_FALLTHROUGH_VERSION = "observation-fallthrough/1.0"


def permits_observation_fallthrough(source: str, *, tool_names: set[str]) -> bool:
    """Only a missing observation control may be adapted, never an answer.

    Static intent is necessary but insufficient: the kernel must also observe
    successful evidence in this cell. A helper definition or dead branch alone
    cannot make a host yield, and historical ledger entries cannot qualify.
    """
    tree = ast.parse(normalize_program_source(source))
    if any((isinstance(node, ast.Name) and node.id in {
            "ReturnAnswer", "YieldObservations", "AnswerPayload"})
           or (isinstance(node, ast.Attribute) and node.attr in {
               "ReturnAnswer", "YieldObservations", "AnswerPayload"})
           for node in ast.walk(tree)):
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif (isinstance(node.func, ast.Attribute)
              and ast.unparse(node.func.value) in {"tools", "ctx.tools"}):
            name = node.func.attr
        else:
            continue
        if name in tool_names | {"ReadObservation"}:
            return True
    return False


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
        if called & (set(tool_names or ()) | {"YieldObservations", "ReadObservation"}):
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
