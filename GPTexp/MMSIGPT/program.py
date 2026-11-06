"""Narrow source compatibility before the unchanged sandbox AST guard."""
from __future__ import annotations

import ast
import hashlib

from .public_input import INTERFACE_VERSION, PUBLIC_CONTEXT_FIELDS


def canonical_source(source):
    """Adapt only injected control imports and literal public ctx lookups.

    Other imports, aliases, reflection and defaults still reach the original guard.
    Keep line counts so execution diagnostics refer to the supplied source lines.
    AST column offsets are UTF-8 byte offsets, not Python string offsets.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(line, column):
        return starts[line - 1] + len(lines[line - 1].encode("utf-8")[:column].decode("utf-8"))

    edits = []
    for node in ast.walk(tree):
        replacement = kind = None
        if (isinstance(node, ast.ImportFrom) and node.level == 0
                and node.module == "harness3d" and node.names
                and all(item.name in {"ReturnAnswer", "AnswerPayload"}
                        and item.asname is None for item in node.names)):
            replacement, kind = "pass", "injected_control_import"
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "getattr" and not node.keywords
              and len(node.args) in (2, 3)
              and isinstance(node.args[0], ast.Name) and node.args[0].id == "ctx"
              and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)
              and node.args[1].value in PUBLIC_CONTEXT_FIELDS
              and (len(node.args) == 2 or (
                  isinstance(node.args[2], ast.Constant) and node.args[2].value is None))):
            replacement, kind = f"ctx.{node.args[1].value}", "literal_public_context_lookup"
        if replacement is not None:
            newlines = node.end_lineno - node.lineno
            if newlines:
                expression = "None" if replacement == "pass" else replacement
                replacement = "(" + expression + "\n" * newlines + ")"
            edits.append((offset(node.lineno, node.col_offset),
                          offset(node.end_lineno, node.end_col_offset), replacement, kind))
    canonical = source
    for start, end, replacement, _kind in sorted(edits, reverse=True):
        canonical = canonical[:start] + replacement + canonical[end:]
    ast.parse(canonical)
    return canonical, [kind for _, _, _, kind in sorted(edits)]


def assemble_program(model_output, *args, adaptations=None, **kwargs):
    from skill3d.synthesis.program_assembler import assemble_program as shared_assemble

    original = shared_assemble(model_output, *args, **kwargs)
    canonical, kinds = canonical_source(original.program_source)
    original_sha = hashlib.sha256(original.program_source.encode()).hexdigest()
    canonical_sha = hashlib.sha256(canonical.encode()).hexdigest()
    if adaptations is not None:
        adaptations.append({
            "interface_version": INTERFACE_VERSION,
            "stage": "program_assembly",
            "original_program_sha256": original_sha,
            "canonical_program_sha256": canonical_sha,
            "adaptations": kinds,
        })
    return type(original)(**{
        **original.model_dump(),
        "program_source": canonical,
        "program_id": canonical_sha[:16],
    })
