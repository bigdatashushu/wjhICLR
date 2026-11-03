"""Native scalar bridge: never convert integral floats to integer predictions."""
from __future__ import annotations

import ast
import json
import math
import re

from skill3d.sandbox.kernel import AnswerTerminate, RestrictedNamespaceKernel


class ProgramFormatError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def program_feedback(code):
    repair = {
        "control_namespace_error": "Call global ReturnAnswer(...) or YieldObservations(...); "
                                   "ctx.tools contains only the four evidence tools.",
        "program_missing_action": "End solve with global YieldObservations(...) or ReturnAnswer(...).",
    }.get(code, "Output one closed python fence with only def solve(ctx):.")
    return [{"code": code, "repair": repair}]


def parse_program(text):
    """Require exactly the advertised code envelope; never execute XML/prose."""
    match = re.fullmatch(r"```python[ \t]*\n([\s\S]*?)\n?```", text.strip())
    if not match or "```" in match[1]:
        raise ProgramFormatError("python_fence_required")
    program = match[1]
    try:
        tree = ast.parse(program)
    except SyntaxError as exc:
        raise ProgramFormatError("python_syntax_error") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ProgramFormatError("solve_entry_required")
    entry = tree.body[0]
    args = entry.args
    if (entry.name != "solve" or entry.decorator_list or entry.returns
            or args.posonlyargs or args.kwonlyargs or args.kwarg or args.vararg
            or args.defaults or len(args.args) != 1
            or args.args[0].arg != "ctx" or args.args[0].annotation):
        raise ProgramFormatError("solve_entry_required")
    return program


def control_namespace_errors(program):
    paths = {"ctx", "tools", "ctx.tools"}
    return sorted({node.attr for node in ast.walk(ast.parse(program))
                   if isinstance(node, ast.Attribute)
                   and node.attr in {"ReturnAnswer", "YieldObservations"}
                   and ast.unparse(node.value) in paths})


def scalar(value):
    if type(value) not in (str, int, float) or (
            type(value) is float and not math.isfinite(value)):
        raise ValueError("native_finite_scalar_required")
    return str(value)


def parse_json(text):
    raw = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", raw)
    if match:
        raw = match[1]

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result

    parsed = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(parsed, dict) or set(parsed) != {"answer"}:
        raise ValueError("scalar_answer_json_required")
    scalar(parsed["answer"])
    return parsed["answer"]


class NativeSlot:
    def __init__(self):
        self.answer = self.payload = self.raw_value = self.evidence = None
        self.given = False
        self.adapter_problems = []

    def __call__(self, value, evidence=None):
        if self.given:
            raise ValueError("duplicate_native_answer")
        self.answer = scalar(value)
        self.raw_value = value
        # Snapshot literals before leaving the generated namespace.
        self.evidence = json.loads(json.dumps(evidence, allow_nan=False))
        self.given = True
        raise AnswerTerminate(self.answer)


class NativeKernel(RestrictedNamespaceKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_slot = NativeSlot()
        self._ns["ReturnAnswer"] = lambda value, evidence=None: self.answer_slot(value, evidence)

    def reset_user_namespace(self):
        super().reset_user_namespace()
        self.answer_slot.raw_value = None
        self.answer_slot.evidence = None
