"""Strict program envelope and option-letter control bridge."""
from __future__ import annotations

import ast
import json
import re

from skill3d.sandbox.kernel import AnswerTerminate, RestrictedNamespaceKernel


class ProgramFormatError(ValueError):
    pass


def parse_program(text):
    match = re.fullmatch(r"```python[ \t]*\n([\s\S]*?)\n?```", text.strip())
    if not match or "```" in match[1]:
        raise ProgramFormatError("python_fence_required")
    try:
        tree = ast.parse(match[1])
    except SyntaxError as exc:
        raise ProgramFormatError("python_syntax_error") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ProgramFormatError("solve_entry_required")
    entry, args = tree.body[0], tree.body[0].args
    if (entry.name != "solve" or entry.decorator_list or entry.returns
            or args.posonlyargs or args.kwonlyargs or args.kwarg or args.vararg or args.defaults
            or len(args.args) != 1 or args.args[0].arg != "ctx" or args.args[0].annotation):
        raise ProgramFormatError("solve_entry_required")
    return match[1]


def wrong_controls(program):
    return sorted({node.attr for node in ast.walk(ast.parse(program))
                   if isinstance(node, ast.Attribute)
                   and node.attr in {"ReturnAnswer", "YieldObservations"}
                   and ast.unparse(node.value) in {"ctx", "tools", "ctx.tools"}})


def option_letter(value, labels):
    if type(value) is not str or value not in labels:
        raise ValueError("current_option_letter_required")
    return value


def parse_json(text, labels):
    raw = text.strip()
    match = re.fullmatch(r"```json[ \t]*\n([\s\S]*?)\n?```", raw)
    if match:
        raw = match[1]

    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(value, dict) or set(value) != {"answer"}:
        raise ValueError("answer_json_required")
    return option_letter(value["answer"], labels)


class OptionSlot:
    def __init__(self, labels):
        self.labels = tuple(labels)
        self.answer = self.payload = None
        self.given = False
        self.adapter_problems = []

    def __call__(self, value):
        if self.given:
            raise ValueError("duplicate_answer")
        self.answer = option_letter(value, self.labels)
        self.given = True
        raise AnswerTerminate(self.answer)


class OptionKernel(RestrictedNamespaceKernel):
    def __init__(self, *args, labels, **kwargs):
        self.trace = None
        super().__init__(*args, **kwargs)
        self.answer_slot = OptionSlot(labels)

    def _make_tool_fn(self, name):
        original = super()._make_tool_fn(name)

        def traced(*args, **kwargs):
            trace = self.trace
            if trace is None:
                return original(*args, **kwargs)
            start_calls, start_results = len(self.tool_calls), len(self.tool_results)
            invocation = f"tool-{len(trace.events):06d}"
            trace.event("tool_started", {"invocation_id": invocation, "tool": name,
                                         "positional_args": args, "keyword_args": kwargs})
            error = None
            try:
                return original(*args, **kwargs)
            except BaseException as exc:
                error = {"type": type(exc).__name__}
                raise
            finally:
                trace.event("tool_finished", {
                    "invocation_id": invocation, "tool": name, "error": error,
                    "calls": [c.model_dump(mode="json") for c in self.tool_calls[start_calls:]],
                    "results": [r.model_dump(mode="json") for r in self.tool_results[start_results:]]})
        return traced
