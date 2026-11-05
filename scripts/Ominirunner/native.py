"""Native scalar bridge: never convert integral floats to integer predictions."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import math
import re

from skill3d.sandbox.kernel import AnswerTerminate, RestrictedNamespaceKernel


MAX_DIAGNOSTIC_CHARS = 160
MAX_EXTRACT_CHARS = 350
MAX_FEEDBACK_CHARS = 1000
PENDING_VERSION = "omni-unexecuted-program/1"
_PYTHON_FENCE = r"```python[ \t]*\n([\s\S]*?)\n?```"
_XML_PROGRAM = r"<program>[ \t]*\n?([\s\S]*?)\n?</program>"
_PROGRAM_EXAMPLE = (
    "```python\n"
    "def solve(ctx):\n"
    "    measurement_reviews = []\n"
    '    YieldObservations([], reason="Review")\n'
    "```"
)


class ProgramFormatError(ValueError):
    def __init__(self, code, **details):
        super().__init__(code)
        self.code = code
        self.record = {**details, "code": code, "currentphase": "program",
                       "executed": False}


def program_feedback(code, phase="program"):
    """Fit one complete repair record inside messages()'s shared errors budget."""
    details = code.record if isinstance(code, ProgramFormatError) else {}
    code = details.get("code", code)
    repair = {
        "control_namespace_error": "Use global ReturnAnswer/YieldObservations, not ctx.tools.",
        "program_missing_action": "End solve with a global YieldObservations or ReturnAnswer call.",
        "python_syntax_error": "Fix reported syntax only; preserve pending code.",
        "solve_entry_required": "Keep only unannotated, undecorated def solve(ctx), no defaults.",
    }.get(code, "Remove wrappers only; resend exact code of the pending program.")
    record = {
        "code": code, "currentphase": phase, "executed": False,
        "repair": f"Not executed. {repair} Resend one closed python fence, no "
                  "prefix/tail/XML/thinking; never revert to an old plan or replay executed code. "
                  "FIRST: measurement_reviews=[] (literal updates if needed); "
                  "ReturnAnswer/YieldObservations are global controls. Example: envelope only.",
        "example": _PROGRAM_EXAMPLE,
    }
    for key in ("field", "reason", "fence_markers"):
        if key in details:
            record[key] = details[key]
    for key in ("invalid_prefix", "invalid_tail", "invalid_envelope"):
        if details.get(key):
            record[key] = {k: details[key][k] for k in ("start_line", "end_line", "chars")}
    if "syntax" in details:
        record["syntax"] = {k: v for k, v in details["syntax"].items() if k != "message"}
    if "unexecuted_program" in details:
        record["unexecuted_program"] = deepcopy(details["unexecuted_program"])
        record["unexecuted_program"].pop("max_code_chars", None)
    # Locations and identity outrank code. Omit the whole code, never truncate a program.
    if len(json.dumps([record])) > MAX_FEEDBACK_CHARS and "unexecuted_program" in record:
        snapshot = record["unexecuted_program"]
        if "code" in snapshot:
            snapshot.pop("code")
            snapshot["code_omitted"] = "feedback_budget"
    for key in ("fence_markers", "reason", "field"):
        if len(json.dumps([record])) > MAX_FEEDBACK_CHARS:
            record.pop(key, None)
    return [record]


def context_feedback(error, phase):
    """A previous program rejection cannot advertise program execution in final."""
    if phase != "final" or error.get("currentphase") != "program":
        return error
    record = deepcopy(error)
    record.update(currentphase="final", rejected_phase="program",
                  repair="The previous program was not executed. Tools are now closed; "
                         "output the final answer JSON with evidence from delivered facts or independent RGB.")
    record.pop("example", None)
    if isinstance(record.get("unexecuted_program"), dict):
        record["unexecuted_program"].pop("code", None)
        record["unexecuted_program"]["code_omitted"] = "tools_closed"
    return record


def _identity(source, kind):
    return {"sha256": hashlib.sha256(source.encode("utf-8", errors="surrogatepass")).hexdigest(),
            "chars": len(source), "source": kind}


def _program_details(text, program=None, source="response"):
    """Keep exact candidate code or omit it whole; never return a runnable excerpt."""
    identity = _identity(text if program is None else program, source)
    unexecuted = {**identity, "max_code_chars": MAX_EXTRACT_CHARS}
    if program is None:
        unexecuted["code_omitted"] = "no_unambiguous_program"
    elif (len(program) <= MAX_EXTRACT_CHARS
          and len(json.dumps(program)) <= MAX_EXTRACT_CHARS):
        unexecuted["code"] = program
    else:
        unexecuted["code_omitted"] = "extract_budget"
    return {"response_sha256": _identity(text, "response")["sha256"],
            "pending_program": dict(identity), "unexecuted_program": unexecuted}


def _span(text, start, end):
    fragment = text[start:end]
    if not fragment.strip():
        return None
    start += len(fragment) - len(fragment.lstrip())
    end -= len(fragment) - len(fragment.rstrip())
    return {"start_line": text.count("\n", 0, start) + 1,
            "end_line": text.count("\n", 0, end) + 1,
            "start_column": start - text.rfind("\n", 0, start),
            "end_column": end - text.rfind("\n", 0, end),
            "chars": end - start,
            "excerpt": text[start:min(end, start + MAX_DIAGNOSTIC_CHARS)],
            "excerpt_truncated": end - start > MAX_DIAGNOSTIC_CHARS}


def _envelope_details(text):
    markers = text.count("```")
    candidate = re.search(_PYTHON_FENCE, text) if markers == 2 else None
    source, reason = "python_fence", "python_fence_required"
    # Recognize this literal wrapper only for feedback, without XML decoding or execution.
    if (candidate is None and markers == 0
            and text.count("<program>") == text.count("</program>") == 1):
        candidate = re.search(_XML_PROGRAM, text)
        source, reason = "xml_program", "xml_envelope"
    details = _program_details(text, candidate[1] if candidate else None,
                               source if candidate else "response")
    details.update(field="program.envelope", fence_markers=markers)
    if candidate:
        start, end = candidate.span() if source == "python_fence" else candidate.span(1)
        prefix, tail = _span(text, 0, start), _span(text, end, len(text))
        if prefix:
            details["invalid_prefix"] = prefix
        if tail:
            details["invalid_tail"] = tail
        if source == "python_fence":
            reason = ("invalid_prefix_and_tail" if prefix and tail else
                      "invalid_prefix" if prefix else "invalid_tail")
    else:
        reason = "multiple_fences" if markers > 2 else reason
        details["invalid_envelope"] = _span(text, 0, len(text))
    details["reason"] = reason
    return details


def pending_program(text, error):
    """Retain the whole unexecuted candidate for the next fresh request.

    Recognition is for model context only. This function never parses a wrapper
    into runnable code or authorizes its execution; parse_program remains strict.
    """
    markers = text.count("```")
    candidate = re.search(_PYTHON_FENCE, text) if markers == 2 else None
    source = "python_fence"
    if (candidate is None and markers == 0
            and text.count("<program>") == text.count("</program>") == 1):
        candidate = re.search(_XML_PROGRAM, text)
        source = "xml_program"
    code = candidate[1] if candidate else text
    source = source if candidate else "invalid_response"
    return {"version": PENDING_VERSION, "status": "unexecuted_model_program",
            **_identity(code, source),
            "response_sha256": _identity(text, "response")["sha256"],
            "rejection_code": error.code, "code": code,
            "execution_authorized": False}


def parse_program(text):
    """Require exactly the advertised code envelope; never execute XML/prose."""
    match = re.fullmatch(_PYTHON_FENCE, text.strip())
    if not match or "```" in match[1]:
        raise ProgramFormatError("python_fence_required", **_envelope_details(text))
    program = match[1]
    try:
        tree = ast.parse(program)
    except SyntaxError as exc:
        details = _program_details(text, program, "python_fence")
        start = len(text) - len(text.lstrip()) + match.start(1)
        details.update(field="program.syntax", syntax={
            "line": exc.lineno, "column": exc.offset,
            "end_line": exc.end_lineno, "end_column": exc.end_offset,
            "response_line": text.count("\n", 0, start) + (exc.lineno or 1),
            "response_column": (exc.offset + start - text.rfind("\n", 0, start) - 1
                                if exc.lineno == 1 and exc.offset is not None else exc.offset),
            "message": exc.msg[:MAX_DIAGNOSTIC_CHARS],
        })
        raise ProgramFormatError("python_syntax_error", **details) from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ProgramFormatError("solve_entry_required", field="program.entry",
                                 **_program_details(text, program, "python_fence"))
    entry = tree.body[0]
    args = entry.args
    if (entry.name != "solve" or entry.decorator_list or entry.returns
            or args.posonlyargs or args.kwonlyargs or args.kwarg or args.vararg
            or args.defaults or len(args.args) != 1
            or args.args[0].arg != "ctx" or args.args[0].annotation):
        raise ProgramFormatError("solve_entry_required", field="program.entry",
                                 **_program_details(text, program, "python_fence"))
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
