"""Strict program envelopes and bounded repair feedback, using synthetic text only."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from Ominirunner.native import (
    MAX_DIAGNOSTIC_CHARS, MAX_EXTRACT_CHARS, MAX_FEEDBACK_CHARS, ProgramFormatError,
    control_namespace_errors, parse_program, program_feedback,
)
from Ominirunner.common import Config
from Ominirunner.protocol import messages
from skill3d.sandbox.action_check import check_action
from skill3d.sandbox.ast_guard import ast_guard


PROGRAM = (
    "def solve(ctx):\n"
    "    measurement_reviews = []\n"
    '    YieldObservations([], reason="Review current observations")'
)
FENCED = f"```python\n{PROGRAM}\n```"


def sha(text):
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def rejected(text, code="python_fence_required"):
    with pytest.raises(ProgramFormatError) as caught:
        parse_program(text)
    error = caught.value
    assert error.code == str(error) == code
    assert error.record["code"] == code
    assert error.record["currentphase"] == "program"
    assert error.record["executed"] is False
    assert error.record["response_sha256"] == sha(text)
    return error


def assert_candidate(error, program=PROGRAM, source="python_fence"):
    identity = {"sha256": sha(program), "chars": len(program), "source": source}
    assert error.record["pending_program"] == identity
    snapshot = error.record["unexecuted_program"]
    assert all(snapshot[key] == value for key, value in identity.items())
    assert snapshot["code"] == program
    assert "code_omitted" not in snapshot
    assert parse_program(f"```python\n{snapshot['code']}\n```") == program


@pytest.mark.parametrize("prefix", [
    "Here is the current program:\n",
    "ReturnAnswer(999)\n",
    "\n\n  Short explanation.\n\n",
])
def test_prefix_is_located_and_never_accepted(prefix):
    error = rejected(prefix + FENCED)
    assert error.record["reason"] == "invalid_prefix"
    span = error.record["invalid_prefix"]
    assert span["chars"] == len(prefix.strip())
    assert span["start_line"] == prefix.count("\n", 0, len(prefix) - len(prefix.lstrip())) + 1
    assert span["excerpt"] == prefix.strip()
    assert "invalid_tail" not in error.record
    assert_candidate(error)


@pytest.mark.parametrize("tail", [
    "\nThat is the program.",
    "\nReturnAnswer(999)",
    "\n\n  Done.\n\n",
])
def test_tail_is_located_and_never_accepted(tail):
    error = rejected(FENCED + tail)
    assert error.record["reason"] == "invalid_tail"
    span = error.record["invalid_tail"]
    first = len(FENCED) + len(tail) - len(tail.lstrip())
    assert span["start_line"] == (FENCED + tail).count("\n", 0, first) + 1
    assert span["chars"] == len(tail.strip())
    assert span["excerpt"] == tail.strip()
    assert "invalid_prefix" not in error.record
    assert_candidate(error)


def test_prefix_and_tail_have_original_line_and_column_locations():
    text = "\n\n  Explain.\n" + FENCED + "\n\n  Done.\n"
    error = rejected(text)
    assert error.record["reason"] == "invalid_prefix_and_tail"
    assert error.record["invalid_prefix"] == {
        "start_line": 3, "end_line": 3, "start_column": 3, "end_column": 11,
        "chars": 8, "excerpt": "Explain.", "excerpt_truncated": False,
    }
    assert error.record["invalid_tail"] == {
        "start_line": 10, "end_line": 10, "start_column": 3, "end_column": 8,
        "chars": 5, "excerpt": "Done.", "excerpt_truncated": False,
    }
    assert_candidate(error)


@pytest.mark.parametrize("text", [
    f"<program>\n{FENCED}\n</program>",
    f"<program><code>\n{FENCED}\n</code></program>",
    f"<thinking>Check the inputs.</thinking>\n{FENCED}",
    f"<think>Check the inputs.</think>\n{FENCED}",
    f"{FENCED}\n<thinking>Next step.</thinking>",
])
def test_xml_and_thinking_wrapped_fences_are_feedback_only(text):
    error = rejected(text)
    assert "invalid_prefix" in error.record or "invalid_tail" in error.record
    assert_candidate(error)


def test_bare_xml_program_can_be_identified_but_never_executed():
    text = f"<program>\n{PROGRAM}\n</program>"
    error = rejected(text)
    assert error.record["reason"] == "xml_envelope"
    assert error.record["invalid_prefix"]["excerpt"] == "<program>"
    assert error.record["invalid_tail"]["excerpt"] == "</program>"
    assert_candidate(error, source="xml_program")


@pytest.mark.parametrize("text", [
    f"<thinking>{PROGRAM}</thinking>",
    f"<program>{PROGRAM}</program><program>{PROGRAM}</program>",
    PROGRAM,
    "",
])
def test_no_unambiguous_candidate_keeps_only_response_identity(text):
    error = rejected(text)
    assert error.record["pending_program"] == {
        "sha256": sha(text), "chars": len(text), "source": "response",
    }
    snapshot = error.record["unexecuted_program"]
    assert snapshot["code_omitted"] == "no_unambiguous_program"
    assert "code" not in snapshot


@pytest.mark.parametrize("text", [
    FENCED + "\n" + FENCED,
    FENCED + '\n```json\n{"answer":1}\n```',
    f"<program>{FENCED}\n{FENCED}</program>",
    '```python\ndef solve(ctx):\n    ReturnAnswer("```")\n```',
    "```text\nExplanation\n```\n" + FENCED,
])
def test_multiple_or_embedded_fences_never_select_a_program(text):
    error = rejected(text)
    assert error.record["reason"] == "multiple_fences"
    assert error.record["fence_markers"] == text.count("```")
    assert error.record["unexecuted_program"]["source"] == "response"
    assert "code" not in error.record["unexecuted_program"]


@pytest.mark.parametrize("text", [
    FENCED[:-3],
    FENCED.replace("```python\n", "```Python\n", 1),
    FENCED.replace("```python\n", "```\n", 1),
    FENCED.replace("```python\n", "```python ", 1),
    FENCED.replace("```python\n", "```python\r\n", 1),
])
def test_existing_exact_fence_guards_remain(text):
    rejected(text)


@pytest.mark.parametrize("text", [
    FENCED,
    "\n \t" + FENCED + "\n \t",
    FENCED.replace("```python\n", "```python \t\n", 1),
    f"```python\n{PROGRAM}```",
])
def test_legal_single_fence_returns_original_source(text):
    assert parse_program(text) == PROGRAM
    assert ast_guard(PROGRAM, allowed_tools=set()).ok
    assert check_action(PROGRAM, tool_names=set()).ok


@pytest.mark.parametrize("program", [
    "def solve(ctx)\n    pass",
    "def solve(ctx):\n    measurement_reviews = []\n    answer = (",
    "def solve(ctx):\nmeasurement_reviews = []",
    "def solve(ctx):\n    # " + "x" * 300 + "\n    ReturnAnswer(1 + )",
])
def test_syntax_errors_report_program_and_response_positions(program):
    text = "\n\n  ```python\n" + program + "\n```"
    try:
        ast.parse(program)
    except SyntaxError as expected:
        error = rejected(text, "python_syntax_error")
        syntax = error.record["syntax"]
        assert syntax["line"] == expected.lineno
        assert syntax["column"] == expected.offset
        assert syntax["end_line"] == expected.end_lineno
        assert syntax["end_column"] == expected.end_offset
        assert syntax["response_line"] == expected.lineno + 3
        assert syntax["response_column"] == expected.offset
        assert syntax["message"] == expected.msg[:MAX_DIAGNOSTIC_CHARS]
        assert error.record["field"] == "program.syntax"
        snapshot = error.record["unexecuted_program"]
        assert snapshot["sha256"] == sha(program)
        if len(program) <= MAX_EXTRACT_CHARS:
            assert snapshot["code"] == program
        else:
            assert "code" not in snapshot
        assert isinstance(error.__cause__, SyntaxError)
    else:
        pytest.fail("synthetic syntax fixture must be invalid")


@pytest.mark.parametrize("program", [
    "",
    "ReturnAnswer(1)",
    "def other(ctx):\n    pass",
    "async def solve(ctx):\n    pass",
    "@decorator\ndef solve(ctx):\n    pass",
    "def solve(ctx) -> int:\n    pass",
    "def solve(ctx: int):\n    pass",
    "def solve(ctx=None):\n    pass",
    "def solve(ctx, extra):\n    pass",
    "def solve():\n    pass",
    "def solve(ctx, /):\n    pass",
    "def solve(*ctx):\n    pass",
    "def solve(**ctx):\n    pass",
    "def solve(*, ctx):\n    pass",
    "def solve(ctx, *, extra):\n    pass",
    "def solve(ctx):\n    pass\nother = 1",
    "def solve(ctx):\n    pass\ndef other(ctx):\n    pass",
])
def test_existing_solve_entry_guards_remain(program):
    error = rejected(f"```python\n{program}\n```", "solve_entry_required")
    assert error.record["field"] == "program.entry"


def test_valid_envelope_does_not_replace_public_security_guard(tmp_path):
    target = tmp_path / "must-not-exist"
    program = f"def solve(ctx):\n    open({str(target)!r}, 'w').write('unsafe')"
    parsed = parse_program(f"```python\n{program}\n```")
    assert parsed == program
    assert not ast_guard(parsed, allowed_tools=set()).ok
    rejected("Explanation.\n```python\n" + program + "\n```")
    assert not target.exists()


def test_prefix_and_tail_diagnostics_are_bounded_without_losing_identity():
    prefix = "\n" * 50 + "p" * 200_000 + "\n"
    tail = "\n" + "t" * 200_000
    error = rejected(prefix + FENCED + tail)
    for field, char in [("invalid_prefix", "p"), ("invalid_tail", "t")]:
        span = error.record[field]
        assert span["chars"] == 200_000
        assert span["excerpt"] == char * MAX_DIAGNOSTIC_CHARS
        assert span["excerpt_truncated"] is True
    assert_candidate(error)
    assert len(json.dumps(program_feedback(error))) <= MAX_FEEDBACK_CHARS


@pytest.mark.parametrize("fill", ["x" * MAX_EXTRACT_CHARS, "\\x01" * 70, "\u00e9" * 100])
def test_large_or_expensive_code_is_omitted_whole_not_truncated(fill):
    program = "def solve(ctx):\n    # " + fill + "\n    ReturnAnswer(1)"
    error = rejected("Explanation.\n```python\n" + program + "\n```")
    snapshot = error.record["unexecuted_program"]
    assert snapshot["sha256"] == sha(program)
    assert snapshot["chars"] == len(program)
    assert snapshot["code_omitted"] == "extract_budget"
    assert snapshot["max_code_chars"] == MAX_EXTRACT_CHARS
    assert "code" not in snapshot
    assert len(json.dumps(program_feedback(error))) <= MAX_FEEDBACK_CHARS


def test_feedback_budget_omits_whole_code_before_dropping_locations():
    program = "def solve(ctx):\n    # " + "x" * 240 + "\n    YieldObservations([])"
    assert len(json.dumps(program)) <= MAX_EXTRACT_CHARS
    error = rejected("Explanation.\n```python\n" + program + "\n```\nDone.")
    assert error.record["unexecuted_program"]["code"] == program
    feedback = program_feedback(error)
    assert len(json.dumps(feedback)) <= MAX_FEEDBACK_CHARS
    row = feedback[0]
    assert row["invalid_prefix"]["start_line"] == 1
    assert row["invalid_tail"]["start_line"] == 7
    assert row["unexecuted_program"]["sha256"] == sha(program)
    assert row["unexecuted_program"]["code_omitted"] == "feedback_budget"
    assert "code" not in row["unexecuted_program"]
    assert error.record["unexecuted_program"]["code"] == program


def test_feedback_accepts_error_and_phase_without_mutating_record():
    error = rejected("Explanation.\n" + FENCED)
    before = deepcopy(error.record)
    feedback = program_feedback(error, phase="program")
    assert feedback[0]["currentphase"] == "program"
    assert feedback[0]["executed"] is False
    assert feedback[0]["invalid_prefix"] == {
        key: before["invalid_prefix"][key] for key in ("start_line", "end_line", "chars")
    }
    assert feedback[0]["unexecuted_program"]["sha256"] == sha(PROGRAM)
    assert "exact code" in feedback[0]["repair"]
    assert "old plan" in feedback[0]["repair"]
    assert "Not executed" in feedback[0]["repair"]
    feedback[0]["unexecuted_program"]["sha256"] = "changed by caller"
    assert error.record == before


@pytest.mark.parametrize("code", [
    "python_fence_required", "python_syntax_error", "solve_entry_required",
    "control_namespace_error", "program_missing_action",
])
def test_legacy_feedback_and_complete_minimal_example_pass_real_guards(code):
    error = ProgramFormatError(code)
    assert str(error) == error.code == code
    for feedback in [program_feedback(code=code), program_feedback(error)]:
        row = feedback[0]
        assert row["code"] == code
        assert row["currentphase"] == "program" and row["executed"] is False
        assert "FIRST:" in row["repair"]
        assert "measurement_reviews=[]" in row["repair"]
        assert "global controls" in row["repair"]
        program = parse_program(row["example"])
        assert ast_guard(program, allowed_tools=set()).ok
        assert check_action(program, tool_names=set()).ok
        assert control_namespace_errors(program) == []
        first = ast.parse(program).body[0].body[0]
        assert isinstance(first, ast.Assign)
        assert first.targets[0].id == "measurement_reviews"
        assert ast.literal_eval(first.value) == []
    assert program_feedback(code, phase="repair")[0]["currentphase"] == "repair"


@pytest.fixture
def context_inputs(monkeypatch):
    import socket

    def denied(*args, **kwargs):
        raise AssertionError("network forbidden in envelope tests")
    monkeypatch.setattr(socket.socket, "connect", denied)
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    depth = np.ones(frame.shape[:2], dtype=float)
    output = BytesIO()
    Image.fromarray(frame).save(output, format="PNG")
    cache = SimpleNamespace(geometries={0: {"depth": depth, "valid_mask": depth > 0}})
    item = {"question": "How many chairs are visible?", "task_type": "int",
            "sample_id": "synthetic-envelope", "options": []}
    return item, output.getvalue(), frame, cache


@pytest.mark.parametrize("text,code,locations", [
    ("Explanation.\n" + FENCED, "python_fence_required", ("invalid_prefix",)),
    (FENCED + "\nExplanation.", "python_fence_required", ("invalid_tail",)),
    ("p" * 100_000 + "\n" + FENCED + "\n" + "t" * 100_000,
     "python_fence_required", ("invalid_prefix", "invalid_tail")),
    (f"<program>{FENCED}</program>", "python_fence_required",
     ("invalid_prefix", "invalid_tail")),
    (f"<thinking>Check.</thinking>\n{FENCED}", "python_fence_required", ("invalid_prefix",)),
    (FENCED + "\n" + FENCED, "python_fence_required", ("invalid_envelope",)),
    ("Explanation.\n```python\ndef solve(ctx):\n    # " + "x" * 10_000 +
     "\n    ReturnAnswer(1)\n```", "python_fence_required", ("invalid_prefix",)),
    ("```python\ndef solve(ctx)\n    pass\n```", "python_syntax_error", ("syntax",)),
])
def test_feedback_survives_real_messages_shared_error_cap(context_inputs, text, code, locations):
    error = rejected(text, code)
    feedback = program_feedback(error)
    assert len(json.dumps(feedback)) <= MAX_FEEDBACK_CHARS
    item, rgb, frame, cache = context_inputs
    # Include oversized competing diagnostics to exercise the real shared cap.
    errors = feedback + [{"code": "other_error", "detail": "x" * 2000}]
    context, receipt = messages(item, rgb, frame, cache, arm="skill", config=Config(),
                                round_index=1, errors=errors)
    content = "\n".join(part["text"] for part in context[0]["content"] if part["type"] == "text")
    delivered, _ = json.JSONDecoder().raw_decode(content.split("\nERRORS ", 1)[1])
    assert delivered == feedback
    assert receipt["phase"] == delivered[0]["currentphase"] == "program"
    assert delivered[0]["executed"] is False
    assert delivered[0]["unexecuted_program"]["sha256"] == error.record["pending_program"]["sha256"]
    for location in locations:
        assert delivered[0][location] == feedback[0][location]
    assert "Resend one closed python fence" in delivered[0]["repair"]
    assert "old plan" in delivered[0]["repair"]
    assert parse_program(delivered[0]["example"])
    assert "other_error" not in content
