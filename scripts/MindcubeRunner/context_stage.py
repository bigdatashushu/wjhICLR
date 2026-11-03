"""Independent request actions and phase-local guidance, without runtime policy."""
from __future__ import annotations

from collections.abc import Mapping


def stage_actions(*, final, errors=(), review_ids=(), hint_ids=()) -> list:
    if final:
        return ["answer_json"]
    actions = []
    if errors:
        actions.append("repair")
    if review_ids:
        actions.append("review_measurements")
    if hint_ids:
        actions.append("inspect_hints")
    return actions or ["plan_or_measure"]


_PROGRAM_ACTIONS = {
    "repair": """REPAIR
Correct the reported code/type/line using the feedback and program format below.
Repair does not cancel pending measurement review or hint inspection; preserve
the supplied evidence while correcting the program.""",
    "review_measurements": """REVIEW MEASUREMENTS
Review every supplied measurement against its original RGB/depth: match result_id,
source_tool, view_index and pixel_xy or a_xy/b_xy to the intended target surfaces.
Check data.status, reason_code, reason and sample_stability; outer status=ok is
execution success, not measurement reliability. Stable sampling can bind the
wrong target. Unreliable notices supply no usable depth/distance. If needed,
selectively resample only unreliable or visually misbound pixels; for pairs check
unreliable_endpoints and retain verified endpoints. New samples require yield
and next-request review; do not repeat rejected pixels or yield old notices.""",
    "inspect_hints": """INSPECT HINTS
Match supplied candidate categories, boxes and view_index to the original RGB
before selecting visible interior pixels; a box center need not hit the target.
IDs and boxes are incomplete location hints, not identities, counts or mask
quality. Reuse delivered candidate queries and depth summaries.""",
    "plan_or_measure": """PLAN OR MEASURE
Identify the deciding comparison in the question's reference frame. When usable
same-view depth or distance resolves it, bind the visible target surfaces and
batch the needed independent comparisons. Clearly visual relations or missing
reliable depth may instead support a visual answer. Query location hints only
for unclear targets, then yield them for next-request inspection before selecting
pixels from new boxes. A depth summary is optional for a needed diagnostic beyond
existing previews/results; it cannot locate objects or refresh depth.""",
}

_PROGRAM_FORMAT = """OUTPUT FORMAT
Output exactly one closed python fence containing only def solve(ctx): and its
body; no prose, XML or top-level execution. End with GLOBAL ReturnAnswer(letter)
or YieldObservations(result_ids, reason=...); these controls are not ctx.tools
methods. Answers use the chosen current legal option letter.
Tool and measurement call counts are unlimited in program phases.
New measurements, including unreliable notices, MUST yield all new result_ids
with YieldObservations for next-request review. Do not answer after measuring in
the same program.
Variables reset each program; evidence remains in the prompt. Reuse needed
prompt values as literals rather than measuring again. There is no ReadObservation
or ctx observation-store interface. Only the four supplied evidence tools are
available; safe math/numpy is allowed, host I/O and introspection are forbidden."""

_FINAL_REVIEW = """REVIEW AVAILABLE EVIDENCE
Use only supplied evidence. Review available measurements and hints against the
original RGB/depth and question: match result_id, source_tool, view_index and
pixel_xy or a_xy/b_xy to the intended surfaces. Decode compact tables using the
global rules. Check data.status, reason_code, reason and sample_stability;
outer status=ok does not establish reliability or correct target binding.
Unreliable or omitted depth/distance supplies no numerical comparison. Decide
from reliable reviewed values and valid visual evidence."""

_FINAL_FORMAT = """OUTPUT FORMAT
Tools are closed. Output only {"answer":"A"} with the chosen current legal option
letter. A is an example: replace A with your chosen letter. No prose or extra keys."""


def guidance(actions, *, final) -> str:
    if final:
        return _FINAL_REVIEW + "\n\n" + _FINAL_FORMAT
    selected = set(actions)
    instructions = [text for action, text in _PROGRAM_ACTIONS.items()
                    if action in selected]
    return "\n\n".join([*instructions, _PROGRAM_FORMAT])


def format_feedback(errors, *, final) -> list:
    """Copy only diagnostic fields; final repairs cannot carry program advice."""
    feedback = []
    for error in errors:
        if not isinstance(error, Mapping):
            continue
        clean = {}
        for field in ("code", "type"):
            value = error.get(field)
            if field in error and (value is None or isinstance(value, str)):
                clean[field] = value
        line = error.get("line")
        if "line" in error and (line is None or type(line) is int):
            clean["line"] = line
        if final:
            clean["repair"] = "Follow the final output format above."
        elif isinstance(error.get("repair"), str):
            clean["repair"] = error["repair"]
        else:
            clean["repair"] = "Correct the reported error using the program output format above."
        feedback.append(clean)
    return feedback
