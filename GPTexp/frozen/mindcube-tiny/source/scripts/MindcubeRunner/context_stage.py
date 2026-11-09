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
The previous source or response in RECENT FEEDBACK is diagnostic data to repair.
Correct the offending line; reuse supplied evidence instead of repeating completed
tool calls or reconsidering an answer solely because its code had a syntax error.
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
quality. Read candidates and count from the compact group's shared_data and rows.
Match each result_id to EXECUTED CANDIDATE QUERIES for its actual view_index and
category. Those are completed calls; code examples and RECENT FEEDBACK are not.
If view_index=-1 and category="" returned no candidates, the exposed catalog has
no hints for this question. Continue from RGB: repeating or filtering that catalog
cannot resolve the binding. An empty filtered or single-view result leaves only
its unqueried scope open. Reuse delivered depth summaries too.""",
    "plan_or_measure": """PLAN OR MEASURE
Identify the deciding reference frame, target bindings and comparison, and check
them against specific visible features. If your interpretation still switches
between target identities or direction mappings that change the answer, it is
unresolved. Choose the next executable action that addresses that evidence gap.
For explicit turns at one camera position, compose the turn and requested
relative direction first using the static rotation rule, then identify the
object in that direction. Do not reconstruct a whole room for a resolved turn.
For an uncertain target or common landmark, query location hints now and yield
their result_ids for next-request inspection. This can help cross-view binding
even when local depth cannot estimate pose. Inspect exposed candidates without
a category filter when exact spelling is unknown; inspect new hints before
selecting pixels from their boxes.
When usable same-view depth or distance resolves the comparison, bind visible
target surfaces and batch the needed independent measurements, then yield them
for next-request review. A direct visual answer is allowed when concrete target
and direction evidence is consistent, or when no available tool addresses the
remaining issue. Do not replace a useful query with repeated imagined layouts.
Once the next useful call is identified, submit it and inspect its actual result
in the next request; do not keep predicting what it might return.
Rotation and cross-view motion use visual correspondences and the question's
reference frame; apparent size changes and independently colored depth previews
cannot alone establish forward movement.
A depth summary is optional for a needed diagnostic beyond existing previews/results;
it cannot locate objects or refresh depth.""",
}

_PROGRAM_FORMAT = """OUTPUT FORMAT
Output exactly one closed python fence containing only def solve(ctx): and its
body; no prose before or after the fence, XML or top-level execution.
Keep explanations and planning out of the submitted response.
End with a direct Python function call
to ReturnAnswer(letter) or YieldObservations(result_ids, reason=...).
Call these controls by their bare names, without a prefix or a global declaration;
they are not ctx.tools methods.

EXECUTION AND CONTINUATION
The host executes your solve(ctx) after you submit the program. Each successful
evidence call returns a dict r inside that program: r["result_id"] identifies the
result and r["data"] contains the tool-specific values. YieldObservations ends
only the current program. The host then sends another model request containing
the original images and TOOL RESULTS; you inspect those results before producing
the next program or answer. This is an executable multi-request interaction:
you will receive the results you yield, not have to guess them in this response.

Tool and measurement call counts are unlimited in program phases.
New candidate hints must be yielded for next-request inspection before using
their boxes to select measurement pixels. Use each returned r["result_id"].
New measurements, including unreliable notices, MUST yield all new result_ids
with YieldObservations for next-request review. Do not answer after measuring in
the same program.
Variables reset each program; evidence remains in the prompt. Reuse needed
prompt values as literals rather than measuring again. The next prompt carries
the returned evidence even though the variable r itself is gone. There is no ReadObservation
or ctx observation-store interface. Only the four supplied evidence tools are
available; safe math/numpy is allowed, host I/O and introspection are forbidden."""

_PROGRAM_EXAMPLES = """These are alternative syntax examples, not past actions.
Only when supplied results do not already cover the needed target locations:
```python
def solve(ctx):
    r = ctx.tools.view_candidates(view_index=-1)
    YieldObservations([r["result_id"]], reason="Inspect target locations in the next request")
```
When concrete target bindings and direction evidence support a direct answer:
```python
def solve(ctx):
    ReturnAnswer("A")
```
A is a syntax example; choose the current legal option letter supported by evidence."""

_FINAL_REVIEW = """REVIEW AVAILABLE EVIDENCE
Use only supplied evidence. Review available measurements and hints against the
original RGB/depth and question: match result_id, source_tool, view_index and
pixel_xy or a_xy/b_xy to the intended surfaces. Decode compact tables using the
global rules. Check data.status, reason_code, reason and sample_stability;
outer status=ok does not establish reliability or correct target binding.
Unreliable or omitted depth/distance supplies no numerical comparison. Decide
from reliable reviewed values and valid visual evidence. Check the observer/object
direction convention and the requested turn once under a consistent mapping;
do not infer cross-view motion from normalized depth colors or size change alone."""

_FINAL_FORMAT = """OUTPUT FORMAT
Tools are closed. Output only {"answer":"A"} with the chosen current legal option
letter. A is an example: replace A with your chosen letter. No prose or extra keys."""


def guidance(actions, *, final) -> str:
    if final:
        return _FINAL_REVIEW + "\n\n" + _FINAL_FORMAT
    selected = set(actions)
    instructions = [text for action, text in _PROGRAM_ACTIONS.items()
                    if action in selected]
    instructions.append(_PROGRAM_FORMAT)
    if not selected & {"inspect_hints", "review_measurements"}:
        instructions.append(_PROGRAM_EXAMPLES)
    return "\n\n".join(instructions)


def response_reminder(*, final) -> str:
    """Repeat the current response contract after all image inputs."""
    if final:
        return ('RESPONSE CONTRACT: Tools are closed. Submit only the JSON object '
                'with the single key "answer" and your chosen legal option letter. '
                'No explanation, Markdown fence or extra keys.')
    return ("RESPONSE CONTRACT: Submit exactly one closed python fence containing "
            "only def solve(ctx): and its body. No explanation before or after it. "
            "For a useful unresolved evidence gap, submit the evidence call and "
            "YieldObservations; use ReturnAnswer only for your supported choice. "
            "The host executes the program and delivers yielded results in the next request.")


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
        for field in ("line", "column"):
            value = error.get(field)
            if field in error and (value is None or type(value) is int):
                clean[field] = value
        if final:
            clean["repair"] = "Follow the final output format above."
        else:
            for field in ("source", "source_utf8_sha256", "source_line",
                          "syntax_message", "previous_response"):
                if isinstance(error.get(field), str):
                    clean[field] = error[field]
            clean["repair"] = (error["repair"] if isinstance(error.get("repair"), str) else
                              "Correct the reported error using the program output format above.")
        feedback.append(clean)
    return feedback
