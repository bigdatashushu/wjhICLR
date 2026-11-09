"""Runner-owned interface supplement; shared evidence and replay rules stay intact."""

from __future__ import annotations

import hashlib
import json

from skill3d.schemas import CANONICAL_UNIT_BY_QUESTION_TYPE

VSI_INTERFACE_VERSION = "vsi-execution-interfaces/1.0"
VSI_INTERFACE_NAMES = frozenset({"ToolData", "ReadNextObservation", "MeasurementAnswer"})

# tool, raw-root field, source unit, submission unit; no inferred alternatives.
MEASUREMENT_FIELDS = {
    "object_abs_distance": ("object_surface_distance", "distance", "m", "m"),
    "object_size_estimation": ("object_3d_extent", "extent_longest_metric", "m", "cm"),
    "room_size_estimation": ("plane_fit_room_size", "room_area_m2", "m2", "m2"),
    "object_counting": ("count_objects", "count", "count", "count"),
}

OBSERVATION_EXAMPLE = """def solve(ctx):
    result = list_objects()
    objects = ToolData(result)
    return YieldObservations([result["result_id"]], "Inspect candidate identities and frames.")
"""
PAGING_EXAMPLE = """def solve(ctx):
    # Replace this literal with the actual host result ID shown in feedback.
    page = ReadNextObservation("ACTUAL_RESULT_ID", fields=["obj_id", "visible_frames"], limit=8)
    rows = ToolData(page)
    return YieldObservations([page["result_id"]], "Inspect the next candidate page.")
"""


def visual_example(question_type: str) -> str:
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE.get(question_type, "option")
    value = "'A'" if unit == "option" else "1" if unit == "count" else "1.25"
    return (
        "def solve(ctx):\n"
        "    # Illustrative syntax only: replace the value with your image-based answer.\n"
        f"    answer = {value}\n"
        "    return ReturnAnswer(AnswerPayload(\n"
        f"        value=answer, unit={unit!r}, basis='visual_estimate',\n"
        "        used_result_ids=[], derivation=None))\n"
    )


def finalization_example(question_type: str) -> str:
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE.get(question_type, "option")
    return json.dumps({
        "action": "answer",
        "value": "A" if unit == "option" else 1 if unit == "count" else 1.25,
        "unit": unit, "basis": "visual_estimate", "used_result_ids": [], "derivation": None,
    }, separators=(",", ":"))


_INTERFACE_RULES = """VSI host interfaces (ordinary Python phase only):
- ToolData(response) extracts data from a tool-response envelope or a ReadObservation page.
  Original tool data keeps its list/dict/scalar shape. Reader page data is ALWAYS a list:
  rows = ToolData(page); a scalar/dict tool page has its original root at rows[0].
  Do not call .get("objects", []) on a list. An absent field is unknown, not an empty list or zero.
  Original tool responses must include their actual data and match the current ledger.
  An ID-only or edited envelope is rejected; use ReadNextObservation for historical lists.
- ReadNextObservation(result_or_id, fields=None, limit=8) reads the first remaining gap
  using the same ReadObservation budget (3 reads per round, at most 32 items and 4000 chars
  per page). It never reruns the original tool. Pass a response/page or its actual host ID.
  Cumulative coverage is supplementary metadata; original reader pages keep their byte cap.
  The host keeps progress across namespace resets. Omitted fields inherit the previous
  selection for that result; changing the selected fields starts an independent coverage.
  page["vsi_read_progress"]["coverage_complete"] reports coverage accumulated across
  successful reads of those fields. page["coverage_complete"] describes this page alone;
  next_cursor=None on a tail page does not prove earlier rows were read.
  A page can be shorter than limit. Use ReadNextObservation, not offset += limit.
  YieldObservations([page["result_id"]], "reason") delivers the actual page to the next request.
  Reuse the literal result ID from feedback; variables from the previous round no longer exist.
- MeasurementAnswer(result_or_id) builds an AnswerPayload from the CURRENT real tool ledger.
  It uses only the registered field for this task: absolute distance=distance with unit m;
  object size=extent_longest_metric in m converted exactly once to cm; room area=room_area_m2;
  object count=count. Failed, invalidated, unresolved, unhealthy, missing, negative,
  nonfinite, wrong-tool or normalized-only measurements are rejected.
  Submit with return ReturnAnswer(MeasurementAnswer(result)).
  The helper keeps full numeric precision, actual used_result_ids and a field/convert
  derivation. Geometry and deterministic replay still decide acceptance.
  Do not round a tool-derived value or stringify it with .2f: replay checks the exact
  underlying measurement (tolerance 1e-9). Round only diagnostic print output.
- These names are reserved host interfaces, not tools. They do not grant new tool
  authorization, geometry, model calls, or extra reader budget. No helper is available
  in finalization. Never fabricate IDs or submit the illustrative values below.
"""


def program_instructions(question_type: str) -> str:
    text = (
        "\n\nVSI executable output supplement [" + VSI_INTERFACE_VERSION + "]\n"
        "Output one fenced Python block. Define every variable before use; Python uses "
        "None/True/False. A valid branch must end in ReturnAnswer or YieldObservations. "
        "Never output a bare answer, pseudocode, or prose as the program.\n"
        + _INTERFACE_RULES
        + "\nComplete visual-submission syntax (illustrative value):\n```python\n"
        + visual_example(question_type) + "```\n"
        + "\nCandidate observation syntax (only when list_objects is authorized):\n```python\n"
        + OBSERVATION_EXAMPLE + "```\n"
        + "\nNext-round paging syntax (replace ACTUAL_RESULT_ID):\n```python\n"
        + PAGING_EXAMPLE + "```\n"
    )
    if question_type in MEASUREMENT_FIELDS:
        tool, field, from_unit, to_unit = MEASUREMENT_FIELDS[question_type]
        text += (
            f"\nFor this task, MeasurementAnswer requires {tool}.{field} "
            f"({from_unit} -> {to_unit}). First select the intended referents and call "
            "the currently authorized tool with its documented arguments, then use "
            "`return ReturnAnswer(MeasurementAnswer(result))`. No early rounding.\n"
        )
    return text


def finalization_instructions(question_type: str) -> str:
    return (
        "\n\nVSI final JSON supplement [" + VSI_INTERFACE_VERSION + "]\n"
        "Output exactly one JSON object: no markdown fences, Python, assignments, "
        "ReturnAnswer, YieldObservations, helper calls, or text outside the object. "
        "JSON uses null/true/false. Include action, value, unit, basis, used_result_ids "
        "and derivation. Numeric answers must be JSON numbers, not strings or arrays. "
        "Retain full measurement precision and any real derivation already established. "
        "Use only valid result IDs shown in this request. Choose basis from the evidence "
        "actually used; changing basis does not repair invalid geometry.\n"
        "Complete syntax example (illustrative value and visual basis; replace both "
        "according to your actual evidence):\n" + finalization_example(question_type) + "\n"
    )


def round_reminder(*, finalizing: bool) -> str:
    if finalizing:
        return "VSI format reminder: one JSON object; null, no Python or code fences."
    return (
        "VSI format reminder: executable Python; define variables and use None. "
        "End with ReturnAnswer(AnswerPayload(...)), ReturnAnswer(MeasurementAnswer(result)), "
        "or YieldObservations([actual_result_id], 'reason'). Reuse host IDs for paging."
    )


VSI_INTERFACE_SHA256 = hashlib.sha256(json.dumps({
    "version": VSI_INTERFACE_VERSION, "names": sorted(VSI_INTERFACE_NAMES),
    "measurement_fields": MEASUREMENT_FIELDS, "rules": _INTERFACE_RULES,
    "program": {task: program_instructions(task) for task in CANONICAL_UNIT_BY_QUESTION_TYPE},
    "finalization": {task: finalization_instructions(task) for task in CANONICAL_UNIT_BY_QUESTION_TYPE},
    "round_reminders": [round_reminder(finalizing=False), round_reminder(finalizing=True)],
}, sort_keys=True).encode()).hexdigest()
