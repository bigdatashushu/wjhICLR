"""Runner-owned interface supplement; shared evidence and replay rules stay intact."""

from __future__ import annotations

import hashlib
import json

from skill3d.schemas import CANONICAL_UNIT_BY_QUESTION_TYPE

VSI_INTERFACE_VERSION = "vsi-execution-interfaces/2.0-compact"
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


_INTERFACE_RULES = """Host helpers are reserved ordinary-phase interfaces, not tools or new authorization.
ToolData(response) and ReadNextObservation use the envelope/paging rules above and the
same reader budget. No helper is available in finalization. Examples are syntax only:
replace their values and IDs with evidence from this request.
"""


def program_instructions(question_type: str, *, authorized_tools=None) -> str:
    text = (
        "\n\nVSI host examples [" + VSI_INTERFACE_VERSION + "]\n"
        + _INTERFACE_RULES
        + "\nComplete visual-submission syntax (illustrative value):\n```python\n"
        + visual_example(question_type) + "```\n"
        + "\nNext-round paging syntax (replace ACTUAL_RESULT_ID):\n```python\n"
        + PAGING_EXAMPLE + "```\n"
    )
    if authorized_tools is None or "list_objects" in authorized_tools:
        text += (
            "\nCandidate observation syntax (only when list_objects is authorized):\n```python\n"
            + OBSERVATION_EXAMPLE + "```\n")
    if question_type in MEASUREMENT_FIELDS and (
            authorized_tools is None or MEASUREMENT_FIELDS[question_type][0] in authorized_tools):
        tool, field, from_unit, to_unit = MEASUREMENT_FIELDS[question_type]
        text += (
            f"\nMeasurementAnswer(result_or_id) uses only the current ledger's {tool}.{field} "
            f"({from_unit} -> {to_unit}), keeping full precision, actual used_result_ids and "
            "a field/convert derivation (replay tolerance 1e-9). Submit with "
            "`return ReturnAnswer(MeasurementAnswer(result))`. Select intended referents and "
            "use authorized tool arguments first. Failed/invalidated/unresolved/unhealthy/missing/"
            "negative/nonfinite/wrong-tool/normalized-only measurements are rejected. "
            "Geometry and replay still decide acceptance.\n"
        )
    return text


def finalization_instructions(question_type: str) -> str:
    return (
        "\n\nVSI final JSON supplement [" + VSI_INTERFACE_VERSION + "]\n"
        "JSON uses null/true/false; no markdown fences, Python or helper calls.\n"
        "Complete syntax example (illustrative value and visual basis; replace both "
        "according to your actual evidence):\n" + finalization_example(question_type) + "\n"
    )


def round_reminder(*, finalizing: bool, question_type: str = "") -> str:
    if finalizing:
        return "VSI format reminder: one JSON object; null, no Python or code fences."
    if question_type in MEASUREMENT_FIELDS:
        tool, field, from_unit, to_unit = MEASUREMENT_FIELDS[question_type]
        return (f"Measurement helper: return ReturnAnswer(MeasurementAnswer(real_result_or_id)); "
                f"requires {tool}.{field} ({from_unit}->{to_unit}), no early rounding.")
    return ""


VSI_INTERFACE_SHA256 = hashlib.sha256(json.dumps({
    "version": VSI_INTERFACE_VERSION, "names": sorted(VSI_INTERFACE_NAMES),
    "measurement_fields": MEASUREMENT_FIELDS, "rules": _INTERFACE_RULES,
    "program": {task: program_instructions(task) for task in CANONICAL_UNIT_BY_QUESTION_TYPE},
    "finalization": {task: finalization_instructions(task) for task in CANONICAL_UNIT_BY_QUESTION_TYPE},
    "round_reminders": {task: round_reminder(finalizing=False, question_type=task)
                        for task in CANONICAL_UNIT_BY_QUESTION_TYPE},
}, sort_keys=True).encode()).hexdigest()
