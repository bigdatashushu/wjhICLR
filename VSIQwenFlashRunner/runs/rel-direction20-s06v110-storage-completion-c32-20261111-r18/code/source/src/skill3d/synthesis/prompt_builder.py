"""Runner-owned compact prompts; complete Skills and shared enforcement retained."""

from __future__ import annotations

import json
from typing import Optional, Sequence

from jinja2 import Environment, StrictUndefined

from skill3d.schemas import SkillSpecV11
from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE, direction_option_mapping
from skill3d.skills.delivery import SkillDeliveryPlan, plan_delivery
from skill3d.synthesis.execution_contract import ExecutionContract
from .prompt_compaction import COMPACTION_VERSION, task_derivation_contract
from .counting_context import COUNTING_CONTEXT_VERSION, COUNTING_PROGRAM, COUNTING_RULES

PROMPT_TEMPLATE_VERSION = "program_synth_vsi_compact_1_en"
FINALIZATION_PROMPT_VERSION = "finalization_vsi_compact_1_en"

_TASK_DEFINITIONS = {
    "object_counting": "Count real instances of the specified category within the stated scope; return a non-negative integer.",
    "object_abs_distance": "Estimate the nearest distance between the two named objects, in the requested unit.",
    "object_rel_distance": "Select the candidate category nearest to the reference object; the reference is not the camera.",
    "object_size_estimation": "Estimate the largest physical dimension among the complete target object's length, width, and height, in the requested unit.",
    "room_size_estimation": "Estimate the floor area of the specified space in square meters.",
    "object_rel_direction": "Determine the target's direction when standing at the specified location and facing the specified object; use the vocabulary of the options.",
    "obj_appearance_order": "Determine the order in which the specified categories first appear in the video.",
    "route_planning": "Determine the turns along the specified route; return the complete action sequence represented by one of the current options.",
}

_TASK_RULES = {
    "object_counting": COUNTING_RULES,
    "object_abs_distance": (
        "Use the named object roles and documented distance definition. Without valid scale, "
        "normalized geometry is not a metric measurement. Convert m/cm exactly once."),
    "object_size_estimation": (
        "Use the complete intended object's longest dimension, not a part, pixel length or volume. "
        "Without valid scale, normalized geometry is not metric. Multiply m by 100 once for cm."),
    "room_size_estimation": (
        "Preserve the specified room/floor scope. Connectivity/span is not floor area or scale. "
        "Area conversions use the square of the length ratio; normalized geometry is not m2."),
    "object_rel_distance": (
        "Tool ranking must use relative_distance_rank covering ALL option categories. Missing "
        "candidates, ambiguous/reused instances, unmeasurable distances or tied minima are unresolved, "
        "never infinity or an omitted option. Partial argmin cannot establish the answer; obtain "
        "further observations or use delivered images in a later zero-tool visual segment. "
        "M11 checks coverage, unique minimum and mapping to the current option letter. No metric scale required."),
    "object_rel_direction": (
        "Preserve observer, facing object and target roles, horizontal reference frame, difficulty "
        "and current direction vocabulary. Map the direction to its current option letter; "
        "a raw direction word is not an option letter."),
    "obj_appearance_order": (
        "Order true first appearances in the video. Frozen frame slots are not physical frame "
        "indices; mask visibility/first registered detection does not certify first appearance. "
        "Check all specified categories and map the complete order to a current option."),
    "route_planning": (
        "Preserve route start, destination, landmarks, facing direction and complete turn sequence. "
        "A connectivity graph is an approximation and all object nodes require usable bindings; "
        "optional metric coordinates depend on current scale. Map the full route to a current option."),
}

_PROGRAM_SYNTH = """You are a spatial reasoning coding agent. Solve from the question, delivered images and valid observations.

## Execution and evidence
{{ execution_contract }}
- Output one fenced Python block, no outside prose. Use top-level code or def solve(ctx); the host calls solve. Define variables; use None/True/False. Comments and observation reasons must be English.
- Imports: numpy/scipy/math/statistics only. No eval/exec/__import__/open, other imports, file writes or network access. Do not overwrite show/ReturnAnswer/YieldObservations/AnswerPayload/ReadObservation/ToolData/ReadNextObservation/MeasurementAnswer/tools/scene/frames.
- End with return ReturnAnswer(payload) or return YieldObservations([actual_result_id,...], "reason"). ReturnAnswer takes one argument and is provisional until geometry and valid-evidence checks pass; rejection revokes it. Budgets and recovery are host-controlled.
- Variables reset between rounds. Rebuild from current feedback and valid IDs. Successful current evidence calls without control may produce observation feedback only; variables, print and prose never submit answers. Caught failures reject submissions in every phase.
- Use only currently listed tools/unit modes. The complete reference method stays fixed; newer failure feedback can withdraw capabilities or invalidate results. A Skill grants no tools, budget, scoring changes or M11 bypass.
- Outer status=ok means call success only: inspect inner status, ambiguity, missing items and degradation. Failed, unresolved, unhealthy or invalidated evidence cannot support answers. Missing/omitted fields and empty detections are unknown, not zero or absence.
- M5/Tool IDs identify candidates, not question referents or scene completeness. Match depicted regions and spatial qualifiers before selecting IDs; keep missing bindings visual. M11 checks technical validity/derivation, not intended identity.
- Do not repeat deterministically invalid calls with unchanged arguments/evidence. Independent valid evidence can support a later answer. Changing comments or basis alone repairs nothing.
- With readable images submit the best legal answer; uncertainty permits an honest visual estimate, not abstain. Never fabricate geometry, scale, observations or IDs. Telemetry selected_object_count/resolved_tool_object_count is read-only, never an answer or required nonzero selection.

## Answer and observation interfaces
AnswerPayload(value=answer, unit={{ answer_unit_repr }}, basis=..., used_result_ids=..., derivation=...) is available without import.
- visual_estimate: zero-tool segment entirely from delivered images, used_result_ids=[], derivation=None.
- mixed: combined visual and valid tool support; list actual IDs used. Use mixed if one registered replay operation cannot fully establish the answer.
- tool_derived: nonempty valid IDs plus {"op":...,"input_result_ids":[...],"parameters":{...}}. A single operation must establish the value; M11 replays actual ToolResults. Paths start at raw tool data, omit "data."; parameters are structured fields, never code/expressions. Keep full measurement precision; no early rounding or .2f strings.
Task derivation help: {{ derivation_contract_json }}
- Short-form ReturnAnswer(value) is recorded as mixed and checks historical successful evidence. Only a genuine zero-tool visual segment with the complete payload may discard history. Current-segment results are always checked, regardless of basis.
- Tool envelope: result_id, source_tool, status, data, schema_version, evidence_version. Read unchanged list/dict/scalar at r["data"] or ToolData(r); use host r["result_id"], not obj_id or inner business IDs. ID-only/edited envelopes are rejected.
- ReadObservation(result_id, fields=None, offset=0, limit=8) reads valid results without rerunning tools. ReadNextObservation(result_or_id, fields=None, limit=8) reads the next uncovered gap; progress survives resets. Reader data is ALWAYS a list: a dict/scalar original root is ToolData(page)[0].
- A page can be shorter than limit; follow returned cursors, never offset += limit. page["coverage_complete"] and next_cursor=None describe only that page/tail; page["vsi_read_progress"]["coverage_complete"] tracks cumulative coverage for selected fields. Omitted fields inherit previous selection; changing fields starts independent coverage.
- YieldObservations([page["result_id"]], "reason") delivers the page next request; YieldObservations([], "reason") returns still-valid successful episode results. Use literal IDs from current feedback. print is diagnostic. Generated crops are usable only after pixels arrive in a completed subsequent request.

## Question
{{ question }}
Question type: {{ question_type or "unclassified" }}. Submission unit: {{ answer_unit }}.
Task meaning: {{ task_definition }}
{% if options %}Options:
{% for opt in options %}{{ opt }}
{% endfor %}Submit exactly one current option letter.
{% else %}Submit one finite non-negative number, never a list/dict/string; counts must be integral.
{% endif %}Preserve object roles, spatial scope, reference frame and units. Follow actual tool computation definitions; visual estimates are not tool measurements.
{{ task_rules }}
{% if direction_mapping_json %}Current direction-to-letter mapping: {{ direction_mapping_json }}. Use option_map on field="direction".
{% endif %}
## Current evidence
{{ route_header }}
{{ scene_summary }}
The public profile has geometry_3d and metric_scale, each available/unavailable. Warnings add no states. Scene geometry does not establish object bindings or collection coverage.

## Authorized tools
{{ tool_docs }}
Signatures, dependencies and allowed modes apply at execution; local binding/call preconditions still apply.
{% if skill_entries %}
## Reference method (complete, unchanged)
{% for e in skill_entries %}{{ e }}{% endfor %}{% endif %}"""


def compact_execution_contract(max_chars: int) -> str:
    contract = ExecutionContract(max_chars)
    facts = contract.to_dict()
    # Identity refers to the COMPLETE shared enforcement contract, not this
    # compact presentation. No replacement enforcement contract is introduced.
    metadata = {key: facts[key] for key in (
        "method_context_max_chars", "finalization_schema_version",
        "finalization_method_delivery", "tool_response_schema_version",
        "observation_fallthrough_version", "progress_policy", "image_pack_version")}
    reader = facts["host_observation_reader"]
    metadata["reader_limits"] = {key: reader[key] for key in (
        "max_page_items", "max_page_chars", "max_reads_per_round")}
    return (
        f"Shared enforcement: {contract.contract_version}; sha256={contract.sha256}. "
        f"Presentation: {COMPACTION_VERSION}.\n"
        "Every episode starts with 32 frozen original frames; each request contains only "
        "its declared pack, with no persistent visual memory from earlier requests.\n"
        + json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


class PromptBuilder:
    def __init__(self) -> None:
        self.template_version = PROMPT_TEMPLATE_VERSION
        self._env = Environment(undefined=StrictUndefined)

    def render(
        self, question: str, scene_summary: str, scene_frame: str, tool_docs: str,
        options: Optional[Sequence[str]] = None, skills: Optional[Sequence[SkillSpecV11]] = None,
        scope: str = "", available_artifacts: Optional[Sequence[str]] = None,
        route_header: str = "", question_type: str = "", evidence_profile=None,
        gate_passed: Optional[bool] = None, gate_missing: Optional[Sequence[str]] = None,
        *, skill_plan: Optional[SkillDeliveryPlan] = None,
        method_context_max_chars: Optional[int] = None,
    ) -> str:
        """Keep the shared public signature and exact complete-Skill delivery."""
        if not route_header and scope:
            from skill3d.tools import REGISTRY

            route_header = REGISTRY.docs_header(
                scope, available_artifacts or (), question_type=question_type,
                gate_passed=gate_passed, evidence_profile=evidence_profile, gate_missing=gate_missing)
        plan = skill_plan
        if plan is None and skills:
            plan = plan_delivery(skills, max_chars=method_context_max_chars)
        unit = CANONICAL_UNIT_BY_QUESTION_TYPE.get(question_type, "")
        template = COUNTING_PROGRAM if question_type == "object_counting" else _PROGRAM_SYNTH
        return self._env.from_string(template).render(
            counting_context_version=COUNTING_CONTEXT_VERSION,
            execution_contract=compact_execution_contract(
                plan.max_chars if plan is not None else (method_context_max_chars or 8000)),
            question=question, scene_summary=scene_summary, scene_frame=scene_frame,
            tool_docs=tool_docs, route_header=route_header or "(Unspecified scope: use listed tools only.)",
            question_type=question_type, answer_unit=unit, answer_unit_repr=repr(unit) if unit else "unit",
            task_definition=_TASK_DEFINITIONS.get(question_type, ""),
            task_rules=_TASK_RULES.get(question_type, ""),
            options=list(options) if options else None,
            derivation_contract_json=json.dumps(task_derivation_contract(question_type),
                                                sort_keys=True, separators=(",", ":")),
            direction_mapping_json=(json.dumps(direction_option_mapping(options or []), sort_keys=True)
                                    if question_type == "object_rel_direction" else ""),
            skill_entries=[entry.text for entry in (plan.entries if plan is not None else [])])

    def render_messages(self, **kwargs) -> list[dict]:
        return [{"role": "user", "content": self.render(**kwargs)}]


def round_action_instructions(question_type: str, *, finalizing: bool = False) -> str:
    if finalizing:
        return ("Required action: one strict JSON answer object under the current schema; "
                "no fences, prose, code, tools or observation actions.")
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE.get(question_type)
    return (
        "Required action: one executable Python fenced block; define variables using None/True/False. "
        "Old variables are cleared; reuse current valid host IDs. Repair statements, not comments; "
        "do not repeat invalid calls with unchanged arguments/evidence. Call only documented tools. "
        "Request pages with return YieldObservations([actual_result_id], 'reason'). "
        "Define answer from current question/images/valid evidence, then submit "
        f"return ReturnAnswer(AnswerPayload(value=answer, unit={repr(unit) if unit else 'unit'}, "
        "basis=basis, used_result_ids=used_ids, derivation=derivation)). "
        "visual_estimate requires a zero-tool image-only segment, used_ids=[], derivation=None; "
        "mixed lists actual valid IDs; tool_derived also needs one replayable derivation. "
        "Keep measurement precision. Invalid/rejected evidence is unusable; changing basis or "
        "short-form answers bypasses nothing. Answer when images are readable; no abstain.")


def build_finalization_prompt(*, question: str, options: Sequence[str], question_type: str,
                              observations: str, remaining_requests: int,
                              method_context_max_chars: int) -> str:
    from skill3d.online.submission import FINALIZATION_DELIVERY_STRATEGY, FINALIZATION_SCHEMA_VERSION

    unit = CANONICAL_UNIT_BY_QUESTION_TYPE[question_type]
    contract = ExecutionContract(method_context_max_chars)
    lines = [
        f"Final answer request ({FINALIZATION_PROMPT_VERSION}); schema={FINALIZATION_SCHEMA_VERSION}.",
        f"Shared enforcement sha256={contract.sha256}; presentation={COMPACTION_VERSION}.",
        f"Method delivery: {FINALIZATION_DELIVERY_STRATEGY}. Remaining requests including this: "
        f"{remaining_requests}; no added repair request.",
        f"Question: {question}\nTask: {question_type}. {_TASK_DEFINITIONS.get(question_type, '')}",
        "Options: " + json.dumps(list(options), ensure_ascii=False),
        f'Output one strict JSON object: action="answer", value, basis, used_result_ids, derivation; '
        f"optional unit must equal {unit!r}. No extra/duplicate keys, prose, fences, NaN or Infinity.",
        "Tools and Python are disabled; the host submits through the same M11 checks.",
        ("value must be one current option letter." if unit == "option" else
         "value must be a finite non-negative JSON number; counts must be integral."),
        "visual_estimate: entirely from currently delivered images, used_result_ids=[], derivation=null. "
        "mixed: list valid IDs actually used. tool_derived: nonempty valid IDs plus one replayable "
        "derivation(op,input_result_ids,parameters); all input IDs must be declared in used_result_ids. "
        "Paths address raw tool data, no data prefix or expressions. Keep full numeric precision.",
        "Task derivation help: " + json.dumps(task_derivation_contract(question_type),
                                            sort_keys=True, separators=(",", ":")),
        _TASK_RULES.get(question_type, ""),
        "Use only currently delivered images and still-valid observations. Missing/rejected evidence "
        "is not zero; invent no IDs/geometry/scale. Changing basis repairs no failed checks. "
        "With readable images give the best legal answer; no abstain.",
        observations,
    ]
    if question_type == "object_rel_direction":
        lines.insert(-2, "Current direction-to-letter mapping: "
                     + json.dumps(direction_option_mapping(options), sort_keys=True)
                     + '; use option_map on field="direction".')
    if question_type == "object_counting":
        lines.insert(2, f"Counting presentation: {COUNTING_CONTEXT_VERSION}.")
    return "\n".join(lines)


SCENE_OVERVIEW_GUIDANCE = (
    "Low-resolution rendered cutaway, not a video/time sample. Gaps or cutaway omissions "
    "prove no absence/completeness. Orientation may differ from the task. Grants no metric "
    "quality/scale, world-frame validity or tools. Use original frames for appearance/time order.")


def scene_overview_manifest_text(record) -> str:
    return (
        f"[Scene overview] image_id={record.image_id} scene_id={record.scene_id} "
        f"delivered_size={record.sent_hw} azimuth_degrees={record.selected_azimuth_degrees} "
        f"elevation_degrees={record.elevation_degrees} world_frame_status={record.world_frame_status}. "
        + SCENE_OVERVIEW_GUIDANCE)


def build_image_messages(text: str, frames, *, max_images: int = 32,
                         jpeg_quality: int = 80, extra: Optional[Sequence] = None,
                         image_labels: Optional[Sequence[str]] = None) -> list[dict]:
    """Shared image encoding/order contract, retained in the local override."""
    import base64
    import cv2
    import numpy as np

    content: list[dict] = [{"type": "text", "text": text}]
    arr = list(frames)
    if extra:
        arr = [np.asarray(img) for _, img in extra] + arr
    if len(arr) > max_images:
        raise ValueError(f"M8 received {len(arr)} images but max_images={max_images}; no silent omission")
    if not arr:
        raise ValueError("M8 requires readable images; zero frames prohibited")
    if image_labels is not None and len(image_labels) != len(arr):
        raise ValueError("Each delivered image must have exactly one identity label")
    for index, image in enumerate(arr):
        try:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        except Exception as exc:
            raise ValueError(f"M8 image {index}/{len(arr)} failed JPEG encoding: {exc}; no omission") from exc
        if not ok:
            raise ValueError(f"M8 image {index}/{len(arr)} failed JPEG encoding; no omission")
        if image_labels is not None:
            content.append({"type": "text", "text": image_labels[index]})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")}})
    return [{"role": "user", "content": content}]
