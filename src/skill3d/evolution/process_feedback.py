"""Label-free, allowlisted process facts for offline method revision.

This module never reads labels, emits source text, or interprets tool activity as
visual understanding. Legacy traces with absent telemetry stay unknown.
"""

from __future__ import annotations

import ast
from collections import Counter
import json
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field
from skill3d.sandbox.action_check import ARGUMENT_ERROR_REASONS

FEEDBACK_SCHEMA_VERSION = "fitset-process-feedback/3.0"
FEEDBACK_WHITELIST_VERSION = "process-facts/1.0"

ERROR_CODES = frozenset({
    "model_output_format_error", "program_syntax_error", "ast_violation",
    "program_missing_action", "finalization_missing_submission",
    "finalization_tool_call_disallowed", "tool_argument_error",
    "tool_precondition_rejected", "geometry_verification_rejected",
    "no_progress_budget_exhausted", "service_error", "input_error",
    "answer_contract_rejected",
    "runtime_error", "unknown_error",
})
EXCEPTION_TYPES = frozenset({
    "SyntaxError", "TypeError", "KeyError", "IndexError", "AttributeError",
    "NameError", "UnboundLocalError", "ValueError", "ToolContractError",
    "DomainValueError", "ArtifactUnavailableError", "ConfidenceGateError",
    "TimeoutError",
})
FIELD_SEMANTICS = {
    "response_completed": "Images sent with a completed response; does not prove understanding.",
    "tool_calls": "Optional additional evidence; zero calls does not mean no visual input.",
    "candidate_count": "Last successfully read candidate list size; not ground truth or the answer. Null means unread.",
    "resolved_tool_object_count": "Distinct IDs resolved by tools; not semantic referent selection.",
    "explicit_selection": "A model declaration, if recorded; does not validate the referent. Missing is unknown.",
    "score_band": "Separate full/partial/zero answered/no answer; partial is not complete failure.",
    "error_chain": "Observed execution events; a protective precondition rejection is not a broken tool.",
}


class _Facts(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputObservation(_Facts):
    original_frames_sent: int | None = Field(default=None, ge=0)
    original_frames_response_completed: int | None = Field(default=None, ge=0)
    derived_images_produced: int | None = Field(default=None, ge=0)
    derived_images_response_completed: int | None = Field(default=None, ge=0)
    current_context_original_count: int | None = Field(default=None, ge=0)
    current_context_derived_count: int | None = Field(default=None, ge=0)


class Referents(_Facts):
    candidate_count: int | None = Field(default=None, ge=0)
    resolved_tool_object_count: int = Field(default=0, ge=0)
    explicit_selection_recorded: bool = False
    explicit_selection_count: int | None = Field(default=None, ge=0)
    selection_validation: Literal["unknown"] = "unknown"


class Execution(_Facts):
    terminal_action: Literal["answer", "yield", "missing", "unknown"] = "unknown"
    answer_submitted: bool = False
    answer_assignment_without_submit: bool = False
    tool_calls_by_name: dict[str, int] = Field(default_factory=dict)
    successful_yields: int = Field(default=0, ge=0)
    repeated_no_progress_rounds: int = Field(default=0, ge=0)
    model_rounds: int = Field(default=0, ge=0)


class ErrorEvent(_Facts):
    stage: Literal["generation", "static", "execution", "verification", "finalization"]
    round_index: int = Field(ge=0)
    event_type: str
    tool_name: str | None = None
    exception_type: str | None = None
    argument_shape: dict[str, int | str] = Field(default_factory=dict)
    return_type: Literal["dict", "list", "float", "int", "bool", "str", "null", "unknown"] = "unknown"
    recovered: bool | None = None
    terminal: bool = False
    capability_change: bool = False


class ObservationQuality(_Facts):
    candidate_coverage: Literal["unknown", "complete", "incomplete"] = "unknown"
    summary_truncated: bool = False
    required_fields_omitted: list[str] = Field(default_factory=list)
    referenced_image_currently_present: bool | None = None


class ProcessFacts(_Facts):
    feedback_schema_version: Literal["fitset-process-feedback/3.0"] = FEEDBACK_SCHEMA_VERSION
    input_observation: InputObservation = Field(default_factory=InputObservation)
    referents: Referents = Field(default_factory=Referents)
    execution: Execution = Field(default_factory=Execution)
    error_chain: list[ErrorEvent] = Field(default_factory=list)
    first_actionable_error: str | None = None
    terminal_error: str | None = None
    observation_quality: ObservationQuality = Field(default_factory=ObservationQuality)


def score_band(score: float, answered: bool) -> str:
    return ("no_answer" if not answered else "full_score" if score == 1
            else "partial_score" if score > 0 else "zero_score_answered")


def _payload(result: Mapping):
    value = result.get("payload")
    if value is not None:
        return value
    try:
        return json.loads(result.get("value", "null"))
    except (ValueError, TypeError):
        return None


def _program_facts(source: str) -> tuple[set[str], bool]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return set(), False
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assignment = any(isinstance(n, ast.Name) and n.id == "answer"
                     and isinstance(n.ctx, ast.Store) for n in ast.walk(tree))
    return calls, assignment


def classify_round(row: Mapping) -> str | None:
    """Classify old and new records without changing their stored scores."""
    code = row.get("error_code") or row.get("cause_code")
    if not code:
        return None
    if code in ERROR_CODES and code not in ("runtime_error", "unknown_error"):
        # Old finalization merged missing actions and tool calls into AST errors.
        if code != "ast_violation" or row.get("trigger") != "finalize":
            return code
    calls, _ = _program_facts(str(row.get("program_source") or ""))
    if row.get("trigger") == "finalize" and code in ("ast_violation", "no_answer"):
        from skill3d.tools import REGISTRY
        if calls & (set(REGISTRY.names()) | {"YieldObservations"}):
            return "finalization_tool_call_disallowed"
        if "ReturnAnswer" not in calls:
            return "finalization_missing_submission"
    if code in ("parse_error", "program_parse_error", "online_model_parse_error"):
        return "model_output_format_error"
    if code == "no_answer":
        return "program_missing_action"
    if code == "geometry_rejected":
        return "geometry_verification_rejected"
    diagnostics = row.get("execution_diagnostics") or {}
    if any((result.get("failure_cause") or {}).get("code") in ARGUMENT_ERROR_REASONS
           for result in row.get("results", [])):
        return "tool_argument_error"
    if code in ("tool_contract", "confidence_gate", "domain_value"):
        return "tool_precondition_rejected"
    if diagnostics.get("error_type") in ("TypeError", "KeyError", "IndexError", "ValueError"):
        return "tool_argument_error"
    if diagnostics.get("error_type") == "SyntaxError":
        return "program_syntax_error"
    if row.get("service_errors") or code == "online_model_service_error":
        return "service_error"
    return "ast_violation" if code == "ast_violation" else "runtime_error"


def extract_process_facts(outcome: Mapping) -> ProcessFacts:
    from skill3d.tools import REGISTRY

    facts = ProcessFacts()
    rounds = list(outcome.get("rounds") or [])
    facts.execution.model_rounds = len(rounds)
    submitted = outcome.get("final_state") == "answer"
    facts.execution.answer_submitted = submitted
    ledger = outcome.get("image_ledger") or {}
    if "images" in ledger:
        images = ledger["images"]
        originals = [im for im in images if im.get("kind") == "frame"]
        derived = [im for im in images if im.get("kind") != "frame"]
        observed = lambda values: sum(bool(im.get("observed_rounds")) for im in values)
        facts.input_observation.original_frames_sent = sum(
            bool(im.get("delivered_rounds")) for im in originals)
        facts.input_observation.original_frames_response_completed = observed(originals)
        facts.input_observation.derived_images_produced = len(derived)
        facts.input_observation.derived_images_response_completed = observed(derived)
    packs = [p for p in ledger.get("rounds", []) if p.get("delivered")]
    if packs:
        facts.input_observation.current_context_original_count = len(packs[-1].get("original_image_ids", []))
        facts.input_observation.current_context_derived_count = len(packs[-1].get("derived_image_ids", []))
    resolved = set()
    tool_counts: Counter = Counter()
    for index, row in enumerate(rounds):
        calls, assignment = _program_facts(str(row.get("program_source") or ""))
        if index == len(rounds) - 1:
            facts.execution.terminal_action = (
                "answer" if submitted else "yield" if "YieldObservations" in calls
                else "missing" if calls or assignment else "unknown")
        facts.execution.answer_assignment_without_submit |= (
            assignment and "ReturnAnswer" not in calls)
        if row.get("executed") and not row.get("error_code") and "YieldObservations" in calls:
            facts.execution.successful_yields += 1
        if (row.get("progress") or {}).get("made_progress") is False:
            facts.execution.repeated_no_progress_rounds += 1
        results = row.get("results") or []
        for result in results:
            name = result.get("source_tool") or result.get("tool")
            tool_counts[name if name in REGISTRY.names() else "unknown_tool"] += 1
            resolved.update(str(v) for v in result.get("resolved_object_ids", []))
            payload = _payload(result)
            if name == "list_objects" and result.get("status") == "ok" and isinstance(payload, list):
                facts.referents.candidate_count = len(payload)
        code = classify_round(row)
        if code:
            failed = next((r for r in reversed(results)
                           if r.get("status") == "failed" or r.get("error")), {})
            tool = failed.get("source_tool") or failed.get("tool")
            args = failed.get("args") or {}
            shape = {f"{k}_length": len(args[k]) for k in ("frame_ids", "boxes")
                     if isinstance(args.get(k), list)}
            diagnostics = row.get("execution_diagnostics") or {}
            exception = diagnostics.get("error_type")
            observed_payload = _payload(failed or (results[-1] if results else {}))
            return_type = type(observed_payload).__name__ if observed_payload is not None else "null"
            facts.error_chain.append(ErrorEvent(
                stage=("finalization" if row.get("trigger") == "finalize" else
                       "verification" if code == "geometry_verification_rejected" else
                       "generation" if code in ("model_output_format_error", "service_error") else
                       "static" if not row.get("executed") else "execution"),
                round_index=int(row.get("index") or index + 1),
                event_type=code, tool_name=tool if tool in REGISTRY.names() else None,
                exception_type=exception if exception in EXCEPTION_TYPES else None,
                argument_shape=shape,
                return_type=return_type if return_type in (
                    "dict", "list", "float", "int", "bool", "str", "null") else "unknown",
                recovered=True if submitted else None,
                terminal=index == len(rounds) - 1 and not submitted,
                capability_change=bool(row.get("downgraded_capabilities")),
            ))
        # Only fixed omission facts survive. Feedback/source text itself never does.
        feedback = str(row.get("feedback") or "")
        if ('"coverage_complete":false' in feedback.replace(" ", "")
                or "coverage_complete=false" in feedback):
            facts.observation_quality.summary_truncated = True
    facts.execution.tool_calls_by_name = dict(sorted(tool_counts.items()))
    facts.referents.resolved_tool_object_count = len(resolved)
    if facts.error_chain:
        facts.first_actionable_error = facts.error_chain[0].event_type
        facts.terminal_error = next((e.event_type for e in reversed(facts.error_chain) if e.terminal), None)
    return facts


def safe_process_dict(raw: Mapping) -> dict:
    """Revalidate and whitelist even persisted/malicious feedback before sending."""
    from skill3d.tools import REGISTRY

    facts = ProcessFacts.model_validate(raw)
    facts.execution.tool_calls_by_name = {
        name: count for name, count in facts.execution.tool_calls_by_name.items()
        if name in REGISTRY.names() or name == "unknown_tool"
    }
    for event in facts.error_chain:
        if event.event_type not in ERROR_CODES:
            event.event_type = "unknown_error"
        if event.tool_name not in REGISTRY.names():
            event.tool_name = None
        if event.exception_type not in EXCEPTION_TYPES:
            event.exception_type = None
        event.argument_shape = {
            key: value for key, value in event.argument_shape.items()
            if key in ("frame_ids_length", "boxes_length")
            and type(value) is int and value >= 0
        }
    facts.first_actionable_error = (
        facts.first_actionable_error if facts.first_actionable_error in ERROR_CODES else None)
    facts.terminal_error = facts.terminal_error if facts.terminal_error in ERROR_CODES else None
    facts.observation_quality.required_fields_omitted = [
        value for value in facts.observation_quality.required_fields_omitted
        if value in ("visible_frames", "track_id", "binding_flags", "grounding_status", "mask_support_ratio")
    ]
    return facts.model_dump(mode="json")


def legacy_process_facts(signature: Mapping) -> ProcessFacts:
    """Read v2 counters without reinterpreting missing telemetry as zero evidence."""
    facts = ProcessFacts()
    facts.execution.answer_submitted = signature.get("final_state") == "answer"
    facts.execution.terminal_action = "answer" if facts.execution.answer_submitted else "unknown"
    for name, target in (("selected_object_count", facts.referents),
                         ("model_rounds", facts.execution)):
        value = signature.get(name)
        if type(value) is int and value >= 0:
            setattr(target, "resolved_tool_object_count" if name == "selected_object_count"
                    else name, value)
    return facts
