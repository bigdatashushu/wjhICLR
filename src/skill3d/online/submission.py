"""v11 提交验收范围与审计；不调用模型，不读取评分真值。"""

from __future__ import annotations

import json
import math
from typing import Sequence

from skill3d.schemas import AnswerPayload, CANONICAL_UNIT_BY_QUESTION_TYPE
from skill3d.verifier.geometry_oracle import GeometryIssue
from skill3d.schemas.tool import FailureCause

EXECUTION_PROTOCOL_VERSION = "solver-v11.9-structured-finalization"
FINALIZATION_SCHEMA_VERSION = "answer-finalization/1.0"
FINALIZATION_DELIVERY_STRATEGY = "omit_complete_method"


class SubmissionFormatError(ValueError):
    """A response failed the strict finalization boundary; never execute it."""

    def __init__(self, reason: str, code: str = "model_output_format_error") -> None:
        super().__init__(reason)
        self.code = code


def _json_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SubmissionFormatError("Duplicate JSON keys are not allowed")
        value[key] = item
    return value


def _nonfinite(value):
    raise SubmissionFormatError("Non-finite JSON numbers are not allowed")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        _nonfinite(value)
    return parsed


def parse_finalization(text: str, *, question_type: str, options: Sequence[str],
                       valid_result_ids: Sequence[str], readable_images: bool) -> AnswerPayload:
    """Accept one typed JSON object, with no extraction, coercion or code execution.

    Only host-issued, currently valid result IDs may be declared. Geometry and
    derivation checks still run at M11 after this format boundary succeeds.
    """
    try:
        raw = json.loads(text, object_pairs_hook=_json_pairs, parse_constant=_nonfinite,
                         parse_float=_finite_float)
    except (ValueError, TypeError, RecursionError) as exc:
        raise SubmissionFormatError("Expected exactly one JSON answer object") from exc
    required = {"action", "value", "basis", "used_result_ids", "derivation"}
    if type(raw) is not dict or not required <= raw.keys() or raw.keys() - required - {"unit"}:
        raise SubmissionFormatError("Use only action, value, basis, used_result_ids, derivation and optional unit")
    if raw["action"] != "answer":
        raise SubmissionFormatError("The only finalization action is answer",
                                    "finalization_tool_call_disallowed")
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE.get(question_type)
    if unit is None or ("unit" in raw and raw["unit"] != unit):
        raise SubmissionFormatError("Unit must match the host task contract", "answer_contract_rejected")
    value = raw["value"]
    if unit == "option":
        if type(value) is not str or value not in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]):
            raise SubmissionFormatError("value must be one current option letter", "answer_contract_rejected")
    else:
        try:
            valid_number = type(value) in (int, float) and math.isfinite(value) and value >= 0
        except OverflowError:
            valid_number = False
        if not valid_number or (unit == "count" and value != int(value)):
            raise SubmissionFormatError("value must be one finite non-negative number; count must be integral",
                                        "answer_contract_rejected")
    ids = raw["used_result_ids"]
    if (type(ids) is not list or not all(type(rid) is str and rid for rid in ids)
            or len(ids) != len(set(ids))):
        raise SubmissionFormatError("used_result_ids must be unique nonempty strings")
    if set(ids) - set(valid_result_ids):
        raise SubmissionFormatError("A result reference is missing, failed, unresolved or invalidated",
                                    "answer_contract_rejected")
    if raw["basis"] not in ("visual_estimate", "mixed", "tool_derived"):
        raise SubmissionFormatError("Unknown answer basis", "answer_contract_rejected")
    if raw["basis"] in ("visual_estimate", "mixed") and not readable_images:
        raise SubmissionFormatError("Visual support requires readable images", "input_error")
    if raw["derivation"] is not None and type(raw["derivation"]) is not dict:
        raise SubmissionFormatError("derivation must be an object or null")
    payload = AnswerPayload(
        value=value, unit=unit, basis=raw["basis"], used_result_ids=ids,
        derivation=raw["derivation"])
    problems = payload.problems()
    if problems:
        raise SubmissionFormatError("; ".join(problems), "answer_contract_rejected")
    return payload


def finalization_program(payload: AnswerPayload):
    """Encode validated data as literals in a single host-built submission call.

    The model response itself is never interpreted as source. Existing M10/M11
    recording and rejection handling remain the sole submission acceptance path.
    """
    from skill3d.synthesis.program_assembler import assemble_program

    arguments = ", ".join(f"{key}={value!r}"
                          for key, value in payload.model_dump(mode="json").items())
    return assemble_program(f"ReturnAnswer(AnswerPayload({arguments}))\n", [])


def submission_scope(kernel, trace):
    """覆盖跨轮有效结果；明确的纯视觉提交可舍弃历史工具依据。

    当前片段实际产生的结果始终接受检查，不能以 visual_estimate 声明绕过。
    短写答案没有可证实的依赖集合，保守检查全部未撤销成功结果。
    """
    payload = kernel.answer_slot.payload
    current = {r.result_id for r in trace.results}
    referenced = (set(payload.used_result_ids) | set(payload.derivation_inputs())
                  if payload is not None else set())
    visual_only = bool(payload and payload.basis == "visual_estimate"
                       and not referenced)
    results = [r for r in kernel.tool_results
               if r.status == "ok" and r.error is None and not r.invalidated_by
               and (
                   (r.result_id in current)
                   or (r.result_id in referenced)
                   or (not visual_only and not referenced)
               )]
    return trace.model_copy(update={"results": results})


def invalid_reference_issues(kernel) -> list[GeometryIssue]:
    """显式引用必须存在、成功且未撤销，不因 mixed 声明而放宽。"""
    payload = kernel.answer_slot.payload
    if payload is None:
        return []
    referenced = set(payload.used_result_ids) | set(payload.derivation_inputs())
    by_id = {r.result_id: r for r in kernel.tool_results}
    issues = []
    for rid in sorted(referenced):
        result = by_id.get(rid)
        if result is None:
            check, reason = "missing_reference", "This answer references a nonexistent tool result"
        elif result.invalidated_by:
            check, reason = "invalidated_reference", "This answer explicitly references an invalidated tool result"
        elif result.status != "ok" or result.error is not None:
            check, reason = "failed_reference", "This answer references a failed tool result"
        else:
            continue
        issues.append(GeometryIssue(
            check=check, tool=result.tool if result else "", result_id=rid, reason=reason,
            failure_cause=FailureCause(
                code=check, scope="local_result", result_ids=[rid],
                object_ids=list(result.object_dependencies) if result else [],
                message=reason)))
    return issues


def rejected_submission(kernel) -> dict:
    slot = kernel.answer_slot
    return {
        "answer": slot.answer,
        "payload": slot.payload.model_dump(mode="json") if slot.payload else None,
        "given": slot.given,
        "adapter_problems": list(slot.adapter_problems),
    }


def accepted_result_ids(kernel, trace) -> list[str]:
    """最终关联集合：显式声明优先；短写答案保守关联整个验收范围。"""
    payload = kernel.answer_slot.payload
    if payload is not None and payload.basis == "visual_estimate":
        return []
    referenced = (set(payload.used_result_ids) | set(payload.derivation_inputs())
                  if payload is not None else set())
    available = [r.result_id for r in trace.results if r.result_id]
    return [rid for rid in available if not referenced or rid in referenced]


def used_evidence_flags(kernel, result_ids, handle=None) -> list[dict]:
    """只追溯最终验收引用；拒绝、撤销、未引用的诊断不混入答案。"""
    import json
    rows = []

    def add(result, layer, flag, **extra):
        flag = flag if isinstance(flag, dict) else {"code": str(flag)}
        code = str(flag.get("code") or flag.get("reason") or "")
        if code:
            rows.append({
                "result_id": result.result_id, "tool": result.tool, "layer": layer,
                "code": code, "audit_id": flag.get("audit_id") or flag.get("audit_sha256"),
                **extra,
            })

    def visit(result, node, path="payload"):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("degradation_flags", "binding_flags") and isinstance(value, list):
                    for flag in value:
                        add(result, "payload" if key == "degradation_flags" else "binding",
                            flag, path=path + "." + key)
                elif key == "duplicate_suspect" and value is True:
                    add(result, "binding", "suspect_duplicate", path=path)
                else:
                    visit(result, value, path + "." + key)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                visit(result, value, f"{path}[{i}]")

    wanted = set(result_ids)
    for result in kernel.tool_results:
        if (result.result_id not in wanted or result.status != "ok"
                or result.error is not None or result.invalidated_by):
            continue
        for flag in result.degraded_evidence:
            add(result, "evidence_profile", flag)
        try:
            payload = result.payload if result.payload is not None else json.loads(result.value)
        except (TypeError, ValueError):
            payload = None
        visit(result, payload)
        if handle is not None:
            for oid in result.object_dependencies:
                try:
                    obj = handle.get_object(oid)
                except (KeyError, ValueError):
                    continue
                for flag in obj.binding_flags:
                    add(result, "binding", flag, object_id=oid)
                if obj.duplicate_suspect:
                    add(result, "binding", "suspect_duplicate", object_id=oid)
    unique = {json.dumps(row, sort_keys=True): row for row in rows}
    return [unique[key] for key in sorted(unique)]
