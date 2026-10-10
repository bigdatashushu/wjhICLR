"""v11 提交验收范围与审计；不调用模型，不读取评分真值。"""

from __future__ import annotations

from skill3d.verifier.geometry_oracle import GeometryIssue

EXECUTION_PROTOCOL_VERSION = "solver-v11.2-m11-acceptance"


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
            check, reason = "missing_reference", "本次答案引用了不存在的工具结果"
        elif result.invalidated_by:
            check, reason = "invalidated_reference", "本次答案显式引用了已撤销的工具结果"
        elif result.status != "ok" or result.error is not None:
            check, reason = "failed_reference", "本次答案引用了失败的工具结果"
        else:
            continue
        issues.append(GeometryIssue(
            check=check, tool=result.tool if result else "", result_id=rid, reason=reason))
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
