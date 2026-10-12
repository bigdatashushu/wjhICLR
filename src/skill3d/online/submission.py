"""v11 提交验收范围与审计；不调用模型，不读取评分真值。"""

from __future__ import annotations

from skill3d.verifier.geometry_oracle import GeometryIssue
from skill3d.schemas.tool import FailureCause

EXECUTION_PROTOCOL_VERSION = "solver-v11.5-core-engineering-repair"


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
