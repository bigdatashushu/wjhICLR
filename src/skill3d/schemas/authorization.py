"""§6.4 工具授权收据与 §6.1 证据不可用原因码（v9）。

规范原文（§6.4）："每次实际调用生成**同一判定函数**产出的
`ToolAuthorizationReceipt`：至少包含 `episode_id, tool_id, argument_digest,
evidence_version, dependency_refs, allowed, reason_codes, metric_gate_result,
decision_version`。提示词中的工具清单只是当前可发现的工具面；实参涉及的具体对象
仍需执行前检查。"

两条纪律：

1. **收据与判定同源**：`allowed` 必须来自真正决定这次调用能否执行的函数
   （`tools.contract.authorize_tool_call`），不能事后拼一个"看起来通过"的收据；
2. **允许也要留痕**：此前授权只以异常形式存在（不允许才有记录），成功调用没有
   "谁授权了它"的凭据 —— 这正是"证据摘要与授权收据不能互相矛盾"无法校验的原因。
"""

from __future__ import annotations

from typing import Literal, Optional

from . import Spec
from .evidence import UNAVAILABLE_REASON_CODES, MetricGateStatus

DECISION_VERSION = "tool-authorization-v9"

# 拒绝原因码 = 既有四类契约错误码 + 作用域/对象级拒绝
DENIAL_REASON_CODES = frozenset({
    "tool_contract", "confidence_gate", "domain_value", "answer_already_given",
    "tools_disabled", "scope_denied", "object_unbound",
})


class AuthorizationDecision(Spec):
    """`authorize_tool_call` 的产出：允许与否 + 原因码 + 依赖 + 米制门状态。"""

    allowed: bool
    reason_codes: list[str] = []
    # 本次调用依赖的产物/证据引用（§6.4 `dependency_refs`）
    dependency_refs: list[str] = []
    # 未通过时缺失的产物（诊断用；通过时为空）
    missing_refs: list[str] = []
    # 米制门：米制工具给题级门结果，非米制工具给 not_applicable（§6.4）
    metric_gate_result: dict = {}
    metric_gate_applicable: bool = False


class ToolAuthorizationReceipt(Spec):
    """§6.4 每次实际调用的统一授权收据。

    `result_id` 把收据与 `ToolResult` 关联起来（§10.1：每个失败调用也必须有可追踪
    result_id）；被拒绝的调用没有工具结果 id 时留空并保持 `allowed=False`。
    """

    episode_id: str
    tool_id: str
    argument_digest: str
    evidence_version: str = ""
    dependency_refs: list[str] = []
    allowed: bool
    reason_codes: list[str] = []
    metric_gate_result: dict = {}
    decision_version: str = DECISION_VERSION
    # 审计补充（§17.1 Tool 层：参数、依赖、失效、耗时；失败也有 result_id）
    args: dict = {}
    result_id: str = ""
    scene_route: str = ""
    question_tool_scope: str = ""
    invalidated_by: list[str] = []


def not_applicable_gate(*, gate_version: str = "") -> dict:
    """非米制工具的 `metric_gate_result`：`not_applicable` + `gate_passed=None`。

    §6.4："非米制工具使用 `not_applicable`，**不能用默认 False 伪装失败**"。
    """
    return {"status": "not_applicable", "gate_passed": None,
            "gate_version": str(gate_version or ""), "sub_results": {},
            "missing_subconditions": []}


def gate_result_payload(gate: Optional[object]) -> tuple[dict, bool]:
    """题级门 → `(收据里的 payload, 是否 applicable)`。门缺失 → 不适用。"""
    if gate is None:
        return not_applicable_gate(), False
    dump = gate.model_dump() if hasattr(gate, "model_dump") else dict(gate)  # type: ignore[arg-type]
    status = str(dump.get("status", "") or "")
    applicable = status != "not_applicable"
    if not status:
        gp = dump.get("gate_passed")
        dump["status"] = ("pass" if gp else "fail") if gp is not None else "not_applicable"
    return dump, applicable
