"""§10.1 / §4 / §12 答案合同与工具归因台账（v9）。

规范原文（§10.1）：``EpisodeStatus = Literal["answered","input_error","run_error"]``、
``AnswerBasis = Literal["visual_estimate","tool_derived","mixed"]``，并给出
``AnswerPayload(value, unit, basis, used_result_ids, derivation)``。

§12 的三条纪律：

1. **声明由框架核验**，不仅依赖模型自称：记录
   ``attempted/succeeded_tool_calls, observed_result_ids, declared_used_result_ids,
   verified_used_result_ids, ignored_result_ids``；
2. ``derivation`` 用可重放的 ``op, input_result_ids, parameters``，**只允许登记操作**
   （字段提取／换算／计数／排序／argmin／选项映射），不执行其中的任意代码；
3. **无法充分验证工具归因时保守记 ``mixed`` 并保留问题记录，不凭此否决格式合法的
   预测**；使用已失效证据按恢复规则处理。

单位来源（§4 题目适配与答案合同）：``unit`` 不是靠猜 —— §4 的表就是官方
"题型 → 单位"合同（计数为非负整数、绝对距离为米、尺寸为厘米、房间面积为平方米，
其余四类为题目给定选项）。历史短写 ``ReturnAnswer(value)`` 经
`parse_answer_payload` 显式适配：单位按 §4 表登记，basis 保守记 ``mixed``，并把
"这是兼容适配、单位来自题型合同而非模型声明"写进问题记录。
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from . import Spec

# ---- §10.1 词汇 ----
EpisodeStatus = Literal["answered", "input_error", "run_error"]
AnswerBasis = Literal["visual_estimate", "tool_derived", "mixed"]
# §4 的单位词汇（选项题用 `option`，负值用题目给定标签而非硬编码 A–D）
AnswerUnit = Literal["option", "count", "m", "cm", "m2"]

# §4 表：规范题型 → 答案单位。**这是官方合同**，不是启发式猜测。
CANONICAL_UNIT_BY_QUESTION_TYPE: dict[str, AnswerUnit] = {
    "object_counting": "count",
    "object_abs_distance": "m",
    "object_size_estimation": "cm",
    "room_size_estimation": "m2",
    "object_rel_distance": "option",
    "object_rel_direction": "option",
    "route_planning": "option",
    "obj_appearance_order": "option",
}

# §12 登记的可重放操作。**不执行其中的任意代码**：这里只有名字，实际重放能力由
# 评估侧实现；本模块只负责"名字必须在册"这一层校验。
DERIVATION_OPS: frozenset[str] = frozenset({
    "field",        # 字段提取
    "convert",      # 单位/尺度换算
    "count",        # 计数
    "sort",         # 排序
    "argmin",       # 取极值项
    "option_map",   # 选项映射
})

ADAPTER_VERSION_LEGACY_SHORT_FORM = "answer-adapter-v9-legacy-short-form"


class AnswerPayload(Spec):
    """§10.1 答案载荷。`derivation` 的形态取 §10.1（`dict | None`）与 §12
    （`op, input_result_ids, parameters` 三字段）的**交集**：一个可重放的变换步骤。
    """

    value: str | int | float
    unit: AnswerUnit
    basis: AnswerBasis
    used_result_ids: list[str] = []
    # {"op": <DERIVATION_OPS 之一>, "input_result_ids": [...], "parameters": {...}}
    derivation: Optional[dict] = None

    def derivation_op(self) -> str:
        return str((self.derivation or {}).get("op", "") or "")

    def derivation_inputs(self) -> list[str]:
        raw = (self.derivation or {}).get("input_result_ids") or []
        return [str(x) for x in raw]

    def problems(self) -> list[str]:
        """静态可判定的合同问题（不依赖任何工具台账）。"""
        out: list[str] = []
        if self.derivation is not None:
            op = self.derivation_op()
            if op not in DERIVATION_OPS:
                out.append(f"derivation.op={op!r} 不在登记操作 {sorted(DERIVATION_OPS)} 中")
            extra = set(self.derivation) - {"op", "input_result_ids", "parameters"}
            if extra:
                out.append(f"derivation 含未登记键 {sorted(extra)}")
            missing = sorted(set(self.derivation_inputs()) - set(self.used_result_ids))
            if missing:
                out.append(f"derivation 引用了未在 used_result_ids 声明的结果 {missing}")
        if self.basis == "tool_derived" and self.derivation is None:
            out.append("basis=tool_derived 但缺少可重放 derivation")
        if self.basis == "tool_derived" and not self.used_result_ids:
            out.append("basis=tool_derived 但未声明任何 used_result_ids")
        if self.basis == "visual_estimate" and (self.used_result_ids or self.derivation):
            out.append("basis=visual_estimate 却声明了工具结果/变换")
        return out


class AttributionLedger(Spec):
    """§12 要求的六个计数/清单字段。"""

    attempted_tool_calls: int = 0
    succeeded_tool_calls: int = 0
    observed_result_ids: list[str] = []
    declared_used_result_ids: list[str] = []
    verified_used_result_ids: list[str] = []
    ignored_result_ids: list[str] = []


class AttributionVerification(Spec):
    """框架核验结果：**保守修正后的 basis** 与保留的问题记录。"""

    declared_basis: AnswerBasis
    verified_basis: AnswerBasis
    downgraded: bool = False
    ledger: AttributionLedger
    problems: list[str] = []
    note: str = ""


def verify_attribution(payload: AnswerPayload, *, observed_result_ids: list[str],
                       succeeded_result_ids: list[str],
                       invalidated_result_ids: list[str],
                       attempted_tool_calls: int = 0,
                       succeeded_tool_calls: int = 0) -> AttributionVerification:
    """§12：用工具台账核验答案声明，必要时保守降级为 `mixed`。

    纪律：**不凭归因问题否决格式合法的预测** —— 只降级 basis 并留痕，答案照常进分。
    `tool_derived` 的成立条件是"声明的每个结果都真实存在、成功、且未被撤销"，
    而不是"本轮调用次数 > 0"（§12：调用不等于使用，零调用也可能用了缓存测量）。
    """
    observed = list(dict.fromkeys(str(x) for x in observed_result_ids))
    succeeded = set(str(x) for x in succeeded_result_ids)
    invalidated = set(str(x) for x in invalidated_result_ids)
    declared = list(dict.fromkeys(str(x) for x in payload.used_result_ids))

    problems = list(payload.problems())
    verified: list[str] = []
    for rid in declared:
        if rid in invalidated:
            problems.append(f"声明的 {rid} 已被级联撤销：不得作为有效依据（§14.1）")
            continue
        if rid not in observed:
            problems.append(f"声明的 {rid} 不在本 episode 工具台账中")
            continue
        if rid not in succeeded:
            problems.append(f"声明的 {rid} 状态非 ok")
            continue
        verified.append(rid)

    verified_basis: AnswerBasis = payload.basis
    note = ""
    if payload.basis == "tool_derived" and (problems or len(verified) != len(declared)
                                            or not declared):
        verified_basis = "mixed"
        note = "tool_derived 无法被台账充分证实 → 保守记 mixed（§12）"
    elif payload.basis == "mixed":
        note = "mixed：工具事实与视觉/语义判断结合，或无法证明完全由工具推导"

    ignored = [rid for rid in observed if rid not in set(declared)]
    ledger = AttributionLedger(
        attempted_tool_calls=int(attempted_tool_calls),
        succeeded_tool_calls=int(succeeded_tool_calls),
        observed_result_ids=observed,
        declared_used_result_ids=declared,
        verified_used_result_ids=verified,
        ignored_result_ids=ignored,
    )
    return AttributionVerification(
        declared_basis=payload.basis, verified_basis=verified_basis,
        downgraded=(verified_basis != payload.basis), ledger=ledger,
        problems=problems, note=note)


def parse_answer_payload(raw: Any, *, question_type: str = "") -> tuple[AnswerPayload, list[str]]:
    """把 `ReturnAnswer` 的入参归一成 `AnswerPayload`（§10.2 显式兼容适配器）。

    - 已是 `AnswerPayload` → 原样通过（模型声明了完整合同）；
    - 历史短写 `ReturnAnswer(value)` → 按 §4 题型合同补 `unit`，**basis 保守记
      `mixed`**（模型没声明工具贡献，无法证明完全由工具推导），并记录适配问题。

    绝不会替模型声明 `tool_derived`，也不会静默把选项题猜成数值题。
    """
    if isinstance(raw, AnswerPayload):
        return raw, []
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE.get(str(question_type or ""), "")
    problems = [f"历史短写 ReturnAnswer(value)：经 {ADAPTER_VERSION_LEGACY_SHORT_FORM} "
                "适配，basis 保守记 mixed"]
    if isinstance(raw, bool):
        # §4 的答案类型只有数值与选项标签，没有布尔。Pydantic 的 `str|int|float`
        # 会把 True 变成 1，因此显式留问题记录，不静默改变语义。
        problems.append("答案入参是 bool；§4 无布尔答案类型，按整数记并留此记录")
    if not unit:
        # 未知题型：不猜单位，记为空选项题的最保守取值并保留问题
        problems.append(f"题型 {question_type!r} 无 §4 单位合同 → unit 无法登记")
        unit = "option"
    return (AnswerPayload(value=raw, unit=unit,  # type: ignore[arg-type]
                          basis="mixed", used_result_ids=[], derivation=None),
            problems)
