"""M10 partial_tool_recovery（v6 §6.4/§14，D7）。

核心规则（逐条实现，不做变通）：

1. **保留未受污染的成功结果并回灌**：答案产生前某 Tool 失败时，不丢弃此前已成功
   执行的 Tool 结果；把成功结果当带来源/可信状态的先验（validated observations）
   回灌给模型继续恢复；
2. **失败不得伪装成功**：每个结果带 `{result_id, source_tool, status, evidence_version}`；
   `status="failed"` 的结果**不进** validated observations；
3. **保留范围**：仅当前 episode、当前 EvidenceProfile 版本下已成功、且**未被失败
   根因污染**的结果；
4. **共享前提失效时按依赖级联撤销**：若失败揭示共享前提无效（坐标/尺度/重建证据
   失效 → EvidenceProfile 对应能力降为 `unavailable`/`degraded`），级联撤销所有
   依赖该前提的旧结果，从 validated observations 移除并标 `invalidated_by`；
5. **重置命名空间后注入**：恢复时重置用户命名空间（避免引用已失效旧变量），
   重新注入 validated observations 摘要 + 失败信息（失败 Tool、根因、已撤销的
   result_id 列表）；
6. **有限次数**：恢复重试 ≤ N（`[TODO_CALIBRATE]`），超限切 `direct_vlm_routed` 或
   `abstain`；
7. **最终答案关联 result_ids**：trace 记录 `used_result_ids`、`recovery_count`、
   `partial_tool_recovery`、级联撤销列表。

性质（§14.3）：纯工程契约，与证据状态无关 —— 规则对所有 Tool / 所有
EvidenceProfile 状态统一适用；级联撤销会**回写** EvidenceProfile，但规则本身
不依赖特定能力值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from skill3d.schemas.evidence import CAPABILITY_ORDER, EvidenceProfile
from skill3d.schemas.reconstruction import SceneState
from skill3d.tools.contract import EVIDENCE_METRIC_SCALE

# 恢复重试上限（TODO_CALIBRATE：起始参考值，必须先标定再当结论）
MAX_RECOVERY_ATTEMPTS: int = 2   # TODO_CALIBRATE

# 失败根因 → 失效的共享前提（能力名）。None = 局部失败，不动共享证据。
#
# 判据来源：
# - `ArtifactUnavailableError` 的 missing 里若含深度/位姿/点云/对象 → 几何前提失效；
# - `ConfidenceGateError` 在米制 Tool 上 → 米制尺度前提失效；
# - 域值错误（负距离/点在相机后方）是**局部**的，不级联（§14.1 明文）。
_PREMISE_BY_ARTIFACT: dict[str, str] = {
    "depth": "geometry_3d",
    "poses": "geometry_3d",
    "point_cloud": "geometry_3d",
    "scale": EVIDENCE_METRIC_SCALE,
    "objects": "object_detection",
}

# 哪些能力失效会连带撤销依赖它的结果（撤销是**传递**的：几何没了，米制也没了）
_DEPENDENTS: dict[str, tuple[str, ...]] = {
    "geometry_3d": ("geometry_3d", "world_frame", EVIDENCE_METRIC_SCALE),
    "world_frame": ("world_frame",),
    "object_detection": ("object_detection", "object_grounding", "track_consensus"),
    EVIDENCE_METRIC_SCALE: (EVIDENCE_METRIC_SCALE,),
}


@dataclass
class ValidatedObservation:
    """一条可信的既有观测（回灌给模型的先验）。"""

    result_id: str
    source_tool: str
    payload: object
    evidence_version: str = ""

    def summary(self, max_chars: int = 400) -> str:
        text = str(self.payload)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        return f"- {self.source_tool}() → {text}   [result_id={self.result_id}]"


@dataclass
class RecoveryPlan:
    """一次恢复动作的完整描述（进 trace，可审计）。"""

    attempt: int
    failed_tool: str
    failure_kind: str                      # tool_contract / confidence_gate / domain_value
    premise: Optional[str]                 # 失效的共享前提（None = 局部失败）
    validated: list[ValidatedObservation] = field(default_factory=list)
    invalidated_result_ids: list[str] = field(default_factory=list)
    invalidated_tools: list[str] = field(default_factory=list)
    feedback: str = ""
    downgraded_capabilities: dict[str, str] = field(default_factory=dict)
    exhausted: bool = False


def premise_of_failure(*, error_code: str, tool: str,
                       requires_evidence: tuple[str, ...] = (),
                       missing_artifacts: tuple[str, ...] = (),
                       unmet_evidence: tuple[str, ...] = ()) -> Optional[str]:
    """从失败事实推断失效的共享前提；`None` = 局部失败，不做级联（§14.1）。

    `unmet_evidence`：证据门给出的**逐项未达标能力**（`contract.evidence_unmet`
    的输出，可能带 `"(degraded 且不容忍)"` 之类的后缀）。**必须优先于产物判据**：
    实测踩过的坑是"证据缺口"被当成"缺产物"从而一律退成 `geometry_3d`，
    级联撤销就把 geometry/world_frame/metric_scale 全部无差别降级了。
    """
    # 证据缺口优先：取**具体**那个能力作前提（去掉括号后缀）
    for item in unmet_evidence:
        cap = str(item).split("(")[0].strip()
        if cap in _DEPENDENTS:
            return cap
    if error_code == "confidence_gate":
        if EVIDENCE_METRIC_SCALE in requires_evidence:
            return EVIDENCE_METRIC_SCALE
        # 其他局部质量门：不视为共享前提失效
        return None
    if error_code == "tool_contract":
        for art in missing_artifacts:
            p = _PREMISE_BY_ARTIFACT.get(str(art))
            if p is not None:
                return p
        return "geometry_3d"      # 缺产物但没标注具体产物 → 保守按几何前提失效
    # domain_value / answer_already_given / 其他：局部失败
    return None


def collect_validated(kernel, *, evidence_version: str = "") -> list[ValidatedObservation]:
    """从 kernel 的 calls/results 里取 `status=ok` 且未被撤销的结果（§14.1）。"""
    out: list[ValidatedObservation] = []
    for r in getattr(kernel, "tool_results", []) or []:
        if str(getattr(r, "status", "ok")) != "ok":
            continue                                  # 失败结果不进 validated
        if getattr(r, "invalidated_by", None):
            continue                                  # 已被级联撤销
        out.append(ValidatedObservation(
            result_id=str(getattr(r, "result_id", "") or getattr(r, "request_digest", "")),
            source_tool=str(getattr(r, "source_tool", "") or getattr(r, "tool", "")),
            payload=getattr(r, "payload", None) if getattr(r, "payload", None) is not None
            else getattr(r, "value", None),
            evidence_version=str(evidence_version or
                                 getattr(r, "evidence_version", "") or "")))
    return out


def cascade_invalidate(kernel, premise: str, *,
                       registry) -> tuple[list[str], list[str]]:
    """按依赖级联撤销（§14.1 第 4 条）。

    撤销范围 = 使用的能力落在 `_DEPENDENTS[premise]` 里的所有既有成功结果。
    返回 `(invalidated_result_ids, invalidated_tool_names)`；同时把被撤销结果的
    `invalidated_by` 写上（从 validated observations 移除的依据）。
    """
    affected = set(_DEPENDENTS.get(str(premise), (str(premise),)))
    ids: list[str] = []
    tools: list[str] = []
    for r in getattr(kernel, "tool_results", []) or []:
        if r.status != "ok" or r.invalidated_by:
            continue
        name = str(getattr(r, "source_tool", "") or getattr(r, "tool", ""))
        try:
            req = set(registry.requires_evidence(name))
        except Exception:  # noqa: BLE001 - 未注册名（回放/历史 trace）→ 不撤销
            continue
        if req & affected:
            rid = str(getattr(r, "result_id", "") or getattr(r, "request_digest", ""))
            if rid and rid not in ids:
                ids.append(rid)
            if name and name not in tools:
                tools.append(name)
            # 标记撤销原因（不可变 Schema：用 model_copy 写回列表元素）
            try:
                idx = kernel.tool_results.index(r)
                kernel.tool_results[idx] = r.model_copy(
                    update={"invalidated_by": sorted(set(
                        (getattr(r, "invalidated_by", None) or []) + [premise]))})
            except (ValueError, AttributeError):  # pragma: no cover - 防御
                pass
    return ids, tools


def invalidate_results(kernel, result_ids: list[str], *, reason: str) -> list[str]:
    """撤销已定位的局部结果，不推测它们与其他调用的共享依赖。"""
    wanted = set(result_ids)
    invalidated = []
    for index, result in enumerate(kernel.tool_results):
        if result.result_id in wanted and result.status == "ok" and not result.invalidated_by:
            kernel.tool_results[index] = result.model_copy(
                update={"invalidated_by": [reason]})
            invalidated.append(result.result_id)
    return invalidated


def downgrade_profile(profile: Optional[EvidenceProfile],
                      premise: str, *, confirmed_invalid: bool = False
                      ) -> tuple[Optional[EvidenceProfile], dict[str, str]]:
    """降级失效前提；M11 确证无效时直接收回对应能力。"""
    if profile is None:
        return None, {}
    changed: dict[str, str] = {}
    update: dict = {}
    reasons = dict(getattr(profile, "state_reasons", None) or {})
    for cap in _DEPENDENTS.get(str(premise), (str(premise),)):
        cur = profile.state(cap)
        if confirmed_invalid and cur != "unavailable":
            update[cap] = "unavailable"
            changed[cap] = f"{cur}→unavailable"
        elif cur == "available":
            update[cap] = "degraded"
            changed[cap] = "available→degraded"
        elif cur == "degraded":
            update[cap] = "unavailable"
            changed[cap] = "degraded→unavailable"
        else:
            continue
        # v9 §6.1：被级联撤销的能力记 `invalidated` —— 与"没跑"和"跑失败"区分开
        reasons[cap] = "invalidated"
    if update:
        update["state_reasons"] = reasons
    return (profile.model_copy(update=update) if update else profile), changed


def build_feedback(plan: RecoveryPlan, *, question_type: str,
                   scope: str, require_answer: bool = False) -> str:
    """恢复用的回灌文本（§14.1 第 5 条）：失败信息 + validated observations 摘要。"""
    lines = [
        "上一轮 program 执行失败，请改写后重新输出**一个** ```python 代码块。",
        f"失败 Tool：{plan.failed_tool}（{plan.failure_kind}）",
    ]
    if plan.premise:
        lines.append(
            f"根因：共享前提 `{plan.premise}` 失效 → 已级联撤销依赖它的既有结果"
            f"（撤销 {len(plan.invalidated_result_ids)} 条："
            f"{plan.invalidated_result_ids[:6]}）；相应证据能力已降级 "
            f"{plan.downgraded_capabilities}")
        lines.append("**不要**再使用上面被撤销的数值。")
    else:
        lines.append("根因：该 Tool 的局部错误（域值/参数），共享证据仍然有效。")
    if plan.validated:
        lines.append("\n仍然可信的既有观测（可直接复用，不必重算）：")
        lines.extend(o.summary() for o in plan.validated)
    else:
        lines.append("\n当前没有任何可信的既有观测，请从 Tool 重新开始。")
    ending = (
        "请使用仍有效的观察或原图给出最佳答案；视觉估计如实标记为 visual_estimate，"
        "不要提交 abstain。" if require_answer else
        "若确实拿不到答案，直接 `ReturnAnswer(\"abstain\")`（主榜按错计，优于编造）。")
    lines.append(
        f"\n当前 question_tool_scope={scope}，本题题型={question_type}。"
        "只允许调用 prompt 里列出的 Tool；" + ending)
    lines.append("注意：`ReturnAnswer` 之后**不得**再调用任何 Tool（会抛 "
                 "AnswerAlreadyGiven）。")
    return "\n".join(lines)


def recovery_exhausted(attempt: int, *, max_attempts: int = MAX_RECOVERY_ATTEMPTS) -> bool:
    """恢复次数是否已用尽（超限 → direct_vlm_routed / abstain，§6.4）。"""
    return int(attempt) > int(max_attempts)


def worst_capability(states: list[str]) -> str:
    """取最差状态（诊断用；与 `schemas.evidence.worst_state` 同口径）。"""
    if not states:
        return "unavailable"
    return min((str(x) for x in states),
               key=lambda x: CAPABILITY_ORDER.get(x, 0))


__all__ = [
    "MAX_RECOVERY_ATTEMPTS",
    "RecoveryPlan",
    "ValidatedObservation",
    "build_feedback",
    "cascade_invalidate",
    "collect_validated",
    "downgrade_profile",
    "invalidate_results",
    "premise_of_failure",
    "recovery_exhausted",
]
