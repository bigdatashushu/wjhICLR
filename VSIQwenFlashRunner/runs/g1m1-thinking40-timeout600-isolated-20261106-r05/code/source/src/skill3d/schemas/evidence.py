"""公共证据合约：几何与尺度两项二值能力，生产者细节只作诊断。

对象候选的技术有效性由工具按实参检查；题目指代由在线模型消歧。
公共画像不声明检测完整性、指代正确性或世界坐标约定可靠性。
"""

from __future__ import annotations

from typing import Literal, Mapping, Optional

from pydantic import field_validator, model_validator

from . import Spec

CapabilityState = Literal["available", "unavailable"]
CAPABILITY_ORDER: dict[str, int] = {"unavailable": 0, "available": 1}
CAPABILITIES: tuple[str, ...] = ("geometry_3d", "metric_scale")
SkillEvidenceState = Literal["g0m0", "g1m0", "g1m1"]
SKILL_EVIDENCE_STATES: tuple[SkillEvidenceState, ...] = ("g0m0", "g1m0", "g1m1")

PROFILE_VERSION: str = "evidence-profile-v11.6-two-capabilities"


def capability_at_least(state: str, required: str) -> bool:
    """`state >= required`（按 available > unavailable 排序）。

    **未知取值一律返回 False**：连 `required="unavailable"` 也不满足 —— 一个不认识
    的状态名说明"这条证据是什么"本身不可信，不得靠"它在序数上等于 0"混过签名检查
    （fail-closed；归纳器把能力名写错时也必须不匹配）。
    """
    if str(state) not in CAPABILITY_ORDER or str(required) not in CAPABILITY_ORDER:
        return False
    return CAPABILITY_ORDER[str(state)] >= CAPABILITY_ORDER[str(required)]


def worst_state(states: list[str]) -> CapabilityState:
    """取最差二值状态。空列表或未知状态 = unavailable。"""
    if not states:
        return "unavailable"
    return "available" if all(s == "available" for s in states) else "unavailable"


class EvidenceProfile(Spec):
    """两项能力；尺度可用必须以几何可用为前提。"""

    profile_version: str = PROFILE_VERSION
    # ---- scene 级能力 ----
    geometry_3d: CapabilityState
    metric_scale: CapabilityState
    # ---- 生产者子条件实算值（可审计；不参与判定）----
    producer_subvalues: dict[str, dict] = {}
    # ---- unavailable 的原因码 ----
    # 至少区分 not_run / producer_failed / invalidated / unsupported。
    # 缺省为空 = 原因未登记（不猜）；有值时受词表校验，避免自由字符串漂移。
    state_reasons: dict[str, str] = {}

    @field_validator("state_reasons")
    @classmethod
    def _validate_state_reasons(cls, v: dict[str, str]) -> dict[str, str]:
        for cap, reason in (v or {}).items():
            if cap not in CAPABILITIES:
                raise ValueError(f"state_reasons 含未知能力 {cap!r}")
            if reason not in UNAVAILABLE_REASON_CODES:
                raise ValueError(
                    f"state_reasons[{cap!r}]={reason!r} 不在 §6.1 词表 "
                    f"{sorted(UNAVAILABLE_REASON_CODES)} 中")
        return v

    @model_validator(mode="after")
    def _metric_requires_geometry(self) -> "EvidenceProfile":
        if self.metric_scale == "available" and self.geometry_3d != "available":
            raise ValueError("metric_scale=available requires geometry_3d=available")
        return self

    def state(self, capability: str) -> CapabilityState:
        """按名取能力；未知能力名 raise（拼错名字不得静默当 available）。"""
        if capability not in CAPABILITIES:
            raise KeyError(
                f"未知证据能力 {capability!r}；词汇表见 schemas.evidence.CAPABILITIES")
        return getattr(self, capability)

    def satisfies(self, signature: Mapping[str, str]) -> bool:
        """`signature` 的每项要求是否都达标（能力→最低可接受状态）。"""
        try:
            return all(capability_at_least(self.state(c), r)
                       for c, r in (signature or {}).items())
        except KeyError:
            # 签名里出现未知能力名 = 归纳器写错 → 不匹配（fail-closed，不静默放行）
            return False

    def unmet(self, signature: Mapping[str, str]) -> list[str]:
        """未达标的能力名列表（诊断用；顺序稳定）。"""
        out: list[str] = []
        for c, req in (signature or {}).items():
            if c not in CAPABILITIES:
                out.append(f"{c}(未知能力)")
                continue
            if not capability_at_least(self.state(c), req):
                out.append(f"{c}(需{req}，实为{self.state(c)})")
        return out

    def as_signature(self) -> dict[str, str]:
        """当前两项状态快照。"""
        return {c: self.state(c) for c in CAPABILITIES}


def skill_evidence_state(
    profile: EvidenceProfile | Mapping[str, str],
) -> SkillEvidenceState:
    """精确编码合法画像；缺画像、未知值和 g0m1 必须拒绝。"""
    values = profile.model_dump() if isinstance(profile, EvidenceProfile) else profile
    validated = EvidenceProfile.model_validate(values)
    if validated.geometry_3d == "unavailable":
        return "g0m0"
    return "g1m1" if validated.metric_scale == "available" else "g1m0"


# §6.1/§6.4：`unavailable` 的原因码。
#
# §6.4 给出四类：`not_run`（未运行）/ `producer_failed`（生产者失败）/
# `invalidated`（被级联撤销）/ `unsupported`（成功执行但无匹配目标）。
#
# **决策记录（用户 2026-09-27）**：规范的四类里**没有**"运行成功但**质量门未过**"
# 这一类 —— 例如 M4 主门未过（质量真算了、生产者也正常，只是数值不达标）、world
# frame 退化、尺度自洽低于阈值。此前这类只能留空（= 原因未登记），于是审计看到的
# 是"没记原因"，而不是"跑了但不达标"。**用户裁定：扩展词表**，新增
# `quality_gate_not_passed`（第五值），与四类并列。它不是对规范的替换，而是补足
# 规范未覆盖的一类；`producer_failed` 仍只表示"生产者自己出错"，两者不得互替。
UNAVAILABLE_REASON_CODES: frozenset[str] = frozenset({
    "not_run", "producer_failed", "invalidated", "unsupported",
    "quality_gate_not_passed",
})

# §8.2 门三态：非米制工具为 not_applicable 且 gate_passed=None
MetricGateStatus = Literal["pass", "fail", "not_applicable"]


class MetricEvidenceGateResult(Spec):
    """§5.5 / §13.1：度量证据门（确定性组件，非 VLM 裁决）。

    它是 **metric evidence 的分项能力门，不是全局 gate** —— 只决定
    `EvidenceProfile.metric_scale` 的二值，不影响 geometry_3d。
    """

    # v9 §8.2：`status` 为 pass／fail／not_applicable；非米制工具 not_applicable
    # 且 `gate_passed=None`，其余与布尔判定一致。
    gate_passed: Optional[bool]
    gate_version: str
    status: "MetricGateStatus" = ""
    sub_results: dict[str, bool] = {}
    values: dict[str, float] = {}
    missing_subconditions: list[str] = []
    invalidated_by: list[str] = []

    @model_validator(mode="after")
    def _consistency(self) -> "MetricEvidenceGateResult":
        """`gate_passed`／`status`／`sub_results` 三者必须自洽，否则伪造门通过。

        - `not_applicable` ⟺ `gate_passed is None`（§8.2：不得用默认 False 伪装失败）；
        - 其余情形 `status` 由 `gate_passed` 派生，且必须与 `sub_results` 全真一致；
        - 空 `sub_results` 只在"未通过/不适用"时允许（例如"根本没跑融合"）。
        """
        if not self.status:
            self.status = ("not_applicable" if self.gate_passed is None
                           else ("pass" if self.gate_passed else "fail"))
        if (self.status == "not_applicable") != (self.gate_passed is None):
            raise ValueError(
                f"MetricEvidenceGateResult.status={self.status!r} 与 "
                f"gate_passed={self.gate_passed!r} 不一致（§8.2：not_applicable 对应 None）")
        if self.gate_passed is None:
            return self                      # 不适用 → 不参与通过/失败判定
        if self.sub_results:
            all_true = all(bool(v) for v in self.sub_results.values())
            if bool(self.gate_passed) != all_true:
                raise ValueError(
                    f"MetricEvidenceGateResult 自相矛盾：gate_passed={self.gate_passed}"
                    f" 但 sub_results={self.sub_results}（§13.1：6 项全真才通过）")
        elif self.gate_passed:
            raise ValueError(
                "MetricEvidenceGateResult：无 sub_results 却声明 gate_passed=True"
                "（§13.1：必须逐项留证）")
        return self


# --------------------------------------------------- 子条件名（§13.1）-------

GATE_SUB_SCENE_ROUTE = "scene_route_full_3d"
GATE_SUB_M4_MAIN_GATE = "m4_main_gate_passed"
GATE_SUB_FUSION_SUCCESS = "scale_fusion_success"
GATE_SUB_SCALE_SELF_CONSISTENCY = "scale_self_consistency_ok"
GATE_SUB_FINITE_INPUTS = "inputs_finite"
GATE_SUB_METRIC_QUESTION = "question_type_is_metric"

GATE_SUBCONDITIONS: tuple[str, ...] = (
    GATE_SUB_SCENE_ROUTE,
    GATE_SUB_M4_MAIN_GATE,
    GATE_SUB_FUSION_SUCCESS,
    GATE_SUB_SCALE_SELF_CONSISTENCY,
    GATE_SUB_FINITE_INPUTS,
    GATE_SUB_METRIC_QUESTION,
)

GATE_VERSION: str = "metric-evidence-gate-v6"


def metric_scale_state_from_gate(
    gate: Optional[MetricEvidenceGateResult],
) -> CapabilityState:
    """仅门的前五项决定尺度能力，题型授权独立判定。"""
    if gate is not None and all(
        gate.sub_results.get(k, False) for k in GATE_SUBCONDITIONS[:5]
    ):
        return "available"
    return "unavailable"
