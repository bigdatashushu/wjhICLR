"""§5.4/§5.5 EvidenceProfile 与 MetricEvidenceGateResult（v6 D3/D6）。

`EvidenceProfile` 是 v6 的**证据单一事实源**：8 项能力 × 三值
（`available > degraded > unavailable`），Tool 暴露（`requires_evidence`）与
Skill 检索（`required_evidence_signature`）都由它派生。

设计纪律（v6 §7）：
- 三值有序，`available > degraded > unavailable`；比较一律走 `CAPABILITY_ORDER`；
- `temporal` / `image_2d` 在统一 32 帧 FrameSet 存在时**恒为 available**
  （是 `direct_vlm_routed` 兜底的基础，永不被重建失败波及）；
- **单项失败只收回依赖该证据的 Tool**（§7.2），不做全局降级；
- `producer_subvalues` 保存各生产者的实算值，供审计（"这个 degraded 是怎么判出来的"）。
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

from pydantic import field_validator, model_validator

from . import Spec

# --------------------------------------------------------------- 三值能力 ----

CapabilityState = Literal["available", "degraded", "unavailable"]

# 有序序数（v6 §5.4：available > degraded > unavailable）
CAPABILITY_ORDER: dict[str, int] = {"unavailable": 0, "degraded": 1, "available": 2}

# 8 项能力的规范名（§7.1）。前 7 项为 scene 级，`object_grounding` 为题级。
SCENE_CAPABILITIES: tuple[str, ...] = (
    "geometry_3d",
    "world_frame",
    "metric_scale",
    "object_detection",
    "track_consensus",
    "temporal",
    "image_2d",
)
QUESTION_CAPABILITIES: tuple[str, ...] = ("object_grounding",)
CAPABILITIES: tuple[str, ...] = SCENE_CAPABILITIES + QUESTION_CAPABILITIES

# 恒 available 的能力（只要统一 FrameSet 存在）：§5.4 明文
ALWAYS_AVAILABLE_CAPABILITIES: frozenset[str] = frozenset({"temporal", "image_2d"})

PROFILE_VERSION: str = "evidence-profile-v6"


def capability_at_least(state: str, required: str) -> bool:
    """`state >= required`（按 available > degraded > unavailable 的有序三值）。

    **未知取值一律返回 False**：连 `required="unavailable"` 也不满足 —— 一个不认识
    的状态名说明"这条证据是什么"本身不可信，不得靠"它在序数上等于 0"混过签名检查
    （fail-closed；归纳器把能力名写错时也必须不匹配）。
    """
    if str(state) not in CAPABILITY_ORDER or str(required) not in CAPABILITY_ORDER:
        return False
    return CAPABILITY_ORDER[str(state)] >= CAPABILITY_ORDER[str(required)]


def worst_state(states: list[str]) -> CapabilityState:
    """取最差状态（用于把多个生产者信号压成一项能力的三值）。空列表 = unavailable。"""
    if not states:
        return "unavailable"
    return min((str(s) for s in states),
               key=lambda s: CAPABILITY_ORDER.get(s, 0))  # type: ignore[return-value]


class EvidenceProfile(Spec):
    """8 项能力 × 三值（§5.4）。

    `profile_version` 进 trace（§19.2）；`producer_subvalues` 让"为什么是 degraded"
    可回查，而不必重跑。
    """

    profile_version: str = PROFILE_VERSION
    # ---- scene 级能力 ----
    geometry_3d: CapabilityState
    world_frame: CapabilityState
    metric_scale: CapabilityState
    object_detection: CapabilityState
    track_consensus: CapabilityState
    temporal: CapabilityState = "available"
    image_2d: CapabilityState = "available"
    # ---- 题级能力 ----
    object_grounding: CapabilityState = "unavailable"
    # ---- 生产者子条件实算值（可审计；不参与判定）----
    producer_subvalues: dict[str, dict] = {}
    # ---- v9 §6.1：`unavailable`/`degraded` 的原因码 ----
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

    @field_validator("temporal", "image_2d")
    @classmethod
    def _always_available(cls, v: Any, info) -> Any:
        """§5.4：统一 FrameSet 存在时 temporal/image_2d 恒 available，不得被降级。"""
        if v != "available":
            raise ValueError(
                f"{info.field_name} 在统一 32 帧 FrameSet 存在时恒为 available"
                "（v6 §5.4：它俩是 direct_vlm_routed 的兜底基础，永不被重建失败波及）")
        return "available"

    def state(self, capability: str) -> CapabilityState:
        """按名取能力三值；未知能力名 raise（拼错名字不得静默当 available）。"""
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
        """当前画像 → 证据签名（Skill 正例按此签名分桶积累，§17.2）。"""
        return {c: self.state(c) for c in CAPABILITIES}


# §6.1/§6.4：`unavailable`/`degraded` 的原因码。
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
    `EvidenceProfile.metric_scale` 的三值，不影响 geometry_3d / object_detection。
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


def metric_scale_state_from_gate(gate: Optional[MetricEvidenceGateResult],
                                 *,
                                 fusion_ok: bool = False,
                                 self_consistency_ok: bool = False
                                 ) -> CapabilityState:
    """§5.5：gate 结果 → `EvidenceProfile.metric_scale` 的三值。

    - 全过 → `available`；
    - 融合成功但自洽松 → `degraded`；
    - 融合失败 / 有限值不过 → `unavailable`。
    """
    if gate is None:
        return "unavailable"
    if gate.gate_passed:
        return "available"
    if fusion_ok and not self_consistency_ok:
        return "degraded"
    return "unavailable"
