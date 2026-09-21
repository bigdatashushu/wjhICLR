"""§5.8 Skill / 治理 Schema（v6：证据签名匹配 + 5 个 Skill 族）。

v6 与 v5 的三处硬性差异：

1. **检索依据从"产物 + 最低质量分"改为"题型 + 证据签名"**（D6/D8）：
   `required_evidence_signature`（能力→最低可接受状态）取代
   `requires_artifacts` + `minimum_quality` + `metric_scale_required`；
2. **允许在特定证据签名下分开积累**（§17.2）：同一题型的 Skill 可以在
   `metric_scale=unavailable` 签名与"全 available"签名下各有一套，
   **互不污染**（这是 v6 证据机制的核心收益之一）；
3. **米制 Skill 双重 fail-closed**（§13.6）：必须声明
   `requires_metric_evidence=True` + `applicable_gate_version`，
   检索（M7）与执行（M10）两次校验当前 gate 通过且版本匹配。

离线治理模型改为 DeepSeek-V4.1-Flash（§3.4）：`DeepSeekGovernanceDecision`。
"""

from typing import Literal, Optional

from . import Spec
from .evidence import CAPABILITIES

SkillState = Literal["draft", "shadow", "canary", "promoted", "quarantined"]

# 5 个 Skill 族（§17.2 基线分组）
SKILL_FAMILIES: tuple[str, ...] = (
    "counting", "metric", "relative_geometry", "route", "appearance",
)
SkillFamily = Literal["counting", "metric", "relative_geometry", "route", "appearance"]

# 族 → 适用的规范题型（族内个体 Skill 只能声明该族的子集）
FAMILY_QUESTION_TYPES: dict[str, tuple[str, ...]] = {
    "counting": ("object_counting",),
    "metric": ("object_abs_distance", "object_size_estimation", "room_size_estimation"),
    "relative_geometry": ("object_rel_distance", "object_rel_direction"),
    "route": ("route_planning",),
    "appearance": ("obj_appearance_order",),
}

SkillSource = Literal["real", "mock_interface", "mock_replay", "mock_light"]


class SkillSpec(Spec):
    """题型级程序合成模板（非可执行工具/非答案，硬约束 15；v6 §5.8）。

    `required_evidence_signature` 的键必须是 `schemas.evidence.CAPABILITIES` 的成员，
    值必须是三值之一。**归纳器不得把该条件泛化/省略掉**（§17.3）—— 例如不得把
    "metric_scale>=degraded" 改写成无条件 Skill（那会让米制 Skill 在尺度不可用的
    场景里被检索出来并给出伪米制答案）。
    """

    skill_id: str
    version: str                          # MAJOR.MINOR.PATCH
    applicable_question_types: list[str]
    # 能力 → 最低可接受状态（"available" | "degraded" | "unavailable"）
    required_evidence_signature: dict[str, str] = {}
    requires_metric_evidence: bool = False
    applicable_gate_version: Optional[str] = None   # 米制 Skill 必填（§13.6）
    skill_family: SkillFamily
    source: SkillSource = "real"

    # ---- 模板内容（沿用 v4/v5 字段；进 prompt 的只有模板本身）----
    description: str = ""
    call_graph_template: str = ""
    supported_coordinate_frames: list[str] = ["world"]
    validation_assertions: list[str] = []

    # ---- v5 只读兼容（过渡期；新 Skill 一律用证据签名）----
    @property
    def semver(self) -> str:
        """v5 字段名别名（版本号语义一致）。"""
        return self.version

    @property
    def task_type(self) -> str:
        """v5 字段名别名：单一题型时返回它，否则返回族名（多题型 Skill）。"""
        qts = list(self.applicable_question_types)
        return qts[0] if len(qts) == 1 else self.skill_family

    def __init__(self, **data):
        """构造期校验（fail-closed）：签名键/值合法、米制声明完整、族与题型一致。"""
        sig = data.get("required_evidence_signature") or {}
        bad_keys = sorted(set(sig) - set(CAPABILITIES))
        if bad_keys:
            raise ValueError(
                f"required_evidence_signature 含未知能力 {bad_keys}；"
                f"词汇表见 schemas.evidence.CAPABILITIES")
        bad_vals = sorted({str(v) for v in sig.values()} - {"available", "degraded",
                                                            "unavailable"})
        if bad_vals:
            raise ValueError(
                f"required_evidence_signature 含非法状态值 {bad_vals}；"
                "只允许 available/degraded/unavailable")
        if data.get("requires_metric_evidence"):
            if "metric_scale" not in sig:
                raise ValueError(
                    "requires_metric_evidence=True 但 required_evidence_signature 未声明 "
                    "metric_scale（§13.6：归纳器不得把米制条件省略掉）")
            if not data.get("applicable_gate_version"):
                raise ValueError(
                    "米制 Skill 必须声明 applicable_gate_version（§13.6 双重 fail-closed）")
        fam = data.get("skill_family")
        if fam in FAMILY_QUESTION_TYPES:
            allowed = set(FAMILY_QUESTION_TYPES[fam])
            got = set(data.get("applicable_question_types") or [])
            extra = sorted(got - allowed)
            if extra:
                raise ValueError(
                    f"skill_family={fam} 不覆盖题型 {extra}；"
                    f"该族只允许 {sorted(allowed)}（§17.2 族基线）")
        super().__init__(**data)


class SkillCandidate(Spec):
    candidate_id: str
    root_candidate_id: str
    revision_id: str
    parent_version: Optional[str]
    candidate_type: Literal["memory", "skill"]
    spec_content: str
    status: Literal["draft", "testing", "promoted", "rejected", "quarantined"]
    # 离线强模型产的 patch（v6：DeepSeek-V4.1-Flash）
    offline_patch_ref: Optional[str] = None
    induction_trace_refs: list[str] = []
    created_by: Literal["offline_induction", "offline_revision", "human"]
    created_at: str
    # v5 字段名兼容（读旧数据）
    @property
    def gpt6_patch_ref(self) -> Optional[str]:
        return self.offline_patch_ref


class DeepSeekGovernanceDecision(Spec):
    """离线治理裁决文本（§3.3/§3.4）。

    **纪律**：离线模型的输出**不能**直接修改 active Skill/Memory，也不能决定
    promote/reject —— promote 由确定性门 + 预注册规则决定（§3.3）。本 Schema 只承载
    "离线模型说了什么"，供审计；它不构成任何准入决定。
    """

    decision_id: str
    candidate_id: str
    review_summary: str
    semantic_risk: str
    generalization_notes: str
    offline_model: str = "DeepSeek-V4.1-Flash"
    provider: str = "deepseek"
    model_id: str = "deepseek-flash"
    prompt_version: str = ""
    # 显式标注：本裁决**不是**准入决定（确定性门才有权 promote）
    advisory_only: Literal[True] = True
    timestamp: str


# v5 名称别名（过渡期只读；新代码一律用 DeepSeekGovernanceDecision）
SkillGovernanceDecision = DeepSeekGovernanceDecision


class RetrievedSkill(Spec):
    """M7 Skill 检索输出（§4 M7 字段 6 + v6 证据签名匹配）。"""

    skill_id: str = ""
    skill_version: str = ""
    score: float = 0.0
    hard_filter_passed: bool = False
    # v6：命中的证据签名（= 检索时的 EvidenceProfile 快照的投影），
    # 让"这条 Skill 是在哪种证据状态下被选中的"可审计（§17.2 分开积累）
    matched_evidence_signature: dict[str, str] = {}
    # 米制 Skill 的 gate 版本匹配结果（§13.6）
    gate_version_matched: Optional[bool] = None

    @property
    def skill_semver(self) -> str:
        """v5 字段名别名。"""
        return self.skill_version


__all__ = [
    "DeepSeekGovernanceDecision",
    "FAMILY_QUESTION_TYPES",
    "RetrievedSkill",
    "SKILL_FAMILIES",
    "SkillCandidate",
    "SkillFamily",
    "SkillGovernanceDecision",
    "SkillSource",
    "SkillSpec",
    "SkillState",
]
