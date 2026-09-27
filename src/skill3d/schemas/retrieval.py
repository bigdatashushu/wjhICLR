"""§13.5/§13.6 检索与交付记录 Schema（v9）。

规范原文（§13.6）：

    「每次检索记录**规范题型、evidence_version、候选及过滤原因、排序分数、
    选中版本、实际交付版本与正文 hash、配置版本**。区分"检索选中但未送达模型"与
    "已交付"。」
    「记录 `retrieved_skill_versions`、`delivered_skill_versions`、
    `declared_selected_skill_versions`、可观察的程序使用线索，以及每轮短方法摘要。
    模型自称选择要与程序和观察交叉检查，不能当作方法成功或因果贡献的充分证明。」

规范原文（§13.5）：

    「顺序固定为：Adapter 得到规范题型 → 精确过滤该题型分区 → 检查整体
    hard_requirements → 在分区内按适用性、当前证据偏好与方法相关性排序 → 选取并
    交付完整方法正文。禁止全库 top-k 后才丢弃其他题型。」
    「证据更新后可在同一快照中重检索，更新实际交付记录；不在 episode 中发布新库。」

本模块只承载**记录**：四个状态（produced → 检索选中 → 交付 → 模型自称）分别有
自己的字段，任何一格都不许拿另一格顶替。

- `produced`：`n_skills_offered` —— 本次交给检索器的方法总数（本快照的候选池）；
- `retrieved`：`retrieved_skill_versions` —— 硬条件通过且排序进入 top-k（= 检索选中，
  §13.6 的"检索选中但未送达模型"指的就是它）；
- `delivered`：`delivered_skill_versions` —— 进了**实际发出**的模型请求；
- `declared`：`declared_selected_skill_versions` —— 模型在程序里**自称**使用，
  它只是线索，不是因果贡献的证明。

自洽性由 validator 强制（fail-closed）：delivered ⊆ retrieved ⊆ eligible，
`delivered=True` ⟺ `delivery_reason="delivered"`，原因码必须在词表内。
"""

from __future__ import annotations

from typing import Optional

from pydantic import field_validator

from . import Spec

RETRIEVAL_RECORD_SCHEMA_VERSION = "1.0"

# 检索触发原因（§13.5：初次检索 / 证据更新后在同一快照内重检索）
RETRIEVAL_TRIGGERS: tuple[str, ...] = ("initial", "evidence_update")

# 候选的硬过滤原因码（§13.5 的固定顺序：题型分区 → 硬条件 → 排序 → 选取）
CANDIDATE_REASON_CODES: frozenset[str] = frozenset({
    "hit",                         # 硬条件全过（是否进 top-k 由 selected 表达）
    "question_type_unknown",       # 题型不可知 → 不检索任何 Skill（fail-closed）
    "question_type_mismatch",      # §13.5 精确过滤题型分区
    "evidence_signature_unmet",    # EvidenceProfile 不满足 Skill 的证据签名
    "metric_gate_not_passed",      # 米制 Skill：当前米制证据门未过
    "metric_gate_version_mismatch",  # 米制 Skill：gate 版本与新声明不一致
    "gate_result_missing",         # 米制 Skill：连 gate 结果都没有（无法证明可用）
    "state_not_active",            # 未 promoted/consolidated 的候选不得生效
    "not_selected_top_k",          # 硬条件通过但排序未进 top-k
})

# 交付状态（§13.6；与 skills.delivery.DELIVERY_REASON_CODES 同源，此处做记录侧校验）
# `not_selected`：没被选中（硬条件未过，或排序未进 top-k）——没选中就谈不上交付。
DELIVERY_STATES: frozenset[str] = frozenset({
    "delivered", "not_selected", "context_cap_exceeded",
    "no_model_request", "request_failed",
})

# 交付渠道
DELIVERY_CHANNELS: frozenset[str] = frozenset({
    "model_request", "request_failed", "not_sent",
})


class SkillCandidateRecord(Spec):
    """单条候选的检索判定 + 交付状态（"候选及过滤原因、排序分数、选中版本"）。"""

    skill_id: str
    version: str
    skill_version: str = ""
    # 该 Skill 声明的适用题型（规范化后）；题型分区过滤的依据
    canonical_question_type: str = ""
    hard_filter_passed: bool = False
    reason_code: str = "hit"
    reason: str = ""                     # 人类可读判定文本（原 RetrievalDecision.reason）
    score: Optional[float] = None        # 排序分数（未参与排序时为 None）
    rank: Optional[int] = None           # 1-based 名次（硬条件未过时为空）
    selected: bool = False               # §13.6"检索选中"（进 top-k）
    content_sha256: str = ""             # 方法正文 hash（交付身份）
    content_chars: int = 0
    matched_evidence_signature: dict[str, str] = {}
    gate_version_matched: Optional[bool] = None
    delivered: bool = False              # §13.6"已交付"（进了实际发出的请求）
    # 未交付原因；缺省 = "没被选中"（默认构造的行都是未选中的候选）
    delivery_reason: str = "not_selected"

    @field_validator("reason_code")
    @classmethod
    def _known_reason(cls, v: str) -> str:
        if v not in CANDIDATE_REASON_CODES:
            raise ValueError(f"未知候选原因码 {v!r}；词表={sorted(CANDIDATE_REASON_CODES)}")
        return v

    @field_validator("delivery_reason")
    @classmethod
    def _known_delivery_reason(cls, v: str) -> str:
        if v not in DELIVERY_STATES:
            raise ValueError(f"未知交付状态 {v!r}；词表={sorted(DELIVERY_STATES)}")
        return v

    @field_validator("selected")
    @classmethod
    def _selected_needs_hit(cls, v: bool, info) -> bool:
        if v and not info.data.get("hard_filter_passed", False):
            raise ValueError("selected=True 但 hard_filter_passed=False（选中必须先过硬条件）")
        return v

    @field_validator("delivered")
    @classmethod
    def _delivered_needs_selected(cls, v: bool, info) -> bool:
        if v and not info.data.get("selected", False):
            raise ValueError("delivered=True 但 selected=False（未选中谈不上交付）")
        return v

    def model_post_init(self, __context) -> None:  # noqa: D105 - pydantic v2 hook
        # delivered 与 delivery_reason 必须同义（§13.6：不能一边说交付了一边写别的状态）
        if self.delivered and self.delivery_reason != "delivered":
            raise ValueError(
                f"delivered=True 但 delivery_reason={self.delivery_reason!r}"
                "（「已交付」必须同源：§13.6 区分「检索选中」与「已交付」）")
        if not self.delivered and self.delivery_reason == "delivered":
            raise ValueError("delivery_reason='delivered' 但 delivered=False")


class SkillRetrievalRecord(Spec):
    """一次检索的完整记录（§13.6 点名字段 + §13.5 的触发与配置版本）。"""

    schema_version: str = RETRIEVAL_RECORD_SCHEMA_VERSION
    retrieval_index: int = 1                  # 本 episode 的第几次检索（1-based）
    trigger: str = "initial"                  # initial | evidence_update
    canonical_question_type: str = ""
    question_type_raw: str = ""
    question_type_known: bool = True
    # §13.5 的过滤顺序是固定的（先题型分区，再硬条件），记录策略名以便核对顺序未被改
    partition_policy: str = "question_type_partition_then_hard_requirements"
    evidence_version: str = ""
    # 证据三值快照（§13.6 只要求 `evidence_version`，但 `profile_version` 是**合同
    # 版本**：同一版本号下不同 episode 的证据状态可以不同，甚至同一 episode 里
    # 级联降级后版本号也不变。因此把能力三值一起落盘 —— 重检索记录之间可直接比对
    # "证据是否真的变了"，不必靠版本号猜。
    evidence_states: dict[str, str] = {}
    evidence_state_reasons: dict[str, str] = {}
    # §13.6"配置版本"：人类标签 + 内容摘要（摘要可复算，标签不可抵赖）
    config_version: str = ""
    config_sha256: str = ""
    config_source: str = ""                   # config | default
    policy: dict = {}                         # top_k/rerank/candidates/rank_weights/method_context_max_chars
    active_snapshot_ref: str = ""
    active_snapshot_manifest_sha256: str = ""
    # ---- produced → retrieved → delivered → declared（四态分离）----
    n_skills_offered: int = 0
    candidates: list[SkillCandidateRecord] = []
    eligible_skill_versions: list[str] = []       # 硬条件通过（top-k 之前）
    retrieved_skill_versions: list[str] = []      # top-k 选中（= 检索结果）
    delivered_skill_versions: list[str] = []      # 实际送达模型
    declared_selected_skill_versions: list[str] = []   # 模型自称（程序文本线索）
    # 程序对象携带的上下文口径（历史字段 `program.skill_semver_used` 的语义：
    # 交付集，**不是**模型自称）—— 两者分开记，禁止互替。
    program_context_skill_versions: list[str] = []
    # ---- 交付明细 ----
    delivery_channel: str = "not_sent"
    delivery_note: str = ""
    delivered_content_sha256: dict[str, str] = {}
    dropped_for_context: list[dict] = []          # §13.5 因上下文上限整条丢弃
    method_summaries: list[dict] = []             # §13.6 每轮短方法摘要
    usage_clues: list[dict] = []                  # §13.6 可观察的程序使用线索

    @field_validator("trigger")
    @classmethod
    def _known_trigger(cls, v: str) -> str:
        if v not in RETRIEVAL_TRIGGERS:
            raise ValueError(f"未知检索触发原因 {v!r}；词表={list(RETRIEVAL_TRIGGERS)}")
        return v

    @field_validator("delivery_channel")
    @classmethod
    def _known_channel(cls, v: str) -> str:
        if v not in DELIVERY_CHANNELS:
            raise ValueError(f"未知交付渠道 {v!r}；词表={sorted(DELIVERY_CHANNELS)}")
        return v

    def model_post_init(self, __context) -> None:  # noqa: D105
        eligible = set(self.eligible_skill_versions)
        retrieved = set(self.retrieved_skill_versions)
        delivered = set(self.delivered_skill_versions)
        extra = retrieved - eligible
        if extra:
            raise ValueError(f"retrieved 不在 eligible 内: {sorted(extra)}")
        extra = delivered - retrieved
        if extra:
            raise ValueError(f"delivered 不在 retrieved 内: {sorted(extra)}")
        missing = delivered - set(self.delivered_content_sha256)
        if missing:
            raise ValueError(
                f"delivered 缺少正文 hash: {sorted(missing)}"
                "（§13.6：实际交付版本必须带正文 hash）")

    def add_delivery(self, plan, *, round_index: int,
                     skills_by_key: Optional[dict] = None,
                     usage_clues: Optional[list] = None,
                     short_summary_chars: int = 120) -> None:
        """把一次合成的交付事实并入本记录（§13.5"更新实际交付记录"）。

        `plan` 是 `skills.delivery.SkillDeliveryPlan`。只有
        `channel == "model_request"` 的条目算"已交付"；因上下文上限被丢弃的条目记
        `context_cap_exceeded`，**不**记成交付（§13.5 禁止"截掉后仍称完整交付"）。
        """
        from skill3d.skills.delivery import (
            DELIVERY_CHANNEL_MODEL_REQUEST,
            DELIVERY_CHANNEL_REQUEST_FAILED,
        )

        self.delivery_channel = str(plan.channel)
        self.delivery_note = str(plan.channel_note or "")
        delivered = list(plan.delivered_skill_versions)
        if delivered:
            self.delivered_skill_versions = sorted(
                set(self.delivered_skill_versions) | set(delivered))
            for key, digest in plan.delivered_content_sha256.items():
                self.delivered_content_sha256[key] = digest
        dropped = {d.skill_version for d in plan.dropped}
        selected_ids = set(plan.selected_skill_versions)
        if plan.dropped:
            self.dropped_for_context = [d.to_dict() for d in plan.dropped]
        for row in self.candidates:
            if not row.selected:
                continue
            if row.skill_version in delivered:
                row.delivered = True
                row.delivery_reason = "delivered"
            elif row.skill_version in dropped:
                row.delivery_reason = "context_cap_exceeded"
            elif plan.channel == DELIVERY_CHANNEL_REQUEST_FAILED:
                row.delivery_reason = "request_failed"
            elif plan.channel == DELIVERY_CHANNEL_MODEL_REQUEST:
                # 进了发出的请求，却既不在交付条目里也不在丢弃清单里 = 计划与实际
                # 不一致（不该发生）。不猜：按"未送达"记，避免虚报交付。
                row.delivery_reason = "no_model_request"
            else:
                row.delivery_reason = "no_model_request"
        self.method_summaries.append({
            "round": int(round_index),
            "channel": str(plan.channel),
            "retrieved_skill_versions": list(self.retrieved_skill_versions),
            "delivered_skill_versions": list(delivered),
            "dropped_skill_versions": sorted(dropped),
            "max_chars": int(plan.max_chars),
            "used_chars": int(plan.used_chars),
            "short": [
                {"skill_version": e.skill_version,
                 "summary": short_method_summary(
                     (skills_by_key or {}).get(e.skill_version),
                     max_chars=short_summary_chars)}
                for e in plan.entries if e.skill_version in selected_ids
            ],
        })
        if usage_clues:
            self.usage_clues.extend(dict(c) for c in usage_clues)


def short_method_summary(skill, *, max_chars: int = 120) -> str:
    """§13.6"每轮短方法摘要"：只截**摘要字段**，正文交付仍必须完整。

    注意区分：这里是给 trace 读的短摘要（可以有损）；交付给模型的正文永远完整
    （§13.5 禁止截断正文后仍称"完整 Skill 已交付"）。
    """
    text = " ".join(str(getattr(skill, "description", "") or "").split())
    return text[:max_chars]


__all__ = [
    "CANDIDATE_REASON_CODES",
    "DELIVERY_CHANNELS",
    "DELIVERY_STATES",
    "RETRIEVAL_RECORD_SCHEMA_VERSION",
    "RETRIEVAL_TRIGGERS",
    "SkillCandidateRecord",
    "SkillRetrievalRecord",
    "short_method_summary",
]
