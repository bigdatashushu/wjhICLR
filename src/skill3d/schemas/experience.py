"""v10 §6.1/§6.3 经验账本 Schema（ExperienceEvent / ExperienceBundle）。

规范原文（§6.2）——只有同时满足以下条件的 learning episode，才能作为某个 Skill 的
**主要修订经验**：

1. episode 使用父快照运行；
2. Skill 被正常检索；
3. Skill 正文真正进入模型请求；
4. 至少存在可观察的程序使用线索；
5. trace、结果和版本身份完整；
6. 不属于 inner/final；
7. 没有 Schema 损坏、缺失映射或无法确认快照身份。

    「`delivered_not_used` 只能用于改进描述和检索适用性；`not_retrieved` 只能用于识别
    覆盖缺口，**不能冒充该 Skill 的执行经验**。」

规范原文（§6.3）：

    「经验包必须按 `skill_id@version` 构建，禁止把多个题型或多个父 Skill 的经验混入
    同一个修订请求。」

因此本模块只承载**事实**：一条经验事件登记一位 (episode, skill 版本) 的检索/交付/
使用状态与资格判定；经验包是按父版本 bucketed 的不可变投影。资格判定本身是确定性
函数（`skill3d.evolution.experience`），Schema 侧只做词汇表与自洽性校验。
"""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from . import Spec

EXPERIENCE_EVENT_SCHEMA_VERSION = "1.0"
EXPERIENCE_BUNDLE_SCHEMA_VERSION = "1.0"

# §6.1 `retrieval_state` 四值：未检索 → 检索到但未交付 → 已交付但未见使用 → 使用了。
RETRIEVAL_STATES: tuple[str, ...] = (
    "not_retrieved", "retrieved_not_delivered", "delivered_not_used", "usage_supported",
)

# §6.1 `split`：首期只用 learning 与 inner_validation（outer/final 不参与）。
EXPERIENCE_SPLITS: tuple[str, ...] = ("learning", "inner_validation")

# 排除原因码（§6.2 七条的机器可读投影 + 结构性原因）。
#
# 每个码必须对应 §6.2 的一条或"身份/结构不完整"这一显式事实；不得用 `other` 兜底
# 掩盖未分类原因（那会让"为什么这批经验不能用于归纳"无从回答）。
EXCLUSION_REASON_CODES: frozenset[str] = frozenset({
    "snapshot_identity_unconfirmed",   # §6.2-7：无法确认快照身份
    "not_parent_snapshot",             # §6.2-1：episode 未使用父快照运行
    "split_not_learning",              # §6.2-6：不属于 learning（inner/final 不得进归纳）
    "split_unknown",                   # §6.2-6：split 无法确认（不猜成 learning）
    "retrieval_record_missing",        # §6.2-5：没有检索记录，无法回答"是否被检索"
    "skill_not_retrieved",             # §6.2-2：Skill 未被正常检索
    "skill_not_delivered",             # §6.2-3：正文未真正进入模型请求
    "delivered_but_no_usage_clue",     # §6.2-4：无任何可观察的程序使用线索
    "result_identity_missing",         # §6.2-5：结果/版本身份不完整
    "schema_corrupted",                # §6.2-7：Schema 损坏
    "skill_identity_unmapped",         # §6.2-7：检索到的版本在快照里找不到映射
    "campaign_generation_mismatch",    # 经验不属于本代（防"同一批旧 trace 计为新一代"）
})

# 行为摘要的受控键（§6.3 `behavior_summary`）：只允许"可观察的程序行为"，
# 不允许任何答案、GT 或 sample id。
BEHAVIOR_SUMMARY_KEYS: tuple[str, ...] = (
    "n_eligible", "n_usage_supported", "n_delivered_not_used",
    "n_retrieved_not_delivered", "n_answer_correct", "n_answer_wrong",
    "n_run_error", "n_abstain", "n_partial_tool_recovery",
    "n_declared_in_program", "n_template_tool_overlap",
    # `answer_correct_rate` 的分母是**有判分布尔量**的题（`n_answer_graded`）；
    # MRA 题型（object_counting 等）没有对错布尔量，另记 `n_mra_graded` / `mra_mean`。
    "n_answer_graded", "answer_correct_rate", "n_mra_graded", "mra_mean",
    "run_error_rate",
)

# 失败摘要的受控键（§6.3 `failure_summary`）：失败**类别**计数，不含题面/答案。
FAILURE_SUMMARY_KEYS: tuple[str, ...] = (
    "by_category", "by_failure_code", "n_without_failure",
)


class ExperienceEvent(Spec):
    """§6.1：一个 episode 对**一条候选 Skill** 产生一个经验事件。

    `experience_id` 是 `(campaign, generation, episode, skill_version)` 的稳定函数，
    不是随机数 —— 同一次真实运行重复构建必须得到同一 id（否则"是否重复消费经验"
    无法判定）。
    """

    schema_version: str = EXPERIENCE_EVENT_SCHEMA_VERSION
    experience_id: str
    campaign_id: str
    generation: int
    episode_id: str
    scene_id: str
    split: Literal["learning", "inner_validation"]
    # 原始层名（本仓库 split 的 `induction` / `inner_validation` ...），
    # 与规范层的 `split` 分开记：`split` 是 v10 语义，`source_split` 是分层事实。
    source_split: str = ""
    snapshot_id: str
    skill_id: str
    skill_version: str
    retrieval_state: Literal[
        "not_retrieved", "retrieved_not_delivered", "delivered_not_used", "usage_supported"
    ]
    retrieval_record_ref: str = ""
    delivery_content_sha256: str | None = None
    usage_clue_refs: list[str] = []
    outcome_ref: str = ""
    failure_categories: list[str] = []
    answer_correct: bool | None = None
    label_access: bool = False
    eligible_for_induction: bool = False
    exclusion_reasons: list[str] = []

    @field_validator("exclusion_reasons")
    @classmethod
    def _known_exclusions(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - EXCLUSION_REASON_CODES)
        if unknown:
            raise ValueError(
                f"未知排除原因码 {unknown}；词表={sorted(EXCLUSION_REASON_CODES)}")
        return v

    def model_post_init(self, __context) -> None:  # noqa: D105 - pydantic v2 hook
        # 资格与排除原因必须同义（§6.2）：eligible=True 时不得同时带排除原因，
        # eligible=False 时必须给出至少一条原因 —— 否则"为什么不能用于归纳"没有答案。
        if self.eligible_for_induction and self.exclusion_reasons:
            raise ValueError(
                f"eligible_for_induction=True 但带排除原因 {self.exclusion_reasons}")
        if not self.eligible_for_induction and not self.exclusion_reasons:
            raise ValueError(
                "eligible_for_induction=False 但没有 exclusion_reasons（§6.2："
                "不合格经验必须给出可回答的原因）")
        # §6.2-3/4：usage_supported/delivered_not_used 都要求正文真正进入请求。
        if self.retrieval_state in ("delivered_not_used", "usage_supported") \
                and not self.delivery_content_sha256:
            raise ValueError(
                f"retrieval_state={self.retrieval_state} 但缺少交付正文 hash"
                "（§6.2-3：正文真正进入请求才算交付）")
        if self.retrieval_state == "usage_supported" and not self.usage_clue_refs:
            raise ValueError("usage_supported 但 usage_clue_refs 为空（§6.2-4）")


class ExperienceBundle(Spec):
    """§6.3：框架生成的**不可变**经验包（按 `skill_id@version` 构建）。

    归纳器只读这个包，不直接读一个目录里的所有 trace：包是经过资格判定与
    排除统计的投影，包含哪些经验、排除了什么都可回答。
    """

    schema_version: str = EXPERIENCE_BUNDLE_SCHEMA_VERSION
    bundle_id: str
    campaign_id: str
    generation: int
    parent_snapshot_id: str
    parent_skill_id: str
    parent_skill_version: str
    canonical_question_type: str
    eligible_experience_refs: list[str] = []
    excluded_experience_refs: list[str] = []
    exclusion_summary: dict[str, int] = {}
    scene_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    behavior_summary: dict = {}
    failure_summary: dict = {}
    source_manifest_hash: str = ""

    def model_post_init(self, __context) -> None:  # noqa: D105
        # §6.3：经验包只属于一个父版本、一个题型 —— 混桶会让"是哪条 Skill 的经验"
        # 无法回答，也不能产出可归因的修订。
        if "@" in self.parent_skill_id or not self.parent_skill_version:
            raise ValueError(
                "经验包必须按 skill_id@version 构建："
                f"parent_skill_id={self.parent_skill_id!r} "
                f"parent_skill_version={self.parent_skill_version!r}")
        if not self.canonical_question_type:
            raise ValueError("经验包缺少 canonical_question_type（§6.3 禁止混题型）")
        if self.scene_count < 0 or self.success_count < 0 or self.failure_count < 0:
            raise ValueError("经验包计数不得为负")
        if self.success_count + self.failure_count > len(self.eligible_experience_refs):
            raise ValueError(
                "success+failure 超过合格经验数：计数不得凭空多出来")

    @property
    def parent_skill_key(self) -> str:
        return f"{self.parent_skill_id}@{self.parent_skill_version}"

    @property
    def n_eligible(self) -> int:
        return len(self.eligible_experience_refs)


__all__ = [
    "BEHAVIOR_SUMMARY_KEYS",
    "EXCLUSION_REASON_CODES",
    "EXPERIENCE_BUNDLE_SCHEMA_VERSION",
    "EXPERIENCE_EVENT_SCHEMA_VERSION",
    "EXPERIENCE_SPLITS",
    "FAILURE_SUMMARY_KEYS",
    "RETRIEVAL_STATES",
    "ExperienceBundle",
    "ExperienceEvent",
]
