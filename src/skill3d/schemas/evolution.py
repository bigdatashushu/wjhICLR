"""§5.6 演进 Schema（含 §5.6b EvolutionSandboxSpec、M20 GPUJob）+ v10 演化合同。

v10 增补（§5.4 / §7.2 / §8.2 / §11.2 / §14.1）：`SkillCandidate`（完整候选）、
`SkillEvaluationBinding`（固定注入标记）、`EvolutionCampaign`（两代唯一调度入口）
与四类代际收据。这些对象是 v10 演化链的**合同**，不是可选审计字段。

命名说明：v5 的 `schemas/skill.py::SkillCandidate`（只有 `spec_content` 字符串、
无父版本、无谱系）已被 §7.2 的完整候选取代，改名为 `SkillCandidateV5`（仅历史
数据在读时解释）；包级 `SkillCandidate` 现在指向本模块的 v10 定义。
"""

from typing import Literal, Optional

from . import Spec
from .skill import SkillSpec

BranchRole = Literal["baseline", "memory_only", "skill_only",
                     "candidate", "old_version", "error_adversarial"]


class EnvironmentSnapshot(Spec):
    snapshot_id: str
    tool_registry_digest: str
    reconstruction_artifact_ref: str  # A/B 必须同源（硬约束 18）
    memory_snapshot_ref: str
    skill_registry_snapshot_ref: str
    prompt_version: str
    code_commit: str
    split_pointer: Literal["induction", "inner_validation"]
    episode_set_hash: str
    seed: int
    base_model_fingerprint: str
    frozen_fields: list[str]
    frozen_policy_ok: bool


class ExperimentBranch(Spec):
    branch_id: str
    snapshot_id: str
    role: BranchRole
    patched_memory_entry_ids: list[str]
    patched_skill_semvers: list[str]
    gpu_rank: int
    container_image_digest: str
    status: Literal["pending", "running", "done", "failed", "cached"]


class MetamorphicTransform(Spec):
    transform_id: str
    kind: Literal["viewpoint_change", "object_permutation",
                  "rigid_transform", "unit_change", "occlusion_dropframe"]
    params: dict
    expected_relation: str


class CounterexampleCase(Spec):
    counterexample_id: str
    branch_id: str
    minimal_episode_ref: str
    trigger_trace_ref: str
    originating_test: Literal["paired_ab", "property_based",
                              "metamorphic", "adversarial_injection"]
    metamorphic_transform: Optional[MetamorphicTransform]
    shrunk_by: Literal["auto", "human"]


class CounterexampleBundle(Spec):
    bundle_id: str
    source_revision_id: str
    failed_episode_refs: list[str]
    minimal_counterexamples: list[CounterexampleCase]
    metamorphic_transforms: list[MetamorphicTransform]
    regression_set_ref: str
    gpt6_visible_summary: str  # 不含答案
    generated_at: str


class PairedOutcome(Spec):
    pair_id: str
    snapshot_id: str
    arm_a_branch_id: str
    arm_b_branch_id: str
    n_episodes: int
    metric: Literal["accuracy", "mra"]
    mean_a: float
    mean_b: float
    delta: float
    ci95_lo: float
    ci95_hi: float
    # §7 / E-2：退化样本（零方差/完全相同组）p 记 None → 判"不显著"（不得声称显著）
    wilcoxon_p: Optional[float] = None
    slice_table: dict
    resource_cost: dict
    slice_no_regression: bool
    within_budget: bool
    # §7：多重比较校正（Bonferroni，按被测切片数）与效应量
    wilcoxon_p_bonferroni: Optional[float] = None
    n_comparisons: int = 1
    degenerate: bool = False
    cliffs_delta: Optional[float] = None
    cohens_d: Optional[float] = None


class AdmissionDecision(Spec):
    decision_id: str
    candidate_id: str
    paired_outcomes: list[PairedOutcome]
    counterexamples: list[CounterexampleCase]
    min_delta_required: float  # TODO_CALIBRATE
    ci_significant: bool
    slice_no_regression: bool
    no_leakage: bool
    cross_scene_multisample: bool
    within_budget: bool
    sim2real_robust: bool  # MVP 恒 True 占位
    promotes: bool
    reason: str
    gpt6_review: Optional[str]  # SkillGovernanceDecision id，事后


class CandidateRevision(Spec):
    revision_id: str
    root_candidate_id: str
    parent_version: Optional[str]
    candidate_type: Literal["memory", "skill"]
    spec_content: str
    status: Literal["draft", "testing", "promoted", "rejected", "quarantined"]
    induction_trace_refs: list[str]
    evidence_lineage_ref: str
    created_by: Literal["gpt6_induction", "gpt6_revision", "human"]
    created_at: str
    # v9 provenance refs are optional for compatibility with historical
    # in-memory candidates; file-backed library candidates should populate them.
    source_split: Optional[str] = None
    experience_relation: Optional[str] = None
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    generated_spec_path: Optional[str] = None
    generated_sha256: Optional[str] = None
    manifest_ref: Optional[str] = None
    candidate_record_ref: Optional[str] = None


class SkillLibraryCandidate(Spec):
    """v9 file-backed candidate record, separate from runtime CandidateRevision.

    The record describes why a candidate exists and how it may be validated;
    ``CandidateRevision.spec_content`` remains the immutable runtime payload
    used by the existing evolution driver.
    """

    candidate_id: str
    operation: str
    canonical_question_type: str
    parent_snapshot_id: Optional[str] = None
    parent_skill_versions: list[str] = []
    source_trace_refs: list[str] = []
    source_split: str
    experience_relation: str
    hypothesis: str
    patch: str
    expected_scope: str
    expected_effect: str
    known_risks: str
    inducer_receipt_ref: Optional[str] = None
    static_check_ref: Optional[str] = None
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    generated_spec_path: Optional[str] = None
    generated_sha256: Optional[str] = None
    manifest_ref: Optional[str] = None
    created_at: str


class RevisionExperiment(Spec):
    experiment_id: str
    revision_id: str
    baseline_revision_id: str
    level: Literal["L1_minimal_slice", "L2_full_inner", "L3_outer_holdout"]
    panel_ref: str
    environment_snapshot_ref: str
    paired_outcome_ref: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]


class BudgetUsage(Spec):
    tokens: int
    gpu_hours: float
    rollouts: int
    revisions: int


class BudgetLimit(Spec):
    max_gpt6_tokens: int  # TODO_CALIBRATE (500000)
    max_gpu_hours: float  # TODO_CALIBRATE (100)
    max_rollouts: int  # TODO_CALIBRATE (50)
    max_revisions: int  # TODO_CALIBRATE (10)
    patience: int  # TODO_CALIBRATE (3)


class OptimizationRun(Spec):
    run_id: str
    root_candidate_id: str
    current_revision_id: str
    budget_used: BudgetUsage
    budget_limit: BudgetLimit
    patience_counter: int
    revision_history: list[str]
    parallel_branches: list[str]
    selection_strategy: Literal["successive_halving", "conservative_bandit", "sequential"]
    status: Literal["running", "promoted", "rejected", "quarantined",
                    "budget_exhausted", "paused"]
    termination_reason: Optional[str]


class GPT6Patch(Spec):
    target_revision_id: str
    patch_type: Literal["modify_skill", "add_assertion", "fix_precondition",
                        "adjust_call_graph", "add_test_case", "revise_description"]
    affected_task_types: list[str]
    affected_skills: list[str]
    patch_content: str
    rationale: str
    expected_improvement: str
    risk_notes: str
    model_id: str = "TODO_USER_INPUT"
    prompt_version: str


class EvolutionSandboxSpec(Spec):
    """§5.6b 一次反事实实验的声明式描述。"""

    spec_id: str
    environment_snapshot_ref: str  # EnvironmentSnapshot id
    branch_roles: list[BranchRole]
    episode_set_ref: str
    gpu_dp_world_size: int = 8
    paired_same_gpu: bool = True
    sandbox_gpus: list[int] = []  # 空 = 沙箱不挂 GPU
    mode: Literal["mock_light", "mock_replay", "real"]  # 准入门必须 real
    auto_counterexample: bool = False  # MVP False，增强版 True
    metamorphic_kinds: list[str] = []


class GPUJob(Spec):
    """M20 调度单元。"""

    gpu_rank: int
    role: Literal["reconstruct", "vllm", "eval"]
    paired_unit_id: str


# =====================================================================================
# v10：候选 / 效果评测 / 两代 campaign 合同（§5.4、§7.2、§8.2、§10.1、§11.2、§14.1）
# =====================================================================================

# §11.1 状态机的显式状态名（与 §11.1 原文逐字一致；顺序即流转顺序）。
EVOLUTION_STATES: tuple[str, ...] = (
    "INIT",
    "RUN_PARENT_LEARNING",
    "BUILD_EXPERIENCE_BUNDLE",
    "GENERATE_CANDIDATE",
    "STATIC_VALIDATE",
    "BUILD_CANDIDATE_SNAPSHOT",
    "RUN_INNER_SEED_0",
    "RUN_INNER_SEED_1",
    "DECIDE",
    "PUBLISH_OR_REJECT",
    "VERIFY_POST_PUBLISH_USE",
    "NEXT_GENERATION",
    "COMPLETE",
)
EvolutionStateValue = Literal[
    "INIT", "RUN_PARENT_LEARNING", "BUILD_EXPERIENCE_BUNDLE", "GENERATE_CANDIDATE",
    "STATIC_VALIDATE", "BUILD_CANDIDATE_SNAPSHOT", "RUN_INNER_SEED_0",
    "RUN_INNER_SEED_1", "DECIDE", "PUBLISH_OR_REJECT", "VERIFY_POST_PUBLISH_USE",
    "NEXT_GENERATION", "COMPLETE",
]

# 候选来源（`SkillCandidate.inducer_receipt_ref` 指向的收据类型）。
CANDIDATE_OPERATIONS: tuple[str, ...] = ("revise",)

# §5.4 静态检查项（名字必须是**稳定标识**，收据里逐项落盘）。
STATIC_CHECK_KEYS: tuple[str, ...] = (
    "json_parseable",
    "schema_extra_forbid",
    "skill_id_matches_parent",
    "version_bump_legal",
    "single_canonical_question_type",
    "method_body_deliverable",
    "tools_known",
    "no_leakage",
    "parent_hash_matches",
    "diff_nonempty_and_consistent",
)

# §8.6 / §13 逐 seed 判定条件（名字稳定，decision.json 里逐项落盘）。
DECISION_CONDITION_KEYS: tuple[str, ...] = (
    "panel_score_strictly_improved",
    "run_error_not_increased",
    "valid_answer_rate_not_decreased",
    "no_schema_or_permission_violation",
    "both_arms_body_entered_request",
    "candidate_delivered_at_least_once",
    "same_episode_and_artifacts",
    "content_and_experience_eligible",
)


class SkillCandidate(Spec):
    """§7.2：离线归纳器产出的**完整候选**（不是自由文本增量）。

    规范原文（§5.4）："候选修订必须产生完整合法 `SkillSpec`，禁止把自由文本直接
    拼接到序列化 JSON 后。" 因此 `full_skill_spec` 是解析后的完整值对象，
    `structured_diff` 是框架对父 / 子做的字段差分（供审计，不参与运行）。
    """

    candidate_id: str
    campaign_id: str
    generation: int
    operation: Literal["revise"]
    parent_snapshot_id: str
    parent_skill_version: str          # skill_id@version
    candidate_skill_version: str       # skill_id@version
    canonical_question_type: str
    hypothesis: str = ""
    expected_effect: str = ""
    known_risks: list[str] = []
    full_skill_spec: SkillSpec
    structured_diff: list[dict] = []
    source_experience_bundle_ref: str = ""
    inducer_receipt_ref: str = ""

    def model_post_init(self, __context) -> None:  # noqa: D105
        parent_id, _, parent_ver = self.parent_skill_version.partition("@")
        cand_id, _, cand_ver = self.candidate_skill_version.partition("@")
        if not parent_ver or not cand_ver:
            raise ValueError(
                "父 / 候选技能版本必须是 skill_id@version 形式："
                f"{self.parent_skill_version!r} → {self.candidate_skill_version!r}")
        if parent_id != cand_id:
            raise ValueError(
                f"候选不得跨谱系：父={parent_id} 候选={cand_id}（§7.2 operation=revise）")
        if parent_id != self.full_skill_spec.skill_id:
            raise ValueError(
                f"候选 SkillSpec.skill_id={self.full_skill_spec.skill_id} 与父谱系 "
                f"{parent_id} 不一致")
        if cand_ver != self.full_skill_spec.version:
            raise ValueError(
                f"candidate_skill_version={cand_ver} 与 full_skill_spec.version="
                f"{self.full_skill_spec.version} 不一致")
        if self.canonical_question_type not in self.full_skill_spec.applicable_question_types:
            raise ValueError(
                f"canonical_question_type={self.canonical_question_type} 不在候选声明的"
                f"题型 {self.full_skill_spec.applicable_question_types} 内（§8.5）")
        if not self.structured_diff:
            raise ValueError("structured_diff 为空（§5.4：diff 必须非空且与候选内容一致）")

    @property
    def skill_id(self) -> str:
        return self.full_skill_spec.skill_id

    @property
    def version(self) -> str:
        return self.full_skill_spec.version


class SkillEvaluationBinding(Spec):
    """§8.2：固定注入的显式标记（只能用于候选效果评测）。

    规范原文（§8.2）："该模式只能用于候选效果评测，不能用于 learning 经验采集、
    发布后运行或最终系统成绩。" 因此这个对象的存在本身就是"这一跑不是正常检索"的
    证据 —— 它必须落进 trace，且不允许被改写成正常检索命中。
    """

    mode: Literal["fixed_skill_evaluation"]
    arm: Literal["parent", "candidate"]
    skill_id: str
    skill_version: str
    content_sha256: str
    bypassed_component: Literal["retrieval_selection"]

    def model_post_init(self, __context) -> None:  # noqa: D105
        if self.skill_version != f"{self.skill_id}@{self.skill_version.split('@')[-1]}":
            raise ValueError(f"skill_version 必须是 skill_id@version：{self.skill_version!r}")
        if not self.content_sha256:
            raise ValueError("固定注入必须带正文 hash（§16.2：核对两臂正文 hash）")


class StaticValidationReceipt(Spec):
    """§14.1 `static_validation.json`：Schema、工具、泄漏、版本、长度检查。"""

    candidate_id: str
    campaign_id: str = ""
    generation: int = 0
    checks: dict[str, bool] = {}
    problems: list[str] = []
    passed: bool = False
    created_at: str = ""

    def model_post_init(self, __context) -> None:  # noqa: D105
        unknown = sorted(set(self.checks) - set(STATIC_CHECK_KEYS))
        if unknown:
            raise ValueError(f"未知静态检查项 {unknown}；词表={list(STATIC_CHECK_KEYS)}")
        if self.passed and self.problems:
            raise ValueError("passed=True 但 problems 非空（不能一边通过一边有问题）")
        if self.passed and not all(self.checks.values()):
            raise ValueError(
                "passed=True 但存在未通过检查项："
                f"{sorted(k for k, v in self.checks.items() if not v)}")


class PairedPanelReceipt(Spec):
    """§14.1 `paired_seed_{0,1}.json`：同一 inner 子面板的父 / 候选逐题配对结果。

    两臂的 `content_sha256` 是**固定注入的正文身份**：§16.2 要求核对两臂正文 hash
    与真实请求一致。`n_run_error` / 合法答案率 / 交付覆盖是 §8.6 与 §13 的判定输入。
    """

    campaign_id: str
    generation: int
    seed: int
    panel_id: str
    panel_hash: str = ""
    n_items: int = 0
    arm_a_skill_version: str = ""
    arm_b_skill_version: str = ""
    arm_a_content_sha256: str = ""
    arm_b_content_sha256: str = ""
    mean_a: float = 0.0
    mean_b: float = 0.0
    delta: float = 0.0
    n_run_error_a: int = 0
    n_run_error_b: int = 0
    valid_answer_rate_a: float = 0.0
    valid_answer_rate_b: float = 0.0
    delivered_a: int = 0
    delivered_b: int = 0
    per_item: list[dict] = []
    created_at: str = ""


class CampaignDecision(Spec):
    """§14.1 `decision.json`：promote/reject 及**每一项条件**。"""

    campaign_id: str
    generation: int
    candidate_id: str
    promote: bool
    conditions: dict[str, bool] = {}
    per_seed: dict[str, dict] = {}
    reasons: list[str] = []
    created_at: str = ""

    def model_post_init(self, __context) -> None:  # noqa: D105
        unknown = sorted(set(self.conditions) - set(DECISION_CONDITION_KEYS))
        if unknown:
            raise ValueError(
                f"未知准入条件 {unknown}；词表={list(DECISION_CONDITION_KEYS)}")
        if self.promote and not all(self.conditions.values()):
            raise ValueError(
                "promote=True 但存在未满足条件："
                f"{sorted(k for k, v in self.conditions.items() if not v)}"
                "（§13：任一 seed 持平/下降/零交付/运行错误增加 → reject）")


class PromotionReceipt(Spec):
    """§14.1 `promotion.json`：快照切换、父子关系和 manifest hash。"""

    campaign_id: str
    generation: int
    candidate_id: str
    skill_version: str
    snapshot_before: str
    snapshot_after: str
    manifest_hash_before: str = ""
    manifest_hash_after: str = ""
    parent_snapshot_id: str | None = None
    competing_versions: list[str] = []
    historical_versions: list[str] = []
    rollback_ref: str = ""             # 父快照文件路径（§10.1-8 保存父快照供回滚）
    created_at: str = ""


class PostPublishUseReceipt(Spec):
    """§14.1 `post_publish_use.json`：新快照 / 新版本**实际**检索与交付证明。

    规范原文（§10.2）："不满足以上条件时，状态只能是 `promoted_not_observed`，
    不能开始下一代归纳。"
    """

    campaign_id: str
    generation: int
    snapshot_id: str
    new_skill_version: str
    episodes_run: int = 0
    retrieved: bool = False
    delivered: bool = False
    delivered_content_sha256: str = ""
    snapshot_content_sha256: str = ""
    request_content_sha256_match: bool = False
    experience_event_refs: list[str] = []
    status: Literal["observed", "promoted_not_observed"] = "promoted_not_observed"
    created_at: str = ""

    def model_post_init(self, __context) -> None:  # noqa: D105
        if self.status == "observed" and not (self.retrieved and self.delivered):
            raise ValueError(
                "status=observed 但 retrieved/delivered 未同时成立（§10.2："
                "发布后必须证明新版本被检索且真实交付）")
        if self.status == "observed" and not self.request_content_sha256_match:
            raise ValueError(
                "status=observed 但请求正文 hash 与快照内容不一致（§10.2）")


class RejectionReceipt(Spec):
    """§14.1 `promotion.json` 的 **reject** 形态（保留 Sn 快照，不动 active 指针）。

    与 `PromotionReceipt` 分开是刻意的：reject 时**没有**新快照、没有 manifest 变化，
    用一个"字段留空"的 promote 收据表示拒绝会让"到底发布没发布"变得含糊。
    两种形态在 `promotion.json` 里以 `outcome` 字段区分（`promoted` / `rejected`）。
    """

    outcome: Literal["rejected"] = "rejected"
    campaign_id: str
    generation: int
    candidate_id: str
    reasons: list[str] = []
    snapshot_before: str = ""
    created_at: str = ""


class EvolutionCampaign(Spec):
    """§11.2：两代演化的唯一调度入口。"""

    campaign_id: str
    target_question_type: str
    target_skill_id: str
    max_generations: int
    current_generation: int
    initial_snapshot_id: str
    current_parent_snapshot_id: str
    state: str
    generation_receipt_refs: list[str] = []
    final_snapshot_id: str | None = None
    completion_status: Literal[
        "running", "completed_two_generations",
        "completed_with_rejection", "blocked", "failed"
    ] = "running"

    def model_post_init(self, __context) -> None:  # noqa: D105
        if self.state not in EVOLUTION_STATES:
            raise ValueError(
                f"未知 campaign 状态 {self.state!r}；词表={list(EVOLUTION_STATES)}")
        if self.current_generation < 0 or self.max_generations < 1:
            raise ValueError("current_generation/max_generations 非法")


__all__ = [
    "CANDIDATE_OPERATIONS",
    "DECISION_CONDITION_KEYS",
    "EVOLUTION_STATES",
    "STATIC_CHECK_KEYS",
    "CampaignDecision",
    "EvolutionCampaign",
    "EvolutionStateValue",
    "PairedPanelReceipt",
    "PostPublishUseReceipt",
    "PromotionReceipt",
    "RejectionReceipt",
    "SkillCandidate",
    "SkillEvaluationBinding",
    "StaticValidationReceipt",
]
