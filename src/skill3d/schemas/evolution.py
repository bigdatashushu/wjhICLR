"""§5.6 演进 Schema（含 §5.6b EvolutionSandboxSpec、M20 GPUJob）。"""

from typing import Literal, Optional

from . import Spec

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
