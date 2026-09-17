"""§5.5 Skill / 治理 Schema。"""

from typing import Literal, Optional

from . import Spec

SkillState = Literal["draft", "shadow", "canary", "promoted", "quarantined"]


class SkillSpec(Spec):
    """题型级程序合成模板（非可执行工具/非答案，硬约束 15）。"""

    skill_id: str
    semver: str  # MAJOR.MINOR.PATCH
    task_type: str
    description: str
    call_graph_template: str  # 题型级程序合成模板
    requires_artifacts: list[str]
    minimum_quality: float  # TODO_CALIBRATE
    supported_coordinate_frames: list[str]
    metric_scale_required: bool
    validation_assertions: list[str]


class SkillCandidate(Spec):
    candidate_id: str
    root_candidate_id: str
    revision_id: str
    parent_version: Optional[str]
    candidate_type: Literal["memory", "skill"]
    spec_content: str
    status: Literal["draft", "testing", "promoted", "rejected", "quarantined"]
    gpt6_patch_ref: Optional[str]
    induction_trace_refs: list[str]
    created_by: Literal["gpt6_induction", "gpt6_revision", "human"]
    created_at: str


class SkillGovernanceDecision(Spec):
    decision_id: str
    candidate_id: str
    review_summary: str
    semantic_risk: str
    generalization_notes: str
    gpt6_model_id: str = "TODO_USER_INPUT"
    gpt6_prompt_version: str
    timestamp: str


class RetrievedSkill(Spec):
    """M7 Skill 检索输出（§4 M7 字段 6）。"""

    skill_semver: str
    score: float
    hard_filter_passed: bool
