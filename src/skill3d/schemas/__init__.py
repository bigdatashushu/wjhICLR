"""Skill3D 系统 Pydantic v2 Schema 单一事实源（对应系统架构.md §5）。

所有模型统一 ``extra="forbid"``，字段 snake_case。
所有阈值字段标注 TODO_CALIBRATE；GPT-6 相关字段标注 TODO_USER_INPUT。

``ser_json_inf_nan="constants"``：G1–G11 等指标在数据缺失时按约定取 NaN
（见 reconstruction_gate/quality_metrics.py），必须能经 JSON 往返而不丢语义；
默认的 null 序列化会让 NaN 变成 None 并在回读时验证失败。
"""

from pydantic import BaseModel, ConfigDict


class Spec(BaseModel):
    """统一基类：禁止额外字段；NaN/Inf 以常量形式序列化（可无损往返）。"""

    model_config = ConfigDict(extra="forbid", protected_namespaces=(),
                              ser_json_inf_nan="constants")


from .episode import (
    DataSplitConfig,
    FrameSet,
    InputFrame,
    InputErrorRecord,
    InputGateVerdict,
    VSIBenchEpisode,
)
from .authorization import (
    DECISION_VERSION,
    DENIAL_REASON_CODES,
    UNAVAILABLE_REASON_CODES,
    AuthorizationDecision,
    MetricGateStatus,
    ToolAuthorizationReceipt,
    gate_result_payload,
    not_applicable_gate,
)
from .answer import (
    ADAPTER_VERSION_LEGACY_SHORT_FORM,
    CANONICAL_UNIT_BY_QUESTION_TYPE,
    DERIVATION_OPS,
    AnswerBasis,
    AnswerPayload,
    AnswerUnit,
    AttributionLedger,
    AttributionVerification,
    EpisodeStatus,
    parse_answer_payload,
    verify_attribution,
)
from .data_access import (
    ACCESS_PURPOSES,
    COMPONENT_ROLES,
    DATA_SPLITS,
    REFUSAL_REASON_CODES,
    DataAccessRecord,
    utcnow_iso,
)
from .evidence import (
    ALWAYS_AVAILABLE_CAPABILITIES,
    CAPABILITIES,
    CAPABILITY_ORDER,
    GATE_SUBCONDITIONS,
    GATE_VERSION,
    PROFILE_VERSION,
    QUESTION_CAPABILITIES,
    SCENE_CAPABILITIES,
    CapabilityState,
    EvidenceProfile,
    MetricEvidenceGateResult,
    capability_at_least,
    metric_scale_state_from_gate,
    worst_state,
)
from .reconstruction import (
    LEGACY_ONLY_FIELDS,
    METRIC_TASK_TYPES,
    QUALITY_METRIC_VERSION,
    ConfidenceMap,
    ObjectInstance,
    ObjectRecord,
    QualityMetrics,
    ReconstructionArtifact,
    SceneState,
)
from .legacy import LegacyArtifact, LegacyArtifactError
from .retrieval import (
    CANDIDATE_REASON_CODES,
    DELIVERY_CHANNELS,
    DELIVERY_STATES,
    RETRIEVAL_RECORD_SCHEMA_VERSION,
    RETRIEVAL_TRIGGERS,
    SkillCandidateRecord,
    SkillRetrievalRecord,
    short_method_summary,
)
# `SparseBAReceipt`（原 `.sparse_ba`）随 v6 §20 废止 `vggt_sparse_ba` 一并归档到
# `skill3d/legacy/retired/sparse_ba.py`，不再由 Schemas 导出（Schema 与 legacy 隔离）。
from .readiness import ExperimentReadiness
from .tool import ToolCall, ToolResult, ToolSpec
from .program import ASTCheckResult, EpisodeProgram, ProgramExecutionTrace
from .skill import (
    DeepSeekGovernanceDecision,
    RetrievedSkill,
    SkillCandidateV11,
    SkillCandidateV5,
    SkillGovernanceDecision,
    SkillSpec,
    SkillSpecV11,
    SkillState,
    normalize_skill_markdown,
    parse_skill_markdown,
)
from .experience import (
    BEHAVIOR_SUMMARY_KEYS,
    EXCLUSION_REASON_CODES,
    EXPERIENCE_BUNDLE_SCHEMA_VERSION,
    EXPERIENCE_EVENT_SCHEMA_VERSION,
    EXPERIENCE_SPLITS,
    FAILURE_SUMMARY_KEYS,
    RETRIEVAL_STATES,
    ExperienceBundle,
    ExperienceEvent,
)
from .evolution import (
    AdmissionDecision,
    BudgetLimit,
    BudgetUsage,
    CANDIDATE_OPERATIONS,
    DECISION_CONDITION_KEYS,
    EVOLUTION_STATES,
    STATIC_CHECK_KEYS,
    CampaignDecision,
    CandidateRevision,
    EvolutionCampaign,
    PairedPanelReceipt,
    PostPublishUseReceipt,
    PromotionReceipt,
    RejectionReceipt,
    SkillCandidate,
    SkillEvaluationBinding,
    StaticValidationReceipt,
    SkillLibraryCandidate,
    CounterexampleBundle,
    CounterexampleCase,
    EnvironmentSnapshot,
    EvolutionSandboxSpec,
    ExperimentBranch,
    GPT6Patch,
    GPUJob,
    MetamorphicTransform,
    OptimizationRun,
    PairedOutcome,
    RevisionExperiment,
)
from .evolution_v11 import (
    V11_CAMPAIGN_SCHEMA_VERSION,
    V11_EVALUATION_SCHEMA_VERSION,
    V11_EXPERIENCE_SCHEMA_VERSION,
    V11CampaignCheckpoint,
    V11CampaignDecision,
    V11CampaignStatus,
    V11EvaluationArm,
    V11ExperienceBundle,
    V11ExperienceCase,
    V11PairedEvaluationReceipt,
    V11PostPublishObservation,
    V11PostPublishReceipt,
    V11PublicationReceipt,
    V11ReceiptRef,
    V11RevisionAttemptReceipt,
    V11RevisionProposal,
    V11StaticValidationReceipt,
)
from .memory import MemoryEntry, MemorySnapshot
from .trace import (
    EpisodeInputTrace,
    EpisodeTrace,
    EvaluationResultTrace,
    EvaluationRun,
    EvolutionGeneration,
    FailureTaxonomy,
    RunManifest,
    TraceRecord,
)

__all__ = [name for name in dir() if not name.startswith("_")]
