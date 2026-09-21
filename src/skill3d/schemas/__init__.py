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
    InputGateVerdict,
    VSIBenchEpisode,
)
from .reconstruction import (
    LEGACY_ONLY_FIELDS,
    METRIC_TASK_TYPES,
    ConfidenceMap,
    ImageGridTransform,
    ObjectInstance,
    QualityMetrics,
    ReconstructionArtifact,
    ScaleAnchorEvidence,
    SceneState,
)
from .legacy import LegacyArtifact, LegacyArtifactError
from .sparse_ba import SparseBAReceipt
from .readiness import ExperimentReadiness
from .tool import ToolCall, ToolResult, ToolSpec
from .program import ASTCheckResult, EpisodeProgram, ProgramExecutionTrace
from .skill import (
    RetrievedSkill,
    SkillCandidate,
    SkillGovernanceDecision,
    SkillSpec,
    SkillState,
)
from .evolution import (
    AdmissionDecision,
    BudgetLimit,
    BudgetUsage,
    CandidateRevision,
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
from .memory import MemoryEntry, MemorySnapshot
from .trace import (
    EpisodeTrace,
    EvaluationRun,
    EvolutionGeneration,
    FailureTaxonomy,
    RunManifest,
)

__all__ = [name for name in dir() if not name.startswith("_")]
