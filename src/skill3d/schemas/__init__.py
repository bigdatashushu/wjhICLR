"""Skill3D 系统 Pydantic v2 Schema 单一事实源（对应系统架构.md §5）。

所有模型统一 ``extra="forbid"``，字段 snake_case。
所有阈值字段标注 TODO_CALIBRATE；GPT-6 相关字段标注 TODO_USER_INPUT。
"""

from pydantic import BaseModel, ConfigDict


class Spec(BaseModel):
    """统一基类：禁止额外字段。"""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


from .episode import DataSplitConfig, InputFrame, InputGateVerdict, VSIBenchEpisode
from .reconstruction import (
    ConfidenceMap,
    CoverageMap,
    ObjectInstance,
    QualityMetrics,
    ReconstructionArtifact,
    SceneState,
)
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
