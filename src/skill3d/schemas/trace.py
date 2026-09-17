"""§5.7 追踪 Schema（含 M21 RunManifest）。"""

from typing import Literal, Optional

from . import Spec


class FailureTaxonomy(Spec):
    episode_id: str
    categories: list[Literal[
        "perception", "reconstruction", "coordinate", "tool_contract",
        "program_syntax", "verifier_reject", "evaluator_noanswer", "scale_unknown"]]
    note: str


class EpisodeTrace(Spec):
    episode_id: str
    qa_id: str
    final_state: str
    program_trace_ref: str
    geometry_check_ref: str
    evaluation_ref: str
    failure: Optional[FailureTaxonomy]
    active_snapshot_ref: str


class EvaluationRun(Spec):
    run_id: str
    split: str
    n_episodes: int
    accuracy: Optional[float]
    mra: Optional[float]
    per_task: dict
    active_snapshot_ref: str
    code_commit: str
    timestamp: str


class EvolutionGeneration(Spec):
    generation: int
    root_candidates: list[str]
    promoted: list[str]
    rejected: list[str]
    quarantined: list[str]
    paired_outcomes_ref: str


class RunManifest(Spec):
    """M21 版本锁定清单。"""

    code_commit: str
    docker_digest: str
    checkpoint_sha256: str
    pip_freeze_hash: str
    config_hash: str
    mlflow_run_id: str
