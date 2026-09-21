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
    # §4 M13 / D-3 归因：契约违规与 abstain 的逐 episode 事实。
    # 与 `failure` 分开记——恢复成功的 episode 不算 failure（否则污染归纳输入），
    # 但"命中过 tool_contract"必须留痕，供审计回溯（G-40）与过程指标统计。
    tool_contract_hits: int = 0
    abstained: bool = False
    answer_untrusted: bool = False
    scene_route: str = ""
    quality_status: str = ""
    # ---- v5 HC35–39：逐 episode 的可审计事实（§7 复现清单 / §14.1-6 smoke 核对）----
    # 这些字段让"质量是否实算、G5 是否被代理、尺度是否越权、帧集是否统一"能在
    # trace 层直接验证，而不必回读 artifact 或相信报告摘要。
    schema_version: str = "5.0"
    quality_metric_version: str = ""
    frame_set_hash: str = ""
    n_frames: int = 0
    n_images_to_synthesizer: int = 0        # M8 实际收到的图像数（HC26：不得静默丢帧）
    reprojection_status: str = "not_available"
    coverage_gate_status: str = "not_defined"   # HC38：本版不设 geometric_coverage 门
    scale_confidence: str = "low"
    allowed_metric_tasks: list[str] = []
    authorized_metric_tasks: list[str] = []     # 逐题门控后实际可用
    overall_quality: Optional[float] = None
    input_degradation_flags: list[str] = []
    # 答案来源：program（沙箱执行程序）/ direct_vlm（C0）/ direct_vlm_routed（题型策略回退）
    # 用途：审计"这一分是程序拿的还是直答拿的"，防止把直答成绩记成程序能力
    answer_source: str = ""


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
    """M21 版本锁定清单（§16.4 复现 checklist）。"""

    code_commit: str
    docker_digest: str
    checkpoint_sha256: str
    pip_freeze_hash: str
    config_hash: str
    mlflow_run_id: str
    # §16.4：split version / split 配置哈希 / seed / 推理环境版本 一并记录
    split_version: str = ""
    split_config_hash: str = ""
    seed: Optional[int] = None
    inference_env: dict = {}
    # §7 / §3 M21 / E-3：影响结果的推理参数全部入册（缺省不臆造）
    vllm_model: str = ""
    n_frames: int = 0
    max_pixels: int = 0
    max_model_len: int = 0
    vllm_endpoints: list[str] = []
    frame_set_hash: str = ""
    # ---- v5 HC35–37/39：重建主线与 BA 状态（默认全部关闭且不得被误开）----
    # 官方 VGGSfM BA 已被 24 GiB OOM 否决（HC35）；`vggt_sparse_ba` 仅在 §10.1 PoC
    # 通过后才可能为真（HC36），且即使通过也不升为主线。
    ba_enabled: bool = False
    recon_method: str = "vggt"
    sparse_ba_enabled: bool = False
    sparse_ba_frontend: str = ""
    sparse_ba_pair_graph_hash: str = ""
    official_vggsfm_ba_enabled: bool = False
    reprojection_status: str = "not_available"
    # ---- v5 版本字段（HC39：schema/质量口径/golden 三重版本必须同时落盘）----
    schema_version: str = "5.0"
    quality_metric_version: str = ""
    golden_version: str = ""
    scale_source: str = ""
    # ---- v4 HC29–34 复现清单（§7：尺度校准复现必须可审计）----
    # 冻结校准器 id（在线只读加载的那个）；未标定时为空 → 所有米制题型被收回
    scale_calibration_id: str = ""
    # 标定集 scene ID 清单哈希 / 被排除的 VSI-Bench 150 ARKitScenes scene ID 哈希
    # （HC32：交集断言 + 两个哈希都要留档，否则无法证明隔离）
    calibration_split_hash: str = ""
    excluded_vsibench_scene_hash: str = ""
    confidence_level: Optional[float] = None      # nominal coverage（如 0.90）
    scale_confidence_distribution: dict = {}
    allowed_metric_tasks_observed: list[str] = []
    # HC34：readiness manifest 引用（四级证据门的快照位置）
    readiness_manifest_ref: str = ""
