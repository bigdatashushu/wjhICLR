"""§5.9 追踪 Schema（TraceRecord / EpisodeTrace / RunManifest，v6 D10）。

v6 新增三类**必须落盘**的内容（§19）：

1. **版本字段**：模板版本 / tool-face 版本 / EvidenceProfile 版本 / gate 版本 /
   距离原语参数 / 尺度融合版本 —— 否则"这个数字是哪版口径算的"无法回答；
2. **证据与路由全状态**：EvidenceProfile、MetricEvidenceGateResult、
   `scene_route` × `question_tool_scope`、`answer_source`、`used_result_ids`、
   `recovery_count`、`partial_tool_recovery`、级联撤销列表；
3. **`synthesis_source` 拆 6 类**（§19.3）：不再用一个 `vllm` 混为一谈 ——
   v5 曾把 M8 解析失败的 episode 也记 `synthesis_source='vllm'`，与服务不可用混淆。

`RunManifest` 另含 §19.2 的离线治理模型字段（`provider="deepseek"`、
`model_id="deepseek-flash"`）；**认证值只从 `DEEPSEEK_API_KEY` 注入，绝不落盘**。
"""

from typing import Literal, Optional

from . import Spec

# 答案来源（§5.3 D9 四值）
AnswerSourceValue = Literal["tool_program", "direct_vlm_routed", "abstain",
                            "tool_contract"]

# synthesis_source 六类（§19.3）+ 一个显式非论文值。
#
# `mock_stub` 是**实现补充**：`mode=mock_light` 下程序来自确定性 stub，既不是
# VLLM 成功也不是任何失败，写成六类中的任何一个都是假话。它被明确排除在
# paper-eligible 之外（§18.6 要求非 mock 证据），因此不削弱"6 类"的目的
# （把解析失败与解析回退、服务不可用区分开）。
SynthesisSourceValue = Literal[
    "vllm_ok", "vllm_parse_error", "vllm_service_error",
    "m8_parse_recovered", "direct_answer_fallback", "partial_tool_recovery",
    "mock_stub",
]

# 失败码（§5.9 / §19.1；不依赖重跑即可归因失败）
FailureCode = Literal[
    "grounding_recall_miss", "detector_fault", "ConfidenceGateError",
    "AnswerAlreadyGiven", "DomainValueError", "metric_evidence_gate_failed",
    "m4_main_gate_failed", "world_frame_unavailable", "ast_violation",
    "degenerate_output_regenerated", "service_unavailable", "unknown",
]


class FailureTaxonomy(Spec):
    episode_id: str
    categories: list[Literal[
        "perception", "reconstruction", "coordinate", "tool_contract",
        "program_syntax", "verifier_reject", "evaluator_noanswer",
        "metric_evidence", "world_frame", "synthesis"]]
    note: str


class TraceRecord(Spec):
    """逐 episode 全套落盘（§5.9 D10）——**不依赖重跑即可归因失败**。"""

    episode_id: str
    # ---- 版本字段（D10）----
    template_version: str = ""
    tool_face_version: str = ""
    evidence_profile_version: str = ""
    gate_version: str = ""
    distance_primitive_params: dict = {}      # quantile_q / voxel_size / conf_warp_version
    metric_fusion_version: str = ""
    # ---- 证据与路由 ----
    evidence_profile: Optional[dict] = None
    metric_evidence_gate_result: Optional[dict] = None
    scene_route: str = ""
    question_tool_scope: str = ""
    answer_source: str = ""
    used_result_ids: list[str] = []
    recovery_count: int = 0
    partial_tool_recovery: bool = False
    invalidated_result_ids: list[str] = []
    # ---- synthesis_source 拆 6 类 ----
    synthesis_source: SynthesisSourceValue = "vllm_ok"
    # ---- 判读信息（P8）----
    n_objects: int = 0
    cache_hit: bool = False
    failure_code: Optional[str] = None
    m5_notes: Optional[str] = None
    m7_notes: Optional[str] = None
    m8_notes: Optional[str] = None


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
    # 但"命中过 tool_contract"必须留痕，供审计回溯与过程指标统计。
    tool_contract_hits: int = 0
    abstained: bool = False
    answer_untrusted: bool = False
    scene_route: str = ""
    question_tool_scope: str = ""
    quality_status: str = ""
    # ---- v6：逐 episode 的可审计事实（§19.1/§19.2）----
    schema_version: str = "6.0"
    quality_metric_version: str = ""
    frame_set_hash: str = ""
    n_frames: int = 0
    n_images_to_synthesizer: int = 0        # M8 实际收到的图像数（不得静默丢帧）
    reprojection_status: str = "not_available"   # v6 恒为 not_available（§10.4）
    overall_quality: Optional[float] = None
    main_gate_passed: Optional[bool] = None
    # 世界系契约（D5）
    world_frame_status: str = "unavailable"
    # 度量证据（D1/D3）
    scale_fusion_status: str = "not_run"
    metric_scale: Optional[float] = None
    metric_gate_passed: bool = False
    metric_model: str = "none"
    # 逐题授权后的米制题型（空 = 未授权）
    authorized_metric_tasks: list[str] = []
    # EvidenceProfile 三值快照（便于按证据状态分组统计，§16.5/§18.5）
    evidence_states: dict[str, str] = {}
    input_degradation_flags: list[str] = []
    answer_source: str = ""
    # v6 D7：partial recovery 事实（§14.1）
    recovery_count: int = 0
    partial_tool_recovery: bool = False
    used_result_ids: list[str] = []
    invalidated_result_ids: list[str] = []
    failure_code: Optional[str] = None
    # synthesis_source 六类（§19.3）
    synthesis_source: str = ""
    degenerate_regenerated: bool = False


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
    # §18.3 噪声底：同配置重复的 seed 数（paper-eligible 需 >=3/>=5）
    n_seeds: int = 1


class EvolutionGeneration(Spec):
    generation: int
    root_candidates: list[str]
    promoted: list[str]
    rejected: list[str]
    quarantined: list[str]
    paired_outcomes_ref: str


class RunManifest(Spec):
    """M21 版本锁定清单（§19.2 复现 checklist）。"""

    code_commit: str
    docker_digest: str
    checkpoint_sha256: str
    pip_freeze_hash: str
    config_hash: str
    mlflow_run_id: str
    split_version: str = ""
    split_config_hash: str = ""
    seed: Optional[int] = None
    inference_env: dict = {}
    vllm_model: str = ""
    n_frames: int = 0
    max_pixels: int = 0
    max_model_len: int = 0
    vllm_endpoints: list[str] = []
    frame_set_hash: str = ""
    # ---- v6：重建/质量/证据版本（§19.2）----
    recon_method: str = "vggt"              # v6 只允许 vggt
    quality_metric_version: str = ""
    template_version: str = ""
    tool_face_version: str = ""
    evidence_profile_version: str = ""
    gate_version: str = ""
    metric_model: str = "none"
    metric_fusion_version: str = ""
    distance_primitive_params: dict = {}
    golden_version: str = ""
    # ---- 离线治理模型（§3.4/§19.2）----
    # `provider="deepseek"`、`model_id="deepseek-flash"` 已核验；
    # 认证值仅从 DEEPSEEK_API_KEY 注入，**绝不落盘**。
    offline_model: str = ""
    provider: str = ""
    model_id: str = ""
    endpoint_hash: str = ""
    prompt_version: str = ""
    latency: Optional[float] = None
    token_usage: dict = {}
    # HC34：readiness manifest 引用（四级证据门的快照位置）
    readiness_manifest_ref: str = ""


__all__ = [
    "AnswerSourceValue",
    "EpisodeTrace",
    "EvaluationRun",
    "EvolutionGeneration",
    "FailureCode",
    "FailureTaxonomy",
    "RunManifest",
    "SynthesisSourceValue",
    "TraceRecord",
]
