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
# 已知取值集合（供"不认识的取值不得被洗成 vllm_ok"的判定使用）。
KNOWN_SYNTHESIS_SOURCES: frozenset[str] = frozenset(
    SynthesisSourceValue.__args__)  # type: ignore[attr-defined]

# 轮次触发原因（v9 §10.1 `ProgramRound.trigger` / §12.2）。
#
# 与 `synthesis_source` **分开记录**（v9 §12.2："`finalization_used`、
# `round_trigger`、`basis` 分开记录"）：`synthesis_source` 回答"这段程序文本从
# 哪里来"（模型 / 解析回退 / mock stub），`round_trigger` 回答"这一轮为什么被
# 生成"（首轮 / 观察回灌 / 错误恢复 / 收口）。把后者塞进前者会丢失程序血统 ——
# v8 曾把 finalization/forced_answer 写进 `synthesis_source`，于是 mock_light 下
# 的收口答案在 TraceRecord 里被记成 `vllm_ok`。
RoundTriggerValue = Literal["initial", "observation", "error_recovery", "finalize"]

# EpisodeTrace 的 Schema 身份（v9 §17.2：旧结果保留原身份，新结果用当前合同）。
#
# 6.0 语料（`data/v5_*`、`data/v6_*`、`data/v7_*`）继续声明 "6.0"，由
# `skill3d.legacy.readers.read_legacy_episode_trace` 只读解释；**不得**把 6.0 的
# JSON 直接 `model_validate` 成当前对象 —— 新增字段会被默认值静默填满，看起来
# 像"当时那个事实不存在"，而真相是"当时没记这个事实"。
EPISODE_TRACE_SCHEMA_VERSION = "9.0"
LEGACY_EPISODE_TRACE_SCHEMA_VERSIONS: tuple[str, ...] = ("6.0",)

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
        "perception", "reconstruction", "coordinate", "tool_contract", "run_error",
        "program_syntax", "verifier_reject", "evaluator_noanswer",
        "metric_evidence", "world_frame", "synthesis"]]
    note: str


class TraceRecord(Spec):
    """逐 episode 全套落盘（§5.9 D10）——**不依赖重跑即可归因失败**。"""

    episode_id: str
    active_snapshot_manifest_sha256: str = ""
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
    # 未知/空值**不再**冒充 `vllm_ok`：空串表示"没拿到可归因的来源"。已知取值
    # 见 `KNOWN_SYNTHESIS_SOURCES`。
    synthesis_source: str = ""
    # ---- v9 轮次事实（§12.2：与 synthesis_source 分开记）----
    round_trigger: RoundTriggerValue = "initial"
    finalization_used: bool = False
    # ---- v9 §10.1/§12：答案合同与工具归因 ----
    # `episode_status` 是 §10.1 的规范终态；`status` 的历史碎片值（含 v6 的
    # `unanswerable`/`abstain`）不再作为本版正常终态，但为历史统计留在
    # EpisodeTrace.final_state。
    episode_status: str = ""
    # 答案依据（tool_derived / visual_estimate / mixed）——由框架核验后的取值，
    # 不是模型自称；核验问题在 `attribution["problems"]`。
    answer_basis: str = ""
    answer: dict = {}
    # §12 六字段台账 + 保守降级记录（declared/verified/ignored 三分离）
    attribution: dict = {}
    # §6.4：每次实际调用的授权收据（allowed 与被拒绝的原因码都在）
    authorization_receipts: list[dict] = []
    # ---- v9 §17.1「检索与 Round」层：轮数/重试/finalization 必须落盘 ----
    # 此前这些只在 runner 内存里，评审无法从 trace 回答"这一题花了多少轮、
    # yield 了几次、是否进了收口阶段、预算上限是多少"。
    agent_rounds: int = 0
    yield_count: int = 0
    round_trace_refs: list[str] = []
    # v9 §17.1 Round 层：逐轮事实（序号/触发/程序文本与哈希/本轮观测/结束方式）。
    rounds: list[dict] = []
    budget: dict = {}
    # ---- 判读信息（P8）----
    n_objects: int = 0
    cache_hit: bool = False
    failure_code: Optional[str] = None
    m5_notes: Optional[str] = None
    m7_notes: Optional[str] = None
    m8_notes: Optional[str] = None
    # v8 P2：把“重建证据 → 题级 scope → Skill 检索 → 首轮合成”的事实链
    # 与结果一起落盘，避免依赖不同 JSONL topic 的文件时间推断顺序。
    state_sequence: list[str] = []
    retrieved_skills: list[dict] = []
    # 注意：证据更新后重检索会**改写**这两个字段，它们表示"当前（最后一轮）的检索结果"。
    # 每一轮的选中/交付事实在 `retrieval_records` 里逐条保留（§13.6）。
    selected_skill_semvers: list[str] = []
    skill_mapping_misses: list[str] = []
    first_synthesis: dict = {}
    # ---- v9 §13.5/§13.6 检索与交付记录（P6）----
    # `retrieval_records` 是每次检索的完整记录（规范题型、evidence_version、候选及
    # 过滤原因、排序分数、选中版本、实际交付版本与正文 hash、配置版本）；下面三个
    # 清单是 §13.6 点名的四态分离口径：
    #   produced（n_skills_offered，在记录内）→ retrieved（检索选中）→
    #   delivered（实际送达模型）→ declared（模型自称，仅线索）。
    # "检索选中"与"已交付"**不得互替**：mock/C0/未发出请求时 retrieved 可以非空而
    # delivered 为空。
    retrieval_records: list[dict] = []
    retrieved_skill_versions: list[str] = []
    delivered_skill_versions: list[str] = []
    declared_selected_skill_versions: list[str] = []
    # v9 §9.4：主动图像账本（produced/delivered/observed 三态、每轮图像清单、
    # 图像布局与 token 成本）。"仅路径、ID 或成功标记不能算看过图" —— 三态与
    # 内容哈希都在这里。
    image_ledger: dict = {}


class EpisodeTrace(Spec):
    episode_id: str
    qa_id: str
    final_state: str
    program_trace_ref: str
    geometry_check_ref: str
    evaluation_ref: str
    failure: Optional[FailureTaxonomy]
    active_snapshot_ref: str
    active_snapshot_manifest_sha256: str = ""
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
    # v9 起该字段声明当前合同身份（见模块头 `EPISODE_TRACE_SCHEMA_VERSION`）。
    schema_version: str = EPISODE_TRACE_SCHEMA_VERSION
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
    # v9 §12.2：这一轮为何被生成 / 是否进入收口阶段（与 synthesis_source 分开）
    round_trigger: RoundTriggerValue = "initial"
    finalization_used: bool = False
    # v9 §10.1/§12：规范终态、答案依据（框架核验后）与工具归因台账。
    # `final_state` 保留历史碎片值（含 v6 的 unanswerable/abstain）供历史统计；
    # `episode_status` 是 §10.1 的三值规范终态。
    episode_status: str = ""
    answer_basis: str = ""
    answer: dict = {}
    attribution: dict = {}
    authorization_receipts: list[dict] = []
    # v9 §17.1「检索与 Round」层：轮数 / yield 次数 / 逐轮引用 / 预算快照
    agent_rounds: int = 0
    yield_count: int = 0
    round_trace_refs: list[str] = []
    # v9 §17.1 Round 层：逐轮事实（序号/触发/程序文本与哈希/本轮观测/结束方式）。
    rounds: list[dict] = []
    budget: dict = {}
    degenerate_regenerated: bool = False
    # v8 P2：可审计的阶段顺序与 Skill 输入快照。
    state_sequence: list[str] = []
    retrieved_skills: list[dict] = []
    selected_skill_semvers: list[str] = []
    skill_mapping_misses: list[str] = []
    first_synthesis: dict = {}
    # v9 §13.5/§13.6：检索与交付记录 + 四态分列的版本清单（见 TraceRecord 同名字段）
    retrieval_records: list[dict] = []
    retrieved_skill_versions: list[str] = []
    delivered_skill_versions: list[str] = []
    declared_selected_skill_versions: list[str] = []
    # v9 §9.4：主动图像账本（produced/delivered/observed 三态、每轮图像清单、
    # 图像布局与 token 成本）。"仅路径、ID 或成功标记不能算看过图" —— 三态与
    # 内容哈希都在这里。
    image_ledger: dict = {}


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
