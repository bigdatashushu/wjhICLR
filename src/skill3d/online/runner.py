"""在线推理链编排（§6.1 在线 FSM 的 driver）：把 M1–M13 接成一条可跑的链。

```
INGEST → INPUT_GATE → RECONSTRUCT → QUALITY_GATE → CLASSIFY_TASK → RETRIEVE_SKILL
→ SYNTHESIZE_PROGRAM → STATIC_CHECK → SANDBOX_EXECUTE → GEOMETRY_VERIFY
→ BENCHMARK_EVAL → ANSWER → LOG_TRACE
```

纪律（与 `系统架构.md` 对齐）：
- 硬约束 1：本模块在线，禁止任何 GPT-6 调用（静态守卫见 tests/unit/test_no_gpt6_online.py）；
- 硬约束 9：`final_test` 默认拒绝进在线链（需 `allow_final_test=True` 显式盲评）；
- 硬约束 14/17：program 只编排 REGISTRY 内 Tool，Tool 只经 SceneHandle 访问产物；
- 硬约束 13：Geometry Verifier 为确定性裁判，不引入第二 VLM；
- 诚实性：`mode="real"` 下缺权重/服务一律抛明确错误并把该 episode 记 `unavailable`；
  只有 `mode="mock_light"` 使用 `online/synthetic.py` 的合成输入与 stub program，
  且在 outcome.synthesis_source 中显式标注，绝不冒充当模型输出。

真值（ground_truth）只在 M12 评测与 M13 落盘时可见；prompt 构造签名刻意不含 GT（§4 M8）。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from skill3d.evaluation.accuracy import extract_option_letter, mca_correct
from skill3d.evaluation.mra import mra_single, parse_numeric_answer
from skill3d.fsm.online_fsm import OnlineFSM, OnlineState
from skill3d.gates.input_gate import annotate_frames, input_gate
from skill3d.reconstruction.vggt_runner import ReconstructionFailed, reconstruct
from skill3d.reconstruction_gate.quality_metrics import (
    QUALITY_METRIC_VERSION,
)
from skill3d.reconstruction_gate.evidence_profile import M5EvidenceSummary
from skill3d.reconstruction_gate.scene_state import (
    build_scene_state,
    scope_scene_to_question,
)
from skill3d.routing.retrieval_policy import RetrievalPolicy
from skill3d.routing.skill_retriever import (
    canonical_question_type,
    retrieval_decision,
    retrieve_ex,
)
from skill3d.routing.task_classifier import TaskClassification, canonical_task, classify
from skill3d.sandbox.ast_guard import ast_guard, normalize_program_source
from skill3d.sandbox.kernel import CellResult, RestrictedNamespaceKernel
from skill3d.sandbox.receipt import ReceiptChain, SandboxReceipt, verify_chain
from skill3d.schemas import (
    EpisodeInputTrace,
    EpisodeProgram,
    EpisodeTrace,
    EvaluationResultTrace,
    EvaluationRun,
    FailureTaxonomy,
    InputErrorRecord,
    InputGateVerdict,
    ProgramExecutionTrace,
    RetrievedSkill,
    SceneState,
    SkillRetrievalRecord,
    SkillSpecV11,
    VSIBenchEpisode,
    parse_answer_payload,
    verify_attribution,
)
from skill3d.schemas.image_ledger import LAYOUT_DERIVED_PLUS_ORIGINALS
from skill3d.schemas.retrieval import SkillCandidateRecord
from skill3d.skills.delivery import (
    SkillDeliveryPlan,
    plan_delivery,
    skill_body_length,
    skill_content_sha256,
    skill_version_key,
)
from skill3d.schemas.evidence import CAPABILITIES
from skill3d.synthesis.program_assembler import (
    SynthesisError,
    assemble_program,
    assemble_program_ex,
    degenerate_reason,
)
from skill3d.synthesis.prompt_builder import (
    PROMPT_TEMPLATE_VERSION,
    PromptBuilder,
)
from skill3d.synthesis.eval_visual_fallback import (
    EVAL_VISUAL_SPLITS,
    request_visual_answer,
)
from skill3d.synthesis.request_context import request_phase
from skill3d.tools import REGISTRY
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION
from skill3d.tools.mock_switch import MockSwitch
from skill3d.tools.scene_handle import SceneHandle
from skill3d.segmentation.open_vocab_detector import (
    capture_detector_attempts, current_detector_attempts,
)
from skill3d.online.recovery import (
    MAX_RECOVERY_ATTEMPTS,
    RecoveryPlan,
    build_feedback,
    cascade_invalidate,
    collect_validated,
    downgrade_profile,
    invalidate_results,
    premise_of_failure,
)
from skill3d.schemas.trace import EPISODE_TRACE_SCHEMA_VERSION, TraceRecord
from skill3d.trace.store import TraceStore
from skill3d.verifier.derivation import replay_derivation
from skill3d.verifier.geometry_oracle import (
    GeometryIssue,
    GeometryVerifyResult,
    geometry_verify,
)

from . import synthetic as synth
from .submission import (
    EXECUTION_PROTOCOL_VERSION,
    accepted_result_ids,
    invalid_reference_issues,
    rejected_submission,
    submission_scope,
)

# 确定性重放模式下的固定时间基准（§4 M8/M17 字节级一致；真实实验不启用）
_REPLAY_EPOCH = 0.0
_Path_ = Path
RuntimeSkillSpec = SkillSpecV11


@dataclass
class OnlineRunConfig:
    """在线链运行配置（阈值/路径一律来自 configs/*.yaml，见 online/config.py）。"""

    mode: str = "real"                      # real | mock_light
    baseline: str = "C1_tools_program"      # 固定的 v11 程序求解入口
    seed: int = 0
    deterministic_replay: bool = False      # 同 seed 字节级一致（时间/id 取确定性占位）
    active_snapshot_ref: str = "genesis"
    skills: list[RuntimeSkillSpec] = field(default_factory=list)
    max_regen: int = 3                      # TODO_CALIBRATE（configs/config.yaml）
    # v9 §5.1：M2 质量诊断默认关闭；仅独立诊断实验开启（开启后 quality_weight 参与
    # 合成质量，会改变 scene_route —— 那是诊断实验的自变量，不是默认口径）
    input_diagnostics: bool = False
    cell_timeout_s: int = 30                # TODO_CALIBRATE
    trace_dir: str = "data/traces"
    recon_dir: str = "data/reconstructions"
    work_dir: str = "data/reconstructions/mock_light"
    vllm_endpoints: list[str] = field(default_factory=list)
    vllm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    model_weights_sha256: str = field(
        default_factory=lambda: os.environ.get("SKILL3D_MODEL_WEIGHTS_SHA256", ""))
    environment_sha256: str = field(
        default_factory=lambda: os.environ.get("SKILL3D_ENVIRONMENT_SHA256", ""))
    recon_method: str = "vggt"              # v6 只允许 vggt（§5.2）
    # v6 D1/D2：零样本度量深度模型（首个 PoC = MoGe-2）。
    # 默认 None → 不跑融合，artifact 记 scale_fusion_status="not_run"、metric_scale=None。
    # 传入实现了 MetricDepthModel 协议的对象即启用（MoGe-2 权重到位后才可能真跑）。
    metric_depth_model: object = None
    metric_model_name: str = "none"          # moge2 | metric3d_v2 | none
    reuse_artifact: Optional[str] = None    # 复用既有 artifact JSON（硬约束 18：A/B 同源）
    max_tokens: int = 4096
    max_images: int = 32                    # M8 送进模型的最大帧数（与 §4 M1 对齐）
    max_pixels: int = 131072                # 实测起点：32 帧 ≈ 9.7k prompt tokens（A-4）
    max_model_len: int = 32768
    # ---- v6 度量证据门（D3）----
    # 融合成功才可能暴露米制 Tool；阈值在 reconstruction_gate.evidence_profile 内
    # （全 [TODO_CALIBRATE]），此处不再重复一份。
    # 总求解轮次包括最终作答轮；finalization 只保留收口轮次。工具调用不再
    # 维护独立总预算，工具是否可用由题级授权、证据和沙箱契约共同决定。
    max_solver_rounds: Optional[int] = None
    max_retries_per_operation: Optional[int] = None
    finalization_rounds: Optional[int] = None
    # 仅评测失败后追加一次干净视觉请求，不占原 solver 轮数；induction 硬禁用。
    eval_visual_fallback: bool = False
    # Content-addressed provenance for the active skill snapshot. The human
    # readable ref remains separate so pointer movement cannot hide lineage.
    active_snapshot_manifest_sha256: str = ""
    allow_final_test: bool = False          # 硬约束 9：默认拒绝 final_test 进在线链
    # v9 §13.5：top-k／排序权重／方法上下文上限的冻结策略（来自 configs/config.yaml
    # 的 `retrieval:` 段）。缺省对象 = 配置缺省值，检索记录里会如实记
    # `config_source="default"`，不会被当成"已冻结配置"。
    retrieval_policy: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    # v9 §9.4：主动图像的声明布局（派生图占位上限；图像上限 = max_images）
    image_layout: str = LAYOUT_DERIVED_PLUS_ORIGINALS
    max_derived_images: int = 8
    # v10 §8.2：候选效果评测的**固定注入**标记。非 None 时本 episode 的 Skill 输入
    # 完全由该绑定决定（跳过检索选择），且记录里带 `evaluation_binding`；
    # 该模式只能用于候选效果评测（不得用于 learning 经验采集 / 发布后运行 / 最终成绩）。
    evaluation_binding: Optional[object] = None
    def __post_init__(self) -> None:
        if self.mode not in {"real", "mock_light"}:
            raise ValueError("mode 只支持 real 或 mock_light")
        for value in (self.model_weights_sha256, self.environment_sha256):
            if value and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("模型/环境身份必须是 SHA256")
        if self.baseline != "C1_tools_program":
            raise ValueError("当前在线入口只支持 C1_tools_program")
        legacy = [type(skill).__name__ for skill in self.skills
                  if not isinstance(skill, SkillSpecV11)]
        if legacy:
            raise ValueError(
                "当前在线协议只接受 SkillSpecV11；历史 SkillSpec 请在对应 Git 提交运行"
                f"（收到 {legacy}）")
        solver_rounds = self.max_solver_rounds if self.max_solver_rounds is not None else 6
        final_rounds = self.finalization_rounds if self.finalization_rounds is not None else 1
        retry_budget = (self.max_retries_per_operation
                        if self.max_retries_per_operation is not None
                        else MAX_RECOVERY_ATTEMPTS)
        self.max_solver_rounds = max(1, int(solver_rounds))
        self.finalization_rounds = max(0, min(
            int(final_rounds), self.max_solver_rounds - 1
        ))
        self.max_retries_per_operation = max(0, int(retry_budget))
        # v10 §8.2：固定注入臂必须恰好一条完整 Skill —— 多于一条就等于"同时提供多个
        # Skill"，§8.3 明确禁止；零条等于"空 Skill 基线"，同样禁止（EV-03）。
        if self.evaluation_binding is not None:
            if len(self.skills) != 1:
                raise ValueError(
                    "evaluation_binding 要求恰好注入一条完整 Skill（§8.2），"
                    f"收到 {len(self.skills)} 条")
            binding = self.evaluation_binding
            version = str(getattr(binding, "skill_version", "")
                          or (binding or {}).get("skill_version", ""))
            spec = self.skills[0]
            if version != f"{spec.skill_id}@{spec.version}":
                raise ValueError(
                    f"固定注入绑定的版本 {version!r} 与被注入的 Skill "
                    f"{spec.skill_id}@{spec.version} 不一致（§8.2）")


@dataclass
class PreparedObjectBinding:
    """Frozen M5 input supplied by a paired driver; consumed as an episode-local copy.

    File references must already point at this episode's private artifact tree.
    The ordinary online path still runs M5 when this argument is absent.
    """

    objects: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    materialized: bool = False


@dataclass
class EpisodeOutcome:
    """单 episode 的完整结果（供评测聚合与审计）。"""

    qa_id: str
    scene_name: str
    question_type: str
    final_state: str                        # answer | input_error | run_error | unavailable
    answer: Optional[str]
    predicted: Optional[str]
    is_mca: bool
    correct: Optional[bool]
    mra_value: Optional[float]
    task: str = ""                          # §4 M7 规范题型（8 类之一，聚合用）
    states: list[str] = field(default_factory=list)
    answer_flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    synthesis_source: str = "none"          # vllm | deterministic_stub | none
    program: Optional[EpisodeProgram] = None
    program_trace: Optional[ProgramExecutionTrace] = None
    verify: Optional[GeometryVerifyResult] = None
    episode_trace: Optional[EpisodeTrace] = None
    receipts: list[SandboxReceipt] = field(default_factory=list)
    receipts_ok: bool = True
    scene_route: Optional[str] = None
    question_tool_scope: str = ""
    g9_tracker_consistency: Optional[float] = None  # G9 SAM2 mask IoU 均值（M5 产物）
    # v6：G5 永久 not_available（D11）；G8 永久退役 —— 都不存在替代门
    reprojection_status: str = "not_available"
    # M2 被动观测的降级 flag（只打权重/标记，不改帧集，硬约束 21）
    input_degradation_flags: list[str] = field(default_factory=list)
    # HC26：M8 实际收到的图像数（必须等于统一 FrameSet 帧数，禁止静默丢帧）
    n_images_to_synthesizer: int = 0
    # `room_size_estimation` medium 档的三态平面质量输入（TODO_CALIBRATE：阈值未标定）。
    # None = 证据缺失 → fail-closed 不授权（不得默认放行）。
    plane_quality_ok: Optional[bool] = None
    # 答案来源（v6 §5.3 四值）：tool_program / direct_vlm_routed / abstain / tool_contract
    answer_source: str = ""
    # v9 §10.1：规范 episode 终态（answered/input_error/run_error），由 final_state 显式映射
    episode_status: str = ""
    input_error_reason: str = ""
    # v9 §12：答案载荷、核验后的 basis、框架核验台账（声明/核实/忽略三分离）
    answer_payload: Optional[Any] = None
    answer_basis: str = ""
    attribution: Optional[Any] = None
    # --- D-3 / 硬约束 22/23：契约与质量的显式记录（过程指标与归因用）---
    quality_status: str = "not_computed"
    overall_quality: Optional[float] = None
    main_gate_passed: Optional[bool] = None   # M4 主门通过与否（§10.1 交叉双指标）
    tool_contract_hits: int = 0             # 本 episode 命中的 tool_contract 次数
    replay_used: bool = False               # 是否用过"回灌一次"
    trimmed_regen_used: bool = False        # 是否用过裁剪 prompt 重生成
    # v6 D7：partial_tool_recovery 事实（§14.1）
    recovery_count: int = 0
    partial_tool_recovery: bool = False
    # v7 §11：多轮执行事实（yield 次数 / 轮次引用 / 是否进入 finalization）
    yield_count: int = 0
    agent_rounds: int = 0
    round_trace_refs: list[str] = field(default_factory=list)
    finalization_used: bool = False
    # v9 §17.1「检索与 Round」层：逐轮事实（序号/触发/程序文本与哈希/本轮观测）。
    # 单个 `program_trace` 只保留最后一轮，无法回答"早先轮次做了什么"。
    rounds: list[dict] = field(default_factory=list)
    static_check_errors: list[str] = field(default_factory=list)
    # v9 §6.4：本 episode 每次实际调用的授权收据（含被拒绝的调用）
    authorization_receipts: list[dict] = field(default_factory=list)
    # v9 §12.2：本 episode 最后一轮的触发原因（initial/observation/error_recovery/finalize）。
    # 与 synthesis_source 分开：来源讲"程序文本从哪来"，触发讲"这一轮为什么生成"。
    round_trigger: str = "initial"
    used_result_ids: list[str] = field(default_factory=list)
    invalidated_result_ids: list[str] = field(default_factory=list)
    failure_code: Optional[str] = None
    terminal_failure: dict = field(default_factory=dict)
    eval_visual_fallback: dict = field(default_factory=dict)
    detector_attempts: list[dict] = field(default_factory=list)
    degenerate_regenerated: bool = False    # 退化输出触发过重生成（§15.3）
    # v6 §5.9 P8：分阶段判读信息（M5/M7/M8），供 TraceRecord 归因
    m5_notes: list[str] = field(default_factory=list)
    m7_notes: list[str] = field(default_factory=list)
    m8_notes: list[str] = field(default_factory=list)
    cache_hit: bool = False                 # M5 场景清单是否命中缓存（§5.9）
    abstained: bool = False                 # 显式 abstain（主榜按错计，不刷分）
    answer_untrusted: bool = False          # 答案依赖过契约失败的 Tool → 不得采纳
    frame_set_hash: str = ""
    scene_summary: str = ""                 # 送给 M8 的场景摘要（错误归因用）
    m8_prompt: str = ""                     # M8 文本 prompt（不含图像；错误归因用）
    # v8 P2：检索与首轮合成的可审计输入快照。
    retrieved_skills: list[dict] = field(default_factory=list)
    selected_skill_semvers: list[str] = field(default_factory=list)
    skill_mapping_misses: list[str] = field(default_factory=list)
    # v9 §13.5/§13.6：每次检索的完整记录（候选/过滤原因/分数/选中/交付/配置版本），
    # 以及四态分列的版本清单（produced → retrieved → delivered → declared）。
    retrieval_records: list[Any] = field(default_factory=list)
    # v9 §9.4：本 episode 的图像账本（produced/delivered/observed 三态 + 每轮清单）
    image_ledger: dict = field(default_factory=dict)
    retrieved_skill_versions: list[str] = field(default_factory=list)
    delivered_skill_versions: list[str] = field(default_factory=list)
    declared_selected_skill_versions: list[str] = field(default_factory=list)
    first_synthesis: dict = field(default_factory=dict)
    scale_source: str = ""                  # 尺度来源标注（mock_light/合成路径用）
    artifact_ref: str = ""                  # 本 episode 用的重建产物 ref（硬约束 18 审计）
    # --- v6 证据（D5/D1/D3/D6）：逐 episode 事实，供 §18/§19 报告与审计 ---
    world_frame_status: str = "unavailable"
    world_up: Optional[list] = None
    handedness: Optional[str] = None
    scale_fusion_status: str = "not_run"
    metric_scale: Optional[float] = None
    scale_self_consistency: Optional[float] = None
    metric_model: str = "none"
    metric_fusion_version: str = ""
    per_frame_scale_ref: Optional[str] = None
    evidence_profile: Optional[object] = None
    metric_evidence_gate_result: Optional[object] = None
    m5_summary: Optional[object] = None
    authorized_metric_tasks: list[str] = field(default_factory=list)  # 本题授权后


class FixedSkillInjectionError(RuntimeError):
    """v10 §8.2：固定注入的评测臂无效（Skill 未过硬条件 / 注入条数不等于 1）。

    评测臂无效必须**吵出来**：静默跑成"无 Skill 基线"会让 A/B 比较失去意义
    （v10 §1.2 EV-03 正是这个错）。
    """


@dataclass
class _SynthResult:
    """M8 program generation result."""
    program: Optional[EpisodeProgram]
    # §19.3 六类 + `mock_stub`：vllm_ok | vllm_parse_error | vllm_service_error |
    # m8_parse_recovered | direct_answer_fallback | partial_tool_recovery | mock_stub
    source: str
    note: str
    response_text: str = ""  # 模型原始回复；不得用解析后的 program_source 回填
    prompt: str = ""        # M8 文本 prompt（错误归因用；图像不入日志）
    # HC26：实际送进模型的多模态图像数（必须等于统一 FrameSet 帧数；0 表示纯文本，
    # 在 real 模式下不允许）。trace 落该值，便于事后核对"没有静默丢帧"。
    n_images: int = 0
    # v9 §13.6：本次合成的**方法交付计划**（哪些方法进了 prompt、哪些被上限丢弃、
    # 请求是否真的发出）。只有 `channel="model_request"` 的条目算"已交付"。
    delivery: Optional[SkillDeliveryPlan] = None
    # v9 §9.4：本轮图像清单（布局、实际装入的图像、被省略的图与 token 成本）
    image_round: dict = field(default_factory=dict)


class _Counter:
    """确定性 id 生成器（重放模式）。"""

    def __init__(self) -> None:
        self.n = 0

    def __call__(self) -> str:
        self.n += 1
        return f"{self.n:08d}"


def _receipt_chain(episode_id: str, cfg: OnlineRunConfig) -> ReceiptChain:
    if cfg.deterministic_replay:
        return ReceiptChain(episode_id, clock=lambda: _REPLAY_EPOCH, id_factory=_Counter())
    return ReceiptChain(episode_id)


def _failure_of(outcome: EpisodeOutcome, episode: VSIBenchEpisode) -> Optional[FailureTaxonomy]:
    """失败归因（仅失败时非空，§5.7 FailureTaxonomy）。categories 取受控枚举（v6）。"""
    eid = episode.qa_id
    if outcome.final_state in ("answer", "answer_best_effort"):
        return None
    if outcome.final_state == "input_error":
        return FailureTaxonomy(
            episode_id=eid, categories=["input_error"],
            note=("输入缺失或全部图像不可读；未请求模型，按 0 分保留在评测分母；"
                  f"reason={outcome.input_error_reason!r}"))
    if outcome.final_state == "run_error":
        return FailureTaxonomy(
            episode_id=eid, categories=["run_error"],
            note=("在线执行或终结阶段失败，按 0 分保留在评测分母；"
                  f"failure_code={outcome.failure_code!r}"))
    if outcome.final_state == "unavailable":
        return FailureTaxonomy(
            episode_id=eid, categories=["synthesis"],
            note="模型/服务不可用 → episode 记 unavailable（服务故障记 "
                 "service_unavailable，见 §6.1）")
    if outcome.abstained or "tool_contract" in outcome.answer_flags:
        return FailureTaxonomy(
            episode_id=eid, categories=["tool_contract"],
            note=("Tool 契约违规（产物/证据缺失、证据门未过、域值错误）→ 显式 abstain，"
                  f"主榜按错计；hits={outcome.tool_contract_hits}，"
                  f"partial_recovery={outcome.partial_tool_recovery}，"
                  f"recovery_count={outcome.recovery_count}"))
    if outcome.metric_evidence_gate_result is not None \
            and not outcome.metric_evidence_gate_result.gate_passed \
            and outcome.answer_source == "direct_vlm_routed":
        return FailureTaxonomy(
            episode_id=eid, categories=["metric_evidence"],
            note=("米制证据门未通过 → 显式 direct_vlm_routed（§13.3）；缺失子条件="
                  f"{sorted(outcome.metric_evidence_gate_result.missing_subconditions)}"))
    if outcome.scene_route == "fallback_2d_only":
        return FailureTaxonomy(
            episode_id=eid, categories=["reconstruction"],
            note="重建质量不足（M4 主门未过）或输入降级 → 受限 Tool 集（§6.2）")
    return None


def _trace_record_of(outcome: EpisodeOutcome, episode: VSIBenchEpisode,
                     cfg: OnlineRunConfig) -> TraceRecord:
    """§5.9 TraceRecord：版本字段 + 证据/路由全状态（D10）。

    目的：**不依赖重跑即可归因失败** —— 每个失败码、每个证据三值、
    每条 used_result_id 都在这里，评审只需读 trace。
    """
    from skill3d.reconstruction.metric_fusion import METRIC_FUSION_VERSION
    from skill3d.schemas.evidence import PROFILE_VERSION
    from skill3d.tools.registry import TOOL_FACE_VERSION

    profile = outcome.evidence_profile
    gate = outcome.metric_evidence_gate_result
    params = {}
    try:
        from skill3d.tools.distance_primitives import DistancePrimitiveParams

        params = DistancePrimitiveParams().snapshot()
    except Exception:  # noqa: BLE001 - 参数快照缺失不得阻断 trace
        params = {}
    return TraceRecord(
        episode_id=episode.qa_id,
        active_snapshot_manifest_sha256=str(
            cfg.active_snapshot_manifest_sha256 or ""),
        template_version=PROMPT_TEMPLATE_VERSION,
        execution_protocol_version=EXECUTION_PROTOCOL_VERSION,
        tool_face_version=TOOL_FACE_VERSION,
        tool_docs_version=TOOL_DOCS_VERSION,
        evidence_profile_version=(getattr(profile, "profile_version", "")
                                  or PROFILE_VERSION),
        gate_version=(getattr(gate, "gate_version", "") or ""),
        distance_primitive_params=params,
        metric_fusion_version=str(outcome.metric_fusion_version or
                                  (METRIC_FUSION_VERSION if outcome.metric_model != "none"
                                   else "")),
        evidence_profile=(profile.model_dump() if profile is not None else None),
        metric_evidence_gate_result=(gate.model_dump() if gate is not None else None),
        scene_route=str(outcome.scene_route or ""),
        question_tool_scope=str(outcome.question_tool_scope or ""),
        answer_source=str(outcome.answer_source or ""),
        input_error_reason=str(outcome.input_error_reason or ""),
        used_result_ids=sorted(set(outcome.used_result_ids or [])),
        recovery_count=int(outcome.recovery_count),
        partial_tool_recovery=bool(outcome.partial_tool_recovery),
        invalidated_result_ids=sorted(set(outcome.invalidated_result_ids or [])),
        synthesis_source=_synthesis_source_enum(outcome.synthesis_source),
        round_trigger=str(outcome.round_trigger or "initial"),
        finalization_used=bool(outcome.finalization_used),
        episode_status=_episode_status_of(outcome.final_state),
        answer_basis=str(outcome.answer_basis or ""),
        answer=(outcome.answer_payload.model_dump()
                if outcome.answer_payload is not None else {}),
        derivation_replay=_final_derivation_replay(outcome),
        attribution=(outcome.attribution.model_dump()
                     if outcome.attribution is not None else {}),
        authorization_receipts=[dict(r) for r in (outcome.authorization_receipts or [])],
        agent_rounds=int(outcome.agent_rounds),
        yield_count=int(outcome.yield_count),
        round_trace_refs=list(outcome.round_trace_refs or []),
        rounds=[dict(r) for r in (outcome.rounds or [])],
        budget=_budget_receipt(cfg),
        n_objects=int((outcome.m5_summary.n_objects
                       if outcome.m5_summary is not None else 0)),
        cache_hit=bool(outcome.cache_hit),
        failure_code=outcome.failure_code,
        terminal_failure=dict(outcome.terminal_failure),
        eval_visual_fallback=dict(outcome.eval_visual_fallback),
        detector_attempts=list(outcome.detector_attempts),
        m5_notes=_joined(outcome.m5_notes),
        m7_notes=_joined(outcome.m7_notes),
        m8_notes=_joined(outcome.m8_notes),
        state_sequence=list(outcome.states or []),
        retrieved_skills=[dict(x) for x in (outcome.retrieved_skills or [])],
        selected_skill_semvers=list(outcome.selected_skill_semvers or []),
        skill_mapping_misses=list(outcome.skill_mapping_misses or []),
        first_synthesis=dict(outcome.first_synthesis or {}),
        # v9 §9.4：主动图像三态（produced/delivered/observed）+ 每轮图像清单
        image_ledger=dict(outcome.image_ledger or {}),
        # v9 §13.5/§13.6：每次检索的完整记录 + 四态分列的版本清单
        retrieval_records=[r.model_dump() if hasattr(r, "model_dump") else dict(r)
                           for r in (outcome.retrieval_records or [])],
        retrieved_skill_versions=list(outcome.retrieved_skill_versions or []),
        delivered_skill_versions=list(outcome.delivered_skill_versions or []),
        declared_selected_skill_versions=list(
            outcome.declared_selected_skill_versions or []),
    )


# §6.3/§14：走 partial_tool_recovery 的 ToolContractError 族（`ToolContractError` 的四个子类）
_RECOVERABLE_CONTRACT_ERRORS = ("tool_contract", "confidence_gate", "domain_value",
                                "answer_already_given")

_SYNTH_SOURCES = {"vllm_ok", "vllm_parse_error", "vllm_service_error",
                  "m8_parse_recovered", "direct_answer_fallback",
                  "partial_tool_recovery", "mock_stub"}


def _synthesis_source_enum(value: str) -> str:
    """把历史/宽松取值归一到 §19.3 的 6 类（+ v6 显式的 `mock_stub`）。

    - `mock_stub`（mock_light 确定性 stub，非模型输出）原样保留 —— v5 曾把它写成
      `deterministic_stub`，这里做别名归一；
    - 空 / `none` / `unavailable`（M8 根本没跑）→ `vllm_service_error`：**只允许**
      断言"没有拿到模型输出"；
    - 其他不认识的取值 → 空串（"来源未知"）。

    **绝不**返回 `vllm_ok`。v5 的 vllm 混写正是 §19.3 要消灭的问题；v8 把
    `forced_answer` / `finalization` 塞进 `synthesis_source` 之后，又被这条兜底
    洗成"模型正常产出" —— mock_light 的收口答案因此在 TraceRecord 里冒充过
    真实模型输出。轮次原因改记在 `round_trigger` / `finalization_used`（§12.2）。
    """
    v = str(value or "").strip()
    if v in _SYNTH_SOURCES:
        return v
    if v == "deterministic_stub":        # v5 历史名 → v6 的 mock_stub
        return "mock_stub"
    if v in ("", "none", "unavailable"):
        return "vllm_service_error"
    return ""


def _round_record(
    *,
    index: int,
    trigger: str,
    source: str,
    program,
    program_trace,
    response_text: str = "",
) -> dict:
    """§17.1 Round 层的逐轮事实（v9 `ProgramRound` 的当前落地子集）。

    记录：轮次序号与触发原因、程序来源、实际执行的程序文本与哈希、本轮产生的
    result_id、本轮工具调用数、结束方式。**不**记录图像布局 —— 图像回灌链尚未
    实现（§10.1 的 `selected_image_refs` 仍是缺口），不在这里假装有。
    """
    source_text = str(getattr(program, "program_source", "") or "")
    results = list(getattr(program_trace, "results", []) or [])
    return {
        "index": int(index),
        "trigger": str(trigger or "initial"),
        "synthesis_source": str(source or ""),
        "program_id": str(getattr(program, "program_id", "") or ""),
        "program_sha256": (hashlib.sha256(source_text.encode("utf-8")).hexdigest()
                           if source_text else ""),
        "program_source": source_text,
        "response_text": str(response_text),
        "observed_result_ids": [str(getattr(r, "result_id", "") or "")
                                for r in results if str(getattr(r, "result_id", "") or "")],
        "tool_calls": len(results),
        "error_code": str(getattr(program_trace, "error_code", "") or ""),
        "service_errors": [
            {"tool": r.tool, "error": (r.payload or {}).get("error", "service fault")}
            for r in results
            if (r.payload or {}).get("status") == "fault"
            or (r.payload or {}).get("service_healthy") is False
        ],
    }


def _evidence_version_of_scene(scene) -> str:
    """当前 EvidenceProfile 版本（§13.6 检索记录要记 `evidence_version`）。"""
    profile = getattr(scene, "evidence_profile", None) if scene is not None else None
    return str(getattr(profile, "profile_version", "") or "")


def _evidence_state_snapshot(scene) -> tuple[dict, dict]:
    """证据三值快照 + 原因码（§13.6 检索记录的可比对证据事实）。

    `profile_version` 是**合同版本**：级联降级前后它不变，所以"证据是否真的变了"
    不能靠版本号判断，必须另记三值快照。
    """
    profile = getattr(scene, "evidence_profile", None) if scene is not None else None
    if profile is None:
        return {}, {}
    states = {cap: str(getattr(profile, cap, "") or "") for cap in CAPABILITIES}
    return states, dict(getattr(profile, "state_reasons", None) or {})


def _fixed_injection_retrieval(episode, scene, cfg: OnlineRunConfig,
                               outcome: EpisodeOutcome, *, trigger: str = "initial",
                               retrieval_index: int = 1) -> list[RuntimeSkillSpec]:
    """§8.2 固定注入臂：跳过**检索选择**，不跳过适用性 / 长度 / 权限 / Schema 校验。

    规范原文（§8.2）："评测目标 Skill 时跳过检索选择，但不跳过 Skill 的适用性、
    内容长度、工具权限和 Schema 校验"；§8.3 禁止"A 使用空 Skill、B 使用候选 Skill"。

    因此这里：

    - 把被评测 Skill 的硬条件判定（题型 / 证据签名 / 米制 gate / 状态）**照跑一遍**：
      不通过就抛 `FixedSkillInjectionError`（评测臂无效要吵出来，不能静默退化成
      "无 Skill 基线" —— 那正是 v10 EV-03 的错）；
    - 记录里显式写 `evaluation_binding`，候选行原因码是 `fixed_injection_*`，
      **不得**写成 `hit`（§8.3：不得把固定注入记录成正常检索命中）。
    """
    skills = list(cfg.skills or [])
    if len(skills) != 1:
        raise FixedSkillInjectionError(
            f"固定注入臂必须恰好一条完整 Skill，收到 {len(skills)} 条"
            "（§8.2：每次模型请求只包含该臂被评测的一条完整 Skill）")
    skill = skills[0]
    task_type = canonical_question_type(getattr(episode, "question_type", ""))
    decision = retrieval_decision(skill, scene, task_type or "")
    row = SkillCandidateRecord(
        skill_id=str(skill.skill_id),
        version=str(skill.version),
        skill_version=skill_version_key(skill),
        canonical_question_type=str(task_type or ""),
        hard_filter_passed=bool(decision.ok),
        reason_code=("fixed_injection_selected" if decision.ok
                     else "fixed_injection_not_applicable"),
        reason=("§8.2 固定注入（跳过检索选择，适用性校验通过）" if decision.ok
                else f"§8.2 固定注入的 Skill 未通过适用性校验: {decision.reason}"),
        matched_evidence_signature=dict(decision.matched_signature),
        gate_version_matched=decision.gate_version_matched,
        content_sha256=skill_content_sha256(skill),
        content_chars=skill_body_length(skill),
        delivered=False,
        delivery_reason="not_selected" if not decision.ok else "no_model_request",
    )
    record = SkillRetrievalRecord(
        retrieval_index=int(retrieval_index),
        trigger=str(trigger),
        canonical_question_type=str(task_type or ""),
        question_type_raw=str(getattr(episode, "question_type", "") or ""),
        question_type_known=bool(task_type),
        evidence_version=str(
            getattr(getattr(scene, "evidence_profile", None), "profile_version", "") or ""),
        config_version=cfg.retrieval_policy.version(),
        config_sha256=cfg.retrieval_policy.sha256(),
        config_source=str(cfg.retrieval_policy.source),
        policy=cfg.retrieval_policy.to_dict(),
        active_snapshot_ref=str(cfg.active_snapshot_ref or ""),
        active_snapshot_manifest_sha256=str(cfg.active_snapshot_manifest_sha256 or ""),
        n_skills_offered=1,
        candidates=[row],
        eligible_skill_versions=([row.skill_version] if decision.ok else []),
        retrieved_skill_versions=([row.skill_version] if decision.ok else []),
        delivery_channel="not_sent",
        delivery_note="§8.2 固定注入：等待模型请求，尚未交付",
    )
    if decision.ok:
        row.selected = True
    record.mark_fixed_injection(cfg.evaluation_binding)
    states, reasons = _evidence_state_snapshot(scene)
    record.evidence_states = states
    record.evidence_state_reasons = reasons
    outcome.retrieval_records.append(record)
    outcome.retrieved_skills = (
        [_hit_from_decision(skill, decision).model_dump()] if decision.ok else [])
    outcome.selected_skill_semvers = ([skill_version_key(skill)] if decision.ok else [])
    outcome.retrieved_skill_versions = list(record.retrieved_skill_versions)
    if not decision.ok:
        raise FixedSkillInjectionError(
            "§8.2 固定注入的 Skill 未通过适用性校验，评测臂无效"
            f"（{skill_version_key(skill)}: {decision.reason}）")
    return [skill]


def _hit_from_decision(skill: SkillSpecV11, decision) -> RetrievedSkill:
    """固定注入臂的 `RetrievedSkill` 记录（硬条件通过；排序分数不适用）。"""
    return RetrievedSkill(
        skill_id=str(skill.skill_id),
        skill_version=skill_version_key(skill),
        score=1.0,
        hard_filter_passed=True,
        matched_evidence_signature=dict(decision.matched_signature),
        gate_version_matched=decision.gate_version_matched,
    )


def _retrieve_for_episode(episode, scene, cfg: OnlineRunConfig, outcome: EpisodeOutcome,
                          *, trigger: str = "initial",
                          llm=None) -> list[RuntimeSkillSpec]:
    """§13.5/§13.6：检索一次并把**完整记录**落进 outcome；返回可用 SkillSpec 列表。

    `trigger="evidence_update"` 是 §13.5 的"证据更新后可在**同一快照**中重检索，
    更新实际交付记录"——重检索只换输入证据，**不**发布新库（`cfg.skills` 不变），
    因此本函数没有重新加载快照的路径。

    `cfg.evaluation_binding` 非空时走固定注入分支；普通运行按规范题型确定唯一
    `SkillSpecV11`。空列表是合法的 B01 baseline。
    """
    index = len(outcome.retrieval_records) + 1
    if cfg.evaluation_binding is not None:
        return _fixed_injection_retrieval(episode, scene, cfg, outcome,
                                          trigger=trigger, retrieval_index=index)
    states, reasons = _evidence_state_snapshot(scene)
    retrieved, record = retrieve_ex(
        episode.question, scene, cfg.skills,  # type: ignore[arg-type]
        question_type=episode.question_type,
        policy=cfg.retrieval_policy,
        trigger=trigger,
        retrieval_index=index,
        evidence_version=_evidence_version_of_scene(scene),
        snapshot_ref=str(cfg.active_snapshot_ref or ""),
        snapshot_manifest_sha256=str(cfg.active_snapshot_manifest_sha256 or ""))
    record.evidence_states = states
    record.evidence_state_reasons = reasons
    by_key = {f"{s.skill_id}@{s.semver}": s for s in cfg.skills}
    selected = [by_key[key] for key in record.retrieved_skill_versions
                if key in by_key]
    outcome.retrieval_records.append(record)
    outcome.retrieved_skills = [r.model_dump() for r in retrieved]
    outcome.selected_skill_semvers = [f"{s.skill_id}@{s.semver}" for s in selected]
    outcome.skill_mapping_misses = [key for key in record.retrieved_skill_versions
                                    if key not in by_key]
    return selected


def _skill_usage_clues(program_source: str, skills: Sequence[RuntimeSkillSpec],
                       delivered: Sequence[str]) -> list[dict]:
    """§13.6"可观察的程序使用线索"（**机械交叉检查**，不是因果证明）。

    只做两件能确证的事：

    1. `declared_in_program`：程序文本里**字面出现**了哪些已交付方法（模型自称）；
    2. `template_tool_overlap`：已交付方法的模板里点名的 Tool，与程序源码里字面出现的
       Tool 名的交集（"程序用了模板建议的工具"这一条可观测线索）。

    两者都**不**证明方法带来了收益（§13.6：模型自称不能当作方法成功或因果贡献的
    充分证明），因此字段名叫 clue / declared，不叫 used / contributed。
    """
    text = str(program_source or "")
    program_tools = {t for t in REGISTRY.names() if t in text}
    out: list[dict] = []
    for skill in skills or []:
        key = f"{skill.skill_id}@{skill.version}"
        if key not in set(delivered):
            continue
        mentioned = key in text or str(skill.skill_id) in text
        template = str(
            getattr(skill, "skill_md", None)
            or getattr(skill, "call_graph_template", "")
            or "")
        template_tools = sorted({t for t in REGISTRY.names() if t in template})
        overlap = sorted(set(template_tools) & program_tools)
        out.append({
            "skill_version": key,
            "clue_source": "static_scan",
            "declared_in_program": bool(mentioned),
            "template_tools": template_tools,
            "template_tool_overlap": overlap,
            "note": "字面交叠只是线索，不构成方法成功或因果贡献的证明（§13.6）",
        })
    return out


def _record_skill_delivery(outcome: EpisodeOutcome, delivery: Optional[SkillDeliveryPlan],
                           *, round_index: int, skills: Sequence[RuntimeSkillSpec],
                           program=None) -> None:
    """§13.5/§13.6：把一次合成的交付事实并进**最近一次**检索记录。

    - `delivered_skill_versions` / `delivered_content_sha256`：只有请求真正发出才算；
    - 因上下文上限被整条丢弃的条目记 `context_cap_exceeded`；
    - `declared_selected_skill_versions`：模型在程序文本里自称使用的方法；
    - 每轮短方法摘要（§13.6）落进记录的 `method_summaries`。
    """
    if delivery is None:
        return
    record: Optional[SkillRetrievalRecord] = (
        outcome.retrieval_records[-1] if outcome.retrieval_records else None)
    if record is None:
        return
    skills_by_key = {f"{s.skill_id}@{s.version}": s for s in skills or []}
    program_source = str(getattr(program, "program_source", "") or "")
    delivered = list(delivery.delivered_skill_versions)
    clues = _skill_usage_clues(program_source, list(skills or []), delivered) \
        if program_source else []
    record.add_delivery(delivery, round_index=round_index,
                        skills_by_key=skills_by_key, usage_clues=clues)
    declared = sorted({c["skill_version"] for c in clues if c.get("declared_in_program")})
    if declared:
        record.declared_selected_skill_versions = sorted(
            set(record.declared_selected_skill_versions) | set(declared))
    outcome.delivered_skill_versions = sorted(
        set(outcome.delivered_skill_versions) | set(delivered))
    outcome.declared_selected_skill_versions = sorted(
        set(outcome.declared_selected_skill_versions)
        | set(record.declared_selected_skill_versions))


def _retrieval_summary_lines(record: SkillRetrievalRecord) -> list[str]:
    """检索记录的人类可读摘要（进 notes / m7_notes，便于日志排障）。"""
    lines = [
        f"M7 检索#{record.retrieval_index}[{record.trigger}] 题型={record.canonical_question_type or '未知'}"
        f" 候选={record.n_skills_offered} 通过硬条件={len(record.eligible_skill_versions)}"
        f" 选中={len(record.retrieved_skill_versions)}"
        f" 配置版本={record.config_version}({record.config_source})"
    ]
    for row in record.candidates:
        if not row.hard_filter_passed:
            lines.append(f"M7 候选 {row.skill_version} 被拦下：{row.reason_code}（{row.reason}）")
    return lines


def _episode_status_of(final_state: str) -> str:
    """§10.1 `EpisodeStatus`：`answered` / `input_error` / `run_error`。

    历史 `final_state` 取值更碎（含 v6 的 `unanswerable` / `abstain` 拒答终态），
    按 §10.1"`unanswerable`、`abstain` 不再是本版的正常 episode 终态"做显式映射，
    并保留原值供历史统计（拒绝"只改标签"的静默迁移）。
    """
    state = str(final_state or "")
    if state in ("answer", "answer_best_effort"):
        return "answered"
    if state == "input_error":
        return "input_error"
    return "run_error"


def _record_answer_attribution(outcome: EpisodeOutcome, episode, kernel, answer: str,
                               *, payload_override=None) -> None:
    """§10.1/§12：记录答案载荷与**框架核验后**的工具归因。

    - 载荷优先取 `ReturnAnswer` 收到的原始对象；没有（如 C0 直答路径）则按 §12
      记 `visual_estimate`（由原图作答、未使用工具事实）；
    - 归因用本 episode 的工具台账核验；无法充分证实时保守降级为 `mixed`，
      **不否决格式合法的预测**；
    - 历史短写经适配器转换的问题记录一并保留。
    """
    slot = getattr(kernel, "answer_slot", None) if kernel is not None else None
    payload = (payload_override if payload_override is not None
               else getattr(slot, "payload", None))
    adapter_problems: list[str] = (
        [] if payload_override is not None
        else list(getattr(slot, "adapter_problems", []) or []))
    if payload is None:
        payload, extra = parse_answer_payload(
            answer, question_type=str(getattr(episode, "question_type", "") or ""))
        adapter_problems.extend(extra)

    results = list(getattr(kernel, "tool_results", []) or []) if kernel is not None else []
    calls = list(getattr(kernel, "tool_calls", []) or []) if kernel is not None else []
    observed = [str(getattr(r, "result_id", "") or "") for r in results]
    succeeded = [str(getattr(r, "result_id", "") or "") for r in results
                 if str(getattr(r, "status", "ok")) == "ok"]

    verification = verify_attribution(
        payload,
        observed_result_ids=[r for r in observed if r],
        succeeded_result_ids=[r for r in succeeded if r],
        invalidated_result_ids=list(outcome.invalidated_result_ids or []),
        attempted_tool_calls=len(calls),
        succeeded_tool_calls=len([r for r in results
                                  if str(getattr(r, "status", "ok")) == "ok"]))
    if adapter_problems:
        verification = verification.model_copy(
            update={"problems": list(verification.problems) + adapter_problems})

    outcome.answer_payload = payload
    outcome.answer_basis = verification.verified_basis
    outcome.attribution = verification
    for p in verification.problems:
        outcome.m7_notes.append(f"§12 归因: {p}")


def _budget_receipt(cfg: OnlineRunConfig) -> dict:
    """轮数/重试/收口预算快照（§17.1「检索与 Round」层要求轮数与重试可审计）。

    只记**配置上限**，不记"用掉多少"——消耗量由 `agent_rounds` / `yield_count` /
    `recovery_count` 单独落盘，避免同一事实两处来源。
    """
    return {
        "max_solver_rounds": int(cfg.max_solver_rounds or 0),
        "finalization_rounds": int(cfg.finalization_rounds or 0),
        "max_retries_per_operation": int(cfg.max_retries_per_operation or 0),
        "max_regen": int(cfg.max_regen),
        "eval_visual_fallback_max_requests": int(cfg.eval_visual_fallback),
    }


def _final_derivation_replay(outcome: EpisodeOutcome) -> dict:
    if outcome.eval_visual_fallback.get("accepted"):
        return dict(outcome.eval_visual_fallback["derivation_replay"])
    return dict(outcome.verify.derivation_replay) if outcome.verify is not None else {}


def _joined(notes: Optional[list[str]]) -> Optional[str]:
    """把逐阶段判读信息压成一行（§5.9 P8：不依赖重跑即可归因失败）。"""
    if not notes:
        return None
    return " | ".join(str(x) for x in notes if str(x).strip()) or None


def _sync_scene_snapshot(outcome: EpisodeOutcome, scene: Optional[SceneState],
                         task: str = "") -> None:
    if scene is None:
        return
    outcome.scene_route = str(scene.scene_route)
    outcome.question_tool_scope = str(scene.question_tool_scope)
    outcome.scene_summary = str(scene.summary or outcome.scene_summary)
    outcome.evidence_profile = scene.evidence_profile
    outcome.metric_evidence_gate_result = scene.metric_evidence_gate_result
    effective_task = str(task or scene.question_type or "")
    outcome.authorized_metric_tasks = (
        [effective_task] if effective_task and scene.metric_task_authorized(effective_task)
        else [])


def _invalidate_metric_gate(gate, premise: str):
    """Return a fail-closed metric gate after a shared premise is invalidated.

    Recovery may invalidate geometry or metric-scale observations after the
    original gate was computed. Keep the gate's subconditions auditable and
    self-consistent instead of leaving a stale ``gate_passed=True`` result.
    """
    if gate is None:
        return None
    sub_results = dict(getattr(gate, "sub_results", None) or {})
    if premise == "geometry_3d":
        affected = {
            "scene_route_full_3d",
            "m4_main_gate_passed",
            "scale_fusion_success",
            "scale_self_consistency_ok",
        }
    elif premise == "metric_scale":
        affected = {"scale_fusion_success", "scale_self_consistency_ok"}
    else:
        # Object binding and other local failures do not invalidate the scene's
        # metric evidence gate; their question scope is handled separately.
        return gate
    # Preserve all existing audit values, but add affected keys when an older
    # gate omitted them. This guarantees a fail-closed gate cannot remain
    # ``gate_passed=False`` with an all-true sub_results mapping.
    for name in affected:
        sub_results[name] = False
    missing = sorted(name for name, passed in sub_results.items() if not passed)
    invalidated = sorted(set(
        list(getattr(gate, "invalidated_by", None) or []) + [str(premise)]
    ))
    updated = gate.model_dump()
    updated.update({
        "gate_passed": False,
        "sub_results": sub_results,
        "missing_subconditions": missing,
        "invalidated_by": invalidated,
    })
    # ``model_copy(update=...)`` intentionally skips validation in Pydantic v2.
    # Re-validate the serialized payload so recovery cannot manufacture a
    # contradictory MetricEvidenceGateResult.
    return type(gate).model_validate(updated)


def _record_first_synthesis(outcome: EpisodeOutcome, res: _SynthResult,
                            scene: Optional[SceneState], cfg: OnlineRunConfig) -> None:
    if outcome.first_synthesis:
        return
    prompt = str(res.prompt or "")
    profile = getattr(scene, "evidence_profile", None) if scene is not None else None
    gate = (getattr(scene, "metric_evidence_gate_result", None)
            if scene is not None else None)
    outcome.first_synthesis = {
        "phase": "first_synthesis",
        "source": str(res.source or ""),
        "template_version": PROMPT_TEMPLATE_VERSION,
        "tool_docs_version": TOOL_DOCS_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if prompt else "",
        "prompt_present": bool(prompt),
        "n_images": int(res.n_images),
        "scene_route": str(getattr(scene, "scene_route", "") or ""),
        "question_tool_scope": str(getattr(scene, "question_tool_scope", "") or ""),
        "evidence_profile_version": str(getattr(profile, "profile_version", "") or ""),
        "metric_gate_version": str(getattr(gate, "gate_version", "") or ""),
        "metric_gate_passed": (None if gate is None else bool(gate.gate_passed)),
        "active_snapshot_ref": str(cfg.active_snapshot_ref or ""),
        "selected_skill_semvers": list(outcome.selected_skill_semvers),
        # v9 §13.6：首轮合成的交付事实与冻结的检索配置版本（"检索选中"≠"已交付"）
        "retrieval_config_version": (
            outcome.retrieval_records[-1].config_version
            if outcome.retrieval_records else ""),
        "delivered_skill_versions": list(outcome.delivered_skill_versions or []),
        "skill_delivery": (res.delivery.to_dict() if res.delivery is not None else {}),
        # v9 §9.4：首轮请求实际装入的图像清单（布局/张数/token 成本）
        "image_round": dict(res.image_round or {}),
    }

# --------------------------------------------------------------- episode 执行 ----

@capture_detector_attempts()
def run_episode(
    episode: VSIBenchEpisode,
    pixels: Sequence[np.ndarray],
    cfg: OnlineRunConfig,
    *,
    geometry: Optional[synth.SyntheticGeometry] = None,
    trace_store: Optional[TraceStore] = None,
    llm=None,
    prepared_binding: Optional[PreparedObjectBinding] = None,
    input_error: Optional[InputErrorRecord] = None,
) -> EpisodeOutcome:
    """跑完一条在线链 episode。

    llm：可注入的 `synthesis.vllm_client.VLLMClient` 兼容对象（测试用 fake）；
    为 None 且 `mode="real"` 时按 `cfg.vllm_endpoints` 构造；`mock_light` 下不需要。
    """
    if any(not isinstance(skill, SkillSpecV11) for skill in cfg.skills):
        raise ValueError(
            "当前在线协议只接受 SkillSpecV11；运行配置在构造后被注入了旧 SkillSpec")
    if prepared_binding is not None and not cfg.reuse_artifact:
        raise ValueError("prepared_binding requires an episode-local reuse_artifact")
    pixels = list(pixels)
    if input_error is not None:
        if pixels:
            raise ValueError("input_error 条目不得同时携带可读像素")
        if input_error.qa_id != episode.qa_id:
            raise ValueError(
                f"input_error.qa_id={input_error.qa_id!r} 与 episode={episode.qa_id!r} 不一致")
    episode_id = episode.qa_id
    states: list[str] = []
    notes: list[str] = []
    receipts = _receipt_chain(episode_id, cfg)
    receipts.append("episode_start", {"qa_id": episode.qa_id, "mode": cfg.mode,
                                      "baseline": cfg.baseline, "split": episode.split})

    outcome = EpisodeOutcome(
        qa_id=episode.qa_id,
        scene_name=episode.scene_name,
        question_type=episode.question_type,
        final_state="unanswerable",
        answer=None,
        predicted=None,
        is_mca=False,
        correct=None,
        mra_value=None,
        notes=notes,
    )
    if input_error is not None:
        outcome.input_error_reason = input_error.reason
        notes.append(
            f"M1 input_error={input_error.reason}；所有输入来源均失败，"
            "不请求模型且保留评测分母")

    fsm = OnlineFSM(max_regen=cfg.max_regen)
    states.append(fsm.state.value)

    # ---------------- M1 INGEST ----------------
    split_ok = episode.split != "final_test" or cfg.allow_final_test
    if not split_ok:
        notes.append("split=final_test：硬约束 9 禁止默认进在线链（需 allow_final_test 显式盲评）")
    fsm.step("episode_ready", {"split_ok": split_ok})
    states.append(fsm.state.value)
    if fsm.terminated:
        return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                     )

    cls: Optional[TaskClassification] = None
    scene: Optional[SceneState] = None
    handle: Optional[SceneHandle] = None
    verdict: Optional[InputGateVerdict] = None
    retrieved: list[RetrievedSkill] = []
    selected_skills: list[RuntimeSkillSpec] = []
    program: Optional[EpisodeProgram] = None
    program_trace: Optional[ProgramExecutionTrace] = None
    kernel: Optional[RestrictedNamespaceKernel] = None
    art = None

    # ---------------- M2 INPUT_GATE（被动观测，硬约束 21）----------------
    if fsm.state is OnlineState.INPUT_GATE:
        frame_set = episode.frame_set
        gate_input = pixels if pixels else list(episode.frames)
        verdict = input_gate(gate_input, frame_set=frame_set,
                            diagnostics=bool(cfg.input_diagnostics))
        notes.append(
            f"M2 被动观测 level={verdict.level} action={verdict.action} "
            f"degraded_frames={len(verdict.degraded_frame_ids)}/{verdict.n_frames} "
            f"flags={verdict.degradation_flags or '无'} "
            f"quality_score={verdict.quality_score:.2f} "
            f"weight={verdict.quality_weight:.2f} "
            f"frame_set_hash={verdict.frame_set_hash[:12]}")
        # 硬约束 21：M2 **不删/不换/不补/不重排帧**，只把观测（flag/weight）写回 InputFrame；
        # 帧集在 M1 之后冻结，主方法与所有 baseline 共用同一帧序。
        pixels_for_stats = pixels if pixels and len(pixels) == len(episode.frames) else None
        episode = episode.model_copy(update={
            "frames": annotate_frames(episode.frames, verdict, pixels_for_stats,
                                      diagnostics=bool(cfg.input_diagnostics))})

        fsm.step("gate_done", {"action": verdict.action})
        states.append(fsm.state.value)
        if fsm.state is OnlineState.ANSWER:
            outcome.final_state = "input_error"
            outcome.failure_code = "input_error"
            if not outcome.input_error_reason:
                outcome.input_error_reason = "all_images_unreadable"
            outcome.input_degradation_flags = list(verdict.degradation_flags)
            notes.append("M2 输入合法性 hard fail → input_error（§4 M2 字段 9；"
                         f"hard_fail_frames={verdict.hard_fail_frame_ids[:5]}）")
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                             verdict=verdict)

    # ---------------- M3 RECONSTRUCT ----------------
    art_path: str = ""
    if fsm.state is OnlineState.RECONSTRUCT:
        if cfg.reuse_artifact:
            # 复用既有 artifact（离线 paired A/B 两臂必须同源，硬约束 18）
            # v5 HC39：运行时**唯一**加载入口是 legacy 门 —— 版本不符/含旧字段的
            # artifact 在这里 hard fail（并提示重跑 v5 pipeline），不得静默反序列化。
            from skill3d.legacy.readers import load_artifact_v5

            art = load_artifact_v5(cfg.reuse_artifact)
            art_path = cfg.reuse_artifact
            notes.append(f"M3 复用既有 artifact（A/B 同源，硬约束 18）: {cfg.reuse_artifact}")
            _validate_artifact_identity(art, episode, pixels)
            fsm.step("done")
        elif cfg.mode == "real":
            # 方案 X（§2.2 / §4 M4）：P1 已把 artifact（含实算 quality）落盘时，
            # P2 **读取复用**、零重算。这也是硬约束 18 的自然形态：同一 scene 的
            # 多个 episode（含 paired A/B 两臂）看到的是**同一个** artifact 对象内容。
            existing, _used_legacy = _resolve_existing_artifact(cfg, episode)
            try:
                if _Path_(existing).is_file():
                    from skill3d.legacy.readers import load_artifact_v5

                    cand = load_artifact_v5(existing)   # v5 门：版本/legacy fail-closed
                    try:
                        _validate_artifact_identity(cand, episode, pixels)
                    except ValueError as exc:
                        notes.append(f"[warn] P1 artifact 身份不符 → 重算: {exc}")
                        raise _ReuseMiss() from exc
                    art, art_path = cand, existing
                    notes.append(
                        f"M3 复用 P1 落盘 artifact（方案 X：quality_status="
                        f"{art.quality_status}，P2 零重算）: {existing}")
                    fsm.step("done")
                else:
                    raise _ReuseMiss()
            except _ReuseMiss:
                try:
                    art = reconstruct(pixels, episode.scene_name, cfg.recon_dir,
                                      method=cfg.recon_method,
                                      frame_set=episode.frame_set,
                                      metric_depth_model=cfg.metric_depth_model,
                                      metric_model_name=cfg.metric_model_name)
                    art_path = existing
                    # 真实缺陷（2026-09-21 实测）：eval 路径下 `reconstruct()` 现场重算的
                    # scene 只把 artifact 留在内存（方案 X 的写回发生在 P1 重建批里），
                    # 于是 next run 会**重新跑一遍 VGGT**（~1 分钟/场景），且 trace 里
                    # 没有可审计的 artifact 文件。这里显式落盘，让方案 X 在 eval 路径也成立。
                    from skill3d.reconstruction_gate.quality_metrics import (
                        write_artifact_json,
                    )

                    try:
                        write_artifact_json(art, art_path)
                        notes.append(f"M3 artifact 已落盘（方案 X）：{art_path}")
                    except Exception as exc:  # noqa: BLE001 - 落盘失败不得阻断作答
                        notes.append(f"[warn] M3 artifact 落盘失败（{exc}）→ "
                                     "下次会重算，不影响本题作答")
                    notes.append(f"M3 重建完成 method={art.recon_method} "
                                 f"quality_status={art.quality_status} "
                                 f"frame_set_hash={art.frame_set_hash[:12]} "
                                 f"world_frame={art.world_frame_status} "
                                 f"scale_fusion={art.scale_fusion_status}")
                    fsm.step("done")
                except ReconstructionFailed as exc:
                    notes.append(f"M3 重建失败（v6 无降级链：VGGT 是唯一主线）: {exc}")
                    fsm.step("failed")
        else:
            if geometry is None:
                raise ValueError("mode=mock_light 必须传入 geometry（online.synthetic 构造）")
            notes.append("M3 跳过真实重建：mock_light 使用合成几何（非 VGGT 产物）")
            fsm.step("skip")   # 世界系契约/度量融合随 artifact 落盘，合成路径跳过两态
        states.append(fsm.state.value)
        if fsm.state is OnlineState.ANSWER:
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                             verdict=verdict)

    # ---------------- M3.5 WORLD_FRAME + METRIC_FUSION（v6 D5/D1）----------------
    # 真实路径：世界系契约与度量融合都已在 M3 内完成并落盘。两态在此只做**观测与落档**：
    # world_up/handedness 的存在性、融合状态、尺度离散度、有效帧占比。二者都
    # **不改变 scene_route**（§6.2：route 只由 M4 质量决定），也都不中止 episode：
    # 世界系缺失 → 方向/路线类 Tool 被 docs() 隐藏并执行期 fail-closed；
    # 融合失败 → metric_scale 能力落 unavailable → 逐题 scope 收窄（米制 Tool 收回）。
    if fsm.state is OnlineState.WORLD_FRAME:
        wf_status = str(getattr(art, "world_frame_status", "unavailable"))
        outcome.world_frame_status = wf_status
        outcome.world_up = getattr(art, "world_up", None)
        outcome.handedness = getattr(art, "handedness", None)
        wf_unavailable = wf_status == "unavailable" or getattr(art, "world_up", None) is None
        notes.append(f"M3.5 WORLD_FRAME status={wf_status} "
                     f"up={getattr(art, 'world_up', None)} "
                     f"handedness={getattr(art, 'handedness', None)}"
                     + ("（缺失 → 方向/路线类 Tool fail-closed，§9.8/§9.10）"
                        if wf_unavailable else ""))
        fsm.step("done", {"world_frame_unavailable": wf_unavailable})
        states.append(fsm.state.value)

    if fsm.state is OnlineState.METRIC_FUSION:
        status = str(getattr(art, "scale_fusion_status", "not_run"))
        disp = getattr(art, "scale_self_consistency", None)
        outcome.scale_fusion_status = status
        outcome.metric_scale = getattr(art, "metric_scale", None)
        outcome.scale_self_consistency = disp
        outcome.metric_model = str(getattr(art, "metric_model", None) or "none")
        outcome.metric_fusion_version = str(getattr(art, "metric_fusion_version", "") or "")
        outcome.per_frame_scale_ref = getattr(art, "per_frame_scale_ref", None)
        from skill3d.reconstruction_gate.evidence_profile import TH_SCALE_DISPERSION

        disp_high = (disp is not None and np.isfinite(float(disp))
                     and float(disp) > TH_SCALE_DISPERSION)
        failed = status != "success" or getattr(art, "metric_scale", None) is None
        notes.append(f"M3.5 METRIC_FUSION status={status} "
                     f"metric_scale={getattr(art, 'metric_scale', None)} "
                     f"dispersion={disp} model={outcome.metric_model}"
                     + ("（融合未成功 → 米制 Tool 逐题收回；不回退多锚点/校准池，§11.4）"
                        if failed else ""))
        fsm.step("done", {"metric_fusion_failed": failed,
                          "scale_dispersion_high": disp_high})
        states.append(fsm.state.value)

    # ---------------- M4 QUALITY_GATE（+ M5 对象绑定）----------------
    if fsm.state is OnlineState.QUALITY_GATE:
        m5_stats: dict = {}
        m5_materialized = False
        if art is not None:
            # M5 先于 M4 统计：对象绑定产物是 G7/G9 的数据源（G-18 接线）
            if prepared_binding is None:
                m5_objects, m5_stats, m5_notes, m5_materialized = _bind_objects_best_effort(
                    art, pixels, None, episode, cfg, llm)
            else:
                from copy import deepcopy

                binding = deepcopy(prepared_binding)
                m5_objects, m5_stats = binding.objects, binding.stats
                m5_notes, m5_materialized = binding.notes, binding.materialized
            notes.extend(m5_notes)
            outcome.m5_notes = list(m5_notes)
            if prepared_binding is not None:
                outcome.detector_attempts = list(m5_stats.get("detector_attempts") or [])
            scene, handle, art = _scene_from_artifact(
                art, pixels, m5_stats, m5_objects, episode, artifact_path=art_path,
                objects_materialized=m5_materialized,
                # paired A/B 复用 frozen artifact（硬约束 18）时不得改写两臂共用文件
                persist_quality=not bool(cfg.reuse_artifact))
            # v9 §9.4：把声明的布局/上限装进图像账本（在 M5/M4 之后、M7 之前）
            _configure_image_ledger(handle, cfg, episode)
            # M5 证据摘要在**此处**落档：M7.5 的逐题 scope 派生与 M10 的级联撤销
            # 都要重算 EvidenceProfile（`scope_scene_to_question(..., m5=...)`），
            # 传 None 会把 object_detection/track_consensus 重算成 unavailable →
            # 对象类 Tool 被整批隐藏（真实链路上等于每题都撞 tool_contract 后 abstain）。
            outcome.m5_summary = _m5_evidence_summary(
                m5_stats, m5_objects, episode, m5_materialized=m5_materialized)
            notes.append(f"M4 质量门禁（唯一事实源）scene_route={scene.scene_route} "
                         f"scope={scene.question_tool_scope} "
                         f"quality_status={art.quality_status} "
                         f"overall_quality={_overall_str(art)} "
                         f"available={sorted(scene.available_artifacts)}")
            outcome.scene_summary = str(getattr(scene, "summary", "") or "")
            outcome.quality_status = art.quality_status
            outcome.overall_quality = _overall_of(art)
            outcome.main_gate_passed = (
                None if getattr(art, "quality", None) is None
                else bool(art.quality.main_gate_passed))
        else:
            # mock_light：合成几何 + 真算 G1–G11；M2 的被动观测（flag/weight）并入 route
            scene, handle, _q = synth.build_scene_state(  # type: ignore[arg-type]
                geometry, pixels,
                input_quality_weight=verdict.quality_weight if verdict else 1.0,
                input_degradation_flags=verdict.degradation_flags if verdict else None)
            outcome.scene_summary = str(getattr(scene, "summary", "") or "")
            outcome.quality_status = "computed"
            outcome.overall_quality = float(_q.overall_quality)
            # M4 主门结论必须落档：TraceRecord/EpisodeTrace 的 main_gate_passed 是
            # 质量归因的第一列（v5 曾只写 overall_quality，主门被"标量阈值"掩盖）
            outcome.main_gate_passed = bool(_q.main_gate_passed)
            outcome.scale_source = "synthetic_mock_light"
            # 合成 M5 证据摘要（构造真值）同样落档：M7.5 重算证据画像时不得丢
            outcome.m5_summary = synth.synthetic_m5_summary(geometry)  # type: ignore[arg-type]
            notes.append(f"M4 合成门禁 scene_route={scene.scene_route} "
                         f"scope={scene.question_tool_scope} "
                         f"overall_quality={_q.overall_quality:.3f}"
                         "（mock_light：M4 主门在合成数据上真算）")
        outcome.scene_route = scene.scene_route
        outcome.question_tool_scope = scene.question_tool_scope
        outcome.artifact_ref = art_path or (art.artifact_id if art is not None else "")
        outcome.g9_tracker_consistency = _mean_track_iou(m5_stats)
        # v6：G5 永久 not_available（D11），G8 永久退役 —— 两者都不得出现代理值
        outcome.reprojection_status = str(
            getattr(art, "reprojection_status", "not_available") or "not_available")
        outcome.input_degradation_flags = list(
            getattr(verdict, "degradation_flags", None) or []) if verdict else []
        outcome.evidence_profile = scene.evidence_profile
        outcome.metric_evidence_gate_result = scene.metric_evidence_gate_result
        action = {"unanswerable": "unanswerable",
                  "fallback_2d_only": "fallback_2d_only"}.get(scene.scene_route, "proceed")
        fsm.step("gate_done", {"action": action})
        states.append(fsm.state.value)
        if fsm.state is OnlineState.ANSWER:
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                             verdict=verdict, scene=scene, handle=handle)

    # ---------------- M7 CLASSIFY_TASK + 逐题 scope 派生 + RETRIEVE_SKILL ----------------
    if fsm.state is OnlineState.CLASSIFY_TASK:
        cls = classify(episode)
        outcome.task = cls.task
        notes.append(f"M7 题型={cls.question_type} task={cls.task} is_mca={cls.is_mca}")
        # 逐题 scope 派生（v6 D4）：只收窄 question_tool_scope，**不改 scene_route**
        decision = None
        if scene is not None:
            scene, decision = scope_scene_to_question(
                scene, cls.task, m5=outcome.m5_summary)
            notes.append(f"M7.5 {decision.note()}")
            outcome.answer_flags.extend(decision.flags)
            outcome.notes.extend(decision.reasons)
            _sync_scene_snapshot(outcome, scene, cls.task)
            outcome.m7_notes = [decision.note()] + list(decision.reasons)
            if "metric_tool_withheld" in decision.flags:
                notes.append(
                    f"M7.5 本题米制 Tool 已收回（scene_route 保持 {scene.scene_route}）："
                    f"非米制 3D 产物保留="
                    f"{sorted(set(scene.available_artifacts) - {'scale'})}")
            handle = _retarget_handle(handle, scene)
        fsm.step("done", {
            "question_ok": True if decision is None else decision.allowed,
            "reject_flag": "metric_task_reject",
            "downgrade_2d_only": False,
        })
        states.append(fsm.state.value)
        if decision is not None and not decision.allowed:
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                             verdict=verdict, scene=scene, handle=handle)

    if fsm.state is OnlineState.RETRIEVE_SKILL:
        selected_skills = _retrieve_for_episode(
            episode, scene, cfg, outcome, trigger="initial",
            llm=llm)
        outcome.retrieved_skill_versions = list(
            outcome.retrieval_records[-1].retrieved_skill_versions)
        notes.extend(_retrieval_summary_lines(outcome.retrieval_records[-1]))
        notes.append(f"M7 Skill 检索命中 {len(outcome.retrieved_skill_versions)} 条" +
                     ("" if outcome.retrieved_skill_versions
                      else "（无命中 → 空 Skill baseline）"))
        if outcome.skill_mapping_misses:
            notes.append(f"M7 Skill 映射缺失: {outcome.skill_mapping_misses}")
        fsm.step("done")
        states.append(fsm.state.value)

    if fsm.state is OnlineState.SYNTHESIZE_PROGRAM:
        scene, handle, selected_skills, kernel, program_trace = _solve_program(
            episode, scene, handle, selected_skills, cfg, llm, outcome, fsm,
            states, receipts, pixels=pixels, geometry=geometry)
        program = outcome.program
        if outcome.final_state == "unavailable":
            return _finalize(
                outcome, fsm, cfg, episode, states, receipts, trace_store,
                verdict=verdict, scene=scene, handle=handle)

    # ---------------- M12 BENCHMARK_EVAL ----------------
    answer: Optional[str] = kernel.answer_slot.answer if kernel is not None else None
    visual_payload = None
    if (cfg.eval_visual_fallback and cfg.mode == "real"
            and episode.split in EVAL_VISUAL_SPLITS
            and outcome.final_state == "run_error" and outcome.agent_rounds > 0
            and answer is None):
        original = {
            "final_state": outcome.final_state, "answer": None,
            "score": (None if any(r.get("service_errors") for r in outcome.rounds) else 0.0),
            "failure_code": outcome.failure_code,
            "terminal_failure": dict(outcome.terminal_failure),
            "answer_untrusted": outcome.answer_untrusted,
            "answer_flags": list(outcome.answer_flags) + list(fsm.answer_flags),
            "agent_rounds": outcome.agent_rounds,
            "geometry_verify": (outcome.verify.model_dump(mode="json")
                                if outcome.verify is not None else None),
        }
        visual_payload, audit = request_visual_answer(
            question=episode.question, options=list(episode.options or []),
            question_type=str(outcome.task or canonical_task(episode.question_type)),
            ledger=getattr(handle, "_ledger", None),
            client=llm if llm is not None else _make_vllm_client(cfg),
            round_index=outcome.agent_rounds + 1, max_images=cfg.max_images,
            max_tokens=cfg.max_tokens, seed=int(cfg.seed))
        outcome.eval_visual_fallback = {**audit, "original_result": original}
        receipts.append("eval_visual_fallback", outcome.eval_visual_fallback)
        if visual_payload is not None:
            answer = str(visual_payload.value)
            outcome.final_state = "answer"
            outcome.failure_code = None
            outcome.terminal_failure = {}
            outcome.answer_untrusted = False
            outcome.abstained = False
            outcome.used_result_ids = []
            outcome.answer_flags = [
                flag for flag in outcome.answer_flags
                if flag not in ("unanswerable", "abstain")
            ]
            fsm.answer_flags = [
                flag for flag in fsm.answer_flags
                if flag not in ("unanswerable", "abstain")
            ]
            # 原 M11 检查留在 original_result 与 rounds；视觉答案只做答案合同验收。
            outcome.verify = None
            outcome.answer_flags.append("eval_visual_fallback")
            fsm.step("eval_visual_answer")
            states.append(fsm.state.value)
        elif audit.get("service_error"):
            outcome.final_state = "unavailable"
            outcome.failure_code = "service_unavailable"
            outcome.terminal_failure = {
                "code": "service_unavailable",
                "stage": "EVAL_VISUAL_FALLBACK",
                "termination_reason": "visual_request_failed",
                "round": outcome.agent_rounds + 1,
            }
        notes.append(
            f"评测增量视觉补答 requests={audit['request_count']} "
            f"accepted={audit['accepted']} "
            f"detail={audit.get('error') or audit.get('skipped_reason') or 'ok'}")
    answer_source = _answer_source_v6(answer=answer, program=program, kernel=kernel,
                                      outcome=outcome)
    if fsm.state is OnlineState.BENCHMARK_EVAL:
        if cls is None:
            cls = classify(episode)
        predicted, correct, mra_value = _evaluate(episode, answer, cls)
        outcome.predicted, outcome.correct, outcome.mra_value = predicted, correct, mra_value
        outcome.is_mca = cls.is_mca
        notes.append(f"M12 评测 is_mca={cls.is_mca} predicted={predicted} "
                     f"correct={correct} mra={mra_value}")
        fsm.step("done")
        states.append(fsm.state.value)

    # ---------------- ANSWER → M13 LOG_TRACE ----------------
    if fsm.state is OnlineState.ANSWER:
        outcome.answer = answer
        outcome.answer_source = answer_source
        # ---- v9 §10.1/§12：答案载荷 + 框架核验的工具归因 ----
        if answer is not None:
            _record_answer_attribution(
                outcome, episode, kernel, answer, payload_override=visual_payload)
        if kernel is not None:
            outcome.authorization_receipts = [
                dict(r) for r in (getattr(kernel, "authorization_receipts", []) or [])]
        if answer is not None and "unanswerable" not in fsm.answer_flags:
            outcome.final_state = "answer"
        outcome.episode_status = _episode_status_of(outcome.final_state)
        fsm.step("done")
        states.append(fsm.state.value)

    return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                     verdict=verdict, scene=scene, handle=handle)


# ------------------------------------------------------------------ 各阶段实现 ----

def _solve_program(episode, scene, handle, skills, cfg, llm, outcome, fsm,
                   states, receipts, *, pixels, geometry):
    """M8–M11 单一循环。每轮最多一次生成请求；所有出口都撤掉未验收的答案。"""
    kernel = None
    trace = None
    feedback = None
    prior_source = None
    observation_images: list[str] = []
    trigger = "initial"
    retry_counts: dict[str, int] = {}
    force_next = False
    unresolved_ids: set[str] = set()
    termination_reason = "solver_round_budget_exhausted"

    def step(event):
        fsm.step(event)
        states.append(fsm.state.value)

    def revoke_premise(premise, *, confirmed=False):
        nonlocal scene, handle, skills
        ids, _ = cascade_invalidate(kernel, premise, registry=REGISTRY)
        profile, changed = downgrade_profile(
            getattr(scene, "evidence_profile", None), premise,
            confirmed_invalid=confirmed)
        if profile is not None and scene is not None:
            scene = scene.model_copy(update={"evidence_profile": profile})
            scene, _ = scope_scene_to_question(
                scene, outcome.task, m5=outcome.m5_summary,
                metric_gate_override=_invalidate_metric_gate(
                    getattr(scene, "metric_evidence_gate_result", None), premise),
                evidence_profile_override=profile)
            _sync_scene_snapshot(outcome, scene, outcome.task)
            handle = _retarget_handle(handle, scene)
            if changed:
                skills = _retrieve_for_episode(
                    episode, scene, cfg, outcome, trigger="evidence_update", llm=llm)
                outcome.retrieved_skill_versions = list(
                    outcome.retrieval_records[-1].retrieved_skill_versions)
        return ids, changed

    def observations():
        return [o for o in collect_validated(kernel) if o.result_id not in unresolved_ids]

    for index in range(1, int(cfg.max_solver_rounds) + 1):
        finalizing = force_next or (
            index > 1 and index > int(cfg.max_solver_rounds) - int(cfg.finalization_rounds))
        if finalizing:
            trigger = "finalize"
            outcome.finalization_used = True
            outcome.answer_flags.append("finalization")
            feedback = (feedback or "") + (
                "\n这是预算内最后一次作答请求。只允许零工具程序，不得 YieldObservations。"
                "请依据仍有效的观察或本次原图给出最佳答案；完全依据原图时提交 "
                "AnswerPayload(value=答案, unit=题型单位, basis='visual_estimate')。"
                "已撤销的结果与被拒绝的答案不能作为依据。")
        outcome.agent_rounds = index
        outcome.round_trigger = trigger
        # _synthesize 对 v11 不在内部追加退化重试，所有请求都经过这里计数。
        if finalizing and cfg.mode == "mock_light":
            forced, source = _forced_answer_program(
                episode, scene, handle, skills, cfg, llm, pixels=pixels, geometry=geometry)
            res = _SynthResult(
                forced,
                source,
                "",
                response_text=str(getattr(forced, "program_source", "") or ""),
            )
        else:
            res = _synthesize(
                episode, scene, handle, skills, cfg, llm, pixels=pixels, geometry=geometry,
                traceback_feedback=feedback, prior_program=prior_source,
                observation_image_ids=observation_images,
                round_index=index, trigger=trigger)
        if index == 1:
            _record_first_synthesis(outcome, res, scene, cfg)
            outcome.m8_prompt = str(res.prompt or "")
        _record_skill_delivery(outcome, res.delivery, round_index=index,
                               skills=skills, program=res.program)
        outcome.synthesis_source = res.source
        outcome.n_images_to_synthesizer = int(res.n_images)
        outcome.m8_notes.append(f"round={index} source={res.source} {res.note}")
        outcome.program = res.program
        record = _round_record(index=index, trigger=trigger, source=res.source,
                               program=res.program, program_trace=None,
                               response_text=res.response_text)
        record.update(execution_protocol_version=EXECUTION_PROTOCOL_VERSION,
                      feedback=feedback or "", executed=False)
        outcome.rounds.append(record)
        outcome.round_trace_refs = [f"round:{r['index']}" for r in outcome.rounds]
        program = res.program
        failure_key = ""
        feedback = None
        observation_images = []
        if program is None:
            record["error_code"] = res.source
            feedback = f"上一轮程序生成失败：{res.note}。请重新输出一个合法 Python 代码块。"
            failure_key = "synthesis"
            if res.source in ("none", "vllm_service_error"):
                outcome.final_state = "unavailable"
                termination_reason = "synthesis_unavailable"
                break
        else:
            prior_source = program.program_source
            step("done")  # SYNTHESIZE_PROGRAM → STATIC_CHECK
            check = ast_guard(program.program_source, allowed_tools=set(REGISTRY.names()))
            if not check.ok or (finalizing and not _is_zero_tool_answer_program(program)):
                problems = list(check.violations) or ["终结程序必须零工具并提交答案"]
                outcome.static_check_errors.extend(problems)
                record["error_code"] = "ast_violation"
                feedback = "上一次生成被 AST 拒绝：" + "; ".join(problems)
                failure_key = "ast"
            else:
                step("pass")
                if kernel is not None:
                    kernel.set_tools_enabled(not finalizing)
                if handle is not None:
                    handle._ledger.current_round = index  # noqa: SLF001
                start = time.perf_counter()
                kernel, cell, trace = _execute_program(
                    episode, program, handle, pixels, cfg, receipts, kernel=kernel,
                    tools_enabled=not finalizing)
                trace = trace.model_copy(update={
                    "wallclock_s": 0.0 if cfg.deterministic_replay else time.perf_counter() - start,
                    "answer_untrusted": bool(cell.answer_untrusted)})
                outcome.program_trace = trace
                outcome.answer_untrusted = bool(cell.answer_untrusted)
                record.update(_round_record(
                    index=index, trigger=trigger, source=res.source,
                    program=program, program_trace=trace,
                    response_text=res.response_text))
                record["executed"] = True
                record["results"] = [r.model_dump(mode="json") for r in trace.results]
                receipts.append("cell_run", {"program_id": program.program_id,
                                             "error_code": cell.error_code})
                if cell.answer_untrusted:
                    outcome.answer_flags.append("tool_contract_observed")
                error = cell.error_code
                if error == "tool_contract" and cell.contract_violations:
                    cause = cell.contract_violations[-1]
                    if cause.get("error_code"):
                        record["cause_code"] = str(cause["error_code"])
                if finalizing and cell.answer_untrusted:
                    error = "tool_contract"
                if error is None and str(cell.answer or "").strip():
                    step("ok")
                    scope = submission_scope(kernel, trace)
                    verify = geometry_verify(scope, handle, cell.answer, options=episode.options)
                    reference_issues = invalid_reference_issues(kernel)
                    if reference_issues:
                        verify = verify.model_copy(update={
                            "passed": False,
                            "checks": {**verify.checks, "invalidated_reference": False},
                            "violations": list(verify.violations) + ["invalidated_reference"],
                            "issues": list(verify.issues) + reference_issues})
                    replay = replay_derivation(
                        kernel.answer_slot.payload,
                        kernel.tool_results,
                        question_type=str(outcome.task or episode.question_type or ""),
                        options=episode.options,
                    )
                    if not replay.passed:
                        derivation_issues = [
                            GeometryIssue(
                                check="derivation_replay",
                                result_id=issue.result_id,
                                reason=issue.reason,
                                invalidate_result=False,
                            )
                            for issue in replay.issues
                        ]
                        verify = verify.model_copy(update={
                            "passed": False,
                            "checks": {**verify.checks, "derivation_replay": False},
                            "violations": list(dict.fromkeys(
                                list(verify.violations) + ["derivation_replay"])),
                            "issues": list(verify.issues) + derivation_issues,
                            "derivation_replay": replay.model_dump(mode="json"),
                        })
                    else:
                        verify = verify.model_copy(update={
                            "checks": {**verify.checks, "derivation_replay": True},
                            "derivation_replay": replay.model_dump(mode="json"),
                        })
                    outcome.verify = verify
                    record["geometry_verify"] = verify.model_dump(mode="json")
                    record["verification_result_ids"] = [r.result_id for r in scope.results]
                    receipts.append("geometry_verify", {
                        "round": index, **verify.model_dump(mode="json")})
                    outcome.notes.append(
                        f"M11 round={index} passed={verify.passed} violations={verify.violations}")
                    if verify.passed:
                        outcome.failure_code = None
                        outcome.terminal_failure = {}
                        record["submission_accepted"] = True
                        outcome.used_result_ids = accepted_result_ids(kernel, scope)
                        step("pass")
                        return scene, handle, skills, kernel, trace
                    record["submission_accepted"] = False
                    record["rejected_submission"] = rejected_submission(kernel)
                    # 保存审计后立即清空 answer/payload/given 和用户变量，账本保留。
                    kernel.reset_user_namespace()
                    ids = invalidate_results(
                        kernel, [i.result_id for i in verify.issues
                                 if i.result_id and i.invalidate_result],
                        reason=f"M11:round:{index}")
                    changes = {}
                    for premise in sorted({i.confirmed_shared_premise for i in verify.issues
                                           if i.confirmed_shared_premise}):
                        extra, changed = revoke_premise(premise, confirmed=True)
                        ids.extend(extra)
                        changes.update(changed)
                    record["invalidated_result_ids"] = sorted(set(ids))
                    record["downgraded_capabilities"] = changes
                    outcome.invalidated_result_ids.extend(ids)
                    located_checks = {
                        i.check for i in verify.issues
                        if i.result_id and i.invalidate_result
                    }
                    unlocated = set(verify.violations) - located_checks
                    if unlocated - {"derivation_replay"}:
                        # 归因不足：不猜共享根因，也不把尚未通过验收的结果回灌成可信观察。
                        unresolved_ids.update(r.result_id for r in scope.results)
                    feedback = (
                        "M11 拒绝上一轮提交，答案槽和程序变量已清空；请修正后重新提交。\n"
                        f"检查失败：{verify.violations}\n"
                        f"定位事实：{[i.model_dump() for i in verify.issues]}\n"
                        f"已撤销结果：{sorted(set(ids))}；证据能力变化：{changes}。\n"
                        "无明确共享根因的失败只作局部处理；未定位结果不当作可信观察。")
                    failure_key = "geometry_verify"
                    record["error_code"] = "geometry_rejected"
                    step("reject")
                elif error is None and cell.terminated == "yield" and not finalizing:
                    outcome.yield_count += 1
                    feedback = _yield_feedback(cell, kernel, scene, cfg)
                    observation_images = _yield_image_ids(cell, kernel, handle)
                    trigger = "observation"
                else:
                    feedback = f"程序执行失败：{error or 'no_answer'}；{cell.error or '没有提交答案'}"
                    failure_key = str(error or "no_answer")
                    record["error_code"] = failure_key
                    if kernel.answer_slot.given:
                        record["rejected_submission"] = rejected_submission(kernel)
                    if error in _RECOVERABLE_CONTRACT_ERRORS:
                        outcome.tool_contract_hits += 1
                        outcome.recovery_count += 1
                        outcome.partial_tool_recovery = True
                        premise = _premise_from_cell(cell, kernel)
                        ids, changes = revoke_premise(premise) if premise else ([], {})
                        outcome.invalidated_result_ids.extend(ids)
                        record["invalidated_result_ids"] = ids
                        record["downgraded_capabilities"] = changes
                        plan = RecoveryPlan(
                            attempt=retry_counts.get(failure_key, 0) + 1,
                            failed_tool=_failed_tool_of(cell), failure_kind=str(error),
                            premise=premise, validated=observations(),
                            invalidated_result_ids=ids, downgraded_capabilities=changes)
                        feedback += "\n" + build_feedback(
                            plan, question_type=outcome.task, scope=outcome.question_tool_scope,
                            require_answer=True)

        if kernel is not None:
            kernel.reset_user_namespace()
        if finalizing:
            termination_reason = "finalization_failed"
            outcome.notes.append(f"M11 终结轮未通过验收（round={index}），不再请求模型")
            break
        if feedback and kernel is not None:
            valid = observations()
            feedback += "\n仍然有效的既有观察（仅这些结果可复用）：\n" + (
                "\n".join(o.summary(max_chars=1200) for o in valid) or "无；请依据原图重新判断。")
            record["recovery_observation_ids"] = [o.result_id for o in valid]
        if failure_key:
            retry_counts[failure_key] = retry_counts.get(failure_key, 0) + 1
            retry_limit = (int(cfg.max_regen) if failure_key in ("ast", "synthesis")
                           else int(cfg.max_retries_per_operation))
            exhausted = retry_counts[failure_key] > retry_limit
            if exhausted and int(cfg.finalization_rounds) <= 0:
                termination_reason = "operation_retry_budget_exhausted"
                outcome.notes.append(
                    f"{failure_key} 重试预算耗尽，且未保留 finalization 轮")
                break
            force_next = exhausted
            trigger = "error_recovery"
        if feedback:
            feedback += f"\n剩余模型请求上限：{int(cfg.max_solver_rounds) - index}（含终结轮）。"
        if fsm.state is not OnlineState.SYNTHESIZE_PROGRAM:
            step("resynthesize")

    if kernel is not None:
        kernel.reset_user_namespace()
    if outcome.final_state != "unavailable":
        outcome.final_state = "run_error"
    last = outcome.rounds[-1] if outcome.rounds else {}
    outcome.failure_code = str(last.get("cause_code") or last.get("error_code") or "no_answer")
    stage = ("M8" if not last.get("program_source") and not last.get("executed")
             else "M9" if outcome.failure_code == "ast_violation"
             else "M11" if outcome.failure_code == "geometry_rejected" else "M10")
    outcome.terminal_failure = {
        "code": outcome.failure_code, "stage": stage,
        "termination_reason": termination_reason, "round": outcome.agent_rounds,
    }
    last["terminal_failure"] = dict(outcome.terminal_failure)
    outcome.notes.append(
        f"当前求解协议结束，无通过验收的答案；已用 "
        f"{outcome.agent_rounds}/{cfg.max_solver_rounds} 轮")
    step("solver_failed")
    return scene, handle, skills, kernel, trace


def _load_npy(ref: str):
    """读 npy ref（缺失/损坏返回 None，不伪造数组）。"""
    if not ref:
        return None
    try:
        return np.load(ref)
    except Exception:  # noqa: BLE001
        return None


def _overall_of(art) -> Optional[float]:
    """artifact 的实算 overall_quality（未计算 → None，不写 NaN 占位）。"""
    q = getattr(art, "quality", None)
    if getattr(art, "quality_status", "") != "computed" or q is None:
        return None
    val = float(q.overall_quality)
    return val if np.isfinite(val) else None


def _overall_str(art) -> str:
    v = _overall_of(art)
    return "not_computed" if v is None else f"{v:.3f}"


class _ReuseMiss(Exception):
    """内部信号：P1 落盘 artifact 不可复用（不存在 / 帧集不符）→ 走重算分支。"""


def _resolve_existing_artifact(cfg: OnlineRunConfig, episode) -> tuple[str, bool]:
    """复用 P1 产物时的定位：先按缓存身份，再按历史命名（§17.2 迁移）。

    返回 `(路径, 是否旧命名)`；旧命名产物仍可复用，但调用方据此知道它不是按新缓存
    身份命名的（同名 scene 跨视频的顶用风险只在旧产物上存在）。
    """
    from skill3d.reconstruction.run import resolve_artifact_path

    frame_set = getattr(episode, "frame_set", None)
    p, used_legacy = resolve_artifact_path(
        cfg.recon_dir, episode.scene_name, cfg.recon_method, frame_set=frame_set)
    return str(p), used_legacy


def _scene_from_artifact(art, pixels, m5_stats: Optional[dict] = None,
                         objects: Optional[list] = None,
                         episode=None,
                         artifact_path: str = "",
                         objects_materialized: Optional[bool] = None,
                         persist_quality: bool = True
                         ) -> tuple[SceneState, SceneHandle, object]:
    """M4 真实路径：从 artifact 读数组 → v6 主门质量 → SceneState + SceneHandle。

    硬约束 22（质量单一事实源）：`quality_status == "computed"` 时直接复用落盘实算值
    （P1 方案 X，零重算）；否则实算并**原子写回** `artifact_path`（方案 Y）。

    v6 数据源接线：主门用 frames/depth/poses/K/point_map/depth_conf；
    诊断用 M5 的 dynamic_masks / track_ious。G5/G8/G11 都已不存在，**不得**补位。
    """
    stats = m5_stats or {}
    depth, c2w, intr = (_load_npy(art.depth_maps), _load_npy(art.c2w_list),
                        _load_npy(art.intrinsics))
    point_map, dconf = _load_npy(art.point_map), _load_npy(art.depth_conf)

    m5_summary = _m5_evidence_summary(stats, objects, episode, m5_materialized=objects_materialized)

    if str(getattr(art, "quality_status", "not_computed")) != "computed":
        from skill3d.reconstruction_gate import quality_metrics as qm

        art = qm.compute_and_store_quality(
            art, frames=pixels, depth_maps=depth, c2w_list=c2w, intrinsics=intr,
            point_map=point_map, depth_conf=dconf,
            dynamic_masks=stats.get("dynamic_masks"),
            track_ious=stats.get("track_ious"),
            artifact_path=(artifact_path if persist_quality else None),
        )
    scene = build_scene_state(
        art,
        m5=m5_summary,
        objects=sorted(o.obj_id for o in (objects or [])),
        artifact_ref=art.artifact_id,
    )
    handle = SceneHandle(scene, objects=objects or [], c2w_list=c2w, intrinsics=intr,
                         quality_overall=_overall_of(art),
                         objects_materialized=objects_materialized,
                         grounding_coverage=stats.get("grounding"))
    # 世界系点图注入句柄（平面拟合 / 连通性图的唯一数据入口）
    handle.set_point_map(point_map)
    # v9 §9.4：把**本 episode 实际可读的帧**装进句柄的图像账本 ——
    # `inspect_frames` 只允许从这套冻结帧集裁剪/查看（不得重新采样视频）。
    _attach_episode_frames(handle, episode, pixels)
    return scene, handle, art


def _m5_evidence_summary(stats: dict, objects: Optional[list], episode,
                         *, m5_materialized: Optional[bool] = None,
                         grounding=None) -> M5EvidenceSummary:
    """把 M5 统计压成证据输入（§7.1 的 object_detection / track_consensus / grounding）。

    不确定的一律留 `None`/False（fail-closed：证据不足落 unavailable，不做乐观默认）。
    """
    objs = list(objects or [])
    fault = bool((stats or {}).get("detector_fault", False))
    if m5_materialized is False:
        fault = True
    tcm = (stats or {}).get("track_consensus") or {}
    ratio = (stats or {}).get("track_stable_ratio")
    n_tracks = len({str(o.track_id) for o in objs if getattr(o, "track_id", None)})
    if n_tracks == 0 and objs:
        # 未记录 track_id 时退回"去重后实例数"作 track 数（不伪造稳定占比）
        n_tracks = len(objs)

    def _opt(v):
        return None if v is None else float(v)

    # §7.1 `object_grounding`：M5 逐题补漏的**实际结果**（不是"没查"）
    g = (stats or {}).get("grounding") or grounding or {}
    if g.get("attempted"):
        if g.get("miss") or g.get("status") in ("missing", "fault"):
            hit, miss = False, True
        elif g.get("status") in ("partial", "ambiguous"):
            hit, miss = None, False
        elif g.get("all_present"):
            hit, miss = True, False
        else:
            # 框数/新增数不能证明全部需求已覆盖；旧统计缺覆盖记录时保守降级。
            hit, miss = None, False
    else:
        hit, miss = None, False      # 未做校验 → 上层按 degraded（不是 available）
    return M5EvidenceSummary(
        # v9 §6.1：这里是把 M5 汇总成证据摘要的地方 —— 它被调用即说明 M5 运行过
        m5_ran=True,
        detection_fault=fault,
        n_objects=len(objs),
        n_tracks=int(n_tracks),
        track_stable_ratio=(None if ratio is None else float(ratio)),
        track_fragmentation_ratio=_opt(tcm.get("track_fragmentation_ratio")),
        duplicate_suspect_ratio=_opt(tcm.get("duplicate_suspect_ratio")),
        grounding_pointed_hit=hit,
        grounding_conf=_opt(g.get("conf")),
        grounding_miss=miss,
        grounding_filled=bool(int(g.get("n_new", 0)) > 0),
        notes=[str(x) for x in ((stats or {}).get("evidence_notes") or [])],
    )


def _mean_track_iou(m5_stats: Optional[dict]) -> Optional[float]:
    """G9 门控输入：SAM2 mask 跨帧 IoU 均值；无数据 → None（不额外门控）。"""
    ious = (m5_stats or {}).get("track_ious")
    if not ious:
        return None
    vals = [float(v) for v in ious if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _configure_image_ledger(handle: Optional[SceneHandle], cfg: OnlineRunConfig,
                            episode) -> None:
    """把 run 配置的主动图像参数装进账本（§9.4：布局与上限必须来自**声明**的配置）。"""
    if handle is None:
        return
    ledger = handle._ledger  # noqa: SLF001
    ledger.episode_id = str(getattr(episode, "qa_id", "") or "")
    ledger.max_images = max(1, int(cfg.max_images))
    ledger.max_derived_images = max(0, int(cfg.max_derived_images))
    if str(cfg.image_layout) != str(ledger_layout := LAYOUT_DERIVED_PLUS_ORIGINALS):
        # 目前只实现了一种声明布局；配置写了别的名字要在 trace 里看得出来，
        # 而不是静默按默认布局跑（"事先声明的布局"不能名不副实）。
        ledger.schema_version = f"{ledger.schema_version}+layout:{cfg.image_layout}"


def _attach_episode_frames(handle: SceneHandle, episode, pixels: Sequence[np.ndarray]) -> None:
    """把 episode 的**实际可读帧**装进句柄图像账本（§9.4 唯一裁剪来源）。

    `pixels` 是本 episode 实际可读的像素（P5f：部分可读时是规划序下的**可读子集**），
    与 `frame_set.readable_frame_ids` 一一对应 —— 不假设 32 帧齐全。

    **帧身份用槽位序号**（0..n-1）：模型与其它工具（`reproject(..., frame_idx)` /
    `object_visible_frames`）都按"第几张"指帧，`FrameSet.frame_ids` 里的物理源帧号
    （真实视频是 0/116/233… 这些采样点）不是模型能对上号的 —— 实测踩过：模型写
    `inspect_frames([8])` 意思是"第 8 张"，按物理号找就直接 KeyError。
    物理源帧号另记进账本 `source_frame_indices`，供审计"这张图来自视频哪一帧"。
    """
    if handle is None or not pixels:
        return
    fs = getattr(episode, "frame_set", None)
    ids = list(getattr(fs, "readable_frame_ids", None) or [])
    if not ids or len(ids) != len(pixels):
        # 锚不下来就宁可**不装**（避免把槽位与像素错配成假映射）
        return
    all_ids = list(getattr(fs, "frame_ids", None) or [])
    # 可读槽位 → 物理源帧号（readable_frame_ids 是 frame_ids 的子集，顺序一致）
    pos = {int(f): k for k, f in enumerate(all_ids)}
    phys = [k for k in (pos.get(int(f)) for f in ids)]
    phys = [p for p in phys if p is not None]
    handle._set_frames(list(range(len(pixels))), list(pixels),  # noqa: SLF001
                       source_frame_indices=phys if len(phys) == len(pixels) else None)


def _retarget_handle(handle: Optional[SceneHandle], scene: SceneState) -> Optional[SceneHandle]:
    """逐题 scope 派生后同步句柄的 SceneState（Tool 与 Verifier 都读它）。

    v6：产物集与证据画像随 scope 变化；点图缓存沿用同一份（不必重读数组）。
    """
    if handle is None:
        return None
    new_handle = SceneHandle(scene, objects=list(handle._objects.values()),  # noqa: SLF001
                             c2w_list=handle._c2w, intrinsics=handle._k,     # noqa: SLF001
                             quality_overall=handle.quality_overall,
                             objects_materialized=handle.objects_materialized,
                             metric_scale=handle.metric_scale,
                             grounding_coverage=handle.grounding_coverage)
    new_handle.set_point_map(handle.get_point_map())
    # v9 §9.4：逐题 scope 重派生**不得丢**帧集与已产出的图像（同一 episode 的账本）
    new_handle._set_image_ledger(handle._ledger)  # noqa: SLF001
    # 内存态对象点集（合成路径注入过的）随句柄一起搬，避免逐题重派生后丢点云
    for oid, pts in getattr(handle, "_points_cache", {}).items():  # noqa: SLF001
        new_handle.set_object_points(oid, pts)
    return new_handle


@capture_detector_attempts()
def _bind_objects_best_effort(art, pixels, _scene, episode, cfg=None,
                              llm=None, point_conf=None) -> tuple[list, dict, list[str], bool]:
    """M5 对象绑定：SAM2 未配置等失败 → 记降级、不阻断（§4 M5 字段 9）。

    返回 `(objects, stats, notes, materialized)`：
    - `stats` 供 M4 的 G7/G9 使用（G-18 接线）；
    - `materialized` 表示 objects 产物**是否已产出**（硬约束 23 的判据）：
      绑定流程跑完（哪怕一个对象都没绑到）→ True；未跑/抛异常 → False
      （此时 `exists_in_scene` 抛 `ArtifactUnavailableError` 而不是假装"场景里没有"）。
    """
    from skill3d.segmentation.sam2_tracker import bind_objects_for_scene

    depth = _load_npy(art.depth_maps)
    c2w = _load_npy(art.c2w_list)
    intr = _load_npy(art.intrinsics)
    if point_conf is None:
        # 逐点置信度（§12.2 软权重来源）；读不到就传 None（**不**伪造 1.0）
        point_conf = _load_npy(art.depth_conf) if art.depth_conf else None
    if depth is None or c2w is None:
        return [], {}, ["M5 跳过：artifact 缺深度/位姿数组（无法反投影到世界系）"], False
    # VLM 框提示：真实模式且有 vLLM 时，由在线 Qwen3-VL-8B 给对象名+bbox（§4 M5 字段 12）
    vlm_client = None
    if cfg is not None and cfg.mode == "real":
        vlm_client = llm if llm is not None else _make_vllm_client(cfg)
    try:
        objects, stats, notes = bind_objects_for_scene(
            pixels, None, episode.question, depth_maps=depth, c2w_list=c2w,
            question_type=canonical_task(episode.question_type),
            options=episode.options,
            intrinsics=intr, out_dir=_m5_out_dir(art), scene_name=art.scene_name,
            # v6：BA 已废止 → 深度网格恒为纯缩放（mask_to_grid 走 resize_mask_nearest），
            # 不再需要 grid_transform 仿射映射（§20 废止表）
            grid_transform=None,
            point_conf=point_conf,
            vlm_client=vlm_client,
            # 场景清单缓存键：同一 (scene, FrameSet) 复用对象清单（M5 是 scene 级产物）
            frame_set_hash=str(getattr(art, "frame_set_hash", "") or ""),
            # 请求级 seed：M5 的 VLM 框提示也要可复现（§7 复现纪律）
            seed=int(cfg.seed),
        )
    except Exception as exc:  # noqa: BLE001 - checkpoint/依赖/传播失败
        return [], {"binding_error": f"{type(exc).__name__}: {exc}",
                    "detector_attempts": current_detector_attempts()}, [
                        f"M5 对象绑定不可用（{type(exc).__name__}: {exc}）→ "
                        "相关 Tool 查询回退全场景（TODO_USER_INPUT: SAM2 checkpoint，§7.1 G-19）"], False
    stats["detector_attempts"] = current_detector_attempts()
    if any(a.get("status") == "error" and not a.get("will_retry")
           for a in stats["detector_attempts"]):
        stats["detector_fault"] = True
    return objects, stats, notes, True


def _m5_out_dir(art) -> Optional[str]:
    """M5 产物目录：与 artifact 数组同目录（mask/对象点云 ref 便于审计与回读）。"""
    try:
        from pathlib import Path as _P

        if art.depth_maps:
            return str(_P(art.depth_maps).parent)
    except Exception:  # noqa: BLE001
        return None
    return None


def _prompt_messages(prompt: str, pixels, cfg: OnlineRunConfig,
                     expected_frames: Optional[int] = None,
                     extra_images: Optional[Sequence] = None) -> list[dict]:
    """文本 prompt → OpenAI messages；`mode=real` 必须带图（多模态，硬约束 26）。

    `mode=real` 下缺帧或帧数与统一 FrameSet 不一致时**直接报错**：既不允许静默
    退化纯文本，也不允许少送帧（硬约束 21/26；`build_image_messages` 另有一道校验）。

    v9 §9.4：`extra_images` = `[(image_id, ndarray), ...]` 本轮要一并送达的**派生图**。
    `expected_frames` 校验的是**本轮实际装入的原帧数**（声明的布局可能少于 FrameSet，
    少装的原帧已在更早轮次交付过，由账本逐条记录，不是静默丢帧）。
    """
    if cfg.mode == "real":
        n = len(pixels) if pixels is not None else 0
        if n == 0:
            raise ValueError(
                "M8 real 模式收到 0 帧：硬约束 26 要求多模态（32 帧 + 文本），"
                "禁止静默退化纯文本生成程序")
        if expected_frames is not None and n != expected_frames:
            raise ValueError(
                f"M8 收到 {n} 帧但本轮布局声明原帧数为 {expected_frames}"
                "（硬约束 21/26：禁止双帧集/静默丢帧）")
        from skill3d.synthesis.prompt_builder import build_image_messages

        return build_image_messages(prompt, pixels, max_images=cfg.max_images,
                                    extra=extra_images)
    return [{"role": "user", "content": prompt}]


def _round_dict(ledger, round_index: int) -> dict:
    """本轮图像清单的落盘形态（无账本/无该轮 → 空 dict，不造假记录）。"""
    if ledger is None:
        return {}
    for r in getattr(ledger, "rounds", []) or []:
        if r.round_index == int(round_index):
            return r.model_dump()
    return {}


def _yield_image_ids(cell: CellResult, kernel, handle) -> list[str]:
    """§9.4：上一轮 `YieldObservations` 点名结果里**产出的图像** → 本轮要送达的派生图。

    只取被点名结果产出的图（模型明确要看的东西）；工具产出但没被点名的图不抢名额。
    """
    ledger = getattr(handle, "_ledger", None) if handle is not None else None  # noqa: SLF001
    if ledger is None or not ledger.has_frames():
        return []
    wanted = list(getattr(cell, "yielded_result_ids", []) or [])
    ids = [i for i in ledger.image_ids_for_results(wanted) if not i.startswith("frame-")]
    if not ids and wanted and kernel is not None:
        # 点名的是 **result_id**（prompt 教的写法）：从该结果的 payload 里取 image_id
        payload_ids: list[str] = []
        for r in (getattr(kernel, "tool_results", []) or []):
            if str(getattr(r, "result_id", "") or "") not in set(wanted):
                continue
            for item in ((getattr(r, "payload", None) or {}).get("images") or []):
                iid = str((item or {}).get("image_id", "") or "")
                if iid:
                    payload_ids.append(iid)
        ids = [i for i in payload_ids if not i.startswith("frame-")]
    if not ids:
        # 兜底：本轮**新产出且从未交付**的派生图（模型让出但没点名具体图）。
        # 有界（≤ max_derived_images）且不含陈图，避免把旧裁剪反复塞进请求。
        ids = ledger.undelivered_images_produced_in_round(int(ledger.current_round))
    return ids


def _image_manifest_text(ledger, plan) -> str:
    """本轮图像的**明确映射**（§9.4"带明确映射的图像布局"）。

    模型必须能把自己看到的每张图与"源帧/裁剪框/尺寸"对上，否则裁剪图与一张无标签的
    小图没有区别。顺序与请求里的图像顺序**严格一致**（派生图在前，原帧随后）。
    """
    if not plan.image_ids:
        return ""
    lines = ["", "## 本轮图像清单（按顺序，与本次请求里的图片一一对应）"]
    idx = 0
    for i in plan.derived_image_ids:
        idx += 1
        rec = ledger.get(i)
        lines.append(
            f"{idx}. [裁剪图] image_id={rec.image_id} 源帧={rec.source_frame_id} "
            f"裁剪框(x0,y0,x1,y1)={rec.box_xyxy} 源尺寸={rec.source_hw} "
            f"送达尺寸={rec.sent_hw} scale={rec.scale}")
    for i in plan.original_image_ids:
        idx += 1
        lines.append(f"{idx}. [原帧] frame_id={i.split('-', 1)[1]}")
    if plan.omitted_originals:
        lines.append(
            f"（此外还有 {len(plan.omitted_originals)} 张原帧本轮未再重复发送："
            "它们已在更早的请求里给过你，本轮为派生图让出名额）")
    return "\n".join(lines) + "\n"


def _synthesize(episode: VSIBenchEpisode, scene, handle, skills, cfg: OnlineRunConfig,
                llm, feedback: Optional[list[str]] = None,
                geometry: Optional[synth.SyntheticGeometry] = None,
                pixels: Optional[Sequence[np.ndarray]] = None,
                scope_override: Optional[str] = None,
                traceback_feedback: Optional[str] = None,
                prior_program: Optional[str] = None,
                observation_image_ids: Optional[Sequence[str]] = None,
                round_index: int = 1,
                trigger: str = "initial") -> _SynthResult:
    """M8：生成 program。

    `scope_override` / `traceback_feedback` / `prior_program` 供 D-3 恢复层使用：
    裁剪 prompt（强制受限 route）或把"上一轮 program + 裁剪 traceback"作为
    额外 turns 回灌（§4 M6 字段 9）。

    v9 §9.4 主动图像：`observation_image_ids` 是上一轮 `YieldObservations` 点名结果
    产出的**派生图**（`inspect_frames` 的裁剪）。它们按声明布局进本次请求，并在
    账本里记 `delivered`（请求发出）/`observed`（响应返回）；图像清单与 token 成本
    一并落盘。没进请求的图保持 unobserved（不静默丢图）。
    """
    if cfg.mode == "mock_light":
        # 有合成几何用合成几何；否则（冻结真实 artifact / golden 重放）用 handle 对象名
        names = None
        if geometry is None:
            if handle is None:
                raise ValueError("mock_light 需要 geometry 或已建立的 SceneHandle")
            names = [_class_hint_of(handle, oid) for oid in handle.list_objects()]
        src = synth.stub_program(episode.question_type, episode, geometry,
                                 object_names=names)
        return _SynthResult(
            assemble_program(src, []),
            "mock_stub",
            "M8 mock_light 确定性 stub program"
            "（非 Qwen3-VL-8B 输出，仅管道验证）",
            response_text=src,
        )

    prompt, delivery = _build_prompt_ex(episode, scene, handle, skills, feedback,
                                        scope_override=scope_override,
                                        # 带 prior_program 时反馈作为下一条 user turn 发送；
                                        # 不再同时复制进首条公共 prompt。
                                        traceback_feedback=(
                                            None if prior_program is not None
                                            else traceback_feedback),
                                        policy=cfg.retrieval_policy)
    client = llm if llm is not None else _make_vllm_client(cfg)
    if client is None:
        # §13.6：prompt 里放了方法，但**没有发出任何请求** → 一条都不算"已交付"。
        delivery.channel_note = "M8 vLLM 未配置：方法进了 prompt 但未发出模型请求"
        return _SynthResult(None, "none",
                            "M8 vLLM 未配置：用 --vllm-endpoint 指定本地 Qwen3-VL-8B endpoint"
                            "（bash scripts/serve_qwen3vl_dp8.sh）；§4 M8 字段 9 → 记 unavailable",
                            prompt=prompt, delivery=delivery)
    # ---- v9 §9.4：主动图像的布局规划（声明布局，图数不超过服务上限）----
    ledger = getattr(handle, "_ledger", None) if handle is not None else None  # noqa: SLF001
    extra_images: list = []
    frames_for_request = list(pixels or [])
    if ledger is not None and ledger.has_frames():
        plan = ledger.plan_round(
            round_index=int(round_index), trigger=str(trigger),
            derived_image_ids=list(observation_image_ids or []),
            original_frame_ids=ledger.frame_ids)
        extra_images = [(i, ledger.pixels_of(i)) for i in plan.derived_image_ids]
        # 本轮实际装入的原帧 = 布局选中的那些（部分观察轮会少于 FrameSet，
        # 少掉的原帧在更早轮次已交付并留痕，不是静默丢帧）
        chosen_orig = {int(i.split("-", 1)[1]) for i in plan.original_image_ids}
        frames_for_request = [p for fid, p in zip(ledger.frame_ids, frames_for_request)
                              if fid in chosen_orig]
        prompt += _image_manifest_text(ledger, plan)

    # 多模态：把本轮布局规定的帧（+ 派生图）连同 prompt 一起送进模型
    expected = (len(frames_for_request) if getattr(episode, "frame_set", None) is not None
                else None)
    messages = _prompt_messages(prompt, frames_for_request, cfg, expected_frames=expected,
                                extra_images=extra_images)
    n_images = len([1 for m in messages
                    if isinstance(m.get("content"), list)
                    for part in m["content"]
                    if isinstance(part, dict) and part.get("type") == "image_url"])
    if ledger is not None and ledger.has_frames():
        # §9.4 delivered = "进了实际发出的请求"：请求发出前的最后一刻标记
        ledger.mark_round_sent(int(round_index))
        ledger.mark_delivered(int(round_index), plan.image_ids)
    if prior_program is not None:
        # 回灌：assistant(上一轮 program) → user(裁剪 traceback + 可用产物)
        messages = messages + [{"role": "assistant", "content": prior_program}]
        if traceback_feedback:
            messages.append({"role": "user", "content": traceback_feedback})
    try:
        # 请求级 seed：钉住 vLLM 采样随机性（§7 复现纪律）
        with request_phase("program"):
            text = client.chat(messages, max_tokens=cfg.max_tokens, seed=int(cfg.seed))
    except Exception as exc:  # noqa: BLE001 - 网络/服务不可用
        from skill3d.synthesis.vllm_client import ServiceUnavailable

        # §13.6：请求失败不能记成"已交付"（模型可能根本没收到正文）。
        delivery.mark_request_failed(f"{type(exc).__name__}: {exc}")
        if ledger is not None and ledger.has_frames():
            # §9.4：请求失败 → 本轮图像保持 **unobserved**，并写明原因（不静默）
            ledger.mark_round_failed(int(round_index), f"{type(exc).__name__}: {exc}")
        if isinstance(exc, ServiceUnavailable):
            return _SynthResult(None, "vllm_service_error",
                                f"M8 service_unavailable（§6.1：服务故障不静默降级）: {exc}",
                                prompt=prompt, n_images=n_images, delivery=delivery)
        return _SynthResult(None, "vllm_service_error",
                            f"M8 vLLM 调用失败: {type(exc).__name__}: {exc}",
                            prompt=prompt, n_images=n_images, delivery=delivery)

    # 请求已发出且模型有返回 → 此刻方法正文确实送达（§13.6"已交付"）
    delivery.mark_delivered()
    # §9.4：模型收到实际内容并完成本轮响应 → 这些图才算 **observed**（仍不证明理解正确）
    if ledger is not None and ledger.has_frames():
        usage = dict(getattr(client, "last_usage", None) or {})
        ledger.mark_round_observed(int(round_index),
                                   prompt_tokens=usage.get("prompt_tokens"))

    # 退化输出由同一个有界求解循环计为一次失败，不在请求函数内暗加重试。
    reason = degenerate_reason(text)
    if reason is not None:
        return _SynthResult(
            None, "vllm_parse_error", f"M8 退化输出：{reason}（交由有界求解循环重试）",
            prompt=prompt, n_images=n_images, delivery=delivery,
            image_round=_round_dict(ledger, round_index), response_text=text)

    try:
        # `skill_semver_used` 承载的是"这一轮**实际交付**给模型的方法集"（v8 以来的
        # 上下文口径）。v9（§13.6）把它收紧到交付计划里的条目：此前这里传的是
        # "检索选中的全部 Skill"，上下文上限丢掉的条目也会被记成"用过"。
        # 模型**自称**使用了哪些方法另有 `declared_selected_skill_versions`，两者不互替。
        program, recovered = assemble_program_ex(
            text, [e.skill_version for e in delivery.entries])
    except SynthesisError as exc:
        return _SynthResult(
            None,
            "vllm_parse_error",
            f"M8 program 解析失败: {exc}",
            prompt=prompt,
            n_images=n_images,
            delivery=delivery,
            image_round=_round_dict(ledger, round_index),
            response_text=text,
        )
    # §15.2：解析回退单列 m8_parse_recovered，与"真解析不出来"区分开
    source = "m8_parse_recovered" if recovered else "vllm_ok"
    return _SynthResult(
        program,
        source,
        "",
        n_images=n_images,
        prompt=prompt,
        delivery=delivery,
        image_round=_round_dict(ledger, round_index),
        response_text=text,
    )


def _class_hint_of(handle, obj_id: str) -> str:
    """取 handle 中某对象的 class_hint（stub program 需要对象名做锚点）。"""
    try:
        return str(handle.get_object(obj_id).class_hint)
    except Exception:  # noqa: BLE001
        return obj_id


def _build_prompt(episode, scene, handle, skills, feedback, *,
                  scope_override: Optional[str] = None,
                  traceback_feedback: Optional[str] = None,
                  policy: Optional[RetrievalPolicy] = None) -> str:
    """M8 prompt：只给 SceneState 摘要 + Tool 文档 + Skill 模板，绝不含 GT（§4 M8）。

    兼容入口：只要文本。需要交付记录（§13.6）的调用方用 `_build_prompt_ex`。
    """
    return _build_prompt_ex(episode, scene, handle, skills, feedback,
                            scope_override=scope_override,
                            traceback_feedback=traceback_feedback,
                            policy=policy)[0]


def _build_prompt_ex(episode, scene, handle, skills, feedback, *,
                     scope_override: Optional[str] = None,
                     traceback_feedback: Optional[str] = None,
                     policy: Optional[RetrievalPolicy] = None
                     ) -> tuple[str, SkillDeliveryPlan]:
    """M8 prompt + **交付计划**（§13.5/§13.6）。

    - `tool_docs = REGISTRY.docs(route=...)`：按 route 静态裁剪（D-3a），
      prompt 头部显式写"当前 route=…，可用产物=…"；
    - `scope_override`：tool_contract 恢复层的"裁剪重生成"（强制 fallback_2d_only）；
    - `traceback_feedback`：回灌层的裁剪 traceback + 可用产物清单；
    - `policy`：冻结的检索策略，其 `method_context_max_chars` 决定**方法上下文上限**；
      放不下的方法条目**整条**丢弃（记 `context_cap_exceeded`），不截断正文。
    """
    scope = scope_override or (scene.question_tool_scope if scene is not None else "")
    # prompt 头部的"可用产物"必须是**实际装载**的集合（硬约束 23）：
    # `scene.available_artifacts` 是 scope 级声明，M5 失败/数组未加载时会高报，
    # 模型照着写就会撞 ArtifactUnavailableError。
    if handle is not None:
        available = sorted(handle.available_artifacts)
    else:
        available = sorted(getattr(scene, "available_artifacts", set()) or set())
    raw_qtype = (str(getattr(scene, "question_type", "") or "")
                 or str(episode.question_type or ""))
    qtype = canonical_question_type(raw_qtype) or raw_qtype
    profile = getattr(scene, "evidence_profile", None) if scene is not None else None
    gate = getattr(scene, "metric_evidence_gate_result", None) if scene is not None else None
    plan = plan_delivery(
        skills, max_chars=(policy.method_context_max_chars if policy is not None else None))
    scene_summary = scene.summary if scene is not None else ""
    if handle is not None and handle.grounding_coverage:
        scene_summary += "\nM5 初始对象覆盖（后续新增对象以工具清单为准）: " + json.dumps(
            handle.grounding_coverage, ensure_ascii=False, sort_keys=True)
    text = PromptBuilder().render(
        question=episode.question,
        scene_summary=scene_summary,
        scene_frame="world",
        # 静态裁剪：scope + EvidenceProfile + 句柄实际装载产物 三条件，
        # 使 prompt 列出的 Tool 在执行期一定不会因缺产物/证据不足抛错
        tool_docs=REGISTRY.docs(scope or None,
                                available=(handle.available_artifacts
                                           if handle is not None else None),
                                evidence_profile=profile,
                                gate_passed=(None if gate is None else gate.gate_passed),
                                question_type=qtype),
        options=episode.options,
        skills=skills,
        scope=scope,
        available_artifacts=available,
        question_type=qtype,
        evidence_profile=profile,
        gate_passed=(None if gate is None else gate.gate_passed),
        gate_missing=(None if gate is None else gate.missing_subconditions),
        skill_plan=plan,
    )
    if feedback:
        text += f"\n上一次生成被 AST 拒绝，原因：{feedback}\n请修正后重新输出。"
    if traceback_feedback:
        text += f"\n{traceback_feedback}"
    return text, plan


def _answer_source_v6(*, answer: Optional[str], program, kernel,
                      outcome: EpisodeOutcome) -> str:
    """Attribute accepted answers to the program; retain failure distinctions."""
    if answer is not None and str(answer) != "":
        if outcome.eval_visual_fallback.get("accepted"):
            return "direct_vlm_routed"
        from_program = (kernel is not None
                        and str(getattr(program, "program_source", "") or "") != "")
        return "tool_program" if from_program else "abstain"
    if outcome.failure_code == "tool_contract" or int(outcome.tool_contract_hits) > 0:
        return "tool_contract"
    return "abstain"


def _failed_tool_of(cell: CellResult) -> str:
    """从 cell 的契约违规记录里取"第一个失败的 Tool"（归因用）。"""
    for v in reversed(list(getattr(cell, "contract_violations", []) or [])):
        name = str(v.get("source_tool") or v.get("tool") or "")
        if name:
            return name
    return ""


def _premise_from_cell(cell: CellResult, kernel) -> Optional[str]:
    """§14.1：失败是否揭示共享前提失效；`None` = 局部失败（不级联撤销）。

    读最后一条契约违规记录的 `error_code` + `missing_artifacts`，按
    `recovery.premise_of_failure` 的判据推断。没有任何记录时返回 None
    （信息不足 → 不做破坏性的级联撤销，交给重生成修复）。
    """
    viols = list(getattr(cell, "contract_violations", []) or [])
    if not viols:
        return None
    v = viols[-1]
    name = str(v.get("source_tool") or v.get("tool") or "")
    try:
        req = tuple(REGISTRY.requires_evidence(name))
    except Exception:  # noqa: BLE001 - 未注册名 → 无法判前提，保守不撤销
        req = ()
    # `missing_artifacts` 里带括号的是**证据项**（如 "metric_scale(degraded 且不容忍)"），
    # 不带括号且命中产物表的才是**产物项**。两者判据不同，不能混在一起。
    raw_missing = [str(x) for x in (v.get("missing_artifacts") or ())]
    unmet = tuple(x for x in raw_missing if "(" in x)
    arts = tuple(x for x in raw_missing if "(" not in x)
    return premise_of_failure(
        error_code=str(v.get("error_code") or "tool_contract"),
        tool=name, requires_evidence=req,
        missing_artifacts=arts, unmet_evidence=unmet)


def _make_vllm_client(cfg: OnlineRunConfig):
    if not cfg.vllm_endpoints:
        return None
    from skill3d.synthesis.vllm_client import VLLMClient

    return VLLMClient(cfg.vllm_endpoints, cfg.vllm_model)


def _execute_program(episode, program, handle, pixels, cfg: OnlineRunConfig, receipts,
                     kernel: Optional[RestrictedNamespaceKernel] = None,
                     tools_enabled: Optional[bool] = None):
    """M10：建/复用 kernel → 执行 program → 收集 ProgramExecutionTrace（§5.4）。

    `kernel` 非空时**先 reset user namespace 再重注入**（SpatialClaw §E.3 先例）：
    D-3 的回灌重执行必须从干净状态起跑，per-episode 状态不得跨次执行泄漏。
    """
    mock_switch = MockSwitch(light_handle=handle) if cfg.mode != "real" else None
    if kernel is not None:
        kernel.reset_user_namespace()
        kernel._mock_switch = mock_switch  # noqa: SLF001 - 重注入 mock 句柄
        if handle is not None:
            kernel.retarget_scene(handle)
    else:
        kernel = RestrictedNamespaceKernel(
            REGISTRY,
            handle,
            frames=pixels,
            mode="real" if cfg.mode == "real" else "mock_light",
            cell_timeout_s=cfg.cell_timeout_s,
            mock_switch=mock_switch,
            call_id_factory=_Counter() if cfg.deterministic_replay else None,
            episode_id=str(getattr(episode, "qa_id", "") or ""),
        )
    if tools_enabled is not None:
        kernel.set_tools_enabled(tools_enabled)
    receipts.append("sandbox_start", {"episode_id": episode.qa_id, "mode": cfg.mode})
    cell = kernel.run_cell(program.program_source)
    # 逐轮视图：账本跨轮只增不减（§17.1），但本轮 trace 只报本轮产生的调用/结果
    results = list(kernel.cell_results)
    if cfg.deterministic_replay:
        # 重放确定性：latency_ms 是实测值，置零以保证同 seed 字节级一致（§4 M17）
        results = [r.model_copy(update={"latency_ms": 0.0}) for r in results]
    program_trace = ProgramExecutionTrace(
        program_id=program.program_id,
        calls=list(kernel.cell_calls),
        results=results,
        stdout_tail=cell.stdout_tail,
        error_code=cell.error_code,  # type: ignore[arg-type]
        steps=len(results) + 1,
        wallclock_s=0.0,
    )
    for r in results:
        receipts.append("tool_call",
                        {"tool": r.tool, "args": r.args, "digest": r.request_digest},
                        source=r.source)
    return kernel, cell, program_trace


def _yield_feedback(cell: CellResult, kernel, scene,
                    cfg: OnlineRunConfig) -> str:
    """构造 yield 后的观察反馈（v7 §11.1/§11.3）。

    硬要求：反馈必须包含**实际 payload 内容**，不能只给"成功标记"，也不能只给模型
    读不到的服务器路径。这里把 `YieldObservations` 点名的结果按 `result_id` 取回，
    逐条给出 `工具名 + 参数 + 实际返回值`（失败结果给出错误码与原因），
    并附上"这一轮还剩多少预算"，让同一 agent 用这些信息写下一段程序。
    """
    wanted = list(getattr(cell, "yielded_result_ids", []) or [])
    # 点名结果按 result_id 从 **episode 账本**取回：id 在 episode 内唯一，早期轮次的
    # 结果也应能被点名（此前账本每轮被清空，只能解析出本轮结果）。
    by_id: dict[str, Any] = {}
    for r in (getattr(kernel, "tool_results", []) or []) if kernel is not None else []:
        rid = str(getattr(r, "result_id", "") or "")
        if rid:
            by_id[rid] = r

    lines = ["## 本轮观察（YieldObservations 回灌）"]
    if getattr(cell, "yielded_reason", ""):
        lines.append(f"让出理由：{cell.yielded_reason}")

    picked = [by_id[i] for i in wanted if i in by_id]
    if wanted and not picked:
        lines.append("（指定的 result_id 在本 episode 结果中不存在："
                     f"{wanted}；下面是本轮全部成功结果）")
        picked = [r for r in (getattr(kernel, "cell_results", []) or [])
                  if str(getattr(r, "status", "ok")) == "ok"]
    if not picked:
        # 未点名或没有可回灌结果 → 给出本轮全部成功结果，保证下一轮不空手
        picked = [r for r in (getattr(kernel, "tool_results", []) or [])
                  if str(getattr(r, "status", "ok")) == "ok"]
    if not picked:
        lines.append("（本轮没有任何成功的工具结果；请改用其他方法或直接视觉判断）")

    for r in picked:
        name = str(getattr(r, "source_tool", "") or getattr(r, "tool", "") or "?")
        rid = str(getattr(r, "result_id", "") or "")
        args = getattr(r, "args", None) or {}
        payload = getattr(r, "payload", None)
        if payload is None:
            raw = getattr(r, "value", None)
            try:
                payload = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:  # noqa: BLE001 - 非 JSON 就原样展示
                payload = raw
        body = json.dumps(payload, ensure_ascii=False, default=str)
        if len(body) > 1200:
            body = body[:1200] + "…(截断)"
        lines.append(f"- {name}({json.dumps(args, ensure_ascii=False, default=str)})"
                     f" [result_id={rid}] → {body}")
        flags = list(getattr(r, "degradation_flags", None) or [])
        if getattr(r, "invalidated_by", None):
            lines.append(f"  ⚠ 该结果已被级联撤销（invalidated_by="
                         f"{getattr(r, 'invalidated_by')}），**不得**再用于答案")
        if flags:
            lines.append(f"  降级标记：{flags}")
    for r in (getattr(kernel, "tool_results", []) or []):
        if str(getattr(r, "status", "ok")) != "ok":
            name = str(getattr(r, "source_tool", "") or getattr(r, "tool", "") or "?")
            lines.append(f"- {name} 调用失败：error_code="
                         f"{getattr(r, 'error_code', None)}"
                         f"（{str(getattr(r, 'error', '') or '')[:200]}）；"
                         "失败的工具不会再给出数值，请换方法或直接用图片判断")

    call_used = len(getattr(kernel, "tool_results", []) or [])
    rounds_left = max(0, int(cfg.max_solver_rounds) - int(cfg.finalization_rounds))
    lines.append(
        f"预算：本轮已观察工具调用 {call_used}（无独立工具总预算）；"
        f"后续最多还能让出/重试 {rounds_left} 轮。")
    lines.append(
        "请基于以上**实际返回值**继续写下一段 program：需要再让出就调用 "
        "`return YieldObservations([...], '理由')`；已经有把握就 "
        "`return ReturnAnswer(...)` 提交。有图必答：证据不足只改变求解方式，"
        "不改变必须作答这一条。")
    return "\n".join(lines)


_FORCED_ANSWER_FEEDBACK = (    "工具路径已经用尽（上一轮仍未能通过工具得到可用的量）。\n"
    "**本题必须作答**：图片是完整可读的，请把缺失的测量值改成你自己的视觉估计。\n"
    "请写一个**零工具** program：不要调用任何 Tool，直接依据你在这些图片里"
    "看到的空间布局和常识给出最佳答案，并用 `return ReturnAnswer(...)` 提交。\n"
    "只输出一个 ```python 代码块，不要解释。"
)


def _forced_answer_program(episode: VSIBenchEpisode, scene, handle, skills,
                           cfg: OnlineRunConfig, llm,
                           pixels: Optional[Sequence[np.ndarray]] = None,
                           geometry=None,
                           delivery_sink: Optional[list] = None,
                           observation_image_ids: Optional[Sequence[str]] = None,
                           round_index: Optional[int] = None,
                           trigger: str = "finalize"):
    """v7 D1：预算/恢复用尽时**必须产出答案**，而不是拒答。

    这是"有图必答"的最后一级：让**同一个在线 agent** 写一个零工具 program，
    把它已经看到的图片转成视觉估计（D2 允许零感知/计算工具调用，是因为视觉推断
    本身发生在模型看图、写代码的时候）。文件、解析、AST 全部复用正常路径，
    因此产出的仍是可审计的 program 轨迹，而不是一条旁路直答。

    `delivery_sink`：调用方传入列表即可拿到本次生成的**方法交付计划**（§13.6）——
    收口轮同样会交付方法，不记就会漏掉一轮的交付事实。
    """
    if cfg.mode == "mock_light":
        src = (f"{synth._STUB_BANNER}\n"
               f"ReturnAnswer({synth.stub_direct_answer(episode)!r})\n")
        return assemble_program(src, []), "mock_stub"
    try:
        res = _synthesize(episode, scene, handle, skills, cfg, llm,
                          feedback=None, geometry=geometry, pixels=pixels,
                          traceback_feedback=_FORCED_ANSWER_FEEDBACK,
                          observation_image_ids=observation_image_ids,
                          round_index=(1 if round_index is None else int(round_index)),
                          trigger=str(trigger))
    except Exception as exc:  # noqa: BLE001 - 兜底链自身失败不能冒泡成崩溃
        return None, f"forced_answer 生成异常: {type(exc).__name__}: {exc}"
    if delivery_sink is not None:
        delivery_sink.append(res.delivery)
    if res.program is None:
        return None, f"forced_answer 生成失败: {res.note}"
    return res.program, res.source


def _is_zero_tool_answer_program(program: EpisodeProgram) -> bool:
    source = str(getattr(program, "program_source", "") or "")
    check = ast_guard(source, allowed_tools=set())
    if not check.ok or check.allowed_tool_calls:
        return False
    try:
        tree = ast.parse(normalize_program_source(source))
    except SyntaxError:
        return False
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return (any(isinstance(node.func, ast.Name) and node.func.id == "ReturnAnswer"
                for node in calls)
            and not any(isinstance(node.func, ast.Name)
                        and node.func.id == "YieldObservations" for node in calls))


def _evaluate(episode: VSIBenchEpisode, answer: Optional[str], cls: TaskClassification):
    """M12：MCA → Accuracy；NA → MRA。不可解析 → 记 wrong / MRA 0（§4 M12 字段 9）。"""
    if cls.is_mca:
        pred = extract_option_letter(answer) if answer is not None else None
        correct = mca_correct(answer, episode.ground_truth) if answer is not None else False
        return pred, correct, None
    pred_f = parse_numeric_answer(answer)
    gt_f = parse_numeric_answer(episode.ground_truth)
    if pred_f is None or gt_f is None:
        return None, False, 0.0
    return f"{pred_f}", None, mra_single(pred_f, gt_f)


# ------------------------------------------------------------------ 收尾落盘 ----

def _finalize(outcome: EpisodeOutcome, fsm: OnlineFSM, cfg: OnlineRunConfig,
              episode: VSIBenchEpisode, states: list[str], receipts: ReceiptChain,
              trace_store: Optional[TraceStore], *, verdict=None,
              scene=None, handle=None) -> EpisodeOutcome:
    """收尾：failure 归因、receipts 链校验、EpisodeTrace 落盘（M13）。"""
    outcome.states = list(states)
    outcome.detector_attempts.extend(current_detector_attempts())
    # v9 §9.4：图像账本在这里**唯一**落一次（handle 只在此可见），保证每条 finalize
    # 路径都带上三态记录，不会因为某条 continue/break 分支漏掉。
    if handle is not None:
        try:
            # §9.4：episode 结束前，把"产出过但从未进过任何请求"的图盖章写明原因 ——
            # 未交付的材料保持 unobserved，但必须**有理由**（不静默丢图）。
            ledger_obj = getattr(handle, "_ledger", None)  # noqa: SLF001
            if ledger_obj is not None:
                ledger_obj.note_undelivered(
                    "episode 在发出携带该图的请求前结束（未交付）")
            outcome.image_ledger = dict(handle.ledger_view)
        except Exception:  # noqa: BLE001 - 账本投影失败不得阻断收尾
            outcome.image_ledger = {}
    outcome.answer_flags = list(dict.fromkeys(outcome.answer_flags + list(fsm.answer_flags)))
    if outcome.final_state == "input_error":
        outcome.answer_flags = [
            flag for flag in outcome.answer_flags
            if flag not in ("unanswerable", "abstain")
        ]
        outcome.answer_flags.append("input_error")
    receipts.append("episode_end", {"final_state": outcome.final_state,
                                    "qa_id": episode.qa_id})
    outcome.receipts = receipts.receipts
    outcome.receipts_ok = verify_chain(outcome.receipts)

    if not outcome.task:
        cls = classify(episode)
        outcome.task = cls.task
        outcome.is_mca = cls.is_mca
    if not outcome.is_mca and outcome.mra_value is None and outcome.correct is None:
        cls = classify(episode)
        outcome.is_mca = cls.is_mca
        if outcome.answer is not None:
            pred, correct, mra_value = _evaluate(episode, outcome.answer, cls)
            outcome.predicted, outcome.correct, outcome.mra_value = pred, correct, mra_value

    # ---- v7 D1：有图必答 —— 已经产出答案就不允许停在 unanswerable/abstain ----
    # 旧口径里只要 FSM 曾转过 `error`/`contract_fail`，episode 就是 `unanswerable`
    # 并按错计；但兜底层可能**已经**在同一 episode 里产出了答案（实测 smoke：
    # appearance_order 拿到 answer=B，却因为 final_state=unanswerable 被记成无答案）。
    # 有答案就是 answered：评分按答案算，格式问题单列 flag 另外报告
    # （§16.1：格式错误计 0 但**原因要分开报**，不能和"没有答案"混在一起）。
    if (outcome.answer is not None and str(outcome.answer).strip() != ""
            and outcome.final_state in ("unanswerable", "abstain")):
        outcome.final_state = "answer"
        outcome.abstained = False
        outcome.answer_flags = [f for f in outcome.answer_flags
                                if f not in ("abstain", "unanswerable")]
        if outcome.is_mca and outcome.predicted is None:
            outcome.answer_flags.append("answer_format_unparsed")
        if not outcome.is_mca and outcome.mra_value in (None, 0.0):
            outcome.answer_flags.append("answer_numeric_unparsed")

    # v3 §4 M6 字段 9 / §5.1：abstain 与 unanswerable **主榜按错计**
    # （MRA=0 / exact_match=0，不刷分）；只有 unavailable（模型/服务不可用）不进分母。
    if outcome.final_state in ("unanswerable", "abstain", "run_error", "input_error"):
        cls_wrong = classify(episode)
        outcome.is_mca = cls_wrong.is_mca
        if cls_wrong.is_mca:
            outcome.correct = False
            outcome.predicted = None
        else:
            outcome.mra_value = 0.0
            outcome.correct = None
        if outcome.final_state not in ("run_error", "input_error"):
            outcome.abstained = True
            if "abstain" not in outcome.answer_flags:
                outcome.answer_flags.append("abstain")
    outcome.episode_status = _episode_status_of(outcome.final_state)

    outcome.frame_set_hash = (episode.frame_set.frame_set_hash
                              if episode.frame_set is not None else "")
    profile = outcome.evidence_profile
    evidence_states = (profile.as_signature() if profile is not None else {})
    gate = outcome.metric_evidence_gate_result
    q = outcome.overall_quality
    main_gate_passed = outcome.main_gate_passed
    outcome.episode_trace = EpisodeTrace(
        episode_id=episode.qa_id,
        qa_id=episode.qa_id,
        final_state=outcome.final_state,
        program_trace_ref=(f"program_trace:{outcome.program.program_id}"
                           if outcome.program is not None else ""),
        geometry_check_ref=(f"geometry_check:{outcome.program.program_id}"
                            if outcome.verify is not None and outcome.program is not None
                            else ""),
        evaluation_ref=f"evaluation_result:{episode.qa_id}",
        failure=_failure_of(outcome, episode),
        active_snapshot_ref=cfg.active_snapshot_ref,
        active_snapshot_manifest_sha256=str(
            cfg.active_snapshot_manifest_sha256 or ""),
        # M13/D-3 归因：即使恢复成功（final_state=answer）也留痕（不进 failure）
        tool_contract_hits=int(outcome.tool_contract_hits),
        abstained=bool(outcome.abstained),
        answer_untrusted=bool(outcome.answer_untrusted),
        scene_route=str(outcome.scene_route or ""),
        question_tool_scope=str(outcome.question_tool_scope or ""),
        quality_status=str(outcome.quality_status or ""),
        # v6：逐 episode 事实落 trace（§19.1/§19.2）
        schema_version=EPISODE_TRACE_SCHEMA_VERSION,
        quality_metric_version=QUALITY_METRIC_VERSION,
        frame_set_hash=str(outcome.frame_set_hash or ""),
        n_frames=int(len(episode.frames)),
        n_images_to_synthesizer=int(outcome.n_images_to_synthesizer),
        # v6 §10.4：G5 永久 not_available；不得出现任何代理值
        reprojection_status="not_available",
        world_frame_status=str(outcome.world_frame_status),
        scale_fusion_status=str(outcome.scale_fusion_status),
        metric_scale=outcome.metric_scale,
        metric_gate_passed=bool(gate is not None and gate.gate_passed),
        metric_model=str(outcome.metric_model or "none"),
        authorized_metric_tasks=sorted(outcome.authorized_metric_tasks),
        evidence_states=evidence_states,
        overall_quality=(None if outcome.overall_quality is None
                         else float(outcome.overall_quality)),
        main_gate_passed=main_gate_passed,
        input_degradation_flags=sorted(set(outcome.input_degradation_flags or [])),
        answer_source=str(outcome.answer_source),
        input_error_reason=str(outcome.input_error_reason or ""),
        recovery_count=int(outcome.recovery_count),
        partial_tool_recovery=bool(outcome.partial_tool_recovery),
        used_result_ids=sorted(set(outcome.used_result_ids or [])),
        invalidated_result_ids=sorted(set(outcome.invalidated_result_ids or [])),
        failure_code=outcome.failure_code,
        terminal_failure=dict(outcome.terminal_failure),
        eval_visual_fallback=dict(outcome.eval_visual_fallback),
        detector_attempts=list(outcome.detector_attempts),
        synthesis_source=str(outcome.synthesis_source or ""),
        round_trigger=str(outcome.round_trigger or "initial"),
        finalization_used=bool(outcome.finalization_used),
        episode_status=_episode_status_of(outcome.final_state),
        answer_basis=str(outcome.answer_basis or ""),
        answer=(outcome.answer_payload.model_dump()
                if outcome.answer_payload is not None else {}),
        derivation_replay=_final_derivation_replay(outcome),
        attribution=(outcome.attribution.model_dump()
                     if outcome.attribution is not None else {}),
        authorization_receipts=[dict(r) for r in (outcome.authorization_receipts or [])],
        agent_rounds=int(outcome.agent_rounds),
        yield_count=int(outcome.yield_count),
        round_trace_refs=list(outcome.round_trace_refs or []),
        rounds=[dict(r) for r in (outcome.rounds or [])],
        budget=_budget_receipt(cfg),
        degenerate_regenerated=bool(outcome.degenerate_regenerated),
        state_sequence=list(outcome.states or []),
        retrieved_skills=[dict(x) for x in (outcome.retrieved_skills or [])],
        selected_skill_semvers=list(outcome.selected_skill_semvers or []),
        skill_mapping_misses=list(outcome.skill_mapping_misses or []),
        first_synthesis=dict(outcome.first_synthesis or {}),
        # v9 §9.4：主动图像三态（produced/delivered/observed）+ 每轮图像清单
        image_ledger=dict(outcome.image_ledger or {}),
        # v9 §13.5/§13.6：每次检索的完整记录 + 四态分列的版本清单
        retrieval_records=[r.model_dump() if hasattr(r, "model_dump") else dict(r)
                           for r in (outcome.retrieval_records or [])],
        retrieved_skill_versions=list(outcome.retrieved_skill_versions or []),
        delivered_skill_versions=list(outcome.delivered_skill_versions or []),
        declared_selected_skill_versions=list(
            outcome.declared_selected_skill_versions or []),
    )
    if trace_store is not None:
        trace_store.append("episode_input", EpisodeInputTrace(
            qa_id=episode.qa_id,
            scene_id=episode.scene_name,
            dataset=episode.dataset,
            question_type=episode.question_type,
            question_text=episode.question,
            options=list(episode.options or []),
            reference_answer=episode.ground_truth,
            source_split=episode.split,
            label_access=False,
            frame_set_hash=str(outcome.frame_set_hash or ""),
        ))
        trace_store.append("episode_trace", outcome.episode_trace)
        # §5.9 TraceRecord：版本字段 + 证据/路由全状态（不依赖重跑即可归因）
        trace_store.append("trace_record", _trace_record_of(outcome, episode, cfg))
        if outcome.program_trace is not None:
            trace_store.append("program_trace", outcome.program_trace)
        if outcome.program is not None:
            # M13 增补：程序源码必须留痕，否则"为什么答错"无法归因（§4 M13）
            trace_store.append("episode_program", {
                "qa_id": episode.qa_id,
                "scene_name": episode.scene_name,
                "task": str(getattr(outcome, "task", "") or ""),
                "program_id": outcome.program.program_id,
                "program_source": outcome.program.program_source,
                "response_text": str(
                    next(
                        (
                            row.get("response_text")
                            for row in reversed(outcome.rounds)
                            if row.get("response_text") is not None
                        ),
                        "",
                    )
                ),
                "response_texts": [
                    str(row["response_text"])
                    for row in outcome.rounds
                    if row.get("response_text") is not None
                ],
                "answer_source": str(outcome.answer_source or ""),
                "skill_semver_used": list(getattr(outcome.program, "skill_semver_used", []) or []),
                # §13.6：`skill_semver_used` 是"进过模型请求的方法集"（上下文口径）；
                # 下面两条分别是"实际交付"与"模型自称"，三者不互替。
                "delivered_skill_versions": list(outcome.delivered_skill_versions or []),
                "declared_selected_skill_versions": list(
                    outcome.declared_selected_skill_versions or []),
                "retrieved_skills": [dict(x) for x in (outcome.retrieved_skills or [])],
                "first_synthesis": dict(outcome.first_synthesis or {}),
                "scene_summary": str(getattr(outcome, "scene_summary", "") or ""),
                "m8_prompt": str(getattr(outcome, "m8_prompt", "") or ""),
            })
        if outcome.verify is not None:
            trace_store.append("geometry_check", outcome.verify)
        trace_store.append("evaluation_result", EvaluationResultTrace(
            qa_id=episode.qa_id,
            question_type=episode.question_type,
            task=outcome.task or episode.question_type,
            is_mca=outcome.is_mca,
            predicted=outcome.predicted,
            ground_truth=episode.ground_truth,
            correct=outcome.correct,
            mra_value=outcome.mra_value,
            # §19.1「不依赖重跑即可归因失败」：`predicted=None` 有两种截然不同的原因——
            # (a) 模型 abstain（无答案），(b) 模型给了自由文本但抽不出选项字母。
            # 不落原始答案就分不开这两者（实测 2770 属 (b)，却只留下一只普通错题）。
            answer_text=outcome.answer,
            answer_source=str(outcome.answer_source or ""),
            episode_status=str(outcome.episode_status or ""),
            label_access=False,
            input_error_reason=str(outcome.input_error_reason or ""),
            derivation_replay=_final_derivation_replay(outcome),
            abstained=bool(outcome.abstained),
            failure_code=outcome.failure_code,
            eval_visual_fallback=dict(outcome.eval_visual_fallback),
        ))
    return outcome


# --------------------------------------------------------------- split 批量 ----


def _validate_artifact_identity(art, episode, pixels) -> None:
    """Reject a cached reconstruction from a different source or frame order."""
    from skill3d.adapters.frame_set import frame_set_hash

    fs = episode.frame_set
    if fs is None or fs.frame_set_hash != frame_set_hash(fs.frame_ids):
        raise ValueError(f"{episode.qa_id}: missing or invalid FrameSet identity")
    for key, actual, expected in (
        ("scene", art.scene_name, episode.scene_name),
        ("frame_set_hash", art.frame_set_hash, fs.frame_set_hash),
    ):
        if actual != expected:
            raise ValueError(f"{episode.qa_id}: artifact {key} mismatch")
    for key, expected in (
        ("dataset_id", episode.dataset), ("scene_name", episode.scene_name),
        ("episode_id", episode.qa_id),
    ):
        if getattr(fs, key) and getattr(fs, key) != expected:
            raise ValueError(f"{episode.qa_id}: FrameSet {key} mismatch")
    slots = [fs.frame_ids.index(fid) for fid in fs.readable_frame_ids]
    if slots != sorted(set(slots)) or len(pixels) != len(slots):
        raise ValueError(f"{episode.qa_id}: readable frame order/count mismatch")
    for key, expected in (
        ("frame_ids", fs.readable_frame_ids),
        ("source_frame_indices", [fs.source_frame_indices[i] for i in slots]),
        ("timestamps", [fs.timestamps[i] for i in slots]),
    ):
        if list(getattr(art, key)) != list(expected):
            raise ValueError(f"{episode.qa_id}: artifact {key} mismatch")

def _run_id(cfg: OnlineRunConfig, split: str, n: int) -> str:
    if cfg.deterministic_replay:
        key = f"{split}|{n}|{cfg.seed}|{cfg.mode}|{cfg.baseline}"
        return "eval-" + hashlib.sha256(key.encode()).hexdigest()[:12]
    import uuid

    return f"eval-{uuid.uuid4().hex[:12]}"


def run_split(items: Sequence, cfg: OnlineRunConfig, *, trace_store: Optional[TraceStore] = None,
              llm=None) -> tuple[list[EpisodeOutcome], EvaluationRun]:
    """跑一批 episode 并聚合为 `EvaluationRun`（§5.7）。items 为 EpisodeItem 序列。"""
    from datetime import datetime, timezone

    if len({it.episode.qa_id for it in items}) != len(items):
        raise ValueError("run_split requires unique qa_id within one run")
    if len({it.episode.split for it in items}) > 1:
        raise ValueError("run_split cannot mix source splits")
    run_id = _run_id(cfg, items[0].episode.split if items else "", len(items))
    store = trace_store
    if store is None and items:
        store = TraceStore(cfg.trace_dir)
    if store is not None and items:
        store.claim_run(run_id)
        store = store.bind(run_id=run_id)
    outcomes = [run_episode(
        it.episode,
        it.pixels,
        cfg,
        geometry=it.geometry,
        trace_store=store.bind(qa_id=it.episode.qa_id) if store is not None else None,
        llm=llm,
        input_error=getattr(it, "input_error", None),
    ) for it in items]

    mca = [o for o in outcomes if o.is_mca]
    na = [o for o in outcomes if not o.is_mca]
    def unavailable(outcome) -> bool:
        return outcome.final_state == "unavailable" or any(
            row.get("service_errors") for row in outcome.rounds)

    n_unavailable = sum(unavailable(o) for o in outcomes)
    # 按 §4 M7 规范题型（8 类）聚合：rel_direction 三档变体合为一类（§16"8 任务分别准确率"）
    per_task: dict[str, dict] = {}
    for o in outcomes:
        slot = per_task.setdefault(o.task or o.question_type,
                                   {"n": 0, "correct": 0, "mra_values": [],
                                    "n_mca": 0, "n_na": 0, "n_unavailable": 0, "levels": {}})
        slot["n"] += 1
        slot["n_unavailable"] += int(unavailable(o))
        slot["n_mca" if o.is_mca else "n_na"] += 1
        if o.correct:
            slot["correct"] += 1
        if o.mra_value is not None:
            slot["mra_values"].append(o.mra_value)
        # 档位明细（§8.3：object_rel_direction 的 easy/medium/hard 先等权聚合）
        lv = slot["levels"].setdefault(
            o.question_type or o.task,
            {"n": 0, "correct": 0, "mra_values": [], "n_mca": 0, "n_na": 0,
             "n_unavailable": 0})
        lv["n"] += 1
        lv["n_unavailable"] += int(unavailable(o))
        lv["n_mca" if o.is_mca else "n_na"] += 1
        if o.correct:
            lv["correct"] += 1
        if o.mra_value is not None:
            lv["mra_values"].append(o.mra_value)
    for slot in per_task.values():
        slot["accuracy"] = (slot["correct"] / slot["n"]) if slot["n"] else None
        vals = slot.pop("mra_values")
        slot["mra"] = (sum(vals) / len(vals)) if vals else None
        for lv in slot["levels"].values():
            lv["accuracy"] = (lv["correct"] / lv["n"]) if lv["n"] else None
            lvals = lv.pop("mra_values")
            lv["mra"] = (sum(lvals) / len(lvals)) if lvals else None
        for row in [slot, *slot["levels"].values()]:
            row["status"] = "incomplete" if row["n_unavailable"] else "completed"
            if row["n_unavailable"]:
                row["accuracy"] = row["mra"] = None

    na_scored = [o.mra_value for o in na if o.mra_value is not None]
    run = EvaluationRun(
        run_id=run_id,
        status="incomplete" if n_unavailable or not outcomes else "completed",
        n_unavailable=n_unavailable,
        split=items[0].episode.split if items else "",
        n_episodes=len(outcomes),
        accuracy=(sum(1 for o in mca if o.correct) / len(mca)) if mca and not n_unavailable else None,
        mra=(sum(na_scored) / len(na_scored)) if na_scored and not n_unavailable else None,
        per_task=per_task,
        active_snapshot_ref=cfg.active_snapshot_ref,
        code_commit=_git_head(),
        timestamp=("deterministic" if cfg.deterministic_replay
                   else datetime.now(timezone.utc).isoformat()),
    )
    if store is not None and items:
        store.append("evaluation_run", run)
        store.append("online_run", {
            "run_id": run.run_id, "mode": cfg.mode, "baseline": cfg.baseline,
            "status": run.status,
            "template_version": PROMPT_TEMPLATE_VERSION,
            "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
            "tool_docs_version": TOOL_DOCS_VERSION,
            "seed": cfg.seed, "source": getattr(items[0], "source", "unknown"),
            "n_episodes": len(outcomes), "split": run.split,
            "label_access": False,
            "n_unavailable": n_unavailable,
            "n_unanswerable": sum(1 for o in outcomes if o.final_state == "unanswerable"),
            "n_input_error": sum(1 for o in outcomes if o.final_state == "input_error"),
            "deterministic_replay": cfg.deterministic_replay,
            "note": ("mock_light/合成数据：仅管道验证，不构成任何精度结论（§9.2）"
                     if cfg.mode != "real" else "real 模式"),
        })
    return outcomes, run


def _git_head() -> str:
    """M21：code_commit（无 git 仓库时 "unknown"，不阻断）。"""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
