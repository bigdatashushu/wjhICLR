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

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.evaluation.accuracy import extract_option_letter, mca_correct
from skill3d.evaluation.mra import mra_single, parse_numeric_answer
from skill3d.fsm.online_fsm import OnlineFSM, OnlineState
from skill3d.memory.online_memory import (
    EpisodicMemory,
    MemoryWriteForbiddenError,
)
from skill3d.gates.input_gate import annotate_frames, input_gate
from skill3d.reconstruction.vggt_runner import ReconstructionFailed, reconstruct
from skill3d.reconstruction_gate.quality_metrics import (
    QUALITY_METRIC_VERSION,
)
from skill3d.reconstruction_gate.evidence_profile import M5EvidenceSummary
from skill3d.reconstruction_gate.scene_state import (
    build_scene_state,
    quality_gate,
    scope_scene_to_question,
)
from skill3d.routing.skill_retriever import retrieve
from skill3d.routing.task_classifier import TaskClassification, classify
from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.kernel import CellResult, RestrictedNamespaceKernel
from skill3d.sandbox.receipt import ReceiptChain, SandboxReceipt, verify_chain
from skill3d.schemas import (
    EpisodeProgram,
    EpisodeTrace,
    EvaluationRun,
    FailureTaxonomy,
    InputGateVerdict,
    InputFrame,
    ProgramExecutionTrace,
    RetrievedSkill,
    SceneState,
    SkillSpec,
    VSIBenchEpisode,
)
from skill3d.synthesis.program_assembler import (
    SynthesisError,
    assemble_program,
    assemble_program_ex,
    degenerate_reason,
)
from skill3d.synthesis.prompt_builder import PromptBuilder
from skill3d.tools import REGISTRY
from skill3d.tools.mock_switch import MockSwitch
from skill3d.tools.scene_handle import SceneHandle
from skill3d.online.recovery import (
    MAX_RECOVERY_ATTEMPTS,
    RecoveryPlan,
    build_feedback,
    cascade_invalidate,
    collect_validated,
    downgrade_profile,
    premise_of_failure,
    recovery_exhausted,
)
from skill3d.schemas.trace import TraceRecord
from skill3d.synthesis.prompt_builder import TEMPLATE_VERSION
from skill3d.trace.store import TraceStore
from skill3d.verifier.geometry_oracle import GeometryVerifyResult, geometry_verify

from . import synthetic as synth

# 确定性重放模式下的固定时间基准（§4 M8/M17 字节级一致；真实实验不启用）
_REPLAY_EPOCH = 0.0
_Path_ = Path


@dataclass
class OnlineRunConfig:
    """在线链运行配置（阈值/路径一律来自 configs/*.yaml，见 online/config.py）。"""

    mode: str = "real"                      # real | mock_light
    baseline: str = "C1_tools_program"      # C0_direct_vlm | C1_tools_program（§16.1）
    seed: int = 0
    deterministic_replay: bool = False      # 同 seed 字节级一致（时间/id 取确定性占位）
    active_snapshot_ref: str = "genesis"
    skills: list[SkillSpec] = field(default_factory=list)
    scene_quality: Optional[float] = None
    max_regen: int = 3                      # TODO_CALIBRATE（configs/config.yaml）
    cell_timeout_s: int = 30                # TODO_CALIBRATE
    use_docker: bool = False                # MVP 用 in-process kernel（§4 M10 字段 12）
    trace_dir: str = "data/traces"
    memory_dir: str = "data/memory_episodic"   # G-26：在线 episodic 记忆（JSONL）
    recon_dir: str = "data/reconstructions"
    work_dir: str = "data/reconstructions/mock_light"
    vllm_endpoints: list[str] = field(default_factory=list)
    vllm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    recon_method: str = "vggt"              # v6 只允许 vggt（§5.2）
    # v6 D1/D2：零样本度量深度模型（首个 PoC = MoGe-2）。
    # 默认 None → 不跑融合，artifact 记 scale_fusion_status="not_run"、metric_scale=None。
    # 传入实现了 MetricDepthModel 协议的对象即启用（MoGe-2 权重到位后才可能真跑）。
    metric_depth_model: object = None
    metric_model_name: str = "none"          # moge2 | metric3d_v2 | none
    # 按题型选择答案来源（任务级策略，非逐题 oracle）：命中的题型改用**直答 VLM**
    # （同一 32 帧 + 问题）。用途：当某题型的程序路径弱于直答时，用直答拿分；
    # 策略必须在 inner_validation 上定、在 outer_holdout 上验证，禁止用 GT 逐题挑选。
    direct_answer_tasks: set[str] = field(default_factory=set)
    reuse_artifact: Optional[str] = None    # 复用既有 artifact JSON（硬约束 18：A/B 同源）
    max_tokens: int = 4096
    max_images: int = 32                    # M8 送进模型的最大帧数（与 §4 M1 对齐）
    max_pixels: int = 131072                # 实测起点：32 帧 ≈ 9.7k prompt tokens（A-4）
    max_model_len: int = 32768
    # ---- v6 度量证据门（D3）----
    # 融合成功才可能暴露米制 Tool；阈值在 reconstruction_gate.evidence_profile 内
    # （全 [TODO_CALIBRATE]），此处不再重复一份。
    # v6 D7：partial_tool_recovery 的最大恢复次数（TODO_CALIBRATE）
    max_recovery: int = MAX_RECOVERY_ATTEMPTS
    allow_final_test: bool = False          # 硬约束 9：默认拒绝 final_test 进在线链


@dataclass
class EpisodeOutcome:
    """单 episode 的完整结果（供评测聚合与审计）。"""

    qa_id: str
    scene_name: str
    question_type: str
    final_state: str                        # answer | unanswerable | unavailable | answer_best_effort
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
    direct_answer: Optional[str] = None     # C0 基线：答案不经沙箱，生成阶段即产出
    # 答案来源（v6 §5.3 四值）：tool_program / direct_vlm_routed / abstain / tool_contract
    answer_source: str = ""
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
    used_result_ids: list[str] = field(default_factory=list)
    invalidated_result_ids: list[str] = field(default_factory=list)
    failure_code: Optional[str] = None
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


@dataclass
class _SynthResult:
    """M8 产出。direct_answer 仅 C0（无 program）时非空。"""

    program: Optional[EpisodeProgram]
    # §19.3 六类 + `mock_stub`：vllm_ok | vllm_parse_error | vllm_service_error |
    # m8_parse_recovered | direct_answer_fallback | partial_tool_recovery | mock_stub
    source: str
    note: str
    direct_answer: Optional[str] = None
    prompt: str = ""        # M8 文本 prompt（错误归因用；图像不入日志）
    # HC26：实际送进模型的多模态图像数（必须等于统一 FrameSet 帧数；0 表示纯文本，
    # 在 real 模式下不允许）。trace 落该值，便于事后核对"没有静默丢帧"。
    n_images: int = 0


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
        template_version=TEMPLATE_VERSION,
        tool_face_version=TOOL_FACE_VERSION,
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
        used_result_ids=sorted(set(outcome.used_result_ids or [])),
        recovery_count=int(outcome.recovery_count),
        partial_tool_recovery=bool(outcome.partial_tool_recovery),
        invalidated_result_ids=sorted(set(outcome.invalidated_result_ids or [])),
        synthesis_source=_synthesis_source_enum(outcome.synthesis_source),
        n_objects=int((outcome.m5_summary.n_objects
                       if outcome.m5_summary is not None else 0)),
        cache_hit=bool(outcome.cache_hit),
        failure_code=outcome.failure_code,
        m5_notes=_joined(outcome.m5_notes),
        m7_notes=_joined(outcome.m7_notes),
        m8_notes=_joined(outcome.m8_notes),
    )


# §6.3/§14：走 partial_tool_recovery 的 ToolContractError 族（`ToolContractError` 的四个子类）
_RECOVERABLE_CONTRACT_ERRORS = ("tool_contract", "confidence_gate", "domain_value",
                                "answer_already_given")

# 契约失败码 → §5.9 `FailureCode` 词汇表取值（trace 归因用）
_FAILURE_CODE_BY_CONTRACT: dict[str, str] = {
    "confidence_gate": "ConfidenceGateError",
    "domain_value": "DomainValueError",
    "answer_already_given": "AnswerAlreadyGiven",
    "tool_contract": "tool_contract",
}

_SYNTH_SOURCES = {"vllm_ok", "vllm_parse_error", "vllm_service_error",
                  "m8_parse_recovered", "direct_answer_fallback",
                  "partial_tool_recovery", "mock_stub"}


def _synthesis_source_enum(value: str) -> str:
    """把历史/宽松取值归一到 §19.3 的 6 类（+ v6 显式的 `mock_stub`）。

    - `mock_stub`（mock_light 确定性 stub，非模型输出）原样保留 —— v5 曾把它写成
      `deterministic_stub`，这里做别名归一；
    - 未知/空值（M8 根本没跑）→ `vllm_service_error`：**只允许**断言"没有拿到模型输出"，
      绝不冒充 `vllm_ok`（v5 的 vllm 混写正是 §19.3 要消灭的问题）。
    """
    v = str(value or "").strip()
    if v in _SYNTH_SOURCES:
        return v
    if v == "deterministic_stub":        # v5 历史名 → v6 的 mock_stub
        return "mock_stub"
    if v in ("", "none", "unavailable"):
        return "vllm_service_error"
    return "vllm_ok"


def _joined(notes: Optional[list[str]]) -> Optional[str]:
    """把逐阶段判读信息压成一行（§5.9 P8：不依赖重跑即可归因失败）。"""
    if not notes:
        return None
    return " | ".join(str(x) for x in notes if str(x).strip()) or None

# --------------------------------------------------------------- episode 执行 ----

def run_episode(
    episode: VSIBenchEpisode,
    pixels: Sequence[np.ndarray],
    cfg: OnlineRunConfig,
    *,
    geometry: Optional[synth.SyntheticGeometry] = None,
    trace_store: Optional[TraceStore] = None,
    llm=None,
    episodic: Optional[EpisodicMemory] = None,
) -> EpisodeOutcome:
    """跑完一条在线链 episode。

    llm：可注入的 `synthesis.vllm_client.VLLMClient` 兼容对象（测试用 fake）；
    为 None 且 `mode="real"` 时按 `cfg.vllm_endpoints` 构造；`mock_light` 下不需要。
    """
    pixels = list(pixels)
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
                     episodic=episodic)

    cls: Optional[TaskClassification] = None
    scene: Optional[SceneState] = None
    handle: Optional[SceneHandle] = None
    verdict: Optional[InputGateVerdict] = None
    retrieved: list[RetrievedSkill] = []
    selected_skills: list[SkillSpec] = []
    program: Optional[EpisodeProgram] = None
    program_trace: Optional[ProgramExecutionTrace] = None
    kernel: Optional[RestrictedNamespaceKernel] = None
    direct_answer: Optional[str] = None
    art = None

    # ---------------- M2 INPUT_GATE（被动观测，硬约束 21）----------------
    if fsm.state is OnlineState.INPUT_GATE:
        frame_set = episode.frame_set
        gate_input = pixels if pixels else list(episode.frames)
        verdict = input_gate(gate_input, frame_set=frame_set)
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
            "frames": annotate_frames(episode.frames, verdict, pixels_for_stats)})

        fsm.step("gate_done", {"action": verdict.action})
        states.append(fsm.state.value)
        if fsm.state is OnlineState.ANSWER:
            notes.append("M2 输入合法性 hard fail → unanswerable（§4 M2 字段 9；"
                         f"hard_fail_frames={verdict.hard_fail_frame_ids[:5]}）")
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                     episodic=episodic,
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
            if episode.frame_set is not None and art.frame_set_hash \
                    and art.frame_set_hash != episode.frame_set.frame_set_hash:
                notes.append(
                    f"[warn] 复用 artifact 的 frame_set_hash={art.frame_set_hash[:12]} 与 "
                    f"episode={episode.frame_set.frame_set_hash[:12]} 不一致"
                    "（硬约束 21：禁止双帧集）")
            fsm.step("done")
        elif cfg.mode == "real":
            # 方案 X（§2.2 / §4 M4）：P1 已把 artifact（含实算 quality）落盘时，
            # P2 **读取复用**、零重算。这也是硬约束 18 的自然形态：同一 scene 的
            # 多个 episode（含 paired A/B 两臂）看到的是**同一个** artifact 对象内容。
            existing = _artifact_json_path(cfg, episode.scene_name)
            try:
                if _Path_(existing).is_file():
                    from skill3d.legacy.readers import load_artifact_v5

                    cand = load_artifact_v5(existing)   # v5 门：版本/legacy fail-closed
                    want = (episode.frame_set.frame_set_hash
                            if episode.frame_set is not None else "")
                    if want and cand.frame_set_hash and cand.frame_set_hash != want:
                        # 帧集身份不符 = 双帧集风险（硬约束 21）→ 不复用，重算并留痕
                        notes.append(
                            f"[warn] P1 artifact 的 frame_set_hash="
                            f"{cand.frame_set_hash[:12]} 与本 episode="
                            f"{want[:12]} 不符 → 不复用，改为重算（硬约束 21）")
                        raise _ReuseMiss()
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
                     episodic=episodic,
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
            m5_objects, m5_stats, m5_notes, m5_materialized = _bind_objects_best_effort(
                art, pixels, None, episode, cfg, llm)
            notes.extend(m5_notes)
            outcome.m5_notes = list(m5_notes)
            scene, handle, art = _scene_from_artifact(
                art, pixels, m5_stats, m5_objects, episode, artifact_path=art_path,
                objects_materialized=m5_materialized,
                # paired A/B 复用 frozen artifact（硬约束 18）时不得改写两臂共用文件
                persist_quality=not bool(cfg.reuse_artifact))
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
                     episodic=episodic,
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
            outcome.question_tool_scope = scene.question_tool_scope
            outcome.scene_summary = str(getattr(scene, "summary", "") or "")
            outcome.m7_notes = [decision.note()] + list(decision.reasons)
            outcome.authorized_metric_tasks = (
                [cls.task] if scene.metric_task_authorized(cls.task) else [])
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
                             episodic=episodic, verdict=verdict, scene=scene, handle=handle)

    if fsm.state is OnlineState.RETRIEVE_SKILL:
        quality = cfg.scene_quality if cfg.scene_quality is not None \
            else (handle.quality_overall if handle is not None else None)
        retrieved = retrieve(episode.question, scene, cfg.skills,  # type: ignore[arg-type]
                             question_type=episode.question_type, scene_quality=quality)
        by_key = {f"{s.skill_id}@{s.semver}": s for s in cfg.skills}
        selected_skills = [by_key[r.skill_semver] for r in retrieved
                           if r.skill_semver in by_key]
        notes.append(f"M7 Skill 检索命中 {len(retrieved)} 条" +
                     ("" if retrieved else "（无命中 → 空 Skill baseline）"))
        fsm.step("done")
        states.append(fsm.state.value)

    # ---------------- M8 SYNTHESIZE_PROGRAM ----------------
    static_rounds = 0
    if fsm.state is OnlineState.SYNTHESIZE_PROGRAM:
        res = _synthesize(episode, scene, handle, selected_skills, cfg, llm,
                          feedback=None, geometry=geometry, pixels=pixels)
        outcome.synthesis_source = res.source
        outcome.n_images_to_synthesizer = int(res.n_images)
        outcome.m8_prompt = str(res.prompt or "")
        direct_answer = res.direct_answer
        program = res.program
        if "退化" in str(res.note or ""):
            outcome.degenerate_regenerated = True
        if program is None:
            # 题型策略命中时，程序路径失败**不**等于不可答：改用直答（同一 32 帧）。
            # 否则像 object_counting/appearance_order 这类"本应直答"的题会因为 M8
            # 解析失败白丢分（实测 32 题里白丢 3 题）。
            try:
                _cls_fb = classify(episode)
            except Exception:  # noqa: BLE001 - 未知题型不触发回退
                _cls_fb = None
            if (_cls_fb is not None
                    and _cls_fb.task in cfg.direct_answer_tasks):
                d_ans, d_note = _direct_vlm_answer(episode, cfg, llm, pixels=pixels)
                if d_ans is not None:
                    program = EpisodeProgram(
                        program_id=f"droute-{episode.qa_id}",
                        program_source="",           # 无程序：直答回退（同 C0 语义）
                        skill_semver_used=[],
                        intended_answer_slot="direct_answer")
                    direct_answer = d_ans
                    outcome.synthesis_source = "direct_answer_fallback"
                    notes.append(f"M8 程序生成失败 → 题型策略命中，改用直答回退"
                                 f"（task={_cls_fb.task}）")
                else:
                    notes.append(f"M8 程序生成失败且直答回退失败: {d_note[:120]}")
        if program is None:
            outcome.final_state = "unavailable"
            outcome.states = list(states)
            notes.append(res.note)
            notes.append("M8 生成失败 → episode 记 unavailable（§4 M8 字段 9）；"
                         "起服务：bash scripts/serve_qwen3vl_dp8.sh")
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                     episodic=episodic,
                             verdict=verdict, scene=scene, handle=handle)
        notes.append(f"M8 program 来源={res.source} program_id={program.program_id}")
        if res.note:
            notes.append(res.note)
        outcome.m8_notes = [f"source={res.source}"] + ([res.note] if res.note else [])
        outcome.program = program
        outcome.n_images_to_synthesizer = int(res.n_images)
        fsm.step("done")
        states.append(fsm.state.value)

    # ---------------- M9 STATIC_CHECK（有限次重生成）----------------
    while fsm.state is OnlineState.STATIC_CHECK and static_rounds <= cfg.max_regen:
        static_rounds += 1
        assert program is not None
        check = ast_guard(program.program_source, allowed_tools=set(REGISTRY.names()))
        if check.ok:
            notes.append(f"M9 AST 通过，Tool 调用={check.allowed_tool_calls}")
            fsm.step("pass")
            states.append(fsm.state.value)
            break
        notes.append(f"M9 AST 拒绝（第 {static_rounds} 次）: {check.violations}")
        fsm.step("fail")
        states.append(fsm.state.value)
        if fsm.state is not OnlineState.SYNTHESIZE_PROGRAM:
            break  # 重生成次数耗尽 → FSM 已转 unanswerable
        res = _synthesize(episode, scene, handle, selected_skills, cfg, llm,
                          feedback=check.violations, geometry=geometry, pixels=pixels)
        outcome.synthesis_source = res.source
        program = res.program
        outcome.program = program
        outcome.n_images_to_synthesizer = int(res.n_images)
        if program is None:
            notes.append(f"M9 重生成失败: {res.note}")
            break
        fsm.step("done")
        states.append(fsm.state.value)

    # ---------------- M10 SANDBOX_EXECUTE（含 D-3 tool_contract 恢复阶梯）----------------
    if fsm.state is OnlineState.SANDBOX_EXECUTE and program is not None:
        if program.program_source == "":
            # §16.1 C0：无 program 可执行（不调沙箱），空 trace 继续走链
            program_trace = _empty_program_trace(program)
            notes.append("M10 C0 基线无 program：跳过沙箱执行（空 ProgramExecutionTrace）")
            fsm.step("ok")
            states.append(fsm.state.value)
        while fsm.state is OnlineState.SANDBOX_EXECUTE:
            wallclock = 0.0
            if not cfg.deterministic_replay:
                t0 = time.perf_counter()
            kernel, cell, program_trace = _execute_program(
                episode, program, handle, pixels, cfg, receipts, kernel=kernel)
            if not cfg.deterministic_replay:
                wallclock = time.perf_counter() - t0
            # 硬约束 23 / §3 M10：程序**自捕获** `ToolContractError` 之后再 ReturnAnswer 时
            # `cell.error_code` 为空但答案不可信（answer_untrusted）——该答案一律不得采纳，
            # 并按 tool_contract 归因（否则"catch 住异常照旧作答"就能绕过 fail-closed）。
            error_code = cell.error_code
            if error_code is None and cell.answer_untrusted:
                error_code = "tool_contract"
                notes.append("M10 程序自捕获了 Tool 契约异常后仍产答案 → 按 tool_contract 归因，"
                             "答案不采纳（硬约束 23）")
            program_trace = program_trace.model_copy(
                update={"wallclock_s": wallclock,
                        "answer_untrusted": bool(cell.answer_untrusted),
                        "error_code": error_code})
            outcome.answer_untrusted = bool(cell.answer_untrusted)

            if error_code is None and cell.answer is not None:
                receipts.append("cell_run", {"program_id": program.program_id,
                                             "answer": cell.answer})
                notes.append(f"M10 执行成功 steps={program_trace.steps}")
                fsm.step("ok")
                states.append(fsm.state.value)
                break

            if error_code is None:
                receipts.append("cell_run", {"program_id": program.program_id})
            else:
                receipts.append("cell_run",
                                {"program_id": program.program_id,
                                 "error_code": error_code},
                                error_code=error_code)

            # ---- v6 D7：partial_tool_recovery（§6.4/§14）----
            # 保留未受污染的成功结果回灌；共享前提失效则级联撤销；恢复次数有限。
            # §6.3：**执行期 ToolContractError 全族**（tool_contract / confidence_gate /
            # domain_value / answer_already_given）都走这条恢复路径 —— 局部失败（域值/
            # 参数）只回灌 validated observations，共享前提失效才级联撤销（§14.1）。
            if error_code in _RECOVERABLE_CONTRACT_ERRORS:
                outcome.tool_contract_hits += 1
                outcome.answer_flags.append("tool_contract")
                detail = cell.error or "程序自捕获契约异常后仍产答案（answer_untrusted）"
                notes.append(
                    f"M10 ToolContractError[{error_code}]"
                    f"（第 {outcome.tool_contract_hits} 次）: {detail}")
                if kernel is not None:
                    # 答案依赖过契约失败的 Tool → 不得采纳
                    kernel.answer_slot.answer = None

                outcome.recovery_count += 1
                attempt = outcome.recovery_count
                premise = _premise_from_cell(cell, kernel)
                plan = RecoveryPlan(
                    attempt=attempt,
                    failed_tool=_failed_tool_of(cell),
                    failure_kind=str(error_code),
                    premise=premise,
                    validated=collect_validated(
                        kernel, evidence_version=str(
                            getattr(outcome.evidence_profile, "profile_version", ""))),
                )
                # 级联撤销（仅当失败揭示共享前提失效；局部失败不撤销，§14.1）
                if premise and kernel is not None:
                    ids, tools = cascade_invalidate(kernel, premise, registry=REGISTRY)
                    plan.invalidated_result_ids = ids
                    plan.invalidated_tools = tools
                    outcome.invalidated_result_ids.extend(ids)
                    # 回写 EvidenceProfile（能力降级）→ 逐题 scope 随之收窄
                    new_profile, changed = downgrade_profile(
                        getattr(scene, "evidence_profile", None), premise)
                    plan.downgraded_capabilities = changed
                    if new_profile is not None and scene is not None:
                        scene = scene.model_copy(update={"evidence_profile": new_profile})
                        outcome.evidence_profile = new_profile
                        scene, _d = scope_scene_to_question(
                            scene, cls.task if cls is not None else "",
                            m5=outcome.m5_summary)
                        handle = _retarget_handle(handle, scene)
                        outcome.question_tool_scope = scene.question_tool_scope
                        notes.append(
                            f"M7.5（级联撤销后重新派生）scope="
                            f"{scene.question_tool_scope}"
                            f"（scene_route={scene.scene_route} 不变，§6.2/D4）")
                    # 撤销后的 validated 重新收集（被撤销的已不在其中）
                    plan.validated = collect_validated(
                        kernel, evidence_version=str(
                            getattr(outcome.evidence_profile, "profile_version", "")))
                    notes.append(
                        f"M10 partial_tool_recovery 级联撤销 premise={premise}："
                        f"撤销 {len(ids)} 条结果 {ids[:6]}，工具 {tools}，"
                        f"能力降级 {changed}")
                else:
                    notes.append(
                        f"M10 partial_tool_recovery 局部失败（无共享前提失效）→ 不撤销"
                        f"，保留 {len(plan.validated)} 条 validated observations")

                if recovery_exhausted(attempt, max_attempts=cfg.max_recovery):
                    plan.exhausted = True
                    outcome.failure_code = _FAILURE_CODE_BY_CONTRACT.get(
                        str(error_code), "tool_contract")
                    outcome.abstained = True
                    notes.append(
                        f"M10 partial_tool_recovery 次数用尽（{attempt}>"
                        f"{cfg.max_recovery}）→ 显式 abstain（主榜按错计）")
                    fsm.step("contract_fail")
                    states.append(fsm.state.value)
                    break

                outcome.partial_tool_recovery = True
                plan.feedback = build_feedback(
                    plan, question_type=str(getattr(cls, "task", "") or ""),
                    scope=str(getattr(scene, "question_tool_scope", "") or ""))
                res, new_program, round_notes = _regenerate_after_contract(
                    episode, scene, handle, selected_skills, cfg, llm, pixels,
                    geometry, prior_program=program, cell=cell,
                    scope_override=None, feedback_text=plan.feedback)
                notes.extend(round_notes)
                if res and new_program is not None:
                    program = new_program
                    outcome.program = program
                    outcome.synthesis_source = "partial_tool_recovery"
                    notes.append(
                        f"M10 partial_tool_recovery 生效（attempt={attempt}）")
                    fsm.step("contract_recover")
                    states.append(fsm.state.value)
                    continue
                outcome.abstained = True
                outcome.failure_code = _FAILURE_CODE_BY_CONTRACT.get(
                    str(error_code), "tool_contract")
                notes.append("M10 partial_tool_recovery 重生成失败 → 显式 abstain"
                             "（主榜按错计，不刷分）")
                fsm.step("contract_fail")
                states.append(fsm.state.value)
                break

            notes.append(f"M10 执行错误（第 {fsm.kernel_restart_count + 1} 次）: "
                         f"{cell.error} / {cell.error_code}")
            fsm.step("error")
            states.append(fsm.state.value)
            if fsm.state is OnlineState.ANSWER:
                # 两级兜底：no-tool CoT → 正则抽取（§6.1），保证 best-effort 答案。
                # 注意：tool_contract 触发的 abstain **不走**兜底（答案不可信，§5.1）。
                best = _best_effort_answer(cell.stdout_tail, llm, episode, cfg,
                                           pixels=pixels)
                if best is not None:
                    cell.answer = best
                    if kernel is not None:  # 回填 kernel 答案槽，供 M11/M12 读取
                        kernel.answer_slot.answer = best
                    outcome.answer_flags.append("no_tool_fallback")
                    if "unanswerable" in fsm.answer_flags:
                        outcome.answer_flags.append("unanswerable")
                        outcome.final_state = "answer_best_effort"
                    notes.append("M10 两级兜底产出 best-effort 答案（标 no_tool_fallback）")
                break
        outcome.program_trace = program_trace

    # ---------------- M11 GEOMETRY_VERIFY ----------------
    if fsm.state is OnlineState.GEOMETRY_VERIFY and program_trace is not None:
        answer = kernel.answer_slot.answer if kernel is not None else None
        verify = geometry_verify(program_trace, handle, answer)  # type: ignore[arg-type]
        outcome.verify = verify
        notes.append(f"M11 几何校验 passed={verify.passed} checks={verify.checks}")
        fsm.step("pass" if verify.passed else "reject")
        states.append(fsm.state.value)
        receipts.append("geometry_verify",
                        {"passed": verify.passed, "violations": verify.violations})

    # ---------------- M12 BENCHMARK_EVAL ----------------
    answer: Optional[str] = kernel.answer_slot.answer if kernel is not None else direct_answer
    answer_source = _answer_source_v6(answer=answer, program=program, kernel=kernel,
                                      outcome=outcome)
    if fsm.state is OnlineState.BENCHMARK_EVAL:
        if cls is None:
            cls = classify(episode)
        # 任务级策略：该题型配置为直答 → 用**同一 32 帧**重新直答并采纳（记录来源）
        if (answer is not direct_answer and cls.task in cfg.direct_answer_tasks):
            d_ans, d_note = _direct_vlm_answer(episode, cfg, llm, pixels=pixels)
            if d_ans is not None:
                answer, answer_source = d_ans, "direct_vlm_routed"
                notes.append(f"M12 题型策略={cls.task} → 改用直答（策略见 "
                             f"direct_answer_tasks）；program 答案被替换")
            else:
                notes.append(f"M12 题型策略={cls.task} → 直答失败，保留 program 答案: "
                             f"{d_note[:120]}")
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
        outcome.direct_answer = direct_answer
        outcome.answer_source = answer_source
        # §14.1：最终答案关联 result_ids（只记 status=ok 且未被级联撤销的结果）
        if kernel is not None and answer is not None:
            outcome.used_result_ids = sorted({
                o.result_id for o in collect_validated(
                    kernel, evidence_version=str(
                        getattr(outcome.evidence_profile, "profile_version", "")))
                if o.result_id})
        if answer is not None and "unanswerable" not in fsm.answer_flags:
            outcome.final_state = "answer"
        fsm.step("done")
        states.append(fsm.state.value)

    return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                     episodic=episodic,
                     verdict=verdict, scene=scene, handle=handle)


# ------------------------------------------------------------------ 各阶段实现 ----

def _frames_with_stats(episode: VSIBenchEpisode, pixels: Sequence[np.ndarray]) -> list[InputFrame]:
    """（历史入口）用真实 IQA 统计填 InputFrame。

    新代码请用 `gates.input_gate.annotate_frames`（M2 被动观测的单一事实源）；
    本函数保留为薄封装，语义等价且同样**不改帧集**（硬约束 21）。
    """
    from skill3d.gates.input_gate import annotate_frames

    if pixels and len(pixels) == len(episode.frames):
        verdict = input_gate(pixels, frame_set=episode.frame_set)
        return annotate_frames(episode.frames, verdict, pixels)
    return list(episode.frames)


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


def _artifact_json_path(cfg: OnlineRunConfig, scene_name: str) -> str:
    """P1 落盘的 artifact JSON 路径（方案 Y 原子写回的目标）。"""
    return str(Path(cfg.recon_dir) / cfg.recon_method / f"{scene_name}.json")


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
                         objects_materialized=objects_materialized)
    # 世界系点图注入句柄（平面拟合 / 连通性图的唯一数据入口）
    handle.set_point_map(point_map)
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
        if g.get("miss") or int(g.get("n_boxes", 0)) <= 0:
            hit, miss = False, True
        elif int(g.get("n_new", 0)) > 0 or g.get("all_present"):
            hit, miss = True, False
        else:
            hit, miss = True, False
    else:
        hit, miss = None, False      # 未做校验 → 上层按 degraded（不是 available）
    return M5EvidenceSummary(
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
                             metric_scale=handle.metric_scale)
    new_handle.set_point_map(handle.get_point_map())
    # 内存态对象点集（合成路径注入过的）随句柄一起搬，避免逐题重派生后丢点云
    for oid, pts in getattr(handle, "_points_cache", {}).items():  # noqa: SLF001
        new_handle.set_object_points(oid, pts)
    return new_handle


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
        return [], {}, [f"M5 对象绑定不可用（{type(exc).__name__}: {exc}）→ "
                        "相关 Tool 查询回退全场景（TODO_USER_INPUT: SAM2 checkpoint，§7.1 G-19）"], False
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
                     expected_frames: Optional[int] = None) -> list[dict]:
    """文本 prompt → OpenAI messages；`mode=real` 必须带图（多模态，硬约束 26）。

    `mode=real` 下缺帧或帧数与统一 FrameSet 不一致时**直接报错**：既不允许静默
    退化纯文本，也不允许少送帧（硬约束 21/26；`build_image_messages` 另有一道校验）。
    """
    if cfg.mode == "real":
        n = len(pixels) if pixels is not None else 0
        if n == 0:
            raise ValueError(
                "M8 real 模式收到 0 帧：硬约束 26 要求多模态（32 帧 + 文本），"
                "禁止静默退化纯文本生成程序")
        if expected_frames is not None and n != expected_frames:
            raise ValueError(
                f"M8 收到 {n} 帧但统一 FrameSet 为 {expected_frames} 帧"
                "（硬约束 21：禁止双帧集/丢帧）")
        from skill3d.synthesis.prompt_builder import build_image_messages

        return build_image_messages(prompt, pixels, max_images=cfg.max_images)
    return [{"role": "user", "content": prompt}]


def _synthesize(episode: VSIBenchEpisode, scene, handle, skills, cfg: OnlineRunConfig,
                llm, feedback: Optional[list[str]] = None,
                geometry: Optional[synth.SyntheticGeometry] = None,
                pixels: Optional[Sequence[np.ndarray]] = None,
                scope_override: Optional[str] = None,
                traceback_feedback: Optional[str] = None,
                prior_program: Optional[str] = None) -> _SynthResult:
    """M8：生成 program（或 C0 直答）。

    `scope_override` / `traceback_feedback` / `prior_program` 供 D-3 恢复层使用：
    裁剪 prompt（强制受限 route）或把"上一轮 program + 裁剪 traceback"作为
    额外 turns 回灌（§4 M6 字段 9）。
    """
    if cfg.baseline == "C0_direct_vlm":
        # §16.1 C0：直接自然语言作答，不编排 Tool → 无 program，不执行沙箱
        # （**必须**带统一 FrameSet 的图像：C0 是"直答 VLM"基线，不是纯文本基线）
        answer, note = _direct_vlm_answer(episode, cfg, llm, pixels=pixels)
        if answer is None:
            return _SynthResult(None, "none", note)
        program = EpisodeProgram(
            program_id=f"c0-{episode.qa_id}",
            program_source="",           # C0 无 program
            skill_semver_used=[],
            intended_answer_slot="direct_answer",
        )
        source = "vllm_ok" if cfg.mode == "real" else "mock_stub"
        return _SynthResult(program, source, note, direct_answer=answer)

    if cfg.mode == "mock_light":
        # 有合成几何用合成几何；否则（冻结真实 artifact / golden 重放）用 handle 对象名
        names = None
        if geometry is None:
            if handle is None:
                raise ValueError("mock_light 需要 geometry 或已建立的 SceneHandle")
            names = [_class_hint_of(handle, oid) for oid in handle.list_objects()]
        src = synth.stub_program(episode.question_type, episode, geometry,
                                 object_names=names)
        return _SynthResult(assemble_program(src, []), "mock_stub",
                            "M8 mock_light 确定性 stub program"
                            "（非 Qwen3-VL-8B 输出，仅管道验证）")

    prompt = _build_prompt(episode, scene, handle, skills, feedback,
                           scope_override=scope_override,
                           traceback_feedback=traceback_feedback)
    client = llm if llm is not None else _make_vllm_client(cfg)
    if client is None:
        return _SynthResult(None, "none",
                            "M8 vLLM 未配置：用 --vllm-endpoint 指定本地 Qwen3-VL-8B endpoint"
                            "（bash scripts/serve_qwen3vl_dp8.sh）；§4 M8 字段 9 → 记 unavailable")
    # 多模态：把 32 帧连同 prompt 一起送进模型（让模型"看到"重建依据的帧）
    expected = (len(episode.frame_set.frame_ids)
                if getattr(episode, "frame_set", None) is not None else None)
    messages = _prompt_messages(prompt, pixels, cfg, expected_frames=expected)
    n_images = len([1 for m in messages
                    if isinstance(m.get("content"), list)
                    for part in m["content"]
                    if isinstance(part, dict) and part.get("type") == "image_url"])
    if prior_program is not None:
        # 回灌：assistant(上一轮 program) → user(裁剪 traceback + 可用产物)
        messages = messages + [{"role": "assistant", "content": prior_program}]
        if traceback_feedback:
            messages.append({"role": "user", "content": traceback_feedback})
    try:
        # 请求级 seed：钉住 vLLM 采样随机性（§7 复现纪律）
        text = client.chat(messages, max_tokens=cfg.max_tokens, seed=int(cfg.seed))
    except Exception as exc:  # noqa: BLE001 - 网络/服务不可用
        from skill3d.synthesis.vllm_client import ServiceUnavailable

        if isinstance(exc, ServiceUnavailable):
            return _SynthResult(None, "vllm_service_error",
                                f"M8 service_unavailable（§6.1：服务故障不静默降级）: {exc}")
        return _SynthResult(None, "vllm_service_error",
                            f"M8 vLLM 调用失败: {type(exc).__name__}: {exc}")

    # §15.3 退化输出（20KB 重复段落/超长无意义输出/重复度超阈）→ **先重生成**
    reason = degenerate_reason(text)
    if reason is not None:
        notes_deg = f"M8 退化输出检测命中：{reason} → 触发一次重生成（§15.3）"
        text = _regenerate_non_degenerate(client, messages, cfg, reason)
        if text is None:
            return _SynthResult(None, "vllm_parse_error",
                                notes_deg + "；重生成仍退化/失败 → vllm_parse_error")

    try:
        program, recovered = assemble_program_ex(
            text, [f"{s.skill_id}@{s.semver}" for s in skills or []])
    except SynthesisError as exc:
        return _SynthResult(None, "vllm_parse_error", f"M8 program 解析失败: {exc}")
    # §15.2：解析回退单列 m8_parse_recovered，与"真解析不出来"区分开
    source = "m8_parse_recovered" if recovered else "vllm_ok"
    return _SynthResult(program, source, "", n_images=n_images, prompt=prompt)


def _regenerate_non_degenerate(client, messages, cfg: OnlineRunConfig,
                               reason: str) -> Optional[str]:
    """§15.3：退化输出触发**一次**重生成；仍退化或失败返回 None。

    重生成会显式告诉模型"上一轮输出退化了"，并要求只输出一个代码块；
    重生成结果必须重过 M9 AST（由调用方在拿到 program 后统一做）。
    """
    hint = (f"\n\n上一次输出被判定为退化（{reason}）。"
            "请只输出一个 ```python 代码块，代码块外不要写任何文字，"
            "不要反复讨论是否 abstain —— 判定一次就直接 ReturnAnswer。")
    try:
        msgs = list(messages)
        if msgs and isinstance(msgs[-1].get("content"), list):
            msgs[-1] = {**msgs[-1],
                        "content": list(msgs[-1]["content"]) + [
                            {"type": "text", "text": hint}]}
        else:
            msgs = msgs + [{"role": "user", "content": hint}]
        return client.chat(msgs, max_tokens=cfg.max_tokens, seed=int(cfg.seed))
    except Exception:  # noqa: BLE001 - 重生成失败即放弃（上层记 parse_error）
        return None


def _class_hint_of(handle, obj_id: str) -> str:
    """取 handle 中某对象的 class_hint（stub program 需要对象名做锚点）。"""
    try:
        return str(handle.get_object(obj_id).class_hint)
    except Exception:  # noqa: BLE001
        return obj_id


def _direct_vlm_answer(episode: VSIBenchEpisode, cfg: OnlineRunConfig, llm,
                       pixels: Optional[Sequence[np.ndarray]] = None
                       ) -> tuple[Optional[str], str]:
    """C0 直答：把**同一套 32 帧** + 问题一起给 VLM，要求直接作答（不编排 Tool）。

    历史缺陷（2026-09-20 实测修出）：real 模式此前只发纯文本 → 模型看不到视频，
    C0 在全部 32 题上 predicted=None（Avg 0.00），把"直答基线"变成了"瞎答基线"。
    C0 必须与 C1 共用同一 FrameSet（硬约束 21/26：同一帧集、多模态且不丢帧）。
    """
    if cfg.mode == "mock_light":
        return synth.stub_direct_answer(episode), (
            "M8/C0 mock_light 确定性 stub 直答（非模型输出，仅管道验证）")
    client = llm if llm is not None else _make_vllm_client(cfg)
    if client is None:
        return None, "M8/C0 vLLM 未配置 → 记 unavailable（TODO_USER_INPUT: endpoint）"
    q = episode.question
    if episode.options:
        q += ("\n选项: " + "; ".join(episode.options)
              + "\n直接给出最终答案：选择题只回答选项字母（如 C），不要解释。")
    else:
        q += "\n直接给出最终答案（一个数字），不要解释。"
    try:
        expected = (len(episode.frame_set.frame_ids)
                    if getattr(episode, "frame_set", None) is not None else None)
        messages = _prompt_messages(q, pixels, cfg, expected_frames=expected)
        return client.chat(messages, max_tokens=256), "vllm"
    except Exception as exc:  # noqa: BLE001
        return None, f"M8/C0 调用失败: {type(exc).__name__}: {exc}"


def _build_prompt(episode, scene, handle, skills, feedback, *,
                  scope_override: Optional[str] = None,
                  traceback_feedback: Optional[str] = None) -> str:
    """M8 prompt：只给 SceneState 摘要 + Tool 文档 + Skill 模板，绝不含 GT（§4 M8）。

    - `tool_docs = REGISTRY.docs(route=...)`：按 route 静态裁剪（D-3a），
      prompt 头部显式写"当前 route=…，可用产物=…"；
    - `scope_override`：tool_contract 恢复层的"裁剪重生成"（强制 fallback_2d_only）；
    - `traceback_feedback`：回灌层的裁剪 traceback + 可用产物清单。
    """
    scope = scope_override or (scene.question_tool_scope if scene is not None else "")
    # prompt 头部的"可用产物"必须是**实际装载**的集合（硬约束 23）：
    # `scene.available_artifacts` 是 scope 级声明，M5 失败/数组未加载时会高报，
    # 模型照着写就会撞 ArtifactUnavailableError。
    if handle is not None:
        available = sorted(handle.available_artifacts)
    else:
        available = sorted(getattr(scene, "available_artifacts", set()) or set())
    qtype = str(getattr(scene, "question_type", "") or "")
    profile = getattr(scene, "evidence_profile", None) if scene is not None else None
    gate = getattr(scene, "metric_evidence_gate_result", None) if scene is not None else None
    text = PromptBuilder().render(
        question=episode.question,
        scene_summary=(scene.summary if scene is not None else ""),
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
    )
    if feedback:
        text += f"\n上一次生成被 AST 拒绝，原因：{feedback}\n请修正后重新输出。"
    if traceback_feedback:
        text += f"\n{traceback_feedback}"
    return text


def _contract_feedback(cell: CellResult, program: EpisodeProgram, scene) -> str:
    """回灌文本（§4 M6 字段 9）：裁剪后的 traceback + 可用产物清单。

    只给"失败的工具调用 + 缺失产物 + 当前 route 可用产物"，不给整段宿主 traceback
    （避免把内部路径/实现细节喂进 prompt）。
    """
    violations = cell.contract_violations or []
    lines = ["上一次 program 执行被 Tool 契约拒绝（产物缺失，硬约束 23）："]
    for v in violations:
        lines.append(f"- Tool {v.get('tool')} 缺产物 {v.get('missing')}"
                     f"；route={v.get('route')}；可用产物={v.get('available')}")
    if not violations:
        lines.append(f"- {cell.error}")
    avail = sorted(getattr(scene, "available_artifacts", set()) or set())
    lines.append(f"当前 route={getattr(scene, 'route', '')}，可用产物={avail}。")
    lines.append("请只用产物的确齐备的 Tool 重写 program；若该量当前无法获得，"
                 "直接 ReturnAnswer(\"abstain\") 而不是猜测。")
    return "\n".join(lines)


def _answer_source_v6(*, answer: Optional[str], program, kernel,
                      outcome: EpisodeOutcome) -> str:
    """`answer_source` 的 v6 四值口径（§5.3/§6.3，D9）：

    `tool_program`（沙箱内程序作答）/ `direct_vlm_routed`（直答，含 C0 基线与题型策略
    回退）/ `abstain`（确实无从作答）/ `tool_contract`（契约失败终止：恢复层用尽）。

    v5 的 `program` / `direct_vlm` 两名已废止（`evaluation.experiment_protocol`
    只为历史 trace 回读保留别名映射）。
    """
    if answer is not None and str(answer) != "":
        from_program = (kernel is not None
                        and str(getattr(program, "program_source", "") or "") != "")
        # 无 program 却能作答 = 直答（C0 基线 / M8 直答回退 / 题型策略）→ 直答来源
        return "tool_program" if from_program else "direct_vlm_routed"
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


def _regenerate_after_contract(episode, scene, handle, skills, cfg, llm, pixels,
                               geometry, *, prior_program: EpisodeProgram,
                               cell: CellResult,
                               scope_override: Optional[str] = None,
                               feedback_text: Optional[str] = None
                               ) -> tuple[bool, Optional[EpisodeProgram], list[str]]:
    """partial_tool_recovery 的重生成（v6 §6.4/§14）。

    与 v5 的"回灌/裁剪两档"不同：v6 只有**一条**恢复路径 ——
    重置命名空间 → 注入 validated observations 摘要 + 失败信息 → 重生成 → 重执行；
    次数由 `cfg.max_recovery` 约束（`[TODO_CALIBRATE]`），超限切
    `direct_vlm_routed` 或 abstain。

    重生成必须重过 M9 AST 检查；重执行前必须 `reset_user_namespace()`（避免引用
    已失效的旧变量）。
    """
    notes: list[str] = []
    if cfg.mode == "mock_light" and llm is None:
        # mock_light 的 stub program 是确定性的：重生成只会得到同一份 → 直接 abstain
        notes.append("M10 恢复层跳过：mock_light 无模型，stub program 重生成无意义")
        return False, None, notes

    feedback = feedback_text or _contract_feedback(cell, prior_program, scene)
    # §14.1：重置用户命名空间（保留 Tool/帧/答案槽），再注入 validated obs 摘要
    if handle is not None:
        pass  # kernel 由 _execute_program 持有；重置在其内部按需执行
    res = _synthesize(episode, scene, handle, skills, cfg, llm,
                      feedback=None, geometry=geometry, pixels=pixels,
                      scope_override=scope_override,
                      traceback_feedback=feedback,
                      prior_program=None)
    if res.program is None:
        notes.append(f"M10 恢复层重生成失败: {res.note}")
        return False, None, notes
    check = ast_guard(res.program.program_source, allowed_tools=set(REGISTRY.names()))
    if not check.ok:
        notes.append(f"M10 恢复层重生成未过 M9 AST: {check.violations}")
        return False, None, notes + ["AST 拒绝"]
    notes.append("M10 恢复层重生成通过 M9 AST（回灌 validated observations + 失败信息）")
    return True, res.program, notes


def _make_vllm_client(cfg: OnlineRunConfig):
    if not cfg.vllm_endpoints:
        return None
    from skill3d.synthesis.vllm_client import VLLMClient

    return VLLMClient(cfg.vllm_endpoints, cfg.vllm_model)


def _empty_program_trace(program: EpisodeProgram) -> ProgramExecutionTrace:
    """C0：无 program 的占位 trace（无可执行内容，非"执行成功"的伪装）。"""
    return ProgramExecutionTrace(
        program_id=program.program_id, calls=[], results=[], stdout_tail="",
        error_code=None, steps=0, wallclock_s=0.0,
    )


def _execute_program(episode, program, handle, pixels, cfg: OnlineRunConfig, receipts,
                     kernel: Optional[RestrictedNamespaceKernel] = None):
    """M10：建/复用 kernel → 执行 program → 收集 ProgramExecutionTrace（§5.4）。

    `kernel` 非空时**先 reset user namespace 再重注入**（SpatialClaw §E.3 先例）：
    D-3 的回灌重执行必须从干净状态起跑，per-episode 状态不得跨次执行泄漏。
    """
    mock_switch = MockSwitch(light_handle=handle) if cfg.mode != "real" else None
    if kernel is not None:
        kernel.reset_user_namespace()
        kernel._mock_switch = mock_switch  # noqa: SLF001 - 重注入 mock 句柄
    else:
        kernel = RestrictedNamespaceKernel(
            REGISTRY,
            handle,
            frames=pixels,
            mode="real" if cfg.mode == "real" else "mock_light",
            cell_timeout_s=cfg.cell_timeout_s,
            mock_switch=mock_switch,
            call_id_factory=_Counter() if cfg.deterministic_replay else None,
        )
    receipts.append("sandbox_start", {"episode_id": episode.qa_id, "mode": cfg.mode})
    cell = kernel.run_cell(program.program_source)
    results = list(kernel.tool_results)
    if cfg.deterministic_replay:
        # 重放确定性：latency_ms 是实测值，置零以保证同 seed 字节级一致（§4 M17）
        results = [r.model_copy(update={"latency_ms": 0.0}) for r in results]
    program_trace = ProgramExecutionTrace(
        program_id=program.program_id,
        calls=list(kernel.tool_calls),
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


def _best_effort_answer(stdout_tail: str, llm, episode: VSIBenchEpisode,
                        cfg: OnlineRunConfig,
                        pixels: Optional[Sequence[np.ndarray]] = None) -> Optional[str]:
    """§6.1 两级兜底：no-tool CoT（需 vLLM）→ 正则抽取。

    **必须带统一 FrameSet 的图像**（实测缺陷，2026-09-20）：此前 real 模式只发文本，
    模型看不到视频 → 兜底答案是无信息的乱猜，而且它会**覆盖程序里已经算对的工具结果**
    （实测 chair 题：count_objects 返回 4（=GT），兜底把答案改成 0）。
    """
    text = ""
    client = llm if llm is not None else _make_vllm_client(cfg)
    if client is not None:
        try:
            q = episode.question + "\n只回答答案本身，不要解释。"
            if episode.options:
                q += "（只回答选项字母）"
            expected = (len(episode.frame_set.frame_ids)
                        if getattr(episode, "frame_set", None) is not None else None)
            messages = _prompt_messages(q, pixels, cfg, expected_frames=expected)
            text = client.chat(messages, max_tokens=128)
        except Exception:  # noqa: BLE001 - 兜底失败即降级到第二级
            text = ""
    elif cfg.mode == "mock_light":
        # mock_light 无模型：用确定性 stub 作为第一级占位（已显式标注）
        text = synth.stub_direct_answer(episode)
    if not text:
        text = stdout_tail
    if episode.options:
        return extract_option_letter(text)
    return text.strip() or None


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
              scene=None, handle=None,
              episodic: Optional[EpisodicMemory] = None) -> EpisodeOutcome:
    """收尾：failure 归因、receipts 链校验、EpisodeTrace 落盘（M13）+ episodic 记忆（G-26）。"""
    outcome.states = list(states)
    outcome.answer_flags = list(dict.fromkeys(outcome.answer_flags + list(fsm.answer_flags)))
    receipts.append("episode_end", {"final_state": outcome.final_state,
                                    "qa_id": episode.qa_id})
    outcome.receipts = receipts.receipts
    outcome.receipts_ok = verify_chain(outcome.receipts)

    # C0 基线：答案不经沙箱，由生成阶段直接产出
    if outcome.program is not None and outcome.program.program_source == "" \
            and outcome.answer is None and outcome.direct_answer is not None:
        outcome.answer = outcome.direct_answer
        if outcome.final_state == "unanswerable":
            outcome.final_state = "answer"

    if not outcome.is_mca and outcome.mra_value is None and outcome.correct is None:
        cls = classify(episode)
        outcome.is_mca = cls.is_mca
        if outcome.answer is not None:
            pred, correct, mra_value = _evaluate(episode, outcome.answer, cls)
            outcome.predicted, outcome.correct, outcome.mra_value = pred, correct, mra_value

    # v3 §4 M6 字段 9 / §5.1：abstain 与 unanswerable **主榜按错计**
    # （MRA=0 / exact_match=0，不刷分）；只有 unavailable（模型/服务不可用）不进分母。
    if outcome.final_state in ("unanswerable", "abstain"):
        cls_wrong = classify(episode)
        outcome.is_mca = cls_wrong.is_mca
        if cls_wrong.is_mca:
            outcome.correct = False
            outcome.predicted = None
        else:
            outcome.mra_value = 0.0
            outcome.correct = None
        outcome.abstained = True
        if "abstain" not in outcome.answer_flags:
            outcome.answer_flags.append("abstain")

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
        # M13/D-3 归因：即使恢复成功（final_state=answer）也留痕（不进 failure）
        tool_contract_hits=int(outcome.tool_contract_hits),
        abstained=bool(outcome.abstained),
        answer_untrusted=bool(outcome.answer_untrusted),
        scene_route=str(outcome.scene_route or ""),
        question_tool_scope=str(outcome.question_tool_scope or ""),
        quality_status=str(outcome.quality_status or ""),
        # v6：逐 episode 事实落 trace（§19.1/§19.2）
        schema_version="6.0",
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
        recovery_count=int(outcome.recovery_count),
        partial_tool_recovery=bool(outcome.partial_tool_recovery),
        used_result_ids=sorted(set(outcome.used_result_ids or [])),
        invalidated_result_ids=sorted(set(outcome.invalidated_result_ids or [])),
        failure_code=outcome.failure_code,
        synthesis_source=str(outcome.synthesis_source or ""),
        degenerate_regenerated=bool(outcome.degenerate_regenerated),
    )
    if trace_store is not None:
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
                "answer_source": str(outcome.answer_source or ""),
                "scene_summary": str(getattr(outcome, "scene_summary", "") or ""),
                "m8_prompt": str(getattr(outcome, "m8_prompt", "") or ""),
            })
        if outcome.verify is not None:
            trace_store.append("geometry_check", outcome.verify)
        trace_store.append("evaluation_result", {
            "qa_id": episode.qa_id,
            "question_type": episode.question_type,
            "task": outcome.task or episode.question_type,
            "is_mca": outcome.is_mca,
            "predicted": outcome.predicted,
            "ground_truth": episode.ground_truth,
            "correct": outcome.correct,
            "mra_value": outcome.mra_value,
            # §19.1「不依赖重跑即可归因失败」：`predicted=None` 有两种截然不同的原因——
            # (a) 模型 abstain（无答案），(b) 模型给了自由文本但抽不出选项字母。
            # 不落原始答案就分不开这两者（实测 2770 属 (b)，却只留下一只普通错题）。
            "answer_text": (outcome.answer if outcome.answer is not None
                            else outcome.direct_answer),
            "answer_source": str(outcome.answer_source or ""),
            "abstained": bool(outcome.abstained),
            "failure_code": outcome.failure_code,
        })
    if episodic is not None:
        _write_episodic_memory(episodic, outcome, episode, trace_store)
    return outcome


def _write_episodic_memory(episodic: EpisodicMemory, outcome: EpisodeOutcome,
                           episode: VSIBenchEpisode,
                           trace_store: Optional[TraceStore]) -> None:
    """G-26：在线写 episodic 记忆（semantic 绝不在此写，硬约束 1/2）。

    final_test 由 `record_episode` 静默跳过（硬约束 9）；写失败只记 note 不阻断在线链
    （记忆是增强项，不能影响单题作答）。
    """
    try:
        entry = episodic.record_episode(
            qa_id=episode.qa_id, scene_name=episode.scene_name,
            task=outcome.task or episode.question_type,
            final_state=outcome.final_state, correct=outcome.correct,
            split=episode.split, route=outcome.scene_route or "",
            flags=outcome.answer_flags,
        )
    except MemoryWriteForbiddenError as exc:
        outcome.notes.append(f"M14 episodic 记忆拒写（硬约束 9/19）: {exc}")
        return
    except Exception as exc:  # noqa: BLE001 - 磁盘/序列化异常不得阻断在线链
        outcome.notes.append(f"M14 episodic 记忆写入失败（{type(exc).__name__}: {exc}）")
        return
    if entry is not None:
        outcome.notes.append(f"M14 episodic 记忆已写 {entry.memory_id}（G-26）")
        if trace_store is not None:
            trace_store.append("memory_entry", entry.model_dump())


# --------------------------------------------------------------- split 批量 ----

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

    store = trace_store
    if store is None and items:
        store = TraceStore(cfg.trace_dir)
    # G-26：在线 episodic 记忆（在线写、离线读；semantic 不在线写）。
    # 构造失败（路径不可写等）→ 记降级、不阻断在线链；EpisodicMemory 内部已自降级。
    episodic = None
    if cfg.memory_dir:
        try:
            episodic = EpisodicMemory(path=_episodic_path(cfg))
        except Exception:  # noqa: BLE001 - 记忆是增强项，绝不阻断评测
            episodic = None
    outcomes = [run_episode(it.episode, it.pixels, cfg, geometry=it.geometry,
                            trace_store=store, llm=llm, episodic=episodic) for it in items]

    mca = [o for o in outcomes if o.is_mca]
    na = [o for o in outcomes if not o.is_mca]
    # 按 §4 M7 规范题型（8 类）聚合：rel_direction 三档变体合为一类（§16"8 任务分别准确率"）
    per_task: dict[str, dict] = {}
    for o in outcomes:
        slot = per_task.setdefault(o.task or o.question_type,
                                   {"n": 0, "correct": 0, "mra_values": [],
                                    "n_mca": 0, "n_na": 0, "levels": {}})
        slot["n"] += 1
        slot["n_mca" if o.is_mca else "n_na"] += 1
        if o.correct:
            slot["correct"] += 1
        if o.mra_value is not None:
            slot["mra_values"].append(o.mra_value)
        # 档位明细（§8.3：object_rel_direction 的 easy/medium/hard 先等权聚合）
        lv = slot["levels"].setdefault(
            o.question_type or o.task,
            {"n": 0, "correct": 0, "mra_values": [], "n_mca": 0, "n_na": 0})
        lv["n"] += 1
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

    na_scored = [o.mra_value for o in na if o.mra_value is not None]
    run = EvaluationRun(
        run_id=_run_id(cfg, items[0].episode.split if items else "", len(outcomes)),
        split=items[0].episode.split if items else "",
        n_episodes=len(outcomes),
        accuracy=(sum(1 for o in mca if o.correct) / len(mca)) if mca else None,
        mra=(sum(na_scored) / len(na_scored)) if na_scored else None,
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
            "seed": cfg.seed, "source": getattr(items[0], "source", "unknown"),
            "n_episodes": len(outcomes), "split": run.split,
            "n_unavailable": sum(1 for o in outcomes if o.final_state == "unavailable"),
            "n_unanswerable": sum(1 for o in outcomes if o.final_state == "unanswerable"),
            "deterministic_replay": cfg.deterministic_replay,
            "note": ("mock_light/合成数据：仅管道验证，不构成任何精度结论（§9.2）"
                     if cfg.mode != "real" else "real 模式"),
        })
    return outcomes, run


def _episodic_path(cfg: OnlineRunConfig) -> str:
    """episodic 记忆落盘路径（按 seed/baseline 分文件，便于按 run 审计）。"""
    return f"{cfg.memory_dir}/episodic_seed{cfg.seed}_{cfg.baseline}.jsonl"


def _git_head() -> str:
    """M21：code_commit（无 git 仓库时 "unknown"，不阻断）。"""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
