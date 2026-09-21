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
from skill3d.reconstruction_gate.scene_state import question_gate, quality_gate
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
from skill3d.synthesis.program_assembler import SynthesisError, assemble_program
from skill3d.synthesis.prompt_builder import PromptBuilder
from skill3d.tools import REGISTRY
from skill3d.tools.mock_switch import MockSwitch
from skill3d.tools.scene_handle import SceneHandle
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
    recon_method: str = "vggt"              # vggt | vggt_sparse_ba | dust3r_mast3r | colmap
    # 按题型选择答案来源（任务级策略，非逐题 oracle）：命中的题型改用**直答 VLM**
    # （同一 32 帧 + 问题）。用途：当某题型的程序路径弱于直答时，用直答拿分；
    # 策略必须在 inner_validation 上定、在 outer_holdout 上验证，禁止用 GT 逐题挑选。
    direct_answer_tasks: set[str] = field(default_factory=set)
    reuse_artifact: Optional[str] = None    # 复用既有 artifact JSON（硬约束 18：A/B 同源）
    max_tokens: int = 4096
    max_images: int = 32                    # M8 送进模型的最大帧数（与 §4 M1 对齐）
    max_pixels: int = 131072                # 实测起点：32 帧 ≈ 9.7k prompt tokens（A-4）
    max_model_len: int = 32768
    ba_enabled: bool = False                # BA route（§10.1 [Conditional Go]）
    # ---- v4 尺度评估（HC29–33）----
    # 冻结 conformal 校准器（在线**只读**）；缺失/版本不符 → scale_confidence=low
    scale_calibration_path: Optional[str] = None
    # v5.1：多校准器目录与被评测数据集（按数据集选同源冻结校准器）
    scale_calibration_dir: str = ""
    evaluation_datasets: list[str] = field(default_factory=list)
    # nominal coverage（如 0.90）；必须与冻结校准器一致（[TODO_CALIBRATE]）
    scale_confidence_level: float = 0.90
    # D-3 回灌恢复层（[Conditional Go]）：回灌修复率 ≥50% 才启用（TODO_CALIBRATE）
    enable_tool_contract_replay: bool = False
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
    g9_tracker_consistency: Optional[float] = None  # G9 SAM2 mask IoU 均值（M5 产物）
    # v5 HC38：本版**不设置** geometric_coverage 门，也不存在替代门。
    # 该字段恒为 "not_defined"；trace/报告**不得**写 "coverage passed"。
    coverage_gate_status: str = "not_defined"
    # v5 HC37：本 episode 所用 artifact 的重投影状态（G5 是否 computed）
    reprojection_status: str = "not_available"
    # M2 被动观测的降级 flag（只打权重/标记，不改帧集，硬约束 21）
    input_degradation_flags: list[str] = field(default_factory=list)
    # HC26：M8 实际收到的图像数（必须等于统一 FrameSet 帧数，禁止静默丢帧）
    n_images_to_synthesizer: int = 0
    # `room_size_estimation` medium 档的三态平面质量输入（TODO_CALIBRATE：阈值未标定）。
    # None = 证据缺失 → fail-closed 不授权（不得默认放行）。
    plane_quality_ok: Optional[bool] = None
    direct_answer: Optional[str] = None     # C0 基线：答案不经沙箱，生成阶段即产出
    # 答案来源：program（沙箱执行）/ direct_vlm（C0）/ direct_vlm_routed（按题型策略回退）
    answer_source: str = ""
    # --- D-3 / 硬约束 22/23：契约与质量的显式记录（过程指标与归因用）---
    quality_status: str = "not_computed"
    overall_quality: Optional[float] = None
    tool_contract_hits: int = 0             # 本 episode 命中的 tool_contract 次数
    replay_used: bool = False               # 是否用过"回灌一次"
    trimmed_regen_used: bool = False        # 是否用过裁剪 prompt 重生成
    abstained: bool = False                 # 显式 abstain（主榜按错计，不刷分）
    answer_untrusted: bool = False          # 答案依赖过契约失败的 Tool → 不得采纳
    frame_set_hash: str = ""
    scene_summary: str = ""                 # 送给 M8 的场景摘要（错误归因用）
    m8_prompt: str = ""                     # M8 文本 prompt（不含图像；错误归因用）
    scale_source: str = ""                  # 尺度来源（§7 RunManifest / D-2）
    artifact_ref: str = ""                  # 本 episode 用的重建产物 ref（硬约束 18 审计）
    # --- v4 尺度（HC29–33）：逐 episode 事实，供 §7/§10.2 报告与审计 ---
    scale_confidence: str = "low"
    scale_ci_rel: Optional[float] = None    # 相对 CI 半宽（分数口径，HC29）
    scale_ci_abs_m: Optional[float] = None
    scale_calibration_id: Optional[str] = None
    scale_conflict: bool = False
    scale_empirical_coverage: Optional[float] = None
    allowed_metric_tasks: list[str] = field(default_factory=list)
    authorized_metric_tasks: list[str] = field(default_factory=list)  # 逐题门控后
    n_anchors_fired: int = 0
    n_anchors_accepted: int = 0


@dataclass
class _SynthResult:
    """M8 产出。direct_answer 仅 C0（无 program）时非空。"""

    program: Optional[EpisodeProgram]
    source: str          # vllm | deterministic_stub | none
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
    """失败归因（仅失败时非空，§5.7 FailureTaxonomy）。categories 取 §5.7 受控枚举。"""
    eid = episode.qa_id
    if outcome.final_state in ("answer", "answer_best_effort"):
        return None
    if outcome.final_state == "unavailable":
        return FailureTaxonomy(
            episode_id=eid, categories=["evaluator_noanswer"],
            note="模型/服务不可用 → episode 记 unavailable（§4 M8 字段 9；"
                 "服务故障记 service_unavailable，见 §6.1）")
    if outcome.abstained or "tool_contract" in outcome.answer_flags:
        return FailureTaxonomy(
            episode_id=eid, categories=["tool_contract"],
            note=("Tool 契约违规（产物缺失/局部质量门未过/域值错误）→ 显式 abstain，"
                  f"主榜按错计（§5.1/D-3，硬约束 23）；hits={outcome.tool_contract_hits}，"
                  f"replay={outcome.replay_used}，trimmed={outcome.trimmed_regen_used}"))
    if outcome.scene_route == "fallback_2d_only":
        note = ("measurement 题因尺度不可用降级 2D-only（§7.1 G-11 验收）"
                if "g11_measurement_2d_only" in outcome.answer_flags
                else "重建质量不足或尺度未知 → 受限 Tool 集（§4 M4 字段 9）")
        return FailureTaxonomy(episode_id=eid, categories=["scale_unknown"], note=note)

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
                                      use_ba=cfg.ba_enabled,
                                      frame_set=episode.frame_set,
                                      scale_calibration_path=cfg.scale_calibration_path,
                                      scale_calibration_dir=cfg.scale_calibration_dir,
                                      evaluation_datasets=cfg.evaluation_datasets,
                                      scale_confidence_level=cfg.scale_confidence_level)
                    art_path = existing
                    notes.append(f"M3 重建完成 method={art.recon_method} "
                                 f"quality_status={art.quality_status} "
                                 f"frame_set_hash={art.frame_set_hash[:12]} "
                                 f"scale_known={art.scale_known}")
                    fsm.step("done")
                except ReconstructionFailed as exc:
                    notes.append(f"M3 重建失败（降级链已走完）: {exc}")
                    fsm.step("failed")
        else:
            if geometry is None:
                raise ValueError("mode=mock_light 必须传入 geometry（online.synthetic 构造）")
            notes.append("M3 跳过真实重建：mock_light 使用合成几何（非 VGGT/COLMAP 产物）")
            fsm.step("skip")   # 尺度评估随 artifact 落盘，合成路径跳过两态
        states.append(fsm.state.value)
        if fsm.state is OnlineState.ANSWER:
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                     episodic=episodic,
                             verdict=verdict)

    # ---------------- M3.5 SCALE_ESTIMATE + SCALE_CALIBRATE（v4 §5.1）----------------
    # 真实路径：尺度评估已在 M3 内完成（`assess_scale` → 写回 artifact）。两态在此
    # 只做**观测与落档**：置信档、冲突、校准器 id、逐题型授权。失败（无锚点）不中止
    # episode —— 落 low 并使 `allowed_metric_tasks=∅`，非尺度 3D 能力不受影响（HC33）。
    if fsm.state is OnlineState.SCALE_ESTIMATE:
        n_fired = len(getattr(art, "scale_anchor_fired", []) or [])
        n_acc = sum(1 for a in (getattr(art, "scale_anchor_fired", []) or [])
                    if getattr(a, "accepted", False))
        outcome.n_anchors_fired = n_fired
        outcome.n_anchors_accepted = n_acc
        notes.append(f"M3.5 SCALE_ESTIMATE 锚点 {n_fired} 个（接受 {n_acc}）；"
                     f"conflict={getattr(art, 'scale_conflict', False)}")
        fsm.step("done")
        states.append(fsm.state.value)

    if fsm.state is OnlineState.SCALE_CALIBRATE:
        allowed = sorted(getattr(art, "allowed_metric_tasks", None) or set())
        outcome.scale_confidence = str(getattr(art, "scale_confidence", "low"))
        outcome.scale_ci_rel = getattr(art, "scale_ci_rel", None)
        outcome.scale_ci_abs_m = getattr(art, "scale_ci_abs_m", None)
        outcome.scale_calibration_id = getattr(art, "scale_calibration_id", None)
        outcome.scale_calibration_dataset = str(
            getattr(art, "scale_calibration_dataset", "") or "")
        outcome.scale_dataset_match = getattr(art, "scale_dataset_match", None)
        outcome.scale_conflict = bool(getattr(art, "scale_conflict", False))
        outcome.scale_empirical_coverage = getattr(art, "scale_empirical_coverage", None)
        outcome.allowed_metric_tasks = list(allowed)
        # HC29 口径复核（fail-closed）：不自洽 / 旧字段（无 v4 CI）→ 一律 low + 收回米制题型。
        # 旧字段被降级是**规格要求**（只读迁移、不准入），不是缺陷。
        from skill3d.schemas.reconstruction import _ci_inconsistency_reason

        ci_reason = _ci_inconsistency_reason(art.model_dump())
        if ci_reason:
            # HC31：降级不写 `scale_conflict`（该字段只表示锚点冲突）
            art = art.model_copy(update={"scale_confidence": "low",
                                        "allowed_metric_tasks": set()})
            allowed = []
            outcome.scale_confidence = "low"
            outcome.allowed_metric_tasks = []
            notes.append(f"M3.5 SCALE_CALIBRATE 口径复核 → low（HC29 fail-closed）：{ci_reason}")
        else:
            notes.append("M3.5 SCALE_CALIBRATE 口径自洽"
                         "（scale_ci_abs_m = metric_scale × scale_ci_rel，HC29）")
        notes.append(f"M3.5 SCALE_CALIBRATE calibration_id="
                     f"{outcome.scale_calibration_id} confidence="
                     f"{outcome.scale_confidence} ci_rel="
                     f"{outcome.scale_ci_rel} allowed_metric_tasks={allowed}")
        fsm.step("done", {"metric_tasks_withdrawn": not allowed,
                          "scale_conflict": outcome.scale_conflict})
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
            scene, handle, art = _scene_from_artifact(
                art, pixels, m5_stats, m5_objects, artifact_path=art_path,
                objects_materialized=m5_materialized,
                # paired A/B 复用 frozen artifact（硬约束 18）时不得改写两臂共用文件；
                # 正常 P2 路径按方案 Y 原子写回（D-5/M5 增量补写）
                persist_quality=not bool(cfg.reuse_artifact))
            notes.append(f"M4 质量门禁（唯一事实源）route={scene.route} "
                         f"quality_status={art.quality_status} "
                         f"overall_quality={_overall_str(art)} "
                         f"scale_known={scene.scale_known} "
                         f"scale_conf={scene.scale_confidence} "
                         f"available={sorted(scene.available_artifacts)}")
            outcome.quality_status = art.quality_status
            outcome.overall_quality = _overall_of(art)
            outcome.scale_source = str(getattr(art, "scale_source", "") or "")
        else:
            # mock_light：合成几何 + 真算 G1–G11；M2 的被动观测（flag/weight）并入 route
            scene, handle, _q = synth.build_scene_state(  # type: ignore[arg-type]
                geometry, pixels,
                input_quality_weight=verdict.quality_weight if verdict else 1.0,
                input_degradation_flags=verdict.degradation_flags if verdict else None)
            outcome.quality_status = "computed"
            outcome.overall_quality = float(_q.overall_quality)
            outcome.scale_source = "synthetic_mock_light"
            notes.append(f"M4 合成门禁 route={scene.route} "
                         f"overall_quality={_q.overall_quality:.3f}"
                         "（mock_light：G1-G11 在合成数据上真算）")
        outcome.scene_route = scene.route
        outcome.artifact_ref = art_path or (art.artifact_id if art is not None else "")
        # v4 HC33：场景级尺度事实（**两条路径都记**）。真实路径的校准器 id / 冲突 /
        # 经验覆盖 / 锚点计数在 M3.5 SCALE_CALIBRATE 补齐；mock_light 走合成 GT 尺度
        # （`scale_source=synthetic_mock_light`，仅管道验证，不得当结果）。
        outcome.scale_confidence = str(scene.scale_confidence)
        outcome.scale_ci_rel = scene.scale_ci_rel
        outcome.allowed_metric_tasks = sorted(scene.allowed_metric_tasks)
        outcome.g9_tracker_consistency = _mean_track_iou(m5_stats)
        # v5 HC38：G8 永久退役且**不设替代 geometric_coverage 门** → 恒记 not_defined。
        # 这里不得再出现任何 `coverage_ok=True` 之类的常量放行。
        outcome.coverage_gate_status = "not_defined"
        outcome.reprojection_status = str(
            getattr(art, "reprojection_status", "not_available") or "not_available")
        outcome.input_degradation_flags = list(
            getattr(verdict, "degradation_flags", None) or []) if verdict else []
        action = {"unanswerable": "unanswerable",
                  "fallback_2d_only": "fallback_2d_only"}.get(scene.route, "proceed")
        fsm.step("gate_done", {"action": action})
        states.append(fsm.state.value)
        if fsm.state is OnlineState.ANSWER:
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                     episodic=episodic,
                             verdict=verdict, scene=scene, handle=handle)

    # ---------------- M7 CLASSIFY_TASK + 逐题门控 + RETRIEVE_SKILL ----------------
    if fsm.state is OnlineState.CLASSIFY_TASK:
        cls = classify(episode)
        outcome.task = cls.task
        notes.append(f"M7 题型={cls.question_type} task={cls.task} is_mca={cls.is_mca}")
        # 逐题门控（v5 HC33 逐题型米制授权；HC38 不设 coverage 门）；无场景则不门控
        decision = None
        if scene is not None:
            decision = question_gate(scene, cls.task,
                                    g9_tracker_consistency=outcome.g9_tracker_consistency,
                                    plane_quality_ok=outcome.plane_quality_ok)
            notes.append(f"M7.5 {decision.note()}")
            outcome.answer_flags.extend(decision.flags)
            outcome.authorized_metric_tasks = sorted(decision.allowed_metric_tasks)
        downgrade = bool(
            decision is not None and decision.allowed
            and scene is not None and decision.route != scene.route
        )
        fsm.step("done", {
            "question_ok": True if decision is None else decision.allowed,
            "reject_flag": "metric_task_reject",
            "downgrade_2d_only": downgrade,
        })
        states.append(fsm.state.value)
        if decision is not None and not decision.allowed:
            outcome.scene_route = "unanswerable"
            if scene is not None:
                scene = scene.model_copy(update={"route": "unanswerable"})
            notes.append("M7.5 逐题门控拒答 → unanswerable")
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                     episodic=episodic,
                             verdict=verdict, scene=scene, handle=handle)
        if scene is not None and decision is not None:
            # v4 HC33：把"本题题型 + 本题授权"写进 scene/handle ——
            # ① 米制 Tool 的逐题授权（`question_type ∈ allowed ∩ supported`）；
            # ② 逐题可用产物集（未授权时收回 `scale`，非米制 3D 产物不受影响）。
            # §3"尺度能力门控"（已确认设计）：尺度为 low 只移除 `scale` 产物与米制
            # Tool，**不得**改变原本合格的 `full_3d` route —— 未授权米制题型时
            # `decision.route` 等于场景 route，此处不再降级 route。
            scene = _scope_scene_to_question(scene, cls.task, decision)
            if decision.route != scene.route:
                scene = scene.model_copy(update={"route": decision.route})
                outcome.scene_route = decision.route
                notes.append(f"M7.5 该题降级 route={decision.route}（米制 Tool 已收回，HC33）")
            if "v5_metric_task_not_authorized" in decision.flags:
                notes.append(
                    f"M7.5 本题米制 Tool 已收回（route 保持 {scene.route}）："
                    f"allowed_metric_tasks=[]；非米制 3D 产物保留="
                    f"{sorted(set(scene.available_artifacts) - {'scale'})}")
            handle = _retarget_handle(handle, scene)

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

            # ---- D-3：tool_contract 恢复阶梯（整个 episode 最多回灌 1 次）----
            if error_code == "tool_contract":
                outcome.tool_contract_hits += 1
                outcome.answer_flags.append("tool_contract")
                detail = cell.error or "程序自捕获契约异常后仍产答案（answer_untrusted）"
                notes.append(
                    f"M10 tool_contract（第 {outcome.tool_contract_hits} 次）: {detail}")
                if kernel is not None:
                    # 硬约束 23 / D-3：答案依赖过契约失败的 Tool → 不得采纳
                    kernel.answer_slot.answer = None
                recovered = False
                # 阶梯：rung1 回灌（[Conditional Go]，默认关）→ rung2 裁剪 prompt（强制
                # fallback_2d_only + 工具子集）→ 两档都用尽才 abstain。rung1 重生成失败时
                # **必须继续落到 rung2**，不得直接 abstain（§5.1）。
                recovered_rung = ""
                for rung, override in _available_contract_rungs(cfg, outcome):
                    if rung == "replay":
                        outcome.replay_used = True
                    else:
                        outcome.trimmed_regen_used = True
                    res, program, round_notes = _regenerate_after_contract(
                        episode, scene, handle, selected_skills, cfg, llm, pixels,
                        geometry, prior_program=program, cell=cell, scene_route=override)
                    notes.extend(round_notes)
                    if res and program is not None:
                        recovered = True
                        recovered_rung = rung
                        break
                if recovered and program is not None:
                    outcome.program = program
                    notes.append(f"M10 恢复层生效（rung={recovered_rung}）")
                    fsm.step("contract_recover")   # 不消耗 kernel 重启预算
                    states.append(fsm.state.value)
                    continue
                # 两档机会用尽 → 显式 abstain（主榜按错计，不刷分）
                outcome.abstained = True
                notes.append("M10 tool_contract 恢复失败 → 显式 abstain"
                             "（§5.1：主榜按错计，答案不得采纳）")
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
    answer_source = ("program" if kernel is not None and answer is not None else
                     ("direct_vlm" if direct_answer is not None else ""))
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
                         artifact_path: str = "",
                         objects_materialized: Optional[bool] = None,
                         persist_quality: bool = True
                         ) -> tuple[SceneState, SceneHandle, object]:
    """M4 真实路径：从 artifact 读数组 → G1-G11 → SceneState + SceneHandle。

    硬约束 22（质量单一事实源）：
    - `art.quality_status == "computed"` → 直接用落盘实算值（P1 方案 X，零重算）；
    - 否则实算并**原子写回** `artifact_path`（方案 Y），返回写回后的新 artifact；
    - route 由 `route_from_quality` fail-closed 判定（NaN/未计算 → 不得 full_3d）。

    G-18 数据源接线：G5 重投影残差读 `art.reproj_errors`（BA 产物），
    G7/G9 从 M5 统计（动态 mask / track IoU）取，缺失则 NaN；
    拿到 M5 统计后会把它们补进已落盘的 quality 并原子写回（D-5 增量补写）。

    `persist_quality=False`（paired A/B 复用 frozen artifact，硬约束 18）：
    补写只进内存副本，不改动两臂共用的那个 artifact。
    """
    stats = m5_stats or {}
    depth, c2w, intr = (_load_npy(art.depth_maps), _load_npy(art.c2w_list),
                        _load_npy(art.intrinsics))
    reproj = _load_npy(art.reproj_errors)
    scene = quality_gate(
        art, frames=pixels, depth_maps=depth, c2w_list=c2w,
        reproj_errors=reproj,
        dynamic_masks=stats.get("dynamic_masks"),
        track_ious=stats.get("track_ious"),
        object_ids=sorted(o.instance_id for o in (objects or [])),
        artifact_ref=art.artifact_id,
        artifact_path=artifact_path or None,
        persist_quality=persist_quality,
    )
    # 写回后的 artifact（quality 已落盘）：后续 M8/M12 一律读它
    art_out = getattr(scene, "artifact", None) or art
    handle = SceneHandle(scene, objects=objects or [], c2w_list=c2w, intrinsics=intr,
                         quality_overall=_overall_of(art_out),
                         objects_materialized=objects_materialized)
    return scene, handle, art_out


def _mean_track_iou(m5_stats: Optional[dict]) -> Optional[float]:
    """G9 门控输入：SAM2 mask 跨帧 IoU 均值；无数据 → None（不额外门控）。"""
    ious = (m5_stats or {}).get("track_ious")
    if not ious:
        return None
    vals = [float(v) for v in ious if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _scope_scene_to_question(scene: SceneState, task: str,
                             decision) -> SceneState:
    """把逐题授权写进 SceneState（v4 HC33）。

    - `question_type` = 本 episode 的规范题型（米制 Tool 授权的第二维）；
    - `allowed_metric_tasks` = 过完全部门控（含 G9 等附加条件）后的本题授权；
    - `available_artifacts` = `route_artifacts_for_question(...)`：未授权时收回
      `scale`，**非米制 3D 产物（depth/poses/point_cloud/objects）不受影响**。
    """
    from skill3d.tools.contract import route_artifacts_for_question

    allowed = set(getattr(decision, "allowed_metric_tasks", None) or set())
    return scene.model_copy(update={
        "question_type": str(task or ""),
        "allowed_metric_tasks": allowed,
        "available_artifacts": route_artifacts_for_question(scene.route, allowed, task),
    })


def _retarget_handle(handle: Optional[SceneHandle], scene: SceneState) -> Optional[SceneHandle]:
    """逐题降级后同步句柄的 SceneState（route/scale_known 供 Tool 与 Verifier 读取）。"""
    if handle is None:
        return None
    return SceneHandle(scene, objects=list(handle._objects.values()),  # noqa: SLF001
                       c2w_list=handle._c2w, intrinsics=handle._k,      # noqa: SLF001
                       quality_overall=handle.quality_overall,
                       objects_materialized=handle.objects_materialized,
                       metric_scale=handle.metric_scale)


def _bind_objects_best_effort(art, pixels, _scene, episode, cfg=None,
                              llm=None) -> tuple[list, dict, list[str], bool]:
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
            # §9：BA route 的深度网格带正方形 pad，mask 必须按仿射映射（非纯缩放）
            grid_transform=(art.grid_transform.model_dump()
                            if getattr(art, "grid_transform", None) is not None else None),
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
                route_override: Optional[str] = None,
                traceback_feedback: Optional[str] = None,
                prior_program: Optional[str] = None) -> _SynthResult:
    """M8：生成 program（或 C0 直答）。

    `route_override` / `traceback_feedback` / `prior_program` 供 D-3 恢复层使用：
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
        source = "vllm" if cfg.mode == "real" else "deterministic_stub"
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
        return _SynthResult(assemble_program(src, []), "deterministic_stub",
                            "M8 mock_light 确定性 stub program"
                            "（非 Qwen3-VL-8B 输出，仅管道验证）")

    prompt = _build_prompt(episode, scene, handle, skills, feedback,
                           route_override=route_override,
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
        program = assemble_program(text, [f"{s.skill_id}@{s.semver}" for s in skills or []])
    except SynthesisError as exc:
        return _SynthResult(None, "vllm", f"M8 program 解析失败: {exc}")
    except Exception as exc:  # noqa: BLE001 - 网络/服务不可用
        from skill3d.synthesis.vllm_client import ServiceUnavailable

        if isinstance(exc, ServiceUnavailable):
            return _SynthResult(None, "vllm",
                                f"M8 service_unavailable（§6.1：服务故障不静默降级）: {exc}")
        return _SynthResult(None, "vllm", f"M8 vLLM 调用失败: {type(exc).__name__}: {exc}")
    return _SynthResult(program, "vllm", "", n_images=n_images, prompt=prompt)


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
                  route_override: Optional[str] = None,
                  traceback_feedback: Optional[str] = None) -> str:
    """M8 prompt：只给 SceneState 摘要 + Tool 文档 + Skill 模板，绝不含 GT（§4 M8）。

    - `tool_docs = REGISTRY.docs(route=...)`：按 route 静态裁剪（D-3a），
      prompt 头部显式写"当前 route=…，可用产物=…"；
    - `route_override`：tool_contract 恢复层的"裁剪重生成"（强制 fallback_2d_only）；
    - `traceback_feedback`：回灌层的裁剪 traceback + 可用产物清单。
    """
    route = route_override or (scene.route if scene is not None else "")
    if route_override and scene is not None:
        from skill3d.tools.contract import available_artifacts_for

        available = sorted(available_artifacts_for(route_override))
    elif handle is not None:
        # prompt 头部的"可用产物"必须是**实际装载**的集合（硬约束 23）：
        # `scene.available_artifacts` 是 route 级声明，M5 失败/数组未加载时会高报，
        # 模型照着写就会撞 ArtifactUnavailableError。
        available = sorted(handle.available_artifacts)
    else:
        available = sorted(getattr(scene, "available_artifacts", set()) or set())
    # v4 HC33：逐题米制授权（未授权时提示词里既没有米制 Tool，头部也写明"无"）
    metric_tasks = sorted(getattr(scene, "allowed_metric_tasks", set()) or set()) \
        if scene is not None else []
    qtype = str(getattr(scene, "question_type", "") or "")
    text = PromptBuilder().render(
        question=episode.question,
        scene_summary=(scene.summary if scene is not None else ""),
        scene_frame=(scene.frame if scene is not None else "world"),
        scale_known=bool(scene.scale_known) if scene is not None else False,
        # 静态裁剪：route（声明）+ 句柄实际装载产物（事实）+ 逐题米制授权 三条件，
        # 使 prompt 列出的 Tool 在执行期一定不会因缺产物/未授权抛错（硬约束 23 / HC33）
        tool_docs=REGISTRY.docs(route=route or None,
                                available=(handle.available_artifacts
                                           if handle is not None else None),
                                allowed_metric_tasks=metric_tasks,
                                question_type=qtype),
        options=episode.options,
        skills=skills,
        route=route,
        available_artifacts=available,
        allowed_metric_tasks=metric_tasks,
        question_type=qtype,
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


def _available_contract_rungs(cfg: OnlineRunConfig, outcome) -> list[tuple[str, Optional[str]]]:
    """本 episode 尚未用掉的 D-3 恢复档位（§5.1 阶梯，每档至多一次）。

    - `("replay", None)`：回灌一次（追加裁剪 traceback + 可用产物清单）——
      [Conditional Go]，仅在 `enable_tool_contract_replay` 打开时可用；
    - `("trimmed", "fallback_2d_only")`：裁剪 prompt（强制受限 route + 工具子集）一次。

    顺序即阶梯顺序；rung1 重生成失败时调用方会继续尝试 rung2。
    """
    rungs: list[tuple[str, Optional[str]]] = []
    if cfg.enable_tool_contract_replay and not outcome.replay_used:
        rungs.append(("replay", None))
    if not outcome.trimmed_regen_used:
        rungs.append(("trimmed", "fallback_2d_only"))
    return rungs


def _regenerate_after_contract(episode, scene, handle, skills, cfg, llm, pixels,
                               geometry, *, prior_program: EpisodeProgram,
                               cell: CellResult,
                               scene_route: Optional[str]) -> tuple[bool, Optional[EpisodeProgram], list[str]]:
    """tool_contract 恢复：回灌一次 / 裁剪 prompt 重生成一次（§5.1 阶梯）。

    返回 `(是否成功, 新 program, notes)`。重生成后必须**重过 M9 AST 检查**，
    并在重执行前 reset kernel user namespace（SpatialClaw §E.3 先例）。

    本层是 [Conditional Go]：正确性不依赖它（默认关闭，需 `enable_tool_contract_replay`
    或在裁剪轮使用），且失败一律落到显式 abstain。
    """
    notes: list[str] = []
    if cfg.mode == "mock_light" and llm is None:
        # mock_light 的 stub program 是确定性的：重生成只会得到同一份 → 直接 abstain
        notes.append("M10 恢复层跳过：mock_light 无模型，stub program 重生成无意义")
        return False, None, notes

    traceback_feedback = _contract_feedback(cell, prior_program, scene)
    res = _synthesize(episode, scene, handle, skills, cfg, llm,
                      feedback=None, geometry=geometry, pixels=pixels,
                      route_override=scene_route,
                      traceback_feedback=traceback_feedback,
                      prior_program=(prior_program.program_source if scene_route is None
                                     else None))
    if res.program is None:
        notes.append(f"M10 恢复层重生成失败: {res.note}")
        return False, None, notes
    check = ast_guard(res.program.program_source, allowed_tools=set(REGISTRY.names()))
    if not check.ok:
        notes.append(f"M10 恢复层重生成未过 M9 AST: {check.violations}")
        return False, None, notes
    label = "回灌" if scene_route is None else "裁剪 prompt（强制 fallback_2d_only）"
    notes.append(f"M10 恢复层：{label}重生成成功，program_id={res.program.program_id}"
                 "（重执行前 reset kernel 命名空间）")
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
        quality_status=str(outcome.quality_status or ""),
        # v5：逐 episode 事实落 trace（§14.1-6 的 smoke 核对项 + §7 复现清单）
        schema_version="5.0",
        quality_metric_version=QUALITY_METRIC_VERSION,
        frame_set_hash=str(outcome.frame_set_hash or ""),
        n_frames=int(len(episode.frames)),
        n_images_to_synthesizer=int(outcome.n_images_to_synthesizer),
        reprojection_status=str(outcome.reprojection_status),
        coverage_gate_status=str(outcome.coverage_gate_status),
        scale_confidence=str(outcome.scale_confidence),
        allowed_metric_tasks=sorted(outcome.allowed_metric_tasks),
        authorized_metric_tasks=sorted(outcome.authorized_metric_tasks),
        overall_quality=(None if outcome.overall_quality is None
                         else float(outcome.overall_quality)),
        input_degradation_flags=sorted(set(outcome.input_degradation_flags or [])),
        answer_source=str(outcome.answer_source),
    )
    if trace_store is not None:
        trace_store.append("episode_trace", outcome.episode_trace)
        if outcome.program_trace is not None:
            trace_store.append("program_trace", outcome.program_trace)
        if outcome.program is not None:
            # M13 增补：程序源码必须留痕，否则"为什么答错"无法归因（§4 M13）
            trace_store.append("episode_program", {
                "qa_id": episode.qa_id,
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
