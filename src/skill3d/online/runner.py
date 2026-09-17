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
from typing import Optional, Sequence

import numpy as np

from skill3d.evaluation.accuracy import extract_option_letter, mca_correct
from skill3d.evaluation.mra import mra_single, parse_numeric_answer
from skill3d.fsm.online_fsm import OnlineFSM, OnlineState
from skill3d.gates import iqa
from skill3d.gates.input_gate import input_gate
from skill3d.reconstruction.vggt_runner import ReconstructionFailed, reconstruct
from skill3d.reconstruction_gate.scene_state import quality_gate
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
# 合成帧的清晰度/曝光判定（与 M2 input_gate 阈值同源，用于填 InputFrame 统计）
_TH_BLUR = 100.0
_TH_EXPOSURE = 0.05


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
    recon_dir: str = "data/reconstructions"
    work_dir: str = "data/reconstructions/mock_light"
    vllm_endpoints: list[str] = field(default_factory=list)
    vllm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    recon_method: str = "vggt"              # vggt | dust3r_mast3r | colmap（§5.2）
    reuse_artifact: Optional[str] = None    # 复用既有 artifact JSON（硬约束 18：A/B 同源）
    max_tokens: int = 4096
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
    direct_answer: Optional[str] = None     # C0 基线：答案不经沙箱，生成阶段即产出


@dataclass
class _SynthResult:
    """M8 产出。direct_answer 仅 C0（无 program）时非空。"""

    program: Optional[EpisodeProgram]
    source: str          # vllm | deterministic_stub | none
    note: str
    direct_answer: Optional[str] = None


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
            note="模型/服务不可用 → episode 记 unavailable（§4 M8 字段 9）")
    if outcome.scene_route == "fallback_2d_only":
        return FailureTaxonomy(
            episode_id=eid, categories=["scale_unknown"],
            note="重建质量不足或尺度未知 → 受限 Tool 集（§4 M4 字段 9）")
    if outcome.verify is not None and not outcome.verify.passed:
        return FailureTaxonomy(
            episode_id=eid, categories=["verifier_reject"],
            note=f"几何校验未过: {outcome.verify.violations}（§4 M11）")
    joined = " | ".join(outcome.notes)
    if "M9 AST 拒绝" in joined:
        return FailureTaxonomy(episode_id=eid, categories=["program_syntax"],
                               note="AST 检查连续失败（§4 M9 字段 9）")
    if "M10 执行错误" in joined:
        return FailureTaxonomy(episode_id=eid, categories=["tool_contract"],
                               note="沙箱执行连续失败（§4 M10 字段 9）")
    if "整体不合格" in joined:
        return FailureTaxonomy(episode_id=eid, categories=["perception"],
                               note="输入门禁整体不合格（§4 M2 字段 9）")
    if "M3 重建失败" in joined:
        return FailureTaxonomy(episode_id=eid, categories=["reconstruction"],
                               note="重建降级链全部失败（§4 M3 字段 9）")
    return FailureTaxonomy(episode_id=eid, categories=["evaluator_noanswer"],
                           note="未产出答案（§6.1 不可恢复）")


# --------------------------------------------------------------- episode 执行 ----

def run_episode(
    episode: VSIBenchEpisode,
    pixels: Sequence[np.ndarray],
    cfg: OnlineRunConfig,
    *,
    geometry: Optional[synth.SyntheticGeometry] = None,
    trace_store: Optional[TraceStore] = None,
    llm=None,
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
        return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store)

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

    # ---------------- M2 INPUT_GATE ----------------
    if fsm.state is OnlineState.INPUT_GATE:
        verdict = input_gate(pixels) if pixels else input_gate(list(episode.frames))
        n_frames = len(pixels) or len(episode.frames)
        notes.append(f"M2 input_gate level={verdict.level} action={verdict.action} "
                     f"degraded_frames={len(verdict.degraded_frame_ids)}/{n_frames}")
        episode = episode.model_copy(update={"frames": _frames_with_stats(episode, pixels)})

        action = verdict.action
        if action == "drop_and_refill":
            dropped = set(verdict.degraded_frame_ids)
            keep = [i for i in range(len(pixels)) if i not in dropped]
            notes.append(f"M2 局部低质：屏蔽 {len(pixels) - len(keep)} 帧后继续，"
                         "不阻断流程（§4 M2 字段 9）")
            pixels = [pixels[i] for i in keep]
            action = "proceed"
        fsm.step("gate_done", {"action": action})
        states.append(fsm.state.value)
        if fsm.state is OnlineState.ANSWER:
            notes.append("M2 整体不合格 → unanswerable（§4 M2 字段 9）")
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                             verdict=verdict)

    # ---------------- M3 RECONSTRUCT ----------------
    if fsm.state is OnlineState.RECONSTRUCT:
        if cfg.reuse_artifact:
            # 复用既有 artifact（离线 paired A/B 两臂必须同源，硬约束 18）
            from pathlib import Path as _P

            from skill3d.schemas import ReconstructionArtifact as _RA

            art = _RA.model_validate_json(_P(cfg.reuse_artifact).read_text(encoding="utf-8"))
            notes.append(f"M3 复用既有 artifact（A/B 同源，硬约束 18）: {cfg.reuse_artifact}")
            fsm.step("done")
        elif cfg.mode == "real":
            try:
                art = reconstruct(pixels, episode.scene_name, cfg.recon_dir,
                                  method=cfg.recon_method)
                notes.append(f"M3 重建完成 method={art.recon_method} "
                             f"scale_known={art.scale_known}")
                fsm.step("done")
            except ReconstructionFailed as exc:
                notes.append(f"M3 重建失败（降级链已走完）: {exc}")
                fsm.step("failed")
        else:
            if geometry is None:
                raise ValueError("mode=mock_light 必须传入 geometry（online.synthetic 构造）")
            notes.append("M3 跳过真实重建：mock_light 使用合成几何（非 VGGT/COLMAP 产物）")
            fsm.step("done")
        states.append(fsm.state.value)
        if fsm.state is OnlineState.ANSWER:
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                             verdict=verdict)

    # ---------------- M4 QUALITY_GATE（+ M5 对象绑定）----------------
    if fsm.state is OnlineState.QUALITY_GATE:
        if art is not None:
            scene, handle = _scene_from_artifact(art, pixels)
            notes.append(f"M4 真实门禁 route={scene.route} "
                         f"overall_quality={art.quality.overall_quality:.3f}")
        else:
            scene, handle, _q = synth.build_scene_state(geometry, pixels)  # type: ignore[arg-type]
            notes.append(f"M4 合成门禁 route={scene.route}"
                         "（mock_light：G1-G11 在合成数据上真算）")
        outcome.scene_route = scene.route
        if cfg.mode == "real":
            handle, m5_note = _bind_objects_best_effort(handle, pixels, scene)
            notes.append(m5_note)
        action = {"unanswerable": "unanswerable",
                  "fallback_2d_only": "fallback_2d_only"}.get(scene.route, "proceed")
        fsm.step("gate_done", {"action": action})
        states.append(fsm.state.value)
        if fsm.state is OnlineState.ANSWER:
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                             verdict=verdict, scene=scene, handle=handle)

    # ---------------- M7 CLASSIFY_TASK + RETRIEVE_SKILL ----------------
    if fsm.state is OnlineState.CLASSIFY_TASK:
        cls = classify(episode)
        notes.append(f"M7 题型={cls.question_type} is_mca={cls.is_mca}")
        fsm.step("done")
        states.append(fsm.state.value)

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
                          feedback=None, geometry=geometry)
        outcome.synthesis_source = res.source
        direct_answer = res.direct_answer
        program = res.program
        if program is None:
            outcome.final_state = "unavailable"
            outcome.states = list(states)
            notes.append(res.note)
            notes.append("M8 生成失败 → episode 记 unavailable（§4 M8 字段 9）；"
                         "起服务：bash scripts/serve_qwen3vl_dp8.sh")
            return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                             verdict=verdict, scene=scene, handle=handle)
        notes.append(f"M8 program 来源={res.source} program_id={program.program_id}")
        if res.note:
            notes.append(res.note)
        outcome.program = program
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
                          feedback=check.violations, geometry=geometry)
        outcome.synthesis_source = res.source
        program = res.program
        outcome.program = program
        if program is None:
            notes.append(f"M9 重生成失败: {res.note}")
            break
        fsm.step("done")
        states.append(fsm.state.value)

    # ---------------- M10 SANDBOX_EXECUTE ----------------
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
                episode, program, handle, pixels, cfg, receipts)
            if not cfg.deterministic_replay:
                wallclock = time.perf_counter() - t0
            program_trace = program_trace.model_copy(update={"wallclock_s": wallclock})
            if cell.error_code is None and cell.answer is not None:
                receipts.append("cell_run", {"program_id": program.program_id,
                                             "answer": cell.answer})
                notes.append(f"M10 执行成功 steps={program_trace.steps}")
                fsm.step("ok")
            else:
                receipts.append("cell_run",
                                {"program_id": program.program_id,
                                 "error_code": cell.error_code},
                                error_code=cell.error_code)
                notes.append(f"M10 执行错误（第 {fsm.kernel_restart_count + 1} 次）: "
                             f"{cell.error} / {cell.error_code}")
                fsm.step("error")
            states.append(fsm.state.value)
            if fsm.state is OnlineState.ANSWER:
                # 两级兜底：no-tool CoT → 正则抽取（§6.1），保证 best-effort 答案
                best = _best_effort_answer(cell.stdout_tail, llm, episode, cfg)
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
        outcome.direct_answer = direct_answer
        if answer is not None and "unanswerable" not in fsm.answer_flags:
            outcome.final_state = "answer"
        fsm.step("done")
        states.append(fsm.state.value)

    return _finalize(outcome, fsm, cfg, episode, states, receipts, trace_store,
                     verdict=verdict, scene=scene, handle=handle)


# ------------------------------------------------------------------ 各阶段实现 ----

def _frames_with_stats(episode: VSIBenchEpisode, pixels: Sequence[np.ndarray]) -> list[InputFrame]:
    """用真实 IQA 统计填 InputFrame（M2 职责；M1 只给占位，§4 M1 字段 7 注）。"""
    out: list[InputFrame] = []
    for i, fr in enumerate(episode.frames):
        if i < len(pixels):
            img = pixels[i]
            blur = iqa.laplacian_var(img)
            p_over, p_under = iqa.exposure_ratios(img)
            out.append(fr.model_copy(update={
                "blur_var": blur,
                "overexposed_ratio": p_over,
                "underexposed_ratio": p_under,
                "quality_ok": not (blur < _TH_BLUR or p_over > _TH_EXPOSURE
                                   or p_under > _TH_EXPOSURE),
            }))
        else:
            out.append(fr)
    return out


def _scene_from_artifact(art, pixels) -> tuple[SceneState, SceneHandle]:
    """M4 真实路径：从 artifact 读数组 → G1-G11 → SceneState + SceneHandle。"""
    def _load(ref: str):
        try:
            return np.load(ref) if ref else None
        except Exception:
            return None

    depth, c2w, intr = _load(art.depth_maps), _load(art.c2w_list), _load(art.intrinsics)
    scene = quality_gate(art, frames=pixels, depth_maps=depth, c2w_list=c2w,
                         artifact_ref=art.artifact_id)
    handle = SceneHandle(scene, objects=[], c2w_list=c2w, intrinsics=intr,
                         quality_overall=art.quality.overall_quality)
    return scene, handle


def _bind_objects_best_effort(handle: SceneHandle, pixels, scene: SceneState):
    """M5 对象绑定：SAM2 未配置等失败 → 记降级、不阻断（§4 M5 字段 9）。"""
    from skill3d.segmentation.sam2_tracker import track_objects

    try:
        objects = list(track_objects(pixels, scene=handle))
    except Exception as exc:  # noqa: BLE001 - checkpoint/依赖/传播失败
        return handle, (f"M5 对象绑定不可用（{type(exc).__name__}: {exc}）→ "
                        "相关 Tool 查询回退全场景（TODO_USER_INPUT: SAM2 checkpoint）")
    if not objects:
        return handle, "M5 未产出对象（SAM2 未配置）"
    return SceneHandle(scene, objects=objects, c2w_list=handle._c2w,  # noqa: SLF001
                       intrinsics=handle._k, quality_overall=handle.quality_overall), \
        f"M5 绑定对象 {len(objects)} 个"


def _synthesize(episode: VSIBenchEpisode, scene, handle, skills, cfg: OnlineRunConfig,
                llm, feedback: Optional[list[str]] = None,
                geometry: Optional[synth.SyntheticGeometry] = None) -> _SynthResult:
    """M8：生成 program（或 C0 直答）。"""
    if cfg.baseline == "C0_direct_vlm":
        # §16.1 C0：直接自然语言作答，不编排 Tool → 无 program，不执行沙箱
        answer, note = _direct_vlm_answer(episode, cfg, llm)
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
        if geometry is None:
            raise ValueError("mock_light 需要 geometry（online.synthetic 构造）")
        src = synth.stub_program(episode.question_type, episode, geometry)
        return _SynthResult(assemble_program(src, []), "deterministic_stub",
                            "M8 mock_light 确定性 stub program"
                            "（非 Qwen3-VL-8B 输出，仅管道验证）")

    prompt = _build_prompt(episode, scene, handle, skills, feedback)
    client = llm if llm is not None else _make_vllm_client(cfg)
    if client is None:
        return _SynthResult(None, "none",
                            "M8 vLLM 未配置：用 --vllm-endpoint 指定本地 Qwen3-VL-8B endpoint"
                            "（bash scripts/serve_qwen3vl_dp8.sh）；§4 M8 字段 9 → 记 unavailable")
    try:
        text = client.chat(prompt, max_tokens=cfg.max_tokens)
        program = assemble_program(text, [f"{s.skill_id}@{s.semver}" for s in skills or []])
    except SynthesisError as exc:
        return _SynthResult(None, "vllm", f"M8 program 解析失败: {exc}")
    except Exception as exc:  # noqa: BLE001 - 网络/服务不可用
        return _SynthResult(None, "vllm", f"M8 vLLM 调用失败: {type(exc).__name__}: {exc}")
    return _SynthResult(program, "vllm", "")


def _direct_vlm_answer(episode: VSIBenchEpisode, cfg: OnlineRunConfig,
                       llm) -> tuple[Optional[str], str]:
    if cfg.mode == "mock_light":
        return synth.stub_direct_answer(episode), (
            "M8/C0 mock_light 确定性 stub 直答（非模型输出，仅管道验证）")
    client = llm if llm is not None else _make_vllm_client(cfg)
    if client is None:
        return None, "M8/C0 vLLM 未配置 → 记 unavailable（TODO_USER_INPUT: endpoint）"
    q = episode.question
    if episode.options:
        q += "\n选项: " + "; ".join(episode.options) + "\n只回答选项字母。"
    try:
        return client.chat([{"role": "user", "content": q}], max_tokens=256), "vllm"
    except Exception as exc:  # noqa: BLE001
        return None, f"M8/C0 调用失败: {type(exc).__name__}: {exc}"


def _build_prompt(episode, scene, handle, skills, feedback) -> str:
    """M8 prompt：只给 SceneState 摘要 + Tool 文档 + Skill 模板，绝不含 GT（§4 M8）。"""
    text = PromptBuilder().render(
        question=episode.question,
        scene_summary=scene.summary,
        scene_frame=scene.frame,
        scale_known=scene.scale_known,
        tool_docs=REGISTRY.docs(),
        options=episode.options,
        skills=skills,
    )
    if feedback:
        text += f"\n上一次生成被 AST 拒绝，原因：{feedback}\n请修正后重新输出。"
    return text


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


def _execute_program(episode, program, handle, pixels, cfg: OnlineRunConfig, receipts):
    """M10：建 kernel → 执行 program → 收集 ProgramExecutionTrace（§5.4）。"""
    mock_switch = MockSwitch(light_handle=handle) if cfg.mode != "real" else None
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
                        cfg: OnlineRunConfig) -> Optional[str]:
    """§6.1 两级兜底：no-tool CoT（需 vLLM）→ 正则抽取。"""
    text = ""
    client = llm if llm is not None else _make_vllm_client(cfg)
    if client is not None:
        try:
            text = client.chat(
                [{"role": "user",
                  "content": episode.question + "\n只回答答案本身，不要解释。"}],
                max_tokens=128)
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
              scene=None, handle=None) -> EpisodeOutcome:
    """收尾：failure 归因、receipts 链校验、EpisodeTrace 落盘（M13）。"""
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
    )
    if trace_store is not None:
        trace_store.append("episode_trace", outcome.episode_trace)
        if outcome.program_trace is not None:
            trace_store.append("program_trace", outcome.program_trace)
        if outcome.verify is not None:
            trace_store.append("geometry_check", outcome.verify)
        trace_store.append("evaluation_result", {
            "qa_id": episode.qa_id,
            "question_type": episode.question_type,
            "is_mca": outcome.is_mca,
            "predicted": outcome.predicted,
            "ground_truth": episode.ground_truth,
            "correct": outcome.correct,
            "mra_value": outcome.mra_value,
        })
    return outcome


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
    outcomes = [run_episode(it.episode, it.pixels, cfg, geometry=it.geometry,
                            trace_store=store, llm=llm) for it in items]

    mca = [o for o in outcomes if o.is_mca]
    na = [o for o in outcomes if not o.is_mca]
    per_task: dict[str, dict] = {}
    for o in outcomes:
        slot = per_task.setdefault(o.question_type,
                                   {"n": 0, "correct": 0, "mra_values": []})
        slot["n"] += 1
        if o.correct:
            slot["correct"] += 1
        if o.mra_value is not None:
            slot["mra_values"].append(o.mra_value)
    for slot in per_task.values():
        slot["accuracy"] = (slot["correct"] / slot["n"]) if slot["n"] else None
        vals = slot.pop("mra_values")
        slot["mra"] = (sum(vals) / len(vals)) if vals else None

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


def _git_head() -> str:
    """M21：code_commit（无 git 仓库时 "unknown"，不阻断）。"""
    import subprocess

    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"
