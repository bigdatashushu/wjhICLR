"""M17/M19 面板执行与 paired A/B 装配（被 `optimize_loop` 与 `offline_driver` 共用）。

从原 `optimize_loop.py` 抽出的公共部分：跑一臂、装配 `PairedOutcome`、把
L1/L2/L3 三层测试接到 `run_optimization_loop`。抽出的原因：§6.2 的完整离线
driver（G-35）与候选级 CLI（§13.5）需要同一套面板语义，避免两份实现漂移。

纪律（硬约束 18）：同一 episode 的 A/B 两臂使用**同一** ReconstructionArtifact
（real 模式按 scene 复用同一 artifact 文件；mock_light 用同 seed 同一合成几何）。
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Callable, Optional, Sequence

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evolution.admission import evaluate_admission
from skill3d.evolution.optimization_loop import (
    DEFAULT_BUDGET_LIMIT,
    CandidateArchive,
    TestReport,
    run_optimization_loop,
)
from skill3d.evolution.paired_score import score_paired
from skill3d.online.runner import OnlineRunConfig, run_episode
from skill3d.reconstruction.run import artifact_path
from skill3d.schemas import (
    AdmissionDecision,
    CandidateRevision,
    PairedOutcome,
    SkillSpec,
)

PanelLevels = dict[str, tuple[PairedOutcome, list, list]]

# G-31 简化 PhaseGate：默认 L1（最小切片）→ L2（inner 全量）→ L3（outer 一次）；
# 论文可简化为 ("L1_minimal_slice", "L3_outer_holdout") 跳过 L2（§5.3 G-31）。
FULL_LEVELS: tuple[str, ...] = ("L1_minimal_slice", "L2_full_inner", "L3_outer_holdout")
SIMPLIFIED_LEVELS: tuple[str, ...] = ("L1_minimal_slice", "L3_outer_holdout")
_LEVEL_PANEL = {"L1_minimal_slice": "L1", "L2_full_inner": "L2", "L3_outer_holdout": "L3"}


def score_of(outcome) -> float:
    """单 episode 分数：MCA → 1/0；NA → MRA。"""
    if outcome.is_mca:
        return 1.0 if outcome.correct else 0.0
    return float(outcome.mra_value) if outcome.mra_value is not None else 0.0


def run_panel(items: Sequence[EpisodeItem], cfg: OnlineRunConfig, trace_store=None,
              llm=None) -> list:
    """跑一臂：逐条 episode 执行在线链（real 模式按 scene 复用 artifact，硬约束 18）。"""
    outcomes = []
    for it in items:
        item_cfg = cfg
        if cfg.mode == "real" and cfg.recon_dir:
            p = artifact_path(cfg.recon_dir, it.episode.scene_name, cfg.recon_method)
            if p.exists():
                item_cfg = replace(cfg, reuse_artifact=str(p))
        outcomes.append(run_episode(it.episode, it.pixels, item_cfg, geometry=it.geometry,
                                    trace_store=trace_store, llm=llm))
    return outcomes


def paired_outcome(pair_id: str, outs_a: list, outs_b: list, task_types: list[str],
                   snapshot_ref: str, seed: int) -> PairedOutcome:
    """由两臂 outcome 装配 `PairedOutcome`（BCa bootstrap CI + Wilcoxon + 效应量，§7）。"""
    # 硬约束 18/21：装配前逐 episode 断言两臂同 artifact 同 frame_set_hash
    from skill3d.skills.paired_ab import assert_paired_outcomes_share_artifact

    assert_paired_outcomes_share_artifact(outs_a, outs_b)
    sa = [score_of(o) for o in outs_a]
    sb = [score_of(o) for o in outs_b]
    st = score_paired(sa, sb, task_types, seed=seed)
    return PairedOutcome(
        pair_id=pair_id,
        snapshot_id=snapshot_ref,
        arm_a_branch_id="arm-a-baseline",
        arm_b_branch_id="arm-b-candidate",
        n_episodes=st["n_episodes"],
        metric="accuracy" if all(o.is_mca for o in outs_a) else "mra",
        mean_a=st["mean_a"],
        mean_b=st["mean_b"],
        delta=st["delta"],
        ci95_lo=st["ci95_lo"],
        ci95_hi=st["ci95_hi"],
        wilcoxon_p=st["wilcoxon_p"],
        wilcoxon_p_bonferroni=st["wilcoxon_p_bonferroni"],
        n_comparisons=st["n_comparisons"],
        degenerate=st["degenerate"],
        cliffs_delta=st["cliffs_delta"],
        cohens_d=st["cohens_d"],
        slice_table=st["slice_table"],
        resource_cost={"n_episodes": st["n_episodes"]},
        slice_no_regression=st["slice_no_regression"],
        within_budget=True,
    )


def delta_by_metric(outs_a: list, outs_b: list) -> tuple[Optional[float], Optional[float]]:
    """分别给出 MCA 精度差与 NA MRA 差（无该类样本则 None）。"""
    mca = [i for i, o in enumerate(outs_a) if o.is_mca]
    na = [i for i, o in enumerate(outs_a) if not o.is_mca]
    d_mca = (sum(1.0 if outs_b[i].correct else 0.0 for i in mca) / len(mca)
             - sum(1.0 if outs_a[i].correct else 0.0 for i in mca) / len(mca)) if mca else None
    d_mra = (sum(float(outs_b[i].mra_value or 0.0) for i in na) / len(na)
             - sum(float(outs_a[i].mra_value or 0.0) for i in na) / len(na)) if na else None
    return d_mca, d_mra


def make_level_report(min_delta_mca: float, min_delta_mra: float,
                      metric: str = "") -> Callable[[PairedOutcome], TestReport]:
    """生成层级判定函数（L1：delta>0 且无 slice 退化；L2/L3：CI 下界>0 且 delta≥min_delta）。"""
    del metric  # 指标名不影响判定（delta 为无量纲比例/相对量）

    def _report(po: PairedOutcome, *, minimal: bool) -> TestReport:
        if minimal:
            ok = po.delta > 0 and po.slice_no_regression
            return TestReport(passed=ok, feedback="" if ok else
                              f"L1 未过: delta={po.delta:+.4f} "
                              f"slice_ok={po.slice_no_regression}")
        min_delta = min_delta_mca if po.metric == "accuracy" else min_delta_mra
        ok = po.ci95_lo > 0 and po.delta >= min_delta and po.slice_no_regression
        return TestReport(passed=ok, feedback="" if ok else
                          f"未过: ci95_lo={po.ci95_lo:+.4f} delta={po.delta:+.4f} "
                          f"min_delta={min_delta:.4f} slice_ok={po.slice_no_regression}")

    return _report


def candidate_skill(cand: CandidateRevision) -> SkillSpec:
    """把候选 spec_content 解析为 SkillSpec（失败即候选非法）。"""
    import json

    return SkillSpec.model_validate(json.loads(cand.spec_content))


def run_candidate_panels(
    root: CandidateRevision,
    *,
    base_cfg: OnlineRunConfig,
    panels: dict[str, list[EpisodeItem]],
    trace_store=None,
    seed: int = 0,
    min_delta_mca: float = 0.02,
    min_delta_mra: float = 0.02,
    revise_fn: Optional[Callable[[CandidateRevision, str], CandidateRevision]] = None,
    static_check_fn: Optional[Callable[[CandidateRevision], bool]] = None,
    archive: Optional[CandidateArchive] = None,
    budget_limit=None,
    levels: Sequence[str] = FULL_LEVELS,
    phase_gate=None,
    print_fn: Callable[[str], None] = print,
) -> tuple[object, PanelLevels]:
    """跑 `levels` 指定的 paired A/B 层级，返回 `(OptimizationRun, level_state)`。

    `panels` 需含 `L1`/`L2`/`L3` 三个 EpisodeItem 序列（L1 通常为 L2 的最小切片）。
    `levels` 默认三层；G-31 简化为 `("L1_minimal_slice", "L3_outer_holdout")` 跳过 L2。
    `phase_gate`（M20 PhaseGate）用于阶段串行：面板执行属 "eval" 阶段，重建期不得并行进入。
    """
    arm_b_cfg = replace(base_cfg, skills=[candidate_skill(root)])
    level_state: PanelLevels = {}
    report = make_level_report(min_delta_mca, min_delta_mra)

    def paired(items: Sequence[EpisodeItem], level: str,
               revision: Optional[CandidateRevision] = None) -> PairedOutcome:
        outs_a = run_panel(items, base_cfg, trace_store)
        outs_b = run_panel(items, arm_b_cfg, trace_store)
        po = paired_outcome(f"pair-{level}-{uuid.uuid4().hex[:8]}", outs_a, outs_b,
                            [it.episode.question_type for it in items],
                            base_cfg.active_snapshot_ref, seed)
        d_mca, d_mra = delta_by_metric(outs_a, outs_b)
        # 退化样本（零方差/完全相同组）→ p 记 None，措辞为"不显著"（§7/E-2）
        p_txt = ("退化(不显著)" if po.degenerate and po.wilcoxon_p is None
                 else f"{po.wilcoxon_p:.4f}" if po.wilcoxon_p is not None else "n/a")
        print_fn(f"[{level}] n={po.n_episodes} delta={po.delta:+.4f} "
                 f"CI95=[{po.ci95_lo:+.4f},{po.ci95_hi:+.4f}] p={p_txt} "
                 f"cliff={po.cliffs_delta:+.3f} d={po.cohens_d:+.3f} "
                 f"d_mca={d_mca} d_mra={d_mra} slice_ok={po.slice_no_regression}")
        if d_mca is not None and d_mra is not None:
            print_fn("        面板同时含 MCA/NA：delta 为混合分数，结论按题型切片看")
        if trace_store is not None:
            # 附候选/版本 id：PairedOutcome schema 无该字段，但审计回溯（G-40）
            # 需要能从 revision_id 直接拉到实验结果，故在 trace 记录里带上。
            trace_store.append("paired_outcome", {
                **po.model_dump(),
                "root_candidate_id": root.root_candidate_id,
                "revision_id": (revision.revision_id if revision is not None else ""),
            })
        return po

    levels_cfg = tuple(levels)
    for lv in levels_cfg:
        if lv not in _LEVEL_PANEL:
            raise ValueError(f"未知层级: {lv}（可选 {FULL_LEVELS}）")

    def run_test_fn(revision: CandidateRevision, level: str) -> TestReport:
        if level not in levels_cfg:
            # 简化 PhaseGate 下被跳过的层级：视作通过（不消耗 rollout，由上层记录）
            print_fn(f"[{level}] 已按 levels={levels_cfg} 跳过（G-31 简化阶段门）")
            return TestReport(passed=True, feedback="")
        if phase_gate is not None:
            if not phase_gate.enter("eval"):
                raise RuntimeError("PhaseGate 拒绝进入 eval 阶段（重建期未退出，M20）")
        po = paired(panels[_LEVEL_PANEL[level]], _LEVEL_PANEL[level], revision)
        level_state[_LEVEL_PANEL[level]] = (po, [], [])
        return report(po, minimal=(level == "L1_minimal_slice"))

    if revise_fn is None:
        raise ValueError("run_candidate_panels 需要 revise_fn（REVISE 阶段由 M16 产 patch）")
    if static_check_fn is None:
        def static_check_fn(_rev):  # type: ignore[misc]
            try:
                candidate_skill(_rev)
                return True
            except Exception:  # noqa: BLE001
                return False

    run = run_optimization_loop(
        root, run_test_fn, revise_fn,
        static_check_fn=static_check_fn,
        archive=archive if archive is not None else CandidateArchive(),
        budget_limit=budget_limit or DEFAULT_BUDGET_LIMIT,
    )
    return run, level_state


def admit(root: CandidateRevision, po: PairedOutcome, *, outer_items: Sequence[EpisodeItem],
          no_leakage: bool, n_min: int, min_delta_mca: float, min_delta_mra: float,
          counterexamples: Optional[list] = None) -> AdmissionDecision:
    """准入门装配（硬约束 13：硬门一票否决）。"""
    n_cross_scene = len({it.episode.scene_name for it in outer_items})
    return evaluate_admission(
        candidate_id=root.root_candidate_id,
        paired_outcomes=[po],
        counterexamples=list(counterexamples or []),
        no_leakage=no_leakage,
        n_cross_scene=n_cross_scene,
        within_budget=po.within_budget,
        min_delta_required=min_delta_mca if po.metric == "accuracy" else min_delta_mra,
        n_min=n_min,
    )
