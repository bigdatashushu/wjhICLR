"""M17/M19 面板执行与 paired A/B 装配（被 `optimize_loop` 与 `offline_driver` 共用）。

从原 `optimize_loop.py` 抽出的公共部分：跑一臂、装配 `PairedOutcome`、把
L1/L2/L3 三层测试接到 `run_optimization_loop`。抽出的原因：§6.2 的完整离线
driver（G-35）与候选级 CLI（§13.5）需要同一套面板语义，避免两份实现漂移。

纪律（硬约束 18）：同一 episode 的 A/B 两臂使用**同一** ReconstructionArtifact
（real 模式按 scene 复用同一 artifact 文件；mock_light 用同 seed 同一合成几何）。
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import replace
from datetime import datetime, timezone
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
from skill3d.reconstruction.run import artifact_path, resolve_artifact_path
from skill3d.schemas import (
    DECISION_CONDITION_KEYS,
    AdmissionDecision,
    CampaignDecision,
    CandidateRevision,
    PairedOutcome,
    PairedPanelReceipt,
    SkillEvaluationBinding,
    SkillSpec,
)
from skill3d.skills.delivery import skill_content_sha256

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
            # §5.2：缓存身份含源标识与帧集内容哈希（旧命名产物按 §17.2 沿用）
            p, _used_legacy = resolve_artifact_path(
                cfg.recon_dir, it.episode.scene_name, cfg.recon_method,
                frame_set=getattr(it.episode, "frame_set", None))
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
    level_state: PanelLevels = {}
    report = make_level_report(min_delta_mca, min_delta_mra)

    def paired(items: Sequence[EpisodeItem], level: str,
               revision: Optional[CandidateRevision] = None) -> PairedOutcome:
        outs_a = run_panel(items, base_cfg, trace_store)
        # Each revision must be the actual B-arm payload. Keeping a root-bound
        # config here would make REVISE a no-op while still recording its id.
        arm_b_cfg = replace(base_cfg, skills=[candidate_skill(revision or root)])
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


def admit(root: CandidateRevision, po: PairedOutcome, *,
          panel_items: Sequence[EpisodeItem],
          no_leakage: bool, n_min: int, min_delta_mca: float, min_delta_mra: float,
          counterexamples: Optional[list] = None) -> AdmissionDecision:
    """准入门装配（硬约束 13：硬门一票否决）。

    §14.3：**候选只在它所属规范题型的固定 inner 面板上准入**，主准入得分在整个
    该题型面板 P 上计算。因此 `po` 必须来自 inner 面板，`panel_items` 是同一个
    面板的题目清单（用于跨场景覆盖计数）。`outer_holdout` 的结果是**快照冻结后的
    独立验证证据**，不作为准入门 —— 用 holdout 选代会让该 holdout 失效。
    """
    n_cross_scene = len({it.episode.scene_name for it in panel_items})
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


# =====================================================================================
# v10 §8.2 / §8.6：固定注入的父 / 候选配对效果评测
# =====================================================================================

def fixed_injection_binding(arm: str, spec: SkillSpec) -> SkillEvaluationBinding:
    """§8.2：为一条被评测 Skill 生成固定注入绑定（正文 hash 即该臂的身份）。"""
    if arm not in ("parent", "candidate"):
        raise ValueError(f"arm 只能是 parent/candidate，收到 {arm!r}")
    key = f"{spec.skill_id}@{spec.version}"
    return SkillEvaluationBinding(
        mode="fixed_skill_evaluation",
        arm=arm,  # type: ignore[arg-type]
        skill_id=str(spec.skill_id),
        skill_version=key,
        content_sha256=skill_content_sha256(spec),
        bypassed_component="retrieval_selection",
    )


def delivered_sha_of(outcome, version_key: str) -> str:
    """从 episode 的检索记录里取"该版本实际交付的正文 hash"（没有则空串）。"""
    for record in (getattr(outcome, "retrieval_records", None) or []):
        payload = (record.model_dump(mode="json") if hasattr(record, "model_dump")
                   else dict(record))
        sha = (payload.get("delivered_content_sha256") or {}).get(version_key, "")
        if sha:
            return str(sha)
    return ""


def _run_error_of(outcome) -> bool:
    status = str(getattr(outcome, "episode_status", "") or "")
    if status:
        return status == "run_error"
    return str(getattr(outcome, "final_state", "")) == "run_error"


def _valid_answer_of(outcome) -> bool:
    """§8.6-3 的"合法答案"：框架解析出合法答案载荷（或 C0 直答文本）。"""
    if _run_error_of(outcome):
        return False
    if getattr(outcome, "answer_payload", None) is not None:
        return True
    return bool(str(getattr(outcome, "direct_answer", "") or ""))


def run_fixed_skill_evaluation(
    *,
    campaign_id: str,
    generation: int,
    seed: int,
    panel_id: str,
    items: Sequence[EpisodeItem],
    parent_spec: SkillSpec,
    candidate_spec: SkillSpec,
    base_cfg: OnlineRunConfig,
    trace_store=None,
    llm=None,
    print_fn: Callable[[str], None] = print,
) -> tuple[PairedPanelReceipt, dict]:
    """§8.2 A/B：A 固定注入父 Skill、B 固定注入候选 Skill（同一批题目与 artifact）。

    规范原文（§8.2）："每次模型请求只包含该臂被评测的一条完整 Skill。除该 Skill 的
    版本和正文外，A/B 的题目、FrameSet、基础 artifact、工具、权限、模型、提示词
    公共部分、求解轮数、重试规则和 seed 必须一致。"

    返回 `(receipt, 原始两臂 outcome)`；`receipt` 里逐题记录了得分、run_error、
    合法答案与**两臂正文 hash**（§16.2 要求核对），供 `decide_promotion` 判定。
    """
    binding_a = fixed_injection_binding("parent", parent_spec)
    binding_b = fixed_injection_binding("candidate", candidate_spec)
    if binding_a.skill_id != binding_b.skill_id:
        raise ValueError(
            f"父 / 候选不同谱系：{binding_a.skill_id} vs {binding_b.skill_id}（§7.2）")
    if not items:
        raise ValueError("固定注入评测需要非空 inner 子面板（§8.5）")

    cfg_a = replace(base_cfg, skills=[parent_spec], evaluation_binding=binding_a)
    cfg_b = replace(base_cfg, skills=[candidate_spec], evaluation_binding=binding_b)
    outs_a = run_panel(items, cfg_a, trace_store, llm=llm)
    outs_b = run_panel(items, cfg_b, trace_store, llm=llm)
    # 硬约束 18/21：逐 episode 断言两臂同 artifact 同 frame_set_hash
    from skill3d.skills.paired_ab import assert_paired_outcomes_share_artifact

    assert_paired_outcomes_share_artifact(outs_a, outs_b)

    scores_a = [score_of(o) for o in outs_a]
    scores_b = [score_of(o) for o in outs_b]
    per_item: list[dict] = []
    for item, oa, ob in zip(items, outs_a, outs_b):
        per_item.append({
            "qa_id": str(item.episode.qa_id),
            "scene_id": str(item.episode.scene_name),
            "score_a": float(score_of(oa)),
            "score_b": float(score_of(ob)),
            "final_state_a": str(getattr(oa, "final_state", "")),
            "final_state_b": str(getattr(ob, "final_state", "")),
            "run_error_a": _run_error_of(oa),
            "run_error_b": _run_error_of(ob),
            "valid_answer_a": _valid_answer_of(oa),
            "valid_answer_b": _valid_answer_of(ob),
            "delivered_sha_a": delivered_sha_of(oa, binding_a.skill_version),
            "delivered_sha_b": delivered_sha_of(ob, binding_b.skill_version),
        })
    n = float(len(per_item))
    body_a = sum(1 for r in per_item
                 if r["delivered_sha_a"] == binding_a.content_sha256)
    body_b = sum(1 for r in per_item
                 if r["delivered_sha_b"] == binding_b.content_sha256)
    receipt = PairedPanelReceipt(
        campaign_id=campaign_id, generation=int(generation), seed=int(seed),
        panel_id=str(panel_id), n_items=len(per_item),
        arm_a_skill_version=binding_a.skill_version,
        arm_b_skill_version=binding_b.skill_version,
        arm_a_content_sha256=binding_a.content_sha256,
        arm_b_content_sha256=binding_b.content_sha256,
        mean_a=sum(scores_a) / n, mean_b=sum(scores_b) / n,
        delta=sum(scores_b) / n - sum(scores_a) / n,
        n_run_error_a=sum(1 for r in per_item if r["run_error_a"]),
        n_run_error_b=sum(1 for r in per_item if r["run_error_b"]),
        valid_answer_rate_a=sum(1 for r in per_item if r["valid_answer_a"]) / n,
        valid_answer_rate_b=sum(1 for r in per_item if r["valid_answer_b"]) / n,
        delivered_a=body_a, delivered_b=body_b, per_item=per_item,
        panel_hash=_panel_hash(items),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    print_fn(f"[固定注入 A/B] seed={seed} n={receipt.n_items} "
             f"父={receipt.mean_a:.4f} 候选={receipt.mean_b:.4f} "
             f"delta={receipt.delta:+.4f} run_error {receipt.n_run_error_a}→"
             f"{receipt.n_run_error_b} 正文入请求 {body_a}/{body_b}（n={receipt.n_items}）")
    return receipt, {"a": outs_a, "b": outs_b}


def _panel_hash(items: Sequence[EpisodeItem]) -> str:
    """面板清单 hash（§8.5：面板清单、scene、qa_id、seed 在看结果前冻结）。"""
    keys = [f"{it.episode.qa_id}:{it.episode.scene_name}" for it in items]
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def decide_promotion(receipts: Sequence[PairedPanelReceipt], *,
                     required_seeds: Sequence[int],
                     candidate_delivered_eligible: bool = True,
                     same_episode_and_artifacts: bool = True,
                     content_and_experience_eligible: bool = True,
                     strict_improvement: bool = True) -> CampaignDecision:
    """§8.6 + §13：双 seed 固定注入结果的确定性准入判定。

    规范原文（§13）："候选在两个固定 seed 上分别满足以下条件才 promote：1. 整个对应
    inner 子面板得分严格提高；2. `run_error` 数量不增加；3. 合法答案率不下降；
    4. 候选至少真实交付一次；…… 任一 seed 持平、下降、候选零交付或运行错误增加，
    则 reject。"
    """
    by_seed = {int(r.seed): r for r in receipts}
    per_seed: dict[str, dict] = {}
    checks: dict[str, bool] = {k: True for k in DECISION_CONDITION_KEYS}
    reasons: list[str] = []
    missing = [s for s in required_seeds if int(s) not in by_seed]
    if missing:
        checks["same_episode_and_artifacts"] = False
        reasons.append(f"缺少 seed {missing} 的固定注入结果（§8.6：两个 seed 均必须满足）")
    for seed in required_seeds:
        r = by_seed.get(int(seed))
        if r is None:
            continue
        improved = (r.mean_b > r.mean_a) if strict_improvement else (r.mean_b >= r.mean_a)
        row = {
            "mean_a": r.mean_a, "mean_b": r.mean_b, "delta": r.delta,
            "panel_score_strictly_improved": improved,
            "run_error_not_increased": r.n_run_error_b <= r.n_run_error_a,
            "valid_answer_rate_not_decreased": (
                r.valid_answer_rate_b >= r.valid_answer_rate_a),
            "candidate_delivered_at_least_once": r.delivered_b > 0,
            "both_arms_body_entered_request": (
                r.delivered_a == r.n_items and r.delivered_b == r.n_items),
            "n_run_error_a": r.n_run_error_a, "n_run_error_b": r.n_run_error_b,
            "valid_answer_rate_a": r.valid_answer_rate_a,
            "valid_answer_rate_b": r.valid_answer_rate_b,
            "delivered_a": r.delivered_a, "delivered_b": r.delivered_b,
            "n_items": r.n_items,
        }
        per_seed[str(seed)] = row
        for key in ("panel_score_strictly_improved", "run_error_not_increased",
                    "valid_answer_rate_not_decreased",
                    "candidate_delivered_at_least_once",
                    "both_arms_body_entered_request"):
            if not row[key]:
                checks[key] = False
                reasons.append(f"seed {seed}: {key}=False")
        if not r.arm_b_content_sha256 or r.arm_a_content_sha256 == r.arm_b_content_sha256:
            checks["no_schema_or_permission_violation"] = False
            reasons.append(f"seed {seed}: 两臂正文 hash 相同或缺失（§8.3 禁止 A/B 同源）")
    checks["same_episode_and_artifacts"] = bool(
        checks["same_episode_and_artifacts"] and same_episode_and_artifacts)
    checks["no_schema_or_permission_violation"] = bool(
        checks["no_schema_or_permission_violation"])
    checks["content_and_experience_eligible"] = bool(content_and_experience_eligible)
    checks["candidate_delivered_at_least_once"] = bool(
        checks["candidate_delivered_at_least_once"] and candidate_delivered_eligible)
    if not content_and_experience_eligible:
        reasons.append("候选内容 / 来源 / 经验资格未通过（§13-6）")
    if not same_episode_and_artifacts:
        reasons.append("A/B 未使用同一 episode / FrameSet / artifact / 配置（§13-5）")
    promote = bool(checks and all(checks.values()))
    if promote:
        reasons = []  # 全部满足：不保留"某条件为假"的误导性说明
    elif not reasons:
        reasons.append("存在未满足的准入条件")
    first = next(iter(receipts), None)
    return CampaignDecision(
        campaign_id=str(first.campaign_id) if first is not None else "",
        generation=int(first.generation) if first is not None else 0,
        candidate_id="",
        promote=promote, conditions=checks, per_seed=per_seed, reasons=reasons,
        created_at=datetime.now(timezone.utc).isoformat())
