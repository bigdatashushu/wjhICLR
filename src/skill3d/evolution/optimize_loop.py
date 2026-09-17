"""P3 离线候选优化 CLI（§13.5）：

```bash
python -m skill3d.evolution.optimize_loop --root-candidate-id <id>
```

把 §6.2 离线演进 FSM 与 §7 准入纪律接起来：

```
CLUSTER_TRACES → GPT6_SYNTHESIZE → LEAKAGE_CHECK
→ OPTIMIZATION_LOOP { SYNTHESIZE → STATIC_CHECK → TEST(L1→L2→L3) → DIAGNOSE → REVISE }
→ PROMOTE / REJECT / QUARANTINE
```

要点：
- 候选不可变（硬约束 11）：修订由 M16 产新版本（parent_version 链），本 CLI 不原地改；
- Inner（L1/L2）可反复迭代，Outer（L3）只跑一次、失败不修订（硬约束 10）；
- paired A/B 复用同一 snapshot（硬约束 18）：real 模式下两臂 `--recon-dir` 指向同一批
  artifact（`reuse_artifact`），mock_light 下两臂用同 seed 的同一合成几何；
- **准入必须 real**（§5.6b）：`--mode mock_light` 只做管道验证，绝不 promote；
- 泄漏检查为硬门（硬约束 13/19）：候选 spec_content 含答案/sample id 即拒；
- GPT-6 不可用（`TODO_USER_INPUT`）→ 候选暂停不 promote，且不得让在线链等待（§8）。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Optional

from skill3d.adapters.episode_source import (
    EpisodeItem,
    EpisodeSourceError,
    load_jsonl_items,
    load_synthetic_items,
    load_vsi_bench_items,
)
from skill3d.evolution.admission import evaluate_admission
from skill3d.evolution.optimization_loop import (
    DEFAULT_BUDGET_LIMIT,
    CandidateArchive,
    TestReport,
    run_optimization_loop,
)
from skill3d.evolution.paired_score import score_paired
from skill3d.governance.gpt6_client import GPT6NotConfiguredError
from skill3d.memory.consolidation import leakage_scan_text
from skill3d.online.config import DEFAULT_CONFIG, load_config, load_yaml, paths_from
from skill3d.online.runner import OnlineRunConfig, run_episode
from skill3d.reconstruction.run import artifact_path
from skill3d.schemas import (
    CandidateRevision,
    CounterexampleBundle,
    PairedOutcome,
    SkillSpec,
)
from skill3d.skills.promote_atomic import promote
from skill3d.trace.store import TraceStore

MODES = ("real", "mock_light")


# ------------------------------------------------------------------ 面板执行 ----

def _score_of(outcome) -> float:
    """单 episode 分数：MCA → 1/0；NA → MRA。"""
    if outcome.is_mca:
        return 1.0 if outcome.correct else 0.0
    return float(outcome.mra_value) if outcome.mra_value is not None else 0.0


def run_panel(items: list[EpisodeItem], cfg: OnlineRunConfig, trace_store=None,
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


def _paired_outcome(pair_id: str, cfg: OnlineRunConfig, outs_a: list, outs_b: list,
                    task_types: list[str], snapshot_ref: str, seed: int) -> PairedOutcome:
    sa = [_score_of(o) for o in outs_a]
    sb = [_score_of(o) for o in outs_b]
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
        slice_table=st["slice_table"],
        resource_cost={"n_episodes": st["n_episodes"]},
        slice_no_regression=st["slice_no_regression"],
        within_budget=True,
    )


def _delta_by_metric(outs_a, outs_b) -> tuple[Optional[float], Optional[float]]:
    """分别给出 MCA 精度差与 NA MRA 差（无该类样本则 None）。"""
    mca = [i for i, o in enumerate(outs_a) if o.is_mca]
    na = [i for i, o in enumerate(outs_a) if not o.is_mca]
    d_mca = (sum(1.0 if outs_b[i].correct else 0.0 for i in mca) / len(mca)
             - sum(1.0 if outs_a[i].correct else 0.0 for i in mca) / len(mca)) if mca else None
    d_mra = (sum(float(outs_b[i].mra_value or 0.0) for i in na) / len(na)
             - sum(float(outs_a[i].mra_value or 0.0) for i in na) / len(na)) if na else None
    return d_mca, d_mra


# ------------------------------------------------------------------ 数据/候选 ----

def _load_items(args, split: str, cfg_yaml: dict) -> list[EpisodeItem]:
    paths = paths_from(cfg_yaml)
    limit = args.limit or None
    if args.panel_source == "synthetic":
        return load_synthetic_items(split, limit=limit, seed=args.seed,
                                    out_dir=str(Path(paths.reconstructions) / "mock_light"))
    if args.panel_source == "jsonl":
        if not args.episodes_jsonl:
            raise EpisodeSourceError("--panel-source jsonl 需要 --episodes-jsonl PATH")
        return load_jsonl_items(args.episodes_jsonl, split=split, limit=limit)
    split_cfg = load_yaml(cfg_yaml.get("split_config", "configs/vsi_bench_split.yaml"))
    return load_vsi_bench_items(split, split_cfg,
                                video_root=args.video_root or paths.raw_videos,
                                cache_dir=paths.vsi_bench_meta, limit=limit,
                                seed=args.seed)


def _load_root_candidate(args) -> CandidateRevision:
    """取根候选：`--candidates` JSONL 中按 root_candidate_id 匹配，或 `--spec-file` 直构。"""
    if args.spec_file:
        spec = SkillSpec.model_validate_json(Path(args.spec_file).read_text(encoding="utf-8"))
        return CandidateRevision(
            revision_id=args.root_candidate_id or "rev-root-0",
            root_candidate_id=args.root_candidate_id or "cand-root-0",
            parent_version=None,
            candidate_type="skill",
            spec_content=spec.model_dump_json(),
            status="draft",
            induction_trace_refs=[],
            evidence_lineage_ref=args.spec_file,
            created_by="human",
            created_at="1970-01-01T00:00:00+00:00",
        )
    p = Path(args.candidates)
    if not p.is_file():
        raise FileNotFoundError(
            f"候选清单不存在: {p}（用 --spec-file 直接指定 SkillSpec，或先由 M16 归纳产出）")
    cands = [CandidateRevision.model_validate_json(l)
             for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    match = [c for c in cands if c.root_candidate_id == args.root_candidate_id]
    if not match:
        raise KeyError(f"候选中未找到 root_candidate_id={args.root_candidate_id}"
                       f"（共 {len(cands)} 条）")
    roots = [c for c in match if c.parent_version is None] or match
    return roots[0]


def _candidate_skill(cand: CandidateRevision) -> SkillSpec:
    return SkillSpec.model_validate(json.loads(cand.spec_content))


# ------------------------------------------------------------------------ CLI ----

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m skill3d.evolution.optimize_loop",
        description="P3 离线候选优化循环（M16/M17/M19）：L1→L2→L3 → 准入 → 原子 promote",
    )
    p.add_argument("--root-candidate-id", default="", help="根候选 id")
    p.add_argument("--candidates", default="data/candidates.jsonl",
                   help="候选清单 JSONL（每行 CandidateRevision）")
    p.add_argument("--spec-file", default="",
                   help="便捷入口：直接给 SkillSpec JSON 构造根候选（管道验证用）")
    p.add_argument("--mode", default="mock_light", choices=list(MODES),
                   help="面板执行模式；准入必须 real（§5.6b）")
    p.add_argument("--panel-source", default="synthetic",
                   choices=["synthetic", "jsonl", "vsi_bench"])
    p.add_argument("--episodes-jsonl", default="", help="panel-source=jsonl 时的清单")
    p.add_argument("--video-root", default="")
    p.add_argument("--l1-limit", type=int, default=4, help="L1 最小切片条数")
    p.add_argument("--limit", type=int, default=0, help="每层面板最多条数（0=全部）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recon-dir", default="", help="real 模式：复用既有 artifact 的目录")
    p.add_argument("--recon-method", default="vggt", choices=["vggt", "dust3r_mast3r", "colmap"])
    p.add_argument("--skill-store", default="data/skill_registry", help="M15 snapshot 存储目录")
    p.add_argument("--active-snapshot", default="", help="active snapshot（只读，用于审计）")
    p.add_argument("--trace-dir", default="", help="离线 trace 目录（默认取 config）")
    p.add_argument("--admission-config", default="configs/admission_thresholds.yaml")
    p.add_argument("--n-cross-scene", type=int, default=0,
                   help="跨 scene 样本数（默认取 admission config 的 N_min_cross_scene）")
    p.add_argument("--max-revisions", type=int, default=0, help="覆盖预算上限（TODO_CALIBRATE）")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_yaml = load_config(args.config)
    paths = paths_from(cfg_yaml)
    adm_cfg = load_yaml(args.admission_config)
    min_delta_mca = float(adm_cfg.get("min_delta_mca", 0.02))
    min_delta_mra = float(adm_cfg.get("min_delta_mra", 0.02))
    n_min = args.n_cross_scene or int(adm_cfg.get("N_min_cross_scene", 3))

    try:
        root = _load_root_candidate(args)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    try:
        cand_skill = _candidate_skill(root)
    except Exception as exc:  # noqa: BLE001 - 候选 spec_content 不是合法 SkillSpec
        print(f"[错误] 候选 spec_content 不是合法 SkillSpec: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 1

    trace_store = TraceStore(args.trace_dir or paths.trace_store)
    store_dir = args.skill_store
    active_snapshot = args.active_snapshot or paths.active_snapshot

    print("=" * 78)
    print(f"P3 优化循环：root_candidate={root.root_candidate_id} "
          f"revision={root.revision_id} skill={cand_skill.skill_id}@{cand_skill.semver}")
    print(f"mode={args.mode} panel_source={args.panel_source} n_min_cross_scene={n_min} "
          f"min_delta_mca={min_delta_mca} min_delta_mra={min_delta_mra}")
    if args.mode != "real":
        print("⚠ mode != real：仅管道验证；准入必须 real（§5.6b）→ 本跑不会 promote")
        print("⚠ mock_light 面板两臂使用同一确定性 stub program：delta 恒为 0，"
              "不具备任何效果判断能力（stub 不是模型，不读 Skill 模板）")
    print("=" * 78)

    # ---- LEAKAGE_CHECK 硬门（§6.2，硬约束 13/19）----
    leaks = leakage_scan_text(root.spec_content)
    no_leakage = not leaks
    if leaks:
        print(f"[拒] LEAKAGE_CHECK 未过: {leaks} → 候选直接 reject（硬约束 13）",
              file=sys.stderr)
        trace_store.append("leakage_check", {"candidate_id": root.root_candidate_id,
                                             "hits": leaks, "passed": False})
        return 1
    trace_store.append("leakage_check", {"candidate_id": root.root_candidate_id,
                                         "hits": [], "passed": True})

    # ---- 面板数据 ----
    try:
        inner = _load_items(args, "inner_validation", cfg_yaml)
        outer = _load_items(args, "outer_holdout", cfg_yaml)
    except EpisodeSourceError as exc:
        print(f"[错误] 面板数据不可用: {exc}", file=sys.stderr)
        return 1
    l1_items = inner[: max(args.l1_limit, 1)]

    base_cfg = OnlineRunConfig(
        mode=args.mode,
        baseline="C1_tools_program",
        seed=args.seed,
        skills=[],                      # arm A：无候选 Skill（baseline）
        recon_dir=args.recon_dir or paths.reconstructions,
        recon_method=args.recon_method,
        trace_dir=args.trace_dir or paths.trace_store,
        active_snapshot_ref=active_snapshot,
    )
    arm_b_cfg = replace(base_cfg, skills=[cand_skill])

    # ---- 逐层 paired A/B（同一批 episode、同一 artifact/合成几何 → 硬约束 18）----
    def paired(items: list[EpisodeItem], level: str) -> tuple[PairedOutcome, list, list]:
        outs_a = run_panel(items, base_cfg, trace_store)
        outs_b = run_panel(items, arm_b_cfg, trace_store)
        po = _paired_outcome(f"pair-{level}-{uuid.uuid4().hex[:8]}", base_cfg,
                             outs_a, outs_b, [it.episode.question_type for it in items],
                             base_cfg.active_snapshot_ref, args.seed)
        d_mca, d_mra = _delta_by_metric(outs_a, outs_b)
        print(f"\n[{level}] n={po.n_episodes} delta={po.delta:+.4f} "
              f"CI95=[{po.ci95_lo:+.4f},{po.ci95_hi:+.4f}] p={po.wilcoxon_p:.4f} "
              f"d_mca={_fmt(d_mca)} d_mra={_fmt(d_mra)} slice_ok={po.slice_no_regression}")
        print(f"        arm_a accuracy/mra: "
              f"{_acc(outs_a)}/{_mra(outs_a)}    arm_b: {_acc(outs_b)}/{_mra(outs_b)}")
        trace_store.append("paired_outcome", po)
        return po, outs_a, outs_b

    def level_report(po: PairedOutcome, min_delta: bool, minimal: bool) -> TestReport:
        if minimal:  # L1：提升 > 0 且无 slice 退化（§6.3）
            ok = po.delta > 0 and po.slice_no_regression
            return TestReport(passed=ok, feedback="" if ok else
                              f"L1 未过: delta={po.delta:+.4f} "
                              f"slice_ok={po.slice_no_regression}")
        ok = po.ci95_lo > 0 and po.delta >= (min_delta_mca if min_delta else 0.0) \
            and po.slice_no_regression
        return TestReport(passed=ok, feedback="" if ok else
                          f"未过: ci95_lo={po.ci95_lo:+.4f} delta={po.delta:+.4f} "
                          f"slice_ok={po.slice_no_regression}")

    level_state: dict[str, tuple[PairedOutcome, list, list]] = {}

    def run_test_fn(revision: CandidateRevision, level: str) -> TestReport:
        if level == "L1_minimal_slice":
            po, a, b = paired(l1_items, "L1")
            level_state["L1"] = (po, a, b)
            return level_report(po, min_delta=False, minimal=True)
        if level == "L2_full_inner":
            po, a, b = paired(inner, "L2")
            level_state["L2"] = (po, a, b)
            return level_report(po, min_delta=True, minimal=False)
        if level == "L3_outer_holdout":
            po, a, b = paired(outer, "L3")
            level_state["L3"] = (po, a, b)
            return level_report(po, min_delta=True, minimal=False)
        raise ValueError(f"未知层级: {level}")

    def static_check_fn(revision: CandidateRevision) -> bool:
        try:
            _candidate_skill(revision)
            return True
        except Exception:  # noqa: BLE001
            return False

    def revise_fn(revision: CandidateRevision, feedback: str) -> CandidateRevision:
        """REVISE：M16 GPT-6 基于反例产新版本（绝不原地改，硬约束 11）。"""
        from skill3d.governance.gpt6_client import GPT6Client
        from skill3d.governance.revise_patch import gpt6_revise

        bundle = CounterexampleBundle(
            bundle_id=f"bundle-{revision.revision_id}",
            source_revision_id=revision.revision_id,
            failed_episode_refs=[],
            minimal_counterexamples=[],
            metamorphic_transforms=[],
            regression_set_ref="",
            gpt6_visible_summary=f"测试反馈（不含答案）: {feedback}",
            generated_at="1970-01-01T00:00:00+00:00",
        )
        return gpt6_revise(revision, bundle, GPT6Client())

    archive = CandidateArchive()
    limit = DEFAULT_BUDGET_LIMIT
    if args.max_revisions:
        limit = limit.model_copy(update={"max_revisions": args.max_revisions})

    print("\n进入 OPTIMIZATION_LOOP（SYNTHESIZE→STATIC_CHECK→TEST(L1→L2→L3)→DIAGNOSE→REVISE）")
    try:
        run = run_optimization_loop(root, run_test_fn, revise_fn,
                                    static_check_fn=static_check_fn,
                                    archive=archive, budget_limit=limit)
    except GPT6NotConfiguredError as exc:
        print(f"\n[paused] GPT-6 不可用（TODO_USER_INPUT）: {exc}\n"
              "        按 §8 策略：候选暂停/进 shadow，不 promote；在线链不等待 GPT-6。",
              file=sys.stderr)
        trace_store.append("optimization_run", {
            "root_candidate_id": root.root_candidate_id, "status": "paused",
            "termination_reason": "gpt6_not_configured"})
        return 3

    print(f"\nOptimizationRun status={run.status} reason={run.termination_reason} "
          f"revisions={run.budget_used.revisions} rollouts={run.budget_used.rollouts}")
    trace_store.append("optimization_run", run.model_dump())

    # ---- 准入（硬约束 13：硬门一票否决；§5.6b：准入必须 real）----
    l3 = level_state.get("L3")
    if run.status != "promoted" or l3 is None:
        print(f"[结束] 未进入准入：status={run.status}（候选留档防重复，§7）")
        return 0

    po, outs_a, outs_b = l3
    n_cross_scene = len({it.episode.scene_name for it in outer})
    decision = evaluate_admission(
        candidate_id=root.root_candidate_id,
        paired_outcomes=[po],
        counterexamples=[],
        no_leakage=no_leakage,
        n_cross_scene=n_cross_scene,
        within_budget=po.within_budget,
        min_delta_required=min_delta_mca if po.metric == "accuracy" else min_delta_mra,
        n_min=n_min,
    )
    trace_store.append("admission_decision", decision.model_dump())
    print(f"\n准入: promotes={decision.promotes} reason={decision.reason}")

    if not decision.promotes:
        print("[结束] 硬门未过 → 不 promote（GPT-6 建议亦不可覆盖，硬约束 13）")
        return 0

    if args.mode != "real":
        print("[暂停] L3 通过但 mode != real：准入必须 real（§5.6b）→ 不 promote。"
              "请用 --mode real 复跑（需真实面板与共享 artifact）。", file=sys.stderr)
        return 3

    promoted = root.model_copy(update={
        "status": "promoted",
        "revision_id": root.revision_id,
    })
    promotion_log: list = []
    snap = promote(store_dir, promoted, promotion_log=promotion_log)
    print(f"[promote] 原子切换完成：snapshot_before={promotion_log[-1]['snapshot_before']} "
          f"→ snapshot_after={snap['snapshot_id']}（旧 snapshot 保留，可回滚，硬约束 12）")
    trace_store.append("promotion", promotion_log[-1] if promotion_log else snap)
    return 0


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:+.4f}"


def _acc(outs) -> str:
    mca = [o for o in outs if o.is_mca]
    return "n/a" if not mca else f"{sum(1 for o in mca if o.correct) / len(mca):.3f}"


def _mra(outs) -> str:
    vals = [o.mra_value for o in outs if o.mra_value is not None]
    return "n/a" if not vals else f"{sum(vals) / len(vals):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
