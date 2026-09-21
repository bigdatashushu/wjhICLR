"""v4 §7 尺度能力报告：逐题型 MRA / refusal / coverage + 锚点触发率 / 冲突率。

规格来源（《系统架构4.md》）：

- §7「尺度校准复现」：报告名义 coverage 与 empirical coverage、每题型
  coverage/MRA/refusal、锚点触发率与冲突率；
- §10.2 L3：只有满足预注册门槛的题型才可加入 `allowed_metric_tasks`；
- §4 M6/§16.2：**不得混主表** —— refusal/tool_contract 等可靠性指标单列小节。

本模块只做聚合，不跑推理、不读 GT 之外的东西（GT 只在 M12 评测阶段可见）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from skill3d.schemas.reconstruction import METRIC_TASK_TYPES

# 逐题型最低可报告样本数（低于此只报 n，不报比率 —— 避免 1/1=100% 的假象）
MIN_REPORT_N = 3   # TODO_CALIBRATE


@dataclass
class ScaleTaskReport:
    """单个题型的尺度相关可靠性指标（§7 / §10.2 L3）。"""

    task: str
    n: int = 0
    n_answered: int = 0
    mra: Optional[float] = None                # 仅在作答 episode 上算
    accuracy: Optional[float] = None           # MCA 题型用
    refusal_rate: Optional[float] = None
    coverage: Optional[float] = None           # 产出答案的比例
    eligible_for_metric_tasks: Optional[bool] = None  # §10.2 L3 预注册门槛
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"task": self.task, "n": self.n, "n_answered": self.n_answered,
                "mra": self.mra, "accuracy": self.accuracy,
                "refusal_rate": self.refusal_rate, "coverage": self.coverage,
                "eligible_for_metric_tasks": self.eligible_for_metric_tasks,
                "notes": list(self.notes)}


@dataclass
class ScaleReport:
    """一次 run 的尺度能力报告（不混主表，§4 M6 字段 9）。"""

    n_episodes: int = 0
    per_task: dict[str, ScaleTaskReport] = field(default_factory=dict)
    # 尺度档位分布
    confidence_distribution: dict[str, int] = field(default_factory=dict)
    n_metric_authorized: int = 0
    n_metric_withdrawn: int = 0
    # 锚点统计（HC31）
    anchor_fire_rate: Optional[float] = None
    anchor_accept_rate: Optional[float] = None
    conflict_rate: Optional[float] = None
    mean_anchors_fired: Optional[float] = None
    # 覆盖率证据（HC32/§7）：名义 vs 经验
    calibration_id: Optional[str] = None
    nominal_coverage: Optional[float] = None
    empirical_coverage: Optional[float] = None
    coverage_gap: Optional[float] = None
    scale_ci_rel_median: Optional[float] = None
    scale_ci_abs_m_median: Optional[float] = None
    # 尺度来源分布（诚实性：mock 的合成 GT 尺度一眼可辨，不得被读成真实能力）
    scale_source_distribution: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def metric_task_rows(self) -> list[ScaleTaskReport]:
        return [self.per_task[t] for t in METRIC_TASK_TYPES if t in self.per_task]

    def as_dict(self) -> dict:
        return {
            "n_episodes": self.n_episodes,
            "per_task": {k: v.as_dict() for k, v in sorted(self.per_task.items())},
            "metric_tasks": {r.task: r.as_dict() for r in self.metric_task_rows()},
            "scale": {
                "confidence_distribution": self.confidence_distribution,
                "n_metric_authorized": self.n_metric_authorized,
                "n_metric_withdrawn": self.n_metric_withdrawn,
                "anchor_fire_rate": self.anchor_fire_rate,
                "anchor_accept_rate": self.anchor_accept_rate,
                "conflict_rate": self.conflict_rate,
                "mean_anchors_fired": self.mean_anchors_fired,
                "calibration_id": self.calibration_id,
                "nominal_coverage": self.nominal_coverage,
                "empirical_coverage": self.empirical_coverage,
                "coverage_gap": self.coverage_gap,
                "scale_ci_rel_median": self.scale_ci_rel_median,
                "scale_ci_abs_m_median": self.scale_ci_abs_m_median,
                "scale_source_distribution": self.scale_source_distribution,
            },
            "notes": list(self.notes),
        }

    def format_lines(self) -> list[str]:
        """打印行（"系统可靠性"小节，**不混主表**）。"""
        out = ["尺度能力（不混主表；HC29–33）:"]
        conf = self.confidence_distribution or {}
        out.append(f"  confidence 分布: " +
                   "  ".join(f"{k}={v}" for k, v in sorted(conf.items())) +
                   f"  |  授权米制题型的 episode={self.n_metric_authorized}"
                   f"  收回={self.n_metric_withdrawn}")
        out.append(f"  锚点: 触发率={_fmt(self.anchor_fire_rate)} "
                   f"接受率={_fmt(self.anchor_accept_rate)} "
                   f"冲突率={_fmt(self.conflict_rate)} "
                   f"平均触发数={_fmt(self.mean_anchors_fired)}")
        out.append(f"  校准: id={self.calibration_id} "
                   f"名义覆盖={_fmt(self.nominal_coverage)} "
                   f"经验覆盖={_fmt(self.empirical_coverage)} "
                   f"差值={_fmt(self.coverage_gap)}")
        out.append(f"  尺度 CI: median(rel)={_fmt(self.scale_ci_rel_median)} "
                   f"median(abs, m)={_fmt(self.scale_ci_abs_m_median)}")
        if self.scale_source_distribution:
            out.append("  来源分布: " + "  ".join(
                f"{k}={v}" for k, v in sorted(self.scale_source_distribution.items())))
        for r in self.metric_task_rows():
            out.append(f"  [{r.task}] n={r.n} answered={r.n_answered} "
                       f"MRA={_fmt(r.mra)} refusal={_fmt(r.refusal_rate)} "
                       f"coverage={_fmt(r.coverage)} "
                       f"metric_eligible={r.eligible_for_metric_tasks}")
        for n in self.notes:
            out.append(f"  [note] {n}")
        return out


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "NA"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not np.isfinite(f) else f"{f:.3f}"


def _answered(o) -> bool:
    return (getattr(o, "final_state", "") in ("answer", "answer_best_effort")
            and getattr(o, "answer", None) is not None
            and not getattr(o, "answer_untrusted", False))


def _mra_of(o) -> Optional[float]:
    v = getattr(o, "mra_value", None)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def build_scale_report(outcomes: Sequence,
                       *, baseline_2d: Optional[dict] = None,
                       min_mra_delta: float = 0.0,
                       min_n: int = MIN_REPORT_N) -> ScaleReport:
    """从 `EpisodeOutcome` 序列聚合尺度报告（§7）。

    `baseline_2d`：`{task: mra}`（2D-only baseline）—— 用于判定题型是否达
    §10.2 L3 的加入门槛（`mra >= baseline + min_mra_delta`）。
    """
    rep = ScaleReport(n_episodes=len(outcomes))
    buckets: dict[str, list] = {}
    conf: dict[str, int] = {}
    fired: list[int] = []
    accepted: list[int] = []
    conflicts = 0
    ci_rels: list[float] = []
    ci_abs: list[float] = []
    cal_ids: set[str] = set()
    nominal: list[float] = []
    empirical: list[float] = []
    src_dist: dict[str, int] = {}

    for o in outcomes:
        task = str(getattr(o, "task", "") or "")
        if task:
            buckets.setdefault(task, []).append(o)
        c = str(getattr(o, "scale_confidence", "low") or "low")
        conf[c] = conf.get(c, 0) + 1
        src = str(getattr(o, "scale_source", "") or "") or "unknown"
        src_dist[src] = src_dist.get(src, 0) + 1
        allowed = list(getattr(o, "allowed_metric_tasks", None) or [])
        auth = list(getattr(o, "authorized_metric_tasks", None) or [])
        if auth or allowed:
            rep.n_metric_authorized += 1
        else:
            rep.n_metric_withdrawn += 1
        nf = int(getattr(o, "n_anchors_fired", 0) or 0)
        na = int(getattr(o, "n_anchors_accepted", 0) or 0)
        if nf:
            fired.append(nf)
            accepted.append(na)
        if getattr(o, "scale_conflict", False):
            conflicts += 1
        for src, dst in ((getattr(o, "scale_ci_rel", None), ci_rels),
                         (getattr(o, "scale_ci_abs_m", None), ci_abs)):
            if src is not None:
                try:
                    f = float(src)
                    if np.isfinite(f):
                        dst.append(f)
                except (TypeError, ValueError):
                    pass
        cid = getattr(o, "scale_calibration_id", None)
        if cid:
            cal_ids.add(str(cid))

    # 校准覆盖率证据（来自校准器本身；逐 episode 相同值 → 取首个非空）
    cov = next((getattr(o, "scale_empirical_coverage", None) for o in outcomes
                if getattr(o, "scale_empirical_coverage", None) is not None), None)
    if cov is not None:
        empirical.append(float(cov))
    rep.calibration_id = (sorted(cal_ids)[0] if len(cal_ids) == 1
                          else (f"multi:{len(cal_ids)}" if cal_ids else None))

    for task, items in sorted(buckets.items()):
        tr = ScaleTaskReport(task=task, n=len(items))
        answered = [o for o in items if _answered(o)]
        tr.n_answered = len(answered)
        if tr.n >= min_n:
            tr.coverage = tr.n_answered / tr.n
            tr.refusal_rate = 1.0 - tr.coverage
        elif tr.n:
            tr.notes.append(f"n={tr.n} < {min_n}：只报计数，不报比率")
        mras = [m for m in (_mra_of(o) for o in answered) if m is not None]
        if mras and tr.n >= min_n:
            tr.mra = float(np.mean(mras))
        accs = [bool(o.correct) for o in answered
                if getattr(o, "correct", None) is not None]
        if accs and tr.n >= min_n:
            tr.accuracy = float(np.mean(accs))
        base = (baseline_2d or {}).get(task)
        if task in METRIC_TASK_TYPES and base is not None and tr.mra is not None:
            tr.eligible_for_metric_tasks = bool(tr.mra >= float(base) + min_mra_delta)
            if not tr.eligible_for_metric_tasks:
                tr.notes.append(
                    f"MRA {tr.mra:.3f} 未达 2D-only baseline {float(base):.3f}"
                    "+ min_delta（§10.2 L3：不得加入 allowed_metric_tasks）")
        rep.per_task[task] = tr

    rep.confidence_distribution = conf
    if fired:
        rep.mean_anchors_fired = float(np.mean(fired))
        rep.anchor_fire_rate = float(np.mean([1.0 if n else 0.0 for n in fired]))
        rep.anchor_accept_rate = float(np.mean(
            [a / n if n else 0.0 for a, n in zip(accepted, fired)]))
    if outcomes:
        rep.conflict_rate = conflicts / len(outcomes)
    if ci_rels:
        rep.scale_ci_rel_median = float(np.median(ci_rels))
    if ci_abs:
        rep.scale_ci_abs_m_median = float(np.median(ci_abs))
    if empirical:
        rep.empirical_coverage = float(np.mean(empirical))
        nominal = [float(getattr(o, "scale_nominal_coverage", 0.0) or 0.0)
                   for o in outcomes
                   if getattr(o, "scale_nominal_coverage", None) is not None]
        if nominal:
            rep.nominal_coverage = float(np.mean(nominal))
            rep.coverage_gap = rep.empirical_coverage - rep.nominal_coverage
    if rep.n_metric_withdrawn and not rep.n_metric_authorized:
        rep.notes.append(
            "全部 episode 都收回了米制题型 —— 与当前实况一致（HC30：未标定一律 low）。"
            "这不是缺陷，而是 fail-closed 的正确表现；不得为提高此数而放宽门槛。")
    rep.scale_source_distribution = src_dist
    synthetic = sorted(s for s in src_dist if s.startswith("synthetic") or "mock" in s)
    if synthetic:
        rep.notes.append(
            f"尺度来源含 mock/合成（{synthetic}）→ 本报告只是**管道验证**，"
            "置信档与米制授权来自合成 GT 尺度，绝不可当结果或进主表（HC24/HC34）。")
    return rep


def write_scale_report(report: ScaleReport, out_path: str) -> str:
    """落盘尺度报告 JSON（论文附录 + 审计用）。"""
    import json
    from pathlib import Path

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return str(p)
