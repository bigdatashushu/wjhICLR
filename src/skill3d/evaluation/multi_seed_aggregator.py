"""G-66 多 seed 聚合与统计检验（§16.3、§16 统计检验表）。

论文规范（§16.3）：
- 所有主表数字 **≥3 seed**，报 mean ± std；
- paired A/B 报 95% BCa bootstrap CI + Wilcoxon（见 `evolution/paired_score.py`）；
- 多 seed 均值比较：双样本 t-test 或 Mann-Whitney U；
- 多方法整体比较：Friedman test + Nemenyi post-hoc；
- 相关性（重建质量 vs MCA/MRA）：Spearman ρ；
- 显著性水平 α=0.05，多比较时 **Bonferroni 校正**；报效应量（Cliff's delta / Cohen's d）。

实现策略：scipy 可用时用 scipy；不可用/版本异常时退化为 numpy 的确定性实现
（t 分布近似、Mann-Whitney 正态近似、Spearman 秩相关、Friedman 卡方近似），
保证在无 scipy 环境下仍能出数且方法在论文中可声明。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

# 论文规范：主表最少 seed 数（§16.3；不足时明确警告，不静默降级）
MIN_SEEDS_FOR_MAIN_TABLE = 3
# 显著性水平（§16.3）
ALPHA = 0.05


def _scipy_stats():
    try:
        from scipy import stats as _st

        return _st
    except Exception:  # noqa: BLE001 - 环境缺 scipy 时走 numpy 退化实现
        return None


# ------------------------------------------------------------------ 聚合 ----

@dataclass
class SeedAggregate:
    """单指标的多 seed 聚合（mean ± std + 95% CI）。"""

    metric: str
    values: list[float]
    mean: float = 0.0
    std: float = 0.0
    ci95_lo: float = 0.0
    ci95_hi: float = 0.0
    n_seeds: int = 0

    def __post_init__(self) -> None:
        """直接构造（只给 values）时自动补统计量，避免"有值但 mean=0"的陷阱。"""
        if not self.values:
            return
        arr = np.asarray([v for v in self.values if v is not None and np.isfinite(v)],
                         dtype=float)
        if arr.size == 0:
            return
        self.mean = float(arr.mean())
        self.std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        self.ci95_lo, self.ci95_hi = _normal_ci(arr)
        self.n_seeds = int(arr.size)

    def as_mean_std(self, digits: int = 4) -> str:
        return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"


def _normal_ci(values: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """均值 95% CI：n≥2 用 t 近似（1.96 保守替代），n=1 退化为点估计。"""
    if values.size < 2:
        return float(values.mean()), float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(values.size))
    return float(values.mean() - 1.96 * se), float(values.mean() + 1.96 * se)


def aggregate_metric(metric: str, values: Sequence[float],
                     alpha: float = ALPHA) -> SeedAggregate:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return SeedAggregate(metric=metric, values=[])
    lo, hi = _normal_ci(arr, alpha)
    return SeedAggregate(metric=metric, values=[float(v) for v in arr],
                         mean=float(arr.mean()),
                         std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                         ci95_lo=lo, ci95_hi=hi, n_seeds=int(arr.size))


def aggregate_runs(runs: Sequence, alpha: float = ALPHA) -> dict[str, SeedAggregate]:
    """把同一配置的多个 seed 的 `EvaluationRun` 聚合为 {metric: SeedAggregate}。

    `runs` 为 `schemas.trace.EvaluationRun` 序列（不同 seed、同 split/配置）。
    """
    if not runs:
        return {}
    mca = [r.accuracy for r in runs if getattr(r, "accuracy", None) is not None]
    mra = [r.mra for r in runs if getattr(r, "mra", None) is not None]
    n_ep = [float(r.n_episodes) for r in runs if getattr(r, "n_episodes", None)]
    out: dict[str, SeedAggregate] = {}
    if mca:
        out["accuracy"] = aggregate_metric("accuracy", mca, alpha)
    if mra:
        out["mra"] = aggregate_metric("mra", mra, alpha)
    if n_ep:
        out["n_episodes"] = aggregate_metric("n_episodes", n_ep, alpha)
    # 8 任务分别准确率（per_task 里的 accuracy/mra）
    tasks = sorted({t for r in runs for t in (r.per_task or {})})
    for t in tasks:
        accs = [(r.per_task.get(t) or {}).get("accuracy") for r in runs]
        accs = [a for a in accs if a is not None]
        if accs:
            out[f"task:{t}:accuracy"] = aggregate_metric(f"task:{t}:accuracy", accs, alpha)
        mras = [(r.per_task.get(t) or {}).get("mra") for r in runs]
        mras = [m for m in mras if m is not None]
        if mras:
            out[f"task:{t}:mra"] = aggregate_metric(f"task:{t}:mra", mras, alpha)
    return out


def check_seed_count(n_seeds: int, min_seeds: int = MIN_SEEDS_FOR_MAIN_TABLE) -> Optional[str]:
    """seed 数不足时返回警告文本（论文主表必须 ≥3，§16.3）。"""
    if n_seeds < min_seeds:
        return (f"仅 {n_seeds} 个 seed < {min_seeds}：不满足论文主表要求（§16.3），"
                "结论不得进主表")
    return None


# ------------------------------------------------------------------ 效应量 ----

def cohens_d(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Cohen's d（合并标准差；n<2 或缺样本返回 None）。"""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.size < 2 or y.size < 2:
        return None
    nx, ny = x.size, y.size
    pooled = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1))
                       / max(nx + ny - 2, 1))
    if pooled <= 0:
        return None
    return float((x.mean() - y.mean()) / pooled)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Cliff's delta ∈ [-1,1]（非参数效应量；O(n·m) 实现）。"""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.size == 0 or y.size == 0:
        return None
    gt = sum(1 for xi in x for yi in y if xi > yi)
    lt = sum(1 for xi in x for yi in y if xi < yi)
    return float((gt - lt) / (x.size * y.size))


# ------------------------------------------------------------------ 两样本 ----

def two_sample_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """多 seed 均值比较：t-test + Mann-Whitney U + 效应量（§16 检验表）。"""
    x, y = np.asarray(a, float), np.asarray(b, float)
    out: dict = {"n_a": int(x.size), "n_b": int(y.size),
                 "mean_a": float(x.mean()) if x.size else None,
                 "mean_b": float(y.mean()) if y.size else None,
                 "cohens_d": cohens_d(x, y), "cliffs_delta": cliffs_delta(x, y),
                 "t_p": None, "mannwhitney_p": None, "method": "numpy"}
    if x.size == 0 or y.size == 0:
        return out
    st = _scipy_stats()
    if st is not None:
        try:
            tp = float(st.ttest_ind(x, y, equal_var=False).pvalue)
            mp = float(st.mannwhitneyu(x, y, alternative="two-sided").pvalue)
            # 退化情形（零方差/完全相同样本）scipy 可能返回 nan → 归一化为 None，
            # 避免 nan 混进论文表格（nan 既不"显著"也不"不显著"）
            out["t_p"] = tp if math.isfinite(tp) else None
            out["mannwhitney_p"] = mp if math.isfinite(mp) else None
            out["method"] = "scipy"
            if out["t_p"] is None and out["mannwhitney_p"] is None:
                out["note"] = "退化样本（零方差/完全相同）→ p 值不可用，判为不显著"
            return out
        except Exception:  # noqa: BLE001 - 版本不匹配等 → 退化
            out["method"] = "numpy(fallback)"
    out["t_p"] = _welch_t_p(x, y)
    out["mannwhitney_p"] = _mannwhitney_p_normal(x, y)
    return out


def _welch_t_p(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Welch t 检验的 p 值（正态近似，双尾）；方差为 0 时返回 None。"""
    if x.size < 2 or y.size < 2:
        return None
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se2 = vx / x.size + vy / y.size
    if se2 <= 0:
        return None
    t = (x.mean() - y.mean()) / math.sqrt(se2)
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2)))))


def _mannwhitney_p_normal(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Mann-Whitney U 的正态近似 p 值（含结校正）；无样本返回 None。"""
    if x.size == 0 or y.size == 0:
        return None
    allv = np.concatenate([x, y])
    ranks = _rankdata(allv)
    rx = ranks[: x.size].sum()
    u = rx - x.size * (x.size + 1) / 2.0
    mu = x.size * y.size / 2.0
    n = allv.size
    _, counts = np.unique(allv, return_counts=True)
    tie = float(sum(c ** 3 - c for c in counts))
    sigma2 = x.size * y.size / 12.0 * ((n + 1) - tie / (n * (n - 1))) if n > 1 else 0.0
    if sigma2 <= 0:
        return None
    z = (u - mu) / math.sqrt(sigma2)
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2)))))


def _rankdata(values: np.ndarray) -> np.ndarray:
    """平均秩（与 scipy.stats.rankdata 口径一致）。"""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_v = values[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i: j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


# ------------------------------------------------------------------ 多方法 ----

def friedman_test(groups: Sequence[Sequence[float]]) -> dict:
    """Friedman 检验（多方法整体比较，§16 检验表）。

    `groups[i]` 为第 i 个方法在各 seed/block 上的取值；要求等长。
    """
    arr = [np.asarray(g, dtype=float) for g in groups]
    if len(arr) < 2 or any(a.size != arr[0].size for a in arr) or arr[0].size < 2:
        return {"statistic": None, "p": None, "n_blocks": 0, "n_methods": len(arr),
                "note": "需要 ≥2 个方法且各 ≥2 个 block（等长）"}
    st = _scipy_stats()
    if st is not None:
        try:
            stat, p = st.friedmanchisquare(*arr)
            return {"statistic": float(stat), "p": float(p),
                    "n_blocks": int(arr[0].size), "n_methods": len(arr),
                    "method": "scipy"}
        except Exception:  # noqa: BLE001
            pass
    n, k = arr[0].size, len(arr)
    ranks = np.stack([_rankdata(np.array([a[i] for a in arr])) for i in range(n)])
    rj = ranks.sum(axis=0)
    stat = 12.0 / (n * k * (k + 1)) * float((rj ** 2).sum()) - 3 * n * (k + 1)
    p = _chi2_sf(max(stat, 0.0), k - 1)
    return {"statistic": float(stat), "p": p, "n_blocks": n, "n_methods": k,
            "method": "numpy(fallback)"}


def _chi2_sf(x: float, df: int) -> Optional[float]:
    """卡方分布上尾概率（Wilson–Hilferty 近似；df<1 返回 None）。"""
    if df < 1:
        return None
    z = ((x / df) ** (1.0 / 3.0) - (1 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
    return float(1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2))))


def nemenyi_posthoc(groups: Sequence[Sequence[float]], alpha: float = ALPHA) -> list[dict]:
    """Nemenyi post-hoc（Friedman 后的两两比较，临界差 CD）。

    CD = q_α · sqrt(k(k+1)/(6N))；q_α 用 α=0.05 的常用查表值近似（TODO_CALIBRATE）。
    """
    res = friedman_test(groups)
    n, k = res["n_blocks"], res["n_methods"]
    if not n or k < 2:
        return []
    arr = [np.asarray(g, dtype=float) for g in groups]
    ranks = np.stack([_rankdata(np.array([a[i] for a in arr])) for i in range(n)])
    mean_ranks = ranks.mean(axis=0)
    q_alpha = _q_alpha_005(k)
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))
    out = []
    for i, j in combinations(range(k), 2):
        diff = abs(float(mean_ranks[i] - mean_ranks[j]))
        out.append({"i": i, "j": j, "rank_diff": diff, "cd": cd,
                    "significant": bool(diff > cd)})
    return out


def _q_alpha_005(k: int) -> float:
    """Nemenyi 临界值 q_0.05（k 个方法）；k>10 用线性外推（TODO_CALIBRATE）。"""
    table = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
             8: 3.031, 9: 3.102, 10: 3.164}
    if k in table:
        return table[k]
    return table[10] + 0.06 * (k - 10)


def spearman_correlation(x: Sequence[float], y: Sequence[float]) -> dict:
    """Spearman ρ（重建质量 vs MCA/MRA 的相关性，§16 检验表）。"""
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.size != b.size or a.size < 3:
        return {"rho": None, "p": None, "n": int(a.size),
                "note": "需要等长且 ≥3 对样本"}
    st = _scipy_stats()
    if st is not None:
        try:
            rho, p = st.spearmanr(a, b)
            return {"rho": float(rho), "p": float(p), "n": int(a.size), "method": "scipy"}
        except Exception:  # noqa: BLE001
            pass
    ra, rb = _rankdata(a), _rankdata(b)
    ra_c, rb_c = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float((ra_c ** 2).sum()) * float((rb_c ** 2).sum()))
    if denom <= 0:
        return {"rho": None, "p": None, "n": int(a.size)}
    rho = float((ra_c * rb_c).sum() / denom)
    t = rho * math.sqrt(max(a.size - 2, 1) / max(1e-12, 1 - rho ** 2))
    p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2)))))
    return {"rho": rho, "p": p, "n": int(a.size), "method": "numpy(fallback)"}


def bonferroni(pvalues: Sequence[float], alpha: float = ALPHA) -> list[bool]:
    """Bonferroni 校正（多比较，§16.3）：返回每个假设是否在 α 下显著。"""
    ps = [p for p in pvalues if p is not None]
    m = len(ps)
    if m == 0:
        return []
    return [bool(p <= alpha / m) for p in ps]


# ------------------------------------------------------------------ 报告 ----

def format_paper_table(aggregates: dict[str, SeedAggregate], *,
                       include_tasks: bool = True) -> str:
    """生成 mean ± std 表格文本（含 seed 数不足警告，§16.3）。"""
    lines: list[str] = []
    main = {k: v for k, v in aggregates.items() if not k.startswith("task:")}
    warn = check_seed_count(max((v.n_seeds for v in main.values()), default=0))
    lines.append(f"{'metric':32s}{'mean ± std':>22s}{'95% CI':>26s}{'n_seeds':>9s}")
    for key in sorted(main):
        v = main[key]
        ci = f"[{v.ci95_lo:.4f}, {v.ci95_hi:.4f}]"
        lines.append(f"{key:32s}{v.as_mean_std():>22s}{ci:>26s}{v.n_seeds:>9d}")
    if include_tasks:
        for key in sorted(k for k in aggregates if k.startswith("task:")):
            v = aggregates[key]
            lines.append(f"{key:32s}{v.as_mean_std():>22s}"
                         f"{'':>26s}{v.n_seeds:>9d}")
    if warn:
        lines.append(f"⚠ {warn}")
    return "\n".join(lines)


def aggregate_to_json(aggregates: dict[str, SeedAggregate]) -> dict:
    """聚合结果落盘用 JSON（RunManifest 引用；CSV/JSONL 足够支撑论文，§13 G-06）。"""
    return {k: asdict(v) for k, v in aggregates.items()}


@dataclass
class MultiSeedReport:
    """多 seed 汇总报告（主表 + 检验结果）。"""

    split: str
    n_seeds: int
    aggregates: dict = field(default_factory=dict)
    seed_warning: Optional[str] = None
    comparisons: dict = field(default_factory=dict)

    def summary(self) -> str:
        s = f"split={self.split} n_seeds={self.n_seeds}"
        if self.seed_warning:
            s += f"\n⚠ {self.seed_warning}"
        return s


# ----------------------------------------------- §8.3 官方 8 任务主表口径 ----

def _slot_score(slot: dict) -> Optional[float]:
    """单个任务/档位的分数：MCA → accuracy；NA → mra；两者都有时按 n 取多数口径。"""
    acc = slot.get("accuracy")
    mra = slot.get("mra")
    n_mca = int(slot.get("n_mca", 0) or 0)
    n_na = int(slot.get("n_na", 0) or 0)
    if n_mca and acc is not None:
        return float(acc)
    if n_na and mra is not None:
        return float(mra)
    if acc is not None:
        return float(acc)
    return float(mra) if mra is not None else None


def macro_average_over_tasks(per_task: dict) -> dict:
    """§8.3 主表口径：4 NA(MRA) + 4 MCA(Acc) 共 8 任务**简单算术平均 × 100**，无加权。

    - `object_rel_direction` 的 easy/medium/hard 三档**先等权聚合**为该任务分
      （`per_task[task]["levels"]`），同时保留三档明细；
    - 缺失任务（该 run 没跑到）不参与平均，但会记进 `missing_tasks`（不臆造 0）；
    - 返回 `{"avg_x100", "per_task_x100": {...}, "scale": 100, "missing_tasks": [...]}`。
    """
    from skill3d.routing.task_classifier import TASK_TYPES

    per_task_x100: dict[str, float] = {}
    level_detail: dict[str, dict] = {}
    missing: list[str] = []
    for task in TASK_TYPES:
        slot = (per_task or {}).get(task)
        if not slot:
            missing.append(task)
            continue
        levels = {k: v for k, v in (slot.get("levels") or {}).items() if v}
        if len(levels) > 1:
            vals = [x for x in (_slot_score(v) for v in levels.values()) if x is not None]
            if not vals:
                missing.append(task)
                continue
            score = float(np.mean(vals))          # 三档等权（§8.3）
            level_detail[task] = {k: (None if _slot_score(v) is None
                                      else round(_slot_score(v) * 100, 4))
                                  for k, v in levels.items()}
        else:
            score = _slot_score(slot)
            if score is None:
                missing.append(task)
                continue
        per_task_x100[task] = round(score * 100, 4)
    # per_task_x100 已是 ×100 的量纲 → 直接取平均（不要再次 ×100）
    avg = round(float(np.mean(list(per_task_x100.values()))), 4) if per_task_x100 else None
    return {
        "avg_x100": avg,
        "per_task_x100": per_task_x100,
        "level_detail_x100": level_detail,
        "n_tasks": len(per_task_x100),
        "missing_tasks": missing,
        "scale": 100,
        "note": "§8.3：4 NA(MRA) + 4 MCA(Acc) 简单算术平均 ×100，无加权；"
                "rel_direction 三档先等权聚合",
    }


def format_main_table_row(name: str, per_task: dict) -> str:
    """按官方 Table 10 版式打印一行：`name | Avg | 4×NA(MRA) | 4×MCA(Acc)`（全 ×100）。"""
    m = macro_average_over_tasks(per_task)
    na = [("object_counting", "object_abs_distance", "object_size_estimation",
           "room_size_estimation")]
    mca = [("object_rel_distance", "object_rel_direction", "route_planning",
            "obj_appearance_order")]
    def _get(t: str) -> str:
        v = m["per_task_x100"].get(t)
        return "n/a" if v is None else f"{v:.2f}"
    cells = [f"{m['avg_x100']:.2f}" if m["avg_x100"] is not None else "n/a"]
    cells += [_get(t) for t in na[0]] + [_get(t) for t in mca[0]]
    return " | ".join([name] + cells)
