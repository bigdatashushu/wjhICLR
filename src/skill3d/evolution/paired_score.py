"""M17 确定性配对评分：

- paired bootstrap 95% CI（1000 次重采样，numpy 实现，固定 seed 可复现）；
- scipy.stats.wilcoxon 符号秩检验；
- slice_table 按题型切片；slice_no_regression 判定（容忍度 TODO_CALIBRATE）。
"""

from __future__ import annotations

import math

import numpy as np

try:
    from scipy import stats as _scipy_stats  # 当前环境 scipy 与 numpy 版本不匹配时降级
except Exception:  # pragma: no cover
    _scipy_stats = None

# TODO_CALIBRATE：起始参考值
N_BOOTSTRAP = 1000
SLICE_REGRESSION_TOL = 0.01
WILCOXON_P_THRESHOLD = 0.05


def paired_bootstrap_ci(scores_a: np.ndarray, scores_b: np.ndarray,
                        n_resamples: int = N_BOOTSTRAP,
                        seed: int = 0) -> tuple[float, float]:
    """配对 bootstrap 95% CI（对每对差值重采样）。"""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1, "paired 评分要求两臂等长一维数组"
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (b[idx] - a[idx]).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def wilcoxon_p(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Wilcoxon 符号秩检验 p 值；全零差或样本不足时返回 1.0（不显著）。

    优先用 scipy.stats.wilcoxon；scipy 不可用时降级为纯 numpy 正态近似实现。
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diff = b - a
    if np.all(diff == 0) or diff.shape[0] < 2:
        return 1.0
    if _scipy_stats is not None:
        try:
            return float(_scipy_stats.wilcoxon(diff).pvalue)
        except ValueError:
            return 1.0
    return _wilcoxon_p_numpy(diff)


def _wilcoxon_p_numpy(diff: np.ndarray) -> float:
    """Wilcoxon 符号秩双侧 p 值（正态近似 + 连续性校正，处理零差与并列秩）。"""
    d = diff[diff != 0]
    n = d.shape[0]
    if n < 2:
        return 1.0
    ranks = _rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if var <= 0:
        return 1.0
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(var)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return float(max(0.0, min(1.0, p)))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩（并列取均值），不依赖 scipy。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # 并列组取平均秩
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def build_slice_table(scores_a: np.ndarray, scores_b: np.ndarray,
                      task_types: list[str]) -> dict:
    """按题型切片：{task_type: {n, mean_a, mean_b, delta}}。"""
    table: dict[str, dict] = {}
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    for tt in sorted(set(task_types)):
        idx = [i for i, t in enumerate(task_types) if t == tt]
        ma, mb = float(a[idx].mean()), float(b[idx].mean())
        table[tt] = {"n": len(idx), "mean_a": ma, "mean_b": mb, "delta": mb - ma}
    return table


def check_slice_no_regression(slice_table: dict,
                              tol: float = SLICE_REGRESSION_TOL) -> bool:
    """slice 无退化判定：每一切片 delta >= -tol（TODO_CALIBRATE）。"""
    return all(v["delta"] >= -tol for v in slice_table.values())


def score_paired(scores_a, scores_b, task_types: list[str],
                 n_resamples: int = N_BOOTSTRAP, seed: int = 0,
                 slice_tol: float = SLICE_REGRESSION_TOL) -> dict:
    """确定性配对评分入口，返回 PairedOutcome 所需统计字段。"""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    lo, hi = paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed)
    table = build_slice_table(a, b, task_types)
    return {
        "n_episodes": int(a.shape[0]),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": float(b.mean() - a.mean()),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "wilcoxon_p": wilcoxon_p(a, b),
        "slice_table": table,
        "slice_no_regression": check_slice_no_regression(table, tol=slice_tol),
        "ci_significant": lo > 0,  # CI 下界 > 0（准入门之一）
    }
