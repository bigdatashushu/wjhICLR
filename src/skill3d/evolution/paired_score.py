"""M17 确定性配对评分（§7 统计与复现）：

- paired **BCa** bootstrap 95% CI（`scipy.stats.bootstrap(paired=True, method="BCa")`，
  `n_resamples=9999`；scipy 不可用/数据退化时回退 numpy 百分位实现，同 seed 可复现）；
- `scipy.stats.wilcoxon` 符号秩检验；**退化样本**（零方差/完全相同组）返回 `None`
  并标注"退化 → 判为不显著"（§7 / E-2），不得声称有显著差异；
- 多重比较校正：per-slice p 值按切片数做 **Bonferroni**（§7）；
- 效应量：**Cliff's delta** 与配对 **Cohen's d**（§7）；
- `slice_table` 按题型切片；`slice_no_regression` 判定（容忍度 TODO_CALIBRATE）。
"""

from __future__ import annotations

import math

import numpy as np

try:
    from scipy import stats as _scipy_stats  # 当前环境 scipy 与 numpy 版本不匹配时降级
except Exception:  # pragma: no cover
    _scipy_stats = None

# §7：paired A/B 用 BCa bootstrap + 9999 次重采样
N_BOOTSTRAP = 9999
SLICE_REGRESSION_TOL = 0.01
WILCOXON_P_THRESHOLD = 0.05


def _as_paired(scores_a, scores_b) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1, "paired 评分要求两臂等长一维数组"
    return a, b


def is_degenerate(diffs: np.ndarray) -> bool:
    """退化样本判定（§7 / E-2）：零方差 / 全部差值为 0 / 样本不足。

    退化时配对检验的 p 值没有定义（scipy 返回 nan），一律判"不显著"，
    **不得**声称有显著差异。
    """
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:
        return True
    return bool(np.all(d == d[0]))


def paired_bootstrap_ci(scores_a: np.ndarray, scores_b: np.ndarray,
                        n_resamples: int = N_BOOTSTRAP,
                        seed: int = 0,
                        method: str = "BCa") -> tuple[float, float]:
    """配对 bootstrap 95% CI（对每对差值重采样）。

    首选 `scipy.stats.bootstrap(paired=True, method="BCa")`（§7 口径）；
    scipy 缺失、BCa 在退化数据上不可解、或样本过少时回退百分位 bootstrap
    （同样受 `seed` 控制，保证同 seed 逐位可复现）。
    """
    a, b = _as_paired(scores_a, scores_b)
    n = a.shape[0]
    if n == 0:
        return float("nan"), float("nan")

    if _scipy_stats is not None and not is_degenerate(b - a) and n >= 3:
        try:
            res = _scipy_stats.bootstrap(
                (a, b),
                # scipy paired+vectorized 的签名：两臂各一个位置参数 + axis
                # （scipy>=1.16 实测；写成单参数 x 会 TypeError 并静默退化为百分位法）
                statistic=lambda x, y, axis=-1: np.mean(y - x, axis=axis),
                paired=True, vectorized=True, n_resamples=n_resamples,
                confidence_level=0.95, method=method,
                random_state=np.random.default_rng(seed),
            )
            lo, hi = res.confidence_interval
            if np.isfinite(lo) and np.isfinite(hi):
                return float(lo), float(hi)
        except Exception:  # noqa: BLE001 - BCa 在退化/并列数据上会抛，按 §7 回退
            pass

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (b[idx] - a[idx]).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def wilcoxon_p(scores_a: np.ndarray, scores_b: np.ndarray) -> float | None:
    """Wilcoxon 符号秩检验 p 值。

    **退化样本返回 `None`**（§7 / E-2：零方差/完全相同组 → 判为不显著，
    不得声称有显著差异）；scipy 不可用时降级为纯 numpy 正态近似实现。
    """
    a, b = _as_paired(scores_a, scores_b)
    diff = b - a
    if is_degenerate(diff):
        return None
    if _scipy_stats is not None:
        try:
            p = float(_scipy_stats.wilcoxon(diff).pvalue)
            return None if math.isnan(p) else p
        except ValueError:
            return None
    return _wilcoxon_p_numpy(diff)


def bonferroni(p: float | None, n_comparisons: int) -> float | None:
    """Bonferroni 多重比较校正（§7）：`p_corrected = min(1, p × m)`。

    退化样本（p=None）保持 None；`n_comparisons` ≤ 1 时原样返回。
    """
    if p is None:
        return None
    if n_comparisons <= 1:
        return float(p)
    return float(min(1.0, p * n_comparisons))


def cliffs_delta(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Cliff's delta 效应量（§7）：P(b>a) − P(b<a)，取值 [-1, 1]。

    |delta| 的经验档位：0.147 小 / 0.33 中 / 0.474 大（Romano et al.）。
    """
    a, b = _as_paired(scores_a, scores_b)
    if a.size == 0:
        return float("nan")
    gt = sum(float(np.sum(bi > a)) for bi in b)
    lt = sum(float(np.sum(bi < a)) for bi in b)
    return float((gt - lt) / (a.size * b.size))


def cohens_d_paired(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """配对 Cohen's d 效应量（§7）：mean(diff) / std(diff, ddof=1)。

    差值标准差为 0（退化）时返回 0.0（无效应），不返回 inf。
    """
    a, b = _as_paired(scores_a, scores_b)
    diff = b - a
    if diff.size < 2:
        return 0.0
    sd = float(np.std(diff, ddof=1))
    if sd <= 0:
        return 0.0
    return float(np.mean(diff) / sd)


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
    """按题型切片：{task_type: {n, mean_a, mean_b, delta, p, p_bonferroni}}。

    §7 多重比较：每个切片各做一次配对 Wilcoxon，按**切片数**做 Bonferroni 校正。
    切片退化（零方差/样本不足）时 p 为 None，并在 `degenerate` 标注。
    """
    table: dict[str, dict] = {}
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    keys = sorted(set(task_types))
    for tt in keys:
        idx = [i for i, t in enumerate(task_types) if t == tt]
        ma, mb = float(a[idx].mean()), float(b[idx].mean())
        p = wilcoxon_p(a[idx], b[idx])
        table[tt] = {
            "n": len(idx), "mean_a": ma, "mean_b": mb, "delta": mb - ma,
            "p": p, "p_bonferroni": bonferroni(p, len(keys)),
            "degenerate": is_degenerate(b[idx] - a[idx]),
        }
    return table


def check_slice_no_regression(slice_table: dict,
                              tol: float = SLICE_REGRESSION_TOL) -> bool:
    """slice 无退化判定：每一切片 delta >= -tol（TODO_CALIBRATE）。"""
    return all(v["delta"] >= -tol for v in slice_table.values())


def score_paired(scores_a, scores_b, task_types: list[str],
                 n_resamples: int = N_BOOTSTRAP, seed: int = 0,
                 slice_tol: float = SLICE_REGRESSION_TOL) -> dict:
    """确定性配对评分入口，返回 PairedOutcome 所需统计字段（§7 口径）。

    含：BCa bootstrap CI、Wilcoxon + Bonferroni 校正、Cliff's delta / Cohen's d、
    退化样本标注（`degenerate` → p 记 None，"判为不显著"）。
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    lo, hi = paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed)
    table = build_slice_table(a, b, task_types)
    p = wilcoxon_p(a, b)
    return {
        "n_episodes": int(a.shape[0]),
        "mean_a": float(a.mean()) if a.size else float("nan"),
        "mean_b": float(b.mean()) if b.size else float("nan"),
        "delta": float(b.mean() - a.mean()) if a.size else float("nan"),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "wilcoxon_p": p,
        # §7 多重比较校正：按被测切片数（无切片时按单次比较）
        "wilcoxon_p_bonferroni": bonferroni(p, max(len(table), 1)),
        "n_comparisons": max(len(table), 1),
        "degenerate": is_degenerate(b - a),
        "cliffs_delta": cliffs_delta(a, b),
        "cohens_d": cohens_d_paired(a, b),
        "slice_table": table,
        "slice_no_regression": check_slice_no_regression(table, tol=slice_tol),
        "ci_significant": bool(lo > 0),  # CI 下界 > 0（准入门之一）
    }
