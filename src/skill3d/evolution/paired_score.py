"""M17 确定性配对评分（§7 统计与复现 + v6 §18.4 paired A/B 口径）：

- paired **BCa** bootstrap 95% CI（`scipy.stats.bootstrap(paired=True, method="BCa")`，
  `n_resamples=9999`；scipy 不可用/数据退化时回退 numpy 百分位实现，同 seed 可复现）；
  实际用的方法由 `paired_bootstrap_ci_detailed` 显式回报（不把百分位法说成 BCa）；
- **McNemar 配对检验**（v6 §18.4：逐题二值结果的首选检验；`mcnemar_test`）——
  小样本用精确二项检验，大样本用带连续性校正的卡方近似；
- `scipy.stats.wilcoxon` 符号秩检验；**退化样本**（零方差/完全相同组）返回 `None`
  并标注"退化 → 判为不显著"（§7 / E-2），不得声称有显著差异；
- 多重比较校正：per-slice p 值按切片数做 **Bonferroni**（§7 / §18.4）；
- 效应量：**Cliff's delta** 与配对 **Cohen's d**（§7 / §18.4）；
- `slice_table` 按题型切片；`slice_no_regression` 判定（容忍度 TODO_CALIBRATE）。

v6 §18.4 的完整实验协议（三级隔离 / 样本量三档 / 噪声底 / evidence-state 报告 /
paper_eligible / 红线 9）在 `skill3d/evaluation/experiment_protocol.py`，本模块只提供
**统计内核**，不携带任何 split 语义。
"""

from __future__ import annotations

import math

import numpy as np

try:
    from scipy import stats as _scipy_stats  # 当前环境 scipy 与 numpy 版本不匹配时降级
except Exception:  # pragma: no cover
    _scipy_stats = None

# §7/§18.4：paired A/B 用 BCa bootstrap + 9999 次重采样
N_BOOTSTRAP = 9999
SLICE_REGRESSION_TOL = 0.01
WILCOXON_P_THRESHOLD = 0.05
# §18.4：McNemar 判别对（b+c）少于此数用精确二项检验，否则用卡方近似
# TODO_CALIBRATE：25 是文献常用的精确/渐近分界，非本系统标定值。
MCNEMAR_EXACT_MAX_DISCOUNT = 25
# §18.4：退化样本的固定注记（不得改写为"不显著但已检验"这类含混表述）
DEGENERATE_NOTE = "退化样本 → 判为不显著"


def normalize_p(p) -> float | None:
    """p 值归一化：`None`/`nan`/`inf` 一律变 `None`（§18.4 退化样本纪律）。

    绝不让 nan 混进论文表格 —— nan 既不"显著"也不"不显著"，它的正确含义是
    "这个检验在退化样本上没有定义"。
    """
    if p is None:
        return None
    try:
        v = float(p)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


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


def paired_bootstrap_ci_detailed(scores_a: np.ndarray, scores_b: np.ndarray,
                                 n_resamples: int = N_BOOTSTRAP,
                                 seed: int = 0,
                                 method: str = "BCa") -> dict:
    """配对 bootstrap 95% CI，并**如实回报实际用的方法**（§18.4）。

    首选 `scipy.stats.bootstrap(paired=True, method="BCa")`；scipy 缺失、BCa 在退化
    数据上不可解、或样本过少时回退百分位 bootstrap（同样受 `seed` 控制，同 seed 逐位
    可复现）。回退时 `method` 字段写明，避免把百分位法当成 BCa 写进论文。
    """
    a, b = _as_paired(scores_a, scores_b)
    n = a.shape[0]
    if n == 0:
        return {"lo": float("nan"), "hi": float("nan"), "n_pairs": 0,
                "method": "none(empty)", "degenerate": True,
                "note": "无配对样本 → CI 未定义"}
    degenerate = is_degenerate(b - a)
    if _scipy_stats is not None and not degenerate and n >= 3:
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
                return {"lo": float(lo), "hi": float(hi), "n_pairs": int(n),
                        "method": f"scipy.bootstrap(paired, {method})",
                        "degenerate": False, "note": ""}
        except Exception:  # noqa: BLE001 - BCa 在退化/并列数据上会抛，按 §7 回退
            pass

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = (b[idx] - a[idx]).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    note = ""
    if degenerate:
        note = (f"{DEGENERATE_NOTE}：差值零方差 → bootstrap 重采样无信息，"
                f"CI 即点估计，不得据此声称显著（§18.4）")
    return {"lo": float(lo), "hi": float(hi), "n_pairs": int(n),
            "method": f"percentile(fallback{'/degenerate' if degenerate else ''})",
            "degenerate": bool(degenerate), "note": note}


def paired_bootstrap_ci(scores_a: np.ndarray, scores_b: np.ndarray,
                        n_resamples: int = N_BOOTSTRAP,
                        seed: int = 0,
                        method: str = "BCa") -> tuple[float, float]:
    """配对 bootstrap 95% CI 的 `(lo, hi)`（细节见 `paired_bootstrap_ci_detailed`）。"""
    d = paired_bootstrap_ci_detailed(scores_a, scores_b, n_resamples=n_resamples,
                                     seed=seed, method=method)
    return d["lo"], d["hi"]


def _binary_arrays(control, treatment) -> tuple[np.ndarray, np.ndarray, int]:
    """两臂配对二值结果 → (control, treatment, 丢弃的配对数)。

    `None`（或 `nan`）成对丢弃（并计数，报告里如实说明丢弃了多少对）；非二值取值
    raise（把 MRA 之类的连续值当二值喂进来是调用方 bug，不得静默取整）。
    """
    a = np.asarray(list(control), dtype=object).ravel()
    b = np.asarray(list(treatment), dtype=object).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired 要求两臂等长：{a.shape[0]} vs {b.shape[0]}")
    keep = [(x, y) for x, y in zip(a, b)
            if not _is_missing(x) and not _is_missing(y)]
    dropped = int(a.shape[0] - len(keep))
    ax = np.array([_to_bit(x) for x, _ in keep], dtype=float)
    bx = np.array([_to_bit(y) for _, y in keep], dtype=float)
    return ax, bx, dropped


def _is_missing(value) -> bool:
    """缺结果判定（`None` / `nan` / `pandas.NA` 之类的 NaN 语义值）。"""
    if value is None:
        return True
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _to_bit(value) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    f = float(value)
    if f not in (0.0, 1.0):
        raise ValueError(f"McNemar/配对二值检验只接受 0/1/bool，收到 {value!r}")
    return f


def mcnemar_test(control, treatment) -> dict:
    """McNemar 配对检验（§18.4）。

    `control` = 直答臂逐题二值结果，`treatment` = 程序臂逐题二值结果（按 qa 对齐）。
    `b` = 程序对有/直答错的判别对（treatment win），`c` = 反向。

    - 判别对 `b+c < MCNEMAR_EXACT_MAX_DISCOUNT` → 精确二项检验（双侧）；
    - 否则 → 带连续性校正的卡方近似（df=1）；
    - **完全相同组**（b+c=0）→ `p=None` 且 `degenerate=True`，注记
      "退化样本 → 判为不显著"：没有判别对时检验没有信息，不得报成"p=1.0 已检验"。
    """
    ax, bx, dropped = _binary_arrays(control, treatment)
    n = int(ax.size)
    b = int(np.sum((bx == 1.0) & (ax == 0.0)))     # treatment 赢
    c = int(np.sum((bx == 0.0) & (ax == 1.0)))     # control 赢
    discount = b + c
    out: dict = {
        "n_pairs": n, "n_dropped": dropped, "b": b, "c": c,
        "n_treatment_win": b, "n_control_win": c, "n_tie": n - discount,
        "degenerate": bool(discount == 0 or n == 0),
        "note": "", "method": "", "statistic": None, "p": None,
    }
    if discount == 0:
        out["method"] = "degenerate"
        out["note"] = DEGENERATE_NOTE + (
            "：全部配对结果相同（无判别对）→ McNemar 无定义" if n else "：无配对样本")
        return out
    if discount < MCNEMAR_EXACT_MAX_DISCOUNT:
        out["method"] = "exact_binomial"
        out["p"] = normalize_p(_mcnemar_exact_p(b, c))
    else:
        stat = (abs(b - c) - 1.0) ** 2 / discount
        out["method"] = "chi2_continuity_corrected"
        out["statistic"] = float(stat)
        out["p"] = normalize_p(_chi2_sf_df1(max(stat, 0.0)))
    if out["p"] is None:
        out["note"] = DEGENERATE_NOTE
    return out


def _mcnemar_exact_p(b: int, c: int) -> float:
    """精确 McNemar 双侧 p（二项分布 n=b+c, p=0.5 的双侧尾概率）。"""
    n = b + c
    if n <= 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return float(min(1.0, 2.0 * tail))


def _chi2_sf_df1(x: float) -> float:
    """卡方分布（df=1）上尾概率：P(X>x) = erfc(sqrt(x/2))。"""
    return float(math.erfc(math.sqrt(max(x, 0.0) / 2.0)))


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
            return normalize_p(_scipy_stats.wilcoxon(diff).pvalue)
        except ValueError:
            return None
    return _wilcoxon_p_numpy(diff)


def bonferroni(p: float | None, n_comparisons: int) -> float | None:
    """Bonferroni 多重比较校正（§18.4）：`p_corrected = min(1, p × m)`。

    退化样本（p=None）保持 None；`n_comparisons` ≤ 1 时原样返回。
    """
    if p is None:
        return None
    if n_comparisons <= 1:
        return float(p)
    return float(min(1.0, p * n_comparisons))


def cliffs_delta(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Cliff's delta 效应量（§18.4）：P(b>a) − P(b<a)，取值 [-1, 1]。

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
