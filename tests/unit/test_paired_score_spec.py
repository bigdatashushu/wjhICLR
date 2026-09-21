"""§7 统计口径测试：BCa bootstrap / 退化样本 / Bonferroni / 效应量。

对应《系统架构3.md》§7「统计与复现」：
- paired A/B 用 `scipy.stats.bootstrap(paired=True, n_resamples=9999, method='BCa')`；
- 退化样本（零方差/完全相同组）→ p 归一化为 None，判"不显著"（E-2），
  **不得**声称有显著差异；
- 多重比较校正（Bonferroni）；效应量报 Cliff's delta / Cohen's d。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.evolution.paired_score import (
    N_BOOTSTRAP,
    bonferroni,
    cliffs_delta,
    cohens_d_paired,
    is_degenerate,
    paired_bootstrap_ci,
    score_paired,
    wilcoxon_p,
)


def test_n_resamples_follows_spec():
    """§7 明确要求 9999 次重采样（旧实现是 1000）。"""
    assert N_BOOTSTRAP == 9999


def test_bca_matches_scipy_bca():
    """默认走 scipy BCa（而非百分位 bootstrap）——与 scipy 直算逐值一致。"""
    from scipy import stats

    rng = np.random.default_rng(3)
    a = rng.normal(0.4, 0.2, size=40)
    b = a + rng.normal(0.1, 0.15, size=40)
    lo, hi = paired_bootstrap_ci(a, b, seed=11)
    ref = stats.bootstrap(
        (a, b), statistic=lambda x, y, axis=-1: np.mean(y - x, axis=axis),
        paired=True, vectorized=True, n_resamples=N_BOOTSTRAP,
        confidence_level=0.95, method="BCa",
        random_state=np.random.default_rng(11))
    assert lo == pytest.approx(float(ref.confidence_interval.low))
    assert hi == pytest.approx(float(ref.confidence_interval.high))


def test_bca_differs_from_percentile_on_skewed_data():
    """BCa 会做偏差/偏度校正：偏态数据上与百分位法不同（证明真的用了 BCa）。"""
    rng = np.random.default_rng(5)
    a = np.zeros(30)
    b = rng.exponential(0.5, size=30)          # 强右偏
    lo_bca, hi_bca = paired_bootstrap_ci(a, b, seed=7)
    n = a.shape[0]
    rng2 = np.random.default_rng(7)
    diffs = np.array([(b[rng2.integers(0, n, n)] - a[rng2.integers(0, n, n)]).mean()
                      for _ in range(N_BOOTSTRAP)])
    lo_pct = float(np.percentile(diffs, 2.5))
    assert lo_bca != pytest.approx(lo_pct, abs=1e-9)
    assert lo_bca <= hi_bca


def test_degenerate_sample_p_is_none_not_significant():
    """退化样本 → p=None（判"不显著"），不得返回 1.0 冒充"测过但不显著"（E-2）。"""
    a = np.array([1.0, 0.0, 1.0, 0.0])
    assert is_degenerate(a - a)
    assert wilcoxon_p(a, a) is None
    res = score_paired(a, a, ["x"] * 4, seed=0)
    assert res["wilcoxon_p"] is None and res["degenerate"] is True
    assert res["ci_significant"] is False
    # 单样本同样退化
    assert wilcoxon_p(np.array([1.0]), np.array([0.0])) is None


def test_bonferroni_correction():
    assert bonferroni(0.01, 5) == pytest.approx(0.05)
    assert bonferroni(0.5, 5) == 1.0            # 上限截断
    assert bonferroni(0.01, 1) == pytest.approx(0.01)
    assert bonferroni(None, 5) is None          # 退化样本保持 None


def test_effect_sizes():
    """Cliff's delta ∈ [-1,1]；配对 Cohen's d = mean(diff)/sd(diff)（退化→0 而非 inf）。"""
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    assert cliffs_delta(a, b) == pytest.approx(1.0)      # b 恒大于 a
    assert cliffs_delta(b, a) == pytest.approx(-1.0)
    assert cliffs_delta(a, a) == pytest.approx(0.0)
    # 差值恒定 → sd=0（退化）：返回 0.0，不返回 inf/NaN
    assert cohens_d_paired(np.array([1.0, 2.0, 3.0]),
                           np.array([2.0, 3.0, 4.0])) == 0.0
    # 有方差的差值：mean/sd(ddof=1)
    a2 = np.array([0.0, 1.0, 2.0, 3.0])
    b2 = np.array([1.0, 1.0, 4.0, 3.0])
    diff = b2 - a2
    assert cohens_d_paired(a2, b2) == pytest.approx(
        float(np.mean(diff) / np.std(diff, ddof=1)))


def test_slice_table_carries_corrected_p_values():
    """切片表带 p 与 Bonferroni 校正后的 p（按切片数校正）。"""
    rng = np.random.default_rng(0)
    a = rng.binomial(1, 0.4, size=60).astype(float)
    b = np.clip(a + rng.binomial(1, 0.5, size=60), 0, 1).astype(float)
    tt = ["t1"] * 20 + ["t2"] * 20 + ["t3"] * 20
    res = score_paired(a, b, tt, seed=0)
    table = res["slice_table"]
    assert res["n_comparisons"] == 3
    for row in table.values():
        assert "p_bonferroni" in row and "degenerate" in row
        if row["p"] is not None:
            assert row["p_bonferroni"] == pytest.approx(min(1.0, row["p"] * 3))
    assert res["wilcoxon_p_bonferroni"] == pytest.approx(
        min(1.0, res["wilcoxon_p"] * 3))
