"""M17 确定性配对评分测试：bootstrap CI 与 wilcoxon。"""

import numpy as np

from skill3d.evolution.paired_score import (
    build_slice_table,
    check_slice_no_regression,
    paired_bootstrap_ci,
    score_paired,
    wilcoxon_p,
)


def test_significant_improvement():
    rng = np.random.default_rng(42)
    a = rng.binomial(1, 0.4, size=50).astype(float)
    b = np.clip(a + rng.binomial(1, 0.5, size=50), 0, 1).astype(float)
    res = score_paired(a, b, ["object_count"] * 50, seed=0)
    assert res["delta"] > 0
    assert res["ci95_lo"] > 0  # 显著提升 → CI 下界 > 0
    assert res["ci_significant"] is True
    assert res["wilcoxon_p"] < 0.05


def test_no_difference_not_significant():
    a = np.array([0.0, 1.0] * 25)
    b = a.copy()
    res = score_paired(a, b, ["abs_dist"] * 50, seed=0)
    assert res["delta"] == 0.0
    assert res["ci95_lo"] <= 0
    assert res["ci_significant"] is False
    assert wilcoxon_p(a, b) == 1.0  # 全零差


def test_bootstrap_ci_deterministic():
    a = np.arange(10.0)
    b = a + 1.0
    lo1, hi1 = paired_bootstrap_ci(a, b, seed=7)
    lo2, hi2 = paired_bootstrap_ci(a, b, seed=7)
    assert (lo1, hi1) == (lo2, hi2)  # 同 seed 可复现
    assert lo1 > 0  # 恒定 +1 差值


def test_slice_table_and_no_regression():
    a = np.array([0.5, 0.5, 0.5, 0.5])
    b = np.array([0.6, 0.6, 0.4, 0.4])
    tt = ["x", "x", "y", "y"]
    table = build_slice_table(a, b, tt)
    import pytest
    assert table["x"]["delta"] == pytest.approx(0.1)
    assert table["y"]["delta"] == pytest.approx(-0.1)
    assert check_slice_no_regression(table, tol=0.01) is False  # y 切片退化超容忍
    assert check_slice_no_regression(table, tol=0.2) is True
