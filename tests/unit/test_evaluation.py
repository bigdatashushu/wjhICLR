"""M12 评测单测：MRA 官方口径 golden + MCA 选项抽取（§4 M12 / §8）。

v3 口径（B-4 决策）：`rel = |pred-gt|/gt`，判定 `rel <= 1-theta`，
`theta ∈ linspace(0.5, 0.95, 10)`；非法值记 0 分（官方 WORST_CASE）。
旧实现（`rel <= theta` 且分母 `max(|gt|,1)`）已作废，NA 结果需重跑。
"""

import numpy as np
import pytest

from skill3d.evaluation.accuracy import extract_option_letter, mca_accuracy, mca_correct
from skill3d.evaluation.mra import (
    MRA_N_THETAS,
    THETAS,
    mean_relative_accuracy,
    mra,
    mra_at_10pct,
    mra_from_text,
    mra_one,
    mra_single,
    mra_with_detail,
    parse_numeric_answer,
    relative_error,
)

# §8.2 golden：rel → MRA（注意 rel=0.2 → 0.7，不是 0.6）
GOLDEN = [(0.0, 1.0), (0.05, 1.0), (0.1, 0.9), (0.2, 0.7), (0.5, 0.1), (1.0, 0.0)]


def test_mra_thetas_count():
    """阈值 = np.linspace(0.5, 0.95, 10)（逐值一致；不能因浮点误差少生成一个）。"""
    assert MRA_N_THETAS == 10 and len(THETAS) == 10
    assert np.isclose(THETAS[0], 0.50) and np.isclose(THETAS[-1], 0.95)
    assert np.allclose(THETAS, np.linspace(0.5, 0.95, 10))


@pytest.mark.parametrize("rel,expected", GOLDEN)
def test_mra_golden_table(rel, expected):
    """§8.2 golden：rel ∈ {0, .05, .1, .2, .5, 1.0} → {1, 1, .9, .7, .1, 0}。"""
    assert mra_one(1.0 + rel, 1.0) == pytest.approx(expected)


def test_mra_b4_episode_hand_computed():
    """§8.2 B-4 episode：pred=1, gt=2（rel=0.5）→ MRA=0.10（仅 θ=0.5 通过）。"""
    assert mra_single(1, 2) == pytest.approx(0.10)
    assert relative_error(1, 2) == pytest.approx(0.5)


def test_mra_perfect_prediction():
    assert mra([2.0, 3.0], [2.0, 3.0]) == pytest.approx(1.0)


def test_mra_denominator_is_gt_without_floor():
    """分母是 gt（不做 max(|gt|,1) 钳制）：gt=0.5、pred=1.05 → rel=1.1 → 0 分。"""
    assert relative_error(1.05, 0.5) == pytest.approx(1.1)
    assert mra_single(1.05, 0.5) == pytest.approx(0.0)


def test_mra_zero_when_far():
    # rel = 9.7/10 = 0.97 → 只有 rel <= 1-θ 才通过；最大 1-θ = 0.5 → 0
    assert mra_single(0.3, 10.0) == pytest.approx(0.0)


def test_mra_illegal_values_score_zero():
    """不可解析 / NaN / 非正 gt → 记 0 分（官方 WORST_CASE，不抛异常不跳过）。"""
    assert mra_from_text("no number here", "2") == pytest.approx(0.0)
    assert mra_one(float("nan"), 2.0) == pytest.approx(0.0)
    assert mra_one(1.0, 0.0) == pytest.approx(0.0)
    assert mra_one(None, 2.0) == pytest.approx(0.0)


def test_mean_relative_accuracy_matches_per_sample_mean():
    """官方同签名接口 = 逐样本 MRA 的算术平均（§8.3 第 8 任务聚合口径）。"""
    preds, gts = [2.0, 1.0, 0.3], [2.0, 2.0, 10.0]
    per_sample = [mra_one(p, g) for p, g in zip(preds, gts)]
    assert mean_relative_accuracy(preds, gts) == pytest.approx(float(np.mean(per_sample)))


def test_mra_numeric_parsing_and_diagnostics():
    """NA 答案解析取数值（带单位）；附录诊断指标 MRA@10% 不进主表（§8.4）。"""
    assert parse_numeric_answer("3.5 m") == pytest.approx(3.5)
    assert parse_numeric_answer("48 m2") == pytest.approx(48.0)
    detail = mra_with_detail([2.0, 1.0], [2.0, 2.0])
    assert detail["mra"] == pytest.approx(0.55)          # (1.0 + 0.10)/2
    assert detail["mra_at_10pct"] == pytest.approx(0.5)
    assert mra_at_10pct([1.0, 1.05], [1.0, 1.0]) == pytest.approx(1.0)  # rel=0.05<=0.1


def test_mca_extract_letter():
    assert extract_option_letter("The answer is B") == "B"
    assert extract_option_letter("A. three chairs") == "A"
    assert extract_option_letter("answer: (C)") == "C"
    assert extract_option_letter("B") == "B"
    assert extract_option_letter("no letter here") is None


def test_mca_correct_and_accuracy():
    assert mca_correct("The answer is A", "A")
    assert not mca_correct("The answer is A", "B")
    assert mca_accuracy(["A", "B", "C"], ["A", "B", "D"]) == pytest.approx(2 / 3)
