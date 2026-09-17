"""M12 评测单测：MRA 手工算例 + MCA 选项抽取（§4 M12 字段 11）。"""

import numpy as np
import pytest

from skill3d.evaluation.accuracy import extract_option_letter, mca_accuracy, mca_correct
from skill3d.evaluation.mra import MRA_THETAS, mra, mra_single


def test_mra_thetas_count():
    assert len(MRA_THETAS) == 10
    assert np.isclose(MRA_THETAS[0], 0.50) and np.isclose(MRA_THETAS[-1], 0.95)


def test_mra_perfect_prediction():
    assert mra([2.0, 3.0], [2.0, 3.0]) == pytest.approx(1.0)


def test_mra_hand_computed():
    # gt=10, pred=4 → rel = 6/10 = 0.6
    # 命中条件 rel <= θ：θ ∈ {0.60,…,0.95} 共 8 个 → 0.8
    assert mra_single(4.0, 10.0) == pytest.approx(0.8)


def test_mra_small_gt_denominator_floor():
    # gt=0.5 → max(|gt|,1)=1；pred=1.05 → rel=0.55
    # θ >= 0.55：{0.55,…,0.95} 共 9 个 → 0.9
    assert mra_single(1.05, 0.5) == pytest.approx(0.9)


def test_mra_zero_when_far():
    # rel = 9.7/10 = 0.97 > 所有 θ（最大 0.95）→ 0
    assert mra_single(0.3, 10.0) == pytest.approx(0.0)


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
