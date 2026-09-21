"""M2 input_gate 单测（§4 M2 字段 11，**被动观测**口径）。

硬约束 21：M2 只打 flag/weight，绝不删/换/补/重排帧；
唯一 hard fail 通道是输入合法性（空帧/尺寸非法/损坏/帧数 < 32）。
"""

import cv2
import numpy as np
import pytest

from skill3d.gates import iqa
from skill3d.gates.input_gate import (
    MIN_FRAMES,
    TH_BLUR_VAR,
    input_gate,
)
from skill3d.schemas.episode import InputFrame


def _sharp_frame(seed: int = 0, size: int = 128) -> np.ndarray:
    """纹理丰富的清晰帧（随机噪声，Laplacian 方差高）。"""
    rng = np.random.default_rng(seed)
    return (rng.random((size, size, 3)) * 200 + 30).astype(np.uint8)


def _blurred_frame(seed: int = 0) -> np.ndarray:
    """高斯模糊帧（大核模糊，Laplacian 方差低）。"""
    return cv2.GaussianBlur(_sharp_frame(seed), (25, 25), 0)


def _mk_input_frame(blur: float, p_over: float = 0.0, p_under: float = 0.0) -> InputFrame:
    return InputFrame(
        frame_idx=0, timestamp=0.0, blur_var=blur,
        overexposed_ratio=p_over, underexposed_ratio=p_under,
        quality_ok=blur >= TH_BLUR_VAR,
    )


def test_blur_metric_drops_on_gaussian_blur():
    """注入高斯模糊帧，blur_var 显著下降（§4 M2 验收）。"""
    sharp = iqa.laplacian_var(_sharp_frame())
    blurred = iqa.laplacian_var(_blurred_frame())
    assert blurred < sharp * 0.1
    assert blurred < TH_BLUR_VAR < sharp


def test_blurred_frames_marked_degraded():
    """注入少量高斯模糊帧 → 被标 degraded（出现在 degraded_frame_ids）。"""
    frames = [_sharp_frame(seed=i) for i in range(MIN_FRAMES)]
    bad_ids = [3, 7, 11]
    for i in bad_ids:
        frames[i] = _blurred_frame(seed=i)
    verdict = input_gate(frames)
    for i in bad_ids:
        assert i in verdict.degraded_frame_ids
    # 3/32 未超 TH_DEGRADED_RATIO → 仍 pass
    assert verdict.level == "pass"
    assert verdict.action == "proceed"


def test_locally_degraded_only_flags_and_downweights():
    """劣化帧占比超阈 → locally_degraded，但 action 仍是 proceed（硬约束 21）。

    旧的 `drop_and_refill`（删帧/补帧）已废弃：帧集在 M1 后冻结，
    这里只降权 + 打 flag，帧数与帧序不变。
    """
    frames = [_mk_input_frame(blur=500.0) for _ in range(MIN_FRAMES)]
    for i in range(0, 16):  # 50% 劣化
        frames[i] = _mk_input_frame(blur=10.0)
    verdict = input_gate(frames)
    assert verdict.level == "locally_degraded"
    assert verdict.action == "proceed"
    assert len(verdict.degraded_frame_ids) == 16
    assert verdict.n_frames == MIN_FRAMES              # 帧数不变
    assert 0.0 < verdict.quality_weight < 1.0          # 只降权
    assert verdict.degradation_flags == ["blur"]


def test_all_degraded_still_proceeds_with_low_weight():
    """全帧劣化也**不是** hard fail：只降权（输入合法性才是唯一 hard fail 通道）。"""
    frames = [_mk_input_frame(blur=1.0) for _ in range(MIN_FRAMES)]
    verdict = input_gate(frames)
    assert verdict.level == "locally_degraded"
    assert verdict.action == "proceed"
    assert verdict.quality_weight == pytest.approx(0.5)
    assert len(verdict.degraded_frame_ids) == MIN_FRAMES


def test_illegal_frames_are_hard_fail():
    """输入合法性（空帧 / 尺寸非法）→ overall_unusable + unanswerable（§4 M2）。"""
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(MIN_FRAMES)]
    frames[7] = np.zeros((0, 0, 3), dtype=np.uint8)      # 空帧
    verdict = input_gate(frames)
    assert verdict.level == "overall_unusable"
    assert verdict.action == "unanswerable"
    assert 7 in verdict.hard_fail_frame_ids


def test_insufficient_frames_unanswerable():
    """G4 帧数不足 32 → 输入合法性 hard fail（§10 G4）。"""
    frames = [_mk_input_frame(blur=500.0) for _ in range(MIN_FRAMES - 1)]
    verdict = input_gate(frames)
    assert verdict.level == "overall_unusable"
    assert verdict.action == "unanswerable"


def test_overexposed_frame_degraded():
    """曝光异常帧（p_over>5%）被标 degraded（§10 G2）。"""
    frames = [_mk_input_frame(blur=500.0) for _ in range(MIN_FRAMES)]
    frames[5] = _mk_input_frame(blur=500.0, p_over=0.2)
    verdict = input_gate(frames)
    assert 5 in verdict.degraded_frame_ids
