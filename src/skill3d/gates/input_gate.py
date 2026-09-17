"""M2 Input Gate：输入帧质量门禁（§4 M2 伪代码实现）。

在线模块，严禁任何 GPT-6 相关依赖（硬约束 1）。
重建/Skill 路由之前必须先过本门禁（硬约束 16）。
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from skill3d.schemas.episode import InputFrame, InputGateVerdict
from skill3d.gates import iqa

# ---- 阈值常量（全部 TODO_CALIBRATE，起始参考值见 §4 M2 字段 12 / §10 G1-G4）----
TH_BLUR_VAR: float = 100.0        # TODO_CALIBRATE: σ²_blur < 100 记模糊
TH_OVER_EXPOSED: float = 0.05     # TODO_CALIBRATE: p_over > 5% 记曝光异常
TH_UNDER_EXPOSED: float = 0.05    # TODO_CALIBRATE: p_under > 5% 记曝光异常
TH_DEGRADED_RATIO: float = 0.25   # TODO_CALIBRATE: 劣化帧占比超过则 drop_and_refill
MIN_FRAMES: int = 32              # TODO_CALIBRATE: G4 帧数完整性（官方 32 帧）
TH_MOTION: float = 20.0           # TODO_CALIBRATE: G3 帧间光流均值阈值（px）


def frame_quality(frame: Union[InputFrame, np.ndarray]) -> tuple[float, float, float]:
    """返回 (blur_var, overexposed_ratio, underexposed_ratio)。

    输入为 InputFrame 时直接读已算好的字段；输入为原始图像 ndarray 时实时计算。
    """
    if isinstance(frame, InputFrame):
        return frame.blur_var, frame.overexposed_ratio, frame.underexposed_ratio
    blur = iqa.laplacian_var(frame)
    p_over, p_under = iqa.exposure_ratios(frame)
    return blur, p_over, p_under


def _is_degraded(blur: float, p_over: float, p_under: float) -> bool:
    return blur < TH_BLUR_VAR or p_over > TH_OVER_EXPOSED or p_under > TH_UNDER_EXPOSED


def input_gate(frames: Sequence[Union[InputFrame, np.ndarray]]) -> InputGateVerdict:
    """按 §4 M2 伪代码实现的整体门禁判定。"""
    scores = [frame_quality(f) for f in frames]
    degraded = [i for i, (b, po, pu) in enumerate(scores) if _is_degraded(b, po, pu)]

    # G3 运动模糊：仅当输入为原始图像时可算；InputFrame 无像素则跳过（记为不过高）
    motion_too_high = False
    if frames and isinstance(frames[0], np.ndarray) and len(frames) >= 2:
        mags = [
            iqa.motion_score(frames[i - 1], frames[i])  # type: ignore[arg-type]
            for i in range(1, len(frames))
        ]
        motion_too_high = bool(np.mean(mags) > TH_MOTION)

    # G4 帧数完整性 + 整体运动模糊 → 整体不可用
    if len(frames) < MIN_FRAMES or motion_too_high:
        return InputGateVerdict(
            level="overall_unusable",
            degraded_frame_ids=degraded,
            action="unanswerable",
        )

    # 全部帧均劣化 → 整体不可用（等效 overall_motion_too_high 的曝光/模糊情形）
    if len(degraded) == len(frames) and len(frames) > 0:
        return InputGateVerdict(
            level="overall_unusable",
            degraded_frame_ids=degraded,
            action="unanswerable",
        )

    # 局部劣化超阈 → 屏蔽补帧（§4 M2 伪代码：degraded 占比超 TH_DEGRADED_RATIO）
    if frames and len(degraded) / len(frames) > TH_DEGRADED_RATIO:
        return InputGateVerdict(
            level="locally_degraded",
            degraded_frame_ids=degraded,
            action="drop_and_refill",
        )

    return InputGateVerdict(level="pass", degraded_frame_ids=degraded, action="proceed")
