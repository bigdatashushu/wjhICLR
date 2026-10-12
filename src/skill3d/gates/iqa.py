"""M2 图像质量评估（IQA）底层算子。

真实实现：cv2.Laplacian 方差 / 曝光直方图占比（G1/G2，§10）。
"""

from __future__ import annotations

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def laplacian_var(image: np.ndarray) -> float:
    """G1 模糊：灰度 Laplacian 方差（σ² 越小越模糊）。"""
    gray = _to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_ratios(
    image: np.ndarray,
    over_thresh: int = 250,
    under_thresh: int = 5,
) -> tuple[float, float]:
    """G2 曝光：直方图过曝/欠曝像素占比（阈值 TODO_CALIBRATE）。"""
    gray = _to_gray(image)
    total = gray.size
    p_over = float(np.count_nonzero(gray >= over_thresh)) / total
    p_under = float(np.count_nonzero(gray <= under_thresh)) / total
    return p_over, p_under


def motion_score(prev: np.ndarray, curr: np.ndarray) -> float:
    """G3 运动模糊辅助：帧间光流幅值均值（px）。

    使用 Farneback 稠密光流；帧差异过大提示运动模糊/抖动风险。
    """
    g0 = _to_gray(prev)
    g1 = _to_gray(curr)
    flow = cv2.calcOpticalFlowFarneback(
        g0, g1, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag))
