"""M2 图像质量评估（IQA）底层算子。

真实实现：cv2.Laplacian 方差 / Tenengrad / 曝光直方图占比（G1/G2，§10）。
pyiqa NIQE/MUSIQ 为可选 GPU 辅助：lazy import，不可用则跳过（返回 None）。
"""

from __future__ import annotations

from typing import Optional

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


def tenengrad(image: np.ndarray) -> float:
    """G1 模糊辅助：Tenengrad（Sobel 梯度能量均值）。"""
    gray = _to_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(gx**2 + gy**2))


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


def pyiqa_score(image: np.ndarray, metric: str = "niqe") -> Optional[float]:
    """可选 GPU 辅助 IQA（NIQE/MUSIQ），pyiqa 未安装或失败时返回 None。

    TODO: pyiqa 具体 API 与权重随版本变动，集成前核对（§4 M2 字段 5）。
    """
    try:
        import pyiqa  # lazy import：可选 GPU 依赖
        import torch
    except Exception:
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metric_fn = pyiqa.create_metric(metric, device=device)
        # pyiqa 接受路径/Tensor；此处将 ndarray 转 Tensor（NCHW, [0,1]）
        img = torch.from_numpy(image).float()
        if img.ndim == 2:
            img = img.unsqueeze(-1).repeat(1, 1, 3)
        img = img.permute(2, 0, 1).unsqueeze(0) / 255.0
        return float(metric_fn(img.to(device)).item())
    except Exception:
        return None
