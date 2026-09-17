"""M3 metric 尺度锚定：PaGeR 全景 metric 深度对齐（§4 M3）。

PaGeR 代码 Apache-2.0 / 权重 CC BY-NC（研究用）；lazy import。
失败 → scale_known=False，metric 类问题（Abs Dist/Obj Size/Room Size）
下游转 unanswerable 或 2D-only（§4 M3 字段 9）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

PAGER_CHECKPOINT: Optional[str] = None  # TODO_USER_INPUT: PaGeR unified checkpoint（~11.5GB fp16）


def anchor_metric_scale(
    depth_ref: str,
    scene_name: str = "",
    checkpoint: Optional[str] = PAGER_CHECKPOINT,
) -> tuple[Optional[float], float, bool]:
    """PaGeR 尺度锚定。

    返回 (scale, scale_ci, scale_known)：
    - scale: 相对深度 → metric 深度的尺度因子（s>0）
    - scale_ci: G11 尺度置信区间半宽（m）
    - scale_known: 锚定是否成功；失败时 scale=None, ci=inf

    TODO: PaGeR inference.py --checkpoint pager 具体接口待核验（§4 M3 字段 5）。
    TODO_CALIBRATE: VSI-Bench 非全景，全景尺度锚定的必要性待校准（§4 M3 字段 12）。
    """
    try:
        import torch  # lazy import
        # import pager  # TODO: PaGeR 真实包名/接口未核验
    except Exception:
        return None, float("inf"), False

    if checkpoint is None:
        return None, float("inf"), False

    # TODO: 加载 PaGeR 权重，对帧做全景 metric 深度推理，
    # 与 VGGT 相对深度做鲁棒尺度拟合（median ratio / RANSAC），
    # 返回尺度因子与 CI。未接入前恒返回失败。
    return None, float("inf"), False


def fit_scale_robust(rel_depth: np.ndarray, metric_depth: np.ndarray) -> tuple[float, float]:
    """鲁棒尺度拟合：s = median(metric/rel)，CI 由 MAD 估计（真实可算部分）。

    G11 尺度 CI（§10）：CI 过宽记 scale_unknown，metric 问题拒答。
    """
    rel = np.asarray(rel_depth, dtype=np.float64).ravel()
    met = np.asarray(metric_depth, dtype=np.float64).ravel()
    mask = (rel > 1e-6) & np.isfinite(rel) & np.isfinite(met) & (met > 0)
    if mask.sum() < 100:  # TODO_CALIBRATE: 最少有效像素数
        return float("nan"), float("inf")
    ratio = met[mask] / rel[mask]
    s = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - s)))
    # 95% CI 近似：1.96 * 1.4826 * MAD / sqrt(N)
    ci = float(1.96 * 1.4826 * mad / np.sqrt(mask.sum()))
    return s, ci
