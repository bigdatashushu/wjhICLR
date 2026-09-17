"""NA 任务 MRA（§4 M12 / §10）：numpy 真实实现。

rel = |gt - pred| / max(|gt|, 1)；对 θ ∈ {0.50, 0.55, …, 0.95} 共 10 阈值
逐阈值算 (rel ≤ θ) 的比例，再对 10 阈值取平均（§10 口径）。
"""

from __future__ import annotations

import numpy as np

MRA_THETAS = np.arange(0.50, 0.951, 0.05)  # 10 阈值（以官方为准 TODO）


def mra(preds, gts) -> float:
    """MRA：逐阈值 (rel ≤ θ) 比例再取 mean（§4 M12 伪代码）。"""
    preds = np.asarray(preds, dtype=np.float64)
    gts = np.asarray(gts, dtype=np.float64)
    if preds.shape != gts.shape:
        raise ValueError("preds 与 gts 形状不一致")
    if preds.size == 0:
        raise ValueError("空输入")
    rel = np.abs(gts - preds) / np.maximum(np.abs(gts), 1.0)
    return float(np.mean([(rel <= th).mean() for th in MRA_THETAS]))


def mra_single(pred: float, gt: float) -> float:
    return mra([pred], [gt])
