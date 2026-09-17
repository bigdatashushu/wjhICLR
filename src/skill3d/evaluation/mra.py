"""NA 任务 MRA（§4 M12 / §10）：numpy 真实实现。

rel = |gt - pred| / max(|gt|, 1)；对 θ ∈ {0.50, 0.55, …, 0.95} 共 10 阈值
逐阈值算 (rel ≤ θ) 的比例，再对 10 阈值取平均（§10 口径）。
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

MRA_THETAS = np.arange(0.50, 0.951, 0.05)  # 10 阈值（以官方为准 TODO）

# 数值抽取：首个带可选正负号/小数/科学计数法的数字
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_numeric_answer(text: Optional[str]) -> Optional[float]:
    """从模型输出抽取数值（§4 M12 字段 9：不可解析 → 记 wrong / MRA 0）。

    NA 题答案可能带单位（"3.5 m"、"12 meters"、"48 m2"），取首个数值。
    """
    if text is None:
        return None
    m = _NUM_RE.search(str(text))
    if m is None:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


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
