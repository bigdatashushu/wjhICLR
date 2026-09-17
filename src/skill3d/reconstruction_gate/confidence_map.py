"""M4 置信度融合：ConfidenceMap / CoverageMap（§4 M4 伪代码）。

fused = point_conf × coverage × reproj 衰减；退化区压零。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# 退化区判定阈值（TODO_CALIBRATE，§4 M4 字段 11：注入短基线退化验证压零）
TH_REPROJ_ERR: float = 5.0        # TODO_CALIBRATE: 重投影残差退化阈值（px）
MIN_COVERAGE_COUNT: int = 2       # TODO_CALIBRATE: 覆盖计数低于此记观测不足
REPROJ_DECAY: float = 1.0         # TODO_CALIBRATE: 重投影误差衰减系数


def fuse_confidence(
    point_conf: np.ndarray,
    coverage_count: np.ndarray,
    reproj_err: Optional[np.ndarray] = None,
) -> np.ndarray:
    """置信度融合（§4 M4）：point_conf × coverage 归一化 × reproj 衰减。

    - point_conf: 逐点置信度（VGGT point_conf）
    - coverage_count: 逐点被观测帧数
    - reproj_err: 逐点重投影残差（无则不衰减）
    退化区（覆盖不足 / 残差过大）压零。
    """
    conf = np.asarray(point_conf, dtype=np.float64)
    cov = np.asarray(coverage_count, dtype=np.float64)
    conf = np.nan_to_num(conf, nan=0.0)
    cov_norm = np.clip(cov / max(MIN_COVERAGE_COUNT, cov.max() if cov.size else 1), 0.0, 1.0)
    fused = conf * cov_norm

    if reproj_err is not None:
        err = np.asarray(reproj_err, dtype=np.float64)
        decay = np.exp(-REPROJ_DECAY * np.nan_to_num(err, nan=TH_REPROJ_ERR))
        fused = fused * decay
        fused[err > TH_REPROJ_ERR] = 0.0  # 退化区压零

    fused[cov < MIN_COVERAGE_COUNT] = 0.0  # 观测不足区压零
    return np.clip(fused, 0.0, 1.0)


def coverage_ratio(fused_conf: np.ndarray, conf_thresh: float = 0.5) -> float:
    """覆盖率：融合置信度超阈的点的占比（TODO_CALIBRATE: conf_thresh）。"""
    f = np.asarray(fused_conf)
    if f.size == 0:
        return 0.0
    return float(np.mean(f >= conf_thresh))


def per_object_coverage(
    fused_conf: np.ndarray,
    object_point_indices: dict[str, np.ndarray],
    conf_thresh: float = 0.5,
) -> dict[str, float]:
    """逐对象覆盖率（object_id -> 覆盖率），供 G8 / CoverageMap 使用。"""
    out: dict[str, float] = {}
    for oid, idx in object_point_indices.items():
        idx = np.asarray(idx)
        if idx.size == 0:
            out[oid] = 0.0
            continue
        out[oid] = float(np.mean(np.asarray(fused_conf).ravel()[idx] >= conf_thresh))
    return out
