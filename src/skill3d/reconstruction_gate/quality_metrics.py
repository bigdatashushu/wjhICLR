"""M4 G1-G11 重建/输入质量指标计算（§10 指标表）。

阈值全部 TODO_CALIBRATE。能用 numpy/cv2 真实算的指标真实计算；
依赖外部产物（重投影残差、动态 mask、跟踪 IoU、基线、尺度 CI）的指标
从传入数据/ReconstructionArtifact 读取，缺失时给 NaN 并注释 TODO。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from skill3d.gates import iqa
from skill3d.gates.input_gate import (
    MIN_FRAMES,
    TH_BLUR_VAR,
    TH_OVER_EXPOSED,
    TH_UNDER_EXPOSED,
)
from skill3d.schemas.reconstruction import QualityMetrics, ReconstructionArtifact

# ---- 阈值常量（全部 TODO_CALIBRATE，起始参考值见 §10）----
TH_G3_MOTION: float = 20.0        # TODO_CALIBRATE: G3 运动模糊光流幅值（px）
TH_G5_MEDIAN: float = 2.0         # TODO_CALIBRATE: G5 重投影残差 median（px）
TH_G5_P95: float = 5.0            # TODO_CALIBRATE: G5 重投影残差 p95（px）
TH_G6_DEPTH_CV: float = 1.0       # TODO_CALIBRATE: G6 深度方差系数 σ/μ
TH_G7_DYNAMIC: float = 0.3        # TODO_CALIBRATE: G7 动态物体占比
TH_G8_COVERAGE: float = 0.8       # TODO_CALIBRATE: G8 包围盒覆盖率 <80% 记不可答
TH_G9_TRACK_IOU: float = 0.5      # TODO_CALIBRATE: G9 跟踪一致性 IoU
TH_G10_BASELINE: float = 0.1      # TODO_CALIBRATE: G10 基线质量（归一化）
TH_G11_SCALE_CI: float = 0.1      # TODO_CALIBRATE: G11 尺度 CI 半宽（m）

NaN = float("nan")


def g1_blur_ok(frames: Sequence[np.ndarray]) -> float:
    """G1 帧模糊：通过阈值（σ²>=TH_BLUR_VAR）的帧占比，[0,1]。"""
    if not frames:
        return NaN
    ok = sum(iqa.laplacian_var(f) >= TH_BLUR_VAR for f in frames)
    return ok / len(frames)


def g2_brightness(frames: Sequence[np.ndarray]) -> float:
    """G2 曝光：曝光正常帧占比，[0,1]。"""
    if not frames:
        return NaN
    ok = 0
    for f in frames:
        p_over, p_under = iqa.exposure_ratios(f)
        if p_over <= TH_OVER_EXPOSED and p_under <= TH_UNDER_EXPOSED:
            ok += 1
    return ok / len(frames)


def g3_motion_blur(frames: Sequence[np.ndarray]) -> float:
    """G3 运动模糊：帧间光流幅值均值（px），越小越好。"""
    if len(frames) < 2:
        return NaN
    mags = [iqa.motion_score(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    return float(np.mean(mags))


def g4_frame_count(frames: Sequence) -> int:
    """G4 帧数完整性：实际可用帧数。"""
    return len(frames)


def g5_reproj_err(reproj_errors: Optional[np.ndarray]) -> tuple[float, float]:
    """G5 重投影残差 median/p95（px）。

    TODO: 残差来自 COLMAP/VGGT-BA 产物；无数据时返回 (NaN, NaN)。
    """
    if reproj_errors is None or len(reproj_errors) == 0:
        return NaN, NaN
    e = np.asarray(reproj_errors, dtype=np.float64)
    return float(np.median(e)), float(np.percentile(e, 95))


def g6_depth_var_coeff(depth_maps: np.ndarray) -> float:
    """G6 深度方差系数 σ/μ（无量纲），跨全部帧的全局统计。

    异常记深度漂移；深度无效（全 0/NaN）时返回 NaN。
    """
    d = np.asarray(depth_maps, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return NaN
    mu = float(np.mean(d))
    if mu <= 0:
        return NaN
    return float(np.std(d) / mu)


def g7_dynamic_ratio(dynamic_masks: Optional[np.ndarray]) -> float:
    """G7 动态物体占比：动态 mask 像素占比。

    TODO: 动态 mask 来自 M5 SAM2；无数据时 NaN。
    """
    if dynamic_masks is None:
        return NaN
    m = np.asarray(dynamic_masks)
    if m.size == 0:
        return NaN
    return float(np.mean(m.astype(bool)))


def g8_bbox_coverage_min(bbox_coverage: Optional[dict]) -> float:
    """G8 包围盒覆盖：所有对象 3D bbox 点云覆盖率的最小值。

    TODO: 覆盖率来自 M5 对象点云绑定（ObjectInstance.bbox 与点云密度）；
    无对象时 NaN。
    """
    if not bbox_coverage:
        return NaN
    return float(min(bbox_coverage.values()))


def g9_tracker_consistency(track_ious: Optional[Sequence[float]]) -> float:
    """G9 跟踪一致性：SAM2 mask 跨帧 IoU 均值，[0,1]。

    TODO: IoU 来自 M5 SAM2 传播；无数据时 NaN。
    """
    if not track_ious:
        return NaN
    return float(np.mean(track_ious))


def g10_baseline_quality(c2w_list: Optional[np.ndarray]) -> float:
    """G10 基线质量：相机基线相对场景尺度的覆盖（归一化）。

    真实可算部分：由 c2w 平移分量估计相邻帧基线中位数 / 场景直径。
    无位姿数据时 NaN（TODO: 需 VGGT/COLMAP 位姿产物）。
    """
    if c2w_list is None:
        return NaN
    c2w = np.asarray(c2w_list, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4) or len(c2w) < 2:
        return NaN
    centers = c2w[:, :3, 3]
    baselines = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    diameter = float(np.max(np.linalg.norm(
        centers[None, :, :] - centers[:, None, :], axis=-1)))
    if diameter <= 0:
        return NaN
    return float(np.median(baselines) / diameter)


def g11_scale_ci(scale_ci: Optional[float]) -> float:
    """G11 尺度 CI：PaGeR 尺度区间半宽（m）；未知时 NaN。"""
    if scale_ci is None:
        return NaN
    return float(scale_ci)


def _norm_scores(q: dict) -> list[float]:
    """将各指标归一化为 [0,1] 的"越高越好"分数，NaN 项跳过。"""
    scores = []
    for v in (q["g1_blur_ok"], q["g2_brightness"]):
        if np.isfinite(v):
            scores.append(v)
    if np.isfinite(q["g3_motion_blur"]):
        scores.append(1.0 if q["g3_motion_blur"] <= TH_G3_MOTION else 0.0)
    scores.append(1.0 if q["g4_frame_count"] >= MIN_FRAMES else q["g4_frame_count"] / MIN_FRAMES)
    for v, th in ((q["g5_reproj_err_median"], TH_G5_MEDIAN),
                  (q["g6_depth_var_coeff"], TH_G6_DEPTH_CV),
                  (q["g7_dynamic_ratio"], TH_G7_DYNAMIC)):
        if np.isfinite(v):
            scores.append(1.0 if v <= th else 0.0)
    if np.isfinite(q["g5_reproj_err_p95"]):
        scores.append(1.0 if q["g5_reproj_err_p95"] <= TH_G5_P95 else 0.0)
    for v, th in ((q["g8_bbox_coverage_min"], TH_G8_COVERAGE),
                  (q["g9_tracker_consistency"], TH_G9_TRACK_IOU),
                  (q["g10_baseline_quality"], TH_G10_BASELINE)):
        if np.isfinite(v):
            scores.append(1.0 if v >= th else v / th if th > 0 else 0.0)
    if np.isfinite(q["g11_scale_ci"]):
        scores.append(1.0 if q["g11_scale_ci"] <= TH_G11_SCALE_CI else 0.0)
    return scores


def compute_g1_g11(
    artifact: Optional[ReconstructionArtifact] = None,
    *,
    frames: Optional[Sequence[np.ndarray]] = None,
    depth_maps: Optional[np.ndarray] = None,
    reproj_errors: Optional[np.ndarray] = None,
    dynamic_masks: Optional[np.ndarray] = None,
    bbox_coverage: Optional[dict] = None,
    track_ious: Optional[Sequence[float]] = None,
    c2w_list: Optional[np.ndarray] = None,
    scale_ci: Optional[float] = None,
) -> QualityMetrics:
    """计算 G1-G11（§10）。

    依赖外部产物的指标从参数读入；artifact 已携带 quality 时可作数据源。
    算不出的指标为 NaN（见各函数 TODO 注释）。
    """
    # artifact 中已有的 quality 字段作为兜底数据源
    art_q = artifact.quality if artifact is not None else None

    def _fallback(val: float, name: str) -> float:
        return val if np.isfinite(val) or art_q is None else getattr(art_q, name)

    g5_med, g5_p95 = g5_reproj_err(reproj_errors)
    q = {
        "g1_blur_ok": _fallback(g1_blur_ok(frames) if frames else NaN, "g1_blur_ok"),
        "g2_brightness": _fallback(g2_brightness(frames) if frames else NaN, "g2_brightness"),
        "g3_motion_blur": _fallback(
            g3_motion_blur(frames) if frames else NaN, "g3_motion_blur"),
        "g4_frame_count": len(frames) if frames is not None else (
            art_q.g4_frame_count if art_q is not None else 0),
        "g5_reproj_err_median": _fallback(g5_med, "g5_reproj_err_median"),
        "g5_reproj_err_p95": _fallback(g5_p95, "g5_reproj_err_p95"),
        "g6_depth_var_coeff": _fallback(
            g6_depth_var_coeff(depth_maps) if depth_maps is not None else NaN,
            "g6_depth_var_coeff"),
        "g7_dynamic_ratio": _fallback(g7_dynamic_ratio(dynamic_masks), "g7_dynamic_ratio"),
        "g8_bbox_coverage_min": _fallback(g8_bbox_coverage_min(bbox_coverage), "g8_bbox_coverage_min"),
        "g9_tracker_consistency": _fallback(g9_tracker_consistency(track_ious), "g9_tracker_consistency"),
        "g10_baseline_quality": _fallback(g10_baseline_quality(c2w_list), "g10_baseline_quality"),
        "g11_scale_ci": _fallback(g11_scale_ci(scale_ci), "g11_scale_ci"),
    }
    scores = _norm_scores(q)
    q["overall_quality"] = float(np.mean(scores)) if scores else 0.0
    return QualityMetrics(**q)
