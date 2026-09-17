"""M4 SceneState 构造与质量门禁分流（§4 M4 伪代码）。

在线模块，严禁任何 GPT-6 相关依赖（硬约束 1）。
重建先于 Skill 路由（硬约束 16）；Tool 只经 SceneState 句柄访问产物（硬约束 17）。
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from skill3d.reconstruction_gate import quality_metrics as qm
from skill3d.reconstruction_gate.confidence_map import (
    coverage_ratio,
    fuse_confidence,
    per_object_coverage,
)
from skill3d.schemas.reconstruction import (
    ConfidenceMap,
    QualityMetrics,
    ReconstructionArtifact,
    SceneState,
)

# 整体质量阈值（TODO_CALIBRATE，§4 M4 字段 7：低于则 route=fallback_2d_only）
TH_OVERALL_QUALITY: float = 0.5   # TODO_CALIBRATE
TH_SCALE_CI: float = 0.1          # TODO_CALIBRATE: G11 CI 过宽记 scale_unknown（m）


def route_from_quality(q: QualityMetrics, scale_known: bool) -> str:
    """§4 M4 伪代码的分流规则（单一事实源）。

    unanswerable（无可用帧）/ fallback_2d_only（质量不足或尺度未知）/ full_3d。
    """
    if q.g4_frame_count < 1:
        return "unanswerable"
    if not scale_known or q.overall_quality < TH_OVERALL_QUALITY:
        return "fallback_2d_only"
    return "full_3d"


def quality_gate(
    art: ReconstructionArtifact,
    *,
    frames: Optional[Sequence[np.ndarray]] = None,
    depth_maps: Optional[np.ndarray] = None,
    reproj_errors: Optional[np.ndarray] = None,
    dynamic_masks: Optional[np.ndarray] = None,
    bbox_coverage: Optional[dict] = None,
    track_ious: Optional[Sequence[float]] = None,
    c2w_list: Optional[np.ndarray] = None,
    object_point_indices: Optional[dict[str, np.ndarray]] = None,
    artifact_ref: str = "",
) -> SceneState:
    """按 §4 M4 伪代码：算 G1-G11 → 融置信度 → 构造 SceneState → 分流。"""
    # G11 尺度 CI 由 artifact 的尺度锚定结果决定
    scale_ci = None
    if art.scale_known and art.metric_scale is not None:
        scale_ci = 0.0  # TODO: 真实 CI 需由 metric_scale 锚定流程写入 artifact
    scale_known = bool(
        art.scale_known and art.metric_scale is not None
        and (scale_ci is None or scale_ci <= TH_SCALE_CI)
    )

    q = qm.compute_g1_g11(
        art,
        frames=frames,
        depth_maps=depth_maps,
        reproj_errors=reproj_errors,
        dynamic_masks=dynamic_masks,
        bbox_coverage=bbox_coverage,
        track_ious=track_ious,
        c2w_list=c2w_list,
        scale_ci=scale_ci,
    )

    # 置信度融合（point_conf × coverage × reproj_err），数据缺失时退化为全 1
    coverage_overall = float("nan")
    try:
        point_conf = np.load(art.point_conf) if art.point_conf else None
    except Exception:
        point_conf = None  # TODO: ref 解析失败时的降级策略
    if point_conf is not None:
        cov_count = (
            np.load(art.confidence.coverage_count_per_frame)
            if art.confidence.coverage_count_per_frame
            else np.full_like(point_conf, qm.MIN_FRAMES if hasattr(qm, "MIN_FRAMES") else 32)
        )
        fused = fuse_confidence(point_conf, cov_count, reproj_errors)
        coverage_overall = coverage_ratio(fused)
        if object_point_indices:
            per_object_coverage(fused, object_point_indices)

    # 分流（§4 M4 伪代码：overall_quality < TH → fallback_2d_only）
    route = route_from_quality(q, scale_known)

    summary = (
        f"scene={art.scene_name} method={art.recon_method} route={route} "
        f"overall_quality={q.overall_quality:.3f} scale_known={scale_known} "
        f"coverage={coverage_overall if np.isfinite(coverage_overall) else 'NA'}"
    )

    return SceneState(
        artifact_ref=artifact_ref or art.artifact_id,
        route=route,  # type: ignore[arg-type]
        frame="world",
        scale_known=scale_known,
        objects=[],
        summary=summary,
    )
