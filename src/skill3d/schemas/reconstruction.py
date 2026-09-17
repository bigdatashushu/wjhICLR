"""§5.2 重建 Schema。"""

from typing import Literal, Optional

from . import Spec


class ConfidenceMap(Spec):
    per_point_confidence: str  # 指向数组文件 ref
    coverage_count_per_frame: str


class CoverageMap(Spec):
    bbox_coverage_ratio: dict  # object_id -> 覆盖率
    overall: float


class QualityMetrics(Spec):
    """G1-G11 重建/输入质量指标（§10，阈值全部 TODO_CALIBRATE）。"""

    g1_blur_ok: float
    g2_brightness: float
    g3_motion_blur: float
    g4_frame_count: int
    g5_reproj_err_median: float
    g5_reproj_err_p95: float
    g6_depth_var_coeff: float
    g7_dynamic_ratio: float
    g8_bbox_coverage_min: float
    g9_tracker_consistency: float
    g10_baseline_quality: float
    g11_scale_ci: float
    overall_quality: float


class ReconstructionArtifact(Spec):
    artifact_id: str
    artifact_version: str  # 内容寻址哈希
    scene_name: str
    recon_method: Literal["vggt", "dust3r_mast3r", "colmap"]
    c2w_list: str  # 世界坐标 SE(3) 序列 ref
    intrinsics: str
    depth_maps: str
    point_map: str
    point_conf: str
    track_list: Optional[str]
    metric_scale: Optional[float]
    scale_known: bool
    quality: QualityMetrics
    confidence: ConfidenceMap


class SceneState(Spec):
    artifact_ref: str
    route: Literal["full_3d", "fallback_2d_only", "unanswerable"]
    frame: Literal["world", "camera", "image"]
    scale_known: bool
    objects: list[str]  # ObjectInstance id ref
    summary: str  # 给 Synthesizer 的场景摘要


class ObjectInstance(Spec):
    """M5 对象绑定/分割输出（§4 M5 字段 6）。"""

    instance_id: str
    class_hint: str
    mask_per_frame: str  # mask 数组文件 ref
    pointcloud_world: str  # 世界坐标点云 ref
    centroid_world: list[float]
    bbox: list[float]
    confidence: float
