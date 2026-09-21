"""M4 质量门单测（v6 §10，D11）：主门 = warp 内点率 ∧ 分组点云重叠率 + 诊断项。

v6 与 v5 的口径差异（本文件的断言全部按 v6 写）：

- 质量是 **M4 一次算清的单一事实源**：`compute_quality(...)` 吃原始数组，产出
  `QualityMetrics`；不存在 v5 那种"先算一半、再由 M5 统计回写 G7/G9"的增量补写
  （`overall_from_metrics` / `compute_g1_g11` / `g5_reproj_err` 都已删除）；
- **主门不得单挑**（§10.1）：`warp_inlier_ratio` 与 `cloud_overlap_ratio` 必须都过，
  任一 NaN/Inf 都判 False 并进 `warnings`（fail-closed，但**不抛异常**）；
- **G5 永久 not_available**、**G8 永久退役**、**G11 随尺度路线废止**：
  这三个字段连"构造出来"都不允许（§5.2/§10.4/§20）。
"""

import numpy as np
import pytest
from pydantic import ValidationError

from skill3d.reconstruction_gate import m4_main_gate as mg
from skill3d.reconstruction_gate import quality_metrics as qm
from skill3d.reconstruction_gate.confidence_map import fuse_confidence
from skill3d.schemas import ConfidenceMap, QualityMetrics, ReconstructionArtifact

# 合成几何参数：正面平行平面（帧间同姿态）→ warp 残差恒 0、重叠率恒 1
_HW = (16, 24)
_N = 32
_DEPTH = 2.0


def _frames(n: int = _N, hw=_HW, seed: int = 0) -> list[np.ndarray]:
    """有纹理的正常曝光帧（G1/G2 满分）：同一世界贴色 → 帧间光度可比。

    用"同一张确定性纹理"而不是随机噪声：完美几何下同一 3D 点的颜色跨帧一致，
    光度 warp 才能真正判到内点（随机噪声的逐像素对齐率约 0）。
    """
    h, w = hw
    base = ((np.add.outer(np.arange(h), np.arange(w)) * 7 + seed) % 200 + 30).astype(np.uint8)
    return [np.repeat(base[..., None], 3, axis=2).copy() for _ in range(n)]


def _intrinsics(hw=_HW) -> np.ndarray:
    h, w = hw
    return np.array([[0.9 * w, 0.0, (w - 1) / 2.0],
                     [0.0, 0.9 * w, (h - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def _perfect_geometry(n: int = _N, hw=_HW, depth_val: float = _DEPTH):
    """最简"完美几何"：同一姿态拍一张正面平行平面（warp 与重叠都必然满分）。

    不是真场景，而是**可解析验证**的最小输入：深度图恒定 → 任意近邻对的相对深度
    差恒 0；点图在两半子云里逐点重合 → 双向 NN 距离恒 0。用它把"主门过"的
    正路径钉死（负路径由注入退化与坏值用例覆盖）。
    """
    h, w = hw
    K = _intrinsics(hw)
    depth = np.full((n, h, w), float(depth_val))
    c2w = np.tile(np.eye(4), (n, 1, 1))
    ks = np.tile(K, (n, 1, 1))
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (uu - K[0, 2]) / K[0, 0] * depth_val
    y = (vv - K[1, 2]) / K[1, 1] * depth_val
    plane = np.stack([x, y, np.full_like(x, depth_val)], axis=-1)
    point_map = np.tile(plane[None], (n, 1, 1, 1))
    return {"frames": _frames(n, hw), "depth_maps": depth, "c2w_list": c2w,
            "intrinsics": ks, "point_map": point_map}


# ------------------------------------------------------------------ 单项诊断 ----

def test_g6_depth_var_coeff_synthetic():
    """合成深度图验证 G6 σ/μ。"""
    # 恒定深度 2.0 → σ/μ = 0
    d_const = np.full((4, 32, 32), 2.0)
    assert qm.g6_depth_var_coeff(d_const) == 0.0

    # 一半 1.0 一半 3.0 → μ=2, σ=1 → σ/μ=0.5
    d = np.concatenate([np.full((2, 32, 32), 1.0), np.full((2, 32, 32), 3.0)])
    assert abs(qm.g6_depth_var_coeff(d) - 0.5) < 1e-9

    # 全零/NaN 深度 → NaN
    assert np.isnan(qm.g6_depth_var_coeff(np.zeros((2, 8, 8))))


def test_g1_g2_with_synthetic_frames():
    """清晰正常曝光帧 → G1/G2 满分；模糊帧 → G1 下降。"""
    rng = np.random.default_rng(0)
    sharp = [(rng.random((64, 64, 3)) * 200 + 30).astype(np.uint8) for _ in range(8)]
    assert qm.g1_blur_ok(sharp) == 1.0
    assert qm.g2_brightness(sharp) == 1.0

    import cv2
    blurred = [cv2.GaussianBlur(f, (25, 25), 0) for f in sharp]
    assert qm.g1_blur_ok(blurred) == 0.0

    # 全白帧 → 过曝 → G2 = 0
    white = [np.full((64, 64, 3), 255, dtype=np.uint8) for _ in range(8)]
    assert qm.g2_brightness(white) == 0.0


def test_g4_frame_count_comes_from_actual_frames():
    """G4 是**实到帧数**（v6 无独立 g4 生产者函数；统一 FrameSet 只能 32 帧）。"""
    geo = _perfect_geometry()
    q = qm.compute_quality(frames=geo["frames"], depth_maps=geo["depth_maps"])
    assert q.g4_frame_count == 32
    # 空帧列表 → 0（不得回填 32 这个"名义帧数"）
    q_empty = qm.compute_quality(frames=[], depth_maps=geo["depth_maps"])
    assert q_empty.g4_frame_count == 0
    # 完全不传 frames 时，从 artifact 的既有质量里继承（审计路径），无 artifact → 0
    q_no_frames = qm.compute_quality(depth_maps=geo["depth_maps"])
    assert q_no_frames.g4_frame_count == 0


def test_g10_baseline_quality():
    """合成圆周轨迹：基线/场景直径为确定值。"""
    n = 16
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles), np.zeros(n)], axis=1)
    c2w = np.repeat(np.eye(4)[None], n, axis=0)
    c2w[:, :3, 3] = centers
    v = qm.g10_baseline_quality(c2w)
    step = np.linalg.norm(centers[1] - centers[0])
    diameter = 2.0
    assert abs(v - step / diameter) < 1e-6
    assert np.isnan(qm.g10_baseline_quality(None))


# --------------------------------------------- v6 废止指标：连构造都不允许 ----

_RETIRED = ("g5_reproj_err_median", "g5_reproj_err_p95",
            "g8_bbox_coverage_min", "g11_scale_ci")


def _artifact(**over) -> ReconstructionArtifact:
    base = dict(artifact_id="a", artifact_version="v", scene_name="s",
                c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
                confidence=ConfidenceMap(per_point_confidence=""))
    base.update(over)
    return ReconstructionArtifact(**base)


def _quality_kwargs(**over) -> dict:
    base = dict(warp_inlier_ratio=0.9, warp_photometric_inlier_ratio=0.9,
                cloud_overlap_ratio=0.9, main_gate_passed=True,
                g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=0.0,
                g4_frame_count=32, g6_depth_var_coeff=0.1, g7_dynamic_ratio=0.0,
                g9_tracker_consistency=0.9, g10_baseline_quality=0.5,
                overall_quality=0.9)
    base.update(over)
    return base


@pytest.mark.parametrize("field", _RETIRED)
def test_retired_metrics_cannot_be_constructed(field):
    """G5 / G8 / G11 字段出现即 hard fail（§10.4/§20：严禁代理值冒充）。"""
    with pytest.raises(ValidationError):
        QualityMetrics(**_quality_kwargs(**{field: float("nan")}))


def test_no_retired_metric_producer_exists():
    """v5 的 G5/G8/G11 生产者函数与总体聚合助手都已删除（不得"顺手加回来"）。"""
    for gone in ("g5_reproj_err", "overall_from_metrics", "compute_g1_g11",
                 "g8_bbox_coverage_min"):
        assert not hasattr(qm, gone), f"{gone} 属 v5 口径，v6 不得再存在"


def test_g5_is_permanently_not_available_on_artifact():
    """G5 在 artifact 上固定 `not_available` + `None`，且不允许被写成 computed。"""
    art = _artifact()
    assert art.reprojection_status == "not_available"
    assert art.g5_reproj_err_median is None and art.g5_reproj_err_p95 is None
    with pytest.raises(ValidationError):
        _artifact(reprojection_status="computed")     # BA 已关闭：只允许 not_available
    with pytest.raises(ValidationError):
        _artifact(g5_reproj_err_median=1.0)           # 代理值同样写不进来


def test_diagnostics_are_nan_without_data_and_measured_with_data():
    """G7/G9 是**输入即测**的诊断项：没数据记 NaN（不是 0），给了就实算。"""
    geo = _perfect_geometry()
    q_plain = qm.compute_quality(**geo)
    assert np.isnan(q_plain.g7_dynamic_ratio)
    assert np.isnan(q_plain.g9_tracker_consistency)

    masks = np.zeros((_N, 4, 4), dtype=bool)
    masks[:8] = True                                  # 动态占比 1/4
    q_m5 = qm.compute_quality(**geo, dynamic_masks=masks, track_ious=[0.9, 0.8])
    assert q_m5.g7_dynamic_ratio == pytest.approx(0.25)
    assert q_m5.g9_tracker_consistency == pytest.approx(0.85)


# --------------------------------------------------------------- 主门（§10.1）----

def test_main_gate_is_and_of_two_submetrics():
    """主门 = warp ∧ 重叠（多指标不得单挑）：任一单项满分都不算过。"""
    both_ok = mg.main_gate({"warp_inlier_ratio": 0.9, "cloud_overlap_ratio": 0.9})
    assert both_ok["main_gate_passed"] is True
    assert both_ok["sub_results"] == {"warp_inlier_ratio": True,
                                      "cloud_overlap_ratio": True}

    # warp 满分、重叠不过 → 必须 False（SysCON3D：自洽幻觉能骗过单项）
    warp_only = mg.main_gate({"warp_inlier_ratio": 0.99, "cloud_overlap_ratio": 0.05})
    assert warp_only["main_gate_passed"] is False
    assert warp_only["sub_results"] == {"warp_inlier_ratio": True,
                                        "cloud_overlap_ratio": False}

    # 重叠满分、warp 不过 → 同样必须 False
    cloud_only = mg.main_gate({"warp_inlier_ratio": 0.05, "cloud_overlap_ratio": 0.99})
    assert cloud_only["main_gate_passed"] is False
    assert cloud_only["sub_results"] == {"warp_inlier_ratio": False,
                                         "cloud_overlap_ratio": True}


def test_main_gate_fails_on_nan_inf_depth_without_exception():
    """NaN/Inf 深度 → 主门不过 + warnings 非空，且**不抛异常**（坏输入 fail-closed）。"""
    geo = _perfect_geometry()
    for bad in (np.nan, np.inf, -np.inf):
        depth = np.full((_N, _HW[0], _HW[1]), bad)
        gate = mg.main_gate({"frames": geo["frames"], "depth_maps": depth,
                             "c2w_list": geo["c2w_list"], "intrinsics": geo["intrinsics"],
                             "point_map": np.full((_N, _HW[0], _HW[1], 3), bad)})
        assert gate["main_gate_passed"] is False
        assert gate["warnings"]
        assert np.isnan(gate["values"]["warp_inlier_ratio"])
        assert np.isnan(gate["values"]["cloud_overlap_ratio"])

    # 完全不给输入（空 dict）也不许抛：两类子项都记 NaN + 告警
    empty = mg.main_gate({})
    assert empty["main_gate_passed"] is False and len(empty["warnings"]) >= 2


def test_overall_quality_is_zero_when_main_gate_fails():
    """主门不过 → `overall_quality` 恒 0.0（路由据此 fail-closed 落 fallback）。"""
    geo = _perfect_geometry()
    nan_depth = np.full((_N, _HW[0], _HW[1]), np.nan)
    q = qm.compute_quality(frames=geo["frames"], depth_maps=nan_depth,
                           c2w_list=geo["c2w_list"], intrinsics=geo["intrinsics"],
                           point_map=np.full((_N, _HW[0], _HW[1], 3), np.nan))
    assert q.main_gate_passed is False
    assert q.overall_quality == 0.0
    assert q.diagnostic_warnings                     # 失败原因必须可审计

    # 什么都不给：同样 0.0（不得靠"少算几项"把均值抬起来）
    q_empty = qm.compute_quality()
    assert q_empty.main_gate_passed is False and q_empty.overall_quality == 0.0


def test_overall_quality_is_positive_when_main_gate_passes():
    """主门过 → 有限诊断项归一化均值（完美几何每一项都满分 → 1.0）。"""
    q = qm.compute_quality(**_perfect_geometry())
    assert q.main_gate_passed is True
    assert q.overall_quality == pytest.approx(1.0)


def test_compute_quality_reports_main_gate_metrics_not_nan():
    """主门实算的 warp/重叠比率必须写进 QualityMetrics（trace 的唯一事实源）。"""
    q = qm.compute_quality(**_perfect_geometry())
    assert q.warp_inlier_ratio == pytest.approx(1.0)
    assert q.warp_photometric_inlier_ratio == pytest.approx(1.0)
    assert q.cloud_overlap_ratio == pytest.approx(1.0)
    assert q.gate_thresholds == {"warp_inlier_ratio": mg.TH_WARP_INLIER,
                                 "cloud_overlap_ratio": mg.TH_CLOUD_OVERLAP}


def test_compute_quality_separates_degraded_geometry_per_submetric():
    """把一组子云打散（跨场景注入）→ warp 仍过、重叠塌 → 主门必须 False。"""
    geo = _perfect_geometry()
    point_map = geo["point_map"].copy()
    point_map[_N // 2:] += 100.0        # 后半帧的点云整体平移（两组子云不再重合）
    q = qm.compute_quality(**{**geo, "point_map": point_map})
    assert q.warp_inlier_ratio >= mg.TH_WARP_INLIER
    assert q.cloud_overlap_ratio < mg.TH_CLOUD_OVERLAP
    assert q.main_gate_passed is False
    assert q.overall_quality == 0.0


def test_conf_warp_monotonic_is_soft_weight_not_gate():
    """conf 只作软权重（§10.3）：不单调只降权、不否决主门；未自检 ≠ 自检没过。"""
    geo = _perfect_geometry()
    mono = mg.main_gate({"warp_inlier_ratio": 1.0, "cloud_overlap_ratio": 1.0,
                         "conf_warp_monotonic": True})
    broken = mg.main_gate({"warp_inlier_ratio": 1.0, "cloud_overlap_ratio": 1.0,
                           "conf_warp_monotonic": False})
    assert mono["main_gate_passed"] and broken["main_gate_passed"]
    assert mono["conf_weight"] == 1.0
    assert broken["conf_weight"] == mg.TH_CONF_DOWNWEIGHT
    assert any("conf_warp_monotonic" in w for w in broken["warnings"])

    # compute_quality 不替调用方跑 conf-warp 自检 → 记 None（"没做"既不放大也不当门）
    q = qm.compute_quality(**geo)
    assert q.conf_warp_monotonic is None and q.conf_warp_spearman is None
    assert q.main_gate_passed is True


# --------------------------------------------------------- M4 置信度退化压零 ----

def test_fuse_confidence_degenerate_zero():
    """退化区（覆盖不足 / 残差过大）压零（§4 M4 字段 11）。"""
    conf = np.full((10,), 0.9)
    cov = np.full((10,), 5)
    err = np.zeros(10)
    fused = fuse_confidence(conf, cov, err)
    assert np.all(fused > 0)

    # 覆盖为 0 的点压零
    cov2 = cov.copy(); cov2[3] = 0
    fused2 = fuse_confidence(conf, cov2, err)
    assert fused2[3] == 0.0

    # 重投影残差超阈压零
    err2 = err.copy(); err2[5] = 100.0
    fused3 = fuse_confidence(conf, cov, err2)
    assert fused3[5] == 0.0
