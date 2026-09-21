"""G-11/G-12 尺度锚定单测（§7.1 G-11 验收：尺度误差 <15%、无锚点降级 low）。

用**已知真值尺度**的合成场景验证：地平面+相机高先验、标准物体尺寸先验、
多锚点鲁棒融合、MAD CI、无锚点降级、固定尺度消融档。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from skill3d.reconstruction.metric_scale import (
    CAMERA_HEIGHT_PRIOR_M,
    MIN_REL_UNCERTAINTY,
    ScaleAnchor,
    anchor_metric_scale,
    apply_scale,
    camera_up_hint,
    fit_scale_robust,
    fuse_scale_anchors,
    gravity_from_planes,
    known_object_priors,
    normalize_class_hint,
    object_anchor,
    ransac_plane,
)

# 真值尺度（m / rel-unit）候选；验收要求 <15%（TODO_CALIBRATE）
TRUE_SCALES = (0.8, 1.0, 1.7, 2.5)
SCALE_TOL = 0.15


# ------------------------------------------------------------------ 合成场景 ----

def _camera_up_world() -> np.ndarray:
    """相机 up 轴（世界 +y）；OpenCV 约定 y_cam 向下 → up = -y_cam。"""
    return np.array([0.0, 1.0, 0.0])


def _c2w(centers: np.ndarray) -> np.ndarray:
    """构造 c2w 序列：相机水平放置、up 为世界 +y（det=+1 的右手系）。"""
    r = np.array([[-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, 1.0]])
    out = np.zeros((len(centers), 4, 4))
    for i, c in enumerate(centers):
        out[i, :3, :3] = r
        out[i, :3, 3] = c
        out[i, 3, 3] = 1.0
    return out


def _floor_points(scale: float, n: int = 4000, seed: int = 0) -> np.ndarray:
    """世界系地板点云（相对单位）：真实高度 0，横向 ±4m/scale。"""
    rng = np.random.default_rng(seed)
    half = 4.0 / scale
    xy = rng.uniform(-half, half, size=(n, 2))
    pts = np.stack([xy[:, 0], np.zeros(n), xy[:, 1]], axis=1)
    pts += rng.normal(0, 0.004 / scale, size=pts.shape)   # 平面噪声
    return pts


def _cameras(scale: float, n: int = 8, seed: int = 0) -> np.ndarray:
    """手持相机轨迹：高度约 1.5m（真值），世界系相对单位 = 1.5/scale。"""
    rng = np.random.default_rng(seed)
    h = CAMERA_HEIGHT_PRIOR_M / scale
    zs = np.linspace(-2.0 / scale, 2.0 / scale, n)
    centers = np.stack([rng.normal(0, 0.05 / scale, n), np.full(n, h), zs], axis=1)
    return _c2w(centers)


@dataclass
class _FakeObject:
    instance_id: str
    class_hint: str
    pointcloud_world: str = ""
    bbox: list = None  # type: ignore[assignment]


def _box_object(scale: float, kind: str, size_m: float, at) -> np.ndarray:
    """在世界系放置一个边长 size_m 的立方体点云（相对单位）。"""
    rng = np.random.default_rng(1)
    half = size_m / 2.0 / scale
    pts = rng.uniform(-half, half, size=(600, 3))
    pts += np.asarray(at, dtype=np.float64) / scale
    return pts


# ------------------------------------------------------------------ RANSAC 平面 ----

def test_ransac_plane_recovers_horizontal_floor():
    pts = _floor_points(scale=1.0)
    normal, offset, mask = ransac_plane(pts, seed=0, min_inliers=500)
    assert normal is not None and offset is not None
    assert abs(abs(float(normal @ _camera_up_world())) - 1.0) < 1e-3  # 法向 = ±up
    assert mask.sum() > 3500


def test_ransac_plane_rejects_plane_off_the_up_axis():
    """竖直墙面点云：法向与 up 轴垂直 → 在 up 约束下不应被当地平面。"""
    rng = np.random.default_rng(0)
    wall = np.stack([rng.uniform(-2, 2, 3000), rng.uniform(0, 3, 3000),
                     np.zeros(3000)], axis=1)
    normal, _, _ = ransac_plane(wall, seed=0, min_inliers=500,
                                normal_hint=_camera_up_world())
    assert normal is None  # 墙不满足地平面法向约束


def test_ransac_plane_insufficient_points():
    normal, offset, mask = ransac_plane(np.zeros((10, 3)), seed=0, min_inliers=500)
    assert normal is None and offset is None and mask.size == 0


def test_camera_up_hint_and_gravity_average():
    up = camera_up_hint(_cameras(1.0))
    assert up is not None and np.allclose(up, _camera_up_world(), atol=1e-9)
    g = gravity_from_planes([_camera_up_world(), _camera_up_world()])
    assert g is not None and np.allclose(g, _camera_up_world(), atol=1e-9)
    assert gravity_from_planes([None]) is None


# ------------------------------------------------------------------ 地平面锚 ----

@pytest.mark.parametrize("scale", TRUE_SCALES)
def test_ground_plane_camera_height_anchor_recovers_scale(scale):
    """地平面 + 相机高先验在合成场景上恢复真值尺度（<15%，§7.1 G-11 验收）。"""
    est = anchor_metric_scale(_floor_points(scale), _cameras(scale), seed=0)
    assert est.scale_known, est.summary()
    assert abs(est.scale / scale - 1.0) < SCALE_TOL, est.summary()
    assert est.scale_ci is not None and est.scale_ci > 0
    # 仅地平面锚点（无标准物体）→ 不确定性大 → low（§7.1 验收口径）
    assert est.scale_confidence == "low", est.summary()


def test_ground_plane_anchor_skipped_without_pose():
    est = anchor_metric_scale(_floor_points(1.0), None)
    assert not est.scale_known and est.scale_confidence == "none"
    assert any("位姿" in n for n in est.notes)


# ------------------------------------------------------------------ 物体先验 ----

def test_known_object_priors_and_aliases():
    assert normalize_class_hint(" Tables ") == "table"
    assert normalize_class_hint("chairs") == "chair"
    assert known_object_priors("door")  # 门高先验存在
    assert known_object_priors("spaceship") == []   # 非标准物体不臆造尺寸


@pytest.mark.parametrize("scale", TRUE_SCALES)
def test_object_anchor_door_height_recovers_scale(scale):
    """门高 2.05m 先验 → 单个对象锚点即可恢复尺度（<15%）。"""
    door = _box_object(scale, "door", size_m=2.05, at=(1.0, 1.025, 0.0))
    anchors, notes = object_anchor(door, "door", up_axis=_camera_up_world(),
                                   plane=(_camera_up_world(), 0.0), instance_id="o1")
    assert anchors, notes
    ratio = anchors[0].ratio
    assert abs(ratio / scale - 1.0) < SCALE_TOL, (ratio, scale)


def test_object_anchor_skips_unknown_class_and_small_cloud():
    anchors, notes = object_anchor(np.zeros((600, 3)), "spaceship")
    assert anchors == [] and any("非标准物体" in n for n in notes)
    anchors, notes = object_anchor(np.zeros((5, 3)), "door")
    assert anchors == [] and any("点数不足" in n for n in notes)
    anchors, notes = object_anchor(None, "door")
    assert anchors == [] and any("无点云" in n for n in notes)


def test_object_anchor_top_height_uses_ground_plane():
    """桌面高 0.75m 先验：需要地面基准才能测 top_height。"""
    scale = 1.0
    table = _box_object(scale, "table", size_m=0.1, at=(0.5, 0.75, 0.5))
    with_plane, _ = object_anchor(table, "table", up_axis=_camera_up_world(),
                                  plane=(_camera_up_world(), 0.0))
    without, _ = object_anchor(table, "table", up_axis=_camera_up_world(), plane=None)
    kinds_with = {a.name.split(":")[1] for a in with_plane}
    kinds_without = {a.name.split(":")[1] for a in without}
    assert "top_height" in kinds_with          # 有地面 → 可测桌面高
    assert "top_height" not in kinds_without   # 无地面 → 该维度跳过


# ------------------------------------------------------------------ 融合 / CI ----

def test_fit_scale_robust_weighted_median_and_ci_floor():
    rel = np.array([1.0, 1.0, 1.0, 1.0])
    metric = np.array([2.0, 2.0, 2.0, 2.0])
    s, ci = fit_scale_robust(rel, metric, min_samples=1)
    assert s == pytest.approx(2.0)
    assert ci >= MIN_REL_UNCERTAINTY          # 完全一致也不给 CI=0（防过度自信）

    # 一个离群锚点被加权中位数压住
    rel2 = np.array([1.0, 1.0, 1.0, 1.0])
    metric2 = np.array([2.0, 2.0, 2.0, 20.0])
    s2, _ = fit_scale_robust(rel2, metric2, weights=np.array([1.0, 1.0, 1.0, 0.1]),
                             min_samples=1)
    assert s2 == pytest.approx(2.0)


def test_fit_scale_robust_needs_min_samples():
    s, ci = fit_scale_robust(np.ones(3), np.ones(3) * 2, min_samples=100)
    assert np.isnan(s) and ci == float("inf")


def test_fuse_scale_anchors_high_confidence_with_many_consistent_anchors():
    anchors = [ScaleAnchor(kind="object_prior", name=f"a{i}", measured=1.0,
                           prior_m=2.0, ratio=2.0, weight=1.0, rel_sigma=0.01)
               for i in range(4)]
    scale, rel_ci, ci_m = fuse_scale_anchors(anchors)
    assert scale == pytest.approx(2.0)
    assert rel_ci <= 0.05 and ci_m == pytest.approx(rel_ci * 2.0, rel=1e-6)


def test_fuse_scale_anchors_no_usable_anchor():
    bad = [ScaleAnchor(kind="object_prior", name="x", measured=0.0, prior_m=1.0,
                       ratio=0.0, weight=1.0, rel_sigma=0.1)]
    scale, rel_ci, ci_m = fuse_scale_anchors(bad)
    assert scale is None and rel_ci == float("inf") and ci_m == float("inf")


def test_fuse_drops_high_leverage_outlier():
    """三锚点中一个严重偏离 → 剔除后尺度回到多数派（鲁棒性）。"""
    anchors = [
        ScaleAnchor(kind="object_prior", name="a", measured=1.0, prior_m=2.0,
                    ratio=2.0, weight=1.0, rel_sigma=0.02),
        ScaleAnchor(kind="object_prior", name="b", measured=1.0, prior_m=2.0,
                    ratio=2.0, weight=1.0, rel_sigma=0.02),
        ScaleAnchor(kind="object_prior", name="c", measured=1.0, prior_m=8.0,
                    ratio=8.0, weight=1.0, rel_sigma=0.02),
    ]
    scale, _, _ = fuse_scale_anchors(anchors)
    assert abs(scale - 2.0) < 0.2, scale


# ------------------------------------------------------------------ 端到端 ----

def test_anchor_metric_scale_multi_anchor_scale_is_accurate():
    """地平面 + 2 个标准物体且一致 → 融合尺度准确、锚点证据完整。

    **v4 口径变更（HC30）**：本函数测的是**锚点提取与融合精度**；`scale_confidence`
    的准入口径已迁到 `assess_scale`（`tests/unit/test_scale_assessment_v4.py`）：
    未加载冻结校准器时一律 low，**不再**因为"锚点多、CI 小"就给 medium/high
    （"尺度置信度不得靠常量或放宽门槛提升"）。
    """
    scale = 1.7
    pts = _floor_points(scale)
    cam = _cameras(scale)
    door = _box_object(scale, "door", 2.05, at=(1.0, 1.025, 0.0))
    table = _box_object(scale, "table", 0.1, at=(-1.0, 0.75, 0.5))
    objs = [_FakeObject("o_door", "door"), _FakeObject("o_table", "table")]
    est = anchor_metric_scale(pts, cam, objects=objs,
                             object_points={"o_door": door, "o_table": table})
    assert est.scale_known
    assert abs(est.scale / scale - 1.0) < SCALE_TOL, est.summary()
    assert est.method == "known_object_prior+ground_plane_camera_height"
    assert est.scale_ci is not None
    # 锚点覆盖地平面 + 标准物体（v4 HC31：多种锚点来源都触发了）
    kinds = {a.kind for a in est.anchors}
    assert "ground_plane_camera_height" in kinds and "object_prior" in kinds


def test_anchor_metric_scale_multi_anchor_without_calibration_stays_low_v4():
    """HC30：无冻结校准器时，即使多锚点一致也不得给 medium/high。"""
    from skill3d.reconstruction.scale_assessment import assess_scale

    scale = 1.7
    door = _box_object(scale, "door", 2.05, at=(1.0, 1.025, 0.0))
    table = _box_object(scale, "table", 0.1, at=(-1.0, 0.75, 0.5))
    objs = [_FakeObject("o_door", "door"), _FakeObject("o_table", "table")]
    a = assess_scale(_floor_points(scale), _cameras(scale), objects=objs,
                     object_points={"o_door": door, "o_table": table})
    assert a.scale_known and abs(a.metric_scale / scale - 1.0) < SCALE_TOL
    assert a.confidence == "low"
    assert a.allowed_metric_tasks == frozenset()


def test_anchor_metric_scale_no_anchor_is_explicit_failure():
    """无点云/无位姿 → scale_known=False 且给出降级原因（不伪造尺度）。"""
    est = anchor_metric_scale(None, None)
    assert not est.scale_known
    assert est.scale is None and est.scale_ci is None
    assert est.scale_confidence == "none"
    assert any("无有效锚点" in n for n in est.notes)


def test_anchor_metric_scale_fixed_scale_ablation():
    """论文消融档：固定尺度 1.0（§7.1 G-11 对照）。"""
    est = anchor_metric_scale(None, None, fixed_scale=1.0)
    assert est.scale == 1.0 and est.scale_known
    assert est.method == "fixed_scale_ablation"
    assert est.scale_confidence == "low"      # 无锚定证据 → low
    assert any("固定尺度" in n for n in est.notes)


def test_apply_scale_uniform_and_rejects_bad_scale():
    pts = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(apply_scale(pts, 2.0), [[2.0, 4.0, 6.0]])
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            apply_scale(pts, bad)


def test_scale_estimate_summary_shapes():
    est = anchor_metric_scale(_floor_points(1.0), _cameras(1.0))
    assert "scale=" in est.summary() and "conf=" in est.summary()
    empty = anchor_metric_scale(None, None)
    assert "scale_unknown" in empty.summary()
