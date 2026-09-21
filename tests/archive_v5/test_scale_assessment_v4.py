"""v4 HC30/HC31/HC33 单测：置信度派生、多锚点鲁棒融合、逐题型授权。

对应《系统架构4.md》：
- HC30 尺度置信度必须由「已触发锚点 × 锚点冲突 × 经验校准覆盖率 × 逐题型授权」
  共同派生；未标定/口径异常/冲突/非有限值一律 low；
- HC31 多锚点鲁棒融合 + 每个锚点留证（来源/估计/不确定性/残差/接受状态/原因码）；
- HC33 尺度按题型授权，`low` 只收回米制 Tool，不得连累非米制 3D 能力。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from skill3d.reconstruction.metric_scale import (
    CONF_HIGH_MAX_REL_CI,
    ScaleAnchor,
    anchor_evidence_of,
    fuse_scale_anchors_robust,
)
from skill3d.reconstruction.scale_assessment import (
    assess_scale,
    grade_confidence_v4,
    pre_authorized_metric_tasks,
)
from skill3d.reconstruction.scale_calibration import (
    build_scene_id_audit,
    fit_conformal_calibrator,
)


# --------------------------------------------------------------- 夹具 ----

@dataclass
class _Obj:
    instance_id: str
    class_hint: str
    pointcloud_world: str = ""


def _c2w(centers: np.ndarray) -> np.ndarray:
    r = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    out = np.zeros((len(centers), 4, 4))
    for i, c in enumerate(centers):
        out[i, :3, :3] = r
        out[i, :3, 3] = c
        out[i, 3, 3] = 1.0
    return out


def _floor(scale: float, n: int = 4000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = 4.0 / scale
    xy = rng.uniform(-half, half, size=(n, 2))
    pts = np.stack([xy[:, 0], np.zeros(n), xy[:, 1]], axis=1)
    return pts + rng.normal(0, 0.004 / scale, size=pts.shape)


def _cameras(scale: float, n: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = 1.5 / scale
    zs = np.linspace(-2.0 / scale, 2.0 / scale, n)
    return _c2w(np.stack([rng.normal(0, 0.05 / scale, n), np.full(n, h), zs], axis=1))


def _box(scale: float, size_m: float, at) -> np.ndarray:
    rng = np.random.default_rng(1)
    half = size_m / 2.0 / scale
    return rng.uniform(-half, half, size=(600, 3)) + np.asarray(at) / scale


def _calibrator(n: int = 80, *, confidence_level: float = 0.90, rel_ci_base: float = 0.02,
                seed: int = 0):
    audit = build_scene_id_audit([f"cal_{i}" for i in range(n)],
                                 excluded_scene_ids=["41069025"])
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(n):
        err = float(np.exp(rng.normal(0.0, 0.03)))
        recs.append({"scale_pred": err, "scale_true": 1.0,
                     "rel_ci": rel_ci_base + 0.4 * abs(np.log(err)),
                     "plane_identity_ok": True, "anchor_fired": True})
    return fit_conformal_calibrator(recs, confidence_level=confidence_level,
                                    split_audit=audit, calibration_id="test-cal")


def _anchor(name: str, ratio: float, *, weight: float = 1.0,
            kind: str = "object_prior", rel_sigma: float = 0.03) -> ScaleAnchor:
    return ScaleAnchor(kind=kind, name=name, measured=1.0, prior_m=ratio, ratio=ratio,
                       weight=weight, rel_sigma=rel_sigma)


# ---------------------------------------------------- HC31 鲁棒融合 ----

def test_single_outlier_does_not_pull_fusion():
    """L1：注入单个离群锚点，鲁棒融合不被拉偏（§10.2）。"""
    anchors = [_anchor("door:extent_up", 2.0),
               _anchor("table:top_height", 2.0, weight=200.0),
               _anchor("ground_plane_camera_height", 2.0,
                       kind="ground_plane_camera_height"),
               _anchor("chair:extent_up", 8.0, weight=5.0)]
    f = fuse_scale_anchors_robust(anchors)
    assert f.scale == pytest.approx(2.0, rel=0.05)
    assert not f.conflict                       # 单个小权重离群不误报冲突
    assert f.reason_codes["chair:extent_up"] == "outlier"
    assert f.n_accepted == 3


def test_heavy_minority_cannot_beat_majority():
    """先验很窄的单个锚点（权重极大）不得赢过两个一致锚点。"""
    f = fuse_scale_anchors_robust([
        _anchor("door:extent_up", 2.0, weight=188.0),
        _anchor("ground_plane_camera_height", 2.0, weight=1.0,
                kind="ground_plane_camera_height"),
        _anchor("table:top_height", 3.4, weight=225.0)])
    assert f.scale == pytest.approx(2.0, rel=0.05)
    assert f.conflict is True                   # 被拒方权重占比超阈 → 冲突


def test_majority_conflict_flags_conflict_and_low():
    """多数锚点冲突（§10.2 L1）→ conflict=True 且置信度必须 low。"""
    f = fuse_scale_anchors_robust([_anchor("door:extent_up", 2.0, weight=100.0),
                                   _anchor("table:top_height", 4.0, weight=100.0)])
    assert f.conflict and f.conflict_ratio == pytest.approx(2.0)
    conf, reasons = grade_confidence_v4(
        n_accepted=3, ci_rel=0.02, conflict=True, has_plane=True,
        has_object_anchor=True, calibration=_calibrator())
    assert conf == "low" and any("冲突" in r for r in reasons)


def test_anchor_evidence_records_full_fields():
    """HC31：每个锚点必须留下来源/估计/不确定性/残差/接受状态/原因码。"""
    anchors = [_anchor("door:extent_up", 2.0),
               _anchor("table:top_height", 2.0, weight=100.0),
               _anchor("chair:extent_up", 6.0, weight=1.0)]
    f = fuse_scale_anchors_robust(anchors)
    ev = anchor_evidence_of(anchors, f)
    assert len(ev) == 3
    by_name = {e.anchor_name: e for e in ev}
    assert by_name["door:extent_up"].anchor_type == "door"
    assert by_name["table:top_height"].anchor_type == "table"
    assert by_name["chair:extent_up"].anchor_type == "chair"
    d = by_name["door:extent_up"]
    assert d.accepted and d.reason_code == "ok"
    assert np.isfinite(d.scale_estimate) and np.isfinite(d.ci_rel)
    assert np.isfinite(d.residual)
    out = by_name["chair:extent_up"]
    assert not out.accepted and out.reason_code == "outlier"


def test_ground_plane_anchor_maps_to_camera_height_floor_type():
    a = _anchor("ground_plane_camera_height", 1.7,
                kind="ground_plane_camera_height")
    ev = anchor_evidence_of([a], fuse_scale_anchors_robust([a]))
    assert ev[0].anchor_type == "camera_height_floor"


# ------------------------------------------------- HC30 置信度派生 ----

def test_uncalibrated_never_leaves_low():
    """HC30：**未标定一律 low**，无论锚点多少个、CI 多小。"""
    for n_acc in (1, 3, 10):
        conf, reasons = grade_confidence_v4(
            n_accepted=n_acc, ci_rel=0.001, conflict=False, has_plane=True,
            has_object_anchor=True, calibration=None)
        assert conf == "low", (n_acc, reasons)
        assert any("未获经验校准支撑" in r for r in reasons)


def test_calibrated_high_requires_plane_object_and_tight_ci():
    cal = _calibrator()
    high, _ = grade_confidence_v4(
        n_accepted=3, ci_rel=CONF_HIGH_MAX_REL_CI / 2, conflict=False,
        has_plane=True, has_object_anchor=True, calibration=cal)
    assert high == "high"
    # 缺地平面或物体锚点 → 不得 high（v2 口径保留：high 需两者都有）
    no_plane, _ = grade_confidence_v4(
        n_accepted=3, ci_rel=0.01, conflict=False, has_plane=False,
        has_object_anchor=True, calibration=cal)
    assert no_plane != "high"
    # 只有地平面 + 标定通过 → 最高只到 medium（HC30：不得仅凭单锚点提升为 high）
    only_plane, _ = grade_confidence_v4(
        n_accepted=2, ci_rel=0.10, conflict=False, has_plane=True,
        has_object_anchor=False, calibration=cal)
    assert only_plane == "medium"


def test_bad_ci_unit_forces_low_even_when_calibrated():
    """HC29+HC30：口径异常（百分数/全宽/NaN）→ low，校准器也救不回来。"""
    cal = _calibrator()
    for bad in (21.875, float("nan"), float("inf"), -0.1, None):
        conf, reasons = grade_confidence_v4(
            n_accepted=5, ci_rel=bad, conflict=False, has_plane=True,
            has_object_anchor=True, calibration=cal)
        assert conf == "low", (bad, reasons)


def test_wide_ci_and_insufficient_anchors_force_low():
    cal = _calibrator()
    wide, _ = grade_confidence_v4(n_accepted=4, ci_rel=0.9, conflict=False,
                                  has_plane=True, has_object_anchor=True,
                                  calibration=cal)
    assert wide == "low"
    few, _ = grade_confidence_v4(n_accepted=1, ci_rel=0.01, conflict=False,
                                has_plane=True, has_object_anchor=True,
                                calibration=cal)
    assert few == "low"


def test_calibration_confidence_level_mismatch_forces_low():
    """§4.1「必须与校准器一致」+ HC29：不同置信水平不得混写。

    校准器按 0.99 标定、本次运行按 0.90 报 CI → 必须 low（否则论文里的
    coverage 声明与区间口径对不上）。
    """
    cal99 = _calibrator(80, confidence_level=0.99)
    conf, reasons = grade_confidence_v4(
        n_accepted=3, ci_rel=0.01, conflict=False, has_plane=True,
        has_object_anchor=True, calibration=cal99, confidence_level=0.90)
    assert conf == "low"
    assert any("不一致" in r for r in reasons)
    # 一致时同一校准器可支撑 high
    conf_ok, _ = grade_confidence_v4(
        n_accepted=3, ci_rel=0.01, conflict=False, has_plane=True,
        has_object_anchor=True, calibration=cal99, confidence_level=0.99)
    assert conf_ok == "high"


def test_calibration_without_coverage_support_cannot_raise():
    """经验覆盖与名义值差太远（同一置信水平下）→ 不得支撑 medium/high。"""
    cal = _calibrator(80, confidence_level=0.90)
    # 人为把经验覆盖改差（模拟"名义 90% 实际只覆盖 50%"）
    bad = cal.model_copy(update={"empirical_coverage": 0.5}) \
        if hasattr(cal, "model_copy") else cal
    object.__setattr__(bad, "empirical_coverage", 0.5)
    conf, reasons = grade_confidence_v4(
        n_accepted=3, ci_rel=0.01, conflict=False, has_plane=True,
        has_object_anchor=True, calibration=bad, confidence_level=0.90)
    assert conf == "low"
    assert any("覆盖" in r for r in reasons)


# --------------------------------------------- HC33 逐题型授权 ----

def test_pre_authorized_metric_tasks_by_confidence():
    assert pre_authorized_metric_tasks("low", ci_rel=0.01) == frozenset()
    med = pre_authorized_metric_tasks("medium", ci_rel=0.10)
    assert med == {"object_size_estimation", "room_size_estimation"}
    high = pre_authorized_metric_tasks("high", ci_rel=0.02)
    assert high == {"object_abs_distance", "object_size_estimation",
                    "room_size_estimation"}
    # medium 必须携带有限区间（v4 明文）
    assert pre_authorized_metric_tasks("medium", ci_rel=None) == frozenset()
    assert pre_authorized_metric_tasks("high", ci_rel=float("nan")) == frozenset()


def test_assess_scale_current_reality_is_low():
    """当前实况（无标定数据）：即使锚点齐全、CI 很小，也必须 low 且收回米制题型。"""
    scale = 1.7
    objs = [_Obj("o_door", "door"), _Obj("o_table", "table")]
    pts = {"o_door": _box(scale, 2.05, (1.0, 1.025, 0.0)),
           "o_table": _box(scale, 0.1, (-1.0, 0.75, 0.5))}
    a = assess_scale(_floor(scale), _cameras(scale), objects=objs, object_points=pts)
    assert a.scale_known and a.metric_scale is not None
    assert a.confidence == "low"
    assert a.allowed_metric_tasks == frozenset()
    assert a.calibration_id is None
    assert any("未标定" in n or "未获经验校准支撑" in n for n in a.notes)
    # 锚点证据仍然完整（供审计与论文报告）
    assert len(a.anchors) >= 2
    assert any(e.accepted for e in a.anchors)


def test_assess_scale_calibrated_authorizes_tasks():
    scale = 1.7
    objs = [_Obj("o_door", "door"), _Obj("o_table", "table")]
    pts = {"o_door": _box(scale, 2.05, (1.0, 1.025, 0.0)),
           "o_table": _box(scale, 0.1, (-1.0, 0.75, 0.5))}
    a = assess_scale(_floor(scale), _cameras(scale), objects=objs, object_points=pts,
                     calibration=_calibrator())
    assert a.confidence in ("medium", "high")
    assert "object_size_estimation" in a.allowed_metric_tasks
    assert a.calibration_id == "test-cal"
    assert a.ci_rel is not None and np.isfinite(a.ci_rel)
    # HC29：abs = scale × rel 必须自洽
    assert a.ci_abs_m == pytest.approx(a.metric_scale * a.ci_rel, rel=1e-9)
    assert a.empirical_coverage is not None


def test_assess_scale_conflict_forces_low_with_calibration():
    """锚点冲突时，即使有校准器也必须 low（§3 M3：不得返回虚假的 medium）。"""
    scale = 1.7
    objs = [_Obj("o_door", "door"), _Obj("o_table", "table")]
    pts = {"o_door": _box(scale, 2.05, (1.0, 1.025, 0.0)),
           # 桌子点云缩小 2 倍 → 与门锚点尺度差 2× → 冲突
           "o_table": _box(scale, 0.1, (-1.0, 0.75, 0.5)) / 2.0}
    a = assess_scale(_floor(scale), _cameras(scale), objects=objs, object_points=pts,
                     calibration=_calibrator())
    assert a.conflict
    assert a.confidence == "low" and a.allowed_metric_tasks == frozenset()


def test_assess_scale_fixed_scale_ablation_never_authorizes():
    """消融档固定尺度：无锚点证据 → 不得授权任何米制题型（HC30）。"""
    a = assess_scale(None, None, fixed_scale=1.0, calibration=_calibrator())
    assert a.scale_known and a.confidence == "low"
    assert a.allowed_metric_tasks == frozenset()
    assert "fixed_scale_ablation_no_anchor_evidence" in a.reason_codes


def test_assess_scale_missing_calibration_path_falls_back_to_low(tmp_path):
    """校准器路径不存在 → 不抛异常，降级 low（fail-closed，不静默放行）。"""
    a = assess_scale(_floor(1.7), _cameras(1.7),
                     calibration_path=str(tmp_path / "nope.json"))
    assert a.confidence == "low" and a.allowed_metric_tasks == frozenset()
    assert "calibration_unavailable" in a.reason_codes


def test_apply_scale_assessment_writes_consistent_fields():
    from skill3d.schemas import ReconstructionArtifact
    from skill3d.reconstruction.scale_assessment import apply_scale_assessment

    art = ReconstructionArtifact(
        artifact_id="a", artifact_version="v", scene_name="s", recon_method="vggt",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, metric_scale=None, scale_known=False,
        confidence={"per_point_confidence": "", "coverage_count_per_frame": ""})
    a = assess_scale(_floor(1.7), _cameras(1.7), calibration=_calibrator())
    out = apply_scale_assessment(art, a)
    assert out.scale_ci_rel == pytest.approx(a.ci_rel)
    assert out.scale_ci_abs_m == pytest.approx(a.metric_scale * a.ci_rel, rel=1e-9)
    assert out.allowed_metric_tasks == set(a.allowed_metric_tasks)
    assert out.scale_calibration_id == "test-cal"
    assert len(out.scale_anchor_fired) == len(a.anchors)
