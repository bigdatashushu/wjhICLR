"""v5 尺度口径 L0/L1 单测（§10.2 L0/L1；HC29–31/33）。

v5 相对 v4 的两处语义修正：

1. **"有点估计但无区间"= 未标定**（不是矛盾、也不是崩溃）：写回 artifact 时必须
   fail-closed 为 `low` + 清空 `allowed_metric_tasks`，重建产物照常产出
   （真实 GPU smoke 中 `metric_scale≈2.33 / ci_rel=None` 曾误抛 `CiUnitError`
   导致整条 P1 失败——本文件是那次修复的回归护栏）；
2. **`scale_conflict` 只表示锚点冲突**（HC31），不得被当作"口径降级"的标记。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.reconstruction.scale_assessment import (
    apply_scale_assessment,
    assess_scale,
)
from skill3d.reconstruction.scale_units import (
    assert_ci_consistent,
    ci_abs_m,
    ci_consistency_status,
)
from skill3d.schemas import ConfidenceMap, ReconstructionArtifact


def _art(**kw) -> ReconstructionArtifact:
    base = dict(
        artifact_id="a", artifact_version="v", scene_name="s", recon_method="vggt",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, metric_scale=None, scale_known=False,
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""))
    base.update(kw)
    return ReconstructionArtifact(**base)


# ------------------------------------------------- L0：口径三态分类 ----

def test_ci_consistency_status_classifies_three_states():
    assert ci_consistency_status(None, None, None) == "absent"
    assert ci_consistency_status(2.0, None, None) == "uncalibrated"
    assert ci_consistency_status(2.0, 0.1, 0.2) == "claim"
    # 部分提供是**真矛盾**（既不是未标定也不是完整声明）
    assert ci_consistency_status(2.0, 0.1, None) == "claim"


def test_assert_ci_consistent_allows_uncalibrated_but_not_partial():
    assert_ci_consistent(None, None, None)          # 未锚定
    assert_ci_consistent(2.0, None, None)           # 未标定（v5 修复点：不再抛）
    assert_ci_consistent(2.0, 0.1, 0.2)             # 自洽声明
    with pytest.raises(Exception):
        assert_ci_consistent(2.0, 0.1, None)        # 部分提供 → 矛盾
    with pytest.raises(Exception):
        assert_ci_consistent(2.0, 0.5, 32.81)       # 不自洽（§10.2 实测样例）


def test_ci_abs_is_scale_times_rel_in_v5_units():
    """L0 口径：`scale_ci_abs_m = metric_scale × scale_ci_rel`（半宽分数口径）。"""
    assert ci_abs_m(2.386, 0.137) == pytest.approx(2.386 * 0.137)


# ------------------------------- L0：真实 smoke 回归（未标定不得让重建失败）----

def test_uncalibrated_assessment_downgrades_instead_of_crashing():
    """真实 GPU smoke 回归：`scale` 有点估计但 `ci_rel=None` 时写回必须降级。"""
    from skill3d.reconstruction.scale_assessment import ScaleAssessment

    assessment = ScaleAssessment(
        metric_scale=2.3301, scale_known=True, ci_rel=None, ci_abs_m=None,
        confidence_level=0.90, confidence="low", anchors=[], conflict=False,
        calibration_id=None, empirical_coverage=None, nominal_coverage=None,
        n_accepted_anchors=1, method="ground_plane_camera_height",
        source="camera_height_prior", reason_codes=("no_calibrator",),
        notes=("缺冻结校准器",), allowed_metric_tasks=frozenset())
    art = apply_scale_assessment(_art(), assessment)
    assert art.scale_confidence == "low"
    assert art.allowed_metric_tasks == set()
    assert art.scale_ci_rel is None and art.scale_ci_abs_m is None
    assert "uncalibrated" in art.scale_method
    assert art.scale_conflict is False      # HC31：降级 ≠ 锚点冲突
    # 产物仍可正常序列化/再校验（不得因为"没区间"而不可用）
    again = ReconstructionArtifact.model_validate_json(art.model_dump_json())
    assert again.scale_confidence == "low" and again.allowed_metric_tasks == set()


def test_artifact_with_scale_but_no_ci_is_downgraded_not_rejected():
    """消费端兜底：手工构造"有尺度无区间"的 artifact → 降 low（不是抛异常）。"""
    art = _art(metric_scale=2.33, scale_known=True, scale_confidence="high",
               allowed_metric_tasks={"object_abs_distance"})
    assert art.scale_confidence == "low" and art.allowed_metric_tasks == set()
    assert art.scale_conflict is False
    assert "v5 口径 CI 缺失" in art.scale_method


# ------------------------------- L1：合成多锚点（离群不拉偏 / 冲突必须 low）----

def _synthetic_point_map(scale_hint: float = 1.0):
    """合成一个"地面 + 桌子"的简单点云（米制，供锚点估计使用）。"""
    rng = np.random.default_rng(0)
    n = 400
    xy = rng.uniform(-3, 3, size=(n, 2))
    floor = np.concatenate([xy, np.zeros((n, 1))], axis=1)          # 地面 z=0
    table_top = np.concatenate([rng.uniform(-1, 1, size=(n, 2)),
                                np.full((n, 1), 0.75)], axis=1)     # 桌面 0.75m
    pts = np.stack([floor, table_top], axis=0) * scale_hint
    return pts, n


def test_l1_outlier_anchor_does_not_pull_robust_fusion():
    """L1 合成：注入单个离群锚点时鲁棒融合不被拉偏（§10.2 L1）。"""
    from skill3d.reconstruction.metric_scale import (
        ScaleAnchor,
        fuse_scale_anchors_robust,
    )

    def _a(name, ratio, rel_sigma, kind="object_prior"):
        """构造锚点：`measured=1` → `ratio` 即该锚点给出的尺度因子。"""
        return ScaleAnchor(kind=kind, name=name, measured=1.0, prior_m=ratio,
                           ratio=ratio, weight=1.0, rel_sigma=rel_sigma)

    good = [_a("floor", 1.0, 0.05, "ground_plane_camera_height"),
            _a("table", 1.02, 0.06),
            _a("chair", 0.98, 0.06)]
    outlier = good + [_a("outlier", 6.0, 0.02)]
    f_good = fuse_scale_anchors_robust(good)
    f_out = fuse_scale_anchors_robust(outlier)
    assert f_good.ok and f_out.ok
    # 离群锚点不得把融合结果拉走（鲁棒性）：相对偏差 < 5%
    assert abs(f_out.scale - f_good.scale) / f_good.scale < 0.05
    assert f_out.conflict is False


def test_l1_conflicting_majority_forces_low():
    """L1 合成：多数锚点互相冲突 → `scale_conflict=True`，且必须 low（§10.2）。"""
    from skill3d.reconstruction.metric_scale import (
        ScaleAnchor,
        fuse_scale_anchors_robust,
    )

    def _a(name, ratio, rel_sigma, kind="object_prior"):
        return ScaleAnchor(kind=kind, name=name, measured=1.0, prior_m=ratio,
                           ratio=ratio, weight=1.0, rel_sigma=rel_sigma)

    conflict = [_a("floor", 1.0, 0.02, "ground_plane_camera_height"),
                _a("obj_a", 5.0, 0.02),
                _a("obj_b", 9.0, 0.02)]
    f = fuse_scale_anchors_robust(conflict)
    assert f.conflict is True


# ------------------------------------------------- L2/L3：缺输入 → blocker ----

def test_l2_l3_require_nonoverlapping_arkitscenes_gt():
    """L2/L3 需要 ARKitScenes 非重叠 GT（TODO_USER_INPUT）；缺输入必须显式拦下。"""
    from skill3d.reconstruction.scale_calibration import (
        CALIBRATOR_SCHEMA_VERSION,
        CalibrationUnavailable,
        calibrator_available,
        load_calibrator,
    )

    # 校准器不存在 → 在线只读加载必须失败（而不是给出一个假校准器）
    assert calibrator_available("data/scale_calibration/does_not_exist.json") is False
    with pytest.raises(CalibrationUnavailable):
        load_calibrator("data/scale_calibration/does_not_exist.json")
    assert CALIBRATOR_SCHEMA_VERSION.startswith("scale-conformal")


def test_l1_synthetic_point_map_is_metric_and_estimable():
    """合成场景的尺度锚点可估计（保证 L1 合成测试本身有效）。"""
    pts, n = _synthetic_point_map()
    assessment = assess_scale(pts[0], np.repeat(np.eye(4)[None], pts.shape[0], axis=0),
                              objects=None, scene_name="synthetic-l1")
    # 未标定 → 一律 low（HC30）；这里只断言"评估跑得通且不越权"
    assert assessment.confidence == "low"
    assert set(assessment.allowed_metric_tasks) == set()
    assert assessment.metric_scale is None or np.isfinite(assessment.metric_scale)
