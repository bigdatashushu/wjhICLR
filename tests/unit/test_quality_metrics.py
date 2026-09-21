"""M4 G1-G11 质量指标单测（§10）。"""

import numpy as np
import pytest

from skill3d.reconstruction_gate import quality_metrics as qm
from skill3d.reconstruction_gate.confidence_map import fuse_confidence


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


def test_g4_frame_count():
    assert qm.g4_frame_count([0] * 32) == 32
    assert qm.g4_frame_count([]) == 0


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


def test_g5_g7_g9_g11_missing_values():
    """v5 HC37：G5 缺失必须是 `None`（不是 NaN/0/1 分）；其余指标缺数据记 NaN。"""
    med, p95 = qm.g5_reproj_err(None)
    assert med is None and p95 is None
    # 未跑真 BA（正式 vggt 主线）→ not_available → 一律 None，禁止代理值
    med2, p95_2 = qm.g5_reproj_err(np.arange(1, 11, dtype=float))
    assert med2 is None and p95_2 is None
    assert np.isnan(qm.g7_dynamic_ratio(None))
    assert np.isnan(qm.g9_tracker_consistency(None))
    assert np.isnan(qm.g11_scale_ci(None))


def test_g5_reproj_err_computed():
    """只有 reprojection_status="computed"（真 BA）才产出 G5 标量。"""
    e = np.arange(1, 101, dtype=float)  # 1..100
    med, p95 = qm.g5_reproj_err(e, status="computed")
    assert med == 50.5
    assert p95 == pytest.approx(95.05, rel=1e-3)
    # 全非有限残差 → 仍为 None（由调用方把状态降为 failed，不得补分）
    med2, p95_2 = qm.g5_reproj_err(np.array([np.nan, np.inf]), status="computed")
    assert med2 is None and p95_2 is None


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


def test_compute_g1_g11_partial():
    """只给 frames + depth 时，可算指标有值、不可算为 NaN，overall 只基于可算项。"""
    rng = np.random.default_rng(1)
    frames = [(rng.random((64, 64, 3)) * 200 + 30).astype(np.uint8) for _ in range(32)]
    depth = np.abs(rng.normal(3.0, 0.1, size=(32, 8, 8)))
    q = qm.compute_g1_g11(frames=frames, depth_maps=depth)
    assert q.g1_blur_ok == 1.0
    assert q.g4_frame_count == 32
    assert np.isfinite(q.g6_depth_var_coeff)
    # v5 HC37：G5 未计算 → None（从 overall 分母排除，不补默认分）
    assert q.g5_reproj_err_median is None and q.g5_reproj_err_p95 is None
    assert np.isnan(q.g7_dynamic_ratio)
    assert 0.0 <= q.overall_quality <= 1.0
    # 与"把 G5 补成满分"的历史口径对比：分母少两项，不得因此虚高
    from skill3d.reconstruction_gate.quality_metrics import overall_from_metrics
    assert q.overall_quality == pytest.approx(overall_from_metrics(q))


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
