"""v3 进阶契约测试：D-3 恢复阶梯 / §9 坐标层 / §10.1 BA route / M4 写回 / D-2 尺度档。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from skill3d.coords import (
    assert_first_frame_is_world_origin,
    camera_to_world,
    mask_to_depth_grid,
    optical_flow_on_depth_grid,
    resize_mask_nearest,
    vlm_box_to_pixels,
    world_to_camera,
)
from skill3d.reconstruction.legacy_vggsfm_ba.route import (
    BAProbe,
    ba_route_available,
    is_square,
    probe_pycolmap,
    run_ba_route,
    run_inlier_policy,
    square_preprocess_wh,
    weak_g5_proxy,
)
from skill3d.reconstruction.metric_scale import (
    SCALE_SOURCE_FUSED,
    SCALE_SOURCE_OBJECTS,
    SCALE_SOURCE_PLANE,
    ScaleCalibration,
    calibrate_scale_on_arkitscenes,
    evaluate_calibration,
    grade_confidence_d2,
    scale_source_of,
)
from skill3d.reconstruction_gate.quality_metrics import (
    apply_quality,
    compute_and_store_quality,
    quality_is_computed,
)
from skill3d.reconstruction_gate.scene_state import route_from_quality
from skill3d.schemas import (
    ConfidenceMap,
    QualityMetrics,
    ReconstructionArtifact,
)

# ------------------------------------------------------------------ §9 坐标层 ----

def test_vlm_normalized_1000_box_maps_to_pixels():
    """C-6：VLM 输出 0–1000 归一化 → 原图像素（640×480）。"""
    px = vlm_box_to_pixels([100, 200, 900, 800], 640, 480)
    assert px == pytest.approx([64.0, 96.0, 576.0, 384.0])
    # 已是像素坐标时不被二次缩放
    assert vlm_box_to_pixels([10, 20, 100, 200], 640, 480) == pytest.approx(
        [10.0, 20.0, 100.0, 200.0])
    # 越界坐标被 clamp 进图像
    x0, y0, x1, y1 = vlm_box_to_pixels([0, 0, 1000, 1000], 640, 480)
    assert (x0, y0) == (0.0, 0.0) and x1 <= 639 and y1 <= 479


def test_mask_resize_uses_nearest_neighbour_and_is_idempotent():
    """C-7：SAM2 mask（原始分辨率）→ VGGT 深度网格，最近邻、布尔语义保持。"""
    mask = np.zeros((480, 640), dtype=bool)
    mask[100:200, 300:400] = True
    small = resize_mask_nearest(mask, (392, 518))
    assert small.shape == (392, 518) and small.dtype == bool
    assert resize_mask_nearest(small, (392, 518)) is small      # 同尺寸不复制
    depth = np.zeros((32, 392, 518))
    assert mask_to_depth_grid(mask, depth[0]).shape == (392, 518)


def test_world_camera_roundtrip_and_world_frame_convention():
    """§9/C-10：c2w 与 w2c 互逆；世界系 = 首帧相机系（c2w[0]=I）。"""
    theta = 0.3
    c2w = np.array([[np.cos(theta), -np.sin(theta), 0, 1.5],
                    [np.sin(theta), np.cos(theta), 0, 0.5],
                    [0, 0, 1, -2.0],
                    [0, 0, 0, 1.0]])
    pts = np.array([[0.1, 0.2, 3.0], [-1.0, 0.5, 2.0]])
    back = camera_to_world(world_to_camera(pts, c2w), c2w)
    assert back == pytest.approx(pts)
    assert_first_frame_is_world_origin(np.stack([np.eye(4), c2w]))
    with pytest.raises(ValueError):
        assert_first_frame_is_world_origin(np.stack([c2w, np.eye(4)]))


def test_optical_flow_is_computed_on_depth_grid():
    """C-8：光流必须在深度网格上算（返回 HxWx2，与深度同分辨率）。"""
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
    frames[1][200:300, 300:400] = 255
    flow = optical_flow_on_depth_grid(frames, (392, 518))
    assert flow.shape == (392, 518, 2)


# ------------------------------------------------------------------ §10.1 BA ----

def _preds() -> dict:
    return {"point_map": np.zeros((2, 8, 8, 3)),
            "point_conf": np.full((2, 8, 8), 0.7),
            "depth_conf": np.full((2, 8, 8), 0.6)}


def test_ba_route_default_off_falls_back_to_feedforward(tmp_path):
    """BA 未启用 → 不生效、method 仍是 vggt、G5 不得伪造（§10.1）。"""
    res = run_ba_route(_preds(), np.zeros((2, 3, 8, 8)), tmp_path, "s", enabled=False)
    assert res.applied is False and res.recon_method == "vggt"
    assert res.g5_reproj_err_median is None and res.g5_reproj_err_p95 is None
    assert res.weak_proxy_name == "point_conf_median"       # 弱代理明确标注来源
    assert "未启用" in res.skip_reason


def test_ba_route_skips_non_square_input_with_actionable_reason(tmp_path):
    """正方形预处理是硬前提（官方 track_predict 断言 H==W）→ 非正方形明确跳过。"""
    assert square_preprocess_wh(392, 518) == (518, 518)
    assert not is_square((2, 3, 392, 518)) and is_square((2, 3, 518, 518))
    res = run_ba_route(_preds(), np.zeros((2, 3, 392, 518)), tmp_path, "s", enabled=True)
    assert res.applied is False
    assert ("正方形" in res.skip_reason) or ("pycolmap" in res.skip_reason
                                            or "track_predict" in res.skip_reason)


def test_min_inlier_policy_skips_whole_ba_unit():
    """min_inlier_per_frame=64：任一帧不足即整单跳过（§3 M3 关键参数）。"""
    assert run_inlier_policy([100, 100, 100], min_inlier_per_frame=64) is None
    reason = run_inlier_policy([100, 63, 100], min_inlier_per_frame=64)
    assert reason and "63" not in reason and "帧号" in reason
    assert run_inlier_policy(None) is None                   # 无统计不判死


def test_weak_proxy_never_claims_to_be_ba_residual():
    val, name = weak_g5_proxy(None, np.full((4, 4), 0.25))
    assert val == pytest.approx(0.25) and name == "point_conf_median"
    assert weak_g5_proxy(None, None) == (None, "")


def test_probe_reports_version_or_reason():
    probe = probe_pycolmap()
    assert isinstance(probe, BAProbe)
    if probe.available:
        assert probe.has_bundle_adjustment and probe.two_arg_signature
    else:
        assert probe.reason
    avail = ba_route_available()
    assert isinstance(avail.available, bool) and avail.reason


# ------------------------------------------------------------------ M4 写回 ----

def _artifact(tmp_path, *, quality_status="not_computed", quality=None) -> ReconstructionArtifact:
    return ReconstructionArtifact(
        artifact_id="a", artifact_version="v", scene_name="s", recon_method="vggt",
        frame_ids=list(range(32)), source_frame_indices=list(range(32)),
        timestamps=[float(i) for i in range(32)], frame_set_hash="h" * 8,
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, metric_scale=None, scale_known=False,
        quality_status=quality_status, quality=quality,
        confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""))


def test_apply_quality_is_immutable_and_writes_atomically(tmp_path):
    """质量写回不可变 + 落盘（方案 X/Y）；原 artifact 不被修改。"""
    art = _artifact(tmp_path)
    q = QualityMetrics(g1_blur_ok=1.0, g2_brightness=1.0, g3_motion_blur=1.0,
                       g4_frame_count=32, g5_reproj_err_median=float("nan"),
                       g5_reproj_err_p95=float("nan"), g6_depth_var_coeff=0.5,
                       g7_dynamic_ratio=0.0, g9_tracker_consistency=0.9, g10_baseline_quality=0.2,
                       g11_scale_ci=0.05, overall_quality=0.88)
    out_path = tmp_path / "art.json"
    new = apply_quality(art, q, artifact_path=str(out_path))
    assert art.quality_status == "not_computed" and art.quality is None   # 原对象不变
    assert new.quality_status == "computed" and new.quality.overall_quality == pytest.approx(0.88)
    assert new.g5_reproj_err_median is None       # NaN → None（不得标成"BA 结果"）
    reloaded = ReconstructionArtifact.model_validate_json(out_path.read_text(encoding="utf-8"))
    assert reloaded.quality_status == "computed"
    assert quality_is_computed(reloaded) and quality_is_computed(new)
    assert not quality_is_computed(art)


def test_failed_quality_status_never_routes_to_full_3d():
    """硬约束 22：quality_status != computed / quality None / NaN 一律不得 full_3d。"""
    q_nan = QualityMetrics(g1_blur_ok=float("nan"), g2_brightness=float("nan"),
                           g3_motion_blur=float("nan"), g4_frame_count=32,
                           g5_reproj_err_median=float("nan"),
                           g5_reproj_err_p95=float("nan"),
                           g6_depth_var_coeff=float("nan"), g7_dynamic_ratio=float("nan"),
                           g9_tracker_consistency=float("nan"),
                           g10_baseline_quality=float("nan"),
                           g11_scale_ci=float("nan"), overall_quality=float("nan"))
    assert route_from_quality(None) == "fallback_2d_only"
    assert route_from_quality(q_nan, quality_status="not_computed") == "fallback_2d_only"
    assert route_from_quality(q_nan) == "fallback_2d_only"      # NaN 不得 full_3d
    ok = q_nan.model_copy(update={"overall_quality": 0.95})
    assert route_from_quality(ok) == "full_3d"
    # Appendix A：route 只由质量决定；尺度不可用时由逐题通道（M7 硬过滤 /
    # question_gate 的 G-11 降级）处置，不把 MCA 题一起砍成 2D-only
    assert route_from_quality(ok.model_copy(update={"overall_quality": 0.4})) \
        == "fallback_2d_only"
    zero_frames = ok.model_copy(update={"g4_frame_count": 0})
    assert route_from_quality(zero_frames) == "unanswerable"


def test_compute_and_store_quality_marks_computed_and_persists(tmp_path):
    art = _artifact(tmp_path)
    out = compute_and_store_quality(
        art, frames=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(32)],
        depth_maps=np.ones((32, 8, 8)), c2w_list=np.tile(np.eye(4), (32, 1, 1)),
        artifact_path=str(tmp_path / "a.json"))
    assert out.quality_status == "computed" and quality_is_computed(out)
    assert (tmp_path / "a.json").is_file()
    assert json.loads((tmp_path / "a.json").read_text())["quality_status"] == "computed"


# ------------------------------------------------------------------ D-2 尺度档 ----

def test_camera_height_uncertainty_propagation():
    """§10.2：解析传播 (σ_s/s)² ≈ (σ_h/h)² + (σ_plane/h_plane)²。"""
    from skill3d.reconstruction.metric_scale import ground_plane_anchor

    rng = np.random.default_rng(0)
    n = 6000
    # 水平地面（世界 y=0，法向 = 相机 up）→ 满足 ransac 的 up 轴约束
    pts = np.stack([rng.uniform(-2, 2, n), np.zeros(n), rng.uniform(-2, 2, n)], axis=1)
    # c2w：相机水平、up 为世界 +y（OpenCV y 向下 → R 第二列为 -up）
    c2w = np.tile(np.diag([-1.0, -1.0, 1.0, 1.0]), (4, 1, 1))
    for i in range(4):
        c2w[i, 0, 3] = 0.05 * i
    c2w[:, 1, 3] = 1.5                                   # 相机高 1.5（相对单位）
    anchor, plane, notes = ground_plane_anchor(pts, c2w, prior_camera_height_m=1.5,
                                               prior_sigma_m=0.15)
    assert anchor is not None and plane is not None
    # σ_h/h = 0.1；平面项 ≈ 0（理想平面）→ 总 rel_sigma ≈ 0.1
    assert anchor.rel_sigma == pytest.approx(0.1, abs=0.03)
    assert anchor.measured == pytest.approx(1.5, rel=0.05)     # 相机高观测 ≈ 1.5
    assert any("解析 σ_s/s" in t for t in notes)


def test_provisional_medium_requires_calibration():
    """§10.2：未标定 → 单地平面锚点恒 low；标定通过才允许 medium（≤40% CI 门）。"""
    assert grade_confidence_d2(1, 0.2, True, False) == "low"                 # 无标定
    assert grade_confidence_d2(1, 0.5, True, False) == "low"                 # CI 超 40%
    good = ScaleCalibration(n_episodes=40, median_rel_err=0.12, spearman_rho=0.7,
                            plane_identity_err=0.05)
    assert evaluate_calibration(good)[0] and good.provisional_medium
    assert grade_confidence_d2(1, 0.2, True, False, calibration=good) == "medium"
    # 有标准物体锚点 → 仍走多锚点口径（单锚点 → low）
    assert grade_confidence_d2(1, 0.2, True, True, calibration=good) == "low"


@pytest.mark.parametrize("cal,ok", [
    (ScaleCalibration(n_episodes=40, median_rel_err=0.12, spearman_rho=0.7,
                      plane_identity_err=0.05), True),
    (ScaleCalibration(n_episodes=10, median_rel_err=0.12, spearman_rho=0.7,
                      plane_identity_err=0.05), False),                       # 样本不足
    (ScaleCalibration(n_episodes=40, median_rel_err=0.60, spearman_rho=0.7,
                      plane_identity_err=0.05), False),                       # 否决：误差过大
    (ScaleCalibration(n_episodes=40, median_rel_err=0.12, spearman_rho=0.1,
                      plane_identity_err=0.05), False),                       # 否决：ρ 过低
    (ScaleCalibration(n_episodes=40, median_rel_err=0.12, spearman_rho=0.7,
                      plane_identity_err=0.5), False),                        # 否决：平面身份
    (ScaleCalibration(n_episodes=40, median_rel_err=0.12, spearman_rho=0.7,
                      plane_identity_err=0.05, measurement_mra=0.3,
                      baseline_2d_mra=0.5), False),                           # 低于 2D-only
    (None, False),
])
def test_calibration_gate(cal, ok):
    assert evaluate_calibration(cal)[0] is ok


def test_arkitscenes_calibration_uses_gt_pairwise_translation_ratio():
    """§10.2：标定用 GT 位姿 pairwise 平移幅值比（不用深度对齐/Umeyama）。"""
    gt = np.tile(np.eye(4), (4, 1, 1))
    for i in range(4):
        gt[i, 0, 3] = i * 1.0
    pred_perfect = gt.copy()
    pred_off = gt.copy()
    pred_off[:, 0, 3] *= 1.2                    # 尺度高估 20%
    records = [{"c2w_pred": pred_perfect, "c2w_gt": gt, "rel_ci": 0.02,
                "plane_identity_ok": True}] * 35 + [
               {"c2w_pred": pred_off, "c2w_gt": gt, "rel_ci": 0.30,
                "plane_identity_ok": True}] * 5
    cal = calibrate_scale_on_arkitscenes(records)
    assert cal.n_episodes == 40
    assert cal.median_rel_err == pytest.approx(0.0)
    assert 0.0 <= cal.plane_identity_err <= 1.0
    assert cal.method == "arkitscenes_pairwise_translation_ratio"


def test_scale_source_labels():
    from skill3d.reconstruction.metric_scale import ScaleAnchor

    plane = ScaleAnchor(kind="ground_plane_camera_height", name="p", measured=1.0,
                        prior_m=1.5, ratio=1.5, weight=1.0, rel_sigma=0.1)
    obj = ScaleAnchor(kind="object_prior", name="table", measured=1.0, prior_m=0.75,
                      ratio=0.75, weight=1.0, rel_sigma=0.05)
    assert scale_source_of([plane], False) == SCALE_SOURCE_PLANE
    assert scale_source_of([obj], True) == SCALE_SOURCE_OBJECTS
    assert scale_source_of([plane, obj], True) == SCALE_SOURCE_FUSED
    assert scale_source_of([], False) == ""


# --------------------------------------------------- §6.1 服务探活与故障归属 ----

def test_service_probe_and_unavailable_attribution():
    """§6.1：健康检查 + 模型名检查；chat 失败归族为 ServiceUnavailable。"""
    from skill3d.synthesis.vllm_client import (
        ServiceUnavailable,
        VLLMClient,
        health_check,
        probe_service,
    )

    # 未起的本地端口 → 健康检查失败（不抛异常，返回原因）
    ok, reason = health_check("http://127.0.0.1:9", timeout_s=0.5)
    assert ok is False and reason
    status = probe_service("http://127.0.0.1:9", model="qwen3vl-8b-r0")
    assert status["healthy"] is False and status["endpoint"].endswith(":9")

    # check_on_init=True 时服务不可用必须抛 ServiceUnavailable（而不是静默降级）
    with pytest.raises(ServiceUnavailable):
        VLLMClient(["http://127.0.0.1:9"], "qwen3vl-8b-r0", timeout_s=0.5,
                   check_on_init=True)

    # endpoint 归一化：重复 /v1 不叠加
    from skill3d.synthesis.vllm_client import _normalize_base_url

    assert _normalize_base_url("http://h:8100/v1") == "http://h:8100/v1"
    assert _normalize_base_url("http://h:8100/v1/v1/") == "http://h:8100/v1"
    assert _normalize_base_url("http://h:8100") == "http://h:8100/v1"


def test_runner_records_service_unavailable_without_silent_downgrade(tmp_path, monkeypatch):
    """M8 服务故障 → episode 记 unavailable（service_unavailable），不静默降级作答。"""
    from skill3d.adapters.episode_source import load_synthetic_items
    from skill3d.online.runner import OnlineRunConfig, run_episode
    from skill3d.synthesis.vllm_client import ServiceUnavailable

    class _DownClient:
        def chat(self, messages, max_tokens: int = 512, **kw):
            raise ServiceUnavailable("APIConnectionError: 连接被拒绝")

    item = load_synthetic_items("inner_validation", question_types=["object_counting"],
                                frame_size=(120, 160), seed=0)[0]
    # 复用冻结 artifact（避免真的起 VGGT：本测试只关心 M8 的服务故障归属）
    import shutil
    from pathlib import Path as _P

    # v5 HC39：夹具随版本迁移到 tests/golden/v5/（旧夹具只读归档 archive_v4）
    golden = (_P(__file__).resolve().parents[1] / "golden" / "v5" / "data"
              / "frozen_artifact.json")
    art_copy = tmp_path / "art.json"
    shutil.copyfile(golden, art_copy)
    cfg = OnlineRunConfig(mode="real", vllm_endpoints=["http://127.0.0.1:9"],
                          reuse_artifact=str(art_copy),
                          trace_dir=str(tmp_path), memory_dir="")
    out = run_episode(item.episode, item.pixels, cfg, llm=_DownClient())
    assert out.final_state == "unavailable"
    assert any("service_unavailable" in n for n in out.notes)
    assert out.episode_trace.failure.categories == ["evaluator_noanswer"]
