"""官方 VGGSfM BA 历史实现单测（v5 HC35：**只读复现 + 生产 hard-disable**）。

v5 口径变更：官方 `VGGSfM tracker + PyCOLMAP BA` 已被 24 GiB OOM 证据否决，代码移到
`reconstruction/legacy_vggsfm_ba/`（只读历史码）。因此：

1. 历史实现仍可被**复现工具**导入（保留失败可追溯性）；
2. **生产路径启用即报 `UnsupportedConfigurationError`**（默认配置不得触发官方 BA）；
3. 正方形预处理/§9 仿射映射等几何契约继续单测（它们也被 v5 稀疏 BA 复用）；
4. `recon_method` 词汇表不再包含历史名 `vggt_ba`。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from skill3d.coords import (
    box_to_grid,
    grid_transform_from_dict,
    grid_transform_identity,
    grid_transform_square_padded,
    mask_to_grid,
)
from skill3d.reconstruction.legacy_vggsfm_ba import (
    OFFICIAL_VGGSFM_BA_ENABLED,
    UnsupportedConfigurationError,
    assert_official_ba_disabled,
    run_official_ba_repro,
)
from skill3d.reconstruction.legacy_vggsfm_ba.route import (
    MIN_INLIER_PER_FRAME,
    ROUTE_BA,
    ROUTE_FEEDFORWARD,
    ba_route_available,
    build_ba_inputs,
    is_square,
    missing_tracker_weights,
    preflight_ba,
    probe_pycolmap,
    run_ba_route,
    run_inlier_policy,
    square_preprocess_frames,
    square_preprocess_wh,
    weak_g5_proxy,
)


# ----------------------------------------------- 正方形预处理（真执行）----

def test_square_preprocess_wh_matches_official_semantics():
    assert square_preprocess_wh(392, 518) == (518, 518)
    assert square_preprocess_wh(518, 518) == (518, 518)
    assert square_preprocess_wh(480, 640) == (640, 640)


def test_is_square():
    assert is_square((32, 3, 518, 518))
    assert not is_square((32, 3, 392, 518))
    assert not is_square(())


def _write_frames(tmp_path, n=3, h=480, w=640):
    import cv2

    paths = []
    for i in range(n):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 1] = 180
        img[10:40, 10:40] = (255, 255, 255)      # 左上白色标记（与通道序无关）
        p = tmp_path / f"{i:04d}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return paths


def test_square_preprocess_frames_actually_squares_and_records_transform(tmp_path):
    """真执行：输出是正方形，且映射与官方 pad 几何逐字一致。"""
    paths = _write_frames(tmp_path, n=2, h=480, w=640)
    images, tf = square_preprocess_frames(paths, resolution=518)
    shape = tuple(int(v) for v in (images.shape[-2], images.shape[-1]))
    assert shape == (518, 518), shape
    t = grid_transform_from_dict(tf)
    assert t["padded_to_square"] is True
    assert t["source_hw"] == [480, 640]
    assert t["grid_hw"] == [518, 518]
    # 官方：max_dim=640, top=(640-480)//2=80, left=0, scale=518/640
    assert t["pad_side"] == 640
    assert t["pad_offset_xy"] == [0, 80]
    assert t["scale_y"] == pytest.approx(518 / 640, rel=1e-9)


def test_square_preprocess_numpy_fallback_geometry_matches_official(tmp_path):
    """numpy 兜底实现与官方几何一致（同一 pad/scale 语义），便于离线单测。"""
    from skill3d.reconstruction.legacy_vggsfm_ba.route import _square_preprocess_numpy

    paths = _write_frames(tmp_path, n=1, h=480, w=640)
    frames, tf, why = _square_preprocess_numpy(paths, 518, 1024)
    assert frames[0].shape == (518, 518, 3)
    assert "兜底" in why
    # 内容检查：标记在原图 y∈[10,40], x∈[10,40]；pad 把 y 下移 80px 后缩放到 518 网格
    #   y_grid = (y_src + top) * scale, x_grid = (x_src + left) * scale
    scale, top, left = 518 / 640, 80, 0
    ys, xs = np.nonzero(frames[0].max(axis=2) > 250)
    assert ys.min() == pytest.approx((10 + top) * scale, abs=2)
    assert ys.max() == pytest.approx((39 + top) * scale, abs=2)
    assert xs.min() == pytest.approx((10 + left) * scale, abs=2)
    # 纯缩放会把标记放在 y≈8，pad 后是 y≈73 → 差值证明 padding 真的执行了
    assert ys.min() - 10 * scale > 50


# --------------------------------------------------- §9 pad 感知映射精度 ----

def test_mask_to_grid_with_pad_differs_from_plain_resize():
    """pad 路径必须用仿射；纯缩放会错位（这正是 C-7 类缺陷）。"""
    t = grid_transform_square_padded((480, 640), (518, 518))
    m = np.zeros((480, 640), dtype=bool)
    m[0:100, 0:100] = True                     # 原图左上角
    g = mask_to_grid(m, t, shape=(518, 518))
    plain = mask_to_grid(m, grid_transform_identity((480, 640), (518, 518)),
                         shape=(518, 518))
    ys, _ = np.nonzero(g)
    ys_plain, _ = np.nonzero(plain)
    # 仿射：y ∈ [(0+80)*0.809, (100+80)*0.809] = [64.8, 145.7]
    assert ys.min() == pytest.approx(65, abs=2)
    assert ys.max() == pytest.approx(146, abs=2)
    # 纯缩放：y ∈ [0, 108] → 起点差 ~65px，证明二者不可混用
    assert ys_plain.min() == 0
    assert ys.min() - ys_plain.min() > 50


def test_mask_to_grid_identity_matches_legacy_behavior():
    """非 pad（feed-forward 主线）行为必须与 v3 的纯缩放逐位一致。"""
    t = grid_transform_identity((480, 640), (392, 518))
    m = np.zeros((480, 640), dtype=bool)
    m[100:200, 200:300] = True
    got = mask_to_grid(m, t, shape=(392, 518))
    from skill3d.coords import resize_mask_nearest

    assert np.array_equal(got, resize_mask_nearest(m, (392, 518)))
    # 缺省 transform（老 artifact）同样走纯缩放
    assert np.array_equal(mask_to_grid(m, None, shape=(392, 518)),
                          resize_mask_nearest(m, (392, 518)))


def test_box_to_grid_handles_pad_and_identity():
    t = grid_transform_square_padded((480, 640), (518, 518))
    box = box_to_grid([0, 0, 100, 100], t)
    assert box[1] == pytest.approx(80 * 518 / 640, rel=1e-6)   # y0 带上 pad
    assert box[0] == pytest.approx(0.0, abs=1e-9)
    t2 = grid_transform_identity((480, 640), (392, 518))
    b2 = box_to_grid([100, 50, 200, 150], t2)
    assert b2[0] == pytest.approx(100 * 518 / 640, rel=1e-6)
    assert b2[1] == pytest.approx(50 * 392 / 480, rel=1e-6)


# ------------------------------------------------------------- 前置探测 ----

def test_probe_pycolmap_reports_two_arg_signature():
    probe = probe_pycolmap()
    if not probe.available:
        pytest.skip(f"pycolmap 不可用: {probe.reason}")
    assert probe.has_bundle_adjustment and probe.two_arg_signature
    assert probe.pycolmap_version


def test_preflight_ba_gives_actionable_hints_when_weights_missing():
    """缺权重时必须给出**可操作的下载指引**（§3 M3 [待实码核验] 的收敛点）。"""
    r = preflight_ba()
    assert set(r) >= {"pycolmap_ok", "tracker_ok", "missing_weights", "ready",
                      "download_hints"}
    if r["missing_weights"]:
        assert r["download_hints"], "缺权重却没有下载指引"
        assert not r["ready"]
    else:
        assert r["download_hints"] == []
    # 探测结果与独立检查一致
    assert (len(missing_tracker_weights()) == 0) == r["tracker_ok"] or not r["pycolmap_ok"]


def test_ba_route_available_reflects_probe():
    probe = ba_route_available()
    if probe.available:
        assert "可用" in probe.reason or "就位" in probe.reason


# --------------------------------------------------------- 输入生产者 ----

def _preds(**kw):
    n, h, w = 2, 8, 8
    base = {
        "c2w": np.tile(np.eye(4), (n, 1, 1)),
        "intrinsic": np.tile(np.eye(3), (n, 1, 1)),
        "depth_map": np.ones((n, h, w), dtype=np.float64),
        "point_map": np.zeros((n, h, w, 3), dtype=np.float64),
        "depth_conf": np.ones((n, h, w), dtype=np.float64),
        "point_conf": np.ones((n, h, w), dtype=np.float64) * 0.7,
    }
    base.update(kw)
    return base


def test_build_ba_inputs_produces_required_keys():
    """A-8 修复点：`extrinsic`(w2c 3×4) / `intrinsic_raw` / `point_map_in_ba_frame`。"""
    out = build_ba_inputs(_preds(), images=None)
    assert set(out) >= {"extrinsic", "intrinsic_raw", "point_map_in_ba_frame"}
    assert out["extrinsic"].shape == (2, 3, 4)
    assert out["intrinsic_raw"].shape == (2, 3, 3)
    # c2w=I → w2c=I，平移列应为 0
    assert np.allclose(out["extrinsic"][:, :3, 3], 0.0)
    assert out["point_map_in_ba_frame"].shape[:3] == (2, 8, 8)


def test_build_ba_inputs_uses_provided_extrinsic():
    extri = np.tile(np.eye(4)[:3, :4], (2, 1, 1))
    extri[:, 0, 3] = 5.0
    out = build_ba_inputs(_preds(extrinsic=extri), images=None)
    assert out["extrinsic"][0, 0, 3] == pytest.approx(5.0)


# --------------------------------------------------------- 回退契约 ----

def test_run_ba_route_disabled_is_feedforward_with_none_g5(tmp_path):
    res = run_ba_route(_preds(), np.zeros((2, 3, 518, 518)), tmp_path, "s", enabled=False)
    assert not res.applied
    assert res.recon_method == ROUTE_FEEDFORWARD
    assert res.g5_reproj_err_median is None and res.g5_reproj_err_p95 is None
    assert "reproj_errors" not in res.preds        # 不得凭空写 G5
    assert res.weak_proxy_name and res.weak_proxy_g5 is not None


def test_run_ba_route_non_square_is_skipped_with_actionable_reason(tmp_path):
    res = run_ba_route(_preds(), np.zeros((2, 3, 392, 518)), tmp_path, "s", enabled=True)
    assert not res.applied and res.recon_method == ROUTE_FEEDFORWARD
    assert "正方形" in res.skip_reason
    assert res.g5_reproj_err_median is None
    assert "reproj_errors" not in res.preds


def test_run_ba_route_claims_square_but_gets_non_square(tmp_path):
    """`square_preprocessed=True` 却拿到非正方形 → 说明预处理没真执行（A-8 原缺陷）。"""
    res = run_ba_route(_preds(), np.zeros((2, 3, 392, 518)), tmp_path, "s",
                       enabled=True, square_preprocessed=True)
    assert not res.applied
    assert "预处理未真正执行" in res.skip_reason


def test_run_ba_route_unavailable_env_falls_back(tmp_path):
    """环境不支持（pycolmap/权重缺失）→ 回退 feed-forward，G5 None，且不标 vggt_ba。"""
    if ba_route_available().available:
        pytest.skip("本机 BA 环境齐备，无法测不可用回退分支")
    res = run_ba_route(_preds(), np.zeros((2, 3, 518, 518)), tmp_path, "s", enabled=True)
    assert not res.applied and res.recon_method == ROUTE_FEEDFORWARD
    assert res.g5_reproj_err_median is None


# ------------------------------------- v5 HC35：生产 hard-disable（只读复现）----

def test_official_ba_disabled_by_default():
    """默认配置不得启用官方 BA（HC35）。"""
    assert OFFICIAL_VGGSFM_BA_ENABLED is False
    assert_official_ba_disabled(enable_official_vggsfm_ba=False)   # 不抛


def test_production_enable_request_raises_unsupported():
    """生产启用请求 → UnsupportedConfigurationError（不静默回退）。"""
    with pytest.raises(UnsupportedConfigurationError) as ei:
        assert_official_ba_disabled(enable_official_vggsfm_ba=True, context="unit-test")
    assert "24 GiB" in str(ei.value) and "vggt_sparse_ba" in str(ei.value)


def test_repro_path_requires_explicit_flag(tmp_path):
    """`run_official_ba_repro` 默认拒绝；显式 allow_repro=True 才转发到历史实现。"""
    with pytest.raises(UnsupportedConfigurationError):
        run_official_ba_repro(_preds(), np.zeros((2, 3, 518, 518)), tmp_path, "s",
                              enabled=False)


def test_reconstruction_rejects_official_ba_and_sparse_ba(tmp_path):
    """v5 生产入口：官方 BA 与未过 PoC 的 sparse BA 都不得被接线。"""
    from skill3d.reconstruction.vggt_runner import reconstruct

    with pytest.raises(UnsupportedConfigurationError):
        reconstruct([np.zeros((8, 8, 3), dtype=np.uint8)], "s", tmp_path, method="vggt",
                    use_ba=True)
    with pytest.raises(UnsupportedConfigurationError):
        reconstruct([np.zeros((8, 8, 3), dtype=np.uint8)], "s", tmp_path,
                    method="vggt_sparse_ba")


def test_inlier_policy_skips_when_any_frame_below_threshold():
    assert run_inlier_policy([100, 200, 300]) is None
    why = run_inlier_policy([100, MIN_INLIER_PER_FRAME - 1, 300])
    assert why and "内点不足" in why
    assert run_inlier_policy(None) is None


def test_weak_g5_proxy_prefers_point_conf():
    v, name = weak_g5_proxy(np.ones((4, 4)) * 0.3, np.ones((4, 4)) * 0.9)
    assert name == "point_conf_median" and v == pytest.approx(0.9)
    v2, name2 = weak_g5_proxy(np.ones((4, 4)) * 0.3, None)
    assert name2 == "depth_conf_median" and v2 == pytest.approx(0.3)
    assert weak_g5_proxy(None, None) == (None, "")


@pytest.mark.skipif(
    os.environ.get("SKILL3D_RUN_BA_POC", "") != "1",
    reason="重测试（加载 VGGSfM tracker + DINOv2 并跑真实追踪，数分钟级/需 GPU）；"
           "注意：该路线已按 HC35 退出生产，只用于复现 OOM 失败证据")
@pytest.mark.skipif(not ba_route_available().available,
                    reason="BA 环境未齐备（pycolmap/tracker 权重）")
def test_ba_route_returns_refined_arrays_or_falls_back(tmp_path):
    """环境齐备时的端到端形状契约：成功 → 精化位姿键齐备；失败 → 干净回退。

    本用例**不**伪造成功：真实 tracker 在合成噪声图上很可能 track 不到点，
    那时必须走回退分支（这正是契约要求的行为）。
    """
    n, h, w = 3, 518, 518
    rng = np.random.default_rng(0)
    preds = _preds()
    preds["c2w"] = np.tile(np.eye(4), (n, 1, 1))
    preds["intrinsic"] = np.tile(np.array([[200.0, 0, w / 2], [0, 200.0, h / 2],
                                           [0, 0, 1.0]]), (n, 1, 1))
    preds["depth_map"] = np.full((n, h, w), 2.0)
    preds["point_map"] = np.zeros((n, h, w, 3))
    preds["depth_conf"] = np.ones((n, h, w))
    preds["point_conf"] = np.ones((n, h, w))
    images = rng.random((n, 3, 518, 518), dtype=np.float32)
    import torch

    res = run_ba_route(preds, torch.from_numpy(images), tmp_path, "s", enabled=True,
                       square_preprocessed=True)
    assert res.recon_method in (ROUTE_FEEDFORWARD, ROUTE_BA)
    if res.applied:
        assert res.g5_reproj_err_median is not None
        assert res.preds["reproj_errors"].size > 0
        assert "c2w_refined" in res.preds or res.notes
        assert res.n_points3D >= 0
    else:
        assert res.g5_reproj_err_median is None
        assert "reproj_errors" not in res.preds
        assert res.skip_reason
