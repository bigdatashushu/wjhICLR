"""v6 §10 M4 无真值主门单测（`reconstruction_gate/m4_main_gate.py`）。

覆盖要求：
- 完美合成几何（盒室解析渲染，两视角组）→ 内点率高、主门过；
- 注入退化（随机深度 / 跨场景帧）→ 比率塌、主门不过；
- NaN/Inf 深度 → 主门不过、warnings 非空、**不抛异常**；
- **不得单挑**：warp 过而重叠不过、重叠过而 warp 不过，两种情形主门都必须 False；
- `conf_warp_monotonic` 在退化/数据不足时返回 None（不是 False）；
- 确定性：同一输入两次 → 逐位一致；点云重叠对尺度不变（报率不报绝对距离）。

合成场景是**解析渲染**（轴对齐盒室的 ray-box 求交），带覆盖光照变化的平滑贴色，
不含任何真值/学习模型 —— 与门本身的"纯几何"口径一致。
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pytest

from skill3d.reconstruction_gate import m4_main_gate as mg

HW = (48, 64)                                  # 单测用小图（性能用例另用 518×392）
BOX_LO = np.array([-2.0, -1.5, -1.5])
BOX_HI = np.array([2.0, 1.5, 1.5])


# --------------------------------------------------------------------------- #
# 合成场景（解析渲染）
# --------------------------------------------------------------------------- #

def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenCV 相机系（x 右 / y 下 / z 前）的 camera→world。"""
    eye = np.asarray(eye, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, d, f, eye
    return m


def _intrinsics(hw=HW) -> np.ndarray:
    h, w = hw
    f = 0.75 * w                                           # 宽视场（≈67°），盒室四面都可见
    return np.array([[f, 0.0, (w - 1) / 2.0],
                     [0.0, f, (h - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def _poses(n: int, *, radius: float = 1.0) -> np.ndarray:
    """房间内绕中心的一小段圆弧轨迹（帧序 = 采集序，基线适中）。"""
    out = []
    for t in range(n):
        th = np.deg2rad(-25.0 + 50.0 * (t / max(1, n - 1)))
        eye = np.array([radius * np.cos(th), 0.15, radius * np.sin(th)])
        out.append(_look_at(eye, np.zeros(3)))
    return np.stack(out)


def _render_box(c2w: np.ndarray, K: np.ndarray, hw, lo: np.ndarray, hi: np.ndarray):
    """轴对齐盒室解析渲染 → (depth (H,W) 相机系 z 深度, points_world (H,W,3))。

    slab 法求交（相机在盒内取出口面，盒外取入口面），无命中记 NaN。
    """
    h, w = hw
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    dirs_cam = np.stack([(uu - K[0, 2]) / K[0, 0],
                         (vv - K[1, 2]) / K[1, 1],
                         np.ones_like(uu)], axis=-1)
    dirs_w = dirs_cam @ c2w[:3, :3].T
    o = c2w[:3, 3]
    t_enter = np.full(hw, -np.inf)
    t_exit = np.full(hw, np.inf)
    for a in range(3):
        d = dirs_w[..., a]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo[a] - o[a]) / d
            t2 = (hi[a] - o[a]) / d
        inside_slab = bool(lo[a] <= o[a] <= hi[a])
        t1 = np.where(np.isnan(t1), -np.inf if inside_slab else np.inf, t1)
        t2 = np.where(np.isnan(t2), np.inf if inside_slab else -np.inf, t2)
        t_enter = np.maximum(t_enter, np.minimum(t1, t2))
        t_exit = np.minimum(t_exit, np.maximum(t1, t2))
    hit = (t_exit >= t_enter) & np.isfinite(t_exit)
    t = np.where(t_enter > 0.0, t_enter, t_exit)
    depth = np.where(hit, t, np.nan)
    points = o + t[..., None] * dirs_w
    return depth, points


def _shade(points_world: np.ndarray) -> np.ndarray:
    """**世界系**平滑贴色（同一 3D 点跨帧同色 → 光度可对齐，且不是常数）。"""
    p = points_world
    c = 0.5 + 0.35 * (np.sin(0.7 * p[..., 0] + 0.3)
                      * np.sin(0.9 * p[..., 1])
                      * np.sin(0.8 * p[..., 2] + 1.1))
    return np.clip(c * 255.0, 0, 255).astype(np.uint8)


def _build_scene(n: int = 8, hw=HW, *, split_at: int | None = None,
                 box_b=(0.6 * BOX_LO, 0.6 * BOX_HI)) -> dict:
    """渲染 n 帧；`split_at` 给定则从该帧起换成**另一个场景**的盒子（跨场景注入）。

    另一个场景取"同心但缩到 0.6 倍"的房间：相机仍在盒内（深度全有效，不会因无命中
    被跳过），但两组子云彼此相距 ≫ 相对阈值 → 重叠率塌到 0，正好用来验证"warp 过、
    重叠不过"的组合。
    """
    K = _intrinsics(hw)
    poses = _poses(n)
    depths, points, frames = [], [], []
    for idx, c2w in enumerate(poses):
        lo, hi = BOX_LO, BOX_HI
        if split_at is not None and idx >= split_at:
            lo, hi = box_b
        d, p = _render_box(c2w, K, hw, np.asarray(lo, float), np.asarray(hi, float))
        depths.append(d)
        points.append(p)
        frames.append(_shade(p))
    return {
        "frames": np.stack(frames),
        "depth_maps": np.stack(depths),
        "c2w_list": poses,
        "intrinsics": K,
        "point_map": np.stack(points),
    }


@lru_cache(maxsize=4)
def _scene(n: int = 8, split_at: int | None = None) -> tuple:
    """缓存好的合成场景（返回不可变元组，避免用例之间互相改动）。"""
    s = _build_scene(n, split_at=split_at)
    return s["frames"], s["depth_maps"], s["c2w_list"], s["intrinsics"], s["point_map"]


def _perfect(n: int = 8) -> dict:
    fr, d, c2w, K, pm = _scene(n)
    return {"frames": fr, "depth_maps": d, "c2w_list": c2w, "intrinsics": K, "point_map": pm}


# --------------------------------------------------------------------------- #
# 1. 完美几何 → 高内点率、主门通过
# --------------------------------------------------------------------------- #

def test_perfect_geometry_warp_and_overlap_pass():
    s = _perfect(8)
    warp = mg.cross_view_warp_inliers(s["frames"], s["depth_maps"], s["c2w_list"], s["intrinsics"])
    assert warp["n_pairs"] > 0
    assert warp["warp_inlier_ratio"] >= mg.TH_WARP_INLIER
    assert warp["warp_inlier_ratio"] > 0.8                  # 完美几何应接近 1
    assert warp["warp_photometric_inlier_ratio"] >= 0.5
    assert len(warp["per_pair"]) == warp["n_pairs"] == 8 * mg.N_NEIGHBOR_FRAMES

    ov = mg.grouped_cloud_overlap(s["point_map"])
    assert ov["cloud_overlap_ratio"] >= mg.TH_CLOUD_OVERLAP
    # 完美几何也不会到 1.0：只在一半帧里可见的表面（相机身后的墙）本来就没有对应点，
    # 这正是 τ_cloud 取"率"而不是取接近 1 的原因。
    assert ov["cloud_overlap_ratio"] > 0.6
    assert ov["n_fwd"] > 0 and ov["n_bwd"] > 0

    gate = mg.main_gate(s)                                   # 原始数组路径
    assert gate["main_gate_passed"] is True
    assert gate["sub_results"] == {"warp_inlier_ratio": True, "cloud_overlap_ratio": True}
    assert gate["warnings"] == []
    assert gate["thresholds"] == {"warp_inlier_ratio": mg.TH_WARP_INLIER,
                                  "cloud_overlap_ratio": mg.TH_CLOUD_OVERLAP}
    assert gate["values"]["warp_inlier_ratio"] > 0.8
    assert gate["values"]["cloud_overlap_ratio"] > 0.6
    assert gate["conf_weight"] == 1.0                        # 未做自检 → 不加罚


def test_main_gate_accepts_precomputed_metrics():
    """第二种取值口径：把三个函数的返回合并进 quality_inputs（显式值优先）。"""
    s = _perfect(8)
    warp = mg.cross_view_warp_inliers(s["frames"], s["depth_maps"], s["c2w_list"], s["intrinsics"])
    ov = mg.grouped_cloud_overlap(s["point_map"])
    gate = mg.main_gate({**warp, **ov})
    assert gate["main_gate_passed"] is True
    assert gate["n_pairs"] == warp["n_pairs"]


# --------------------------------------------------------------------------- #
# 2. 注入退化 → 比率塌、主门不过
# --------------------------------------------------------------------------- #

def test_random_depth_collapses_warp_ratio():
    s = _perfect(8)
    rng = np.random.default_rng(0)
    garbage = rng.random(s["depth_maps"].shape) * 10.0 + 0.1      # 全在相机前方但完全不对
    warp = mg.cross_view_warp_inliers(s["frames"], garbage, s["c2w_list"], s["intrinsics"])
    assert not np.isfinite(warp["warp_inlier_ratio"]) or warp["warp_inlier_ratio"] < 0.2

    gate = mg.main_gate({"frames": s["frames"], "depth_maps": garbage,
                         "c2w_list": s["c2w_list"], "intrinsics": s["intrinsics"],
                         "point_map": s["point_map"]})
    assert gate["main_gate_passed"] is False
    assert gate["sub_results"]["warp_inlier_ratio"] is False
    assert gate["sub_results"]["cloud_overlap_ratio"] is True     # 点图仍完好
    assert gate["warnings"]


def test_cross_scene_frames_collapse_cloud_overlap():
    """跨场景帧：warp 在两个子段内自洽（过），但前后子云不重叠（不过）→ 主门必须 False。"""
    mixed = _build_scene(32, split_at=16)
    # 前 16 帧 = 房间 A，后 16 帧 = 平移 30 单位的房间 B
    assert not np.allclose(mixed["point_map"][0], mixed["point_map"][-1])

    warp = mg.cross_view_warp_inliers(mixed["frames"], mixed["depth_maps"],
                                      mixed["c2w_list"], mixed["intrinsics"])
    ov = mg.grouped_cloud_overlap(mixed["point_map"])
    assert warp["warp_inlier_ratio"] >= mg.TH_WARP_INLIER          # 幻觉式的"跨视图一致"
    assert ov["cloud_overlap_ratio"] <= 0.2                        # 分组重叠把它戳穿
    assert ov["cloud_overlap_ratio"] < mg.TH_CLOUD_OVERLAP

    gate = mg.main_gate({**warp, **ov})
    assert gate["main_gate_passed"] is False
    assert gate["sub_results"] == {"warp_inlier_ratio": True, "cloud_overlap_ratio": False}
    # 好场景对照：同一批次里若没有跨场景注入，两组子云是重叠的
    assert mg.grouped_cloud_overlap(_perfect(32)["point_map"])["cloud_overlap_ratio"] > 0.6


# --------------------------------------------------------------------------- #
# 3. NaN/Inf：fail-closed、告警、不抛异常
# --------------------------------------------------------------------------- #

def test_nan_inf_depth_fails_closed_without_exception():
    s = _perfect(8)
    broken = np.array(s["depth_maps"], copy=True)
    broken[:] = np.nan
    broken[0, 0, 0] = np.inf
    broken[1, :, :] = -np.inf

    warp = mg.cross_view_warp_inliers(s["frames"], broken, s["c2w_list"], s["intrinsics"])
    assert np.isnan(warp["warp_inlier_ratio"])                      # 未知记 NaN，不是 0/False
    assert np.isnan(warp["warp_photometric_inlier_ratio"])

    gate = mg.main_gate({"frames": s["frames"], "depth_maps": broken,
                         "c2w_list": s["c2w_list"], "intrinsics": s["intrinsics"],
                         "point_map": s["point_map"]})
    assert gate["main_gate_passed"] is False
    assert gate["sub_results"]["warp_inlier_ratio"] is False
    assert np.isnan(gate["values"]["warp_inlier_ratio"])
    assert gate["warnings"]
    assert any("warp_inlier_ratio" in w for w in gate["warnings"])

    # 子云全 NaN → 重叠率 NaN + reason，不得返回 0
    ov = mg.grouped_cloud_overlap(np.full_like(s["point_map"], np.nan))
    assert np.isnan(ov["cloud_overlap_ratio"])
    assert ov["reason"] == "insufficient_points"


def test_missing_inputs_and_bad_values_fail_closed():
    gate = mg.main_gate({})                                          # 什么都没给
    assert gate["main_gate_passed"] is False
    assert gate["sub_results"] == {"warp_inlier_ratio": False, "cloud_overlap_ratio": False}
    assert np.isnan(gate["values"]["warp_inlier_ratio"])
    assert np.isnan(gate["values"]["cloud_overlap_ratio"])
    assert len(gate["warnings"]) >= 2

    gate = mg.main_gate({"warp_inlier_ratio": float("inf"),
                         "cloud_overlap_ratio": float("nan")})
    assert gate["main_gate_passed"] is False
    assert gate["sub_results"] == {"warp_inlier_ratio": False, "cloud_overlap_ratio": False}

    gate = mg.main_gate({"warp_inlier_ratio": None, "cloud_overlap_ratio": "bad"})
    assert gate["main_gate_passed"] is False
    assert gate["warnings"]

    gate = mg.main_gate({"warp_inlier_ratio": 0.9, "cloud_overlap_ratio": 0.9,
                         "frames": None, "depth_maps": None})
    assert gate["main_gate_passed"] is True                          # 显式值优先，None 不参与
    assert gate["warnings"] == []

    # 坏几何数组（形状不匹配）→ 内部捕获 → NaN + 告警，而不是抛异常
    s = _perfect(4)
    gate = mg.main_gate({"frames": s["frames"][:2], "depth_maps": s["depth_maps"],
                         "c2w_list": s["c2w_list"], "intrinsics": s["intrinsics"],
                         "point_map": s["point_map"]})
    assert gate["main_gate_passed"] is False
    assert any("cross_view_warp_inliers 失败" in w for w in gate["warnings"])


# --------------------------------------------------------------------------- #
# 4. 多指标不得单挑
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("warp_val,cloud_val", [(0.99, 0.01), (0.01, 0.99), (0.99, 0.0),
                                                (0.49, 0.29), (0.51, 0.0), (0.0, 0.51)])
def test_neither_submetric_passes_alone(warp_val, cloud_val):
    """任一单项都不得单独放行：主门恒为两个子项的 AND（最多一个子项能过）。"""
    vals = {"warp_inlier_ratio": warp_val, "cloud_overlap_ratio": cloud_val}
    gate = mg.main_gate(vals)
    assert gate["main_gate_passed"] is False                         # 最多只有一个过 → 仍不过
    assert sum(gate["sub_results"].values()) <= 1

    # 阈值门槛严格更高 → 仍不过；阈值调到与观测值恰好相等（≥ 口径）→ 两个都过才过
    gate = mg.main_gate(vals, thresholds={"warp_inlier_ratio": warp_val + 1e-9,
                                          "cloud_overlap_ratio": cloud_val})
    assert gate["main_gate_passed"] is False
    gate = mg.main_gate(vals, thresholds={"warp_inlier_ratio": warp_val,
                                          "cloud_overlap_ratio": cloud_val})
    assert gate["main_gate_passed"] is True


def test_each_submetric_can_veto_but_never_alone():
    """真值构造：warp 过 / 重叠过 各一例，主门都必须是 False。"""
    s = _perfect(8)
    # (a) 重叠过、warp 不过：点图完好，深度换成垃圾
    rng = np.random.default_rng(0)
    bad_depth = rng.random(s["depth_maps"].shape) * 5.0 + 0.5
    g_a = mg.main_gate({"depth_maps": bad_depth, "frames": s["frames"],
                        "c2w_list": s["c2w_list"], "intrinsics": s["intrinsics"],
                        "point_map": s["point_map"]})
    assert g_a["sub_results"]["cloud_overlap_ratio"] is True
    assert g_a["sub_results"]["warp_inlier_ratio"] is False
    assert g_a["main_gate_passed"] is False

    # (b) warp 过、重叠不过：跨场景帧（上面单测已证 warp 仍高）
    mixed = _build_scene(16, split_at=8)
    g_b = mg.main_gate(mixed)
    assert g_b["sub_results"]["warp_inlier_ratio"] is True
    assert g_b["sub_results"]["cloud_overlap_ratio"] is False
    assert g_b["main_gate_passed"] is False


# --------------------------------------------------------------------------- #
# 5. conf 纪律：只降权、不作硬门、可选掩码
# --------------------------------------------------------------------------- #

def test_conf_never_gates_main_gate():
    """§10.3/§10.4：conf 不能否决主门，也不是硬阈值。"""
    bad_conf = np.full((8, *HW), np.nan)                             # conf 全坏
    gate = mg.main_gate({"warp_inlier_ratio": 0.9, "cloud_overlap_ratio": 0.9,
                         "depth_conf": bad_conf, "point_conf": bad_conf})
    assert gate["main_gate_passed"] is True
    assert gate["conf_warp_monotonic"] is None

    # 单向不可救：conf 再自信也救不了不过的子项
    gate = mg.main_gate({"warp_inlier_ratio": 0.1, "cloud_overlap_ratio": 0.9,
                         "depth_conf": np.full((8, *HW), 99.0)})
    assert gate["main_gate_passed"] is False

    mask = mg.conf_optional_mask(np.array([0.5, 2.0, 3.0, np.nan, np.inf]))
    assert mask.tolist() == [False, True, True, False, False]        # 仅掩码，默认不参与门


def test_conf_warp_monotonic_true_false_and_downweight():
    rng = np.random.default_rng(0)
    conf = rng.random((20, 32, 32)) * 4.0 + 1.0                     # C = exp(Σ)+1 口径
    resid_good = (5.0 - conf) / 10.0 + rng.random(conf.shape) * 1e-3  # conf ↑ → 残差 ↓
    good = mg.conf_warp_monotonic(conf, resid_good, n_bins=5)
    assert good["monotonic"] is True
    assert good["spearman"] is not None and good["spearman"] < 0
    assert len(good["bin_medians"]) >= mg.MIN_CONF_WARP_BINS
    assert np.all(np.diff(good["bin_medians"]) <= 0)

    gate = mg.main_gate({"warp_inlier_ratio": 0.9, "cloud_overlap_ratio": 0.9,
                         "conf_warp_monotonic": True})
    assert gate["conf_weight"] == 1.0 and gate["main_gate_passed"] is True

    resid_bad = conf / 10.0 + rng.random(conf.shape) * 1e-3
    bad = mg.conf_warp_monotonic(conf, resid_bad, n_bins=5)
    assert bad["monotonic"] is False
    gate = mg.main_gate({"warp_inlier_ratio": 0.9, "cloud_overlap_ratio": 0.9,
                         "conf_warp_monotonic": False,
                         "conf_warp_spearman": bad["spearman"]})
    assert gate["conf_weight"] == mg.TH_CONF_DOWNWEIGHT              # 降权
    assert gate["main_gate_passed"] is True                          # 但**不否决**
    assert any("不单调" in w for w in gate["warnings"])
    assert gate["conf_warp_spearman"] == bad["spearman"]


def test_conf_warp_monotonic_none_when_degenerate():
    """退化/数据不足 → None（不是 False），调用方据此降权而非否决。"""
    # (a) 样本太少
    few = mg.conf_warp_monotonic(np.ones((2, 4, 4)), np.ones((2, 4, 4)))
    assert few["monotonic"] is None and few["monotonic"] is not False
    assert few["spearman"] is None and few["bin_medians"] == []
    assert few["reason"] == "insufficient_samples"

    rng = np.random.default_rng(0)
    conf = rng.random((20, 32, 32)) * 4.0 + 1.0
    # (b) 残差处处相同 → 无从判单调
    same = mg.conf_warp_monotonic(conf, np.full(conf.shape, 0.25))
    assert same["monotonic"] is None and same["reason"] == "degenerate_residuals"
    # (c) 全 NaN / 全非有限
    nan_case = mg.conf_warp_monotonic(np.full(conf.shape, np.nan), np.full(conf.shape, np.nan))
    assert nan_case["monotonic"] is None and nan_case["reason"] == "insufficient_samples"
    # (d) conf 常数 → 只有一个桶
    const = mg.conf_warp_monotonic(np.full(conf.shape, 3.0), rng.random(conf.shape))
    assert const["monotonic"] is None and const["reason"] == "conf_constant"

    # None 进主门 → 降权，但不否决
    gate = mg.main_gate({"warp_inlier_ratio": 0.9, "cloud_overlap_ratio": 0.9,
                         "conf_warp_monotonic": None})
    assert gate["conf_weight"] == mg.TH_CONF_DOWNWEIGHT
    assert gate["main_gate_passed"] is True
    assert any("证据不足" in w for w in gate["warnings"])

    # 形状不一致是接线 bug → 显式抛错（不是静默 NaN）
    with pytest.raises(ValueError):
        mg.conf_warp_monotonic(np.ones((2, 4, 4)), np.ones((2, 4, 5)))


# --------------------------------------------------------------------------- #
# 6. 报率不报绝对距离 / 尺度不变 / 确定性
# --------------------------------------------------------------------------- #

def test_cloud_overlap_is_scale_invariant():
    """VGGT 点云无米制尺度：整体乘任意倍数不得改变重叠率（判据是相对阈值）。"""
    s = _perfect(8)
    base = mg.grouped_cloud_overlap(s["point_map"])
    for factor in (1e-3, 1e1, 1e3):
        scaled = mg.grouped_cloud_overlap(s["point_map"] * factor)
        assert scaled["cloud_overlap_ratio"] == pytest.approx(
            base["cloud_overlap_ratio"], abs=1e-9)
        assert scaled["n_fwd"] == base["n_fwd"] and scaled["n_bwd"] == base["n_bwd"]
    assert 0.0 <= base["cloud_overlap_ratio"] <= 1.0
    assert base["n_fwd"] == base["n_bwd"]                            # 32 帧 → 16 vs 16


def test_cloud_overlap_requires_two_frames_and_rate_only():
    pm = np.zeros((1, 8, 8, 3))
    ov = mg.grouped_cloud_overlap(pm)
    assert np.isnan(ov["cloud_overlap_ratio"]) and ov["reason"] == "n_frames_lt_2"
    assert ov["n_fwd"] == 0 and ov["n_bwd"] == 0


def test_determinism_byte_identical():
    """同一输入两次 → 逐位一致（含下采样路径：max_points 故意压到触发 rng.choice）。"""
    s = _perfect(8)
    runs = []
    for _ in range(2):
        warp = mg.cross_view_warp_inliers(s["frames"], s["depth_maps"],
                                          s["c2w_list"], s["intrinsics"])
        ov = mg.grouped_cloud_overlap(s["point_map"], max_points=500)   # 触发固定种子抽样
        resid = mg.warp_residual_map(s["depth_maps"], s["c2w_list"], s["intrinsics"])
        mono = mg.conf_warp_monotonic(np.abs(resid) * 10.0 + 1.0, np.nan_to_num(resid, nan=0.0))
        gate = mg.main_gate(s)
        runs.append(json.dumps({"warp": warp, "ov": ov, "mono": mono, "gate": gate},
                               sort_keys=True))
    assert runs[0] == runs[1]
    assert 0 < mg.grouped_cloud_overlap(s["point_map"], max_points=500)["n_fwd"] <= 500


def test_warp_residual_map_shape_and_nan():
    s = _perfect(4)
    resid = mg.warp_residual_map(s["depth_maps"], s["c2w_list"], s["intrinsics"])
    assert resid.shape == s["depth_maps"].shape
    fin = np.isfinite(resid)
    assert fin.any() and np.all(resid[fin] >= 0.0)
    assert np.isnan(resid[~fin]).all()                              # 无观测 → NaN，不是 0
    # 残差口径与 warp 内点率自洽：以 rel_depth_tol 截断后占比 ≈ 深度内点率
    warp = mg.cross_view_warp_inliers(s["frames"], s["depth_maps"], s["c2w_list"], s["intrinsics"])
    frac_in = float(np.count_nonzero(fin & (resid <= mg.TH_WARP_REL_DEPTH_TOL))) / float(fin.sum())
    assert abs(frac_in - warp["warp_inlier_ratio"]) < 0.05


# --------------------------------------------------------------------------- #
# 7. 阈值口径 / 性能
# --------------------------------------------------------------------------- #

def test_threshold_overrides_and_unknown_keys():
    default = mg.default_thresholds()
    assert default == {"warp_inlier_ratio": mg.TH_WARP_INLIER,
                       "cloud_overlap_ratio": mg.TH_CLOUD_OVERLAP}
    gate = mg.main_gate({"warp_inlier_ratio": 0.6, "cloud_overlap_ratio": 0.4},
                        thresholds={"warp_inlier_ratio": 0.7, "cloud_overlap_ratio": 0.4,
                                    "gate_thresholds_snapshot": 1.0})
    assert gate["thresholds"]["warp_inlier_ratio"] == 0.7
    assert gate["sub_results"]["warp_inlier_ratio"] is False
    assert gate["sub_results"]["cloud_overlap_ratio"] is True
    assert gate["main_gate_passed"] is False
    assert any("未知键" in w for w in gate["warnings"])
    # 非有限覆盖 → 忽略并告警（不静默改门）
    gate = mg.main_gate({"warp_inlier_ratio": 0.6, "cloud_overlap_ratio": 0.6},
                        thresholds={"warp_inlier_ratio": float("nan")})
    assert gate["thresholds"]["warp_inlier_ratio"] == mg.TH_WARP_INLIER
    assert gate["main_gate_passed"] is True
    assert any("非有限" in w for w in gate["warnings"])
    assert gate["gate_version"] == "v6-warp-overlap-no-g5"


def test_all_thresholds_annotated_todo_calibrate():
    """硬要求：每个阈值常量都必须带 TODO_CALIBRATE 注释（可标定、不是拍脑袋的定论）。"""
    import inspect

    src = inspect.getsource(mg)
    for name in ("TH_WARP_INLIER", "TH_CLOUD_OVERLAP", "TH_WARP_REL_DEPTH_TOL",
                 "TH_WARP_PHOTO_TOL", "TH_CLOUD_REL_TOL", "MAX_CLOUD_POINTS",
                 "TH_CONF_DOWNWEIGHT", "CONF_OPTIONAL_MASK_C", "TH_PHOTO_WARN"):
        line = next(ln for ln in src.splitlines() if ln.startswith(f"{name}:"))
        assert "TODO_CALIBRATE" in line, f"{name} 缺 TODO_CALIBRATE 标注"


def test_performance_518x392_32frames():
    """§10 规模验收：32 帧 @518×392 主门全流程须在秒级跑完（CPU）。"""
    import time

    s = _build_scene(32, hw=(392, 518))
    t0 = time.perf_counter()
    gate = mg.main_gate(s)
    elapsed = time.perf_counter() - t0
    assert gate["main_gate_passed"] is True
    assert gate["values"]["warp_inlier_ratio"] > 0.8
    assert gate["values"]["cloud_overlap_ratio"] > 0.5
    assert elapsed < 30.0, f"主门耗时 {elapsed:.1f}s，超出秒级预算"
