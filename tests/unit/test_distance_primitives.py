"""§12 稳健低分位距离原语单测（纯 numpy/scipy，无 Tool / 无重建产物）。

覆盖规格要求的十条：解析分位、飞点稳健性、体素保低分位、`N_min` 降级、
污染判据、`suspect_duplicate` 传播、米制换算、NaN/Inf 剔除、确定性、
以及 20k×20k 双向 NN 的性能冒烟（< 5 s）；另加 §9.4/§9.5 的 extent/房间几何
与 §12.5 消融表。

全部算例自造（无外部数据、无网络、无随机种子漂移）。
"""

from __future__ import annotations

import time

import numpy as np
import pytest


def _load_dp():
    """导入被测模块：优先走正常包路径。

    `skill3d.tools.__init__` 会**急切导入并注册全部 Tool**（含 `geometry_tools`）；
    工具注册表正处在并行改造中，一旦它的中间态不可导入，本纯 numpy 原语模块不该被连坐，
    故退化为按文件路径加载（模块自身只依赖 numpy/scipy/度量融合版本号）。
    """
    try:
        from skill3d.tools import distance_primitives as mod
        return mod
    except Exception:  # pragma: no cover - 仅在注册表中间态触发
        import importlib.util
        import sys
        from pathlib import Path

        path = (Path(__file__).resolve().parents[2] / "src" / "skill3d" / "tools"
                / "distance_primitives.py")
        spec = importlib.util.spec_from_file_location(
            "distance_primitives_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod          # dataclass 需要能在 sys.modules 里找到自己
        spec.loader.exec_module(mod)
        return mod


dp = _load_dp()

# v6 里 conf 纪律与可选掩码的**唯一事实源**是 M4 主门模块，这里交叉校验常量一致。
from skill3d.reconstruction_gate.m4_main_gate import (  # noqa: E402
    CONF_OPTIONAL_MASK_C as M4_CONF_OPTIONAL_MASK_C,
)


# ---------------------------------------------------------------------------
# 造数工具（确定性）
# ---------------------------------------------------------------------------

def fib_sphere(n: int, radius: float) -> np.ndarray:
    """斐波那契球壳（确定性、近似均匀）—— 用于"已知距离"的解析算例。"""
    i = np.arange(n, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * i / max(n - 1, 1)
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    theta = phi * i
    return np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1) * radius


def line_cloud(n: int, r0: float, r1: float) -> np.ndarray:
    """沿 +x 轴、半径从 r0 线性排到 r1 的点（到原点的距离可知，便于解析核对）。"""
    radii = np.linspace(r0, r1, n)
    return np.stack([radii, np.zeros(n), np.zeros(n)], axis=1)


def box_room(n_floor=6000, n_ceil=6000, n_wall=12000, size=(5.0, 4.0), height=2.4):
    """轴对齐长方体房间点云（地面 + 天花 + 四墙）：竖直轴 = y，房间尺寸 5×4×2.4。

    采样确定性（`default_rng(0)` 固定种子）：仅用于造**输入数据**，
    被测函数的确定性不依赖它。
    """
    rng = np.random.default_rng(0)
    w, d = float(size[0]), float(size[1])
    h = float(height)
    x = rng.uniform(-w / 2, w / 2, n_floor)
    z = rng.uniform(-d / 2, d / 2, n_floor)
    floor = np.stack([x, np.zeros(n_floor), z], axis=1)
    x = rng.uniform(-w / 2, w / 2, n_ceil)
    z = rng.uniform(-d / 2, d / 2, n_ceil)
    ceil = np.stack([x, np.full(n_ceil, h), z], axis=1)
    n4 = int(n_wall // 4)
    parts = []
    for wall_idx in range(4):
        y = rng.uniform(0.0, h, n4)
        t = rng.uniform(-(d / 2) if wall_idx < 2 else -(w / 2),
                        (d / 2) if wall_idx < 2 else (w / 2), n4)
        if wall_idx == 0:
            parts.append(np.stack([np.full(n4, -w / 2), y, t], axis=1))
        elif wall_idx == 1:
            parts.append(np.stack([np.full(n4, w / 2), y, t], axis=1))
        else:
            sign = -1.0 if wall_idx == 2 else 1.0
            parts.append(np.stack([t, y, np.full(n4, sign * d / 2)], axis=1))
    return np.concatenate([floor, ceil, *parts], axis=0)


def prep_off(n_min: int = 1, **kw):
    """关掉体素降采样（解析算例用；默认 0.01 的体素会把细密点集压掉）。"""
    return dp.DistancePrimitiveParams(voxel_size=0.0, n_min=n_min, **kw)


# ---------------------------------------------------------------------------
# 1) 解析分位：q 精确等于分位定义；q=0 等于真最小值
# ---------------------------------------------------------------------------

def test_quantile_matches_analytic_and_q0_is_true_minimum():
    pts = line_cloud(2000, 1.0, 3.0)
    radii = np.linalg.norm(pts, axis=1)
    for q in (0.0, 0.005, 0.01, 0.02, 0.5, 0.9):
        res = dp.robust_distance_to_reference(
            pts, [0.0, 0.0, 0.0], params=prep_off(quantile_q=q))
        expected = float(np.quantile(radii, q, method="linear"))
        assert res.degradation_flags == []
        assert res.n_valid_points == 2000
        assert res.distance_normalized == pytest.approx(expected, abs=1e-12)
    # q=0 必须**精确**等于真最近点（稳健低分位的极限情形，§12.1）
    res0 = dp.robust_distance_to_reference(pts, [0.0, 0.0, 0.0], params=prep_off(quantile_q=0.0))
    assert res0.distance_normalized == pytest.approx(1.0, abs=1e-12)
    assert res0.distance_normalized == float(np.min(radii))


def test_default_quantile_q_is_one_percent():
    """§12.1：q 默认候选 1%，仅 inner 选择 outer 冻结。"""
    assert dp.QUANTILE_Q_DEFAULT == pytest.approx(0.01)
    assert dp.DistancePrimitiveParams().snapshot() == {
        "quantile_q": 0.01,
        "voxel_size": dp.VOXEL_SIZE_DEFAULT,
        "conf_warp_version": "conf-warp-v6",
        "n_min": dp.N_MIN_DEFAULT,
    }


# ---------------------------------------------------------------------------
# 2) 飞点：低分位稳、分位/均值被拖走（本模块的核心主张）
# ---------------------------------------------------------------------------

def test_flying_points_leave_low_quantile_near_surface():
    surface = fib_sphere(2000, 1.0)                     # 表面：到参考点距离恒为 1
    flyers = fib_sphere(50, 50.0)                       # 飞点：距离 50
    pts = np.concatenate([surface, flyers], axis=0)
    ref = [0.0, 0.0, 0.0]

    rob = dp.robust_distance_to_reference(pts, ref, params=prep_off(quantile_q=0.01))
    med = dp.robust_distance_to_reference(pts, ref, params=prep_off(quantile_q=0.5))
    mean_d = float(np.mean(np.linalg.norm(pts - np.asarray(ref), axis=1)))

    assert rob.distance_normalized is not None
    assert rob.distance_normalized <= 1.05              # 低分位贴近真实表面
    assert abs(med.distance_normalized - 1.0) < 0.5     # 中位数仍在表面附近但被推高
    assert mean_d > 2.0                                 # 均值被飞点彻底带走
    # 低分位必须显著优于中位数/均值（本模块存在的理由）
    assert rob.distance_normalized < med.distance_normalized < mean_d


# ---------------------------------------------------------------------------
# 3) 体素降采样保低分位
# ---------------------------------------------------------------------------

def test_voxel_downsampling_preserves_low_quantile():
    # 斜平面上的稠密网格（距离随位置缓慢变化 → 降采样不应改变低分位）
    g = np.linspace(-1.0, 1.0, 200)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel(),
                    0.5 + 0.3 * xx.ravel() + 0.2 * yy.ravel()], axis=1)
    ref = [0.0, 0.0, 0.0]
    full = dp.robust_distance_to_reference(
        pts, ref, params=dp.DistancePrimitiveParams(voxel_size=0.0, n_min=1, quantile_q=0.01))
    ds = dp.robust_distance_to_reference(
        pts, ref, params=dp.DistancePrimitiveParams(voxel_size=0.05, n_min=1, quantile_q=0.01))
    assert ds.audit["voxel_applied"] is True
    assert ds.n_valid_points < full.n_valid_points / 5      # 确实压缩了
    assert ds.distance_normalized == pytest.approx(
        full.distance_normalized, rel=0.05)                 # 低分位保持（±5%）
    # 体素降采样后点集仍应"贴在同一张面上"：点仍来自原集合
    assert ds.n_valid_points > 0


def test_voxel_downsampling_is_deterministic_and_keeps_real_points():
    pts = fib_sphere(3000, 1.0)
    a = dp.preprocess_points(pts, params=dp.DistancePrimitiveParams(voxel_size=0.05))
    b = dp.preprocess_points(pts, params=dp.DistancePrimitiveParams(voxel_size=0.05))
    assert np.array_equal(a["points"], b["points"])
    # 取的是"离质心最近的真实点"：每个输出点都能在原集合里找到
    s1 = {tuple(np.round(p, 12)) for p in pts}
    assert all(tuple(np.round(p, 12)) in s1 for p in a["points"])


# ---------------------------------------------------------------------------
# 4) N_min 违规 → degraded（且没有伪精确值）
# ---------------------------------------------------------------------------

def test_n_min_violation_degrades_without_pseudo_value():
    pts = fib_sphere(dp.N_MIN_DEFAULT - 1, 1.0)         # 差一个点
    res = dp.robust_distance_to_reference(pts, [0, 0, 0],
                                          params=dp.DistancePrimitiveParams(voxel_size=0.0))
    assert "degraded" in res.degradation_flags
    assert res.distance_normalized is None
    assert res.distance_metric is None
    assert res.n_valid_points == dp.N_MIN_DEFAULT - 1
    assert res.to_trace()["distance_normalized"] is None
    # 恰好达标（= N_min）→ 不降级
    ok = dp.robust_distance_to_reference(
        fib_sphere(dp.N_MIN_DEFAULT, 1.0), [0, 0, 0],
        params=dp.DistancePrimitiveParams(voxel_size=0.0))
    assert "degraded" not in ok.degradation_flags
    assert ok.distance_normalized is not None


def test_empty_point_set_is_degraded_not_zero():
    res = dp.robust_distance_to_reference(np.zeros((0, 3)), [1, 2, 3])
    assert res.degradation_flags == ["degraded"]
    assert res.distance_normalized is None and res.distance_metric is None
    assert res.n_valid_points == 0


# ---------------------------------------------------------------------------
# 5) 污染：互相咬合/重合的点集 → point_contamination_suspect（且不输出精确值）
# ---------------------------------------------------------------------------

def test_interpenetrating_sets_flagged_as_contamination():
    a = fib_sphere(1000, 1.0)
    b = a + np.array([1e-4, 0.0, 0.0])                  # 平移一个远小于采样间距的量
    res = dp.robust_distance_between_pointsets(
        a, b, params=dp.DistancePrimitiveParams(voxel_size=0.0))
    assert "point_contamination_suspect" in res.degradation_flags
    assert res.distance_normalized is None               # §12.4：污染不作精确值输出
    assert res.suppressed_distance_normalized is not None
    assert res.contamination_ratio is not None and res.contamination_ratio > 0.5
    assert res.n_nn_samples == 2000


def test_well_separated_sets_not_flagged():
    a = fib_sphere(1000, 1.0)
    b = fib_sphere(1000, 1.0) + np.array([10.0, 0.0, 0.0])   # 间距远大于采样间距
    res = dp.robust_distance_between_pointsets(
        a, b, params=dp.DistancePrimitiveParams(voxel_size=0.0))
    assert "point_contamination_suspect" not in res.degradation_flags
    # 两个半径 1、中心相距 10 的球壳 → 表面间距 ≈ 10 − 1 − 1 = 8
    assert res.distance_normalized == pytest.approx(8.0, abs=0.2)
    assert res.n_nn_samples == 2000


def test_between_pointsets_uses_bidirectional_nn_not_cartesian():
    """双向 NN：两集合点数不等时样本数 = n_A + n_B（而不是 n_A×n_B）。"""
    a = fib_sphere(500, 1.0)
    b = fib_sphere(700, 1.0) + np.array([3.0, 0.0, 0.0])
    res = dp.robust_distance_between_pointsets(
        a, b, params=dp.DistancePrimitiveParams(voxel_size=0.0))
    assert res.n_nn_samples == 1200
    assert res.n_valid_points == 1200
    assert res.audit["surface_distance_note"].startswith("双向 NN")


# ---------------------------------------------------------------------------
# 6) suspect_duplicate 传播
# ---------------------------------------------------------------------------

def test_suspect_duplicate_propagates():
    pts = fib_sphere(500, 1.0)
    res = dp.robust_distance_to_reference(
        pts, [0, 0, 0], params=prep_off(), duplicate_suspect=True)
    assert "suspect_duplicate" in res.degradation_flags
    assert res.duplicate_suspect is True
    assert "suspect_duplicate" in res.to_trace()["degradation_flags"]
    # 未指定时不应出现该标记（不许凭空加降级）
    clean = dp.robust_distance_to_reference(pts, [0, 0, 0], params=prep_off())
    assert "suspect_duplicate" not in clean.degradation_flags
    # 点集↔点集同路径
    res2 = dp.robust_distance_between_pointsets(
        pts, fib_sphere(500, 1.0) + 5.0, params=prep_off(), duplicate_suspect=True)
    assert "suspect_duplicate" in res2.degradation_flags


# ---------------------------------------------------------------------------
# 7) 米制尺度：有系数才给米制值，且换算正确（面积按平方）
# ---------------------------------------------------------------------------

def test_metric_scale_conversion_and_absence():
    pts = line_cloud(500, 1.0, 2.0)
    ref = [0.0, 0.0, 0.0]
    no_metric = dp.robust_distance_to_reference(pts, ref, params=prep_off())
    with_metric = dp.robust_distance_to_reference(pts, ref, metric_scale=2.0, params=prep_off())
    assert no_metric.distance_metric is None             # 缺系数 → fail-closed（不默认 1.0）
    assert no_metric.scale_version is None
    assert with_metric.distance_metric == pytest.approx(2.0 * with_metric.distance_normalized)
    assert with_metric.scale_version == dp.SCALE_VERSION_DEFAULT
    # 非法系数（非正/非有限）→ 不出米制值 + 显式标记
    bad = dp.robust_distance_to_reference(pts, ref, metric_scale=-1.0, params=prep_off())
    assert bad.distance_metric is None
    assert "metric_scale_invalid" in bad.degradation_flags
    assert bad.distance_normalized is not None           # 归一化值不受影响


def test_object_extent_metric_and_area_squared():
    rng = np.random.default_rng(1)
    box = np.stack([rng.uniform(0, 1.0, 8000), rng.uniform(0, 0.5, 8000),
                    rng.uniform(0, 0.3, 8000)], axis=1)
    norm = dp.object_extent(box, params=dp.DistancePrimitiveParams(voxel_size=0.0))
    metric = dp.object_extent(box, metric_scale=2.0, params=dp.DistancePrimitiveParams(voxel_size=0.0))
    assert norm["extent_metric"] is None and norm["area_metric"] is None
    assert norm["n_valid_points"] == 8000
    ext = np.asarray(metric["extent_normalized"])
    assert np.allclose(np.asarray(metric["extent_metric"]), 2.0 * ext)
    # 稳健 extent ≈ 真实边长（q=0.01 两端各切 1%：均匀分布下 ≈ 0.98 倍，容差 3%）
    assert np.all(np.abs(ext - np.array([1.0, 0.5, 0.3])) < 0.03)
    assert metric["area_metric"] == pytest.approx(norm["area_normalized"] * 4.0)  # 面积按平方
    assert metric["extent_longest_metric"] == pytest.approx(2.0 * norm["extent_longest_normalized"])


# ---------------------------------------------------------------------------
# 8) NaN/Inf 剔除，且不崩
# ---------------------------------------------------------------------------

def test_nan_inf_points_excluded_without_crash():
    pts = fib_sphere(600, 1.0)
    dirty = np.concatenate([pts, np.array([[np.nan, 0.0, 0.0],
                                           [np.inf, 0.0, 0.0],
                                           [0.0, -np.inf, 2.0]])], axis=0)
    clean = dp.robust_distance_to_reference(pts, [0, 0, 0], params=prep_off())
    dirty_res = dp.robust_distance_to_reference(dirty, [0, 0, 0], params=prep_off())
    assert dirty_res.n_valid_points == 600
    assert dirty_res.n_points_raw == 603
    assert dirty_res.audit["n_drop_nonfinite"] == 3
    assert dirty_res.distance_normalized == pytest.approx(clean.distance_normalized)
    # 密度图输入 (H,W,3) 也接受
    grid = pts[:600].reshape(20, 30, 3)
    assert dp.robust_distance_to_reference(
        grid, [0, 0, 0], params=prep_off()).n_valid_points == 600


def test_invalid_reference_fails_closed():
    with pytest.raises(ValueError):
        dp.robust_distance_to_reference(fib_sphere(200, 1.0), [np.nan, 0.0, 0.0])
    with pytest.raises(ValueError):
        dp.robust_distance_to_reference(fib_sphere(200, 1.0), [0.0, 0.0])


# ---------------------------------------------------------------------------
# 9) 确定性
# ---------------------------------------------------------------------------

def test_determinism_same_input_same_result():
    pts = box_room()
    ref = [0.0, 0.5, 0.0]
    a = dp.robust_distance_to_reference(pts, ref)
    b = dp.robust_distance_to_reference(pts, ref)
    assert a == b
    assert a.to_trace() == b.to_trace()
    aa = dp.robust_distance_to_reference(pts, ref, metric_scale=1.7)
    bb = dp.robust_distance_to_reference(pts, ref, metric_scale=1.7)
    assert aa.to_trace() == bb.to_trace()
    e1 = dp.object_extent(pts)
    e2 = dp.object_extent(pts)
    assert e1 == e2
    s1 = dp.robust_distance_between_pointsets(pts, fib_sphere(400, 1.0) + 6.0)
    s2 = dp.robust_distance_between_pointsets(pts, fib_sphere(400, 1.0) + 6.0)
    assert s1 == s2
    r1 = dp.room_size_from_planes(pts)
    r2 = dp.room_size_from_planes(pts)
    assert r1 == r2


def test_params_accepts_dict_and_validates():
    assert dp.DistancePrimitiveParams(quantile_q=0.02).snapshot()["quantile_q"] == 0.02
    d = dp.robust_distance_to_reference(
        fib_sphere(300, 1.0), [0, 0, 0], params={"voxel_size": 0.0, "n_min": 1})
    assert d.voxel_size == 0.0 and d.n_valid_points == 300
    with pytest.raises(ValueError):
        dp.DistancePrimitiveParams(quantile_q=1.5)
    with pytest.raises(TypeError):
        dp.robust_distance_to_reference(fib_sphere(300, 1.0), [0, 0, 0], params=object())


# ---------------------------------------------------------------------------
# 10) 性能冒烟：20k × 20k 双向 NN（不许出现全笛卡尔积）
# ---------------------------------------------------------------------------

def test_performance_smoke_20k_bidirectional_nn_under_5s():
    """冒烟测试：20k×20k 双向 NN（KD-tree）远快于 5 s；若退化成笛卡尔积必然超时。"""
    n = 20000
    rng = np.random.default_rng(7)
    a = rng.uniform(-1.0, 1.0, size=(n, 3))
    b = rng.uniform(-1.0, 1.0, size=(n, 3))
    t0 = time.monotonic()
    res = dp.robust_distance_between_pointsets(
        a, b, params=dp.DistancePrimitiveParams(voxel_size=0.0))
    elapsed = time.monotonic() - t0
    assert res.n_nn_samples == 2 * n        # 双向各 n 次查询，而不是 n×n 笛卡尔积
    assert res.distance_normalized is not None
    assert elapsed < 5.0, f"20k×20k 双向 NN 用时 {elapsed:.2f}s，超出冒烟上限 5s"


def test_performance_smoke_20k_single_reference_under_5s():
    n = 20000
    rng = np.random.default_rng(8)
    a = rng.uniform(-1.0, 1.0, size=(n, 3))
    t0 = time.monotonic()
    res = dp.robust_distance_to_reference(
        a, [0.0, 0.0, 0.0], params=dp.DistancePrimitiveParams(voxel_size=0.0))
    elapsed = time.monotonic() - t0
    assert res.distance_normalized is not None
    assert elapsed < 5.0, f"20k 点→参考点用时 {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# §12.5 消融表：形状 + 单调性 + 相对官方最近点的偏差
# ---------------------------------------------------------------------------

def test_ablation_quantiles_table_shape_and_monotonicity():
    problems = [
        {"name": "sphere1", "points": fib_sphere(2000, 1.0), "reference_xyz": [0, 0, 0]},
        {"name": "sphere2", "points": fib_sphere(2000, 1.0) + 4.0, "reference_xyz": [0, 0, 0]},
    ]
    table = dp.ablation_quantiles(
        problems, params=dp.DistancePrimitiveParams(voxel_size=0.0))
    assert table["quantiles"] == [0.0, 0.005, 0.01, 0.02, 0.05]
    assert len(table["summary"]) == 5
    assert len(table["rows"]) == 2 * 5
    assert set(table["params"]) == {"quantile_q", "voxel_size", "conf_warp_version", "n_min"}
    assert table["official_reference"]["requested"] == "min_q0"
    for row in table["rows"]:
        assert row["n_valid_points"] == 2000
        assert row["deviation_abs"] is not None
        assert row["deviation_abs"] >= 0.0
    # 逐样本：距离随 q 单调不减（分位数性质）
    for name in ("sphere1", "sphere2"):
        vals = [r["distance_normalized"] for r in table["rows"] if r["name"] == name]
        assert vals == sorted(vals)
    # 汇总：偏差随 q 单调不减（干净点集上，q 越大离最近点越远）
    devs = [s["mean_deviation_abs"] for s in table["summary"]]
    assert devs == sorted(devs)
    assert table["summary"][0]["mean_deviation_abs"] == pytest.approx(0.0, abs=1e-12)
    assert all(s["n_degraded"] == 0 for s in table["summary"])


def test_ablation_quantiles_uses_provided_official_reference():
    pts = line_cloud(1000, 1.0, 3.0)
    table = dp.ablation_quantiles(
        [{"name": "line", "points": pts, "reference_xyz": [0, 0, 0],
          "official_nearest": 1.25}],
        params=dp.DistancePrimitiveParams(voxel_size=0.0), official_reference="provided")
    row0 = [r for r in table["rows"] if r["q"] == 0.0][0]
    assert row0["official_nearest_normalized"] == pytest.approx(1.25)
    assert row0["official_reference_source"] == "provided"
    assert row0["deviation_abs"] == pytest.approx(abs(1.0 - 1.25))
    assert "provided" in table["official_reference"]["source"]
    # 单个样本（dict）也能跑
    single = dp.ablation_quantiles(
        {"points": pts, "reference_xyz": [0, 0, 0]},
        params=dp.DistancePrimitiveParams(voxel_size=0.0))
    assert len(single["summary"]) == 5


def test_ablation_quantiles_marks_degraded_samples_without_faking():
    small = fib_sphere(10, 1.0)                     # < N_min
    table = dp.ablation_quantiles(
        [{"name": "tiny", "points": small, "reference_xyz": [0, 0, 0]}],
        params=dp.DistancePrimitiveParams(voxel_size=0.0, n_min=dp.N_MIN_DEFAULT))
    assert all(r["distance_normalized"] is None for r in table["rows"])
    assert all("degraded" in r["degradation_flags"] for r in table["rows"])
    assert all(s["n_degraded"] == 1 and s["n_measured"] == 0 for s in table["summary"])
    assert all(s["mean_deviation_abs"] is None for s in table["summary"])


# ---------------------------------------------------------------------------
# §12.2-2 conf 纪律（软权重/掩码，绝不当硬门）
# ---------------------------------------------------------------------------

def test_conf_discipline_monotonicity_gate():
    pts = line_cloud(1000, 1.0, 3.0)                    # 半径 1..3
    ref = [0.0, 0.0, 0.0]
    conf = np.where(np.arange(1000) < 500, 0.1, 5.0)    # 近端低 conf、远端高 conf
    p = dp.DistancePrimitiveParams(voxel_size=0.0, quantile_q=0.05)
    unverified = dp.robust_distance_to_reference(pts, ref, params=p, point_conf=conf,
                                                 conf_warp_monotonic=None)
    passed = dp.robust_distance_to_reference(pts, ref, params=p, point_conf=conf,
                                             conf_warp_monotonic=True)
    failed = dp.robust_distance_to_reference(pts, ref, params=p, point_conf=conf,
                                             conf_warp_monotonic=False)
    # None（未自检）→ 完全不用 conf：与无 conf 的结果完全一致
    plain = dp.robust_distance_to_reference(pts, ref, params=p)
    assert unverified.conf_usage == "ignored_unverified"
    assert unverified.distance_normalized == pytest.approx(plain.distance_normalized)
    # True → 软权重，把分位拉向高 conf（远）端
    assert passed.conf_usage == "weighted"
    assert passed.distance_normalized > plain.distance_normalized
    # False → 只降权（不丢弃点），效果弱于 True 但仍高于无权重
    assert failed.conf_usage == "downweighted"
    assert failed.n_valid_points == plain.n_valid_points
    assert plain.distance_normalized <= failed.distance_normalized <= passed.distance_normalized
    # C>2 只能作**可选掩码**，且只在自检通过时才允许启用
    masked = dp.robust_distance_to_reference(pts, ref, params=p, point_conf=conf,
                                             conf_warp_monotonic=True,
                                             conf_optional_mask=True)
    assert masked.conf_usage == "masked_optional"
    assert masked.n_valid_points == 500
    blocked = dp.robust_distance_to_reference(pts, ref, params=p, point_conf=conf,
                                              conf_warp_monotonic=False,
                                              conf_optional_mask=True)
    assert blocked.conf_usage == "downweighted" and blocked.n_valid_points == 1000


def test_conf_optional_mask_cross_checks_m4_constant():
    """§10.3 `C>2` 可选掩码的数值必须与 M4 主门同口径（单一事实源交叉校验）。"""
    assert dp.CONF_OPTIONAL_MASK_C == M4_CONF_OPTIONAL_MASK_C


# ---------------------------------------------------------------------------
# §9.5 房间几何 / §9.4 extent 的端到端算例
# ---------------------------------------------------------------------------

def test_planarity_and_ground_on_synthetic_room():
    room = box_room()
    g = dp.planarity_and_ground(room)
    assert 0 < g["n_valid_points"] <= room.shape[0]
    assert g["n_valid_points"] > 0.95 * room.shape[0]          # 体素只压掉极少数重合格
    assert g["up_source"].startswith("pca_min_variance")
    assert g["up_axis"] == 1                                   # 竖直轴 = y
    assert g["up"][1] == pytest.approx(1.0, abs=1e-6)          # 已被地面平面法向精修
    assert "up_axis_ambiguous" not in g["degradation_flags"]
    assert g["ground_plane"] is not None
    assert abs(g["ground_plane"]["normal"][1]) > 0.99          # 地面法向 ≈ 上方向
    assert g["ground_plane"]["inlier_ratio"] > 0.2
    assert g["floor_height_normalized"] == pytest.approx(0.0, abs=0.02)
    assert g["ceiling_height_normalized"] == pytest.approx(2.4, abs=0.02)
    assert g["room_height_normalized"] == pytest.approx(2.4, abs=0.03)
    assert g["ground_planarity"] > 0.9
    ext = sorted(g["horizontal_extent_normalized"])
    assert ext == pytest.approx([4.0, 5.0], rel=0.05)
    # `plane_inlier_ratio` 是**地面**平面内点率（真实房间里通常 0.2–0.5）；
    # 墙内点率与"地面∪墙"的联合覆盖率另见 wall_inlier_ratio / audit
    assert 0.2 < g["plane_inlier_ratio"] < 0.5
    assert g["wall_inlier_ratio"] > 0.2
    assert g["audit"]["room_surface_inlier_ratio"] > 0.6
    assert g["fit_quality"] is not None and g["fit_quality"] > dp.TH_FIT_QUALITY_MIN


def test_planarity_and_ground_accepts_up_hint_and_degrades_on_too_few_points():
    room = box_room(n_floor=200, n_ceil=200, n_wall=400)
    hint = dp.planarity_and_ground(room, up_hint=[0.0, 1.0, 0.0])
    assert hint["up_source"].startswith("provided")
    assert hint["up"][1] == pytest.approx(1.0)
    assert np.isfinite(hint["fit_quality"])
    tiny = dp.planarity_and_ground(room[:10])
    assert "degraded" in tiny["degradation_flags"]
    assert tiny["ground_plane"] is None and tiny["up"] is None
    assert tiny["fit_quality"] is None
    with pytest.raises(ValueError):
        dp.planarity_and_ground(room, up_hint=[0.0, 0.0, 0.0])


def test_room_size_from_planes_diagonal_area_and_metric():
    room = box_room()
    norm = dp.room_size_from_planes(room)
    metric = dp.room_size_from_planes(room, metric_scale=1.0)
    bare = dp.room_size_from_planes(room, metric_scale=2.0)
    assert norm["room_area_m2"] is None and norm["room_diagonal_metric"] is None
    assert norm["room_diagonal_normalized"] == pytest.approx(
        float(np.hypot(5.0, 4.0)), rel=0.06)
    assert norm["room_area_normalized"] == pytest.approx(20.0, rel=0.12)
    assert metric["room_area_m2"] == pytest.approx(norm["room_area_normalized"])
    # 面积按**平方**换算（不是线性 ×2）
    assert bare["room_area_m2"] == pytest.approx(norm["room_area_normalized"] * 4.0)
    assert bare["room_diagonal_metric"] == pytest.approx(norm["room_diagonal_normalized"] * 2.0)
    assert "degraded" not in norm["degradation_flags"]
    assert norm["plane_inlier_ratio"] > 0.2                    # 地面平面内点率
    assert norm["fit_quality"] is not None
    for key in ("quantile_q", "voxel_size", "conf_warp_version", "scale_version",
                "n_valid_points", "degradation_flags"):
        assert key in norm
    assert norm["scale_version"] is None
    assert metric["scale_version"] == dp.SCALE_VERSION_DEFAULT


def test_room_size_degrades_on_point_blob_without_floor():
    rng = np.random.default_rng(3)
    blob = rng.normal(0.0, 1.0, size=(2000, 3))
    out = dp.room_size_from_planes(blob)
    # 球状团块没有"地面"：要么显式降级、要么给出低拟合质量标记（不许假装是好房间）
    assert out["fit_quality"] is None or "plane_fit_low_quality" in out["degradation_flags"]
    assert out["room_area_m2"] is None                       # 无米制系数 → 永不给 m²


# ---------------------------------------------------------------------------
# §12.5 trace 字段完整性
# ---------------------------------------------------------------------------

def test_trace_fields_follow_section_12_5():
    res = dp.robust_distance_to_reference(
        fib_sphere(500, 1.0), [0, 0, 0], metric_scale=1.5, params=prep_off())
    trace = res.to_trace()
    required = {"distance_normalized", "distance_metric", "scale_version", "quantile_q",
                "voxel_size", "conf_warp_version", "n_valid_points", "degradation_flags"}
    assert required <= set(trace)
    assert trace["conf_warp_version"] == "conf-warp-v6"
    assert trace["quantile_q"] == dp.QUANTILE_Q_DEFAULT
    assert trace["voxel_size"] == 0.0
    assert trace["scale_version"] == dp.SCALE_VERSION_DEFAULT
    assert isinstance(trace["degradation_flags"], list)


def test_preprocess_points_columns_and_flags():
    prep = dp.preprocess_points(fib_sphere(400, 1.0))
    assert prep["points"].shape[1] == 3
    assert prep["weights"] is None and prep["conf_usage"] == "disabled"
    assert prep["degraded"] is False
    assert np.isfinite(prep["robust_scale"])
    bad = dp.preprocess_points(fib_sphere(5, 1.0), params=dp.DistancePrimitiveParams(voxel_size=0.0))
    assert bad["degraded"] is True and bad["degradation_flags"] == ["degraded"]


def test_public_interface_names_present():
    for name in ("QUANTILE_Q_DEFAULT", "VOXEL_SIZE_DEFAULT", "N_MIN_DEFAULT", "TAU_CONTAM",
                 "CONF_WARP_VERSION", "DistancePrimitiveParams", "DistanceResult",
                 "robust_distance_to_reference", "robust_distance_between_pointsets",
                 "object_extent", "room_size_from_planes", "planarity_and_ground",
                 "ablation_quantiles"):
        assert hasattr(dp, name), f"缺少要求的公开接口：{name}"
    assert isinstance(dp.N_MIN_DEFAULT, int) and dp.N_MIN_DEFAULT > 0
    assert isinstance(dp.TAU_CONTAM, float) and 0.0 < dp.TAU_CONTAM < 1.0
