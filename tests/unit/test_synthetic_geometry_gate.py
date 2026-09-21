"""合成几何 × M4 主门（v6 §10 / §6.2）—— mock_light 不许"绕过门"。

背景：v6 把 M4 主门从 v5 的标量 `overall_quality` 阈值换成
**跨视图 warp 内点率 ∧ 分组点云重叠率**（`reconstruction_gate/m4_main_gate.py`）。
mock_light 的合成 bundle 必须**真的**满足这套约定，否则 `scene_route` 会落到
`fallback_2d_only`、3D Tool 全被收回 → 每个 mock episode 都 unanswerable
（这正是本次修复前的实际状态）。

本文件锁三件事（**不放松任何阈值、不改门**）：

1. 干净合成 bundle 过**真门**：`main_gate_passed=True` 且
   `overall_quality > TH_OVERALL_QUALITY` → `scene_route == "full_3d"`；
2. 约定逐字对齐：`point_map` 必须等于"M4 门反投影公式"的正算结果
   （`p_cam = ((u−cx)/fx·z, (v−cy)/fy·z, z)` → `p_world = R·p_cam + t`），
   且 `K` 必须是**深度网格**像素单位（拿全分辨率 K 去配深度网格必须把门打挂）；
3. 注入破坏后门**必须 fail**（洗牌半段位姿 / 破坏半段深度）—— 这条保证
   mock 路径是"真实烟囱测试"，不是靠合成数据伪造通过。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.online import synthetic as syn
from skill3d.reconstruction_gate import m4_main_gate as mg
from skill3d.reconstruction_gate.scene_state import (
    TH_OVERALL_QUALITY,
    scene_route_from_quality,
)

FRAME_SIZE = (120, 160)


@pytest.fixture(scope="module")
def bundle():
    """一条合成 episode（几何 + 帧）；模块级复用（构造 + 真算门 ≈ 秒级）。"""
    se = syn.make_synthetic_episode("room_size_estimation", scene_name="gate-scene",
                                    qa_id="gate-0", seed=0, frame_size=FRAME_SIZE)
    return se


def _gate_inputs(geometry, frames, *, c2w=None, depth=None, point_map=None,
                 intrinsics=None) -> dict:
    return {
        "frames": list(frames),
        "depth_maps": geometry.depth_maps if depth is None else depth,
        "c2w_list": geometry.c2w if c2w is None else c2w,
        "intrinsics": geometry.intrinsics if intrinsics is None else intrinsics,
        "point_map": geometry.point_map if point_map is None else point_map,
    }


def _backproject(depth: np.ndarray, c2w: np.ndarray, K: np.ndarray) -> np.ndarray:
    """按 M4 门的反投影口径独立正算世界点图（测试侧不 import 生产实现）。

    `x = (u−cx)/fx·z`、`y = (v−cy)/fy·z`（`u,v` 为**整数像素索引**）、
    `p_world = R·p_cam + t`；与 `m4_main_gate._pair_warp_stats` 逐字一致。
    """
    h, w = int(depth.shape[1]), int(depth.shape[2])
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    out = np.empty((depth.shape[0], h, w, 3), dtype=np.float64)
    for i in range(depth.shape[0]):
        z = np.asarray(depth[i], dtype=np.float64)
        p_cam = np.stack([(uu - K[0, 2]) / K[0, 0] * z,
                          (vv - K[1, 2]) / K[1, 1] * z, z], axis=-1)
        out[i] = p_cam @ c2w[i, :3, :3].T + c2w[i, :3, 3]
    return out


# ------------------------------------------------- 1. 干净 bundle 必须过真门 ----

def test_clean_bundle_passes_real_main_gate(bundle):
    """主门（warp ∧ 重叠）真算通过；两个子项都不得靠"单挑"放行（§10.1）。"""
    g = bundle.geometry
    out = mg.main_gate(_gate_inputs(g, bundle.frames))
    assert out["main_gate_passed"] is True, out["warnings"]
    assert out["sub_results"] == {"warp_inlier_ratio": True, "cloud_overlap_ratio": True}
    # 余量：不是"刚过线"的脆弱构造（合成 bundle 是精确自洽的）
    assert out["values"]["warp_inlier_ratio"] >= 0.9
    assert out["values"]["cloud_overlap_ratio"] >= mg.TH_CLOUD_OVERLAP + 0.1
    assert out["values"]["warp_inlier_ratio"] >= mg.TH_WARP_INLIER + 0.1
    assert out["n_pairs"] >= 32                     # 配对没有被"可比像素不足"整批跳过


def test_build_scene_state_routes_to_full_3d(bundle):
    """`scene_route == "full_3d"`（M4 质量决定）+ 米制题 scope 到 metric_enabled。"""
    scene, handle, q = syn.build_scene_state(bundle.geometry, bundle.frames)
    assert q.main_gate_passed is True
    assert float(q.overall_quality) > TH_OVERALL_QUALITY
    assert scene_route_from_quality(q) == "full_3d"
    assert scene.scene_route == "full_3d"
    gate = scene.metric_evidence_gate_result
    assert gate is not None and gate.gate_passed is True
    # 合成场景的 8 项能力都按真实判定给出（不是"无画像 → 全放行"）
    assert scene.evidence_profile is not None
    assert scene.evidence_profile.state("geometry_3d") == "available"
    assert scene.evidence_profile.state("metric_scale") == "available"
    # 3D 产物真的挂上句柄（Tool 拿得到点图/对象点云）
    assert {"depth", "poses", "point_cloud", "objects"} <= handle.available_artifacts
    assert handle.get_point_map() is not None
    assert handle.object_points("sofa-0").shape[0] > 100


def test_metric_gate_fails_when_question_type_is_not_metric(bundle):
    """单项失败只收窄 scope：非米制题不给 scale，但 3D 能力不受影响（§7.2/D4）。"""
    scene, _handle, _q = syn.build_scene_state(
        bundle.geometry, bundle.frames, metric_tasks=set())
    assert scene.scene_route == "full_3d"            # route 只由质量决定
    assert scene.question_tool_scope == "full_3d"
    assert scene.metric_evidence_gate_result.gate_passed is False
    assert "scale" not in scene.available_artifacts


# ----------------------------------------------------- 2. 约定逐字对齐 ----

def test_point_map_matches_gate_backprojection(bundle):
    """`point_map` == 门反投影公式的正算结果（约定不一致 → 门会正确地报 ~0 重叠）。"""
    g = bundle.geometry
    ref = _backproject(g.depth_maps, g.c2w, g.intrinsics)
    assert np.allclose(g.point_map, ref, atol=1e-4)
    # 精度足以支撑相对阈值 3% 的重叠判据（点间距远小于它）
    assert float(np.nanmax(np.abs(np.asarray(g.point_map) - ref))) < 1e-4


def test_depth_is_camera_frame_z_of_that_pose(bundle):
    """深度 = 该位姿下的**相机系 z**（不是斜距）：重投影回自身必须落回原像素。"""
    g = bundle.geometry
    K = g.intrinsics
    pm = g.point_map
    for i in (0, 15, 31):
        R, t = g.c2w[i, :3, :3], g.c2w[i, :3, 3]
        p_cam = (pm[i] - t) @ R
        z = p_cam[..., 2]
        u = K[0, 0] * p_cam[..., 0] / z + K[0, 2]
        v = K[1, 1] * p_cam[..., 1] / z + K[1, 2]
        uu, vv = np.meshgrid(np.arange(g.depth_maps.shape[2]),
                             np.arange(g.depth_maps.shape[1]))
        assert np.allclose(u, uu, atol=1e-3)         # 像素索引口径自洽
        assert np.allclose(v, vv, atol=1e-3)
        assert np.allclose(z, g.depth_maps[i], rtol=1e-4)


def test_full_resolution_intrinsics_must_fail_the_gate(bundle):
    """K 必须是**深度网格**像素单位：拿全分辨率 K（fx=500, cx=320）配深度网格 → 挂。

    注意这里判的是"**自洽性**"而不是"标定准确"：把 K 整体换成一个自洽但不同的相机
    模型（例如主点整体挪 5px）仍能过门（§10.5 诚实边界：M4 只判同一静态场景且几何
    自洽，不判绝对精度）。但 K 与 depth/point_map **不同网格**时必须立刻 fail-closed。
    """
    g = bundle.geometry
    K_full = np.array([[syn.FOCAL, 0.0, syn.FRAME_W / 2.0],
                       [0.0, syn.FOCAL, syn.FRAME_H / 2.0],
                       [0.0, 0.0, 1.0]])
    out = mg.main_gate(_gate_inputs(g, bundle.frames, intrinsics=K_full))
    assert out["main_gate_passed"] is False
    assert out["sub_results"]["warp_inlier_ratio"] is False


def test_depth_conf_is_finite_and_positive(bundle):
    """`depth_conf` 有限且 > 0（§10.3：只作软权重，但必须真的是合法置信度）。"""
    conf = bundle.geometry.depth_conf
    assert np.isfinite(conf).all()
    assert float(conf.min()) > 0.0
    assert float(conf.max()) > float(conf.min())      # 非常数：自检才有意义


# --------------------------------------------- 3. 注入破坏后门必须失败 ----

def test_shuffled_poses_fail_the_gate(bundle):
    """洗牌位姿序列（点图用被洗牌的位姿重建）→ 主门必须失败。

    （只洗牌后半段时，箱型房间的深度场本身缓慢变化、且在"两半都错"的配对里误差会
    部分自消 —— 实测 warp 仍达 0.75；故这里用**整段**确定性洗牌，warp 掉到 ~0.32。）
    """
    g = bundle.geometry
    perm = np.random.default_rng(0).permutation(g.c2w.shape[0])
    c2w_bad = g.c2w[perm].copy()
    pm_bad = _backproject(g.depth_maps, c2w_bad, g.intrinsics).astype(np.float32)
    out = mg.main_gate(_gate_inputs(g, bundle.frames, c2w=c2w_bad, point_map=pm_bad))
    assert out["main_gate_passed"] is False
    assert out["sub_results"]["warp_inlier_ratio"] is False
    assert out["values"]["warp_inlier_ratio"] < mg.TH_WARP_INLIER


def test_corrupted_depth_fails_the_warp_submetric(bundle):
    """逐帧独立扰动深度（并用被破坏的深度重建点图）→ warp 子项必须挂。"""
    g = bundle.geometry
    scale = np.random.default_rng(3).uniform(0.8, 1.25, size=(g.depth_maps.shape[0], 1, 1))
    depth_bad = (np.asarray(g.depth_maps, dtype=np.float64) * scale).astype(np.float32)
    pm_bad = _backproject(depth_bad, g.c2w, g.intrinsics).astype(np.float32)
    out = mg.main_gate(_gate_inputs(g, bundle.frames, depth=depth_bad, point_map=pm_bad))
    assert out["main_gate_passed"] is False
    assert out["sub_results"]["warp_inlier_ratio"] is False
    assert out["values"]["warp_inlier_ratio"] < mg.TH_WARP_INLIER


def test_corrupted_half_depth_breaks_cloud_overlap(bundle):
    """只破坏**后半段**深度 → 前/后两个子云不再对齐，重叠率子项必须挂。"""
    g = bundle.geometry
    depth_bad = np.array(g.depth_maps, dtype=np.float32, copy=True)
    depth_bad[16:] *= 1.5                        # 后半段深度被系统性拉远
    pm_bad = _backproject(depth_bad, g.c2w, g.intrinsics).astype(np.float32)
    out = mg.main_gate(_gate_inputs(g, bundle.frames, depth=depth_bad, point_map=pm_bad))
    assert out["main_gate_passed"] is False
    assert out["sub_results"]["cloud_overlap_ratio"] is False
    assert out["values"]["cloud_overlap_ratio"] < 0.1


def test_uncorrupted_rebuild_of_point_map_keeps_gate_passing(bundle):
    """对照：用**未破坏**的深度重建点图（走同一条测试侧公式）→ 门仍通过。

    这条把"上面几条的失败"钉死在"数据被破坏"上，而不是"测试侧公式与门不一致"。
    """
    g = bundle.geometry
    pm = _backproject(g.depth_maps, g.c2w, g.intrinsics).astype(np.float32)
    out = mg.main_gate(_gate_inputs(g, bundle.frames, point_map=pm))
    assert out["main_gate_passed"] is True
