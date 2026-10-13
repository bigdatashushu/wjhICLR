"""简化版整场门：统计定义、四项否决、旧版读取与独立质量收据。"""

import json

import numpy as np
import pytest

from skill3d.legacy.readers import load_artifact_v5
from skill3d.reconstruction_gate import m4_main_gate as mg
from skill3d.reconstruction_gate import quality_metrics as qm
from skill3d.reconstruction_gate.scene_state import build_scene_state, quality_gate
from skill3d.schemas import ConfidenceMap, QualityMetrics, ReconstructionArtifact
from skill3d.schemas.reconstruction import LEGACY_QUALITY_METRIC_VERSION, QUALITY_METRIC_VERSION


def _geometry():
    """解析平面与不同机位，固定网格有足够不重复的有效点。"""
    n, h, w = 8, 64, 80
    depth = np.full((n, h, w), 2.0)
    k = np.array([[80.0, 0, 40], [0, 80.0, 32], [0, 0, 1]])
    c2w = np.repeat(np.eye(4)[None], n, axis=0)
    c2w[:, 0, 3] = np.arange(n) * 0.01
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cam = np.stack(((xx - 40) / 40, (yy - 32) / 40, np.full((h, w), 2)), axis=-1)
    points = np.repeat(cam[None], n, axis=0)
    points[..., 0] += c2w[:, 0, 3, None, None]
    frames = [np.full((h, w, 3), 120, dtype=np.uint8) for _ in range(n)]
    return dict(depth_maps=depth, c2w_list=c2w, intrinsics=k, point_map=points, frames=frames)


def _old_artifact(tmp_path, geo):
    refs = {}
    for key in ("depth_maps", "c2w_list", "intrinsics", "point_map"):
        path = tmp_path / f"{key}.npy"
        np.save(path, geo[key])
        refs[key] = str(path)
    q = QualityMetrics(
        warp_inlier_ratio=0.9, cloud_overlap_ratio=0.9,
        warp_photometric_inlier_ratio=0.9, main_gate_passed=True,
        g1_blur_ok=1, g2_brightness=1, g3_motion_blur=0, g4_frame_count=8,
        g6_depth_var_coeff=0, g7_dynamic_ratio=0, g9_tracker_consistency=1,
        g10_baseline_quality=0.5, overall_quality=0.9,
    )
    return ReconstructionArtifact(
        artifact_id="scene", artifact_version="frozen-arrays", scene_name="scene",
        quality_metric_version=LEGACY_QUALITY_METRIC_VERSION,
        quality_status="computed", quality=q, **refs, point_conf="",
        confidence=ConfidenceMap(per_point_confidence=""),
    )


def test_head_difference_uses_same_pixels_and_world_coordinates():
    geo = _geometry()
    # 世界坐标整体刚体变换必须同时施加到两头。
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    geo["c2w_list"][:, :3, :3] = rotation
    geo["c2w_list"][:, :3, 3] = geo["c2w_list"][:, :3, 3] @ rotation.T + [3, 4, 5]
    geo["point_map"] = geo["point_map"] @ rotation.T + [3, 4, 5]
    geo["point_map"][..., 2] += 0.4  # 欧氏差 0.4 / 源深度 2 = 0.2
    out = mg.scene_head_disagreement(
        geo["depth_maps"], geo["c2w_list"], geo["intrinsics"], geo["point_map"])
    assert out["point_depth_relative_p95"] == pytest.approx(0.2)
    assert len(out["depth_points"]) == 8 * 4 * 5
    assert out["reason"] == ""


def test_scene_statistics_are_scale_invariant():
    geo = _geometry()
    geo["point_map"] = geo["point_map"] + [0, 0, 0.1]
    results = []
    for factor in (1e-3, 1.0, 1e3):
        c2w = geo["c2w_list"].copy()
        c2w[:, :3, 3] *= factor
        head = mg.scene_head_disagreement(
            geo["depth_maps"] * factor, c2w, geo["intrinsics"], geo["point_map"] * factor)
        tail = mg.cloud_neighbor_tail_ratio(head["depth_points"])
        results.append((head["point_depth_relative_p95"], tail["cloud_neighbor_tail_ratio"]))
    assert np.allclose(results, results[1], rtol=1e-9, atol=1e-9)


def test_neighbor_tail_matches_direct_pairwise_definition():
    rng = np.random.default_rng(137)
    cloud = rng.normal(size=(110, 3))
    distance = np.linalg.norm(cloud[:, None] - cloud[None, :], axis=-1)
    means = np.sort(distance, axis=1)[:, 1:21].mean(axis=1)
    expected = np.quantile(means, 0.95) / np.quantile(means, 0.50)
    assert mg.cloud_neighbor_tail_ratio(cloud)["cloud_neighbor_tail_ratio"] == pytest.approx(expected)


def test_insufficient_nonfinite_and_zero_distance_samples_are_unknown():
    for cloud in (np.zeros((99, 3)), np.full((100, 3), np.nan), np.zeros((100, 3))):
        out = mg.cloud_neighbor_tail_ratio(cloud)
        assert np.isnan(out["cloud_neighbor_tail_ratio"])
        assert out["reason"]
    geo = _geometry()
    geo["point_map"][:] = np.nan
    out = mg.scene_head_disagreement(
        geo["depth_maps"], geo["c2w_list"], geo["intrinsics"], geo["point_map"])
    assert np.isnan(out["point_depth_relative_p95"])
    assert out["reason"] == "insufficient_common_samples"
    with pytest.raises(ValueError, match="网格不一致"):
        mg.scene_head_disagreement(
            geo["depth_maps"], geo["c2w_list"], geo["intrinsics"], geo["point_map"][:, :32])


@pytest.mark.parametrize("key,bad", [
    ("warp_inlier_ratio", 0.49), ("cloud_overlap_ratio", 0.29),
    ("point_depth_relative_p95", 0.151), ("cloud_neighbor_tail_ratio", 3.01),
])
def test_each_of_four_metrics_vetoes_the_whole_scene(key, bad):
    good = dict(warp_inlier_ratio=0.5, cloud_overlap_ratio=0.3,
                point_depth_relative_p95=0.15, cloud_neighbor_tail_ratio=3)
    assert mg.main_gate(good)["main_gate_passed"]
    out = mg.main_gate({**good, key: bad})
    assert not out["main_gate_passed"]
    assert not out["sub_results"][key]
    assert sum(out["sub_results"].values()) == 3
    for missing in (None, float("nan"), float("inf"), -1):
        out = mg.main_gate({**good, key: missing})
        assert not out["main_gate_passed"] and out["warnings"]


def test_old_two_metrics_cannot_pass_without_new_statistics():
    assert not mg.main_gate({"warp_inlier_ratio": 1, "cloud_overlap_ratio": 1})["main_gate_passed"]


def test_new_array_path_does_not_mutate_and_repeats_identically():
    geo = _geometry()
    before = {k: v.tobytes() for k, v in geo.items() if isinstance(v, np.ndarray)}
    a = mg.main_gate(geo)
    b = mg.main_gate(geo)
    assert a["main_gate_passed"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert before == {k: v.tobytes() for k, v in geo.items() if isinstance(v, np.ndarray)}


def test_old_quality_is_readable_but_recomputed_in_separate_receipt(tmp_path, monkeypatch):
    geo = _geometry()
    old = _old_artifact(tmp_path, geo)
    path = tmp_path / "frozen.json"
    payload = old.model_dump()
    payload["quality"].pop("point_depth_relative_p95")
    payload["quality"].pop("cloud_neighbor_tail_ratio")
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    loaded = load_artifact_v5(path)
    assert loaded.quality_metric_version == LEGACY_QUALITY_METRIC_VERSION
    assert not qm.quality_is_computed(loaded)
    assert build_scene_state(loaded).scene_route == "fallback_2d_only"
    updated = qm.compute_and_store_quality(loaded, frames=geo["frames"], artifact_path=path)
    assert updated.quality_metric_version == QUALITY_METRIC_VERSION
    assert updated.quality and updated.quality.main_gate_passed
    assert build_scene_state(updated).scene_route == "full_3d"
    assert path.read_bytes() == before
    receipt = qm.current_quality_path(path)
    assert receipt.is_file()
    assert load_artifact_v5(receipt).quality_metric_version == QUALITY_METRIC_VERSION
    # 同场景第二题必须复用收据；若再次调用计算则测试失败。
    def unexpected_compute(*args, **kwargs):
        pytest.fail("same frozen scene quality recomputed")
    monkeypatch.setattr(qm, "compute_quality", unexpected_compute)
    again = qm.compute_and_store_quality(loaded, frames=geo["frames"], artifact_path=path)
    assert again.quality_metric_version == QUALITY_METRIC_VERSION
    assert path.read_bytes() == before


def test_old_quality_can_be_recomputed_without_writing_frozen_record(tmp_path):
    geo = _geometry()
    old = _old_artifact(tmp_path, geo)
    path = tmp_path / "old.json"
    path.write_text(old.model_dump_json())
    before = path.read_bytes()
    scene = quality_gate(old, artifact_path=str(path), persist_quality=False)
    assert scene.quality and np.isfinite(scene.quality.point_depth_relative_p95)
    assert path.read_bytes() == before
    assert not qm.current_quality_path(path).exists()
