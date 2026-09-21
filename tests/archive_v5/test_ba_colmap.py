"""G-13/G-15 单测：VGGT BA 残差回填与 COLMAP → ReconstructionArtifact。

pycolmap 未安装，故用**最小 fake pycolmap**（实现本模块实际用到的 API：
`Reconstruction` / `Image.cam_from_world()` / `Camera.calibration_matrix()` /
`Point2D.xy` / `Point2D.has_point3D()` / `Point3D.xyz` / `Camera.img_from_cam`）
驱动真实代码路径，验证几何与降级行为。
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from skill3d.reconstruction import ba as ba_mod
from skill3d.reconstruction.ba import (
    ColmapUnavailable,
    artifact_data_from_sparse,
    find_sparse_model,
    reproj_errors_from_model,
    save_artifact_arrays,
)
from skill3d.reconstruction.colmap_baseline import (
    colmap_available,
    reconstruct_colmap,
    write_frames,
)
from skill3d.reconstruction.legacy_vggsfm_ba.route import run_ba_route
from skill3d.reconstruction.vggt_runner import ReconstructionFailed

# ------------------------------------------------------------------ fake pycolmap ----

K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
W, H = 640, 480


class _Rigid:
    def __init__(self, m: np.ndarray):
        self._m = np.asarray(m, dtype=np.float64)

    def matrix(self) -> np.ndarray:
        # 与真实 pycolmap 一致：返回 4x4 齐次矩阵
        return self._m if self._m.shape == (4, 4) else np.vstack([self._m, [0, 0, 0, 1]])

    def __mul__(self, xyz):        # cam_from_world * X
        xyz = np.asarray(xyz, dtype=np.float64)
        m = self.matrix()
        return m[:3, :3] @ xyz + m[:3, 3]


class _Cam:
    width, height, model_name = W, H, "PINHOLE"

    def calibration_matrix(self) -> np.ndarray:
        return K

    def img_from_cam(self, xy):
        p = K @ np.array([xy[0], xy[1], 1.0])
        return p[:2] / p[2]


class _P2D:
    def __init__(self, xy, pid=None):
        self.xy = np.asarray(xy, dtype=np.float64)
        self.point3D_id = pid

    def has_point3D(self) -> bool:
        return self.point3D_id is not None


class _P3D:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, dtype=np.float64)


class _Image:
    def __init__(self, name, w2c, points):
        self.name = name
        self.camera_id = 1
        self._w2c = _Rigid(w2c)
        self.points2D = points

    @property
    def num_points3D(self) -> int:
        return sum(1 for p in self.points2D if p.has_point3D())

    def cam_from_world(self) -> _Rigid:
        return self._w2c


class _Reconstruction:
    def __init__(self, images):
        self.images = {i: im for i, im in enumerate(images)}
        self.cameras = {1: _Cam()}
        self.points3D = {}


def _rot_x(deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def _make_scene(noise_px: float = 0.0, seed: int = 0, n_points: int = 40):
    """构造 3 帧模型：世界点投影到各帧为观测（可注入像素噪声 → 已知残差）。"""
    rng = np.random.default_rng(seed)
    world_points = rng.uniform(-1.0, 1.0, size=(n_points, 3))
    world_points[:, 2] += 4.0                       # 置于相机前方
    images, p3d = [], {}
    for t in range(3):
        r = _rot_x(5.0 * t)
        c = np.array([0.2 * t, 0.0, 0.0])
        w2c = np.eye(4)
        w2c[:3, :3] = r.T
        w2c[:3, 3] = -r.T @ c
        pts = []
        for j, X in enumerate(world_points):
            p_cam = r.T @ X + w2c[:3, 3]
            uv = K @ (p_cam / p_cam[2])
            obs = uv[:2] + (rng.normal(0, noise_px, 2) if noise_px else 0.0)
            pts.append(_P2D(obs, pid=j))
        images.append(_Image(f"{t:06d}.png", w2c, pts))
        for j, X in enumerate(world_points):
            p3d[j] = _P3D(X)
    model = _Reconstruction(images)
    model.points3D = p3d
    return model


def _install_fake_pycolmap(monkeypatch, model=None, root=None):
    mod = types.ModuleType("pycolmap")

    def _read(path):
        if model is None:
            raise RuntimeError("no model")
        return model

    mod.Reconstruction = _read
    mod.read_array = lambda p: np.zeros((2, 2))
    monkeypatch.setitem(sys.modules, "pycolmap", mod)
    return mod


# ------------------------------------------------------------------ G-13 残差 ----

def test_reproj_errors_are_zero_for_consistent_projection(monkeypatch):
    model = _make_scene(noise_px=0.0)
    _install_fake_pycolmap(monkeypatch, model)
    errs = reproj_errors_from_model(model)
    assert errs.size == 3 * 40
    assert float(np.median(errs)) < 1e-6        # 无噪声 → 残差为 0


def test_reproj_errors_detect_injected_noise(monkeypatch):
    """注入 2px 高斯噪声 → 残差 median 应 ~2px（G5 数据源可信）。"""
    model = _make_scene(noise_px=2.0, seed=3)
    _install_fake_pycolmap(monkeypatch, model)
    errs = reproj_errors_from_model(model)
    assert 1.0 < float(np.median(errs)) < 3.5, float(np.median(errs))


def test_compute_reproj_errors_returns_none_without_pycolmap(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pycolmap", None)
    assert ba_mod.compute_reproj_errors_from_colmap(tmp_path) is None


def test_read_sparse_model_none_when_model_dir_missing(monkeypatch, tmp_path):
    _install_fake_pycolmap(monkeypatch, _make_scene())
    assert ba_mod.read_sparse_model(tmp_path) is None    # 无 cameras.bin


def test_find_sparse_model_variants(tmp_path):
    for rel in ("sparse/0", "sparse"):
        d = tmp_path / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "cameras.bin").write_bytes(b"x")
        assert find_sparse_model(tmp_path) is not None
        (d / "cameras.bin").unlink()
    assert find_sparse_model(tmp_path) is None


def test_pycolmap_module_raises_when_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycolmap", None)
    with pytest.raises(ColmapUnavailable):
        ba_mod.pycolmap_module()


# ------------------------------------------------------------------ G-15 转换 ----

def test_artifact_data_from_sparse_geometry(monkeypatch):
    model = _make_scene(noise_px=0.0, seed=1)
    data = artifact_data_from_sparse(model)
    assert data.c2w_list.shape == (3, 4, 4)
    assert data.intrinsics.shape == (3, 3, 3)
    assert data.depth_maps.shape == (3, H, W)
    assert data.point_map.shape == (3, H, W, 3)
    assert data.image_names == ["000000.png", "000001.png", "000002.png"]
    # c2w 为 SE(3)
    for m in data.c2w_list:
        assert np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(m[:3, :3]), 1.0)
        assert np.allclose(m[3], [0, 0, 0, 1])
    # 稀疏深度：有观测处为正、其余为 0
    assert (data.depth_maps > 0).sum() == 3 * 40
    assert float(data.depth_maps.max()) > 0
    # 点云与深度一致（反投影回来还原世界点）
    total = (data.point_conf > 0).sum()
    assert total == 3 * 40


def test_artifact_data_roundtrip_world_points(monkeypatch):
    """c2w × K 反投影应还原世界坐标（G-15 几何正确性核心）。

    稀疏深度图按整像素 z-buffer 落点，故世界坐标误差的上界是"半像素投影误差"
    `z/f × 0.5`；像素域检查则是半像素内的严格断言（量化不变）。
    """
    model = _make_scene(noise_px=0.0, seed=5)
    data = artifact_data_from_sparse(model)
    mask = data.depth_maps[0] > 0
    vv, uu = np.nonzero(mask)
    assert vv.size == 40

    # ① 像素域：反投影点重投影回原帧，应落在原观测的半像素内
    pts = data.point_map[0][vv, uu]
    c2w = data.c2w_list[0]
    w2c = np.linalg.inv(c2w)
    cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
    uv = (K @ (cam / cam[:, 2:3]).T).T[:, :2]
    assert float(np.abs(uv - np.stack([uu, vv], axis=1)).max()) <= 0.5 + 1e-9

    # ② 世界域：误差不超过半像素在深度 z 处的等效世界尺度
    world = np.array([model.points3D[j].xyz for j in range(40)])
    d = np.linalg.norm(pts[:, None, :] - world[None, :, :], axis=-1).min(axis=1)
    z_max = float(data.depth_maps[0].max())
    bound = 0.5 * z_max / K[0, 0] * np.sqrt(2)
    assert float(d.max()) <= bound + 1e-9, (float(d.max()), bound)


def test_save_artifact_arrays_refs_exist(tmp_path, monkeypatch):
    data = artifact_data_from_sparse(_make_scene())
    refs = save_artifact_arrays(data, tmp_path, "sceneA")
    assert set(refs) == {"c2w_list", "intrinsics", "depth_maps", "point_map",
                         "point_conf", "reproj_errors"}
    for r in refs.values():
        assert np.load(r) is not None


def test_write_frames_writes_numbered_pngs(tmp_path):
    import cv2

    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    names = write_frames(frames, tmp_path / "images")
    assert names == ["000000.png", "000001.png", "000002.png"]
    for n in names:
        img = cv2.imread(str(tmp_path / "images" / n))
        assert img is not None and img.shape == (8, 8, 3)


# ------------------------------------------------------------------ 降级路径 ----

def test_colmap_available_false_for_bogus_binary():
    assert colmap_available("definitely-not-a-real-colmap-binary") is False


def test_reconstruct_colmap_raises_clear_error_without_cli(tmp_path):
    with pytest.raises(ReconstructionFailed, match="colmap CLI 不可用"):
        reconstruct_colmap([np.zeros((8, 8, 3), dtype=np.uint8)], "s", tmp_path,
                           colmap_bin="definitely-not-a-real-colmap-binary")


def test_reconstruct_colmap_raises_without_pycolmap(monkeypatch, tmp_path):
    monkeypatch.setattr("skill3d.reconstruction.colmap_baseline.colmap_available",
                        lambda *_: True)
    monkeypatch.setitem(sys.modules, "pycolmap", None)
    with pytest.raises(ReconstructionFailed, match="pycolmap 不可用"):
        reconstruct_colmap([np.zeros((8, 8, 3), dtype=np.uint8)], "s", tmp_path)


# ------------------------------------------------------------------ G-13 BA note ----

def test_run_ba_reports_degradation_instead_of_silent_pass(tmp_path):
    """G-13/§10.1：BA 未生效必须可见（原因进 summary），且不得标成 "BA 结果"。"""
    preds = {"point_map": np.zeros((1, 8, 8, 3)), "point_conf": np.full((1, 8, 8), 0.9),
             "depth_conf": np.full((1, 8, 8), 0.8)}
    res = run_ba_route(preds, np.zeros((1, 3, 8, 8)), tmp_path, "s", enabled=True)
    assert res.applied is False
    assert res.recon_method == "vggt"          # 未被标成 vggt_ba
    assert res.g5_reproj_err_median is None    # G5 不得伪造
    assert res.skip_reason                     # 回退原因必须可见
    assert "未生效" in res.summary()
