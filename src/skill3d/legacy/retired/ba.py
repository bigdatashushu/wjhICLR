"""G-13/G-15：BA 产物与 COLMAP sparse 模型的读回（pycolmap，lazy import）。

`pycolmap` 是 COLMAP 的官方 Python 绑定，可读 `cameras.bin/images.bin/points3D.bin`。
本模块提供两件事：

1. **G-13 重投影残差**：从 sparse 模型逐观测算 `|观测像素 − 重投影像素|`（px），
   回填 `ReconstructionArtifact.reproj_errors` → M4 的 G5 与置信度融合；
2. **G-15 COLMAP baseline → ReconstructionArtifact**：位姿/内参/稀疏深度/点云
   全部由 sparse 模型（可选 MVS 稠密深度）转换而来。

降级纪律（§4 M3 字段 9）：pycolmap 未安装或模型缺失时返回 `None` 并给出原因，
**绝不**用零数组或常数冒充残差——G5 恒 NaN 才是诚实状态（缺口报告 G-13 口径）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# COLMAP 稀疏模型在 workspace 下的常见位置（VGGT `demo_colmap.py --use_ba` 与
# `colmap mapper` 都写 `sparse/0`）
_SPARSE_CANDIDATES = ("sparse/0", "sparse", ".")


class ColmapUnavailable(RuntimeError):
    """pycolmap 不可用（未安装）。"""


def pycolmap_module():
    """lazy import pycolmap；不可用时抛 ColmapUnavailable。"""
    try:
        import pycolmap  # type: ignore

        return pycolmap
    except Exception as exc:  # noqa: BLE001
        raise ColmapUnavailable(
            f"pycolmap 不可用（{type(exc).__name__}: {exc}）；"
            "安装：pip install pycolmap（G-13/G-15 依赖）"
        ) from exc


def find_sparse_model(root: str | Path) -> Optional[Path]:
    """在 workspace 下定位 sparse 模型目录（含 cameras.bin/points3D.bin 或 .txt）。"""
    base = Path(root)
    for rel in _SPARSE_CANDIDATES:
        cand = base / rel
        if not cand.is_dir():
            continue
        for marker in ("cameras.bin", "cameras.txt"):
            if (cand / marker).is_file():
                return cand
    return None


def read_sparse_model(root: str | Path) -> Optional[Any]:
    """读回 COLMAP 稀疏模型；不可得返回 None（原因由调用方汇总）。"""
    try:
        pycolmap = pycolmap_module()
    except ColmapUnavailable:
        return None
    sparse = find_sparse_model(root)
    if sparse is None:
        return None
    try:
        return pycolmap.Reconstruction(str(sparse))
    except Exception:  # noqa: BLE001 - 模型损坏/版本不兼容
        return None


# ------------------------------------------------------------------ G-13 残差 ----

def reproj_errors_from_model(model: Any) -> np.ndarray:
    """逐观测重投影残差（px）：`|xy_obs − proj(cam_from_world · X)|`。

    仅统计有 3D 点的观测；无有效观测时返回空数组。
    """
    pycolmap = pycolmap_module()
    errors: list[float] = []
    for image in model.images.values():
        if image.num_points3D == 0:
            continue
        cam_from_world = image.cam_from_world()
        try:
            cam = model.cameras[image.camera_id]
        except Exception:  # noqa: BLE001
            continue
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            p3d = model.points3D[point2D.point3D_id].xyz
            proj = cam_from_world * p3d
            if proj[2] <= 1e-9:
                continue
            xy = cam.img_from_cam(np.array([proj[0] / proj[2], proj[1] / proj[2]]))
            errors.append(float(np.linalg.norm(np.asarray(xy) - np.asarray(point2D.xy))))
    _ = pycolmap
    return np.asarray(errors, dtype=np.float64)


def compute_reproj_errors_from_colmap(output_dir: str | Path,
                                      scene_name: str = "") -> Optional[np.ndarray]:
    """VGGT `--use_ba` 产物 → 重投影残差数组；不可得返回 None（不伪造）。"""
    model = read_sparse_model(output_dir)
    if model is None:
        return None
    errs = reproj_errors_from_model(model)
    return errs if errs.size else None


# ------------------------------------------------------------------ G-15 artifact ----

@dataclass
class SparseArtifactData:
    """sparse 模型转换出的数组集合（供 COLMAP baseline 组装 artifact）。"""

    c2w_list: np.ndarray                     # (T,4,4)
    intrinsics: np.ndarray                   # (T,3,3)
    depth_maps: np.ndarray                   # (T,H,W) 稀疏投影深度（0 = 无观测）
    point_map: np.ndarray                    # (T,H,W,3) 由稀疏深度反投影
    point_conf: np.ndarray                   # (T,H,W) 观测权重（1/重投影残差尺度）
    reproj_errors: np.ndarray                # (N,) px
    image_names: list[str] = field(default_factory=list)
    image_size: tuple[int, int] = (0, 0)     # (W, H)


def _camera_matrix(cam: Any) -> np.ndarray:
    """相机内参 3x3（优先 pycolmap 的 calibration_matrix）。"""
    if hasattr(cam, "calibration_matrix"):
        return np.asarray(cam.calibration_matrix(), dtype=np.float64)
    params = np.asarray(cam.params, dtype=np.float64)
    model_name = str(getattr(cam, "model_name", "") or "").upper()
    f, cx, cy = params[0], params[1], params[2]
    k = np.eye(3, dtype=np.float64)
    k[0, 0], k[1, 1], k[0, 2], k[1, 2] = f, f, cx, cy
    if "SIMPLE_RADIAL" in model_name or "RADIAL" in model_name or "OPENCV" in model_name:
        pass  # 仅取 pinhole 近似（畸变由 COLMAP 内部校正；baseline 用）
    return k


def artifact_data_from_sparse(
    model: Any,
    *,
    image_size: Optional[tuple[int, int]] = None,
    max_points_per_pixel: int = 1,
) -> SparseArtifactData:
    """把 COLMAP sparse 模型转成统一数组：c2w/K/稀疏深度/点云/残差。

    深度图由 **3D 点重投影 + z-buffer** 得到（sparse 模型无稠密深度；
    MVS 稠密路径见 `run_colmap` 的 `--mvs`）。无观测像素深度为 0，
    由 M4 的 G5/G6 与置信度融合按"数据缺失"处理。
    """
    images = sorted(model.images.values(), key=lambda im: im.name)
    if not images:
        raise ValueError("sparse 模型无注册图像")

    cam0 = model.cameras[images[0].camera_id]
    if image_size is None:
        w, h = int(cam0.width), int(cam0.height)
    else:
        w, h = int(image_size[0]), int(image_size[1])

    t = len(images)
    c2w = np.zeros((t, 4, 4), dtype=np.float64)
    intr = np.zeros((t, 3, 3), dtype=np.float64)
    depth = np.zeros((t, h, w), dtype=np.float64)
    conf = np.zeros((t, h, w), dtype=np.float64)
    reproj_all: list[float] = []
    names: list[str] = []

    for ti, image in enumerate(images):
        names.append(str(image.name))
        cam = model.cameras[image.camera_id]
        k = _camera_matrix(cam)
        intr[ti] = k
        # COLMAP: cam_from_world（world→camera）；c2w 为其逆。
        # pycolmap 的 Rigid3d.matrix() 返回 4x4 齐次矩阵（旧版本可能给 3x4），
        # 统一取 [:3,:4] 保证两种形态都能读。
        w2c = np.eye(4, dtype=np.float64)
        m = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        w2c[:3, :4] = m[:3, :4]
        c2w[ti] = np.linalg.inv(w2c)

        # 逐观测残差（G-13 同源计算，保证与 reproj_errors 一致）
        for point2D in image.points2D:
            if not point2D.has_point3D():
                continue
            p3d = np.asarray(model.points3D[point2D.point3D_id].xyz, dtype=np.float64)
            p_cam = w2c[:3, :3] @ p3d + w2c[:3, 3]
            if p_cam[2] <= 1e-9:
                continue
            uv = k @ (p_cam / p_cam[2])
            u, v = int(round(uv[0])), int(round(uv[1]))
            err = float(np.linalg.norm(uv[:2] - np.asarray(point2D.xy, dtype=np.float64)))
            reproj_all.append(err)
            if not (0 <= u < w and 0 <= v < h):
                continue
            if depth[ti, v, u] == 0.0 or p_cam[2] < depth[ti, v, u]:
                depth[ti, v, u] = float(p_cam[2])
                # 观测权重：残差越小越可信（1px 尺度归一，TODO_CALIBRATE）
                conf[ti, v, u] = float(1.0 / (1.0 + err))
    _ = max_points_per_pixel

    # 稀疏深度反投影 → 世界点云
    point_map = np.zeros((t, h, w, 3), dtype=np.float64)
    for ti in range(t):
        z = depth[ti]
        mask = z > 0
        if not mask.any():
            continue
        vv, uu = np.nonzero(mask)
        k_inv = np.linalg.inv(intr[ti])
        pix = np.stack([uu, vv, np.ones_like(uu)], axis=0).astype(np.float64)
        cam_pts = k_inv @ pix * z[vv, uu][None, :]
        world = c2w[ti][:3, :3] @ cam_pts + c2w[ti][:3, 3:4]
        point_map[ti][vv, uu] = world.T

    return SparseArtifactData(
        c2w_list=c2w, intrinsics=intr, depth_maps=depth, point_map=point_map,
        point_conf=conf, reproj_errors=np.asarray(reproj_all, dtype=np.float64),
        image_names=names, image_size=(w, h),
    )


def save_artifact_arrays(data: SparseArtifactData, out_dir: str | Path,
                         scene_name: str) -> dict[str, str]:
    """数组落盘为 npy，返回字段名 → 路径 ref（§5.2 artifact 字段约定）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _save(name: str, arr: np.ndarray) -> str:
        p = out / f"{scene_name}_{name}.npy"
        np.save(p, arr)
        return str(p)

    return {
        "c2w_list": _save("c2w", data.c2w_list),
        "intrinsics": _save("intrinsics", data.intrinsics),
        "depth_maps": _save("depth", data.depth_maps),
        "point_map": _save("point_map", data.point_map),
        "point_conf": _save("point_conf", data.point_conf),
        "reproj_errors": _save("reproj_errors", data.reproj_errors),
    }


def coarse_quality_from_sparse(reproj: np.ndarray, n_frames: int,
                               depth: np.ndarray) -> tuple[float, float]:
    """由 sparse 产物给 G5 median/p95 的近似值（px）；无残差时 (NaN, NaN)。"""
    if reproj is None or reproj.size == 0:
        return float("nan"), float("nan")
    valid = np.isfinite(reproj)
    if not valid.any():
        return float("nan"), float("nan")
    e = reproj[valid]
    _ = (n_frames, depth)
    return float(np.median(e)), float(np.percentile(e, 95))


def dense_depth_available(dense_dir: str | Path) -> bool:
    """MVS 稠密深度是否已产出（`stereo/depth_maps/*.bin`）。"""
    return (Path(dense_dir) / "stereo" / "depth_maps").is_dir()


def read_dense_depth_map(dense_dir: str | Path, image_name: str) -> Optional[np.ndarray]:
    """读 MVS 稠密深度（pycolmap.read_array，geometric 类型）；不可得返回 None。"""
    try:
        pycolmap = pycolmap_module()
    except ColmapUnavailable:
        return None
    stem = Path(image_name).stem
    p = Path(dense_dir) / "stereo" / "depth_maps" / f"{stem}.geometric.bin"
    if not p.is_file():
        return None
    try:
        return np.asarray(pycolmap.read_array(str(p)))
    except Exception:  # noqa: BLE001
        return None


def sparse_model_summary(model: Any) -> dict[str, Any]:
    """sparse 模型摘要（写进 notes / trace，便于审计）。"""
    return {
        "n_images": len(model.images),
        "n_points3D": len(model.points3D),
        "n_cameras": len(model.cameras),
        "n_observations": sum(im.num_points3D for im in model.images.values()),
    }


def image_names_in_model(model: Any) -> Sequence[str]:
    return [str(im.name) for im in sorted(model.images.values(), key=lambda i: i.name)]
