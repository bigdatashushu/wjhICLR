"""PyCOLMAP 三角化 + 真 BA + 重投影残差（§10.1 L1；pycolmap 4.2 实测 API）。

**HC36 约束**：后端是 PyCOLMAP 三角化 + `bundle_adjustment`；前端必须是 LightGlue
稀疏匹配（`features.py` / `matching.py`），**不得**复用官方 VGGSfM tracker（HC35 否决）。

**初始化**：用 VGGT 的 `c2w` 与内参作为初值；世界系保持"首帧相机系"，BA 后重新固定
`c2w[0]=I`。**BA 不恢复米制尺度**（gauge 自由度）。

**pycolmap 4.2 实测要点**（本文件按实测 API 写，不用记忆中的旧签名）：

- `Image.cam_from_world` 是**只读方法**，位姿挂在 `Frame`：`frame.rig_from_world = Rigid3d(...)`
  且必须 `rec.register_frame(frame_id)`，否则 BA 报 "Need at least one registered frame"；
- 模块级 `triangulate_points(rec, db, images, out)` 现在要求 **COLMAP database**，
  因此本模块用**自实现 numpy DLT** 三角化（已知位姿 + 内参，等价且更可控）；
- `pycolmap.bundle_adjustment(rec, opts)` 返回 `None`（cost 只进日志），
  故 `initial_cost/final_cost` 由我们按"BA 前后的均方重投影误差"计算并显式注明口径；
- `Point2D(xy)` 收单个数组；观测用 `TrackElement(image_id, point2D_idx)` 挂到 `Point3D`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .features import FrameFeatures
from .matching import MatchReport, matches_to_track_input
from .tracks import Track, merge_tracks, validate_tracks

# 三角化/过滤门槛（全部 TODO_CALIBRATE）
MAX_TRIANGULATION_REPROJ_PX: float = 4.0   # TODO_CALIBRATE：三角化阶段单观测残差上限
MIN_PARALLAX_DEG: float = 1.0              # TODO_CALIBRATE：最小视差（度过小 → 深度不可靠）


class PycolmapUnavailable(RuntimeError):
    """pycolmap 不可用或 API 不符 → L1 止损（不得换别的后端硬凑）。"""


@dataclass
class BAInputs:
    """BA 输入（特征分辨率下的坐标与内参；世界系 = VGGT 首帧相机系）。"""

    tracks: list[Track]
    c2w_init: np.ndarray            # (S, 4, 4)
    intrinsics_init: np.ndarray     # (S, 3, 3)
    feature_hw: tuple[int, int]


@dataclass
class BAResult:
    """BA 结果。失败时 `ok=False` 且 `reason` 非空（调用方写 receipt 并止损）。"""

    ok: bool = False
    reason: str = ""
    c2w_refined: Optional[np.ndarray] = None
    intrinsics_refined: Optional[np.ndarray] = None
    points3D: Optional[np.ndarray] = None
    reproj_errors: Optional[np.ndarray] = None
    initial_cost: Optional[float] = None
    final_cost: Optional[float] = None
    n_tracks: int = 0
    n_observations: int = 0
    n_triangulated: int = 0
    notes: list[str] = field(default_factory=list)

    def _stats(self) -> tuple[Optional[float], Optional[float]]:
        if self.reproj_errors is None or len(self.reproj_errors) == 0:
            return None, None
        e = np.asarray(self.reproj_errors, dtype=np.float64)
        e = e[np.isfinite(e)]
        if e.size == 0:
            return None, None
        return float(np.median(e)), float(np.percentile(e, 95))

    @property
    def g5_median(self) -> Optional[float]:
        return self._stats()[0]

    @property
    def g5_p95(self) -> Optional[float]:
        return self._stats()[1]


def build_ba_inputs(feats: Sequence[FrameFeatures], report: MatchReport, *,
                    c2w: np.ndarray, intrinsics: np.ndarray,
                    min_track_length: int) -> BAInputs:
    """逐 pair 内点 → 多帧 track，并跑 L0 不变量校验。"""
    groups = matches_to_track_input(report)
    merged = merge_tracks(groups, min_length=int(min_track_length))
    ok, problems = validate_tracks(merged.tracks, min_length=int(min_track_length))
    if not ok:
        raise PycolmapUnavailable(
            "track 不变量未满足（L0 合同）：" + "; ".join(problems[:3]))
    if not merged.tracks:
        raise PycolmapUnavailable(
            f"无有效 track（matches={report.n_matches} inliers={report.n_inliers}；"
            f"skip={report.skip_reasons}）")
    return BAInputs(tracks=merged.tracks,
                    c2w_init=np.asarray(c2w, dtype=np.float64),
                    intrinsics_init=np.asarray(intrinsics, dtype=np.float64),
                    feature_hw=feats[0].feature_hw)


def _projection_matrix(k: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """P = K [R|t]（world→image），c2w 给的是 camera→world。"""
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    return np.asarray(k, dtype=np.float64) @ w2c[:3, :4]


def triangulate_dlt(proj_mats: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
    """多视图 DLT 三角化（`proj_mats` (N,3,4)，`pts` (N,2)）。退化/失败返回 None。"""
    if len(proj_mats) < 2:
        return None
    rows = []
    for p, (x, y) in zip(proj_mats, pts):
        rows.append(x * p[2] - p[0])
        rows.append(y * p[2] - p[1])
    a = np.asarray(rows, dtype=np.float64)
    try:
        _, _, vt = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None
    x_h = vt[-1]
    if abs(x_h[3]) < 1e-12:
        return None
    x = x_h[:3] / x_h[3]
    return x if np.all(np.isfinite(x)) else None


def _cheirality_ok(x: np.ndarray, proj_mats: np.ndarray, pts: np.ndarray,
                   max_err: float) -> bool:
    """所有观测都在相机前方，且重投影误差在阈值内（至少两视图）。"""
    n_ok = 0
    for p, (px, py) in zip(proj_mats, pts):
        cam = p @ np.append(x, 1.0)
        if cam[2] <= 1e-9:
            return False
        uv = cam[:2] / cam[2]
        if float(np.hypot(uv[0] - px, uv[1] - py)) > max_err:
            return False
        n_ok += 1
    return n_ok >= 2


def _max_parallax_deg(centers: np.ndarray, x: np.ndarray) -> float:
    """最大视差角（度）：过小 → 深度不可靠。"""
    rays = x[None, :] - centers
    norms = np.linalg.norm(rays, axis=1)
    norms[norms < 1e-12] = 1e-12
    rays = rays / norms[:, None]
    best = 0.0
    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            c = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
            best = max(best, float(np.degrees(np.arccos(c))))
    return best


def _mean_sq_reproj(errs: Sequence[float]) -> Optional[float]:
    e = np.asarray([x for x in errs if np.isfinite(x)], dtype=np.float64)
    return float(np.mean(np.square(e))) if e.size else None


def _triangulate_tracks(inputs: BAInputs) -> tuple[list[tuple[np.ndarray, list, float]], list[float]]:
    """逐 track DLT 三角化 + 几何筛选。

    返回 `([(xyz, [(frame_id, keypoint_id, xy), ...], mean_err)], [所有观测残差])`。
    筛掉的 track 不进入 BA（不参与、也不"补一个点"）。
    """
    s = len(inputs.c2w_init)
    proj = [_projection_matrix(inputs.intrinsics_init[i], inputs.c2w_init[i])
            for i in range(s)]
    centers = np.asarray([inputs.c2w_init[i][:3, 3] for i in range(s)])
    out: list[tuple[np.ndarray, list, float]] = []
    err_acc: list[float] = []
    for t in inputs.tracks:
        frames = [o.frame_id for o in t.observations]
        if len(set(frames)) != len(frames):
            continue
        pm = np.asarray([proj[f] for f in frames])
        pts = np.asarray([[o.xy[0], o.xy[1]] for o in t.observations], dtype=np.float64)
        x = triangulate_dlt(pm, pts)
        if x is None or not _cheirality_ok(x, pm, pts, MAX_TRIANGULATION_REPROJ_PX):
            continue
        if _max_parallax_deg(centers[frames], x) < MIN_PARALLAX_DEG:
            continue
        obs_errs = []
        obs = []
        for p, f, (px, py), o in zip(pm, frames, pts, t.observations):
            cam = p @ np.append(x, 1.0)
            uv = cam[:2] / cam[2]
            obs_errs.append(float(np.hypot(uv[0] - px, uv[1] - py)))
            obs.append((int(f), int(o.keypoint_id), (float(o.xy[0]), float(o.xy[1]))))
        out.append((x, obs, float(np.mean(obs_errs))))
        err_acc.extend(obs_errs)
    return out, err_acc


def _build_reconstruction(pycolmap, inputs: BAInputs, tri: list[tuple[np.ndarray, list, float]]):
    """把三角化结果组装成 pycolmap.Reconstruction（位姿挂 Frame 并注册）。"""
    s = len(inputs.c2w_init)
    h, w = int(inputs.feature_hw[0]), int(inputs.feature_hw[1])
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=w, height=h, camera_id=1,
                          params=[float(inputs.intrinsics_init[0, 0, 0]),
                                  float(inputs.intrinsics_init[0, 0, 2]),
                                  float(inputs.intrinsics_init[0, 1, 2])])
    rec.add_camera_with_trivial_rig(cam)
    for i in range(s):
        rec.add_image_with_trivial_frame(
            pycolmap.Image(name=f"frame_{i}", camera_id=cam.camera_id, image_id=i + 1))
    for i in range(s):
        st = rec.image(i + 1)
        fr = rec.frame(st.frame_id)
        w2c = np.linalg.inv(inputs.c2w_init[i])
        fr.rig_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(w2c[:3, :3]), w2c[:3, 3])
        rec.register_frame(fr.frame_id)   # 必需：否则 BA 报 "Need at least one registered frame"

    # 逐帧收集需要的 2D 观测（同一 (frame, keypoint) 只放一个 Point2D）
    per_frame: dict[int, list[tuple[int, tuple[float, float]]]] = {}
    index_of: dict[tuple[int, int], int] = {}
    for _, obs, _ in tri:
        for f, kp_id, xy in obs:
            key = (f, kp_id)
            if key in index_of:
                continue
            lst = per_frame.setdefault(f, [])
            index_of[key] = len(lst)
            lst.append((kp_id, xy))
    for f, lst in per_frame.items():
        rec.image(f + 1).points2D = [pycolmap.Point2D(np.asarray(xy, dtype=np.float64))
                                     for _, xy in lst]

    n_linked = 0
    for xyz, obs, mean_err in tri:
        pid = rec.add_point3D(np.asarray(xyz, dtype=np.float64), pycolmap.Track(),
                              np.array([0, 0, 0]))
        for f, kp_id, _xy in obs:
            te = pycolmap.TrackElement()
            te.image_id = int(f + 1)
            te.point2D_idx = int(index_of[(f, kp_id)])
            rec.add_observation(pid, te)
            n_linked += 1
    return rec, n_linked


def _reprojection_errors(pycolmap, rec, k: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """按（可能已精化的）位姿与 3D 点重算全部观测的重投影残差（px）。"""
    errs: list[float] = []
    for pid in rec.point3D_ids():
        p3d = rec.point3D(int(pid))
        xyz = np.asarray(p3d.xyz, dtype=np.float64)
        if not np.all(np.isfinite(xyz)):
            continue
        for el in p3d.track.elements:
            i = int(el.image_id) - 1
            if i < 0 or i >= len(c2w):
                continue
            w2c = np.linalg.inv(c2w[i])
            cam = w2c[:3, :3] @ xyz + w2c[:3, 3]
            if cam[2] <= 1e-9:
                continue
            uv = k[i] @ (cam / cam[2])
            pt2d = rec.image(int(el.image_id)).points2D[int(el.point2D_idx)]
            xy = np.asarray(pt2d.xy, dtype=np.float64).ravel()
            errs.append(float(np.hypot(uv[0] - xy[0], uv[1] - xy[1])))
    return np.asarray(errs, dtype=np.float64)


def triangulate_and_ba(inputs: BAInputs, *, min_track_length: int) -> BAResult:
    """DLT 三角化 → 组装 Reconstruction → 全局 BA → 读回位姿与 G5。"""
    try:
        import pycolmap
    except Exception as exc:  # noqa: BLE001
        raise PycolmapUnavailable(f"pycolmap 不可用: {exc}") from exc

    res = BAResult(n_tracks=len(inputs.tracks),
                   n_observations=sum(len(t.observations) for t in inputs.tracks))
    tri, pre_errs = _triangulate_tracks(inputs)
    res.n_triangulated = len(tri)
    res.notes.append(f"dlt_triangulated={len(tri)}/{len(inputs.tracks)} "
                     f"(parallax>={MIN_PARALLAX_DEG}°, reproj<={MAX_TRIANGULATION_REPROJ_PX}px)")
    if len(tri) < 3:
        res.reason = f"三角化成功点数不足（{len(tri)} < 3）"
        return res

    rec, n_linked = _build_reconstruction(pycolmap, inputs, tri)
    res.notes.append(f"observations_linked={n_linked}")

    # BA 前后各算一次均方重投影误差（pycolmap 4.2 不返回内部 cost，口径见模块 docstring）
    c2w = np.asarray(inputs.c2w_init, dtype=np.float64)
    pre = _reprojection_errors(pycolmap, rec, inputs.intrinsics_init, c2w)
    res.initial_cost = _mean_sq_reproj(pre)

    try:
        opts = pycolmap.BundleAdjustmentOptions()
        opts.refine_focal_length = False       # VGGT 内参可信：只精化位姿与 3D 点
        opts.refine_principal_point = False
        opts.refine_extra_params = False
        opts.print_summary = False
        pycolmap.bundle_adjustment(rec, opts)
    except Exception as exc:  # noqa: BLE001
        res.reason = f"bundle_adjustment 失败: {type(exc).__name__}: {exc}"
        return res

    # 读回精化位姿（camera→world），并重新固定首帧为世界原点（gauge 规范化）
    s = len(inputs.c2w_init)
    c2w_ref = np.tile(np.eye(4), (s, 1, 1))
    for i in range(s):
        w2c = np.asarray(rec.image(i + 1).cam_from_world().matrix(), dtype=np.float64)
        c2w_ref[i] = np.linalg.inv(w2c)
    if not np.all(np.isfinite(c2w_ref)):
        res.reason = "BA 后位姿含非有限值"
        return res
    fix = np.linalg.inv(c2w_ref[0])
    c2w_ref = np.einsum("ij,njk->nik", fix, c2w_ref)

    post = _reprojection_errors(pycolmap, rec, inputs.intrinsics_init, c2w_ref)
    res.final_cost = _mean_sq_reproj(post)
    if res.initial_cost is None or res.final_cost is None:
        res.reason = "无法计算 BA 前后重投影误差"
        return res
    if not res.final_cost < res.initial_cost:
        res.reason = (f"BA 未收敛（mean_sq_reproj {res.initial_cost:.6g} → "
                      f"{res.final_cost:.6g}）")
        return res

    res.ok = True
    res.reason = ""
    res.c2w_refined = c2w_ref
    res.intrinsics_refined = np.asarray(inputs.intrinsics_init, dtype=np.float64)
    res.points3D = np.asarray([rec.point3D(int(p)).xyz for p in rec.point3D_ids()],
                              dtype=np.float64)
    res.reproj_errors = post
    res.notes.append(f"reproj_obs={len(post)} median={res.g5_median:.4f} "
                     f"p95={res.g5_p95:.4f}")
    return res


__all__ = [
    "BAInputs",
    "BAResult",
    "PycolmapUnavailable",
    "build_ba_inputs",
    "triangulate_and_ba",
    "triangulate_dlt",
]
