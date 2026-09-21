"""M4 质量指标计算（v6 §10：无真值质量门，D11）。

v6 结构：

- **主门**（§10.1）= 跨视图 warp 内点率 ≥ τ_warp **且** 分组点云重叠率 ≥ τ_cloud。
  实测计算在 `reconstruction_gate.m4_main_gate`；本模块只负责组装、聚合与写回。
  **多指标不得单挑**（SysCON3D 依据：前馈 backbone 会幻觉跨视图一致性）；
- **诊断**（§10.2）：track 重投影残差、相邻帧旋转平滑 —— 只产告警，不决定主门；
- **conf-warp 单调自检**（§10.3）：VGGT conf 只作软权重；不单调则降权，
  **绝不**用 conf 作硬门（`C>2` 至多是可选掩码）；
- **G5 永久 not_available**、**G8 永久退役**、**G11 随尺度路线废止**：
  本模块**不产生**这三个指标，`QualityMetrics` 也不声明它们（出现即 hard fail）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from skill3d.gates import iqa
from skill3d.gates.input_gate import MIN_FRAMES, blur_floor_for
from skill3d.reconstruction_gate import m4_main_gate as mg
from skill3d.schemas.reconstruction import (
    QUALITY_METRIC_VERSION,
    QualityMetrics,
    ReconstructionArtifact,
)

# ---- 诊断阈值（全部 TODO_CALIBRATE，只产告警）----
TH_G3_MOTION: float = 20.0        # TODO_CALIBRATE: G3 运动模糊光流幅值（px）
TH_G6_DEPTH_CV: float = 1.0       # TODO_CALIBRATE: G6 深度方差系数 σ/μ
TH_G7_DYNAMIC: float = 0.3        # TODO_CALIBRATE: G7 动态物体占比
TH_G9_TRACK_IOU: float = 0.5      # TODO_CALIBRATE: G9 跟踪一致性 IoU
TH_G10_BASELINE: float = 0.1      # TODO_CALIBRATE: G10 基线质量（归一化）
TH_REPROJ_RESIDUAL_PX: float = 2.0  # TODO_CALIBRATE: track 重投影残差告警（px，诊断）
TH_ROT_SMOOTH_DEG: float = 15.0   # TODO_CALIBRATE: 相邻帧旋转跳变告警（度，诊断）

NaN = float("nan")


def _finite(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


# ------------------------------------------------------------ 输入/几何诊断 ----

def g1_blur_ok(frames: Sequence[np.ndarray]) -> float:
    """G1 帧模糊：通过**双判据**的帧占比，[0,1]（绝对下界 ∧ episode 相对判据）。"""
    if not frames:
        return NaN
    blurs = [iqa.laplacian_var(f) for f in frames]
    floor = blur_floor_for(blurs)
    return sum(b >= floor for b in blurs) / len(frames)


def g2_brightness(frames: Sequence[np.ndarray]) -> float:
    """G2 曝光：曝光正常帧占比，[0,1]。"""
    if not frames:
        return NaN
    from skill3d.gates.input_gate import TH_OVER_EXPOSED, TH_UNDER_EXPOSED

    ok = 0
    for f in frames:
        p_over, p_under = iqa.exposure_ratios(f)
        if p_over <= TH_OVER_EXPOSED and p_under <= TH_UNDER_EXPOSED:
            ok += 1
    return ok / len(frames)


def g3_motion_blur(frames: Sequence[np.ndarray],
                   depth_shape: Optional[tuple[int, int]] = None) -> float:
    """G3 运动模糊：帧间光流幅值均值（px）。**必须在 VGGT-depth-grid 上算**。"""
    if len(frames) < 2:
        return NaN
    if depth_shape is None:
        mags = [iqa.motion_score(frames[i - 1], frames[i]) for i in range(1, len(frames))]
        return float(np.mean(mags))
    from skill3d import coords

    mags = []
    for i in range(1, len(frames)):
        flow = coords.optical_flow_on_depth_grid([frames[i - 1], frames[i]], depth_shape)
        mags.append(float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))))
    return float(np.mean(mags))


def g6_depth_var_coeff(depth_maps: Optional[np.ndarray]) -> float:
    """G6 深度方差系数 σ/μ（无量纲）。"""
    if depth_maps is None:
        return NaN
    d = np.asarray(depth_maps, dtype=np.float64)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return NaN
    mu = float(np.mean(d))
    return NaN if mu <= 0 else float(np.std(d) / mu)


def g7_dynamic_ratio(dynamic_masks: Optional[np.ndarray]) -> float:
    """G7 动态物体占比：动态 mask 像素占比（数据来自 M5 SAM2 刚性残差）。"""
    if dynamic_masks is None:
        return NaN
    m = np.asarray(dynamic_masks)
    return NaN if m.size == 0 else float(np.mean(m.astype(bool)))


def g9_tracker_consistency(track_ious: Optional[Sequence[float]]) -> float:
    """G9 跟踪一致性：SAM2 mask 跨帧 IoU 均值，[0,1]。"""
    return NaN if not track_ious else float(np.mean(track_ious))


def g10_baseline_quality(c2w_list: Optional[np.ndarray]) -> float:
    """G10 基线质量：相邻帧基线中位数 / 场景直径（归一化）。"""
    if c2w_list is None:
        return NaN
    c2w = np.asarray(c2w_list, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4) or len(c2w) < 2:
        return NaN
    centers = c2w[:, :3, 3]
    baselines = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    diameter = float(np.max(np.linalg.norm(
        centers[None, :, :] - centers[:, None, :], axis=-1)))
    return NaN if diameter <= 0 else float(np.median(baselines) / diameter)


def track_reprojection_residual(program_trace=None) -> float:
    """§10.2 诊断：track 重投影残差中位数（px）。

    数据源是 M5 的 SAM2 track 传播残差，而不是 BA 残差 —— 它是**诊断**：
    只用于产生告警与降权，**不得**当作 G5 重投影残差（§10.4 禁止代理值）。
    无数据 → NaN。
    """
    return NaN


def rotation_smoothness(c2w_list: Optional[np.ndarray]) -> float:
    """§10.2 诊断：相邻帧旋转跳变（度，越小越平滑）。无位姿 → NaN。"""
    if c2w_list is None:
        return NaN
    c2w = np.asarray(c2w_list, dtype=np.float64)
    if c2w.ndim != 3 or len(c2w) < 2:
        return NaN
    jumps = []
    for i in range(1, len(c2w)):
        R0, R1 = c2w[i - 1][:3, :3], c2w[i][:3, :3]
        c = float(np.clip((np.trace(R1.T @ R0) - 1.0) / 2.0, -1.0, 1.0))
        jumps.append(float(np.degrees(np.arccos(c))))
    return float(np.median(jumps)) if jumps else NaN


def _norm_scores(q: dict) -> list[float]:
    """把诊断指标归一化为 [0,1] 的"越高越好"分数（NaN 项跳过，**不补分**）。

    v6：G5/G8/G11 都不参与（G5 永久 not_available、G8 退役、G11 随尺度路线废止）。
    """
    scores = []
    for v in (q["g1_blur_ok"], q["g2_brightness"]):
        if np.isfinite(v):
            scores.append(float(v))
    if np.isfinite(q["g3_motion_blur"]):
        scores.append(1.0 if q["g3_motion_blur"] <= TH_G3_MOTION else 0.0)
    scores.append(1.0 if q["g4_frame_count"] >= MIN_FRAMES
                  else q["g4_frame_count"] / MIN_FRAMES)
    for v, th in ((q["g6_depth_var_coeff"], TH_G6_DEPTH_CV),
                  (q["g7_dynamic_ratio"], TH_G7_DYNAMIC)):
        if np.isfinite(v):
            scores.append(1.0 if v <= th else 0.0)
    for v, th in ((q["g9_tracker_consistency"], TH_G9_TRACK_IOU),
                  (q["g10_baseline_quality"], TH_G10_BASELINE)):
        if np.isfinite(v):
            scores.append(1.0 if v >= th else (v / th if th > 0 else 0.0))
    return scores


def compute_quality(
    artifact: Optional[ReconstructionArtifact] = None,
    *,
    frames: Optional[Sequence[np.ndarray]] = None,
    depth_maps: Optional[np.ndarray] = None,
    c2w_list: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    point_map: Optional[np.ndarray] = None,
    depth_conf: Optional[np.ndarray] = None,
    dynamic_masks: Optional[np.ndarray] = None,
    track_ious: Optional[Sequence[float]] = None,
) -> QualityMetrics:
    """计算 v6 质量指标（主门 + 诊断），算不出的项为 NaN / None，**不伪造**。

    `overall_quality` 的口径：**主门未过 → 0.0**；主门过了 → 有限诊断项的归一化均值。
    这样 `overall_quality` 单调反映主门，路由用它做 NaN 检查（§6.2 fail-closed）。
    """
    art_q = artifact.quality if artifact is not None else None
    depth_shape = None
    if depth_maps is not None:
        try:
            from skill3d.coords import depth_grid_shape

            depth_shape = depth_grid_shape(depth_maps)
        except Exception:  # noqa: BLE001 - 深度数组异常：G3 退化为原始分辨率
            depth_shape = None

    gate = mg.main_gate({
        "frames": frames,
        "depth_maps": depth_maps,
        "c2w_list": c2w_list,
        "intrinsics": intrinsics,
        "point_map": point_map,
        "depth_conf": depth_conf,
    })
    # 主门把实算比率放在返回值的 `values` 子字典里（§10.1；`main_gate` 的输入才是
    # 扁平键，输出不是）——必须从 `values` 取，否则三个比率恒为 NaN，
    # 让 trace 里的主门数字变成"算过但没记"，违反 §5.2 的"quality 是唯一事实源"。
    gate_values = dict(gate.get("values") or {})

    def _fb(val: float, name: str) -> float:
        return val if np.isfinite(val) or art_q is None else float(getattr(art_q, name))

    diag_warns: list[str] = list(gate.get("warnings") or [])
    reproj_res = track_reprojection_residual()
    rot_smooth = rotation_smoothness(c2w_list)
    if np.isfinite(reproj_res) and reproj_res > TH_REPROJ_RESIDUAL_PX:
        diag_warns.append(f"track 重投影残差中位数 {reproj_res:.2f}px 超阈"
                          "（诊断告警，不计入主门；§10.2）")
    if np.isfinite(rot_smooth) and rot_smooth > TH_ROT_SMOOTH_DEG:
        diag_warns.append(f"相邻帧旋转跳变中位数 {rot_smooth:.1f}° 超阈"
                          "（诊断告警，不计入主门；§10.2）")

    q = {
        "warp_inlier_ratio": float(gate_values.get("warp_inlier_ratio", NaN)),
        "warp_photometric_inlier_ratio": float(
            gate_values.get("warp_photometric_inlier_ratio", NaN)),
        "cloud_overlap_ratio": float(gate_values.get("cloud_overlap_ratio", NaN)),
        "main_gate_passed": bool(gate.get("main_gate_passed", False)),
        "gate_thresholds": {k: float(v) for k, v in
                            (gate.get("thresholds") or {}).items()},
        "g1_blur_ok": _fb(g1_blur_ok(frames) if frames else NaN, "g1_blur_ok"),
        "g2_brightness": _fb(g2_brightness(frames) if frames else NaN, "g2_brightness"),
        "g3_motion_blur": _fb(g3_motion_blur(frames, depth_shape) if frames else NaN,
                              "g3_motion_blur"),
        "g4_frame_count": len(frames) if frames is not None else (
            art_q.g4_frame_count if art_q is not None else 0),
        "g6_depth_var_coeff": _fb(g6_depth_var_coeff(depth_maps)
                                  if depth_maps is not None else NaN,
                                  "g6_depth_var_coeff"),
        "g7_dynamic_ratio": _fb(g7_dynamic_ratio(dynamic_masks), "g7_dynamic_ratio"),
        "g9_tracker_consistency": _fb(g9_tracker_consistency(track_ious),
                                      "g9_tracker_consistency"),
        "g10_baseline_quality": _fb(g10_baseline_quality(c2w_list),
                                    "g10_baseline_quality"),
        "track_reproj_residual_median": _finite(reproj_res),
        "rotation_smoothness": _finite(rot_smooth),
        "diagnostic_warnings": diag_warns,
        "conf_warp_monotonic": gate.get("conf_warp_monotonic"),
        "conf_warp_spearman": _finite(gate.get("conf_warp_spearman")),
    }
    if not q["main_gate_passed"]:
        q["overall_quality"] = 0.0
    else:
        scores = _norm_scores(q)
        q["overall_quality"] = float(np.mean(scores)) if scores else 1.0
    return QualityMetrics(**q)


# --------------------------------------------------- 质量写回（硬约束 22）----

def quality_is_computed(art: Optional[ReconstructionArtifact]) -> bool:
    """artifact 的质量是否可用（fail-closed 判定的唯一入口）。"""
    if art is None:
        return False
    if getattr(art, "quality_status", "not_computed") != "computed":
        return False
    q = getattr(art, "quality", None)
    if q is None:
        return False
    return bool(np.isfinite(float(q.overall_quality)))


def apply_quality(
    art: ReconstructionArtifact,
    q: Optional[QualityMetrics],
    *,
    status: Literal["computed", "failed"] = "computed",
    artifact_path: Optional[str | Path] = None,
) -> ReconstructionArtifact:
    """把质量写回 artifact（**不可变**：返回新版本，不原地改老对象）。

    G5 在 v6 恒为 `not_available` + `None`（§10.4）：本函数**不再**读取或写入任何
    G5 标量，也不允许任何代理值进入。
    """
    update: dict = {"quality_status": status,
                    "quality": q if status == "computed" else None}
    new_art = art.model_copy(update=update)
    if artifact_path:
        write_artifact_json(new_art, artifact_path)
    return new_art


def write_artifact_json(art: ReconstructionArtifact, path: str | Path) -> Path:
    """原子写 artifact JSON（先写临时文件再 replace，避免半截文件被 P2 读到）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(art.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def compute_and_store_quality(
    art: ReconstructionArtifact,
    *,
    frames: Optional[Sequence[np.ndarray]] = None,
    depth_maps: Optional[np.ndarray] = None,
    c2w_list: Optional[np.ndarray] = None,
    intrinsics: Optional[np.ndarray] = None,
    point_map: Optional[np.ndarray] = None,
    depth_conf: Optional[np.ndarray] = None,
    dynamic_masks: Optional[np.ndarray] = None,
    track_ious: Optional[Sequence[float]] = None,
    artifact_path: Optional[str | Path] = None,
) -> ReconstructionArtifact:
    """M4 单一入口：算质量 → 写回 artifact（可选落盘）。

    数据源全缺（无 frames / depth）时**不得**标 "computed"：此时主门必然算不出，
    `overall_quality=0.0` 会让场景被判 fallback —— 但那是"缺少输入"而不是"算过且差"。
    故这种情况记 `quality_status="failed"` + `quality=None`（route 必落 fallback）。
    """
    if frames is None and depth_maps is None:
        return apply_quality(art, None, status="failed", artifact_path=artifact_path)
    q = compute_quality(art, frames=frames, depth_maps=depth_maps, c2w_list=c2w_list,
                        intrinsics=intrinsics, point_map=point_map,
                        depth_conf=depth_conf, dynamic_masks=dynamic_masks,
                        track_ious=track_ious)
    return apply_quality(art, q, status="computed", artifact_path=artifact_path)


__all__ = [
    "QUALITY_METRIC_VERSION",
    "apply_quality",
    "compute_and_store_quality",
    "compute_quality",
    "quality_is_computed",
    "write_artifact_json",
]
