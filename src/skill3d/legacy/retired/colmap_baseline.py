"""M3 传统 baseline：COLMAP SfM(+可选 MVS)，产出统一 ReconstructionArtifact（G-15）。

COLMAP 为外部 CLI 程序（BSD-3），通过 subprocess 调用；
预算：SfM 1-3min、MVS 5-15min（§4 M3 字段 10）。

G-15 缺口（待修）：原来 `run_colmap` 只返回 sparse 目录路径、**不产出 artifact**
→ §10「三类 baseline 对比矩阵」无法产出。现补齐：

```
write_frames → feature_extractor → exhaustive_matcher → mapper
   → [可选] image_undistorter + patch_match_stereo + stereo_fusion   (MVS 稠密)
   → pycolmap 读回 sparse/0  → SparseArtifactData（位姿/K/稀疏深度/点云/残差）
   → ReconstructionArtifact(recon_method="colmap") + 自研尺度锚定（G-11）
```

降级纪律：pycolmap 缺失或 MVS 未开启时，深度为稀疏投影深度（0 = 无观测），
并由 M4 的 G5/G6 按"数据缺失"处理；不伪造稠密深度。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.reconstruction.vggt_runner import ReconstructionFailed

COLMAP_BIN = "colmap"  # TODO_USER_INPUT: colmap 可执行文件路径（§15.1 G-60）


def colmap_available(colmap_bin: str = COLMAP_BIN) -> bool:
    """COLMAP CLI 是否可用（未安装时 CLI 应给出明确提示而非 traceback）。"""
    return shutil.which(colmap_bin) is not None


def _run(cmd: list[str], timeout_s: int = 3600) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise ReconstructionFailed(f"COLMAP 命令失败: {cmd[1]}\n{proc.stderr[-2000:]}")


def write_frames(frames: Sequence[np.ndarray], image_dir: str | Path) -> list[str]:
    """将帧序列写为 COLMAP 输入图像；返回图像文件名序列（与帧序一致）。"""
    import cv2

    d = Path(image_dir)
    d.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i, img in enumerate(frames):
        name = f"{i:06d}.png"
        cv2.imwrite(str(d / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        names.append(name)
    return names


def run_sfm(image_dir: str | Path, workspace: str | Path,
            colmap_bin: str = COLMAP_BIN) -> Path:
    """SfM 三件套（feature_extractor → exhaustive_matcher → mapper）。返回 sparse 根目录。"""
    ws = Path(workspace)
    db = ws / "database.db"
    sparse_dir = ws / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    _run([colmap_bin, "feature_extractor", "--database_path", str(db),
          "--image_path", str(image_dir)])
    _run([colmap_bin, "exhaustive_matcher", "--database_path", str(db)])
    _run([colmap_bin, "mapper", "--database_path", str(db),
          "--image_path", str(image_dir), "--output_path", str(sparse_dir)])
    return sparse_dir


def run_mvs(image_dir: str | Path, sparse_dir: str | Path, workspace: str | Path,
            colmap_bin: str = COLMAP_BIN) -> Path:
    """MVS 稠密化（image_undistorter → patch_match_stereo → stereo_fusion）。

    返回 dense 目录；稠密深度图位于 `<dense>/stereo/depth_maps/*.bin`。
    """
    dense_dir = Path(workspace) / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    _run([colmap_bin, "image_undistorter", "--image_path", str(image_dir),
          "--input_path", str(sparse_dir / "0"), "--output_path", str(dense_dir),
          "--output_type", "COLMAP"])
    _run([colmap_bin, "patch_match_stereo", "--workspace_path", str(dense_dir),
          "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true"],
         timeout_s=3600)
    _run([colmap_bin, "stereo_fusion", "--workspace_path", str(dense_dir),
          "--workspace_format", "COLMAP", "--input_type", "geometric",
          "--output_path", str(dense_dir / "fused.ply")], timeout_s=3600)
    return dense_dir


def run_colmap(
    frames: Sequence[np.ndarray],
    scene_name: str,
    workspace: str | Path,
    colmap_bin: str = COLMAP_BIN,
    *,
    mvs: bool = False,
) -> tuple[Path, Path, Optional[Path]]:
    """跑 COLMAP SfM(+MVS)。返回 `(稀疏模型目录, workspace, dense 目录或 None)`。"""
    ws = Path(workspace) / scene_name
    image_dir = ws / "images"
    write_frames(frames, image_dir)
    sparse_dir = run_sfm(image_dir, ws, colmap_bin=colmap_bin)
    dense_dir = run_mvs(image_dir, sparse_dir, ws, colmap_bin=colmap_bin) if mvs else None
    return sparse_dir / "0", ws, dense_dir


def reconstruct_colmap(
    frames: Sequence[np.ndarray],
    scene_name: str,
    output_dir: str | Path,
    *,
    colmap_bin: str = COLMAP_BIN,
    mvs: bool = False,
    workspace: Optional[str | Path] = None,
    objects: Optional[Sequence[object]] = None,
    scale_calibration_path: Optional[str] = None,
    scale_confidence_level: Optional[float] = None,
) -> "object":
    """G-15：COLMAP → `ReconstructionArtifact`（统一 Schema，可进 §10 baseline 矩阵）。

    步骤：SfM(+MVS) → pycolmap 读回 → 数组落盘 → v4 尺度评估 → 组装 artifact。
    CLI 不可用 / pycolmap 不可用时抛 `ReconstructionFailed`（由降级链汇总）。
    """
    from skill3d.reconstruction import ba as ba_mod
    from skill3d.schemas.reconstruction import (
        ConfidenceMap,
        QualityMetrics,
        ReconstructionArtifact,
    )

    if not colmap_available(colmap_bin):
        raise ReconstructionFailed(
            f"colmap CLI 不可用（{colmap_bin}）；TODO_USER_INPUT: COLMAP_BIN 路径（§15.1 G-60）"
        )
    try:
        ba_mod.pycolmap_module()
    except ba_mod.ColmapUnavailable as exc:
        raise ReconstructionFailed(str(exc)) from exc

    ws_root = Path(workspace) if workspace else Path(output_dir) / f"colmap_ws_{scene_name}"
    sparse_model_dir, ws, dense_dir = run_colmap(
        frames, scene_name, ws_root, colmap_bin=colmap_bin, mvs=mvs)

    model = ba_mod.read_sparse_model(ws)
    if model is None:
        raise ReconstructionFailed(f"COLMAP sparse 模型读回失败: {sparse_model_dir}")
    summary = ba_mod.sparse_model_summary(model)

    ws_frame = frames[0].shape[:2] if frames else None
    image_size = (int(ws_frame[1]), int(ws_frame[0])) if ws_frame else None
    data = ba_mod.artifact_data_from_sparse(model, image_size=image_size)

    # MVS 稠密深度可用时替换稀疏深度（同名图像对齐）
    dense_note = "稀疏投影深度（未开 MVS）"
    if dense_dir is not None and ba_mod.dense_depth_available(dense_dir):
        replaced = 0
        for ti, name in enumerate(data.image_names):
            dense = ba_mod.read_dense_depth_map(dense_dir, name)
            if dense is None:
                continue
            h, w = data.depth_maps.shape[1], data.depth_maps.shape[2]
            if dense.shape[:2] != (h, w):
                continue
            data.depth_maps[ti] = np.where(np.isfinite(dense) & (dense > 0), dense, 0.0)
            data.point_conf[ti] = np.where(data.depth_maps[ti] > 0, 1.0, 0.0)
            replaced += 1
        dense_note = f"MVS 稠密深度已替换 {replaced}/{len(data.image_names)} 帧"
        data.point_map = _reproject_depth_to_world(data)

    refs = ba_mod.save_artifact_arrays(data, output_dir, scene_name)
    g5_med, g5_p95 = ba_mod.coarse_quality_from_sparse(
        data.reproj_errors, len(data.image_names), data.depth_maps)

    # v4 尺度评估（与 VGGT 主线同一实现，保证 baseline 可比）
    from skill3d.reconstruction.scale_assessment import (
        apply_scale_assessment,
        assess_scale,
    )

    assessment = assess_scale(data.point_map, data.c2w_list, objects=objects,
                              scene_name=scene_name,
                              calibration_path=scale_calibration_path,
                              confidence_level=scale_confidence_level)
    est = assessment

    # 质量由 M4 计算（P1 的 run_jobs / P2 的 quality_gate 会写回）。
    # 硬约束 22：这里**不得**用 NaN 占位冒充实算值 —— quality=None + not_computed，
    # route 判定 fail-closed（非 computed 一律不得停 full_3d）。
    _g5_med, _g5_p95 = g5_med, g5_p95

    artifact = ReconstructionArtifact(
        artifact_id=f"colmap-{scene_name}",
        artifact_version=str(abs(hash(
            (scene_name, len(data.image_names), summary["n_points3D"]))))[:16],
        scene_name=scene_name,
        recon_method="colmap",
        c2w_list=refs["c2w_list"],
        intrinsics=refs["intrinsics"],
        depth_maps=refs["depth_maps"],
        point_map=refs["point_map"],
        point_conf=refs["point_conf"],
        track_list=None,
        metric_scale=est.metric_scale,
        scale_known=est.scale_known,
        quality_status="not_computed",
        quality=None,
        # v5 HC37：COLMAP baseline 自带真 BA → 有真重投影残差；两个标量都有才算
        # computed，否则 not_available（不得声明算过却留空）。
        reprojection_status=(
            "computed" if (_g5_med is not None and _g5_p95 is not None
                           and np.isfinite(_g5_med) and np.isfinite(_g5_p95))
            else "not_available"),
        g5_reproj_err_median=(None if _g5_med is None or not np.isfinite(_g5_med)
                              else float(_g5_med)),
        g5_reproj_err_p95=(None if _g5_p95 is None or not np.isfinite(_g5_p95)
                           else float(_g5_p95)),
        confidence=ConfidenceMap(per_point_confidence=refs["point_conf"],
                                 coverage_count_per_frame=""),
        scale_confidence=est.confidence,
        scale_method=f"colmap+{est.method} [{dense_note}; sparse={summary}]",
        reproj_errors=refs["reproj_errors"],
    )
    return apply_scale_assessment(artifact, assessment)


def _reproject_depth_to_world(data) -> np.ndarray:
    """稠密深度替换后重新反投影点云（与 sparse 路径同一几何约定）。"""
    t = data.depth_maps.shape[0]
    point_map = np.zeros((t, *data.depth_maps.shape[1:], 3), dtype=np.float64)
    for ti in range(t):
        z = data.depth_maps[ti]
        mask = z > 0
        if not mask.any():
            continue
        vv, uu = np.nonzero(mask)
        k_inv = np.linalg.inv(data.intrinsics[ti])
        pix = np.stack([uu, vv, np.ones_like(uu)], axis=0).astype(np.float64)
        cam_pts = k_inv @ pix * z[vv, uu][None, :]
        c2w = data.c2w_list[ti]
        point_map[ti][vv, uu] = (c2w[:3, :3] @ cam_pts + c2w[:3, 3:4]).T
    return point_map
