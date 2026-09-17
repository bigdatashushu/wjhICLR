"""M3 传统 baseline：COLMAP SfM+MVS（subprocess CLI 调用，§4 M3）。

COLMAP 为外部 CLI 程序（BSD-3），通过 subprocess 调用；
预算：SfM 1-3min、MVS 5-15min（§4 M3 字段 10）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.reconstruction.vggt_runner import ReconstructionFailed

COLMAP_BIN = "colmap"  # TODO_USER_INPUT: colmap 可执行文件路径


def _run(cmd: list[str], timeout_s: int = 3600) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise ReconstructionFailed(f"COLMAP 命令失败: {cmd[1]}\n{proc.stderr[-2000:]}")


def write_frames(frames: Sequence[np.ndarray], image_dir: str | Path) -> None:
    """将帧序列写为 COLMAP 输入图像。"""
    import cv2

    d = Path(image_dir)
    d.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(frames):
        cv2.imwrite(str(d / f"{i:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def run_colmap(
    frames: Sequence[np.ndarray],
    scene_name: str,
    workspace: str | Path,
    colmap_bin: str = COLMAP_BIN,
) -> str:
    """COLMAP SfM+MVS baseline。

    返回 sparse 模型目录路径；完整 ReconstructionArtifact 组装
    （读取 sparse 模型 → c2w/深度/点云）TODO。
    """
    ws = Path(workspace) / scene_name
    image_dir = ws / "images"
    write_frames(frames, image_dir)
    db = ws / "database.db"
    sparse_dir = ws / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    _run([colmap_bin, "feature_extractor", "--database_path", str(db),
          "--image_path", str(image_dir)])
    _run([colmap_bin, "exhaustive_matcher", "--database_path", str(db)])
    _run([colmap_bin, "mapper", "--database_path", str(db),
          "--image_path", str(image_dir), "--output_path", str(sparse_dir)])

    # MVS（稠密化）可选：dense 目录
    dense_dir = ws / "dense"
    dense_dir.mkdir(exist_ok=True)
    _run([colmap_bin, "image_undistorter", "--image_path", str(image_dir),
          "--input_path", str(sparse_dir / "0"), "--output_path", str(dense_dir),
          "--output_type", "COLMAP"])
    # TODO: patch_match_stereo / stereo_fusion（MVS 5-15min），按需开启

    # TODO: 解析 sparse/0 的 images.bin/cameras.bin/points3D.bin
    # 组装 ReconstructionArtifact（recon_method="colmap"）
    return str(sparse_dir / "0")
