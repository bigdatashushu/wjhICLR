"""M3 备选重建：DUSt3R + MASt3R 两阶段 fallback（§4 M3）。

License: CC BY-NC-SA 4.0（仅研究用）。torch/dust3r/mast3r 未安装，lazy import。
TODO: 具体类名/接口以 https://github.com/naver/dust3r 与
https://github.com/naver/mast3r README 为准（§4 M3 字段 5）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.schemas.reconstruction import ConfidenceMap, QualityMetrics, ReconstructionArtifact
from skill3d.reconstruction.vggt_runner import ReconstructionFailed, _artifact_version

DUST3R_CHECKPOINT: Optional[str] = None  # TODO_USER_INPUT
MAST3R_CHECKPOINT: Optional[str] = None  # TODO_USER_INPUT


def run_dust3r_mast3r(
    frames: Sequence[np.ndarray],
    scene_name: str,
    output_dir: str | Path,
) -> ReconstructionArtifact:
    """DUSt3R/MASt3R 两阶段相对点图重建（TODO: API 待核验）。"""
    try:
        import torch  # lazy import
        import dust3r  # noqa: F401  # TODO: 具体 API 未核验
    except Exception as e:  # noqa: BLE001
        raise ReconstructionFailed(f"dust3r/mast3r/torch 不可用: {e}") from e

    # TODO: 完整两阶段流程（成对点图 → 全局对齐 → BA）待接入真实权重后实现
    raise ReconstructionFailed("dust3r_mast3r fallback 尚未接入真实权重（TODO）")


def _placeholder_artifact(scene_name: str, out_refs: dict) -> ReconstructionArtifact:
    """组装 artifact 的公共辅助（供未来真实实现复用）。"""
    # 硬约束 22：质量未算就是 not_computed + None（不得用 NaN 占位冒充实算值）；
    # 由 M4（P1 run_jobs 或 P2 quality_gate）实算后写回。
    confidence = ConfidenceMap(
        per_point_confidence=out_refs.get("point_conf", ""),
        coverage_count_per_frame="",
    )
    return ReconstructionArtifact(
        artifact_id=f"dust3r_mast3r-{scene_name}",
        artifact_version=_artifact_version(scene_name.encode()),
        scene_name=scene_name,
        recon_method="dust3r_mast3r",
        c2w_list=out_refs.get("c2w", ""),
        intrinsics=out_refs.get("intrinsics", ""),
        depth_maps=out_refs.get("depth", ""),
        point_map=out_refs.get("point_map", ""),
        point_conf=out_refs.get("point_conf", ""),
        track_list=None,
        metric_scale=None,
        scale_known=False,
        quality_status="not_computed",
        quality=None,
        confidence=confidence,
    )
