"""M3 VGGT 主线重建（§4 M3）。

在线模块，严禁任何 GPT-6 相关依赖（硬约束 1）。
torch/vggt 未安装，全部 lazy import；权重路径 TODO。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.schemas.reconstruction import ReconstructionArtifact

# VGGT 权重 checkpoint（TODO：VGGT-1B-Commercial 申请流程，§4 M3 字段 12）
VGGT_CHECKPOINT: Optional[str] = None  # TODO_USER_INPUT
# bf16 推理（§4 M3：~12-16GB VRAM/32帧，4090 可跑）
VGGT_DTYPE = "bfloat16"


def _artifact_version(payload: bytes) -> str:
    """内容寻址哈希（§5 artifact_version）。"""
    return hashlib.sha256(payload).hexdigest()[:16]


class ReconstructionFailed(RuntimeError):
    """重建失败，触发降级链。"""


def run_vggt(
    frames: Sequence[np.ndarray],
    scene_name: str,
    output_dir: str | Path,
    checkpoint: Optional[str] = VGGT_CHECKPOINT,
    use_ba: bool = True,
) -> ReconstructionArtifact:
    """VGGT 单网络前向（+可选 BA），输出组装为 ReconstructionArtifact。

    TODO: vggt 具体 API（load_and_preprocess_images / model 前向签名）
    以官方 repo README 为准（§4 M3 字段 5）。
    """
    try:
        import torch  # lazy import：未安装库
        from vggt.models.vggt import VGGT  # TODO: 类名以官方 repo 为准
        from vggt.utils.load_fn import load_and_preprocess_images  # TODO
    except Exception as e:  # noqa: BLE001
        raise ReconstructionFailed(f"vggt/torch 不可用: {e}") from e

    if checkpoint is None:
        raise ReconstructionFailed("VGGT checkpoint 未配置（TODO_USER_INPUT）")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = VGGT.from_pretrained(checkpoint).to(device).eval()

    images = load_and_preprocess_images(list(frames)).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=device == "cuda"):
        preds = model(images)

    if use_ba:
        # 官方 demo_colmap.py --use_ba 的 BA 流程（TODO: 具体函数以 repo 为准）
        try:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # TODO

            preds = dict(preds)
            pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
        except Exception:
            pass  # BA 失败不致命，继续用前向结果

    # ---- 落盘并组装 artifact（字段为数组文件 ref，见 §5.2）----
    def _save(name: str, arr) -> str:
        p = out / f"{scene_name}_{name}.npy"
        arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
        np.save(p, arr)
        return str(p)

    c2w_ref = _save("c2w", preds["extrinsic"])      # 世界坐标 SE(3) 序列
    intr_ref = _save("intrinsics", preds["intrinsic"])
    depth_ref = _save("depth", preds["depth_map"])
    pmap_ref = _save("point_map", preds["point_map"])
    pconf_ref = _save("point_conf", preds["point_conf"])
    track_ref = _save("tracks", preds["track_list"]) if "track_list" in preds else None

    # metric 尺度锚定（PaGeR），失败则 scale_known=False
    from skill3d.reconstruction.metric_scale import anchor_metric_scale

    scale, _ci, scale_known = anchor_metric_scale(depth_ref, scene_name=scene_name)

    # quality/confidence 由 M4 计算，此处填占位（真实值由 quality_gate 覆写）
    from skill3d.schemas.reconstruction import (
        ConfidenceMap,
        QualityMetrics,
    )

    nan = float("nan")
    quality = QualityMetrics(
        g1_blur_ok=nan, g2_brightness=nan, g3_motion_blur=nan, g4_frame_count=len(frames),
        g5_reproj_err_median=nan, g5_reproj_err_p95=nan, g6_depth_var_coeff=nan,
        g7_dynamic_ratio=nan, g8_bbox_coverage_min=nan, g9_tracker_consistency=nan,
        g10_baseline_quality=nan, g11_scale_ci=nan, overall_quality=nan,
    )
    confidence = ConfidenceMap(per_point_confidence=pconf_ref, coverage_count_per_frame="")

    return ReconstructionArtifact(
        artifact_id=f"vggt-{scene_name}",
        artifact_version=_artifact_version(scene_name.encode() + str(len(frames)).encode()),
        scene_name=scene_name,
        recon_method="vggt",
        c2w_list=c2w_ref,
        intrinsics=intr_ref,
        depth_maps=depth_ref,
        point_map=pmap_ref,
        point_conf=pconf_ref,
        track_list=track_ref,
        metric_scale=scale if scale_known else None,
        scale_known=scale_known,
        quality=quality,
        confidence=confidence,
    )


def reconstruct(
    frames: Sequence[np.ndarray],
    scene_name: str,
    output_dir: str | Path,
    method: str = "vggt",
) -> ReconstructionArtifact:
    """重建入口 + 降级链（§4 M3 字段 9）：
    vggt → dust3r_mast3r → 单目深度 2D-only（后者不产生 3D artifact）。

    每次降级都保留上一级失败原因，最终异常里汇总（便于定位缺权重/缺依赖）。
    """
    reasons: list[str] = []

    if method == "vggt":
        try:
            return run_vggt(frames, scene_name, output_dir)
        except ReconstructionFailed as exc:
            reasons.append(f"vggt: {exc}")
            method = "dust3r_mast3r"  # 降级链

    if method == "dust3r_mast3r":
        from skill3d.reconstruction.dust32_mast3r_fallback import run_dust3r_mast3r

        try:
            return run_dust3r_mast3r(frames, scene_name, output_dir)
        except ReconstructionFailed as exc:
            reasons.append(f"dust3r_mast3r: {exc}")
            method = "monocular"

    if method == "monocular":
        # 单目深度 2D-only：不产出 ReconstructionArtifact，
        # 上游应将 SceneState.route 置为 "fallback_2d_only"
        raise ReconstructionFailed(
            "所有 3D 重建路径失败，降级为单目深度 2D-only（不产生 3D artifact）；"
            "各级原因: " + " | ".join(reasons)
        )

    raise ValueError(f"未知重建方法: {method}")
