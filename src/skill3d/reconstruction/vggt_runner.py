"""M3 VGGT 主线重建（v6 §3.1 三层结构）。

在线模块，严禁任何离线强模型相关依赖。torch/vggt 全部 lazy import。

**API 已对官方 repo 核验**（`third_party/vggt`，2026-09-18 从 facebookresearch/vggt 取源码）：
- `VGGT.from_pretrained(ckpt)`（`PyTorchModelHubMixin`），权重 `VGGT-1B/model.safetensors`
- `load_and_preprocess_images(list_of_PATHS, mode="crop") -> (N,3,H,W) in [0,1]`
  —— 注意收的是**文件路径**，不是 ndarray（本模块负责把帧写盘）
- `model(images) -> dict`，键为 `pose_enc / depth / depth_conf / world_points /
  world_points_conf / images`（+ 传 query_points 时才有 `track/vis/conf`）
- `pose_encoding_to_extri_intri(pose_enc, (H,W)) -> (extri, intri)`，其中
  **extri 是 world→camera（OpenCV 约定，B×S×3×4）**，故 `c2w = inv(extri)`
- VGGT 的世界系即**第一帧相机坐标系** → `c2w[0] = I`

v6 三层（§3.1）：

- 第①层：VGGT 前馈 → c2w / K / depth / point_map / conf（**不变**）；
- 第②层：**世界系契约**（D5）—— `estimate_world_frame(c2w)` 出 `world_up` +
  `handedness`，落进 artifact；缺失时方向/路线类 Tool fail-closed；
- 第③层：**度量尺度融合**（D1/D2，`[待实验]`）—— 零样本度量深度模型逐帧出
  `s_k`，跨帧 median 融合。默认 `metric_model="none"` →
  `scale_fusion_status="not_run"` + `metric_scale=None`（PoC 未通过前**不许**冒充）。

**BA 正式关闭**（D11）：`recon_method` 只允许 `vggt`；G5 永久 `not_available`。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.reconstruction_gate.world_frame import estimate_world_frame
from skill3d.schemas.reconstruction import (
    ConfidenceMap,
    QualityMetrics,
    ReconstructionArtifact,
)

# VGGT-1B 权重目录（本机已就位：weights 4.7GB + config.json）
VGGT_CHECKPOINT: Optional[str] = "/home/cvailab/.cache/huggingface/geothinker/VGGT-1B"
VGGT_DTYPE = "bfloat16"


def _artifact_version(payload: bytes) -> str:
    """内容寻址哈希（§5.2 artifact_version）。"""
    return hashlib.sha256(payload).hexdigest()[:16]


class ReconstructionFailed(RuntimeError):
    """重建失败（v6 无降级链：VGGT 是唯一正式主线）。"""


def run_vggt(
    frames: Sequence[np.ndarray],
    scene_name: str,
    output_dir: str | Path,
    checkpoint: Optional[str] = VGGT_CHECKPOINT,
    objects: Optional[Sequence[object]] = None,
    *,
    frame_set: Optional[object] = None,
    compute_quality: bool = True,
    quality_artifact_path: Optional[str] = None,
    metric_depth_model=None,
    metric_model_name: str = "none",
) -> ReconstructionArtifact:
    """VGGT 单网络前向 → 世界系契约 → 度量尺度融合（可选）→ 组装 artifact。

    `metric_depth_model`：实现了 `MetricDepthModel` 协议的零样本度量深度模型
    （首个 PoC = MoGe-2，见 `reconstruction/metric_fusion.make_moge2_model`）。
    为 `None` 时**不做**融合，artifact 写 `scale_fusion_status="not_run"` ——
    这是当前默认状态（§11 全部为 `[待实验]`，PoC 通过前 metric_scale 必须为 None）。
    """
    try:
        import torch  # lazy import：未安装库
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
    except Exception as e:  # noqa: BLE001
        raise ReconstructionFailed(f"vggt/torch 不可用: {e}") from e

    if checkpoint is None:
        raise ReconstructionFailed("VGGT checkpoint 未配置")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = VGGT.from_pretrained(checkpoint).to(device).eval()

    # 官方 API 收**文件路径**：把 32 帧写到 work 目录再喂进去（帧已由 M1 抽好）
    frame_dir = out / f"{scene_name}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    import cv2

    frame_paths: list[str] = []
    for i, fr in enumerate(frames):
        p = frame_dir / f"{i:04d}.png"
        if not p.is_file():
            cv2.imwrite(str(p), cv2.cvtColor(np.asarray(fr), cv2.COLOR_RGB2BGR))
        frame_paths.append(str(p))

    images = load_and_preprocess_images(frame_paths)
    images = images.to(device)
    with torch.no_grad():
        autocast = getattr(torch, "amp", None)
        if autocast is not None and hasattr(autocast, "autocast"):
            ctx = autocast.autocast("cuda", dtype=dtype, enabled=device == "cuda")
        else:  # pragma: no cover - 旧 torch
            ctx = torch.cuda.amp.autocast(dtype=dtype, enabled=device == "cuda")
        with ctx:
            preds = model(images)

    preds = _normalize_preds(dict(preds), images)

    # ---------------- 第②层：世界系契约（D5）----------------
    wf = estimate_world_frame(preds["c2w"])

    # ---------------- 第③层：度量尺度融合（D1/D2 [待实验]）----------------
    fusion = _run_metric_fusion(
        metric_depth_model, metric_model_name,
        frames=frames, preds=preds, intrinsics=preds["intrinsic"],
        out_dir=out, scene_name=scene_name)

    # ---- 落盘并组装 artifact（字段为数组文件 ref，见 §5.2）----
    def _save(name: str, arr) -> str:
        p = out / f"{scene_name}_{name}.npy"
        arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
        np.save(p, arr)
        return str(p)

    c2w_ref = _save("c2w", preds["c2w"])
    intr_ref = _save("intrinsics", preds["intrinsic"])
    depth_ref = _save("depth", preds["depth_map"])
    pmap_ref = _save("point_map", preds["point_map"])
    pconf_ref = _save("point_conf", preds["point_conf"])
    dconf_ref = _save("depth_conf", preds["depth_conf"]) if preds.get("depth_conf") is not None else ""
    track_ref = _save("tracks", preds["track_list"]) if preds.get("track_list") is not None else None

    frame_ids = list(range(len(frames)))
    source_frame_indices = list(range(len(frames)))
    timestamps = [float(i) for i in range(len(frames))]
    fsh = ""
    if frame_set is not None:
        frame_ids = [int(i) for i in getattr(frame_set, "frame_ids", frame_ids)]
        source_frame_indices = [int(i) for i in getattr(
            frame_set, "source_frame_indices", frame_ids)]
        timestamps = [float(t) for t in getattr(frame_set, "timestamps", timestamps)]
        fsh = str(getattr(frame_set, "frame_set_hash", "") or "")
    if not fsh:  # 无 FrameSet 传入时按规范槽位算哈希（jsonl/合成来源）
        from skill3d.adapters.frame_set import frame_set_hash as _fsh

        fsh = _fsh(frame_ids)

    artifact = ReconstructionArtifact(
        artifact_id=f"vggt-{scene_name}",
        artifact_version=_artifact_version(scene_name.encode() + str(len(frames)).encode()),
        scene_name=scene_name,
        recon_method="vggt",
        frame_ids=frame_ids,
        source_frame_indices=source_frame_indices,
        timestamps=timestamps,
        frame_set_hash=fsh,
        c2w_list=c2w_ref,
        intrinsics=intr_ref,
        depth_maps=depth_ref,
        point_map=pmap_ref,
        point_conf=pconf_ref,
        depth_conf=dconf_ref,
        track_list=track_ref,
        # 世界系契约（D5）：M3 估计并落盘，M4 校验存在性
        world_up=wf.world_up,
        handedness=wf.handedness,      # type: ignore[arg-type]
        world_frame_status=wf.status,  # type: ignore[arg-type]
        # 度量尺度融合（默认 not_run；PoC 通过前 metric_scale 必须为 None）
        metric_scale=fusion.metric_scale,
        scale_self_consistency=fusion.scale_self_consistency,
        per_frame_scale_ref=fusion.receipt_ref,
        metric_model=fusion.model,     # type: ignore[arg-type]
        metric_fusion_version=fusion.version,
        scale_fusion_status=fusion.status,  # type: ignore[arg-type]
        quality_status="not_computed",
        quality=None,
        confidence=ConfidenceMap(per_point_confidence=pconf_ref,
                                 coverage_count_per_frame=""),
        # D11：G5 永久 not_available（本 Schema 已固定，无需显式赋值）
    )
    if compute_quality:
        from skill3d.reconstruction_gate import quality_metrics as qm

        artifact = qm.compute_and_store_quality(
            artifact, frames=frames, depth_maps=preds["depth_map"],
            c2w_list=preds["c2w"], intrinsics=preds["intrinsic"],
            point_map=preds["point_map"], depth_conf=preds.get("depth_conf"),
            artifact_path=quality_artifact_path,
        )
    return artifact


class _FusionOutcome:
    """融合结果的 artifact 视图（把 `ScaleFusionResult` 映射到 §5.2 字段）。"""

    __slots__ = ("status", "metric_scale", "scale_self_consistency",
                 "receipt_ref", "version", "model")

    def __init__(self, *, status: str, metric_scale: Optional[float],
                 scale_self_consistency: Optional[float],
                 receipt_ref: Optional[str], version: str, model: str) -> None:
        self.status = status
        self.metric_scale = metric_scale
        self.scale_self_consistency = scale_self_consistency
        self.receipt_ref = receipt_ref
        self.version = version
        self.model = model


def _run_metric_fusion(model, model_name: str, *, frames, preds, intrinsics,
                       out_dir: Path, scene_name: str) -> _FusionOutcome:
    """逐帧度量深度 → 跨帧尺度融合（§11.1）；模型缺失即 `not_run`。

    **绝不**在没有真模型时编造尺度（§11.4 失败纪律：不达标即关闭支路，
    不回退多锚点/校准池）。
    """
    from skill3d.reconstruction import metric_fusion as mf

    if model is None:
        return _FusionOutcome(
            status="not_run", metric_scale=None, scale_self_consistency=None,
            receipt_ref=None, version=mf.METRIC_FUSION_VERSION, model="none")

    depth_vggt = np.asarray(preds["depth_map"], dtype=np.float64)
    n = depth_vggt.shape[0]
    metric_list: list[Optional[np.ndarray]] = []
    valid_masks: list[Optional[np.ndarray]] = []
    conf_list: list[Optional[np.ndarray]] = []
    dconf = preds.get("depth_conf")
    for i in range(n):
        K = np.asarray(intrinsics, dtype=np.float64)
        K_i = K[i] if K.ndim == 3 else K
        try:
            out = model.infer(np.asarray(frames[i]), K_i)
        except Exception:  # noqa: BLE001 - 单帧失败不阻断（该帧 s_k 记 None）
            metric_list.append(None)
            valid_masks.append(None)
            conf_list.append(None)
            continue
        d = np.asarray(out.depth_metric, dtype=np.float64)
        vggt_hw = depth_vggt[i].shape
        if d.shape != vggt_hw:
            # 度量模型输出在输入 RGB 分辨率；必须重采样到 VGGT 深度网格，
            # 否则逐像素比值没有意义（§11.2 相机对齐的一部分）。
            import cv2

            d = cv2.resize(d, (vggt_hw[1], vggt_hw[0]), interpolation=cv2.INTER_LINEAR)
            vm = np.asarray(out.valid_mask, dtype=np.uint8)
            vm = cv2.resize(vm, (vggt_hw[1], vggt_hw[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            vm = np.asarray(out.valid_mask, dtype=bool)
        metric_list.append(d)
        valid_masks.append(vm)
        conf_list.append(None if dconf is None else np.asarray(dconf[i], dtype=np.float64))

    # conf 的 conf-warp 单调性自检属于 M4（需要 warp 残差，此时还没算）；
    # 按 §10.3 的纪律，**未自检的 conf 不作任何过滤**（传 None = 保守不用）。
    fusion = mf.fuse_metric_scale(
        metric_list, [depth_vggt[i] for i in range(n)], valid_masks,
        conf_list,
        conf_warp_monotonic=None,
        model=str(model_name or getattr(model, "name", "none")),
    )
    receipt_ref = None
    try:
        receipt_path = out_dir / f"{scene_name}_scale_receipt.json"
        mf.write_per_frame_receipt(fusion, receipt_path)
        receipt_ref = str(receipt_path)
    except Exception:  # noqa: BLE001 - receipt 缺失 → valid_frame_ratio 读不到 → 门不通过
        receipt_ref = None
    return _FusionOutcome(
        status=fusion.status,
        metric_scale=fusion.metric_scale,
        scale_self_consistency=fusion.scale_self_consistency,
        receipt_ref=receipt_ref,
        version=fusion.version,
        model=fusion.model)


def _to_numpy(x) -> np.ndarray:
    """torch.Tensor / ndarray → float64 ndarray（去掉 batch 维与尾部单通道）。"""
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    a = np.asarray(x, dtype=np.float64)
    if a.ndim >= 1 and a.shape[0] == 1:      # VGGT 输出带 B 维（B=1）
        a = a[0]
    if a.ndim == 4 and a.shape[-1] == 1:     # depth: (S,H,W,1) → (S,H,W)
        a = a[..., 0]
    return a


def _normalize_preds(raw: dict, images) -> dict:
    """VGGT 官方输出键 → §5.2 字段（已对官方 repo 核验）。"""
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    out: dict = {}
    pose_enc = raw.get("pose_enc")
    if pose_enc is None:
        raise ReconstructionFailed("VGGT 输出缺 pose_enc（前向失败）")
    hw = (int(images.shape[-2]), int(images.shape[-1]))
    extri, intri = pose_encoding_to_extri_intri(pose_enc, hw)
    extri_np = _to_numpy(extri)                      # (S,3,4) world→cam
    n_frames = extri_np.shape[0]
    w2c = np.tile(np.eye(4), (n_frames, 1, 1))
    w2c[:, :3, :4] = extri_np
    out["c2w"] = np.linalg.inv(w2c)                  # (S,4,4)
    out["intrinsic"] = _to_numpy(intri)

    for src, dst in (("depth", "depth_map"), ("world_points", "point_map"),
                     ("world_points_conf", "point_conf"), ("depth_conf", "depth_conf")):
        if src in raw:
            out[dst] = _to_numpy(raw[src])
    if "track" in raw:
        out["track_list"] = _to_numpy(raw["track"])
    if "images" in raw:
        out["images"] = _to_numpy(raw["images"])
    return out


def reconstruct(
    frames: Sequence[np.ndarray],
    scene_name: str,
    output_dir: str | Path,
    method: str = "vggt",
    *,
    frame_set: Optional[object] = None,
    compute_quality: bool = True,
    metric_depth_model=None,
    metric_model_name: str = "none",
) -> ReconstructionArtifact:
    """重建入口（v6：**唯一**正式主线是 `vggt`，无降级链、无 BA）。

    v6 §5.2 把 `recon_method` 收成 `Literal["vggt"]`：COLMAP / DUSt3R-MAST3R /
    `vggt_sparse_ba` / 官方 VGGSfM BA 全部退出生产路线（依据见 §20 废止表），
    它们的历史实现在 `skill3d/legacy/retired/` 只读归档。因此"降级链"不再存在：
    VGGT 失败即该 episode 重建失败，scene_route 落 `fallback_2d_only`。
    """
    if method != "vggt":
        raise ValueError(
            f"未知/已废止重建方法: {method}（v6 §5.2：recon_method 只允许 'vggt'；"
            "历史 colmap / dust3r_mast3r / vggt_sparse_ba / vggt_ba 见 "
            "skill3d/legacy/retired/ 与 §20 废止表）")
    return run_vggt(frames, scene_name, output_dir,
                    frame_set=frame_set, compute_quality=compute_quality,
                    metric_depth_model=metric_depth_model,
                    metric_model_name=metric_model_name)


__all__ = [
    "VGGT_CHECKPOINT",
    "ReconstructionFailed",
    "reconstruct",
    "run_vggt",
]
