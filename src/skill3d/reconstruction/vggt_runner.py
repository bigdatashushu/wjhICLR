"""M3 VGGT 主线重建（§4 M3）。

在线模块，严禁任何 GPT-6 相关依赖（硬约束 1）。torch/vggt 全部 lazy import。

**API 已对官方 repo 核验**（`third_party/vggt`，2026-09-18 从 facebookresearch/vggt 取源码）：
- `VGGT.from_pretrained(ckpt)`（`PyTorchModelHubMixin`），权重 `VGGT-1B/model.safetensors`
- `load_and_preprocess_images(list_of_PATHS, mode="crop") -> (N,3,H,W) in [0,1]`
  —— 注意收的是**文件路径**，不是 ndarray（本模块负责把帧写盘）
- `model(images) -> dict`，键为 `pose_enc / depth / depth_conf / world_points /
  world_points_conf / images`（+ 传 query_points 时才有 `track/vis/conf`）
- `pose_encoding_to_extri_intri(pose_enc, (H,W)) -> (extri, intri)`，其中
  **extri 是 world→camera（OpenCV 约定，B×S×3×4）**，故 `c2w = inv(extri)`
- VGGT 的世界系即**第一帧相机坐标系** → `c2w[0] = I`，使 G-11 的"相机 up 轴"先验成立
- `demo_colmap.py --use_ba` 存在（BA/COLMAP 导出的复用点，G-13）
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.reconstruction import legacy_vggsfm_ba
from skill3d.reconstruction.legacy_vggsfm_ba import (
    UnsupportedConfigurationError,
    assert_official_ba_disabled,
)
from skill3d.schemas.reconstruction import ConfidenceMap, ReconstructionArtifact

# VGGT-1B 权重目录（本机已就位：weights 4.7GB + config.json）
# 商用版（VGGT-1B-Commercial）需另行申请；学术研究用当前权重（§15 许可表）
VGGT_CHECKPOINT: Optional[str] = "/home/cvailab/.cache/huggingface/geothinker/VGGT-1B"
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
    use_ba: bool = False,
    objects: Optional[Sequence[object]] = None,
    *,
    frame_set: Optional[object] = None,
    compute_quality: bool = True,
    quality_artifact_path: Optional[str] = None,
    scale_calibration_path: Optional[str] = None,
    scale_calibration_dir: Optional[str] = None,
    evaluation_datasets: Optional[Sequence[str]] = None,
    scale_confidence_level: Optional[float] = None,
) -> ReconstructionArtifact:
    """VGGT 单网络前向（+可选 BA + 尺度评估），输出组装为 ReconstructionArtifact。

    v4 尺度支路（§3 M3）：`assess_scale` 做多锚点鲁棒融合 + 冻结 conformal 校准 +
    逐题型授权；`scale_calibration_path` 指向**在线只读**的冻结校准器
    （`TODO_USER_INPUT`：ARKitScenes 标定数据到位前无校准器 → 一律 low）。
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

    # v5 HC35：官方 VGGSfM BA 已被 24 GiB OOM 证据否决 → 生产路径不得启用。
    # 唯一允许的 BA 候选 `vggt_sparse_ba` 走 reconstruction/sparse_ba（默认关闭，
    # 且必须先过 §10.1 L0→L1→L2）。这里的护栏保证"配置写错也不会偷偷跑官方 BA"。
    assert_official_ba_disabled(
        enable_official_vggsfm_ba=bool(use_ba),
        context="vggt_runner.run_vggt(use_ba=True)")

    from skill3d.coords import grid_transform_identity

    grid_tf: dict
    images = load_and_preprocess_images(frame_paths)
    src_hw = (int(np.asarray(frames[0]).shape[0]), int(np.asarray(frames[0]).shape[1]))
    grid_tf = grid_transform_identity(
        src_hw, (int(images.shape[-2]), int(images.shape[-1])))
    images = images.to(device)
    with torch.no_grad():
        # torch>=2.4 用 torch.amp.autocast('cuda', ...)；旧签名在 2.10 已弃用
        autocast = getattr(torch, "amp", None)
        if autocast is not None and hasattr(autocast, "autocast"):
            ctx = autocast.autocast("cuda", dtype=dtype, enabled=device == "cuda")
        else:  # pragma: no cover - 旧 torch
            ctx = torch.cuda.amp.autocast(dtype=dtype, enabled=device == "cuda")
        with ctx:
            preds = model(images)

    # 输出键名统一（真实键 → 本系统 §5.2 字段）
    preds = _normalize_preds(dict(preds), images)

    # v5 HC35/37：正式主线是 feed-forward，**不跑任何 BA**。
    # 无真 BA ⇒ `reprojection_status="not_available"`、G5=None（禁止代理值）。
    # `vggt_sparse_ba` 是唯一允许的 BA 候选，且必须先过 §10.1 L0→L1→L2；通过后由
    # `reconstruction/sparse_ba` 产出独立 receipt 与独立 artifact，**不在此处内联**。
    from skill3d.reconstruction.sparse_ba import BAOutcome, sparse_ba_enabled

    ba = BAOutcome.feed_forward(
        reason=("官方 VGGSfM BA 已按 HC35 退出生产（24 GiB OOM 否决）"
                if use_ba is False else "sparse BA 未启用"))
    if sparse_ba_enabled():  # pragma: no cover - 需 §10.1 PoC 通过后才可打开
        raise UnsupportedConfigurationError(
            "vggt_sparse_ba 尚未通过 §10.1 PoC，禁止启用（HC36 一次性止损纪律）。")
    ba_note = ba.summary()
    if ba.applied:  # pragma: no cover - 当前恒 False（保留真实接线位点）
        if preds.get("c2w_refined") is not None:
            preds["c2w"] = preds["c2w_refined"]
            preds["intrinsic"] = preds["intrinsic_refined"]
        if preds.get("point_map_refined") is not None:
            preds["point_map"] = preds["point_map_refined"]
        # point_conf 仍是 VGGT 逐点置信度（BA 只精化相机与稀疏点，不改深度置信度）

    # ---- 落盘并组装 artifact（字段为数组文件 ref，见 §5.2）----
    def _save(name: str, arr) -> str:
        p = out / f"{scene_name}_{name}.npy"
        arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.asarray(arr)
        np.save(p, arr)
        return str(p)

    c2w_ref = _save("c2w", preds["c2w"])            # 世界坐标 SE(3) 序列（extri 取逆）
    intr_ref = _save("intrinsics", preds["intrinsic"])
    depth_ref = _save("depth", preds["depth_map"])
    pmap_ref = _save("point_map", preds["point_map"])
    pconf_ref = _save("point_conf", preds["point_conf"])
    track_ref = _save("tracks", preds["track_list"]) if preds.get("track_list") is not None else None

    # G5 重投影残差（BA 产物）：有则落盘并回填 artifact，供 M4 的 G5 与置信度融合使用
    reproj_ref = None
    reproj = getattr(preds, "get", lambda *_: None)("reproj_errors")
    if reproj is not None:
        reproj_ref = _save("reproj_errors", np.asarray(reproj, dtype=np.float64))

    # metric 尺度锚定（v4：多锚点鲁棒融合 + 冻结 conformal 校准 + 逐题型授权）
    from skill3d.reconstruction.scale_assessment import assess_scale, apply_scale_assessment

    assessment = assess_scale(preds["point_map"], preds["c2w"], objects=objects,
                              scene_name=scene_name, calibration_path=scale_calibration_path,
                              calibration_dir=scale_calibration_dir,
                              evaluation_datasets=evaluation_datasets,
                              confidence_level=scale_confidence_level)
    est = assessment  # 统一走 ScaleAssessment（v4 口径）

    # 帧集身份（硬约束 21）：M1 冻结的 FrameSet 原样落 artifact，全链共用同一帧序
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

    confidence = ConfidenceMap(per_point_confidence=pconf_ref, coverage_count_per_frame="")

    artifact = ReconstructionArtifact(
        artifact_id=f"vggt-{scene_name}",
        artifact_version=_artifact_version(scene_name.encode() + str(len(frames)).encode()),
        scene_name=scene_name,
        recon_method=ba.recon_method if ba.applied else "vggt",
        frame_ids=frame_ids,
        source_frame_indices=source_frame_indices,
        timestamps=timestamps,
        frame_set_hash=fsh,
        c2w_list=c2w_ref,
        intrinsics=intr_ref,
        depth_maps=depth_ref,
        point_map=pmap_ref,
        point_conf=pconf_ref,
        track_list=track_ref,
        # §9：原图→深度网格 的仿射映射（BA 正方形 pad 会改变它；主线为纯缩放）
        grid_transform=grid_tf,
        # 尺度字段由 ScaleAssessment 统一写回（HC29/30/31/33：口径自洽 + 逐题授权）
        metric_scale=est.metric_scale,
        scale_known=est.scale_known,
        # quality 未算前是 None + status="not_computed"（硬约束 22：禁止用占位 NaN
        # 冒充实算值——旧实现在此处填 NaN 占位，日志里会被误读成"门禁失效"）
        quality_status="not_computed",
        quality=None,
        confidence=confidence,
        scale_confidence=est.confidence,
        # 最终值由下方 apply_scale_assessment 统一写（含 BA 说明后缀），
        # 这里只填必需字段，避免两处各拼一次导致重复
        scale_method=est.method,
        scale_source=est.source,
        reproj_errors=reproj_ref,
        # v5 HC37：G5 只在真 BA 跑通且两个标量都有限时才是 computed；
        # 正式 `vggt` feed-forward 主线（无真 BA）固定 not_available + G5=None。
        reprojection_status=(
            "computed" if (
                ba.applied
                and ba.g5_reproj_err_median is not None
                and ba.g5_reproj_err_p95 is not None) else "not_available"),
        g5_reproj_err_median=(ba.g5_reproj_err_median if ba.applied else None),
        g5_reproj_err_p95=(ba.g5_reproj_err_p95 if ba.applied else None),
    )
    # v4：把 ScaleAssessment 的 v4 字段（ci_rel/ci_abs_m/anchor evidence/conflict/
    # calibration_id/coverage/allowed_metric_tasks）写回 artifact（含 HC29 自洽断言）
    artifact = apply_scale_assessment(
        artifact, assessment,
        method_suffix=f"[{ba_note}]" if ba_note else "")

    # 方案 X（§4 M4）：P1 顺手把 G1–G11 算好写回 artifact，P2 加载即得实算值、零重算。
    if compute_quality:
        from skill3d.reconstruction_gate import quality_metrics as qm

        artifact = qm.compute_and_store_quality(
            artifact, frames=frames, depth_maps=preds["depth_map"],
            c2w_list=preds["c2w"], reproj_errors=preds.get("reproj_errors"),
            artifact_path=quality_artifact_path,
        )
    return artifact


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
    """VGGT 官方输出键 → 本系统 §5.2 字段（已对官方 repo 核验）。

    - `pose_enc` [B,S,9] --pose_encoding_to_extri_intri--> **extri(world→cam)** → 取逆得 c2w
      （VGGT 世界系 = 第一帧相机系，故 c2w[0] 应为单位阵）
    - `depth` [B,S,H,W,1] → depth_map
    - `world_points` [B,S,H,W,3] → point_map
    - `world_points_conf` [B,S,H,W] → point_conf
    - `track`/`vis`/`conf` 仅在有 query_points 时存在 → track_list
    """
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
    mvs: bool = False,
    use_ba: bool = False,
    frame_set: Optional[object] = None,
    compute_quality: bool = True,
    scale_calibration_path: Optional[str] = None,
    scale_calibration_dir: Optional[str] = None,
    evaluation_datasets: Optional[Sequence[str]] = None,
    scale_confidence_level: Optional[float] = None,
) -> ReconstructionArtifact:
    """重建入口 + 降级链（§4 M3 字段 9 / v5 HC35）：
    vggt → dust3r_mast3r → 单目深度 2D-only（后者不产生 3D artifact）。

    `method="colmap"` 为 §10 baseline 矩阵的独立一列（不进降级链：COLMAP 是
    对照重建方案，不是 VGGT 失败后的回退）。

    v5 HC35/36：`method="vggt"` 是**唯一**正式主线；`vggt_sparse_ba` 是尚未通过
    §10.1 PoC 的限定候选（启用即报 `UnsupportedConfigurationError`）。官方
    `VGGSfM tracker + PyCOLMAP BA`（历史名 `vggt_ba`）已被 24 GiB OOM 证据否决，
    只保留在 `reconstruction/legacy_vggsfm_ba/` 供失败复现。
    每次降级都保留上一级失败原因，最终异常里汇总（便于定位缺权重/缺依赖）。
    """
    reasons: list[str] = []

    if method == "colmap":
        from skill3d.reconstruction.colmap_baseline import reconstruct_colmap

        return reconstruct_colmap(frames, scene_name, output_dir, mvs=mvs,
                                  scale_calibration_path=scale_calibration_path,
                                  scale_confidence_level=scale_confidence_level)

    if method == "vggt_sparse_ba":
        # HC36：未过 §10.1 全部门槛前不得接线（一次性止损纪律）
        raise UnsupportedConfigurationError(
            "`vggt_sparse_ba` 尚未通过 §10.1 L0→L1→L2 门槛，禁止作为生产 route；"
            "正式主线请用 `vggt`（feed-forward）。")

    if method == "vggt":
        try:
            return run_vggt(frames, scene_name, output_dir,
                            use_ba=use_ba,
                            frame_set=frame_set, compute_quality=compute_quality,
                            scale_calibration_path=scale_calibration_path,
                            scale_calibration_dir=scale_calibration_dir,
                            evaluation_datasets=evaluation_datasets,
                            scale_confidence_level=scale_confidence_level)
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

    raise ValueError(
        f"未知重建方法: {method}（v5 受控枚举：vggt / vggt_sparse_ba / "
        "dust3r_mast3r / colmap；历史 `vggt_ba` 已按 HC35 作废）")
