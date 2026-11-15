"""SAM2 propagation, world backprojection and whole-object conflict audit.

Question parsing and the sole online M5 path live in question_binding.py.
G7/G9 remain diagnostics; geometry confidence never certifies object semantics.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.schemas.reconstruction import ObjectRecord
from skill3d.tools.category_match import (
    MASK_LABEL_RELATION_VERSION,
    canonical,
    mask_label_relation,
    normalize,
)


# 明确不相容的新题级绑定共享同一 mask 时硬拦；历史来源只读。
SEMANTIC_MASK_CONFLICT_VERSION = "semantic-mask-conflict-v3-question-driven"
SEMANTIC_MASK_CONFLICT_IOU = 0.95
SEMANTIC_MASK_CONFLICT_MIN_FRAMES = 2

TRACK_PREFIX_BASE_LIST = "trk"  # primitive default for historical read-only records

# G-19 档位（本机实测，2026-09-18）：可用权重为 SAM2.1 Hiera-Large 与 Hiera-Base+，
# **Tiny 未下载**。实测加载峰值显存：Large 0.98 GiB / Base+ 0.44 GiB（权重部分）。
# 默认取 Large（质量优先，仍可与 VGGT 同卡共存）；低显存场景改 Base+：
#   SKILL3D_SAM2_CHECKPOINT=<base+ .pt> SKILL3D_SAM2_CONFIG=configs/sam2/sam2_hiera_b+.yaml
# 注意 `SAM2_CONFIG` 必须用官方文档的**相对形式**（hydra 以 sam2/configs 为搜索根；
# 传绝对路径会 MissingConfigException）。
SAM2_CHECKPOINT: Optional[str] = (
    "/home/cvailab/.cache/huggingface/hub/models--facebook--sam2.1-hiera-large/"
    "snapshots/227a114a2f535cd147f82442e7d2038cdd2e5d68/sam2.1_hiera_large.pt")
SAM2_CONFIG: Optional[str] = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_LOCAL_CACHE: str = "data/checkpoints"                      # 预拉取目录（§7.1 G-19）
_ENV_CHECKPOINT = "SKILL3D_SAM2_CHECKPOINT"
_ENV_CONFIG = "SKILL3D_SAM2_CONFIG"
_ENV_ALLOW_DOWNLOAD = "SKILL3D_ALLOW_HF_DOWNLOAD"

# G7 刚性残差阈值（px）：观测光流与自运动预测光流之差超此值记为动态（TODO_CALIBRATE）
TH_RIGIDITY_RESIDUAL_PX: float = 3.0
# G7 自适应阈值的稳健系数：thr = max(下界, median + k×1.4826×MAD)（TODO_CALIBRATE）
RIGIDITY_MAD_K: float = 3.0
# 动态占比上限：超此值记重建污染（§10 G7 触发动作，TODO_CALIBRATE）
TH_DYNAMIC_RATIO: float = 0.3


def resolve_sam2_config(checkpoint: Optional[str] = None,
                        config: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """解析 checkpoint/config：显式参数 > 环境变量 > 模块默认。"""
    ckpt = checkpoint or os.environ.get(_ENV_CHECKPOINT) or SAM2_CHECKPOINT
    cfg = config or os.environ.get(_ENV_CONFIG) or SAM2_CONFIG
    return ckpt, cfg


def ensure_checkpoint(checkpoint: Optional[str] = None,
                      cache_dir: str = SAM2_LOCAL_CACHE) -> Optional[str]:
    """把 checkpoint 解析为本地路径（HF id 时预拉取到 cache，§7.1 G-19）。

    已是本地文件则原样返回；HF id 走 `huggingface_hub.hf_hub_download`。
    自动下载需显式开启 `SKILL3D_ALLOW_HF_DOWNLOAD=1`：离线/受限网络环境下
    HF 连接会长时间悬挂，默认不隐式发起网络请求（返回 None → 调用方报降级）。
    """
    ckpt, _ = resolve_sam2_config(checkpoint=checkpoint)
    if ckpt is None:
        return None
    p = Path(ckpt)
    if p.is_file():
        return str(p)
    target = Path(cache_dir) / str(ckpt).replace("/", "__")
    if target.is_file():
        return str(target)
    if not os.environ.get(_ENV_ALLOW_DOWNLOAD):
        return None
    try:
        from huggingface_hub import hf_hub_download  # lazy import
    except Exception:  # noqa: BLE001
        return None
    try:
        got = hf_hub_download(repo_id=str(ckpt), filename="model.safetensors",
                              local_dir=str(target.parent / target.name))
        return str(got)
    except Exception:  # noqa: BLE001 - 无网络/无权限 → 交由调用方报降级
        return None


def build_video_predictor(checkpoint: Optional[str] = None,
                          config: Optional[str] = None):
    """构建 SAM2 视频预测器（lazy import）。

    TODO: `build_sam2_video_predictor` 具体签名以官方 repo 为准（§4 M5 字段 5）。
    配置缺失 / sam2 未安装 / 权重不可得，统一抛 RuntimeError 并保留根因，
    由 M5 降级分支接住（§4 M5 字段 9）。
    """
    ckpt, cfg = resolve_sam2_config(checkpoint, config)
    if ckpt is None or cfg is None:
        raise RuntimeError("SAM2 checkpoint/config 未配置（TODO_USER_INPUT：档位见 §7.1 G-19）")
    try:
        from sam2.build_sam import build_sam2_video_predictor  # lazy import
    except ImportError as exc:
        raise RuntimeError(f"sam2 未安装（pip install sam2==1.1.0）：{exc}") from exc
    local = ensure_checkpoint(ckpt)
    if local is None:
        raise RuntimeError(
            f"SAM2 权重不可得（{ckpt}）：预拉取失败或未配置本地路径；"
            f"可设 {_ENV_CHECKPOINT}=<本地 .pt 路径>，或先下载到 {SAM2_LOCAL_CACHE}"
            f"（自动下载需 {_ENV_ALLOW_DOWNLOAD}=1）"
        )
    return build_sam2_video_predictor(cfg, local)


def build_image_predictor(model=None, checkpoint: Optional[str] = None,
                          config: Optional[str] = None):
    """Build SAM2's still-image predictor over the same frozen model weights."""
    if model is not None and all(hasattr(model, name) for name in ("set_image", "predict")):
        return model
    if model is None:
        model = build_video_predictor(checkpoint, config)
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise RuntimeError(f"SAM2 image predictor unavailable: {exc}") from exc
    return SAM2ImagePredictor(model)


def segment_frame_boxes(frame: np.ndarray, boxes: Sequence[Sequence[float]],
                        predictor) -> list[np.ndarray]:
    """Segment all NMS-kept boxes in one frame without temporal propagation."""
    prompts = [np.asarray(box, dtype=np.float32) for box in boxes]
    if not prompts:
        return []
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("SAM2 image segmentation requires a readable RGB frame")
    try:
        import torch

        inference = torch.inference_mode()
    except Exception:  # pragma: no cover - fake/CPU-only test environment
        inference = contextlib.nullcontext()
    results = []
    with inference, _autocast_ctx():
        predictor.set_image(image)
        for box in prompts:
            masks, _scores, _logits = predictor.predict(
                box=box, multimask_output=False)
            mask = np.asarray(masks)
            if mask.ndim == 3:
                mask = mask[0]
            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"SAM2 image mask shape {mask.shape} != frame {image.shape[:2]}")
            results.append(mask.astype(bool, copy=False))
    return results


def track_objects(
    frames: Sequence[np.ndarray],
    box_prompts: Optional[Sequence[Sequence[float]]] = None,
    predictor=None,
    *,
    prompts: Optional[Sequence[tuple[int, int, Sequence[float]]]] = None,
    class_hints: Optional[Sequence[str]] = None,
    handle=None,
    question: str = "",
    checkpoint: Optional[str] = None,
    config: Optional[str] = None,
) -> list[dict[int, np.ndarray]]:
    """SAM2 视频 mask 传播（§4 M5 伪代码）。

    `box_prompts`：每个候选一个真实 [x0,y0,x1,y1] 检测框，无框则不可用。

    返回 per-object `{frame_idx: mask(H,W) bool}`。
    """
    boxes = list(box_prompts) if box_prompts is not None else []
    if not boxes:
        raise RuntimeError("SAM2 requires actual detector box prompts")

    if predictor is None:
        predictor = build_video_predictor(checkpoint, config)
    # 已核验的官方 API：init_state 接受 **MP4 或 JPEG 文件夹路径**（不接受 ndarray），
    # 故把帧写到临时 JPEG 目录；帧名需为连续数字（load_video_frames 的约定）。
    frames_dir = _write_jpeg_sequence(frames)
    try:
        # 官方 demo 会在调用前**全局进入 bf16 autocast**；不进入会出现
        # "mat1 and mat2 must have the same dtype, but got BFloat16 and Float"。
        with _autocast_ctx():
            state = predictor.init_state(
                frames_dir, offload_video_to_cpu=True, offload_state_to_cpu=True)
        # 提示形式：`prompts=[(frame_idx, obj_id, box)]`（多帧各自给框，SAM2 从该帧起传播）
        # 或退化形式 `boxes=[box,...]`（统一在第 0 帧给框）
        prompt_list = ([(int(fi), int(oid), box) for fi, oid, box in prompts]
                       if prompts else [(0, i, b) for i, b in enumerate(boxes)])
        n_objects = max([oid for _fi, oid, _b in prompt_list], default=-1) + 1
        for fi, oid, box in prompt_list:
            # 已核验签名：add_new_points_or_box(state, frame_idx, obj_id, box=...)
            # dtype 必须与模型参数一致（SAM2 在 bf16 下运行；传 float32 会
            # "mat1 and mat2 must have the same dtype"）
            with _autocast_ctx():
                predictor.add_new_points_or_box(state, frame_idx=fi, obj_id=oid,
                                                box=_prompt_box(box, predictor))
        # propagate_in_video 产出 (frame_idx, obj_ids, video_res_masks)
        per_object: list[dict[int, np.ndarray]] = [dict() for _ in range(max(n_objects, 1))]
        for frame_idx, obj_ids, mask_logits in _propagate_bidirectional(predictor, state):
            for oid, logits in zip(obj_ids, mask_logits):
                mask = _to_bool_mask(logits)
                if mask.ndim == 3:        # (1,H,W) → (H,W)
                    mask = mask[0]
                per_object[int(oid)][int(frame_idx)] = mask
        return per_object
    finally:
        import shutil
        shutil.rmtree(frames_dir, ignore_errors=True)


def _propagate_bidirectional(predictor, state):
    """SAM2 **双向**传播（§4 M5 字段 7 / D-1）：先正向，再逆向补齐提示帧之前的帧。

    只正向传播时，在第 n-1 帧给框的对象拿不到之前的 mask（单帧漏检 = 对象不存在），
    逆向 pass 失败使该候选不可用，不以正向结果冒充双向成功。
    """
    # C-4：mask decoder 在 bf16 下计算，传播**也必须**在 autocast 内，否则
    # "mat1 and mat2 must have the same dtype, but got BFloat16 and Float"。
    with _autocast_ctx():
        for item in predictor.propagate_in_video(state):
            yield item
        # A failed reverse pass invalidates this candidate; forward-only is not success.
        for item in predictor.propagate_in_video(state, reverse=True):
            yield item


def _autocast_ctx():
    """官方 SAM2 demo 的全局 bf16 autocast 上下文（CUDA 可用时）。

    SAM2 的 mask decoder 内部按 autocast 计算，而 prompt/权重为 fp32；
    不在外层进入 autocast 会触发 dtype 不一致（见 sam2 README 的推理示例）。
    """
    try:
        import torch

        if torch.cuda.is_available():
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                return torch.amp.autocast("cuda", dtype=torch.bfloat16)
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)  # pragma: no cover
        return contextlib.nullcontext()
    except Exception:  # noqa: BLE001
        return contextlib.nullcontext()


def _prompt_box(box: Sequence[float], predictor):
    """构造与模型 dtype 一致的 box 提示张量（SAM2 常以 bf16 运行）。"""
    dtype = None
    try:
        params = list(predictor.parameters())      # type: ignore[attr-defined]
        if params:
            dtype = params[0].dtype
        elif getattr(predictor, "device", None) is None:
            dtype = None
    except Exception:  # noqa: BLE001 - fake predictor 无 parameters()
        dtype = None
    try:
        import torch

        return torch.as_tensor(np.asarray(box, dtype=np.float32),
                               dtype=dtype or torch.float32)
    except Exception:  # noqa: BLE001 - 无 torch 时退回 ndarray
        return np.asarray(box, dtype=np.float32)


def _resize_mask_to(mask: np.ndarray, shape: tuple[int, int],
                    transform: Optional[dict] = None) -> np.ndarray:
    """原始 SAM2 掩码 → VGGT crop 网格；保留非 VGGT 合成测试网格。"""
    from skill3d.coords import mask_to_grid

    del transform          # 不恢复已废止的 BA grid_transform 字段。
    if int(shape[1]) == 518:
        from skill3d.reconstruction.image_grid import mask_on_vggt_grid

        return mask_on_vggt_grid(mask, shape)
    return mask_to_grid(mask, None, shape=(int(shape[0]), int(shape[1])))


def _to_bool_mask(logits) -> np.ndarray:
    """SAM2 输出（可能是 CUDA tensor）→ 布尔 mask ndarray。

    已核验：`propagate_in_video` 的 `video_res_masks` 是 **GPU 上的 torch.Tensor**，
    直接 `np.asarray` 会抛 "can't convert cuda:0 device type tensor to numpy"。
    """
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu().numpy()
    return np.asarray(logits) > 0.0


def _write_jpeg_sequence(frames: Sequence[np.ndarray],
                         quality: int = 92) -> str:
    """把帧写成连续编号的 JPEG 目录（SAM2 `init_state` 的唯一非-MP4 输入形式）。

    返回目录路径。放在临时目录下（M10 沙箱纪律：不往产物目录写中间态）。
    """
    import tempfile

    import cv2

    d = Path(tempfile.mkdtemp(prefix="skill3d_sam2_"))
    for i, fr in enumerate(frames):
        img = np.asarray(fr)
        cv2.imwrite(str(d / f"{i:05d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return str(d)


# ------------------------------------------------------------------ G7 动态 mask ----

def dynamic_mask_from_rigidity(
    frames: Sequence[np.ndarray],
    depth_maps: np.ndarray,
    c2w_list: np.ndarray,
    intrinsics: np.ndarray,
    *,
    th_residual_px: float = TH_RIGIDITY_RESIDUAL_PX,
    mad_k: float = RIGIDITY_MAD_K,
) -> tuple[np.ndarray, list[str]]:
    """G7 动态 mask：由**自运动刚性残差**判定动态像素（确定性，无额外模型）。

    对相邻帧 (t, t+1)：用 depth+位姿把帧 t 的像素投影到帧 t+1 得"自运动预测光流"，
    与 Farneback 观测光流比较；残差超阈的像素记为动态（相机静止/纯自运动场景该
    残差为 0）。

    **阈值是自适应 + 绝对下界**：光流估计本身有噪声地板（实测在平滑纹理上
    ~3–5px），固定阈值会大面积误报。故取
    `thr = max(th_residual_px, median(res) + mad_k × 1.4826 × MAD(res))`，
    以场景自身的残差分布标定噪声地板（对"多数像素静止"的常见情形成立）。

    TODO_CALIBRATE：`th_residual_px` 下界与 `mad_k`；真实手持视频的噪声地板更高，
    需在自有数据上重新标定（§15.2）。
    """
    import cv2

    depth = np.asarray(depth_maps, dtype=np.float64)
    c2w = np.asarray(c2w_list, dtype=np.float64)
    k_all = np.asarray(intrinsics, dtype=np.float64)
    n = min(len(frames), len(depth), len(c2w))
    notes: list[str] = []
    if n < 2:
        return np.zeros((0, 0, 0), dtype=bool), ["帧数不足 2，无法估计动态 mask"]

    h, w = depth.shape[1], depth.shape[2]
    dyn = np.zeros((n, h, w), dtype=bool)
    thr_used: list[float] = []
    # 光流必须与深度在同一像素网格：VGGT 会改动帧分辨率（如 640x480 竖屏视频 →
    # 518x392 预处理网格），直接在原始帧上算光流会导致坐标越界/错位。
    gray = [cv2.resize(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY), (w, h),
                       interpolation=cv2.INTER_AREA)
            for f in frames[:n]]
    for t in range(n - 1):
        k = k_all[t] if k_all.ndim == 3 else k_all
        z = depth[t]
        try:
            obs = cv2.calcOpticalFlowFarneback(
                gray[t], gray[t + 1], None, 0.5, 3, 21, 3, 5, 1.2, 0)
        except Exception as exc:  # noqa: BLE001 - 分辨率/类型异常 → 跳过该帧对
            notes.append(f"帧对 {t}->{t+1} 光流失败（{type(exc).__name__}）")
            continue
        # 自运动预测：X_{t+1} = T_{t+1<-t} X_t，T = inv(c2w_{t+1}) @ c2w_t
        t_rel = np.linalg.inv(c2w[t + 1]) @ c2w[t]
        vv, uu = np.nonzero(np.isfinite(z) & (z > 1e-6))
        if vv.size == 0:
            continue
        pix = np.stack([uu, vv, np.ones_like(uu)], axis=0).astype(np.float64)
        k_inv = np.linalg.inv(k)
        xyz = (k_inv @ pix) * z[vv, uu][None, :]
        moved = t_rel[:3, :3] @ xyz + t_rel[:3, 3:4]
        depth_next = moved[2]
        valid = depth_next > 1e-6
        if not valid.any():
            continue
        proj = k @ (moved[:, valid] / depth_next[valid])
        u2 = np.clip(np.round(proj[0]).astype(int), 0, w - 1)
        v2 = np.clip(np.round(proj[1]).astype(int), 0, h - 1)
        pred_u = proj[0] - uu[valid]
        pred_v = proj[1] - vv[valid]
        res = np.hypot(obs[v2, u2, 0] - pred_u, obs[v2, u2, 1] - pred_v)
        med = float(np.median(res))
        mad = float(np.median(np.abs(res - med)))
        thr = max(float(th_residual_px), med + float(mad_k) * 1.4826 * mad)
        thr_used.append(thr)
        dyn[t][vv[valid], uu[valid]] = res > thr
    ratio = float(dyn[: n - 1].mean()) if n > 1 else 0.0
    if thr_used:
        notes.append(f"G7 刚性残差动态占比={ratio:.4f}（自适应阈值 "
                     f"min={min(thr_used):.2f}px max={max(thr_used):.2f}px，下界 "
                     f"{th_residual_px}px/U+2B k={mad_k}，TODO_CALIBRATE）")
    if ratio > TH_DYNAMIC_RATIO:
        notes.append(f"G7 动态占比超阈 {TH_DYNAMIC_RATIO} → 记重建污染（§10 G7）")
    return dyn, notes


# ------------------------------------------------------------------ 绑定世界点云 ----

def _as_point_conf(point_conf, depth_maps: np.ndarray) -> Optional[np.ndarray]:
    """规整逐点置信度到与 `depth_maps` 同形的 (N,H,W)；不可用 → `None`。

    **不伪造**：形状对不上（旧产物/别的网格上的 conf）一律返回 None，
    由调用方写空 ref（§12.2：conf 不可用时不得当作 1.0 参与加权）。
    """
    if point_conf is None:
        return None
    arr = np.asarray(point_conf, dtype=np.float64)
    d = np.asarray(depth_maps)
    if d.ndim != 3:
        return None
    if arr.ndim == 2 and arr.shape == d.shape[1:]:
        arr = np.broadcast_to(arr, (d.shape[0], arr.shape[0], arr.shape[1]))
    if arr.ndim != 3 or arr.shape != d.shape:
        return None
    return arr


def bind_masks_to_world(
    masks_per_object: Sequence[dict[int, np.ndarray]],
    depth_maps: np.ndarray,
    c2w_list: np.ndarray,
    intrinsics: np.ndarray,
    class_hints: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    det_conf_sources: Optional[Sequence[str]] = None,
    max_points: int = 20000,
    out_dir: Optional[str | Path] = None,
    scene_name: str = "scene",
    grid_transform: Optional[dict] = None,
    point_conf: Optional[np.ndarray] = None,
    track_prefix: str = TRACK_PREFIX_BASE_LIST,
    grounding_status: str = "base_list",
    prompt_provenance: Optional[Sequence[dict]] = None,
) -> list[ObjectRecord]:
    """mask × 深度反投影 → 世界坐标点云绑定（numpy 真实实现，§4 M5）。

    - masks_per_object: per-object {frame_idx: mask(H,W)}
    - depth_maps: (N,H,W) 逐帧深度（与相机坐标系对齐，正值）
    - c2w_list: (N,4,4) 世界坐标 SE(3)
    - intrinsics: (N,3,3) 或 (3,3) 相机内参
    - out_dir: 给出时把逐帧 mask 与对象点云落盘并填 `ObjectRecord` 的 ref 字段
      （M3 尺度锚定与 M4 覆盖率都需要从 ref 回读对象点云）
    - point_conf: (N,H,W) 或 (H,W) **逐点** VGGT 置信度（与 `depth_maps` 同网格）。
      给出时按与点云**同一套有效掩码/同一套下采样下标**取点，落盘 `pointconf_world`
      并与 `pointcloud_world` 逐点 1:1 对齐；**不给则写空 ref**（不伪造置信度）。
    - track_prefix: 本题生成身份与候选序号的命名空间；历史原语默认前缀只供审计。
    - grounding_status: 新 M5 为 `question_driven`；历史来源可读取，不授予在线准入。
    - confidences / det_conf_sources: 与候选严格对齐的检测置信度及来源。没有检测器
      分数时置信度固定为 0；传播覆盖率单独写入 `mask_support_ratio`。

    `centroid_ref` 留空串：质心已内联在 `centroid_world`（同一批数字不写第二份文件，
    避免两处事实源漂移），**不写假路径**。
    """
    depth_maps = np.asarray(depth_maps, dtype=np.float64)
    c2w_list = np.asarray(c2w_list, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim == 2:
        intrinsics = np.broadcast_to(intrinsics, (len(c2w_list), 3, 3))
    pconf = _as_point_conf(point_conf, depth_maps)

    instances: list[ObjectRecord] = []
    for prop_idx, masks in enumerate(masks_per_object):
        # 有效支撑帧（掩码非空）→ 供"逐帧可见性"类题型程序化作答
        saved_masks = _stack_masks(masks, len(depth_maps), depth_maps.shape[1:])
        masks = {i: mask.astype(bool) for i, mask in enumerate(saved_masks)}
        vis_frames = np.flatnonzero(saved_masks.any(axis=(1, 2))).tolist()
        support_ratio = min(1.0, len(vis_frames) / max(1, len(depth_maps)))
        hint = class_hints[prop_idx] if class_hints and prop_idx < len(class_hints) else "unknown"
        conf = (float(confidences[prop_idx])
                if confidences is not None and prop_idx < len(confidences) else 0.0)
        conf_source = (str(det_conf_sources[prop_idx])
                       if det_conf_sources is not None
                       and prop_idx < len(det_conf_sources) else "unknown")
        source_label = str(getattr(hint, "source_label", hint))
        provenance = {
            "source": grounding_status, "det_conf_source": conf_source,
            "source_label": source_label, "prompt_category": str(hint),
        }
        if prompt_provenance is not None and prop_idx < len(prompt_provenance):
            provenance.update(prompt_provenance[prop_idx])
        pts_world_all = []
        conf_all = []
        for frame_idx, mask in sorted(masks.items()):
            if not 0 <= frame_idx < len(depth_maps):
                continue
            depth = depth_maps[frame_idx]
            K = intrinsics[min(frame_idx, len(intrinsics) - 1)]
            c2w = c2w_list[min(frame_idx, len(c2w_list) - 1)]
            # 原图掩码必须经过 VGGT 同样的缩放、中心裁剪及批内填充。
            m = _resize_mask_to(np.asarray(mask, dtype=bool), depth.shape, grid_transform)
            vs, us = np.nonzero(m)
            z = depth[vs, us]
            valid = np.isfinite(z) & (z > 0)
            vs, us, z = vs[valid], us[valid], z[valid]
            if z.size == 0:
                continue
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            # 相机坐标系反投影
            x = (us - cx) * z / fx
            y = (vs - cy) * z / fy
            pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)  # (4,M)
            pts_world = (c2w @ pts_cam)[:3].T                        # (M,3)
            pts_world_all.append(pts_world)
            if pconf is not None:
                conf_all.append(pconf[frame_idx][vs, us])            # 与 pts 同序同长

        iid = f"obj_{prop_idx}"
        # v6 §5.6：跨帧 track 身份 = 产生该对象的 SAM2 传播序号（确定性）
        track_id = f"{track_prefix}{prop_idx}"
        mask_ref = ""
        if out_dir is not None:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            mask_path = d / f"{scene_name}_{iid}_mask.npy"
            np.save(mask_path, saved_masks)
            mask_ref = str(mask_path)
        if not pts_world_all:
            # 分割失败 → 该对象标记 unverified（§4 M5 字段 9）
            instances.append(ObjectRecord(
                obj_id=iid,
                category_name=hint if class_hints else "unverified",
                source_label=source_label, binding_provenance=[provenance],
                track_id=track_id,
                grounding_status=grounding_status,
                mask_per_frame=mask_ref,
                pointcloud_world="",
                pointconf_world="",          # 无点云 → 无逐点 conf（不伪造）
                centroid_ref="",
                centroid_world=[0.0, 0.0, 0.0],
                bbox=[0.0] * 6,
                det_conf=conf,
                det_conf_source=conf_source,
                mask_support_ratio=support_ratio,
                visible_frames=vis_frames,
            ))
            continue

        pts = np.concatenate(pts_world_all, axis=0)
        conf_pts = np.concatenate(conf_all, axis=0) if conf_all else None
        if len(pts) > max_points:
            sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
            pts = pts[sel]
            if conf_pts is not None:
                conf_pts = conf_pts[sel]     # 同一套下标 → 下采样后仍逐点对齐
        centroid = pts.mean(axis=0)
        pmin, pmax = pts.min(axis=0), pts.max(axis=0)
        cloud_ref, conf_ref = "", ""
        if out_dir is not None:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            cloud_path = d / f"{scene_name}_{iid}_points.npy"
            np.save(cloud_path, pts)
            cloud_ref = str(cloud_path)
            if conf_pts is not None:
                conf_path = d / f"{scene_name}_{iid}_pointconf.npy"
                np.save(conf_path, conf_pts)     # 与 pts 同长（逐点对齐）
                conf_ref = str(conf_path)

        instances.append(ObjectRecord(
            obj_id=iid,
            category_name=hint,
            source_label=source_label, binding_provenance=[provenance],
            track_id=track_id,
            grounding_status=grounding_status,
            mask_per_frame=mask_ref,
            pointcloud_world=cloud_ref,
            pointconf_world=conf_ref,
            centroid_ref="",
            centroid_world=centroid.tolist(),
            bbox=(list(pmin) + list(pmax)),
            det_conf=conf,
            det_conf_source=conf_source,
            mask_support_ratio=support_ratio,
            visible_frames=vis_frames,
        ))
    return instances


def _stack_masks(masks: dict[int, np.ndarray], n_frames: int,
                 shape: Sequence[int]) -> np.ndarray:
    """把 per-frame mask 堆成 (N,H,W) uint8（缺帧置 0），便于落盘与 G9 统计。

    **必须按 `_resize_mask_to` 映射到深度网格后再堆**（2026-09-22 实测缺陷）：
    SAM2 掩码在**原始视频分辨率**，本函数的 `shape` 来自 `depth_maps.shape[1:]`
    （VGGT 深度网格 518×392）。旧实现写的是 `if np.asarray(m).shape == out.shape[1:]`
    —— 尺寸不等就**静默跳过**，于是每个对象的 `*_mask.npy` 落盘成**整片 0**
    （实测 7 个 scene、全部 54 个对象的 mask 数组 sum 均为 0），而清单里的
    `visible_frames` 由**未缩放的** mask 算出、依然非空 →
    落盘产物与清单自相矛盾，审计与回读（G9/掩码类复算）全部失真。
    映射遵循 VGGT crop 预处理；不使用已废止的 BA square-pad 仿射。
    """
    out = np.zeros((n_frames, int(shape[0]), int(shape[1])), dtype=np.uint8)
    for idx, m in masks.items():
        if not (0 <= idx < n_frames):
            continue
        mm = _resize_mask_to(np.asarray(m, dtype=bool), out.shape[1:])
        if mm.shape == out.shape[1:]:
            out[idx] = np.asarray(mm, dtype=np.uint8)
    return out


def audit_semantic_mask_conflicts(
    bindings: Sequence[tuple[str, Sequence[ObjectRecord]]],
    *,
    scene_identity: str,
    iou_threshold: float = SEMANTIC_MASK_CONFLICT_IOU,
    contract_version: str = SEMANTIC_MASK_CONFLICT_VERSION,
) -> tuple[dict[str, list[ObjectRecord]], list[dict]]:
    """审计全部可读 mask；明确不相容且至少一端新题级绑定时硬拦。"""
    threshold = float(iou_threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"iou_threshold must be within [0,1], got {iou_threshold!r}")

    cleaned: dict[str, list[ObjectRecord]] = {}
    candidates: dict[tuple, dict] = {}
    audits: list[dict] = []

    def publish(body):
        audit_sha = hashlib.sha256(json.dumps(
            body, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        audit = {**body, "audit_sha256": audit_sha}
        audits.append(audit)
        return audit

    contract = {
        "contract_version": str(contract_version),
        "relation_rule_version": MASK_LABEL_RELATION_VERSION,
        "scene_identity": str(scene_identity),
        "iou_threshold": threshold,
        "min_common_nonempty_frames": SEMANTIC_MASK_CONFLICT_MIN_FRAMES,
    }
    for qa_id, objects in bindings:
        rows = []
        for index, obj in enumerate(objects):
            flags = [dict(flag) for flag in (obj.binding_flags or [])
                     if not (
                         flag.get("code") == "semantic_label_mask_conflict"
                         and flag.get("contract_version") in (
                             "semantic-mask-conflict-v1", "semantic-mask-conflict-v2",
                             SEMANTIC_MASK_CONFLICT_VERSION,
                             contract_version))]
            current = obj.model_copy(update={"binding_flags": flags})
            rows.append(current)
            category = canonical(current.category_name)
            mask_path = Path(str(current.mask_per_frame or ""))
            endpoint = {
                "qa_id": str(qa_id), "obj_id": current.obj_id,
                "category_name": current.category_name,
                "source_label": current.source_label or current.category_name,
                "canonical_category": category, "source": current.grounding_status,
            }
            try:
                from skill3d.tools.scene_handle import file_sha256
                mask_sha = file_sha256(mask_path)
                raw = normalize(endpoint["source_label"])
                compound = raw if raw in ("desk counter", "table dresser") else ""
                key = (category, compound, current.grounding_status, mask_sha)
                mask = (candidates[key]["mask"] if key in candidates else
                        np.load(mask_path, allow_pickle=False, mmap_mode="r"))
                if mask.ndim != 3:
                    raise ValueError(f"expected 3D mask, got {mask.shape}")
            except (OSError, ValueError) as exc:
                publish({
                    **contract, "code": "semantic_mask_audit_skipped", "status": "skipped",
                    "skip_reason": "unreadable_or_invalid_mask",
                    "error": f"{type(exc).__name__}: {exc}",
                    "endpoints": [endpoint], "qa_ids": [str(qa_id)],
                    "object_ids": [current.obj_id],
                })
                continue
            endpoint["mask_sha256"] = mask_sha
            entry = candidates.setdefault(key, {
                "object": current, "category": category, "mask": mask,
                "endpoint": endpoint, "aliases": [],
            })
            entry["aliases"].append({"qa_id": str(qa_id), "index": index,
                                     "obj_id": current.obj_id,
                                     "source_label": endpoint["source_label"],
                                     "category_name": current.category_name})
        cleaned[str(qa_id)] = rows

    representatives = list(candidates.values())
    for left_index, left in enumerate(representatives):
        for right in representatives[left_index + 1:]:
            relation = mask_label_relation(left["endpoint"]["source_label"],
                                           right["endpoint"]["source_label"])
            if any(entry["object"].grounding_status == "question_driven"
                   for entry in (left, right)):
                requested_relation = mask_label_relation(
                    left["object"].category_name, right["object"].category_name)
                if requested_relation == "incompatible":
                    relation = "incompatible"
            if relation == "equivalent":
                continue
            endpoints = sorted([
                {**entry["endpoint"],
                 "aliases": sorted(entry["aliases"], key=lambda a: (a["qa_id"], a["obj_id"]))}
                for entry in (left, right)
            ], key=lambda e: (e["qa_id"], e["obj_id"], e["canonical_category"], e["mask_sha256"]))
            identity = {
                **contract, "relation": relation, "endpoints": endpoints,
                "qa_ids": sorted({a["qa_id"] for e in endpoints for a in e["aliases"]}),
                "object_ids": sorted({a["obj_id"] for e in endpoints for a in e["aliases"]}),
                "categories": [e["category_name"] for e in endpoints],
            }
            if left["mask"].shape[1:] != right["mask"].shape[1:]:
                publish({**identity, "code": "semantic_mask_audit_skipped",
                         "status": "skipped", "skip_reason": "mask_shape_mismatch"})
                continue
            n_frames = min(len(left["mask"]), len(right["mask"]))
            common = [
                frame for frame in range(n_frames)
                if left["mask"][frame].shape == right["mask"][frame].shape
                and left["mask"][frame].any() and right["mask"][frame].any()
            ]
            if len(common) < SEMANTIC_MASK_CONFLICT_MIN_FRAMES:
                publish({**identity, "code": "semantic_mask_audit_skipped",
                         "status": "skipped", "skip_reason": "insufficient_common_nonempty_frames",
                         "common_nonempty_frames": common})
                continue
            intersections = sum(
                int(np.logical_and(left["mask"][frame], right["mask"][frame]).sum())
                for frame in common)
            unions = sum(
                int(np.logical_or(left["mask"][frame], right["mask"][frame]).sum())
                for frame in common)
            stacked_iou = float(intersections / unions) if unions else 0.0
            if stacked_iou < threshold:
                continue

            hard_block = relation == "incompatible" and any(
                e["source"] in ("question_targeted_fill", "question_driven")
                for e in endpoints)
            audit = publish({
                **identity, "hard_block": hard_block, "status": "high_overlap",
                "code": ("semantic_label_mask_conflict" if hard_block
                         else "semantic_mask_overlap_audit"),
                "common_nonempty_frames": common,
                "stacked_iou": stacked_iou,
            })
            if not hard_block:
                continue
            for own, other in ((left, right), (right, left)):
                for alias in own["aliases"]:
                    flag = {
                        **audit, "object_qa_id": alias["qa_id"], "object_id": alias["obj_id"],
                        "counterpart_qa_id": other["endpoint"]["qa_id"],
                        "counterpart_object_id": other["object"].obj_id,
                        "counterpart_category": other["object"].category_name,
                    }
                    row = cleaned[alias["qa_id"]][alias["index"]]
                    flags = list(row.binding_flags or [])
                    if not any(item.get("audit_sha256") == audit["audit_sha256"] for item in flags):
                        cleaned[alias["qa_id"]][alias["index"]] = row.model_copy(
                            update={"binding_flags": flags + [flag]})
    audits.sort(key=lambda item: item["audit_sha256"])
    return cleaned, audits


def refresh_grounding_coverage(grounding, objects):
    """Refresh audited question coverage without model requests."""
    from .question_binding import refresh_coverage
    return refresh_coverage(grounding, objects)


def inventory_scope_statistics(objects, *, n_frames, scope, n_propagations=None,
                               n_dedup_groups=None, n_dedup_absorbed=None, consensus=None):
    """Explicit denominators; multi-pass final fragmentation has no proven denominator."""
    objects = list(objects)
    n_suspect = sum(bool(o.duplicate_suspect) for o in objects)
    return {
        "scope": scope, "object_ids": sorted(o.obj_id for o in objects),
        "n_objects": len(objects),
        "n_tracks": len({o.track_id for o in objects if o.track_id}),
        "n_propagations": n_propagations, "n_dedup_groups": n_dedup_groups,
        "n_dedup_absorbed": n_dedup_absorbed, "n_suspect": n_suspect,
        "duplicate_suspect_ratio": n_suspect / len(objects) if objects else None,
        "track_stable_ratio": track_stable_ratio(objects, n_frames),
        "visible_support": {o.obj_id: {
            "visible_frames": list(o.visible_frames or []),
            "mask_support_ratio": o.mask_support_ratio} for o in objects},
        "track_fragmentation_ratio": (consensus or {}).get("track_fragmentation_ratio"),
        "fragmentation_reason": ("question_candidate_denominator" if consensus is not None
                                 else "no_unified_propagation_denominator"),
        "track_consensus": consensus,
    }


def track_stable_ratio(objects: Sequence[ObjectRecord],
                       n_frames: int) -> Optional[float]:
    """可见帧率均值（**诊断量**，不再是 `track_consensus` 的判据；[TODO_CALIBRATE]）。

    定义：每条 track 的存活率 = 该对象在帧集里可见的帧数 / 帧集帧数，再对清单取均值。

    **2026-09-21 真实数据修正**：本函数曾是 `track_consensus` 的判据，但实测
    （scene 41069043）该值只有 0.14–0.19 —— 手持扫描里单个物体本来只在少数帧出现，
    这个量回答的是"物体可见多久"，**不是**"同一实例是否被跨帧一致地跟成一条 track"。
    用它当门会让 `count_objects` 结构性不可用（counting 题的程序路径直接没了），
    而 v5 实测的计数失败根因是**重复实例**（清单 12 个重复）与**碎片化**，
    不是"可见帧少"。故仅作为可见帧率诊断，计数统计使用本题候选失败与存疑实例。
    """
    if n_frames <= 0 or not objects:
        return None
    vals = [min(1.0, len(set(int(i) for i in (o.visible_frames or []))) / float(n_frames))
            for o in objects]
    return float(sum(vals) / len(vals))


def _merge_mask_groups(masks_per_object, groups) -> list[dict]:
    """把去重分组还原成"每个保留对象一份掩码字典"（帧 → 该组掩码的并集）。

    用于在**去重后**的对象集上计算 G9，使统计口径与 `SceneState.objects` 一致。
    """
    merged: list[dict] = []
    for g in groups:
        frames: dict[int, np.ndarray] = {}
        for idx in g:
            if idx >= len(masks_per_object):
                continue
            for fi, m in masks_per_object[idx].items():
                m = np.asarray(m, dtype=bool)
                prev = frames.get(fi)
                if prev is None:
                    frames[fi] = m
                elif prev.shape == m.shape:
                    frames[fi] = np.logical_or(prev, m)
                # 形状不一致（同帧两条候选分辨率不同）：保留先到者，不伪造并集
        merged.append(frames)
    return merged


def _bbox_of(pts: np.ndarray) -> list[float]:
    pmin, pmax = pts.min(axis=0), pts.max(axis=0)
    return [float(v) for v in list(pmin) + list(pmax)]


def bind_objects_for_scene(frames, predictor=None, question="", **kwargs):
    """Shared ordinary/paired M5 entry point; legacy inventory has no online path."""
    from .question_binding import bind_question_objects
    return bind_question_objects(frames, predictor, question, **kwargs)
