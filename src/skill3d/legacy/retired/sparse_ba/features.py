"""逐帧稀疏特征提取（§10.1 L1；HC36 指定前端：SuperPoint 或 ALIKED）。

**流式纪律**（HC36 / §6.3）：一帧一次前向，提完立刻把张量挪到 CPU numpy 并释放显存；
不把 32 帧的特征同时留在 GPU 上。中间结果可选落盘（`.npz`），便于复现与 L2 复用。

坐标口径：特征在**统一特征分辨率**（默认 `FEATURE_MAX_SIDE=512` 缩放后的尺寸）上提取；
调用方必须用 `scale_intrinsics` 把 VGGT 内参映射到同一分辨率，否则 BA 的初值会错位。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

FrontendName = Literal["superpoint_lightglue", "aliked_lightglue"]

FEATURE_MAX_SIDE: int = 512      # TODO_CALIBRATE：特征长边（SuperPoint 要求 8 的倍数）
KEYPOINT_MIN_SCORE: float = 0.005   # TODO_CALIBRATE：低于该分的 keypoint 丢弃
MAX_KEYPOINTS: int = 2048        # TODO_CALIBRATE：单帧上限（官方 demo 量级）

# 各前端的描述子维度（用于缓存/匹配前的形状校验：**不同前端的缓存不得混用**）
FRONTEND_DESCRIPTOR_DIM: dict[str, int] = {
    "superpoint_lightglue": 256,
    "aliked_lightglue": 128,
}


def expected_descriptor_dim(frontend: FrontendName) -> int:
    try:
        return FRONTEND_DESCRIPTOR_DIM[str(frontend)]
    except KeyError as exc:  # pragma: no cover - 受控枚举已限
        raise FeatureExtractionError(f"未知前端 {frontend!r}") from exc


class FeatureExtractionError(RuntimeError):
    """特征提取不可用（缺权重/依赖）→ L1 直接止 loss，不得静默降级。"""


@dataclass
class FrameFeatures:
    """单帧稀疏特征（CPU numpy，便于落盘/释放显存）。"""

    frame_id: int
    keypoints: np.ndarray      # (N, 2) float32，特征分辨率像素坐标
    descriptors: np.ndarray    # (N, D) float32
    scores: np.ndarray         # (N,) float32
    feature_hw: tuple[int, int]


def _load_extractor(name: FrontendName, device: str, max_keypoints: int):
    """构造 LightGlue 官方前端（延迟导入，缺依赖时给出可操作错误）。"""
    try:
        from lightglue import ALIKED, SuperPoint
    except Exception as exc:  # noqa: BLE001
        raise FeatureExtractionError(
            f"lightglue 不可用（HC36 指定前端）：{exc}。"
            "请确认 skill3d-exp 环境已装 lightglue。") from exc
    if name == "superpoint_lightglue":
        # SuperPoint：top-k 模式（detection_threshold=0）+ 峰值上限，避免固定阈值在不同
        # 数据集亮度/纹理下大幅改变关键点数量（TODO_CALIBRATE）
        return SuperPoint(max_num_keypoints=int(max_keypoints),
                          detection_threshold=0.0).eval().to(device)
    if name == "aliked_lightglue":
        # ALIKED 默认 detection_threshold=0.2 会**先按分数阈值筛、再取 top-k**（源码：
        # threshold>0 时 top_k=-1），在真实室内帧上会只剩个位数关键点（实测 4~512）。
        # 改成 threshold=0 + max_num_keypoints → 纯 top-k，帧间数量稳定（TODO_CALIBRATE）。
        return ALIKED(max_num_keypoints=int(max_keypoints),
                      detection_threshold=0.0).eval().to(device)
    raise FeatureExtractionError(f"未知前端 {name!r}（HC36 只允许 superpoint/aliked + LightGlue）")


def feature_hw_for(src_hw: tuple[int, int],
                   max_side: int = FEATURE_MAX_SIDE) -> tuple[int, int]:
    """把原图尺寸缩放到"长边 = max_side 且两边都是 8 的倍数"的特征尺寸。"""
    h, w = int(src_hw[0]), int(src_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"非法帧尺寸 {src_hw}")
    scale = float(max_side) / float(max(h, w))
    nh = max(8, int(round(h * scale / 8.0)) * 8)
    nw = max(8, int(round(w * scale / 8.0)) * 8)
    return nh, nw


def scale_intrinsics(k: np.ndarray, src_hw: tuple[int, int],
                     dst_hw: tuple[int, int]) -> np.ndarray:
    """把 VGGT 内参从源分辨率映射到特征分辨率（`K' = S K S⁻¹`，S = diag(sx, sy, 1)）。"""
    k = np.asarray(k, dtype=np.float64)
    src_h, src_w = int(src_hw[0]), int(src_hw[1])
    dst_h, dst_w = int(dst_hw[0]), int(dst_hw[1])
    sx, sy = dst_w / float(src_w), dst_h / float(src_h)
    out = k.copy()
    out[0, :] *= sx
    out[1, :] *= sy
    return out.astype(np.float64)


def _to_tensor(rgb: np.ndarray, dst_hw: tuple[int, int]):
    import cv2
    import torch

    img = cv2.resize(np.asarray(rgb), (int(dst_hw[1]), int(dst_hw[0])),
                     interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).float().permute(2, 0, 1)[None] / 255.0
    return t


def extract_frames(
    frames: Sequence[np.ndarray],
    *,
    frontend: FrontendName = "superpoint_lightglue",
    device: str = "cuda",
    max_keypoints: int = MAX_KEYPOINTS,
    min_score: float = KEYPOINT_MIN_SCORE,
    max_side: int = FEATURE_MAX_SIDE,
    cache_path: Optional[str | Path] = None,
) -> tuple[list[FrameFeatures], tuple[int, int]]:
    """逐帧提取稀疏特征（流式；返回特征列表与特征分辨率 `(H, W)`）。

    `cache_path` 给定且已存在 → 直接读回（可复现 + 省 GPU）。
    """
    import torch

    dst_hw = feature_hw_for((int(np.asarray(frames[0]).shape[0]),
                             int(np.asarray(frames[0]).shape[1])), max_side)
    # 缓存按**前端**分文件，并在读回时校验形状：换前端后旧缓存不得被静默复用
    # （踩过的真实缺陷：SuperPoint 的 256 维描述子喂给 ALIKED 的 LightGlue → 断言崩溃）
    cache = (Path(cache_path).with_name(f"{Path(cache_path).stem}_{frontend}.npz")
             if cache_path else None)
    want_dim = expected_descriptor_dim(frontend)
    if cache is not None and cache.is_file():
        data = np.load(cache, allow_pickle=False)
        n = int(data["n_frames"])
        dim = int(data["desc_0"].shape[1]) if n else 0
        if dim != want_dim:
            # 维度不符 = 缓存来自别的前端 → 重新提取（不静默使用，也不抛给上层）
            import warnings

            warnings.warn(
                f"特征缓存 {cache} 描述子维度 {dim} ≠ 前端 {frontend} 期望 {want_dim}；"
                "忽略该缓存并重新提取。", stacklevel=2)
        else:
            out = [FrameFeatures(frame_id=i,
                                 keypoints=data[f"kp_{i}"],
                                 descriptors=data[f"desc_{i}"],
                                 scores=data[f"score_{i}"],
                                 feature_hw=dst_hw)
                   for i in range(n)]
            return out, dst_hw

    extractor = _load_extractor(frontend, device, max_keypoints)
    feats: list[FrameFeatures] = []
    try:
        with torch.no_grad():
            for i, fr in enumerate(frames):
                t = _to_tensor(fr, dst_hw).to(device)
                raw = extractor.extract(t)
                kp = raw["keypoints"][0].detach().cpu().numpy().astype(np.float32)
                desc = raw["descriptors"][0].detach().cpu().numpy().astype(np.float32)
                sc = (raw["keypoint_scores"][0].detach().cpu().numpy().astype(np.float32)
                      if "keypoint_scores" in raw else np.ones(len(kp), dtype=np.float32))
                keep = sc >= float(min_score)
                feats.append(FrameFeatures(frame_id=int(i), keypoints=kp[keep],
                                           descriptors=desc[keep], scores=sc[keep],
                                           feature_hw=dst_hw))
                del t, raw
                if device == "cuda":
                    torch.cuda.empty_cache()
    finally:
        del extractor
        if device == "cuda":
            torch.cuda.empty_cache()

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        payload = {"n_frames": np.asarray(len(feats)),
                   "frontend": np.asarray(frontend),
                   "descriptor_dim": np.asarray(want_dim)}
        for f in feats:
            payload[f"kp_{f.frame_id}"] = f.keypoints
            payload[f"desc_{f.frame_id}"] = f.descriptors
            payload[f"score_{f.frame_id}"] = f.scores
        np.savez_compressed(cache, **payload)
    return feats, dst_hw


def features_summary(feats: Sequence[FrameFeatures]) -> dict:
    counts = [len(f.keypoints) for f in feats]
    if not counts:
        return {"n_frames": 0}
    return {
        "n_frames": len(feats),
        "keypoints_min": int(min(counts)),
        "keypoints_max": int(max(counts)),
        "keypoints_mean": float(np.mean(counts)),
        "descriptor_dim": int(feats[0].descriptors.shape[1]),
        "feature_hw": list(feats[0].feature_hw),
    }


__all__ = [
    "FEATURE_MAX_SIDE",
    "KEYPOINT_MIN_SCORE",
    "MAX_KEYPOINTS",
    "FeatureExtractionError",
    "FrameFeatures",
    "extract_frames",
    "feature_hw_for",
    "features_summary",
    "scale_intrinsics",
]
