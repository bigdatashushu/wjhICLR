"""逐 pair 匹配 + 几何过滤（§10.1 L1；前端 LightGlue + RANSAC）。

**流式纪律**：一次只把一对帧的张量放上 GPU，匹配完立刻 `.detach().cpu()`、
`del`、`torch.cuda.empty_cache()`；显存 allocated 不得随 pair 数单调累积（L0 合同）。

**几何过滤**：RANSAC 估计基础矩阵（`cv2.FM_RANSAC`），只保留内点。
不做"按分数阈值再筛"的额外启发式——L0 要求 RANSAC 可复现，多余的自由度会让
复现性无法证明。动态区域排除（如有 SAM2 mask）由调用方在 `mask_out` 里传入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .features import FrameFeatures

RANSAC_REPROJ_THRESHOLD: float = 2.0   # TODO_CALIBRATE：内点阈值（特征分辨率像素）
MIN_INLIERS_PER_PAIR: int = 20         # TODO_CALIBRATE：单 pair 最少内点数


class MatchingError(RuntimeError):
    """匹配不可用（缺 lightglue 权重/依赖）→ L1 直接止损。"""


@dataclass
class PairMatch:
    """一次 pair 匹配的结果（CPU numpy）。"""

    frame_a: int
    frame_b: int
    kp_a: np.ndarray            # (M, 2) 内点坐标（a 帧，特征分辨率）
    kp_b: np.ndarray            # (M, 2)
    kp_id_a: np.ndarray         # (M,) a 帧 keypoint 索引
    kp_id_b: np.ndarray         # (M,)
    n_raw_matches: int = 0
    n_inliers: int = 0
    rejected_reason: str = ""

    @property
    def ok(self) -> bool:
        return (not self.rejected_reason) and self.n_inliers >= MIN_INLIERS_PER_PAIR


def _feat_dict(feat: FrameFeatures, device) -> dict:
    """构造 LightGlue 输入（含 `image_size`：`normalize_keypoints` 依赖它）。

    注意 `image_size` 必须是 `(W, H)` 顺序（LightGlue 内部按 x/y 归一化）。
    """
    import torch

    h, w = int(feat.feature_hw[0]), int(feat.feature_hw[1])
    return {
        "keypoints": torch.from_numpy(feat.keypoints)[None].to(device),
        "descriptors": torch.from_numpy(feat.descriptors)[None].to(device),
        "image_size": torch.tensor([[w, h]], dtype=torch.float32, device=device),
    }


def _load_matcher(device: str, features: str):
    try:
        from lightglue import LightGlue
    except Exception as exc:  # noqa: BLE001
        raise MatchingError(f"lightglue 不可用（HC36 指定后端）：{exc}") from exc
    return LightGlue(features=features).eval().to(device)


def match_pair(feat_a: FrameFeatures, feat_b: FrameFeatures, matcher, *,
               ransac_threshold: float = RANSAC_REPROJ_THRESHOLD) -> PairMatch:
    """单 pair：LightGlue 匹配 → RANSAC 内点过滤（纯函数，无全局状态）。"""
    import cv2
    import torch

    dev = next(matcher.parameters()).device
    out = PairMatch(frame_a=feat_a.frame_id, frame_b=feat_b.frame_id,
                    kp_a=np.zeros((0, 2), np.float32), kp_b=np.zeros((0, 2), np.float32),
                    kp_id_a=np.zeros((0,), np.int64), kp_id_b=np.zeros((0,), np.int64))
    if len(feat_a.keypoints) == 0 or len(feat_b.keypoints) == 0:
        out.rejected_reason = "empty_keypoints"
        return out
    inputs = {
        "image0": _feat_dict(feat_a, dev),
        "image1": _feat_dict(feat_b, dev),
    }
    res = None
    try:
        with torch.no_grad():
            res = matcher(inputs)
        m = res["matches"][0].detach().cpu().numpy().astype(np.int64)
    except AssertionError as exc:
        # 形状/维度不符 → 直接暴露上下文（便于定位前端问题，而不是静默 0 匹配）
        raise MatchingError(
            f"LightGlue 断言失败 pair=({feat_a.frame_id},{feat_b.frame_id})：{exc}；"
            f"kp0={tuple(inputs['image0']['keypoints'].shape)} "
            f"d0={tuple(inputs['image0']['descriptors'].shape)} "
            f"kp1={tuple(inputs['image1']['keypoints'].shape)} "
            f"d1={tuple(inputs['image1']['descriptors'].shape)}") from exc
    finally:
        del inputs
        if res is not None:
            del res
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    out.n_raw_matches = int(len(m))
    if len(m) < 8:
        out.rejected_reason = "too_few_raw_matches"
        return out
    ia, ib = m[:, 0], m[:, 1]
    p_a = feat_a.keypoints[ia].astype(np.float64)
    p_b = feat_b.keypoints[ib].astype(np.float64)
    # RANSAC 基础矩阵（确定性：固定阈值、固定方法；不做额外随机源）
    f_mat, inlier_mask = cv2.findFundamentalMat(
        p_a.reshape(-1, 1, 2), p_b.reshape(-1, 1, 2),
        cv2.FM_RANSAC, float(ransac_threshold), 0.999, maxIters=10000)
    if f_mat is None or inlier_mask is None:
        out.rejected_reason = "ransac_failed"
        return out
    mask = inlier_mask.ravel().astype(bool)
    out.kp_a = p_a[mask].astype(np.float32)
    out.kp_b = p_b[mask].astype(np.float32)
    out.kp_id_a = ia[mask]
    out.kp_id_b = ib[mask]
    out.n_inliers = int(mask.sum())
    if out.n_inliers < MIN_INLIERS_PER_PAIR:
        out.rejected_reason = "too_few_inliers"
    return out


@dataclass
class MatchReport:
    """整批 pair 的匹配汇总（L1/L2 报告用）。"""

    pairs: list[PairMatch] = field(default_factory=list)
    peak_gpu_gib: float = 0.0
    n_skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def n_pairs_ok(self) -> int:
        return sum(1 for p in self.pairs if p.ok)

    @property
    def n_matches(self) -> int:
        return sum(p.n_raw_matches for p in self.pairs)

    @property
    def n_inliers(self) -> int:
        return sum(p.n_inliers for p in self.pairs)

    def summary(self) -> str:
        return (f"pairs={len(self.pairs)} ok={self.n_pairs_ok} "
                f"matches={self.n_matches} inliers={self.n_inliers} "
                f"peak_gpu={self.peak_gpu_gib:.2f}GiB")


def match_pairs_streaming(
    feats: Sequence[FrameFeatures],
    pairs: Sequence[tuple[int, int]],
    *,
    frontend: str = "superpoint_lightglue",
    device: str = "cuda",
    ransac_threshold: float = RANSAC_REPROJ_THRESHOLD,
) -> MatchReport:
    """按预注册 pair graph 逐 pair 流式匹配（每对结束即释放显存）。"""
    import torch

    lg_features = ("superpoint" if frontend == "superpoint_lightglue" else "aliked")
    matcher = _load_matcher(device, lg_features)
    by_id = {f.frame_id: f for f in feats}
    report = MatchReport()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        for (a, b) in pairs:
            fa, fb = by_id.get(int(a)), by_id.get(int(b))
            if fa is None or fb is None:
                report.n_skipped += 1
                report.skip_reasons["missing_frame_features"] = (
                    report.skip_reasons.get("missing_frame_features", 0) + 1)
                continue
            pm = match_pair(fa, fb, matcher, ransac_threshold=ransac_threshold)
            report.pairs.append(pm)
            if pm.rejected_reason:
                report.skip_reasons[pm.rejected_reason] = (
                    report.skip_reasons.get(pm.rejected_reason, 0) + 1)
    finally:
        del matcher
        if device == "cuda":
            report.peak_gpu_gib = float(torch.cuda.max_memory_allocated() / (2 ** 30))
            torch.cuda.empty_cache()
    return report


def matches_to_track_input(report: MatchReport) -> list[list[tuple[tuple[int, int],
                                                                  tuple[float, float]]]]:
    """把内点匹配转成 `tracks.merge_tracks` 的输入形态（逐 pair 的扁平序列）。"""
    groups: list[list[tuple[tuple[int, int], tuple[float, float]]]] = []
    for pm in report.pairs:
        if pm.n_inliers <= 0:
            continue
        flat: list[tuple[tuple[int, int], tuple[float, float]]] = []
        for i in range(pm.n_inliers):
            flat.append(((pm.frame_a, int(pm.kp_id_a[i])),
                         (float(pm.kp_a[i, 0]), float(pm.kp_a[i, 1]))))
            flat.append(((pm.frame_b, int(pm.kp_id_b[i])),
                         (float(pm.kp_b[i, 0]), float(pm.kp_b[i, 1]))))
        groups.append(flat)
    return groups


def track_observations(track) -> np.ndarray:
    """`Track` → `(N, 2)` 观测坐标数组（喂 PyCOLMAP）。"""
    return np.asarray([o.xy for o in track.observations], dtype=np.float64)


def track_frame_ids(track) -> np.ndarray:
    return np.asarray([o.frame_id for o in track.observations], dtype=np.int64)


def optional_mask_filter(pm: PairMatch, masks: Optional[dict[int, np.ndarray]]) -> PairMatch:
    """如提供动态 mask（M5 SAM2 产物，特征分辨率），剔除落在动态区域的观测。

    `masks`：`{frame_id: bool ndarray(H, W)}`。未提供时原样返回（不做任何猜测）。
    """
    if not masks:
        return pm
    keep = np.ones(pm.n_inliers, dtype=bool)
    for i in range(pm.n_inliers):
        ma = masks.get(pm.frame_a)
        if ma is not None and ma[int(pm.kp_a[i, 1]), int(pm.kp_a[i, 0])]:
            keep[i] = False
            continue
        mb = masks.get(pm.frame_b)
        if mb is not None and mb[int(pm.kp_b[i, 1]), int(pm.kp_b[i, 0])]:
            keep[i] = False
    out = PairMatch(frame_a=pm.frame_a, frame_b=pm.frame_b,
                    kp_a=pm.kp_a[keep], kp_b=pm.kp_b[keep],
                    kp_id_a=pm.kp_id_a[keep], kp_id_b=pm.kp_id_b[keep],
                    n_raw_matches=pm.n_raw_matches, n_inliers=int(keep.sum()),
                    rejected_reason="" if keep.sum() >= MIN_INLIERS_PER_PAIR else "too_few_inliers")
    return out


__all__ = [
    "MIN_INLIERS_PER_PAIR",
    "MatchReport",
    "MatchingError",
    "PairMatch",
    "RANSAC_REPROJ_THRESHOLD",
    "match_pair",
    "match_pairs_streaming",
    "matches_to_track_input",
    "optional_mask_filter",
    "track_frame_ids",
    "track_observations",
]
