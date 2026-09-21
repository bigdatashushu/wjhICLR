"""M1 统一固定 FrameSet（§4 M1 / 硬约束 21）：32 帧时间均匀采样的**单一事实源**。

规则（经用户确认的设计决策，不得擅自更改）：

- `frame_ids = np.linspace(0, n_frames_total - 1, 32).round().astype(int)`；
- 必须得到 **32 个唯一物理帧**（时间戳严格单调递增）；不到 32 帧的视频属于
  **输入合法性硬失败**（G4 帧数完整性）→ 抛 `EpisodeUnavailable`，整 episode
  `unavailable`，不进入任何 split（**不**用重复末帧补齐，硬约束 21）；
- `frame_set_hash = sha256(json.dumps(frame_ids, sort_keys=True))`：纯算术、确定性、
  可复现、**不读题目**（无答案泄漏风险）；
- 采样后帧集**冻结**：M2 只做被动质量观测（不删/不换/不补/不重排帧），
  M3/M5/M7/M8 及所有 baseline/消融共用同一帧序与同一 `frame_set_hash`。

不做题目感知选帧（不引入 CLIP/SigLIP、不维护双帧集）：专项调研显示 CLIP/SigLIP
选帧在空间推理任务上持平或下降，且引入新学习模型违反硬约束 1（§4 M1、§12 术语表）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional, Sequence

import numpy as np

from skill3d.schemas.episode import FrameSet

# 与 VSI-Bench 官方开源模型对齐的均匀采样帧数（§4 M1）
N_FRAMES: int = 32


class FrameSetError(ValueError):
    """帧集不合法（无法构造 32 个唯一物理帧）。"""


def uniform_frame_ids(total_frames: int, n_frames: int = N_FRAMES) -> list[int]:
    """时间均匀采样 `n_frames` 个**唯一**物理帧索引（严格单调递增）。

    与官方一致：`np.linspace(0, n-1, 32).round()`。视频总帧数不足 → 抛
    `FrameSetError`（输入合法性硬失败，硬约束 21：不允许重复帧凑数）。
    """
    if total_frames < n_frames:
        raise FrameSetError(
            f"视频仅 {total_frames} 帧，不足 {n_frames} 帧：输入合法性硬失败"
            "（G4 帧数完整性，硬约束 21 禁止重复帧补齐）"
        )
    ids = np.linspace(0, total_frames - 1, num=n_frames).round().astype(int)
    unique = np.unique(ids)
    if unique.size != n_frames:
        # 只有 total_frames < n_frames 时才会发生，上面已拦截；此处兜底防御
        raise FrameSetError(
            f"均匀采样只得 {unique.size} 个唯一帧（期望 {n_frames}）"
            f"：total_frames={total_frames}"
        )
    return [int(i) for i in ids]


def frame_set_hash(frame_ids: Sequence[int]) -> str:
    """帧集内容哈希（确定性；与 topic 无关，不读题目）。"""
    return hashlib.sha256(
        json.dumps([int(i) for i in frame_ids], sort_keys=True).encode()
    ).hexdigest()


def build_frame_set(
    total_frames: int,
    *,
    n_frames: int = N_FRAMES,
    fps: float = 0.0,
    source_frame_indices: Optional[Sequence[int]] = None,
) -> FrameSet:
    """构造统一固定 FrameSet（M1 唯一入口）。

    `fps<=0` 时时间戳退化为帧号（仍严格单调，便于对齐第三方抽帧工具）；
    `source_frame_indices` 缺省与 `frame_ids` 相同——直接从原始视频解码时两者一致，
    上游若先落盘抽帧图片（jsonl 数据源）则显式传入源视频帧号。
    """
    ids = uniform_frame_ids(total_frames, n_frames)
    if fps and fps > 0:
        stamps = [float(i) / float(fps) for i in ids]
    else:
        stamps = [float(i) for i in ids]
    return FrameSet(
        frame_ids=ids,
        source_frame_indices=[int(i) for i in (source_frame_indices or ids)],
        timestamps=stamps,
        frame_set_hash=frame_set_hash(ids),
        n_frames=int(n_frames),
        n_total_frames=int(total_frames),
        fps=float(fps or 0.0),
    )


def assert_same_frame_set(a: Optional[FrameSet], b: Optional[FrameSet]) -> None:
    """硬约束 21/18 断言：两条臂必须共用同一 FrameSet（paired A/B、复用 artifact）。"""
    if a is None or b is None:
        return
    if a.frame_set_hash != b.frame_set_hash:
        raise FrameSetError(
            f"帧集不一致：{a.frame_set_hash[:12]} != {b.frame_set_hash[:12]}"
            "（硬约束 21 禁止双帧集；硬约束 18 要求 paired A/B 同源）"
        )
