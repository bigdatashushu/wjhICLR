"""M1 统一固定 FrameSet（§4 M1 / 硬约束 21）：32 帧时间均匀采样的**单一事实源**。

规则（经用户确认的设计决策，不得擅自更改）：

- `frame_ids = np.linspace(0, n_frames_total - 1, 32).round().astype(int)`；
- 目标为 **32 个唯一物理帧**（时间戳严格单调递增）；**源视频不足 32 帧时按 v9 §5.2
  用全部可用帧继续**（不重复填充、不改采样算法），由 `n_frames` / `decode_status`
  与 `input_degraded` 显式记录 —— 此条取代原硬约束 21 的"不足 32 帧即整题 unavailable"
  （用户 2026-09-27 决定：三种解法中按 v9）；
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
# 预处理口径版本（进 FrameSet；改动取帧/缩放/解码口径时必须升版）
PREPROCESSING_VERSION: str = "preprocess-v1"


class FrameSetError(ValueError):
    """帧集不合法（无法构造 32 个唯一物理帧）。"""


def uniform_frame_ids(total_frames: int, n_frames: int = N_FRAMES) -> list[int]:
    """时间均匀采样帧索引（唯一、有序、严格单调递增）。

    与官方一致：`np.linspace(0, n-1, 32).round()`。

    v9 §5.2 取代硬约束 21 的"不足 32 帧即硬失败"：**源视频不足 32 帧但仍有真实可读
    帧时，保留帧身份与缺失掩码、用可读帧继续作答**。因此总帧数不足时返回**全部可用
    帧**（`0..total-1`，仍唯一且有序）——既不重复填充凑数（硬约束 21 禁止的做法），
    也不偷偷改采样算法（仍是时间均匀采样，只是可采样的时间轴更短）。帧数不足这一
    事实由 `FrameSet.n_frames` / `decode_status` 与 `input_degraded` 显式记录。

    只有"视频一帧都没有"才是真正无法继续的输入错误。
    """
    if total_frames < 1:
        raise FrameSetError(f"视频总帧数为 {total_frames}：无任何可用帧（§5.2 input_error）")
    if total_frames < n_frames:
        return [int(i) for i in range(total_frames)]
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
    episode_id: str = "",
    dataset_id: str = "",
    video_id: str = "",
    scene_name: str = "",
    frame_refs: Optional[Sequence[str]] = None,
    decode_status: str = "ok",
    preprocessing_version: str = PREPROCESSING_VERSION,
    readable_frame_ids: Optional[Sequence[int]] = None,
) -> FrameSet:
    """构造统一固定 FrameSet（M1 唯一入口）。

    `fps<=0` 时时间戳退化为帧号（仍严格单调，便于对齐第三方抽帧工具）；
    `source_frame_indices` 缺省与 `frame_ids` 相同——直接从原始视频解码时两者一致，
    上游若先落盘抽帧图片（jsonl 数据源）则显式传入源视频帧号。

    v9 §5.2：`episode_id / dataset_id / video_id / scene_name` 是**源标识**，
    与 `frame_set_hash`（内容校验）一起构成缓存身份（见 `FrameSet.cache_identity`）——
    否则两个不同视频采样出相同索引时会错误复用同一份重建产物。
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
        episode_id=episode_id,
        dataset_id=dataset_id,
        video_id=video_id,
        scene_name=scene_name,
        frame_refs=[str(x) for x in (frame_refs or [])],
        # §5.2：缺失掩码 —— 只有真正解码成功的规划帧进 readable_frame_ids；
        # 缺省（None）= 规划帧全部可读，与既有构造点语义一致。
        readable_frame_ids=([int(i) for i in readable_frame_ids]
                            if readable_frame_ids is not None else list(ids)),
        # §5.2：帧数必须**如实**记录实际规划帧数，不得沿用名义 32
        n_frames=len(ids),
        decode_status=("ok" if (len(ids) == n_frames
                                and (readable_frame_ids is None
                                     or len(readable_frame_ids) == len(ids)))
                       else "partial"),   # §5.2：帧数不足／部分可读 → partial
        preprocessing_version=preprocessing_version,
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
    if list(a.readable_frame_ids) != list(b.readable_frame_ids):
        # §5.2："成对实验必须复用同一**实际可读**集合" —— 规划帧相同还不够，
        # 缺失掩码也必须一致，否则两条臂实际看到的东西不同。
        raise FrameSetError(
            f"实际可读帧集不一致：{a.readable_frame_ids} != {b.readable_frame_ids}"
            "（§5.2：成对实验必须复用同一实际可读集合）"
        )
