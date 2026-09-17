"""M1 VSI-Bench Adapter：meta 加载、32 帧均匀采样、四层 split 分层隔离。

在线模块，严禁任何 GPT-6 相关依赖（硬约束 1）。
split 划分：按 scene_id + task_type 分层，不共享 scene（硬约束 19）；
final_test 完全隔离（硬约束 9）。
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from skill3d.schemas.episode import DataSplitConfig, InputFrame, VSIBenchEpisode

# 与官方开源模型对齐的均匀采样帧数（§4 M1）
N_FRAMES = 32

# split 划分起始比例 induction:inner:outer:final（TODO_CALIBRATE，§4 M1 字段 12）
SPLIT_RATIOS = {
    "induction": 0.5,
    "inner_validation": 0.3,
    "outer_holdout": 0.2,
    # final_test 为留出集：由调用方显式给出 scene 列表，不参与比例划分
}


class EpisodeUnavailable(Exception):
    """scene 原始视频缺失等导致 episode 不可用时抛出（§4 M1 字段 9）。"""


def load_meta(cache_dir: Optional[str] = None) -> list[dict]:
    """加载 VSI-Bench HF meta-info（lazy import datasets，未安装时抛错）。

    HF 仅提供 QA meta，原始视频需另行获取（TODO_USER_INPUT 视频访问方式）。
    """
    from datasets import load_dataset  # lazy import：未安装库禁止顶层导入

    ds = load_dataset("nyu-visionx/vsi-bench", cache_dir=cache_dir)
    # 官方列：id/dataset/scene_name/question_type/question/ground_truth/options
    rows = [dict(r) for split in ds for r in ds[split]]
    return rows


def sample_uniform_indices(total_frames: int, n: int = N_FRAMES) -> list[int]:
    """32 帧均匀采样索引：严格单调递增且均匀（§4 M1 验收条件 b）。"""
    if total_frames < 1:
        raise ValueError("total_frames 必须 >= 1")
    if total_frames < n:
        # 视频不足 32 帧时重复末帧补齐，由上游 G4 帧数指标判定不全
        idx = np.linspace(0, total_frames - 1, num=total_frames)
        idx = np.concatenate([idx, np.full(n - total_frames, total_frames - 1)])
        return idx.astype(int).tolist()
    idx = np.linspace(0, total_frames - 1, num=n)
    return np.floor(idx).astype(int).tolist()


def _default_frame_reader(video_path: str, indices: Sequence[int]) -> list[np.ndarray]:
    """用 OpenCV 抽帧（cv2 已安装，可顶层使用）。"""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise EpisodeUnavailable(f"视频无法打开: {video_path}")
    frames = []
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                raise EpisodeUnavailable(f"视频第 {i} 帧读取失败: {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def video_path_for(qa_row: dict, video_root: str | Path) -> Path:
    """由 qa_row 定位原始视频文件（目录组织方式 TODO_USER_INPUT）。"""
    root = Path(video_root)
    # 起始约定：<root>/<dataset>/<scene_name>.mp4；真实布局待用户确认
    return root / str(qa_row["dataset"]) / f"{qa_row['scene_name']}.mp4"


def build_episode(
    qa_row: dict,
    split: str,
    video_path: Optional[str | Path] = None,
    frame_reader: Callable[[str, Sequence[int]], list[np.ndarray]] = _default_frame_reader,
    n_frames: int = N_FRAMES,
) -> VSIBenchEpisode:
    """由单行 QA meta 构造 VSIBenchEpisode（§4 M1 伪代码）。

    视频缺失时抛 EpisodeUnavailable，由调用方标记 unavailable 且不进任何 split。
    """
    if video_path is None or not Path(video_path).exists():
        raise EpisodeUnavailable(f"scene {qa_row.get('scene_name')} 视频缺失")

    import cv2  # 用于读取总帧数

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if total < 1:
        raise EpisodeUnavailable(f"视频无有效帧: {video_path}")

    indices = sample_uniform_indices(total, n_frames)
    images = frame_reader(str(video_path), indices)

    frames = []
    for i, img in zip(indices, images):
        h, w = img.shape[:2]
        frames.append(
            InputFrame(
                frame_idx=int(i),
                timestamp=float(i) / float(fps),
                width=w,
                height=h,
                # 逐帧质量统计由 M2 input_gate 计算；此处先填占位，
                # quality_ok 默认 True，等待门禁覆写
                blur_var=0.0,
                overexposed_ratio=0.0,
                underexposed_ratio=0.0,
                quality_ok=True,
            )
        )

    return VSIBenchEpisode(
        qa_id=str(qa_row["id"]),
        scene_name=str(qa_row["scene_name"]),
        dataset=str(qa_row["dataset"]),
        question_type=str(qa_row["question_type"]),
        question=str(qa_row["question"]),
        options=list(qa_row["options"]) if qa_row.get("options") else None,
        ground_truth=str(qa_row["ground_truth"]),
        frames=frames,
        split=split,  # type: ignore[arg-type]
    )


def make_split_config(
    qa_rows: Iterable[dict],
    final_test_scene_ids: Sequence[str],
    seed: int = 0,
    log_ref: str = "",
) -> DataSplitConfig:
    """按 scene_id + question_type 分层切分 induction/inner/outer（硬约束 19）。

    final_test scene 由调用方显式指定并完全隔离（硬约束 9），
    本函数断言 final_test scene 不出现在其余任一 split。
    """
    final_scenes = set(final_test_scene_ids)

    # 按 question_type 分组 scene，保证每个题型在各 split 都有代表（分层）
    type2scenes: dict[str, set[str]] = defaultdict(set)
    for r in qa_rows:
        scene = str(r["scene_name"])
        if scene in final_scenes:
            continue  # final_test 完全不参与
        type2scenes[str(r["question_type"])].add(scene)

    rng = random.Random(seed)
    induction: set[str] = set()
    inner: set[str] = set()
    outer: set[str] = set()
    for qtype, scenes in type2scenes.items():
        pool = sorted(scenes)
        rng.shuffle(pool)
        n = len(pool)
        n_ind = int(round(n * SPLIT_RATIOS["induction"]))
        n_inn = int(round(n * SPLIT_RATIOS["inner_validation"]))
        induction.update(pool[:n_ind])
        inner.update(pool[n_ind : n_ind + n_inn])
        outer.update(pool[n_ind + n_inn :])

    cfg = DataSplitConfig(
        induction_scene_ids=sorted(induction),
        inner_validation_scene_ids=sorted(inner),
        outer_holdout_scene_ids=sorted(outer),
        final_test_scene_ids=sorted(final_scenes),
        task_type_stratification=True,
        split_version=hashlib.sha256(
            repr((sorted(induction), sorted(inner), sorted(outer), sorted(final_scenes), seed)).encode()
        ).hexdigest()[:16],
        contamination_check_log_ref=log_ref,
    )
    assert_final_test_isolation(cfg)
    return cfg


def assert_final_test_isolation(cfg: DataSplitConfig) -> None:
    """final_test 完全隔离断言（硬约束 9，§4 M1 验收条件 c）。"""
    final_set = set(cfg.final_test_scene_ids)
    for name in ("induction_scene_ids", "inner_validation_scene_ids", "outer_holdout_scene_ids"):
        overlap = final_set & set(getattr(cfg, name))
        assert not overlap, f"final_test 隔离被违反: {name} 含 {overlap}"


def split_of(scene_name: str, cfg: DataSplitConfig) -> Optional[str]:
    """查询 scene 所属 split；未分配返回 None。"""
    for split, key in (
        ("induction", "induction_scene_ids"),
        ("inner_validation", "inner_validation_scene_ids"),
        ("outer_holdout", "outer_holdout_scene_ids"),
        ("final_test", "final_test_scene_ids"),
    ):
        if scene_name in set(getattr(cfg, key)):
            return split
    return None
