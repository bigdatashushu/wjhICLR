"""episode 数据源（M1 上层装配）：把"episode + 像素 + 合成几何"装成统一条目。

三种来源：
- `vsi_bench`：HF meta（`nyu-visionx/vsi-bench`）+ 原始视频抽帧。
  视频需自行获取与预处理（`TODO_USER_INPUT`，§15.1）；本系统不虚构视频路径。
- `jsonl`：本系统约定格式的预抽帧清单（VSI-Bench meta 只给 QA，帧需自备），
  每行：qa_id / scene_name / dataset / question_type / question / options /
  ground_truth / split / frame_paths（32 张图的路径列表）。
- `synthetic`：`online/synthetic.py` 合成（仅 mock_light 管道验证）。

在线模块，禁止 import governance / gpt6（硬约束 1）。`reconstruction.run` 与
`online.eval` 共用本模块，避免两条 CLI 各写一套装配逻辑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from skill3d.online import synthetic as _synth
from skill3d.routing.task_classifier import MCA_TYPES, NA_TYPES
from skill3d.schemas import VSIBenchEpisode

# 8 题型（官方 question_type 取值集合，§4 M7）
ALL_QUESTION_TYPES = tuple(sorted(MCA_TYPES | NA_TYPES))


@dataclass
class EpisodeItem:
    """一条待跑条目：episode 元数据 + 帧像素（+ mock_light 合成几何）。"""

    episode: VSIBenchEpisode
    pixels: list[np.ndarray]
    geometry: Optional[_synth.SyntheticGeometry] = None
    source: str = "vsi_bench"  # vsi_bench | jsonl | synthetic
    video_path: str = ""


class EpisodeSourceError(RuntimeError):
    """数据源不可用（缺 meta / 缺视频 / 缺 datasets 依赖）。"""


# ------------------------------------------------------------------ synthetic ----

def load_synthetic_items(
    split: str,
    *,
    question_types: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    seed: int = 0,
    n_frames: int = _synth.N_FRAMES,
    frame_size: tuple[int, int] = (_synth.FRAME_H, _synth.FRAME_W),
    degrade: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> list[EpisodeItem]:
    """合成条目（确定性与 seed 绑定；题型按需子集，默认 8 题型各一条）。"""
    types = list(question_types) if question_types else list(ALL_QUESTION_TYPES)
    items: list[EpisodeItem] = []
    for i, qt in enumerate(types):
        scene = f"synthetic-{split}-{qt}"
        se = _synth.make_synthetic_episode(
            qt,
            scene_name=scene,
            qa_id=f"synth-{split}-{qt}",
            split=split,
            seed=seed + i,
            n_frames=n_frames,
            frame_size=frame_size,
            degrade=degrade,
            out_dir=out_dir,
        )
        items.append(
            EpisodeItem(episode=se.episode, pixels=se.frames, geometry=se.geometry,
                        source="synthetic")
        )
        if limit is not None and len(items) >= limit:
            break
    return items


# ------------------------------------------------------------------ vsi_bench ----

def _read_frames(video_path: str | Path, indices: Sequence[int]) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise EpisodeSourceError(f"视频无法打开: {video_path}（TODO_USER_INPUT 视频获取）")
    frames: list[np.ndarray] = []
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                raise EpisodeSourceError(f"视频第 {i} 帧读取失败: {video_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames


def load_vsi_bench_items(
    split: str,
    split_cfg: Optional[dict] = None,
    *,
    video_root: str | Path = "data/raw_videos",
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    n_frames: int = _synth.N_FRAMES,
    seed: int = 0,
) -> list[EpisodeItem]:
    """从 HF meta + 原始视频装配条目（需 `datasets` 与本地视频，均 TODO_USER_INPUT）。"""
    from skill3d.adapters import vsibench_loader as vl

    try:
        rows = vl.load_meta(cache_dir=cache_dir)
    except ImportError as exc:
        raise EpisodeSourceError(
            "未安装 datasets（pip install datasets）；VSI-Bench meta 加载不可用"
        ) from exc

    split_key = {
        "induction": "induction_scene_ids",
        "inner_validation": "inner_validation_scene_ids",
        "outer_holdout": "outer_holdout_scene_ids",
        "final_test": "final_test_scene_ids",
    }.get(split)
    allowed_scenes: Optional[set[str]] = None
    if split_cfg and split_key:
        allowed = split_cfg.get(split_key) or []
        if allowed:
            allowed_scenes = set(allowed)

    items: list[EpisodeItem] = []
    for row in rows:
        scene = str(row["scene_name"])
        if allowed_scenes is not None and scene not in allowed_scenes:
            continue
        vpath = vl.video_path_for(row, video_root)
        if not Path(vpath).exists():
            continue  # 视频缺失 → 该 episode 不进任何 split（§4 M1 字段 9）
        import cv2

        cap = cv2.VideoCapture(str(vpath))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total < 1:
            continue
        indices = vl.sample_uniform_indices(total, n_frames)
        pixels = _read_frames(vpath, indices)
        episode = _episode_from_row(row, split, pixels, n_frames)
        items.append(EpisodeItem(episode=episode, pixels=pixels, source="vsi_bench",
                                 video_path=str(vpath)))
        if limit is not None and len(items) >= limit:
            break

    if not items:
        raise EpisodeSourceError(
            f"split={split} 无可用 episode：请确认原始视频已放在 {video_root}"
            "（TODO_USER_INPUT：VSI-Bench 视频获取与目录约定见 §4 M1 字段 12）"
        )
    return items


def _episode_from_row(row: dict, split: str, pixels: Sequence[np.ndarray],
                      n_frames: int) -> VSIBenchEpisode:
    """由 meta 行 + 已读像素构造 VSIBenchEpisode（帧统计由 M2 覆写）。"""
    frames = [
        dict(
            frame_idx=i,
            timestamp=float(i),
            width=int(img.shape[1]),
            height=int(img.shape[0]),
            blur_var=0.0,
            overexposed_ratio=0.0,
            underexposed_ratio=0.0,
            quality_ok=True,
        )
        for i, img in enumerate(pixels)
    ]
    return VSIBenchEpisode(
        qa_id=str(row["id"]),
        scene_name=str(row["scene_name"]),
        dataset=str(row["dataset"]),
        question_type=str(row["question_type"]),
        question=str(row["question"]),
        options=list(row["options"]) if row.get("options") else None,
        ground_truth=str(row["ground_truth"]),
        frames=frames,
        split=split,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------- jsonl ----

_JSONL_REQUIRED = ("qa_id", "scene_name", "question_type", "question", "ground_truth",
                   "split", "frame_paths")


def load_jsonl_items(
    path: str | Path,
    split: Optional[str] = None,
    *,
    limit: Optional[int] = None,
) -> list[EpisodeItem]:
    """读本系统约定格式的预抽帧清单（见模块 docstring）。"""
    import cv2

    p = Path(path)
    if not p.is_file():
        raise EpisodeSourceError(f"episodes jsonl 不存在: {p}")
    items: list[EpisodeItem] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        missing = [k for k in _JSONL_REQUIRED if k not in rec]
        if missing:
            raise EpisodeSourceError(f"{p}:{lineno} 缺字段 {missing}")
        if split is not None and rec["split"] != split:
            continue
        frame_paths = list(rec["frame_paths"])
        pixels = []
        for fp in frame_paths:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                raise EpisodeSourceError(f"{p}:{lineno} 帧读取失败: {fp}")
            pixels.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        episode = _episode_from_row(
            {
                "id": rec["qa_id"],
                "scene_name": rec["scene_name"],
                "dataset": rec.get("dataset", "unknown"),
                "question_type": rec["question_type"],
                "question": rec["question"],
                "options": rec.get("options"),
                "ground_truth": rec["ground_truth"],
            },
            rec["split"],
            pixels,
            len(pixels),
        )
        items.append(EpisodeItem(episode=episode, pixels=pixels, source="jsonl"))
        if limit is not None and len(items) >= limit:
            break
    if not items:
        raise EpisodeSourceError(f"{p} 未匹配到 split={split} 的条目")
    return items


def write_episode_meta_jsonl(items: Sequence[EpisodeItem], out_path: str | Path) -> Path:
    """把条目的 meta（不含像素）落盘为清单，便于复核与复现（不写 GT 给 GPT-6）。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            ep = it.episode
            f.write(json.dumps({
                "qa_id": ep.qa_id, "scene_name": ep.scene_name, "dataset": ep.dataset,
                "question_type": ep.question_type, "question": ep.question,
                "options": ep.options, "split": ep.split, "source": it.source,
                "n_frames": len(it.pixels),
            }, ensure_ascii=False) + "\n")
    return out
