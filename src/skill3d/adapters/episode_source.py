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
from skill3d.routing.task_classifier import QUESTION_TYPE_VALUES, TASK_TYPES
from skill3d.schemas import InputErrorRecord, VSIBenchEpisode

# 8 个规范题型（§4 M7 / §16"8 任务分别准确率"）；合成数据在此 8 类上各造一条
ALL_TASKS = TASK_TYPES
# 官方 meta 的 10 个 question_type 原始取值（8 题型 + rel_direction 三档变体，§1.2）
ALL_QUESTION_TYPE_VALUES = QUESTION_TYPE_VALUES
# 兼容别名：规范题型集合（历史命名，等于 ALL_TASKS）
ALL_QUESTION_TYPES = ALL_TASKS


@dataclass
class EpisodeItem:
    """一条待跑条目：episode 元数据 + 帧像素（+ mock_light 合成几何）。"""

    episode: VSIBenchEpisode
    pixels: list[np.ndarray]
    geometry: Optional[_synth.SyntheticGeometry] = None
    source: str = "vsi_bench"  # vsi_bench | jsonl | synthetic
    video_path: str = ""
    input_error: Optional[InputErrorRecord] = None


class EpisodeSourceError(RuntimeError):
    """数据源不可用（缺 meta / 缺视频 / 缺 datasets 依赖）。"""


def _row_id(row: dict) -> str:
    return str(row.get("qa_id") or row.get("id") or "")


def _input_error_record(
    row: dict,
    split: str,
    reason: str,
    attempts: Sequence[dict] = (),
) -> InputErrorRecord:
    return InputErrorRecord(
        qa_id=_row_id(row),
        scene_name=str(row.get("scene_name", "") or ""),
        dataset=str(row.get("dataset", "unknown") or "unknown"),
        question_type=str(row.get("question_type", "") or ""),
        question=str(row.get("question", "") or ""),
        options=list(row["options"]) if row.get("options") else None,
        ground_truth=str(row.get("ground_truth", "") or ""),
        split=split,  # type: ignore[arg-type]
        reason=str(reason),
        source_attempts=[dict(attempt) for attempt in attempts],
    )


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

def _read_frames(video_path: str | Path,
                 indices: Sequence[int]) -> tuple[list[np.ndarray], list[int]]:
    """按规划帧号读像素 → `(可读帧像素, 缺失帧号)`。

    v9 §5.2："**部分**帧无法解码…保留帧身份和缺失掩码，使用可读帧继续作答；
    **不重复填充或偷偷改采样**。"因此单帧读失败**不再**让整个 episode 失败：
    失败帧只进缺失清单，返回的像素是**规划序下**的可读子集（不插占位帧、不重复邻近帧）。

    只有"一帧都读不出来"才由调用方按 `input_error` 记排除（§5.2 全部缺失）。
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise EpisodeSourceError(f"视频无法打开: {video_path}（TODO_USER_INPUT 视频获取）")
    frames: list[np.ndarray] = []
    missing: list[int] = []
    try:
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                missing.append(int(i))       # 缺失掩码：帧身份保留，内容不伪造
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return frames, missing


def load_vsi_bench_items(
    split: str,
    split_cfg: Optional[dict] = None,
    *,
    video_root: str | Path = "data/raw_videos",
    video_fallback_roots: Optional[Sequence[str | Path]] = None,
    cache_dir: Optional[str] = None,
    limit: Optional[int] = None,
    n_frames: int = _synth.N_FRAMES,
    seed: int = 0,
    question_types: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None,
    max_per_scene: int = 0,
    stratified_per_task: int = 0,
    sampling_seed: int = 0,
    exclusions: Optional[list] = None,
    sampling_receipt: Optional[dict] = None,
    include_input_errors: bool = False,
) -> list[EpisodeItem]:
    """从 HF meta + 原始视频装配条目（需 `datasets` 与本地视频，均 TODO_USER_INPUT）。

    `question_types`：按**规范题型**过滤（`object_rel_direction` 之类；easy/medium/hard
    变体一并归入其规范题型），在**抽帧之前**生效——分层子集实验必须走这里，
    否则会白解码大量视频（`--limit` 是过滤后的行数上限）。

    `datasets`：按来源数据集过滤（`scannet` / `scannetpp` / `arkitscenes`），同样在
    抽帧之前生效。用途：在 ARKitScenes 标定数据到位前，先把实验限定在
    scannet + scannetpp 上；**限定范围必须写进 RunManifest**（口径要可审计，不得
    把子集数字冒充全量）。

    `stratified_per_task`：**按规范题型均匀采样**——每个题型最多保留 N 条（0 = 不限）。
    按 meta 行顺序"先到先得"，因此采样确定、可复现，且在**抽帧之前**生效。
    用途：小规模分层子集实验（快出结果）。采样参数必须写进 RunManifest；
    小样本数字不得当主表（HC34）。

    `max_per_scene`：每个 scene 最多**抽帧**多少条 episode（0 = 不限）。它**不影响
    抽样集合**（抽样只由 split / datasets / question_types / stratified_per_task /
    limit 决定），因此重建与评测看到的是同一样本。重建路径用 `max_per_scene=1`：
    同 scene 的 episode 共用同一段视频与同一套帧，只抽一份即可（否则 induction
    1993 条会把同一视频解码几十遍、常驻 ~58 GB 内存）。
    """
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

    wanted: Optional[set[str]] = None
    if question_types:
        from skill3d.routing.task_classifier import canonical_task

        wanted = {canonical_task(t) for t in question_types}
    wanted_ds: Optional[set[str]] = ({str(d).strip().lower() for d in datasets}
                                     if datasets else None)

    from skill3d.routing.task_classifier import canonical_task as _canon

    # ---- 第一段：**抽样**（不抽帧）----
    # 采样口径（题型分层 / 数据集 / 题型子集 / limit）必须与评测路径**完全一致**，
    # 否则重建出来的 scene 集合与评测要跑的题对不上（曾实测：重建挑到 30 个 scene、
    # 评测只用其中 7 个）。故 scene 上限**不参与**抽样，只在第二段决定抽哪些帧。
    eligible: list[dict] = []
    for row in rows:
        scene = str(row["scene_name"])
        if allowed_scenes is not None and scene not in allowed_scenes:
            continue
        if wanted_ds is not None:
            if str(row.get("dataset", "")).strip().lower() not in wanted_ds:
                continue
        if wanted is not None:
            from skill3d.routing.task_classifier import (
                UnknownQuestionTypeError,
                canonical_task,
            )

            try:
                if canonical_task(str(row.get("question_type", ""))) not in wanted:
                    continue
            except UnknownQuestionTypeError:
                continue   # 非官方题型取值 → 不进任何 split（§1.2 contract error）
        eligible.append(row)

    # ---- 抽样（§5.3）----
    # 规范原文："评估清单从题目元数据与预登记的 scene／任务采样规则生成，**不以视频
    # 存在、成功解码或工具成功为筛选条件**"；"开发抽样以 **scene** 为单位控制覆盖与
    # 缓存成本，同时保证任务和方向难度覆盖"；"**禁止**默认按文件行序取每类前 N 题"。
    # 因此这里按 (seed, 题型) 对 scene 做确定性洗牌后顺序取，行内按 qa_id 稳定排序，
    # 视频存在性检查移到下面的加载段（缺失记为排除行，不进抽样条件）。
    if stratified_per_task:
        selected = _sample_by_scene(eligible, canon=_canon,
                                    per_task_cap=int(stratified_per_task),
                                    seed=int(sampling_seed))
    else:
        selected = list(eligible)
    if limit is not None:
        selected = selected[:int(limit)]
    if sampling_receipt is not None:
        sampling_receipt.clear()
        sampling_receipt.update(_sampling_receipt(
            selected, seed=int(sampling_seed),
            strategy=("seeded_scene_stratified" if stratified_per_task
                      else "declared_order"),
            per_task_cap=int(stratified_per_task or 0)))

    # ---- 第二段：抽帧（`max_per_scene` 只在此处生效）----
    items: list[EpisodeItem] = []
    per_scene: dict[str, int] = {}
    def _exclude(
        row: dict,
        reason: str,
        *,
        input_failure: bool = False,
        attempts: Sequence[dict] = (),
    ) -> Optional[InputErrorRecord]:
        """§5.3："每个预登记 qa_id 必须有结果行；不足 32 帧不能通过 skip 静默消失。"""
        record = None
        if input_failure:
            record = _input_error_record(row, split, reason, attempts)
        if exclusions is not None:
            exclusions.append(
                record.model_dump(mode="json") if record is not None else {
                    "status": "excluded",
                    "qa_id": _row_id(row),
                    "scene_name": str(row.get("scene_name", "") or ""),
                    "reason": reason,
                })
        return record

    for row in selected:
        scene = str(row["scene_name"])
        if max_per_scene and per_scene.get(scene, 0) >= int(max_per_scene):
            _exclude(row, "scene_cap_reached")
            continue
        loaded, attempts = _load_from_sources(
            row, video_root, video_fallback_roots, n_frames)
        if loaded is None:
            # §5.2："全部指定图像缺失或无法解码：`input_error`，记录原因" —— 换过
            # 所有来源副本之后仍然失败才走到这里，**不生成伪答案**（§5.3 分母保留）。
            failure = _exclude(
                row,
                _source_failure_reason(attempts),
                input_failure=True,
                attempts=attempts,
            )
            _record_source_attempts(sampling_receipt, row, attempts, used=None)
            if include_input_errors and failure is not None:
                items.append(EpisodeItem(
                    episode=failure.as_episode(),
                    pixels=[],
                    source="vsi_bench",
                    input_error=failure,
                ))
            continue
        pixels, planned, missing = loaded["pixels"], loaded["planned"], loaded["missing"]
        total, fps, source = loaded["total"], loaded["fps"], loaded["source"]
        if len(attempts) > 1:
            # 用了非首选来源 = 发生了**换来源副本**（§5.2 可重试加载）。必须留痕：
            # 副本可能是另一种编码，像素与主来源不同，事后要能查出来。
            _record_source_attempts(sampling_receipt, row, attempts, used=str(source.path))
        readable = [i for i in planned if i not in set(missing)]
        # §5.2：源标识参与缓存身份（不能只凭帧索引列表跨视频复用）；
        # 从副本加载时 `video_id` 带来源标签（`vsibench_loader.video_id_for`）。
        fset = vl.build_frame_set(
            total, n_frames=n_frames, fps=fps,
            episode_id=_row_id(row),
            dataset_id=str(row.get("dataset", "") or ""),
            video_id=vl.video_id_for(row, source),
            scene_name=scene,
            readable_frame_ids=readable)
        if missing:
            # §5.2："部分缺帧样本标记 `input_degraded`，**独立报告**；不伪称完整 32 帧
            # 输入，也**不从评分分母静默删除**。" —— 所以这里 **不** 排除该行，
            # 它照常产出 episode（用可读帧继续），只把缺失掩码单独登记。
            # `input_degraded` 标记本身由 M2 在帧数不足时自动加上。
            if sampling_receipt is not None:
                sampling_receipt.setdefault("partially_readable", []).append({
                    "qa_id": _row_id(row),
                    "n_planned": len(planned), "n_readable": len(readable),
                    "missing_frame_ids": list(missing)})
        episode = _episode_from_row(row, split, pixels, n_frames, frame_set=fset)
        items.append(EpisodeItem(episode=episode, pixels=pixels, source="vsi_bench",
                                 video_path=str(source.path)))
        per_scene[scene] = per_scene.get(scene, 0) + 1

    if not items:
        hint = (f"（question_types={sorted(wanted)}）" if wanted else "")
        if wanted_ds:
            hint += f"（datasets={sorted(wanted_ds)}）"
        raise EpisodeSourceError(
            f"split={split} 无可用 episode{hint}：请确认原始视频已放在 {video_root}"
            "（TODO_USER_INPUT：VSI-Bench 视频获取与目录约定见 §4 M1 字段 12）"
        )
    return items


def _load_from_sources(row: dict, video_root, fallback_roots,
                       n_frames: int) -> tuple[Optional[dict], list[dict]]:
    """§5.2"可重试加载"：按来源候选逐个尝试，直到拿到**真实可读帧**。

    **决策记录（用户 2026-09-27）**："重试"=**换来源/换副本**（另一个根目录、镜像目录
    或同名不同容器的副本），不是对同一文件反复解码。候选顺序见
    `vsibench_loader.video_source_candidates`。

    返回 `(loaded, attempts)`：

    - `loaded is None` = **所有来源都失败**（调用方记 `input_error` 排除行，不生成伪答案）；
    - 否则 `loaded` 含实际使用的来源、规划帧、可读像素、缺失掩码与 total/fps；
    - `attempts` 逐个来源记结果（`missing` / `undecodable` / `no_frames` /
      `all_frames_unreadable` / `used`），是"记录原因"的原始事实。
    """
    import cv2

    from skill3d.adapters import vsibench_loader as vl

    attempts: list[dict] = []
    for source in vl.video_source_candidates(row, video_root, fallback_roots):
        attempt = {"source": str(source.path), "convention": source.convention}
        if not Path(source.path).exists():
            attempts.append({**attempt, "result": "missing"})
            continue
        cap = cv2.VideoCapture(str(source.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if total < 1:
            attempts.append({**attempt, "result": "undecodable"})
            continue
        try:
            planned = vl.uniform_frame_ids(total, n_frames)
        except vl.FrameSetError:
            # 只有"一帧都没有"才到这里（§5.2 input_error）
            attempts.append({**attempt, "result": "no_frames"})
            continue
        # §5.2：先把规划帧读出来，才知道哪些真正可读（缺失掩码的来源）
        try:
            pixels, missing = _read_frames(source.path, planned)
        except EpisodeSourceError:
            attempts.append({**attempt, "result": "unreadable"})
            continue
        if not pixels:
            attempts.append({**attempt, "result": "all_frames_unreadable"})
            continue
        attempts.append({**attempt, "result": "used"})
        return ({"source": source, "planned": planned, "pixels": pixels,
                 "missing": missing, "total": total, "fps": fps}, attempts)
    return None, attempts


def _source_failure_reason(attempts: Sequence[dict]) -> str:
    """所有来源都失败时的排除原因（沿用既有词表，不新增自由字符串）。

    取"走得最远"的那个失败作为原因，因此主来源缺失、副本能打开但解不出帧时，
    记的是 `video_undecodable`（比 `video_missing` 更准确），而不是笼统的一句"缺失"。
    """
    results = {str(a.get("result", "")) for a in attempts}
    if results & {"undecodable", "all_frames_unreadable", "unreadable"}:
        return "video_undecodable"
    if "no_frames" in results:
        return "insufficient_frames"
    return "video_missing"      # §4 M1 字段 9：来源校验失败单独记录


def _record_source_attempts(sampling_receipt: Optional[dict], row: dict,
                            attempts: Sequence[dict], *, used: Optional[str]) -> None:
    """把"换来源副本"的事实写进抽样收据（§5.2 可重试加载必须可审计）。

    只在**确实换过来源**（用了非首选候选）或**全部来源失败**时登记；主来源一次成功
    的常规路径不写任何东西，收据保持干净。
    """
    if sampling_receipt is None:
        return
    primary = str(attempts[0]["source"]) if attempts else ""
    if used is not None and len(attempts) <= 1:
        return
    sampling_receipt.setdefault("source_retries", []).append({
        "qa_id": _row_id(row),
        "scene_name": str(row.get("scene_name", "") or ""),
        "primary_source": primary,
        "used_source": str(used or ""),
        "retried": bool(used is not None and len(attempts) > 1),
        "attempts": [dict(a) for a in attempts],
    })


def _sample_by_scene(rows: list[dict], *, canon, per_task_cap: int,
                     seed: int) -> list[dict]:
    """§5.3：以 **scene** 为单位、按 (seed, 题型) 的确定性抽样。

    - 单位是 scene（控制覆盖与缓存成本），组内行按 qa_id 稳定排序；
    - scene 顺序由 `random.Random(f"{seed}:{task}")` 洗牌决定 —— **与文件行序无关**，
      因此"每类前 N 题"不再等于"文件里最先出现的 N 行"；
    - 同 seed 完全可复现（抽样收据里的 hash 即由此保证）。
    """
    import random

    by_task_scene: dict[str, dict[str, list[dict]]] = {}
    for row in rows:
        try:
            task = canon(str(row.get("question_type", "")))
        except Exception:  # noqa: BLE001 - 未知题型不参与分层（§1.2 适配错误）
            continue
        buckets = by_task_scene.setdefault(task, {})
        buckets.setdefault(str(row.get("scene_name", "")), []).append(row)

    out: list[dict] = []
    for task in sorted(by_task_scene):
        scenes = sorted(by_task_scene[task])
        random.Random(f"{seed}:{task}").shuffle(scenes)
        taken = 0
        for scene in scenes:
            if taken >= per_task_cap:
                break
            for row in sorted(by_task_scene[task][scene],
                              key=_row_id):
                if taken >= per_task_cap:
                    break
                out.append(row)
                taken += 1
    return out


def _sampling_receipt(selected: list[dict], *, seed: int, strategy: str,
                      per_task_cap: int) -> dict:
    """§5.3：抽样算法、scene／qa_id 清单、seed 和 hash 都要落盘。"""
    import hashlib

    qa_ids = [_row_id(r) for r in selected]
    scenes = sorted({str(r.get("scene_name", "") or "") for r in selected})
    canon = hashlib.sha256(
        "\n".join(qa_ids).encode("utf-8")).hexdigest()
    return {"strategy": strategy, "seed": int(seed),
            "per_task_cap": int(per_task_cap),
            "n_preregistered": len(qa_ids),
            "qa_ids": qa_ids, "scenes": scenes,
            "qa_id_sha256": canon}


def _episode_from_row(row: dict, split: str, pixels: Sequence[np.ndarray],
                      n_frames: int, *, frame_set=None) -> VSIBenchEpisode:
    """由 meta 行 + 已读像素构造 VSIBenchEpisode（帧统计由 M2 覆写）。

    `frame_set` 为 M1 冻结的统一帧集（硬约束 21）；缺省时按"已读像素即 0..N-1 帧"
    构造等价的 FrameSet（jsonl 预抽帧来源：传输帧序 = 规范槽位顺序）。
    """
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
    if frame_set is None:
        from .frame_set import frame_set_hash as _fsh

        ids = list(range(len(pixels)))
        frame_set = {
            "frame_ids": ids,
            "source_frame_indices": list(row.get("source_frame_indices") or ids),
            "timestamps": [float(i) for i in ids],
            "frame_set_hash": _fsh(ids),
            "n_frames": len(pixels),
            "n_total_frames": int(row.get("n_total_frames") or len(pixels)),
            "fps": float(row.get("fps") or 0.0),
        }
    return VSIBenchEpisode(
        qa_id=_row_id(row),
        scene_name=str(row["scene_name"]),
        dataset=str(row["dataset"]),
        question_type=str(row["question_type"]),
        question=str(row["question"]),
        options=list(row["options"]) if row.get("options") else None,
        ground_truth=str(row["ground_truth"]),
        frames=frames,
        split=split,  # type: ignore[arg-type]
        frame_set=frame_set,
    )


# ---------------------------------------------------------------------- jsonl ----

_JSONL_REQUIRED = ("qa_id", "scene_name", "question_type", "question", "ground_truth",
                   "split", "frame_paths")


def load_jsonl_items(
    path: str | Path,
    split: Optional[str] = None,
    *,
    limit: Optional[int] = None,
    exclusions: Optional[list] = None,
    include_input_errors: bool = False,
) -> list[EpisodeItem]:
    """读本系统约定格式的预抽帧清单（见模块 docstring）。"""
    import cv2

    p = Path(path)
    if not p.is_file():
        raise EpisodeSourceError(f"episodes jsonl 不存在: {p}")
    items: list[EpisodeItem] = []
    matched = 0
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
        if limit is not None and matched >= limit:
            break
        matched += 1
        frame_paths = list(rec["frame_paths"])
        pixels = []
        readable: list[int] = []
        attempts: list[dict] = []
        for index, fp in enumerate(frame_paths):
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
            if img is None:
                attempts.append({"source": str(fp), "result": "unreadable"})
                continue
            attempts.append({"source": str(fp), "result": "used"})
            readable.append(index)
            pixels.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        row = {
            "id": rec["qa_id"],
            "qa_id": rec["qa_id"],
            "scene_name": rec["scene_name"],
            "dataset": rec.get("dataset", "unknown"),
            "question_type": rec["question_type"],
            "question": rec["question"],
            "options": rec.get("options"),
            "ground_truth": rec["ground_truth"],
        }
        if not pixels:
            failure = _input_error_record(
                row, rec["split"], "all_frames_unreadable", attempts)
            if exclusions is not None:
                exclusions.append(failure.model_dump(mode="json"))
            if include_input_errors:
                items.append(EpisodeItem(
                    episode=failure.as_episode(),
                    pixels=[],
                    source="jsonl",
                    input_error=failure,
                ))
            continue
        frame_set = None
        if len(readable) != len(frame_paths):
            from .frame_set import build_frame_set

            frame_set = build_frame_set(
                len(frame_paths),
                n_frames=len(frame_paths),
                source_frame_indices=list(rec.get("source_frame_indices")
                                          or range(len(frame_paths))),
                episode_id=str(rec["qa_id"]),
                dataset_id=str(rec.get("dataset", "unknown")),
                video_id=str(p.resolve()),
                scene_name=str(rec["scene_name"]),
                frame_refs=[str(value) for value in frame_paths],
                readable_frame_ids=readable,
            )
        episode = _episode_from_row(
            row,
            rec["split"],
            pixels,
            len(pixels),
            frame_set=frame_set,
        )
        items.append(EpisodeItem(episode=episode, pixels=pixels, source="jsonl"))
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
                "input_error": ({
                    "status": it.input_error.status,
                    "reason": it.input_error.reason,
                    "source_attempts": it.input_error.source_attempts,
                } if it.input_error is not None else None),
            }, ensure_ascii=False) + "\n")
    return out
