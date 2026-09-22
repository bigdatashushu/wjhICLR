"""P1 批量重建 CLI（§13.5）：

```bash
python -m skill3d.reconstruction.run --split induction,inner,outer --method vggt
```

- **v6 §5.2**：正式主线固定 `vggt` feed-forward，`recon_method` 受控枚举**只有** `vggt`；
  BA（官方 VGGSfM + PyCOLMAP / `vggt_sparse_ba`）与 `colmap` / `dust3r_mast3r` 对照基线
  已按 §20 整体废止，只读归档在 `skill3d/legacy/retired/`（运行时代码不得 import）；
- 重建以 **scene** 为单位（同 scene 的多个 episode 共享一次重建，§4 M3）；
- 已存在 artifact 的 scene 默认跳过（断点恢复幂等，§4 M21 字段 7/8）；
- DP 分配用 M20 `GPUScheduler`（每卡一个 scene，paired 同卡规则不影响本阶段）；
- **诚实性**：本命令只跑真实重建（vggt）。合成几何只允许经
  `python -m skill3d.online.eval --mode mock_light` 走管道验证，`recon_method`
  （§5.2 受控枚举）不得被合成产物冒用；
- 缺原始视频 → 明确报错（`TODO_USER_INPUT`，§15.1），不静默跳过、不伪造。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from skill3d.adapters.episode_source import (
    EpisodeItem,
    EpisodeSourceError,
    load_jsonl_items,
    load_vsi_bench_items,
)
from skill3d.online.config import DEFAULT_CONFIG, load_config, load_yaml, paths_from
from skill3d.reconstruction.vggt_runner import ReconstructionFailed, reconstruct
from skill3d.scheduling.gpu_scheduler import GPUScheduler
from skill3d.schemas import ReconstructionArtifact


@dataclass
class SceneJob:
    """一个待重建的 scene（同 scene 的 episode 合并）。"""

    scene_name: str
    split: str
    n_episodes: int
    video_path: str = ""
    artifact_ref: str = ""          # 已存在的 artifact（跳过时填）
    status: str = "pending"         # pending | done | skipped | failed
    gpu_rank: Optional[int] = None
    note: str = ""
    artifact: Optional[ReconstructionArtifact] = None


def artifact_path(recon_dir: str | Path, scene_name: str, method: str) -> Path:
    """artifact 落盘约定：<recon_dir>/<method>/<scene>.json + 同名 .npy 数组。"""
    return Path(recon_dir) / method / f"{scene_name}.json"


def plan_scene_jobs(items: Sequence[EpisodeItem], recon_dir: str | Path, method: str,
                    force: bool = False) -> list[SceneJob]:
    """按 scene 归并 episode → 场景作业表；已有 artifact 且 !force → 标 skipped。"""
    by_scene: dict[str, SceneJob] = {}
    for it in items:
        ep = it.episode
        key = ep.scene_name
        job = by_scene.get(key)
        if job is None:
            by_scene[key] = SceneJob(scene_name=key, split=ep.split, n_episodes=1,
                                     video_path=it.video_path)
        else:
            job.n_episodes += 1
    jobs = sorted(by_scene.values(), key=lambda j: (j.split, j.scene_name))
    for job in jobs:
        p = artifact_path(recon_dir, job.scene_name, method)
        if p.exists() and not force:
            job.status = "skipped"
            job.artifact_ref = str(p)
            job.note = "artifact 已存在（断点恢复幂等，§4 M21）"
    return jobs


def run_jobs(jobs: list[SceneJob], items_by_scene: dict[str, list[EpisodeItem]],
             recon_dir: str | Path, method: str, gpus: Optional[list[int]] = None,
             n_frames: int = 32, metric_depth_model=None) -> list[SceneJob]:
    """执行场景作业（DP 逻辑分配；重建本身按 §4 M3 主线）。

    **M4 前移 P1（方案 X，§2.2）**：重建时顺手算 G1–G11 并写回 artifact，
    quality 随 artifact 落盘，P2 加载即得实算值、零重算。
    """
    pending = [j for j in jobs if j.status == "pending"]
    scheduler = GPUScheduler(world_size=len(gpus) if gpus else 1)
    for job in pending:
        gpu = scheduler.assign(job.scene_name, role="reconstruct")
        job.gpu_rank = gpu.gpu_rank
        # 同 scene 的多个 episode 来自同一段视频 → 只取一份 32 帧（不拼接多份帧）
        scene_items = items_by_scene.get(job.scene_name, [])
        frames = list(scene_items[0].pixels) if scene_items else []
        if not frames:
            job.status, job.note = "failed", "无可用帧（视频缺失或抽帧失败）"
            continue
        try:
            art = reconstruct(
                frames, job.scene_name, Path(recon_dir) / method, method=method,
                frame_set=scene_items[0].episode.frame_set if scene_items else None,
                metric_depth_model=metric_depth_model,
                metric_model_name=("moge2" if metric_depth_model is not None else "none"),
            )
        except ReconstructionFailed as exc:
            job.status, job.note = "failed", f"重建失败（降级链已走完）: {exc}"
            continue
        except Exception as exc:  # noqa: BLE001 - 依赖/显存等运行时错误
            job.status, job.note = "failed", f"{type(exc).__name__}: {exc}"
            continue
        out = artifact_path(recon_dir, job.scene_name, method)
        out.parent.mkdir(parents=True, exist_ok=True)
        # 方案 X（§4 M4 / §2.2）：M4 前移到 P1 —— 质量随 artifact 一起落盘。
        # v6 主线只有 vggt，其质量已在 run_vggt 内部算过（此处为零重算的兜底补齐）。
        try:
            from skill3d.reconstruction_gate import quality_metrics as qm

            if not qm.quality_is_computed(art):
                art = qm.compute_and_store_quality(
                    art, frames=frames,
                    depth_maps=_load_npy(art.depth_maps),
                    c2w_list=_load_npy(art.c2w_list),
                    artifact_path=str(out))
        except Exception as exc:  # noqa: BLE001 - 质量算不出不阻断重建（P2 会按 fail-closed 降级）
            job.note = f"[warn] M4 质量写回失败: {type(exc).__name__}: {exc}"
        out.write_text(art.model_dump_json(indent=2), encoding="utf-8")
        job.status, job.artifact, job.artifact_ref = "done", art, str(out)
        job.note = (f"method={art.recon_method} "
                    f"quality_status={art.quality_status} "
                    f"overall_quality={_overall_of(art)} "
                    f"frame_set_hash={art.frame_set_hash[:12]}")
    return jobs


def _load_npy(ref):
    """读数组 ref（缺失/损坏返回 None，不伪造）。"""
    if not ref:
        return None
    try:
        import numpy as np

        return np.load(ref)
    except Exception:  # noqa: BLE001
        return None


def _overall_of(art) -> str:
    """artifact 的实算 overall_quality（未计算时显式写 not_computed，不写 NaN）。"""
    q = getattr(art, "quality", None)
    if getattr(art, "quality_status", "") != "computed" or q is None:
        return "not_computed"
    return f"{float(q.overall_quality):.3f}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m skill3d.reconstruction.run",
        description="P1 批量 3D 重建（M3）：按 scene 跑 VGGT feed-forward 并落 artifact",
    )
    p.add_argument("--split", default="induction,inner_validation,outer_holdout",
                   help="逗号分隔的 split 列表（默认 induction,inner_validation,outer_holdout）")
    # v6 §11：度量尺度融合开关（默认关；§11 全部 [待实验]）
    p.add_argument("--moge2", action="store_true",
                   help="启用 MoGe-2 度量尺度融合（§11.2），artifact 将带 metric_scale")
    p.add_argument("--method", default="vggt", choices=["vggt"],
                   help="重建方法；v6 §5.2 受控枚举只有 vggt（BA / colmap / dust3r 已废止）")
    p.add_argument("--source", default="vsi_bench", choices=["vsi_bench", "jsonl"])
    p.add_argument("--episodes-jsonl", default="")
    p.add_argument("--video-root", default="")
    p.add_argument("--recon-dir", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--question-types", default="",
                   help="逗号分隔的题型子集（默认 8 题型；用于按任务分层重建/子集实验）")
    p.add_argument("--episodes-per-scene", type=int, default=1,
                   help="每个 scene 最多抽几条 episode 的帧（默认 1：同 scene 共用同一段视频/"
                        "同一套帧，重建只需一份；0 = 不限，会白解码并吃掉大量内存）")
    p.add_argument("--sampling-per-task", type=int, default=0,
                   help="按题型均匀采样：每题型最多 N 条（0=不限）。小样本快跑用；"
                        "采样参数进 manifest，小样本不得当主表口径（HC34）")
    p.add_argument("--scene-shard", default="",
                   help="scene 分片 I/N（如 0/2、1/2）：多卡并行重建同一批 scene 时按 "
                        "scene 名哈希确定性切分")
    p.add_argument("--datasets", default="",
                   help="逗号分隔的来源数据集（scannet,scannetpp,arkitscenes；默认全部）。"
                        "限定范围会写进 manifest，不得当全量口径")
    p.add_argument("--n-frames", type=int, default=32, help="均匀采样帧数（默认 32）")
    p.add_argument("--gpus", default="", help="逗号分隔 GPU 编号（默认 0；M20 DP 用）")
    p.add_argument("--force", action="store_true", help="忽略已有 artifact 重跑")
    p.add_argument("--plan-only", action="store_true", help="只打印作业表，不执行重建")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    return p


def _maybe_moge2(args):
    """`--moge2` 时构造 MoGe-2（懒加载）；否则 None（→ scale_fusion_status=not_run）。

    v6 §11 全部为 [待实验]：默认不融合，只有显式开启才跑；构造失败直接抛错
    （静默关闭会把"米制三题全 0 分"误读成几何问题）。
    """
    if not getattr(args, "moge2", False):
        return None
    from skill3d.reconstruction.metric_fusion import make_moge2_model

    print("[info] 启用 MoGe-2 度量尺度融合（§11 [待实验]）")
    return make_moge2_model(device="cuda")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    paths = paths_from(cfg)
    recon_dir = args.recon_dir or paths.reconstructions
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()] or [0]

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    if not splits:
        print("[错误] --split 为空", file=sys.stderr)
        return 2
    if "final_test" in splits:
        print("[错误] 硬约束 9：final test 不参与候选搜索/批量重建（不进任何离线流程）。",
              file=sys.stderr)
        return 2

    items: list[EpisodeItem] = []
    for split in splits:
        try:
            if args.source == "jsonl":
                if not args.episodes_jsonl:
                    print("[错误] --source jsonl 需要 --episodes-jsonl PATH", file=sys.stderr)
                    return 2
                part = load_jsonl_items(args.episodes_jsonl, split=split,
                                        limit=args.limit or None)
            else:
                split_cfg = load_yaml(cfg.get("split_config", "configs/vsi_bench_split.yaml"))
                qtypes = [t.strip() for t in args.question_types.split(",") if t.strip()]
                dsets = [d.strip() for d in args.datasets.split(",") if d.strip()]
                part = load_vsi_bench_items(
                    split, split_cfg,
                    video_root=args.video_root or paths.raw_videos,
                    cache_dir=paths.vsi_bench_meta, limit=args.limit or None,
                    n_frames=args.n_frames,
                    question_types=qtypes or None,
                    datasets=dsets or None,
                    max_per_scene=int(args.episodes_per_scene),
                    stratified_per_task=int(args.sampling_per_task),
                )
        except EpisodeSourceError as exc:
            print(f"[错误] split={split} 数据源不可用: {exc}", file=sys.stderr)
            return 1
        items.extend(part)

    # v6 §20：官方 VGGSfM BA 与 `vggt_sparse_ba` 已整体废止（生产 hard-disable 的依据
    # 与历史码归档在 `skill3d/legacy/retired/`），CLI 不再提供任何 BA 开关。
    jobs = plan_scene_jobs(items, recon_dir, args.method, force=args.force)
    if args.scene_shard:
        import hashlib as _hl

        i_s, _, n_s = args.scene_shard.partition("/")
        shard_i, shard_n = int(i_s), int(n_s or 1)
        if shard_n > 1:
            def _bucket(name: str) -> int:
                return int(_hl.sha256(name.encode()).hexdigest()[:8], 16) % shard_n

            before = len(jobs)
            jobs = [j for j in jobs if _bucket(j.scene_name) == shard_i]
            print(f"[shard {shard_i}/{shard_n}] scenes {len(jobs)}/{before}")
    items_by_scene: dict[str, list[EpisodeItem]] = {}
    for it in items:
        items_by_scene.setdefault(it.episode.scene_name, []).append(it)

    print("=" * 78)
    print(f"P1 重建：method={args.method} splits={splits} scenes={len(jobs)} "
          f"episodes={len(items)} gpus={gpus} recon_dir={recon_dir}")
    print("=" * 78)
    if args.plan_only:
        for j in jobs:
            print(f"  {j.status:8s} {j.split:18s} {j.scene_name:28s} "
                  f"episodes={j.n_episodes} {j.note}")
        return 0

    jobs = run_jobs(jobs, items_by_scene, recon_dir, args.method, gpus=gpus,
                    metric_depth_model=_maybe_moge2(args),
                    n_frames=args.n_frames)
    manifest = Path(recon_dir) / args.method / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "method": args.method, "splits": splits, "recon_dir": recon_dir,
        "moge2_enabled": bool(getattr(args, "moge2", False)),
        "schema_version": "6.0",
        "quality_metric_version": "v6-m4-main-gate",
        "n_frames": args.n_frames,
        "dataset_scope": [d.strip() for d in args.datasets.split(",") if d.strip()] or "all",
        "sampling_per_task": int(args.sampling_per_task),
        "sampling_strategy": ("first_n_per_task_by_meta_order"
                              if args.sampling_per_task else "none"),
        "scene_shard": args.scene_shard or "none",
        "question_type_scope": [t.strip() for t in args.question_types.split(",")
                                if t.strip()] or "all",
        "n_scenes": len(jobs),
        "n_done": sum(1 for j in jobs if j.status == "done"),
        "n_skipped": sum(1 for j in jobs if j.status == "skipped"),
        "n_failed": sum(1 for j in jobs if j.status == "failed"),
        "jobs": [{"scene_name": j.scene_name, "split": j.split,
                  "n_episodes": j.n_episodes, "status": j.status,
                  "gpu_rank": j.gpu_rank, "artifact_ref": j.artifact_ref,
                  "recon_method": (j.artifact.recon_method if j.artifact else ""),
                  "quality_status": (j.artifact.quality_status if j.artifact else ""),
                  "frame_set_hash": (j.artifact.frame_set_hash if j.artifact else ""),
                  "note": j.note} for j in jobs],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for j in jobs:
        print(f"  {j.status:8s} {j.split:18s} {j.scene_name:28s} gpu={j.gpu_rank} {j.note}")
    print(f"\nmanifest: {manifest}")
    n_failed = sum(1 for j in jobs if j.status == "failed")
    if n_failed:
        print(f"[warn] {n_failed} 个 scene 重建失败：确认权重/显存/依赖（VGGT checkpoint 见 "
              "§4 M3 字段 12 TODO_USER_INPUT）", file=sys.stderr)
    return 0 if n_failed < len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
