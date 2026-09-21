"""P1 批量重建 CLI（§13.5）：

```bash
python -m skill3d.reconstruction.run --split induction,inner,outer --method vggt
```

- **v5 HC35**：正式主线固定 `vggt` feed-forward；官方 `VGGSfM tracker + PyCOLMAP BA`
  已在 24 GiB 环境被真实 OOM 证据否决，生产启用请求直接报
  `UnsupportedConfigurationError`（`reconstruction/legacy_vggsfm_ba/`）；唯一允许的 BA
  候选 `vggt_sparse_ba` 默认关闭（未过 §10.1 PoC 前启用同样报错）；
- 重建以 **scene** 为单位（同 scene 的多个 episode 共享一次重建，§4 M3）；
- 已存在 artifact 的 scene 默认跳过（断点恢复幂等，§4 M21 字段 7/8）；
- DP 分配用 M20 `GPUScheduler`（每卡一个 scene，paired 同卡规则不影响本阶段）；
- **诚实性**：本命令只跑真实重建（vggt / dust3r_mast3r / colmap）。合成几何只允许经
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
             n_frames: int = 32, use_ba: bool = False) -> list[SceneJob]:
    """执行场景作业（DP 逻辑分配；重建本身按 §4 M3 主线/降级链）。

    **M4 前移 P1（方案 X，§2.2）**：重建时顺手算 G1–G11 并写回 artifact，
    quality 随 artifact 落盘，P2 加载即得实算值、零重算。
    `use_ba` → BA route（[Conditional Go]，§10.1），不可用时自动回退 feed-forward。
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
                use_ba=use_ba,
                frame_set=scene_items[0].episode.frame_set if scene_items else None,
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
        # run_vggt 已在内部算过（零重算）；其余方法（colmap / dust3r）在此补齐。
        try:
            from skill3d.reconstruction_gate import quality_metrics as qm

            if not qm.quality_is_computed(art):
                art = qm.compute_and_store_quality(
                    art, frames=frames,
                    depth_maps=_load_npy(art.depth_maps),
                    c2w_list=_load_npy(art.c2w_list),
                    reproj_errors=_load_npy(art.reproj_errors),
                    artifact_path=str(out))
        except Exception as exc:  # noqa: BLE001 - 质量算不出不阻断重建（P2 会按 fail-closed 降级）
            job.note = f"[warn] M4 质量写回失败: {type(exc).__name__}: {exc}"
        out.write_text(art.model_dump_json(indent=2), encoding="utf-8")
        job.status, job.artifact, job.artifact_ref = "done", art, str(out)
        job.note = (f"method={art.recon_method} scale_known={art.scale_known} "
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
        description="P1 批量 3D 重建（M3）：按 scene 跑 VGGT/DUSt3R/COLMAP 并落 artifact",
    )
    p.add_argument("--split", default="induction,inner_validation,outer_holdout",
                   help="逗号分隔的 split 列表（默认 induction,inner_validation,outer_holdout）")
    p.add_argument("--method", default="vggt",
                   choices=["vggt", "vggt_sparse_ba", "dust3r_mast3r", "colmap"],
                   help="重建方法；v5 正式主线为 vggt（HC35）")
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
    p.add_argument("--sparse-ba", action="store_true",
                   help="启用 `vggt_sparse_ba` 限定 PoC（默认关闭；未过 §10.1 前启用报错，HC36）")
    p.add_argument("--allow-official-ba-repro", action="store_true",
                   help="[仅复现] 允许跑官方 VGGSfM BA 以重放 OOM 失败证据（HC35：不得用于生产）")
    p.add_argument("--plan-only", action="store_true", help="只打印作业表，不执行重建")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    return p


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

    # v5 HC35/36：生产禁用的两条 BA 路径在 CLI 入口就 fail-closed（不落到 worker 里才报错）
    from skill3d.reconstruction.legacy_vggsfm_ba import (
        UnsupportedConfigurationError,
        assert_official_ba_disabled,
    )

    try:
        assert_official_ba_disabled(
            enable_official_vggsfm_ba=bool(args.allow_official_ba_repro),
            context="reconstruction.run CLI")
    except UnsupportedConfigurationError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
    if args.method == "vggt_sparse_ba" or args.sparse_ba:
        print("[错误] `vggt_sparse_ba` 尚未通过 §10.1 L0→L1→L2，禁止启用"
              "（HC36：一次性 PoC 止损纪律；正式主线继续用 vggt）。", file=sys.stderr)
        return 2

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
                    n_frames=args.n_frames,
                    # v5：正式主线不带 BA（上面已 fail-closed 掉所有启用请求）
                    use_ba=False)
    manifest = Path(recon_dir) / args.method / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "method": args.method, "splits": splits, "recon_dir": recon_dir,
        "schema_version": "5.0",
        "quality_metric_version": "v5-no-g8-g5-optional",
        # v5 HC35/36：官方 BA 与 sparse BA 均为 False（任何启用都在 CLI 处报错）
        "ba_enabled": False,
        "official_vggsfm_ba_enabled": False,
        "sparse_ba_enabled": False,
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
