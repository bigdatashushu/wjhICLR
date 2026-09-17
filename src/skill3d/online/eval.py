"""在线评测 CLI（§13.5）：

```bash
python -m skill3d.online.eval --split test --active-snapshot data/active_snapshot.json
```

- 数据源 `--source`：`synthetic`（合成，管道验证）| `vsi_bench`（HF meta + 原始视频，
  `TODO_USER_INPUT`）| `jsonl`（本系统约定格式的预抽帧清单）；
- 两种运行模式 `--mode`：`real`（真实 M1–M13）| `mock_light`（合成输入 + 确定性 stub
  program，仅管道验证，§9.2）；
- 两档 baseline `--baseline`：`C0_direct_vlm`（无 Tool 直答）| `C1_tools_program`（§16.1）；
- 硬约束 9：`final_test`（或 `--split test`）需显式 `--allow-final-test`，且仅应盲评一次；
- 硬约束 1：本模块在线，禁止任何 GPT-6 调用。

结果：控制台汇总 + `EvaluationRun` 落 TraceStore（§5.7）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from skill3d.adapters.episode_source import (
    ALL_QUESTION_TYPES,
    EpisodeSourceError,
    load_jsonl_items,
    load_synthetic_items,
    load_vsi_bench_items,
    write_episode_meta_jsonl,
)
from skill3d.online.config import (
    DEFAULT_CONFIG,
    load_config,
    load_yaml,
    paths_from,
    sandbox_from,
    vllm_from,
)
from skill3d.online.runner import OnlineRunConfig, run_split
from skill3d.skills.registry import load_active_skills

# §13.5 用 `--split test`；本系统的四层切分用 induction/inner/outer/final_test
_SPLIT_ALIASES = {"test": "final_test", "final": "final_test"}

BANNER_MOCK = (
    "⚠ mode=mock_light：使用 online/synthetic.py 合成输入与确定性 stub program，"
    "仅用于管道验证，不构成任何精度结论（§9.2 / §12 M0-M1 验收口径）"
)
BANNER_FINAL = "⚠ final_test：按硬约束 9 完全隔离、仅应盲评一次；本次结果请单独留档"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m skill3d.online.eval",
        description="Skill3D 在线评测（M1-M13）：跑 split → EvaluationRun + EpisodeTrace",
    )
    p.add_argument("--split", default="test",
                   choices=["test", "induction", "inner_validation", "outer_holdout",
                            "final_test", "final"])
    p.add_argument("--source", default="synthetic", choices=["synthetic", "vsi_bench", "jsonl"])
    p.add_argument("--episodes-jsonl", default="", help="--source jsonl 时的清单路径")
    p.add_argument("--video-root", default="", help="--source vsi_bench 时的原始视频根目录")
    p.add_argument("--question-types", default="",
                   help="逗号分隔的题型子集（默认 8 题型）")
    p.add_argument("--limit", type=int, default=0, help="最多跑 N 条（0 = 不限）")
    p.add_argument("--mode", default="mock_light", choices=["real", "mock_light"])
    p.add_argument("--baseline", default="C1_tools_program",
                   choices=["C0_direct_vlm", "C1_tools_program"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic-replay", action="store_true",
                   help="重放确定性：时间/id/latency 取确定性占位，保证同 seed 字节级一致")
    p.add_argument("--active-snapshot", default="", help="active snapshot 文件或目录（§13.5）")
    p.add_argument("--vllm-endpoint", action="append", default=[],
                   help="本地 vLLM endpoint（可重复，DP×8 时给 8 个）")
    p.add_argument("--vllm-model", default="", help="served model name")
    p.add_argument("--trace-dir", default="")
    p.add_argument("--recon-dir", default="")
    p.add_argument("--recon-method", default="vggt", choices=["vggt", "dust3r_mast3r", "colmap"])
    p.add_argument("--frame-size", default="", help="合成帧尺寸 HxW（默认 480x640，与 VSI-Bench 对齐）")
    p.add_argument("--degrade", default="", choices=["", "blur_all", "blur_some",
                                                     "overexposed_all", "few_frames"],
                   help="合成输入退化注入（仅 mock_light，用于 M2 门禁验收）")
    p.add_argument("--allow-final-test", action="store_true",
                   help="硬约束 9：显式允许 final_test 进在线链（盲评一次）")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out-meta", default="", help="可选：把本次 episode meta 落盘为 jsonl")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_yaml = load_config(args.config)
    paths = paths_from(cfg_yaml)
    vllm = vllm_from(cfg_yaml)
    sandbox = sandbox_from(cfg_yaml)

    split = _SPLIT_ALIASES.get(args.split, args.split)
    if split == "final_test" and not args.allow_final_test:
        print("[错误] split=final_test 被硬约束 9 拦截：final test 完全隔离、仅盲评一次。\n"
              "       确认要盲评请显式加 --allow-final-test（结果单独留档）。", file=sys.stderr)
        return 2

    if args.mode == "real" and args.source == "synthetic":
        print("[错误] mode=real 需要真实数据：请用 --source vsi_bench 或 --source jsonl。\n"
              "       合成数据只允许配 mode=mock_light（§9.2）。", file=sys.stderr)
        return 2

    # ---- active snapshot → 在线只读的 SkillSpec（M15，硬约束 12）----
    snapshot_ref = "genesis"
    skills, warnings = [], []
    snap_arg = args.active_snapshot or paths.active_snapshot
    if snap_arg:
        skills, warnings, snapshot_ref = load_active_skills(snap_arg)
    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)

    # ---- 数据源 ----
    qtypes = [q.strip() for q in args.question_types.split(",") if q.strip()]
    limit = args.limit or None
    try:
        if args.source == "synthetic":
            frame_size = (480, 640)
            if args.frame_size:
                h, _, w = args.frame_size.partition("x")
                frame_size = (int(h), int(w))
            items = load_synthetic_items(
                split, question_types=qtypes or None, limit=limit, seed=args.seed,
                frame_size=frame_size, degrade=args.degrade or None,
                out_dir=str(Path(paths.reconstructions) / "mock_light"),
            )
        elif args.source == "jsonl":
            if not args.episodes_jsonl:
                print("[错误] --source jsonl 需要 --episodes-jsonl PATH", file=sys.stderr)
                return 2
            items = load_jsonl_items(args.episodes_jsonl, split=split, limit=limit)
        else:
            split_cfg = load_yaml(cfg_yaml.get("split_config", "configs/vsi_bench_split.yaml"))
            items = load_vsi_bench_items(
                split, split_cfg,
                video_root=args.video_root or paths.raw_videos,
                cache_dir=paths.vsi_bench_meta, limit=limit, seed=args.seed,
            )
    except EpisodeSourceError as exc:
        print(f"[错误] 数据源不可用: {exc}", file=sys.stderr)
        return 1

    if args.out_meta:
        print(f"[info] episode meta 已写: {write_episode_meta_jsonl(items, args.out_meta)}")

    # ---- 运行配置 ----
    run_cfg = OnlineRunConfig(
        mode=args.mode,
        baseline=args.baseline,
        seed=args.seed,
        deterministic_replay=args.deterministic_replay,
        active_snapshot_ref=snapshot_ref,
        skills=skills,
        max_regen=int(sandbox.max_regenerate),
        cell_timeout_s=int(sandbox.cell_timeout_s),
        trace_dir=args.trace_dir or paths.trace_store,
        recon_dir=args.recon_dir or paths.reconstructions,
        vllm_endpoints=list(args.vllm_endpoint),
        vllm_model=args.vllm_model or vllm.model,
        recon_method=args.recon_method,
        allow_final_test=args.allow_final_test,
    )
    if not run_cfg.vllm_endpoints and args.mode == "real":
        print("[warn] mode=real 但未给 --vllm-endpoint：program 生不出来的 episode 会记 "
              "unavailable（§4 M8 字段 9）。离线起服务：bash scripts/serve_qwen3vl_dp8.sh",
              file=sys.stderr)

    print("=" * 78)
    print(f"split={split} source={args.source} n_episodes={len(items)} mode={args.mode} "
          f"baseline={args.baseline} active_snapshot={snapshot_ref} "
          f"skills={len(skills)}")
    if args.mode != "real" or args.source == "synthetic":
        print(BANNER_MOCK if args.mode != "real" else "")
    if split == "final_test":
        print(BANNER_FINAL)
    print("=" * 78)

    outcomes, run = run_split(items, run_cfg)

    print(f"\nEvaluationRun  run_id={run.run_id}  split={run.split}  n={run.n_episodes}")
    print(f"  accuracy(MCA) = {_fmt(run.accuracy)}   mra(NA) = {_fmt(run.mra)}")
    print(f"  code_commit   = {run.code_commit}")
    print("\nper-task:")
    for qt in ALL_QUESTION_TYPES:
        slot = run.per_task.get(qt)
        if not slot:
            continue
        print(f"  {qt:22s} n={slot['n']}  acc={_fmt(slot['accuracy'])}  mra={_fmt(slot['mra'])}")
    print("\nepisodes:")
    for o in outcomes:
        print(f"  {o.qa_id:34s} {o.question_type:20s} {o.final_state:20s} "
              f"answer={str(o.answer)[:16]:18s} predicted={str(o.predicted):10s} "
              f"correct={str(o.correct):5s} mra={_fmt(o.mra_value)}")
    print(f"\ntrace 已写: {run_cfg.trace_dir}（episode_trace / program_trace / "
          f"geometry_check / evaluation_result / evaluation_run / online_run）")
    print(f"synthesis_source 分布: "
          f"{ {s: sum(1 for o in outcomes if o.synthesis_source == s) for s in set(o.synthesis_source for o in outcomes)} }")
    return 0


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


if __name__ == "__main__":
    raise SystemExit(main())
