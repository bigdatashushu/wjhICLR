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
import os
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
    p.add_argument("--direct-answer-tasks", default="",
                   help="逗号分隔的题型：这些题型改用直答 VLM 作答（任务级策略，"
                        "须在 inner 上定、outer 上验证）。留空 = 全部走程序（默认）")
    p.add_argument("--sampling-per-task", type=int, default=0,
                   help="按题型均匀采样：每题型最多 N 条（与重建批使用同一参数，保证同一样本）")
    p.add_argument("--datasets", default="",
                   help="逗号分隔的来源数据集（scannet,scannetpp,arkitscenes）；"
                        "限定范围会写进 RunManifest，报告不得冒充全量口径")
    p.add_argument("--limit", type=int, default=0, help="最多跑 N 条（0 = 不限）")
    p.add_argument("--mode", default="mock_light", choices=["real", "mock_light"])
    p.add_argument("--baseline", default="C1_tools_program",
                   choices=["C0_direct_vlm", "C1_tools_program"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic-replay", action="store_true",
                   help="重放确定性：时间/id/latency 取确定性占位，保证同 seed 字节级一致")
    p.add_argument("--active-snapshot", default="", help="active snapshot 文件或目录（§13.5）")
    p.add_argument("--skill-spec", default="",
                   help="C2 消融：直接注入手写 SkillSpec JSON（静态人工 Skill，无归纳）")
    p.add_argument("--inject-wrong-skill", action="store_true",
                   help="C5 消融：注入已知错误 Skill，观察回退/退化（可证伪 §17.5 #5）")
    p.add_argument("--vllm-endpoint", action="append", default=[],
                   help="本地 vLLM endpoint（可重复，DP×8 时给 8 个）")
    p.add_argument("--vllm-model", default="", help="served model name")
    p.add_argument("--trace-dir", default="")
    p.add_argument("--memory-dir", default="",
                   help="episodic 记忆目录（G-26；空=用 config 的 paths.memory_db 同级）")
    p.add_argument("--no-memory", action="store_true", help="关闭在线 episodic 记忆写入")
    p.add_argument("--recon-dir", default="")
    p.add_argument("--recon-method", default="vggt",
                   choices=["vggt", "vggt_sparse_ba", "dust3r_mast3r", "colmap"],
                   help="重建方法；v5 正式主线为 vggt（HC35）")
    p.add_argument("--sparse-ba", action="store_true",
                   help="启用 `vggt_sparse_ba` 限定 PoC（未过 §10.1 前启用报错，HC36）")
    p.add_argument("--scale-calibration", default="",
                   help="冻结尺度 conformal 校准器 JSON（HC32；缺省取 configs/config.yaml "
                        "的 scale.calibration_path；不存在 → scale_confidence 恒 low）")
    p.add_argument("--allow-tool-contract-replay", action="store_true",
                   help="启用 D-3 回灌恢复层（[Conditional Go]：回灌修复率 >=50%% 才启用）")
    p.add_argument("--frame-size", default="", help="合成帧尺寸 HxW（默认 480x640，与 VSI-Bench 对齐）")
    p.add_argument("--degrade", default="", choices=["", "blur_all", "blur_some",
                                                     "overexposed_all", "few_frames"],
                   help="合成输入退化注入（仅 mock_light，用于 M2 门禁验收）")
    p.add_argument("--allow-final-test", action="store_true",
                   help="硬约束 9：显式允许 final_test 进在线链（盲评一次）")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--out-meta", default="", help="可选：把本次 episode meta 落盘为 jsonl")
    p.add_argument("--run-manifest", default="",
                   help="RunManifest 落盘路径（§16.4 复现 checklist；默认 data/run_manifest.json）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # M5 检测器兜底地址：config 里配了就用它（除非环境变量已显式指定）。
    # 纪律：检测器是确定性 Tool（非 LLM），只产出 SAM2 的 box prompt，不参与出答案。
    try:
        _det = (load_config(args.config).get("detection") or {}).get("endpoint")
        if _det and not os.environ.get("SKILL3D_DETECTOR_ENDPOINT"):
            os.environ["SKILL3D_DETECTOR_ENDPOINT"] = str(_det)
    except Exception:  # noqa: BLE001 - 配置缺失不阻断
        pass
    # v5 HC35/36：生产禁用的两条 BA 路径在入口 fail-closed
    from skill3d.reconstruction.legacy_vggsfm_ba import UnsupportedConfigurationError

    if args.recon_method == "vggt_sparse_ba" or args.sparse_ba:
        print("[错误] `vggt_sparse_ba` 尚未通过 §10.1 L0→L1→L2，禁止启用（HC36）；"
              "正式主线继续用 vggt。", file=sys.stderr)
        return 2
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

    # ---- 消融注入（C2 静态人工 Skill / C5 已知错误 Skill，§16.1）----
    try:
        skills, extra_ref = _apply_skill_ablations(args, skills)
    except Exception as exc:  # noqa: BLE001 - spec 非法 → 明确报错退出
        print(f"[错误] Skill 消融注入失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if extra_ref:
        snapshot_ref = f"{snapshot_ref}+{extra_ref}"
    if args.skill_spec or args.inject_wrong_skill:
        print(f"[消融] C2/C5 Skill 注入：共 {len(skills)} 条（"
              f"{'含已知错误 Skill' if args.inject_wrong_skill else '静态人工 Skill'}）")

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
                # v5 修复：vsi_bench 源此前**静默忽略** --question-types（分层子集实验
                # 会误跑成全量）→ 现在如实传递过滤条件。
                question_types=qtypes or None,
                datasets=[d.strip() for d in args.datasets.split(",") if d.strip()] or None,
                stratified_per_task=int(args.sampling_per_task),
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
        memory_dir=("" if args.no_memory
                    else (args.memory_dir or str(Path(paths.memory_db).parent
                                                 / "memory_episodic"))),
        recon_dir=args.recon_dir or paths.reconstructions,
        vllm_endpoints=list(args.vllm_endpoint),
        vllm_model=args.vllm_model or vllm.model,
        recon_method=args.recon_method,
        direct_answer_tasks={t.strip() for t in args.direct_answer_tasks.split(",")
                             if t.strip()},
        ba_enabled=False,   # v5 HC35/36：正式评测不带任何 BA（启用请求在 CLI 处报错）
        # v4 尺度（HC29–33）：冻结校准器路径 + nominal coverage（在线只读）
        scale_calibration_path=(args.scale_calibration
                                or (cfg_yaml.get("scale") or {}).get("calibration_path")
                                or None),
        # v5.1：多校准器目录 + 被评测数据集（按数据集选同源冻结校准器）
        scale_calibration_dir=str((cfg_yaml.get("scale") or {}).get("calibration_dir") or ""),
        evaluation_datasets=list((cfg_yaml.get("scale") or {}).get("evaluation_datasets") or []),
        scale_confidence_level=float(
            (cfg_yaml.get("scale") or {}).get("confidence_level", 0.90) or 0.90),
        enable_tool_contract_replay=bool(args.allow_tool_contract_replay),
        max_pixels=int(getattr(vllm, "max_pixels", 131072) or 131072),
        max_model_len=int(getattr(vllm, "max_model_len", 32768) or 32768),
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
        levels = slot.get("levels") or {}
        extra = ""
        if len(levels) > 1:
            extra = "  档位=" + ", ".join(
                f"{k.split('object_rel_direction_')[-1]}:{_fmt(v['accuracy'])}"
                for k, v in sorted(levels.items()))
        print(f"  {qt:22s} n={slot['n']}  acc={_fmt(slot['accuracy'])}  "
              f"mra={_fmt(slot['mra'])}{extra}")

    # ---- §8.3 官方主表口径：Avg + 4×NA(MRA) + 4×MCA(Acc)，全 ×100 ----
    try:
        from skill3d.evaluation.multi_seed_aggregator import (
            format_main_table_row,
            macro_average_over_tasks,
        )

        macro = macro_average_over_tasks(run.per_task)
        print("\n主表口径（§8.3：8 任务简单算术平均 ×100，无加权）:")
        print("  " + format_main_table_row(args.baseline, run.per_task))
        if macro["missing_tasks"]:
            print(f"  [note] 本 run 未覆盖任务（不计入平均，未臆造 0）: "
                  f"{macro['missing_tasks']}")
        if macro["level_detail_x100"]:
            print(f"  [note] 档位明细（等权聚合前）: {macro['level_detail_x100']}")
    except Exception as exc:  # noqa: BLE001 - 汇总失败不阻断评测
        print(f"[warn] 主表口径汇总失败: {type(exc).__name__}: {exc}", file=sys.stderr)
    print("\nepisodes:")
    for o in outcomes:
        print(f"  {o.qa_id:34s} {o.question_type:20s} {o.final_state:20s} "
              f"answer={str(o.answer)[:16]:18s} predicted={str(o.predicted):10s} "
              f"correct={str(o.correct):5s} mra={_fmt(o.mra_value)}")
    print(f"\ntrace 已写: {run_cfg.trace_dir}（episode_trace / program_trace / "
          f"geometry_check / evaluation_result / evaluation_run / online_run）")
    if run_cfg.memory_dir:
        print(f"episodic 记忆: {run_cfg.memory_dir}（G-26；semantic 不在线写，硬约束 1/2）")
    print(f"synthesis_source 分布: "
          f"{ {s: sum(1 for o in outcomes if o.synthesis_source == s) for s in set(o.synthesis_source for o in outcomes)} }")

    # ---- 系统可靠性小节（§4 M6 字段 9 / §16.2：与主表分离报告，不混主表）----
    try:
        from skill3d.evaluation.process_metrics import aggregate_process_metrics

        pm = aggregate_process_metrics(outcomes)
        print("\n系统可靠性（不混主表）:")
        print(f"  coverage={_fmt(pm.coverage)}  refusal_rate={_fmt(pm.refusal_rate)}  "
              f"abstain_rate={_fmt(pm.abstain_rate)}")
        print(f"  coverage-conditioned: MRA={_fmt(pm.coverage_conditioned_mra)}  "
              f"Acc={_fmt(pm.coverage_conditioned_accuracy)}")
        print(f"  tool_contract: episode 率={_fmt(pm.tool_contract_rate)}  "
              f"调用数={pm.tool_contract_calls}  恢复 episode={pm.tool_contract_recovered}")
        for n in pm.notes:
            print(f"  [note] {n}")
    except Exception as exc:  # noqa: BLE001 - 统计是附加产物，不阻断评测
        print(f"[warn] 过程指标聚合失败: {type(exc).__name__}: {exc}", file=sys.stderr)

    # ---- v4 尺度能力小节（§7：逐题型 MRA/refusal/coverage + 锚点/冲突率，不混主表）----
    try:
        from skill3d.evaluation.scale_report import build_scale_report, write_scale_report

        sr = build_scale_report(outcomes)
        print("\n尺度能力（不混主表；HC29–33）:")
        for line in sr.format_lines()[1:]:
            print(line)
        sr_out = (str(Path(run_cfg.trace_dir) / "scale_report.json")
                  if run_cfg.trace_dir else "")
        if sr_out:
            print(f"  尺度报告已写: {write_scale_report(sr, sr_out)}")
    except Exception as exc:  # noqa: BLE001 - 附加产物，不阻断评测
        print(f"[warn] 尺度报告生成失败: {type(exc).__name__}: {exc}", file=sys.stderr)

    # ---- RunManifest（G-67/§16.4 + §7：代码/环境/split/seed/推理参数全记录）----
    manifest_path = _write_manifest(args, cfg_yaml, run, run_cfg, split,
                                    outcomes=outcomes, items=items)
    print(f"RunManifest: {manifest_path}（code_commit / pip_freeze_hash / "
          f"checkpoint_sha256 / split_version / seed / inference_env）")
    return 0


def _frame_set_summary(items) -> tuple[str, int]:
    """本次 run 的帧集身份：全一致 → 该 hash；否则 multi:<digest> + 去重个数。

    硬约束 21 要求全链共用同一 FrameSet；一 run 内出现多个 hash 只可能来自
    多 scene（每个 scene 一段视频，各自 32 帧），此时记录去重后的聚合摘要。
    """
    import hashlib as _h

    hashes = []
    for it in items or []:
        fs = getattr(it.episode, "frame_set", None)
        if fs is not None and fs.frame_set_hash:
            hashes.append(fs.frame_set_hash)
    uniq = sorted(set(hashes))
    if not uniq:
        return "", 0
    if len(uniq) == 1:
        return uniq[0], 1
    return "multi:" + _h.sha256("|".join(uniq).encode()).hexdigest()[:16], len(uniq)


def _write_manifest(args, cfg_yaml: dict, run, run_cfg, split: str, *,
                    outcomes=None, items=None):
    """写 RunManifest（失败不阻断评测：复现信息是附加产物）。"""
    from skill3d.infra.version_lock import build_run_manifest, write_run_manifest

    try:
        from skill3d.adapters.split_builder import load_split_config

        split_cfg_path = cfg_yaml.get("split_config", "configs/vsi_bench_split.yaml")
        split_version = load_split_config(split_cfg_path).split_version
    except Exception:  # noqa: BLE001
        split_cfg_path, split_version = "", ""
    try:
        m = build_run_manifest(
            docker_image=str((cfg_yaml.get("sandbox") or {}).get("image", "")),
            checkpoint_path=str((cfg_yaml.get("vllm") or {}).get("model", "")),
            config_path=args.config, repo_dir=".",
            split_version=split_version, seed=args.seed,
            split_config_path=split_cfg_path)
        out = args.run_manifest or "data/run_manifest.json"
        vllm_cfg = cfg_yaml.get("vllm") or {}
        fsh, n_fsh = _frame_set_summary(items or [])
        scale_sources = sorted({str(getattr(o, "scale_source", "") or "")
                                for o in (outcomes or [])} - {""})
        cal_ids, cal_hashes, conf_dist, allowed_tasks = _scale_manifest_facts(
            outcomes, run_cfg)
        from skill3d.evaluation.golden_v5 import golden_versions
        from skill3d.reconstruction_gate.quality_metrics import (
            QUALITY_METRIC_VERSION,
        )
        from skill3d.reconstruction.sparse_ba import sparse_ba_enabled

        return write_run_manifest(m, out, extra={
            # ---- v5 版本三元组（HC39：schema/质量口径/golden 必须同时留档）----
            "schema_version": "5.0",
            "quality_metric_version": QUALITY_METRIC_VERSION,
            **golden_versions(),
            # ---- v5 HC35–37：主线与 BA 状态（全部默认关闭，启用即报错）----
            "recon_method": str(run_cfg.recon_method),
            "sparse_ba_enabled": bool(sparse_ba_enabled()),
            "sparse_ba_frontend": "",
            "sparse_ba_pair_graph_hash": "",
            "official_vggsfm_ba_enabled": False,
            "reprojection_status": (
                "computed" if any(
                    str(getattr(o, "reprojection_status", "")) == "computed"
                    for o in (outcomes or [])) else "not_available"),
            "dataset_scope": [d.strip() for d in args.datasets.split(",")
                              if d.strip()] or "all",
            "question_type_scope": ([q.strip() for q in args.question_types.split(",")
                                     if q.strip()] or "all"),
            "direct_answer_tasks": sorted(t.strip()
                                          for t in args.direct_answer_tasks.split(",")
                                          if t.strip()),
            "sampling_per_task": int(args.sampling_per_task),
            "sampling_strategy": ("first_n_per_task_by_meta_order"
                                  if args.sampling_per_task else "none"),
            "gpt6_latency_stats": {},   # 在线链无 GPT-6（HC1）；离线链在此登记
            "run_id": run.run_id, "split": split, "mode": args.mode,
            "baseline": args.baseline, "n_episodes": run.n_episodes,
            "active_snapshot_ref": run.active_snapshot_ref,
            # 影响结果的推理参数必须记录（§16.4 / §7：冻结模型 + 一致推理环境）
            "vllm_model": run_cfg.vllm_model,
            "n_frames": vllm_cfg.get("n_frames"),
            "max_pixels": vllm_cfg.get("max_pixels"),
            "max_model_len": vllm_cfg.get("max_model_len"),
            "vllm_endpoints": list(run_cfg.vllm_endpoints),
            # §7 / D-6 / A-8 / D-2：帧集身份、BA 开关、尺度来源
            "frame_set_hash": fsh,
            "n_distinct_frame_sets": n_fsh,
            "ba_enabled": bool(getattr(run_cfg, "ba_enabled", False)),
            "scale_source": ",".join(scale_sources),
            # ---- v4 HC29–34：尺度校准复现三件套（§7 明文要求留档）----
            "scale_calibration_id": ",".join(cal_ids),
            "evaluation_datasets": list(run_cfg.evaluation_datasets),
            "calibration_split_hash": cal_hashes.get("calibration_split_hash", ""),
            "excluded_vsibench_scene_hash":
                cal_hashes.get("excluded_vsibench_scene_hash", ""),
            "confidence_level": float(getattr(run_cfg, "scale_confidence_level", 0.0) or 0.0),
            "scale_confidence_distribution": conf_dist,
            "allowed_metric_tasks_observed": sorted(allowed_tasks),
            "readiness_manifest_ref": str(
                (cfg_yaml.get("readiness") or {}).get("manifest_path", ""))})
    except Exception as exc:  # noqa: BLE001
        return f"(写失败: {type(exc).__name__}: {exc})"


def _scale_manifest_facts(outcomes, run_cfg) -> tuple[list[str], dict, dict, list[str]]:
    """从 outcomes 与冻结校准器提取 v4 复现事实（§7 / HC32）。

    返回 `(calibration_ids, split_hashes, confidence_distribution, allowed_tasks)`。
    `split_hashes` 来自**冻结校准器自带的审计清单**（不是临时重算），
    保证记录的是本次实际使用的那份校准器的隔离证据。
    """
    cal_ids = sorted({str(getattr(o, "scale_calibration_id", "") or "")
                      for o in outcomes or []} - {""})
    conf_dist: dict = {}
    for o in outcomes or []:
        c = str(getattr(o, "scale_confidence", "low") or "low")
        conf_dist[c] = conf_dist.get(c, 0) + 1
    allowed = {t for o in outcomes or []
               for t in (getattr(o, "authorized_metric_tasks", None) or [])}
    hashes: dict = {}
    if cal_ids:
        try:
            from skill3d.reconstruction.scale_calibration import load_calibrator

            cal = load_calibrator(getattr(run_cfg, "scale_calibration_path", None))
            if cal.split_audit is not None:
                hashes = {"calibration_split_hash": cal.split_audit.calibration_split_hash,
                          "excluded_vsibench_scene_hash":
                              cal.split_audit.excluded_vsibench_scene_hash}
        except Exception:  # noqa: BLE001 - 校准器不可用时有 cal_id 却没有哈希
            hashes = {"calibration_split_hash": "unavailable",
                      "excluded_vsibench_scene_hash": "unavailable"}
    return cal_ids, hashes, conf_dist, sorted(allowed)


def _apply_skill_ablations(args, skills: list) -> tuple[list, str]:
    """C2（--skill-spec）/ C5（--inject-wrong-skill）消融注入。

    C5 的"已知错误 Skill"是本系统内置的**故意错误**模板（调用不存在的 Tool 名称 +
    与题型不符的模板），用于验证：错误 Skill 应被 M9/AST 或 M11/T4 或 retrieval 硬过滤
    拦住、至少不提升分数（§17.5 #5 可证伪命题）。
    """
    from skill3d.schemas import SkillSpec

    out = list(skills)
    ref = ""
    if args.skill_spec:
        spec = SkillSpec.model_validate_json(Path(args.skill_spec).read_text(encoding="utf-8"))
        out = [s for s in out if s.skill_id != spec.skill_id] + [spec]
        ref = f"static:{spec.skill_id}@{spec.semver}"
    if args.inject_wrong_skill:
        wrong = SkillSpec(
            skill_id="sk-known-wrong", semver="0.0.1",
            task_type="object_counting",
            description="【C5 消融】已知错误模板：调用不存在的 Tool 并返回常数",
            call_graph_template=(
                "answer = definitely_not_a_registered_tool(1, 2)\n"
                "ReturnAnswer(\"42\")"),
            requires_artifacts=["nonexistent_artifact"],
            minimum_quality=0.0,
            supported_coordinate_frames=["world"],
            metric_scale_required=False,
            validation_assertions=["never_true()"],
        )
        out = [s for s in out if s.skill_id != wrong.skill_id] + [wrong]
        ref = (ref + "+" if ref else "") + "wrong:sk-known-wrong@0.0.1"
    return out, ref


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


if __name__ == "__main__":
    raise SystemExit(main())
