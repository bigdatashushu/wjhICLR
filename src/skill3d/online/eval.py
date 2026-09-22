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
- 硬约束 1：本模块在线，禁止任何离线强模型调用（v6 §3.3/§20：离线治理模型为
  DeepSeek-V4.1-Flash，仅离线；RunManifest 只**登记**其冻结配置，不发起调用）。

结果：控制台汇总 + `EvaluationRun` 落 TraceStore（§5.7）+ RunManifest（§19.2）。
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
from skill3d.online.runner import (
    MAX_RECOVERY_ATTEMPTS,
    OnlineRunConfig,
    run_split,
)
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
    # v6 §11：零样本度量深度跨帧融合（首个 PoC = MoGe-2）。
    # 默认**关闭**（metric_scale=None + scale_fusion_status="not_run"）——§11 全部
    # 为 [待实验]，只有显式开启才跑，避免把 PoC 数字混进默认口径。
    p.add_argument("--moge2", action="store_true",
                   help="启用 MoGe-2 度量尺度融合（§11.2；需本机有 moge 包与权重）")
    p.add_argument("--moge2-checkpoint", default="",
                   help="覆盖 MoGe-2 权重（HF repo id 或本地目录）")
    p.add_argument("--recon-method", default="vggt", choices=["vggt"],
                   help="重建方法；v6 §5.2 受控枚举只有 vggt（BA 路线与 colmap/dust3r "
                        "对照基线均已废止并入 legacy/retired）")
    p.add_argument("--max-recovery", type=int, default=MAX_RECOVERY_ATTEMPTS,
                   help=("partial_tool_recovery 的最大恢复次数（v6 §14，[TODO_CALIBRATE]）；"
                         "超限即显式 abstain / direct_vlm_routed。"
                         "v5 的 --allow-tool-contract-replay 开关已随两档阶梯废止"))
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


def _maybe_moge2(args):
    """`--moge2` 时构造 MoGe-2（懒加载）；否则 None（→ scale_fusion_status=not_run）。

    构造失败**直接抛错**而不是静默关闭融合：静默关闭会把"米制三题全 0 分"
    误读成几何/工具的问题，而实际是模型没加载上。
    """
    if not getattr(args, "moge2", False):
        return None
    from skill3d.reconstruction.metric_fusion import make_moge2_model

    ckpt = getattr(args, "moge2_checkpoint", "") or None
    print(f"[info] 启用 MoGe-2 度量尺度融合（§11 [待实验]）；checkpoint={ckpt or '默认'}")
    return make_moge2_model(device="cuda", checkpoint=ckpt)


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
    # v6 §5.2：`recon_method` 受控枚举只有 `vggt`；旧 BA / colmap / dust3r 路线
    # 已整体废止（§20），CLI 不再提供开关，也不再 import 已归档模块。
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
        metric_depth_model=_maybe_moge2(args),
        direct_answer_tasks={t.strip() for t in args.direct_answer_tasks.split(",")
                             if t.strip()},
        # v6 §20：BA（官方 VGGSfM / vggt_sparse_ba）与整套"需校准的尺度"路线已废止，
        # 相关配置项（ba_enabled / scale_calibration_* / scale_confidence_level）不再传入。
        # v6 §14/D7：partial_tool_recovery 恒开（次数由 --max-recovery 约束），
        # v5 的 --allow-tool-contract-replay 开关随"回灌/裁剪"两档阶梯一并废止。
        max_recovery=int(args.max_recovery),
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

    # ---- v6 §20：v4/v5 尺度能力小节（scale_report / 校准池诊断）随校准路线一并废止，
    #      米制尺度证据改由 reconstruction/metric_fusion.py 的逐帧 receipt 承载。----

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


def v6_version_fields() -> dict:
    """§19.2 v6 版本字段（单一定义点：在线评测与离线演进两条链共用同一份口径）。"""
    from skill3d.reconstruction.metric_fusion import (
        METRIC_FUSION_VERSION,
        METRIC_MODEL_NONE,
    )
    from skill3d.schemas.evidence import GATE_VERSION, PROFILE_VERSION
    from skill3d.tools.distance_primitives import DistancePrimitiveParams
    from skill3d.tools.registry import TOOL_FACE_VERSION
    from skill3d.synthesis.prompt_builder import TEMPLATE_VERSION

    return {
        "template_version": TEMPLATE_VERSION,
        "tool_face_version": TOOL_FACE_VERSION,
        "evidence_profile_version": PROFILE_VERSION,
        "gate_version": GATE_VERSION,
        "metric_model": METRIC_MODEL_NONE,          # 未跑融合时如实标 none
        "metric_fusion_version": METRIC_FUSION_VERSION,
        # 距离原语冻结参数（quantile_q / voxel_size / conf_warp_version / n_min）
        "distance_primitive_params": DistancePrimitiveParams().snapshot(),
    }


# manifest 里绝不允许出现的键/值（§3.4：密钥、Authorization 头、完整环境变量、
# base URL 本体——只允许 endpoint_hash）。注意 `request_params.endpoint` 只含
# 哈希与路径，不在禁止之列。
_FORBIDDEN_MANIFEST_KEYS = frozenset({
    "api_key", "apikey", "authorization", "auth_header", "bearer",
    "env", "environ", "base_url",
})


def offline_manifest_fields(offline_client=None) -> dict:
    """§19.2「离线治理模型字段」：`offline_model`/`provider`/`model_id`/`endpoint_hash`/
    `prompt_version`/`latency`/`token_usage`。

    - **给了离线客户端**（`DeepSeekClient` 或任何提供 `manifest_fields()` 的对象）→
      直接取其元数据（只含形状 / 呼号哈希 / 延迟 / token 计数），`invoked=true`；
    - **没给**（在线链的常态：硬约束 1 禁止在线链接触离线治理模型）→ 如实登记
      `invoked=false` + 空字段，**不虚构**任何值，也绝不"用 mock 顶上"。

    两条路径都经 `_strip_secrets` 清洗：`api_key` / `authorization` / `base_url` 本体
    都不落盘（只保留 `endpoint_hash`）。

    注意：本模块**不得 import `skill3d.governance`**（`tests/unit/test_no_gpt6_online.py`
    的硬约束 1 静态守卫），故这里只做 duck-typing，不引用 `DeepSeekClient` 类型。
    """
    if offline_client is not None:
        if hasattr(offline_client, "manifest_fields"):
            fields = dict(offline_client.manifest_fields())
        else:
            # 替身 / 自定义 transport：只登记它**自报**的身份，不虚构延迟与用量
            fields = {
                "offline_model": str(getattr(offline_client, "offline_model_name", "") or ""),
                "provider": str(getattr(offline_client, "provider", "") or ""),
                "model_id": str(getattr(offline_client, "model_id", "") or ""),
                "endpoint_hash": str(getattr(offline_client, "endpoint_hash", "") or ""),
                "prompt_version": str(getattr(offline_client, "prompt_version", "") or ""),
                "latency": {}, "token_usage": {},
            }
        fields["invoked"] = True
    else:
        fields = {
            "offline_model": "", "provider": "", "model_id": "", "endpoint_hash": "",
            "prompt_version": "", "latency": {}, "token_usage": {},
            "invoked": False,
            "note": ("在线链不调用离线治理模型（硬约束 1 / §3.3）；离线模型身份由离线链"
                     "（evolution/offline_driver.py）的 RunManifest 登记"),
        }
    return _strip_secrets(fields)


def _strip_secrets(value, *, _depth: int = 0):
    """递归剔除密钥类键并做值级兜底（§3.4：绝不落盘 API key/Authorization）。

    键名命中 `_FORBIDDEN_MANIFEST_KEYS` → 丢弃；字符串值若与环境里的
    `DEEPSEEK_API_KEY` 相同 → 替换为 `***`（双保险，防上游把密钥塞进别的字段）。
    """
    secret = os.environ.get("DEEPSEEK_API_KEY", "")
    if _depth > 6:
        return "<max_depth>"
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if str(k).lower() in _FORBIDDEN_MANIFEST_KEYS:
                continue
            out[str(k)] = _strip_secrets(v, _depth=_depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_strip_secrets(v, _depth=_depth + 1) for v in value]
    if isinstance(value, str) and secret and secret in value:
        return "***"
    return value


def _write_manifest(args, cfg_yaml: dict, run, run_cfg, split: str, *,
                    outcomes=None, items=None, offline_client=None):
    """写 RunManifest（失败不阻断评测：复现信息是附加产物）。

    §19.2：模板版本 / tool-face 版本 / EvidenceProfile 版本 / MetricEvidenceGate 版本 /
    距离原语参数 / 尺度融合版本 + 离线治理模型字段（离线模型只在**离线链**被调用，
    本函数只登记其冻结配置；绝不写 API key）。
    """
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
        from skill3d.evaluation.golden_v5 import golden_versions
        from skill3d.reconstruction_gate.quality_metrics import (
            QUALITY_METRIC_VERSION,
        )

        # §19.2 离线治理模型块（无离线客户端 → 只登记冻结配置，invoked=false）
        offline = offline_manifest_fields(offline_client)
        # HC39 golden 三元组：**必须带 `golden_` 前缀**。`golden_versions()` 里的
        # `schema_version` / `quality_metric_version` 是 **golden 夹具**的版本
        # （v5 golden = 5.0 / v5-no-g8-g5-optional），与本 run 的 v6 口径不同名同义——
        # 直接展开会把 "6.0 / v6-warp-overlap-no-g5" 覆盖成 golden 的版本号，
        # 让 RunManifest 谎报本 run 的 schema 口径。
        golden = golden_versions()
        golden_fields = {
            "golden_version": golden.get("golden_version", ""),
            "golden_schema_version": golden.get("schema_version", ""),
            "golden_quality_metric_version": golden.get("quality_metric_version", ""),
        }
        return write_run_manifest(m, out, extra={
            # ---- v6 版本三元组（HC39：schema/质量口径/golden 必须同时留档）----
            "schema_version": "6.0",
            "quality_metric_version": QUALITY_METRIC_VERSION,
            **golden_fields,
            # ---- §19.2 版本字段（模板 / tool-face / 证据画像 / gate / 距离原语）----
            **v6_version_fields(),
            # ---- §19.2 离线治理模型块（扁平键 + 嵌套块双写，便于审计脚本直读）----
            "offline_model": offline.get("offline_model", ""),
            "provider": offline.get("provider", ""),
            "model_id": offline.get("model_id", ""),
            "endpoint_hash": offline.get("endpoint_hash", ""),
            "prompt_version": offline.get("prompt_version", ""),
            "latency": offline.get("latency", {}),
            "token_usage": offline.get("token_usage", {}),
            "offline_governance": offline,
            # ---- v6 §5.2：重建路线受控枚举只有 vggt；BA 与校准尺度路线已废止（§20）----
            "recon_method": str(run_cfg.recon_method),
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
            "run_id": run.run_id, "split": split, "mode": args.mode,
            "baseline": args.baseline, "n_episodes": run.n_episodes,
            "active_snapshot_ref": run.active_snapshot_ref,
            # 影响结果的推理参数必须记录（§16.4 / §7：冻结模型 + 一致推理环境）
            "vllm_model": run_cfg.vllm_model,
            "n_frames": vllm_cfg.get("n_frames"),
            "max_pixels": vllm_cfg.get("max_pixels"),
            "max_model_len": vllm_cfg.get("max_model_len"),
            "vllm_endpoints": list(run_cfg.vllm_endpoints),
            # §7 / D-6 / A-8 / D-2：帧集身份（v6 不再记录 BA 开关与校准尺度来源）
            "frame_set_hash": fsh,
            "n_distinct_frame_sets": n_fsh,
            "readiness_manifest_ref": str(
                (cfg_yaml.get("readiness") or {}).get("manifest_path", ""))})
    except Exception as exc:  # noqa: BLE001
        return f"(写失败: {type(exc).__name__}: {exc})"


def _apply_skill_ablations(args, skills: list) -> tuple[list, str]:
    """C2（`--skill-spec`）/ C5（`--inject-wrong-skill`）消融注入。scope:

    C5 的"已知错误 Skill"是 v6 形态的**故意错误**模板：

    - 调用不存在的 Tool 名称（`definitely_not_a_registered_tool`）→ M9/AST 应拒；
    - 声明一条**当前证据状态不可能满足**的证据签名（要求 `metric_scale=available` +
      gate 版本匹配，而 mock/普通场景的 gate 未过）→ M7 检索硬过滤应拦。

    这样"错误 Skill 被拦住"的可证伪命题（§17.5 #5）在 v6 的两道硬门上都被覆盖：
    检索（证据签名 / gate 双重 fail-closed，§13.6）与执行（AST/沙箱）。
    v5 的 `requires_artifacts=["nonexistent_artifact"]` + `minimum_quality` 已随
    §5.8 的签名化检索一并废止（§20）。
    """
    from skill3d.schemas import SkillSpec

    out = list(skills)
    ref = ""
    if args.skill_spec:
        spec = SkillSpec.model_validate_json(Path(args.skill_spec).read_text(encoding="utf-8"))
        out = [s for s in out if s.skill_id != spec.skill_id] + [spec]
        ref = f"static:{spec.skill_id}@{spec.version}"
    if args.inject_wrong_skill:
        wrong = wrong_skill_spec()
        out = [s for s in out if s.skill_id != wrong.skill_id] + [wrong]
        ref = (ref + "+" if ref else "") + f"wrong:{wrong.skill_id}@{wrong.version}"
    return out, ref


def wrong_skill_spec():
    """C5 内置的"已知错误 Skill"（v6 SkillSpec；见 `_apply_skill_ablations` 文档）。

    三处故意错误：① 调用未注册 Tool；② 声明 `metric_scale=available` 且挂了
    gate 版本（普通场景 gate 未过 → §13.6 检索硬过滤拦下）；③ 断言恒假
    （`never_true()`）。**只在消融档使用**，绝不进 active snapshot。
    """
    from skill3d.schemas import SkillSpec
    from skill3d.schemas.evidence import GATE_VERSION

    return SkillSpec(
        skill_id="sk-known-wrong",
        version="0.0.1",
        applicable_question_types=["object_counting"],
        required_evidence_signature={"metric_scale": "available"},
        requires_metric_evidence=True,
        applicable_gate_version=GATE_VERSION,
        skill_family="counting",
        source="mock_interface",
        description="【C5 消融】已知错误模板：调用不存在的 Tool 并返回常数",
        call_graph_template=(
            "answer = definitely_not_a_registered_tool(1, 2)\n"
            "ReturnAnswer(\"42\")"),
        validation_assertions=["never_true()"],
    )


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


if __name__ == "__main__":
    raise SystemExit(main())
