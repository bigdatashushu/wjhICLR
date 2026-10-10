"""在线评测 CLI（§13.5）：

```bash
python -m skill3d.online.eval --split test --active-snapshot data/active_snapshot.json
```

- 数据源 `--source`：`synthetic`（合成，管道验证）| `vsi_bench`（HF meta + 原始视频，
  `TODO_USER_INPUT`）| `jsonl`（本系统约定格式的预抽帧清单）；
- 两种运行模式 `--mode`：`real`（真实 M1–M13）| `mock_light`（合成输入 + 确定性 stub
  program，仅管道验证，§9.2）；
- 在线入口固定 `C1_tools_program`；B01/B11 与父/候选比较由 v11 实验驱动器执行；
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
    active_vision_from,
    paths_from,
    retrieval_policy_from,
    sandbox_from,
    vllm_from,
)
from skill3d.online.runner import (
    MAX_RECOVERY_ATTEMPTS,
    OnlineRunConfig,
    run_split,
)
from skill3d.online.submission import EXECUTION_PROTOCOL_VERSION
from skill3d.schemas import SkillSpecV11
from skill3d.schemas.trace import EPISODE_TRACE_SCHEMA_VERSION
from skill3d.skills.registry import active_snapshot_provenance, load_active_skills
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION

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
    p.add_argument("--video-fallback-root", action="append", default=None,
                   help="§5.2 可重试加载的来源副本根（可重复；缺省用配置 "
                        "paths.raw_video_fallbacks）")
    p.add_argument("--question-types", default="",
                   help="逗号分隔的题型子集（默认 8 题型）")
    p.add_argument("--sampling-per-task", type=int, default=0,
                   help="按题型均匀采样：每题型最多 N 条（与重建批使用同一参数，保证同一样本）")
    p.add_argument("--datasets", default="",
                   help="逗号分隔的来源数据集（scannet,scannetpp,arkitscenes）；"
                        "限定范围会写进 RunManifest，报告不得冒充全量口径")
    p.add_argument("--limit", type=int, default=0, help="最多跑 N 条（0 = 不限）")
    p.add_argument("--mode", default="mock_light", choices=["real", "mock_light"])
    p.add_argument("--baseline", default="C1_tools_program",
                   choices=["C1_tools_program"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic-replay", action="store_true",
                   help="重放确定性：时间/id/latency 取确定性占位，保证同 seed 字节级一致")
    p.add_argument("--active-snapshot", default="", help="active snapshot 文件或目录（§13.5）")
    p.add_argument("--vllm-endpoint", action="append", default=[],
                   help="本地 vLLM endpoint（可重复，DP×8 时给 8 个）")
    p.add_argument("--vllm-model", default="", help="served model name")
    p.add_argument("--trace-dir", default="")
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
                        "历史对照基线请在对应 Git 提交运行）")
    p.add_argument("--max-retries-per-operation", type=int, default=None,
                   help="每类失败在有界求解循环中的最大恢复次数")
    p.add_argument("--eval-visual-fallback", action=argparse.BooleanOptionalAction,
                   default=None, help="评测失败后追加一次独立的 32 原帧视觉回答")
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
    # v9 §17.1/§18.1：环境依赖先检后跑。缺依赖时 M4 子项会 fail-closed 成
    # `fallback_2d_only`，把环境故障伪装成场景质量，因此这里直接拒绝启动。
    from skill3d.env_preflight import RuntimeEnvironmentError, assert_runtime_dependencies

    try:
        inference_env = assert_runtime_dependencies(context=f"online.eval:{args.mode}")
    except RuntimeEnvironmentError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2
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
    active_vision = active_vision_from(cfg_yaml)

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
    snapshot_manifest_sha256 = ""
    skills, warnings = [], []
    snap_arg = args.active_snapshot or paths.active_snapshot
    if snap_arg:
        try:
            skills, warnings, snapshot_ref = load_active_skills(snap_arg)
        except ValueError as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            return 2
        _active_ref, snapshot_manifest_sha256 = active_snapshot_provenance(snap_arg)
    legacy = [type(skill).__name__ for skill in skills
              if not isinstance(skill, SkillSpecV11)]
    if legacy:
        print("[错误] 当前在线协议只接受 runtime-skill-snapshot/2.0 与 "
              f"SkillSpecV11；收到旧格式 {legacy}", file=sys.stderr)
        return 2
    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)

    # ---- 数据源 ----
    qtypes = [q.strip() for q in args.question_types.split(",") if q.strip()]
    limit = args.limit or None
    sampling_receipt: dict = {}
    exclusions: list = []
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
            items = load_jsonl_items(
                args.episodes_jsonl,
                split=split,
                limit=limit,
                exclusions=exclusions,
                include_input_errors=True,
            )
        else:
            split_cfg = load_yaml(cfg_yaml.get("split_config", "configs/vsi_bench_split.yaml"))
            # §5.3：抽样算法的 seed／qa_id 清单／hash 与**排除行**都要落盘 ——
            # "每个预登记 qa_id 必须有结果行；不足 32 帧不能通过 skip 静默消失"。
            items = load_vsi_bench_items(
                split, split_cfg,
                video_root=args.video_root or paths.raw_videos,
                # §5.2"可重试加载"（用户 2026-09-27 裁定：重试 = 换来源/换副本）
                video_fallback_roots=(args.video_fallback_root
                                      if args.video_fallback_root is not None
                                      else paths.raw_video_fallbacks),
                cache_dir=paths.vsi_bench_meta, limit=limit, seed=args.seed,
                # v5 修复：vsi_bench 源此前**静默忽略** --question-types（分层子集实验
                # 会误跑成全量）→ 现在如实传递过滤条件。
                question_types=qtypes or None,
                datasets=[d.strip() for d in args.datasets.split(",") if d.strip()] or None,
                stratified_per_task=int(args.sampling_per_task),
                # §5.3：抽样口径必须与 seed 绑定，不能靠文件行序
                sampling_seed=args.seed,
                exclusions=exclusions,
                sampling_receipt=sampling_receipt,
                include_input_errors=True,
            )
            if exclusions:
                print(f"[warn] {len(exclusions)} 条预登记 qa_id 输入不可用"
                      f"（将生成 input_error 零分结果并保留分母；§5.3）",
                      file=sys.stderr)
                for e in exclusions[:5]:
                    print(f"        {e['qa_id']} → {e['reason']}", file=sys.stderr)
            # §5.2 可重试加载：换过来源副本的题必须报出来（副本可能是另一种编码，
            # 像素与主来源不同；静默使用会让两次 run 不可比）。
            retries = [r for r in (sampling_receipt.get("source_retries") or [])
                       if r.get("retried")]
            if retries:
                print(f"[warn] {len(retries)} 条使用了**来源副本**（§5.2 可重试加载）："
                      f"主来源失败后改用镜像/同名副本，已记 source_retried_samples",
                      file=sys.stderr)
                for r in retries[:5]:
                    print(f"        {r['qa_id']} → {r['used_source']}", file=sys.stderr)
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
        active_snapshot_manifest_sha256=snapshot_manifest_sha256,
        skills=skills,
        max_regen=int(sandbox.max_regenerate),
        cell_timeout_s=int(sandbox.cell_timeout_s),
        trace_dir=args.trace_dir or paths.trace_store,
        recon_dir=args.recon_dir or paths.reconstructions,
        vllm_endpoints=list(args.vllm_endpoint),
        vllm_model=args.vllm_model or vllm.model,
        max_images=int(getattr(vllm, "n_frames", 32)),
        max_tokens=int(cfg_yaml.get("vllm", {}).get("max_tokens", 4096)),
        recon_method=args.recon_method,
        metric_depth_model=_maybe_moge2(args),
        # v6 §20：BA（官方 VGGSfM / vggt_sparse_ba）与整套"需校准的尺度"路线已废止，
        # 相关配置项（ba_enabled / scale_calibration_* / scale_confidence_level）不再传入。
        max_retries_per_operation=(int(args.max_retries_per_operation)
                                   if args.max_retries_per_operation is not None
                                   else int(cfg_yaml.get("max_retries_per_operation",
                                                        MAX_RECOVERY_ATTEMPTS))),
        input_diagnostics=bool(cfg_yaml.get("input_diagnostics", False)),
        max_solver_rounds=int(cfg_yaml.get("max_solver_rounds", 6)),
        finalization_rounds=int(cfg_yaml.get("finalization_rounds", 1)),
        eval_visual_fallback=(args.eval_visual_fallback if args.eval_visual_fallback is not None
                              else bool(cfg_yaml["eval_visual_fallback"])),
        max_pixels=int(getattr(vllm, "max_pixels", 131072) or 131072),
        max_model_len=int(getattr(vllm, "max_model_len", 32768) or 32768),
        allow_final_test=args.allow_final_test,
        # §13.5：top-k／排序权重／方法上下文上限来自冻结配置（不在代码里另设一份）
        retrieval_policy=retrieval_policy_from(cfg_yaml),
        # §9.4：主动图像的声明布局与派生图占位上限
        image_layout=active_vision.layout,
        max_derived_images=int(active_vision.max_derived_images),
    )
    if not run_cfg.vllm_endpoints and args.mode == "real":
        print("[warn] mode=real 但未给 --vllm-endpoint：program 生不出来的 episode 会记 "
              "unavailable（§4 M8 字段 9）。离线起服务：bash scripts/serve_qwen3vl_dp8.sh",
              file=sys.stderr)

    print("=" * 78)
    print(f"split={split} source={args.source} n_episodes={len(items)} mode={args.mode} "
          f"baseline={args.baseline} active_snapshot={snapshot_ref} "
          f"skills={len(skills)} template={PROMPT_TEMPLATE_VERSION}")
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
    print(f"\ntrace 已写: {run_cfg.trace_dir}（episode_input / episode_trace / "
          f"program_trace / geometry_check / evaluation_result / evaluation_run / "
          f"online_run）")
    print(f"synthesis_source 分布: "
          f"{ {s: sum(1 for o in outcomes if o.synthesis_source == s) for s in set(o.synthesis_source for o in outcomes)} }")

    # ---- 系统可靠性小节（§4 M6 字段 9 / §16.2：与主表分离报告，不混主表）----
    try:
        from skill3d.evaluation.process_metrics import aggregate_process_metrics

        pm = aggregate_process_metrics(outcomes)
        print("\n系统可靠性（不混主表）:")
        print(f"  coverage={_fmt(pm.coverage)}  refusal_rate={_fmt(pm.refusal_rate)}  "
              f"abstain_rate={_fmt(pm.abstain_rate)}  input_error={pm.n_input_error}")
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
                                    sampling_receipt=locals().get("sampling_receipt"),
                                    exclusions=locals().get("exclusions"),
                                    outcomes=outcomes, items=items)
    print(f"RunManifest: {manifest_path}（code_commit / pip_freeze_hash / "
          f"checkpoint_sha256 / split_version / seed / inference_env）")
    # §13.5：检索策略必须"写入配置并冻结" —— 启动时把**实际生效**的版本打出来，
    # 免得跑了半天才发现用的不是配置里那一份（标签 + 内容摘要一起打）。
    _rp = run_cfg.retrieval_policy
    print(f"[info] 检索策略 {_rp.version()}（sha256={_rp.sha256()[:12]}，"
          f"来源={_rp.source}）：按题型选择唯一 active 方法，"
          f"方法上下文上限={_rp.method_context_max_chars} 字符")
    if run.status != "completed":
        print(f"[error] run incomplete: {run.n_unavailable} service-unavailable episodes")
        return 1
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


def current_version_fields() -> dict:
    """当前在线合同身份（在线评测与离线清单共用同一份口径）。"""
    from skill3d.reconstruction.metric_fusion import (
        METRIC_FUSION_VERSION,
        METRIC_MODEL_NONE,
    )
    from skill3d.schemas.evidence import GATE_VERSION, PROFILE_VERSION
    from skill3d.tools.distance_primitives import DistancePrimitiveParams
    from skill3d.tools.registry import TOOL_FACE_VERSION
    return {
        "template_version": PROMPT_TEMPLATE_VERSION,
        "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
        "tool_face_version": TOOL_FACE_VERSION,
        "tool_docs_version": TOOL_DOCS_VERSION,
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


def _denominator_fields(outcomes, sampling_receipt, exclusions) -> dict:
    result_ids = [str(getattr(outcome, "qa_id", "") or "")
                  for outcome in (outcomes or [])]
    planned_ids = list((sampling_receipt or {}).get("qa_ids") or result_ids)
    return {
        "n_excluded": len(exclusions or []),
        "exclusions": list(exclusions or []),
        "n_input_error": sum(
            1 for outcome in (outcomes or [])
            if getattr(outcome, "episode_status", "") == "input_error"),
        "denominator_preserved": (
            len(result_ids) == len(set(result_ids))
            and len(planned_ids) == len(set(planned_ids))
            and set(result_ids) == set(planned_ids)
        ),
        "missing_result_qa_ids": sorted(set(planned_ids) - set(result_ids)),
        "unexpected_result_qa_ids": sorted(set(result_ids) - set(planned_ids)),
    }


def _write_manifest(args, cfg_yaml: dict, run, run_cfg, split: str, *,
                    outcomes=None, items=None, offline_client=None,
                    sampling_receipt=None, exclusions=None):
    """写 RunManifest（失败不阻断评测：复现信息是附加产物）。

    §19.2：模板版本 / tool-face 版本 / EvidenceProfile 版本 / MetricEvidenceGate 版本 /
    距离原语参数 / 尺度融合版本 + 离线治理模型字段（离线模型只在**离线链**被调用，
    本函数只登记其冻结配置；绝不写 API key）。
    """
    from skill3d.infra.version_lock import build_run_manifest, write_run_manifest

    retriev = retrieval_policy_from(cfg_yaml)
    try:
        from skill3d.adapters.split_builder import load_split_config

        split_cfg_path = cfg_yaml.get("split_config", "configs/vsi_bench_split.yaml")
        split_version = load_split_config(split_cfg_path).split_version
    except Exception:  # noqa: BLE001
        split_cfg_path, split_version = "", ""
    try:
        m = build_run_manifest(
            checkpoint_path=str((cfg_yaml.get("vllm") or {}).get("model", "")),
            config_path=args.config, repo_dir=".",
            split_version=split_version, seed=args.seed,
            split_config_path=split_cfg_path)
        out = args.run_manifest or "data/run_manifest.json"
        vllm_cfg = cfg_yaml.get("vllm") or {}
        fsh, n_fsh = _frame_set_summary(items or [])
        from skill3d.reconstruction_gate.quality_metrics import (
            QUALITY_METRIC_VERSION,
        )

        # §19.2 离线治理模型块（无离线客户端 → 只登记冻结配置，invoked=false）
        offline = offline_manifest_fields(offline_client)

        # §5.3：抽样算法／seed／qa_id 清单／hash 与排除行必须落盘 ——
        # "每个预登记 qa_id 必须有结果行"，且总体分母不得因排除而消失。
        sampling_fields = {}
        if sampling_receipt:
            sampling_fields = {
                "sampling_strategy": sampling_receipt.get("strategy", ""),
                "sampling_seed": sampling_receipt.get("seed"),
                "sampling_per_task_cap": sampling_receipt.get("per_task_cap"),
                "sampling_n_preregistered": sampling_receipt.get("n_preregistered"),
                "sampling_scenes": sampling_receipt.get("scenes", []),
                "sampling_qa_ids": sampling_receipt.get("qa_ids", []),
                "sampling_qa_id_sha256": sampling_receipt.get("qa_id_sha256", ""),
            }
            # §5.2：部分可读样本"独立报告"（仍在分母里，但不得伪称完整 32 帧输入）
            if sampling_receipt.get("partially_readable"):
                sampling_fields["partially_readable_samples"] =                     sampling_receipt["partially_readable"]
            # §5.2 可重试加载：换来源副本的事实（哪些题、用了哪个副本、试过哪些来源）
            source_retries = list(sampling_receipt.get("source_retries") or [])
            if source_retries:
                sampling_fields["source_retried_samples"] = source_retries
                sampling_fields["n_source_retries"] = sum(
                    1 for r in source_retries if r.get("retried"))
        if exclusions is not None:
            sampling_fields.update(_denominator_fields(
                outcomes, sampling_receipt, exclusions))
        return write_run_manifest(m, out, extra={
            **sampling_fields,
            # ---- 版本三元组（HC39：schema/质量口径必须同时留档）----
            # v9：本 run 的 episode trace 声明当前合同（§17.2），manifest 必须
            # 与之一致 —— 否则 manifest 说 6.0、trace 说 9.0，又是一处自相矛盾。
            "schema_version": EPISODE_TRACE_SCHEMA_VERSION,
            "quality_metric_version": QUALITY_METRIC_VERSION,
            # ---- §19.2 版本字段（模板 / tool-face / 证据画像 / gate / 距离原语）----
            **current_version_fields(),
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
            # §13.5：检索策略必须"写入配置并冻结"，manifest 里同时留人类标签与
            # 内容摘要 —— 只留标签的话，配置被改过也看不出来。
            "retrieval_config_version": retriev.version(),
            "retrieval_config_sha256": retriev.sha256(),
            "retrieval_config_source": str(retriev.source),
            "retrieval_method_context_max_chars": int(retriev.method_context_max_chars),
            "retrieval_policy": retriev.canonical(),
            # §13.6：实测交付事实（本 run 里"检索选中"与"已送达模型"各有多少条）
            "n_retrieval_records": sum(
                len(getattr(o, "retrieval_records", []) or []) for o in (outcomes or [])),
            "retrieved_skill_versions": sorted({
                v for o in (outcomes or [])
                for v in (getattr(o, "retrieved_skill_versions", []) or [])}),
            "delivered_skill_versions": sorted({
                v for o in (outcomes or [])
                for v in (getattr(o, "delivered_skill_versions", []) or [])}),
            "declared_selected_skill_versions": sorted({
                v for o in (outcomes or [])
                for v in (getattr(o, "declared_selected_skill_versions", []) or [])}),
            "n_episodes_with_delivered_skills": sum(
                1 for o in (outcomes or [])
                if getattr(o, "delivered_skill_versions", None))})
    except Exception as exc:  # noqa: BLE001
        return f"(写失败: {type(exc).__name__}: {exc})"


def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


if __name__ == "__main__":
    raise SystemExit(main())
