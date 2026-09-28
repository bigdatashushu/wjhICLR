#!/usr/bin/env python3
"""把 v10 campaign 的收据汇总成 §18 的报告骨架（**只读**，不改任何收据）。

规范原文（§18 首期报告模板）：

```
代码版本：
初始快照：
目标题型/Skill：
冻结配置 hash：
Generation 1
- 父版本：
- learning 运行与合格经验数：
- 候选版本与 diff：
- seed 0 父/候选得分、错误数、交付覆盖：
- seed 1 父/候选得分、错误数、交付覆盖：
- 决定：promote/reject
- 发布后真实使用证据：
Generation 2
...
可声明结论：
不可声明结论：
未完成项：
```

用法：`PYTHONPATH=src $EXP scripts/report_evolution_campaign.py --run-root data/v10_campaign/runs/camp-v10-001`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--markdown", default="", help="写到该 Markdown 文件（默认打印）")
    args = parser.parse_args(argv)
    root = Path(args.run_root)
    checkpoint = _read(root / "campaign.json") or {}
    campaign = checkpoint.get("campaign") or {}
    panels = _read(root / "panels.json") or {}
    lines: list[str] = []
    add = lines.append

    add(f"# v10 两代 campaign 报告（{campaign.get('campaign_id', root.name)}）")
    add("")
    code = _first_run_state(root)
    add(f"- 代码版本：{code.get('commit', '') or '（未登记）'}"
        f"（工作区脏={code.get('dirty_worktree')}；关键源码聚合 sha256="
        f"{str(code.get('sources_sha256', ''))[:16]}）")
    add(f"- 初始快照：{campaign.get('initial_snapshot_id', '')}")
    add(f"- 目标题型 / Skill：{campaign.get('target_question_type', '')} / "
        f"{campaign.get('target_skill_id', '')}")
    add(f"- 终态：{campaign.get('completion_status', '')} "
        f"（state={campaign.get('state', '')}）")
    add(f"- 停止原因：{checkpoint.get('stop_reason', '') or '（无）'}")
    add(f"- 冻结面板：learning={len(panels.get('learning_qa_ids') or [])} 题 / "
        f"{len(panels.get('learning_scenes') or [])} scene；"
        f"inner_g0={len(panels.get('inner_g0_qa_ids') or [])}；"
        f"inner_g1={len(panels.get('inner_g1_qa_ids') or [])}；"
        f"互不重叠={panels.get('inner_panels_disjoint')}")
    add("")

    for gen_dir in sorted(root.glob("gen*")):
        generation = gen_dir.name
        add(f"## {generation}")
        run = _read(gen_dir / "parent_run_manifest.json") or {}
        add(f"- learning 运行：run_id={run.get('run_id', '')} 题目 {run.get('n_items')} "
            f"snapshot={run.get('snapshot_id', '')} panel_hash="
            f"{str(run.get('panel_hash', ''))[:12]}")
        add(f"- 冻结配置：retrieval={run.get('retrieval_policy_version', '')} "
            f"sha={str(run.get('retrieval_policy_sha256', ''))[:12]} "
            f"manifest_hash={str(run.get('manifest_hash', ''))[:12]}")
        bundle = _read(gen_dir / "experience_bundle.json") or {}
        if bundle:
            add(f"- 经验包：{bundle.get('bundle_id', '')} 父="
                f"{bundle.get('parent_skill_id')}@{bundle.get('parent_skill_version')} "
                f"合格={len(bundle.get('eligible_experience_refs') or [])} "
                f"scene={bundle.get('scene_count')} 成功={bundle.get('success_count')} "
                f"失败={bundle.get('failure_count')}")
            add(f"  - 行为摘要：{json.dumps(bundle.get('behavior_summary') or {}, ensure_ascii=False)}")
            add(f"  - 排除摘要：{json.dumps(bundle.get('exclusion_summary') or {}, ensure_ascii=False)}")
        candidate = _read(gen_dir / "candidate.json") or {}
        if candidate:
            diff_fields = [d.get("field") for d in candidate.get("structured_diff") or []]
            add(f"- 候选：{candidate.get('candidate_id', '')} "
                f"{candidate.get('parent_skill_version')} → "
                f"{candidate.get('candidate_skill_version')}"
                f"（diff 字段={diff_fields}）")
            add(f"  - 假设：{candidate.get('hypothesis', '')}")
        static = _read(gen_dir / "static_validation.json") or {}
        if static:
            add(f"- 静态检查：passed={static.get('passed')} "
                f"problems={static.get('problems')}")
        for seed in (0, 1):
            paired = _read(gen_dir / f"paired_seed_{seed}.json")
            if not paired:
                continue
            add(f"- seed {seed}：父 {paired.get('mean_a'):.4f}（{paired.get('arm_a_skill_version')}）"
                f" vs 候选 {paired.get('mean_b'):.4f}（{paired.get('arm_b_skill_version')}）"
                f" Δ={paired.get('delta'):+.4f}；run_error {paired.get('n_run_error_a')}→"
                f"{paired.get('n_run_error_b')}；合法答案率 "
                f"{paired.get('valid_answer_rate_a'):.3f}→{paired.get('valid_answer_rate_b'):.3f}；"
                f"正文入请求 {paired.get('delivered_a')}/{paired.get('n_items')}"
                f"→{paired.get('delivered_b')}/{paired.get('n_items')}")
        decision = _read(gen_dir / "decision.json") or {}
        if decision:
            add(f"- 决定：{'promote' if decision.get('promote') else 'reject'}")
            for key, value in (decision.get("conditions") or {}).items():
                add(f"  - {key}: {value}")
            if decision.get("reasons"):
                add(f"  - 原因：{decision['reasons']}")
        promotion = _read(gen_dir / "promotion.json") or {}
        if promotion:
            outcome = str(promotion.get("outcome", "") or (
                "rejected" if promotion.get("promoted") is False else "promoted"))
            if outcome != "promoted":
                add(f"- 发布：未发布（reject；{promotion.get('reasons')}）")
            else:
                add(f"- 发布：{promotion.get('snapshot_before')} → "
                    f"{promotion.get('snapshot_after')}；竞争集合="
                    f"{promotion.get('competing_versions')}；历史="
                    f"{promotion.get('historical_versions')}；manifest_hash_before="
                    f"{str(promotion.get('manifest_hash_before', ''))[:12]} → "
                    f"{str(promotion.get('manifest_hash_after', ''))[:12]}")
        use = _read(gen_dir / "post_publish_use.json") or {}
        if use:
            add(f"- 发布后使用：status={use.get('status')} "
                f"retrieved={use.get('retrieved')} delivered={use.get('delivered')} "
                f"hash_match={use.get('request_content_sha256_match')} "
                f"经验事件={len(use.get('experience_event_refs') or [])}")
        add("")

    add("## 可声明结论")
    add("")
    add("见报告正文（本骨架只汇总收据事实；口径声明必须人工书写，避免机器替项目下结论）。")
    add("")
    add("## 不可声明结论")
    add("")
    add("- 未跑 outer_holdout / final_test：不得声明「独立泛化收益」（§2.3/§8.5）。")
    add("- inner 小面板双 seed 改善只表示「开发门通过」，不表示统计显著（§8.6）。")
    add("")
    add("## 未完成项")
    add("")
    add("- 见 `docs/skill_library_v10_implementation.md` 第 4 节。")

    text = "\n".join(lines) + "\n"
    if args.markdown:
        Path(args.markdown).write_text(text, encoding="utf-8")
        print(f"报告骨架已写: {args.markdown}")
    else:
        print(text)
    return 0


def _first_run_state(root: Path) -> dict:
    for gen_dir in sorted(root.glob("gen*")):
        run = _read(gen_dir / "parent_run_manifest.json") or {}
        if run.get("code_state"):
            return dict(run["code_state"])
    return {}


def _first_run(root: Path, key: str) -> str:
    for gen_dir in sorted(root.glob("gen*")):
        run = _read(gen_dir / "parent_run_manifest.json") or {}
        if run.get(key):
            return str(run[key])
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
