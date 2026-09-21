"""§7/§16.3 多 seed 主表入口（G-66 的生产调用者）。

**为什么需要它**：`multi_seed_aggregator` 的统计内核早已实现，但**没有生产入口**
（§10.5 B 类 dead code），而 §7 要求"主表报告 ≥3 个 seed，mean±std"——没有入口
就等于主表永远出不来。本模块把「N 个 seed 的 run 产物 → 主表行 + 统计 + 资格判定」
接成一条可跑的命令。

三道门（缺一不可进主表，HC24/HC34）：

1. **seed 数**：每个方法必须 ≥3 seed（`MIN_SEEDS_FOR_MAIN_TABLE`），否则标
   `eligible_for_main_table=false` 并给警告；退出码 1；
2. **真实性**：`mode != "real"`（mock_light/合成）的 run 一律**不得**进主表
   （HC24：mock 仅开发用），默认直接拒绝，需 `--allow-mock` 才允许生成
   "仅管道验证"的非资格表；
3. **可复现**：输出里带每个 seed 的 `run_id / code_commit / timestamp / split`，
   供 RunManifest 与审计回溯引用。

输入读取约定：`--runs` 收**trace 目录**（`online/eval.py --trace-dir` 的产物），
从 `evaluation_run.jsonl` 取指标、从 `online_run.jsonl` 取 mode/seed/split 等事实
（按 `run_id` 关联）；也接受直接给单个 JSON 文件（含 EvaluationRun 字段 +
可选的 mode/seed）。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from skill3d.schemas import EvaluationRun

from .multi_seed_aggregator import (
    MIN_SEEDS_FOR_MAIN_TABLE,
    aggregate_metric,
    aggregate_runs,
    aggregate_to_json,
    check_seed_count,
    format_main_table_row,
    format_paper_table,
    macro_average_over_tasks,
)


class MainTableError(RuntimeError):
    """主表无法生成（输入缺失/格式错误/未过门禁）。"""


@dataclass
class RunRecord:
    """一个 seed 的 run 产物（指标 + 运行事实）。"""

    run: EvaluationRun
    mode: str = ""
    baseline: str = ""
    seed: Optional[int] = None
    split: str = ""
    source: str = ""
    trace_dir: str = ""

    @property
    def is_mock(self) -> bool:
        """非 real 模式的 run（含空 mode：来源不明 → 按不可信处理，fail-closed）。"""
        return str(self.mode) != "real"

    def as_facts(self) -> dict:
        return {
            "run_id": self.run.run_id,
            "mode": self.mode,
            "baseline": self.baseline or self.run.active_snapshot_ref,
            "seed": self.seed,
            "split": self.split or self.run.split,
            "source": self.source,
            "n_episodes": self.run.n_episodes,
            "code_commit": self.run.code_commit,
            "timestamp": self.run.timestamp,
            "accuracy": self.run.accuracy,
            "mra": self.run.mra,
            "trace_dir": self.trace_dir,
        }


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def load_run(path: str | Path) -> RunRecord:
    """读一个 seed 的产物：trace 目录（JSONL）或单个 JSON 文件。"""
    p = Path(path)
    if p.is_dir():
        eval_rows = _read_jsonl(p / "evaluation_run.jsonl")
        if not eval_rows:
            raise MainTableError(f"{p} 下没有 evaluation_run.jsonl（用 --trace-dir 产出）")
        run = EvaluationRun.model_validate(eval_rows[-1])
        online_rows = _read_jsonl(p / "online_run.jsonl")
        facts = online_rows[-1] if online_rows else {}
        return RunRecord(run=run, mode=str(facts.get("mode", "")),
                         baseline=str(facts.get("baseline", "")),
                         seed=facts.get("seed"), split=str(facts.get("split", "")),
                         source=str(facts.get("source", "")), trace_dir=str(p))
    if not p.is_file():
        raise MainTableError(f"run 路径不存在: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "evaluation_run" in data:            # 合并形态：{evaluation_run: {...}, facts...}
        payload = dict(data["evaluation_run"])
        facts = data
    else:
        payload, facts = dict(data), data
    run = EvaluationRun.model_validate(payload)
    return RunRecord(run=run, mode=str(facts.get("mode", "")),
                     baseline=str(facts.get("baseline", "")),
                     seed=facts.get("seed"), split=str(facts.get("split", "")),
                     source=str(facts.get("source", "")), trace_dir=str(p))


@dataclass
class GroupTable:
    """一个方法/配置的多 seed 主表行。"""

    label: str
    records: list[RunRecord] = field(default_factory=list)
    aggregates: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    eligible_for_main_table: bool = False
    reasons: list[str] = field(default_factory=list)
    # mock/来源不明被检出（HC24 硬规则：**任何开关都不能**把它变成 eligible）
    mock_detected: bool = False
    pipeline_only: bool = False

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n_seeds": len(self.records),
            "eligible_for_main_table": self.eligible_for_main_table,
            "mock_detected": self.mock_detected,
            "pipeline_only": self.pipeline_only,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "macro": self.macro,
            "aggregates": aggregate_to_json(self.aggregates),
            "seeds": [r.as_facts() for r in self.records],
        }


def build_group(label: str, records: Sequence[RunRecord], *,
                allow_mock: bool = False,
                min_seeds: int = MIN_SEEDS_FOR_MAIN_TABLE) -> GroupTable:
    """聚合一个方法的多 seed run，并判定它是否有资格进主表。"""
    g = GroupTable(label=label, records=list(records))
    if not g.records:
        g.reasons.append("没有任何 run 产物")
        return g

    # 门 ②：真实性（HC24 硬规则：**永远**是 blocking reason，allow_mock 也不能绕）
    mocks = [r for r in g.records if r.is_mock]
    if mocks:
        modes = sorted({r.mode or "(未记录)" for r in mocks})
        msg = (f"{len(mocks)}/{len(g.records)} 个 run 不是 real 模式（mode={modes}）"
               "→ 按 HC24 不得进主表")
        g.mock_detected = True
        g.reasons.append(msg)
        if allow_mock:
            # allow_mock 只放行"生成表"这一步（便于验证管道与版式），
            # 不改变资格判定：本表仍标记为仅管道验证、不得写进论文
            g.pipeline_only = True
            g.warnings.append(msg + "（--allow-mock：仅生成管道验证表，"
                                   "eligible 仍为 False）")

    # 门 ①：seed 数（§7/§16.3）
    n = len(g.records)
    warn = check_seed_count(n, min_seeds)
    if warn:
        g.reasons.append(warn)
    else:
        g.warnings.append(f"n_seeds={n} ≥ {min_seeds} ✓")

    # split 一致性：不同 split 混在一起会污染主表
    splits = sorted({(r.split or r.run.split) for r in g.records} - {""})
    if len(splits) > 1:
        g.reasons.append(f"split 不一致（{splits}）→ 不得聚合成一行")

    g.aggregates = aggregate_runs([r.run for r in g.records])
    # 逐题型的**多 seed 均值**（原始 0–1 量纲）→ 交给 macro_average_over_tasks 统一 ×100
    # （它内部已乘 100，这里再乘会得到 3750 这种双倍缩放值）
    per_task_agg: dict[str, dict] = {}
    for key, agg in g.aggregates.items():
        if key.startswith("task:") and key.endswith(":mra"):
            per_task_agg.setdefault(key.split(":")[1], {})["mra"] = agg.mean
        elif key.startswith("task:") and key.endswith(":accuracy"):
            per_task_agg.setdefault(key.split(":")[1], {})["accuracy"] = agg.mean
    if per_task_agg:
        g.macro = macro_average_over_tasks(per_task_agg)
        g.macro["_source"] = "per-task means over seeds (raw scale)" 
    if g.aggregates.get("accuracy") is not None:
        g.macro.setdefault("_acc_mean", g.aggregates["accuracy"].mean)
        g.macro.setdefault("_acc_std", g.aggregates["accuracy"].std)
    if g.aggregates.get("mra") is not None:
        g.macro.setdefault("_mra_mean", g.aggregates["mra"].mean)
        g.macro.setdefault("_mra_std", g.aggregates["mra"].std)

    g.eligible_for_main_table = not g.reasons
    return g


def build_main_table(groups: dict[str, Sequence[RunRecord]], *,
                     allow_mock: bool = False,
                     min_seeds: int = MIN_SEEDS_FOR_MAIN_TABLE) -> dict:
    """生成主表（多方法 × 多 seed）并给出资格判定。"""
    tables = [build_group(label, recs, allow_mock=allow_mock, min_seeds=min_seeds)
              for label, recs in groups.items()]
    # 生成门（是否让 CLI 成功退出）：mock 在 allow_mock 下不再阻断**生成**，
    # 但绝不影响 `eligible_for_main_table`（论文资格）
    generation_blocked = [
        t.label for t in tables
        if [r for r in t.reasons if not (allow_mock and t.mock_detected and "HC24" in r)]
    ]
    return {
        "schema": "skill3d-main-table-v1",
        "min_seeds_required": min_seeds,
        "allow_mock": bool(allow_mock),
        "groups": [t.as_dict() for t in tables],
        "any_eligible": any(t.eligible_for_main_table for t in tables),
        "all_eligible": all(t.eligible_for_main_table for t in tables) and bool(tables),
        "generation_blocked_groups": generation_blocked,
        "notes": [
            "资格判定即 HC34 的 paper_eligible 前置：seed 数、真实性、split 一致性",
            "写论文前仍须过 `python -m skill3d.readiness.manifest --check <capability>`",
        ],
    }


def format_main_table(table: dict) -> str:
    """控制台版式：每方法一行（官方 Table 10 口径）+ 逐方法 mean±std 明细。"""
    lines: list[str] = []
    rows = []
    for g in table["groups"]:
        agg = g["aggregates"] or {}
        per_task: dict[str, dict] = {}
        for key, val in agg.items():
            if not key.startswith("task:"):
                continue
            _, tname, kind = key.split(":", 2)
            per_task.setdefault(tname, {})[kind] = val["mean"]
        rows.append(format_main_table_row(g["label"], per_task))
    if rows:
        lines.append("主表（§8.3：Avg + 4 NA(MRA) + 4 MCA(Acc)，×100）:")
        lines.extend("  " + r for r in rows)
        miss = {g["label"]: (g["macro"].get("missing_tasks") or []) for g in table["groups"]}
        miss = {k: v for k, v in miss.items() if v}
        if miss:
            lines.append(f"  注意：以下配置缺任务（不参与平均，不臆造 0）: {miss}")
    for g in table["groups"]:
        lines.append(f"\n[{g['label']}] n_seeds={g['n_seeds']} "
                     f"eligible={g['eligible_for_main_table']}")
        for w in g["warnings"]:
            lines.append(f"  · {w}")
        for r in g["reasons"]:
            lines.append(f"  ✗ {r}")
        acc = (g["aggregates"] or {}).get("accuracy")
        mra = (g["aggregates"] or {}).get("mra")
        if acc:
            lines.append(f"  accuracy: {acc['mean']:.4f} ± {acc['std']:.4f} "
                         f"(n={acc['n_seeds']})")
        if mra:
            lines.append(f"  mra     : {mra['mean']:.4f} ± {mra['std']:.4f} "
                         f"(n={mra['n_seeds']})")
    return "\n".join(lines)


def _parse_groups(args) -> dict[str, list[RunRecord]]:
    """`--runs d1 d2 d3 --label Ours` 或重复 `--group name=d1,d2`。"""
    groups: dict[str, list[RunRecord]] = {}
    if args.runs:
        label = args.label or "run"
        groups[label] = [load_run(p) for p in args.runs]
    for spec in args.group or []:
        if "=" not in spec:
            raise MainTableError(f"--group 需要 name=dir[,dir…] 形式，收到 {spec!r}")
        name, paths = spec.split("=", 1)
        groups[name] = [load_run(p) for p in paths.split(",") if p.strip()]
    return groups


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="§7 多 seed 主表：≥3 seed mean±std + 资格判定（HC24/HC34）")
    p.add_argument("--runs", nargs="*", default=[],
                   help="同一方法的各 seed trace 目录（或单个 run JSON）")
    p.add_argument("--label", default="", help="--runs 的配置名（主表行名）")
    p.add_argument("--group", action="append", default=[],
                   help="name=dir[,dir…]（可重复，用于多方法对比）")
    p.add_argument("--out", default="", help="主表 JSON 落盘路径")
    p.add_argument("--allow-mock", action="store_true",
                   help="允许非 real 模式入表（仅管道验证，仍标 non-eligible）")
    p.add_argument("--min-seeds", type=int, default=MIN_SEEDS_FOR_MAIN_TABLE)
    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        groups = _parse_groups(args)
    except MainTableError as exc:
        print(f"[错误] {exc}", file=__import__("sys").stderr)
        return 2
    if not groups:
        print("[错误] 未提供任何 run：用 --runs <trace_dir…> 或 --group name=dir,dir",
              file=__import__("sys").stderr)
        return 2

    table = build_main_table(groups, allow_mock=args.allow_mock,
                             min_seeds=args.min_seeds)
    print(format_main_table(table))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n主表 JSON: {out}")

    if table["generation_blocked_groups"]:
        print(f"\n[门禁] 以下配置未过生成门禁（{table['generation_blocked_groups']}）"
              "→ 不得写入论文主表（§7/HC24/HC34）", file=__import__("sys").stderr)
        return 1
    if not table["all_eligible"]:
        print("\n[门禁] 已生成表，但存在 eligible=False 的配置"
              "（mock/来源不明等）→ 该表仅管道验证，不得写进论文主表"
              "（§7/HC24/HC34）", file=__import__("sys").stderr)
        return 0
    print("\n[门禁] 全部配置均满足 ≥3 seed + real 模式 + split 一致 → 可进主表"
          "（写论文前再过 readiness --check）")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI 入口
    raise SystemExit(main())
