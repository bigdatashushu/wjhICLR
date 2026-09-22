#!/usr/bin/env python3
"""§18.3 噪声底报告（同配置 ≥3 seed）+ §18.2 样本量口径。

用法：
    python scripts/noise_floor.py \
        --arm C1=data/v6_inner128/c1_seed0/traces,data/v6_inner128/c1_seed1/traces,data/v6_inner128/c1_seed2/traces \
        --arm C0=data/v6_inner128/c0_seed0/traces,data/v6_inner128/c0_seed1/traces,data/v6_inner128/c0_seed2/traces

每个 `--arm NAME=dir1,dir2,...` 是一组**同配置、不同 seed** 的 trace 目录。
脚本做三件事：

1. **配对前提校验**（§18.4 硬前提）：同一 arm 内各 seed 的 qa_id 集合必须完全一致
   （抽样在 meta 行序上先到先得、与 seed 无关，理应一致）；不一致即拒绝，不print 数字。
2. **逐 seed 主表 + 跨 seed 波动**：8 题型算术平均 ×100，并给出 per-task 的
   min/max/极差 —— 这是"某个改动值不值得信"的直接标尺。
3. **§18.3 噪声底**：逐题一致率（同一题 3 个 seed 结果全同的比例）；
   `<3 seed` 时显式标注"不满足噪声底，任何差异都不能当增益"。

纪律：本脚本**不做显著性宣称**，也不把 seed 波动当"增益"；它唯一的用途是给出
"小于这个波动就不算改动有效"的门槛。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill3d.evaluation.experiment_protocol import noise_floor_report  # noqa: E402

TASKS = ("object_counting", "object_abs_distance", "object_size_estimation",
         "room_size_estimation", "object_rel_distance", "object_rel_direction",
         "route_planning", "obj_appearance_order")
NA_TASKS = frozenset({"object_counting", "object_abs_distance",
                      "object_size_estimation", "room_size_estimation"})


def load_results(d: Path) -> dict[str, dict]:
    f = Path(d) / "evaluation_result.jsonl"
    if not f.is_file():
        raise SystemExit(f"缺 {f}（该目录不是一次在线评测的 trace 目录）")
    out: dict[str, dict] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["qa_id"])] = r
    return out


def score(r: dict, task: str):
    """单题得分：NA 题取 MRA，MCA 题取 0/1；无答案 → None（不进分母，§16.4）。"""
    if task in NA_TASKS:
        return None if r.get("mra_value") is None else float(r["mra_value"])
    c = r.get("correct")
    return None if c is None else float(bool(c))


def per_task(runs: list[dict[str, dict]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for task in TASKS:
        vals = []
        for res in runs:
            qa = [q for q in res if res[q].get("task") == task]
            v = [score(res[q], task) for q in qa]
            v = [x for x in v if x is not None]
            vals.append(sum(v) / len(v) if v else float("nan"))
        out[task] = vals
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=dir1,dir2,...",
                    help="一组同配置不同 seed 的 trace 目录（逗号分隔）")
    ap.add_argument("--json-out", default="", help="可选：把结构化结果写到这里")
    args = ap.parse_args()

    payload: dict = {"arms": {}}
    for spec in args.arm:
        name, _, dirs_s = spec.partition("=")
        dirs = [Path(d) for d in dirs_s.split(",") if d.strip()]
        if not dirs:
            raise SystemExit(f"--arm {name} 没有给出目录")
        runs = [load_results(d) for d in dirs]

        # ---- 1. 配对前提：各 seed 的 qa_id 集合必须完全一致 ----
        base = set(runs[0])
        for i, r in enumerate(runs[1:], 1):
            if set(r) != base:
                raise SystemExit(
                    f"[{name}] seed#{i} 样本与 seed#0 不一致（§18.4 硬前提）："
                    f"仅 seed#{i} 有 {sorted(set(r) - base)[:6]}；"
                    f"仅 seed#0 有 {sorted(base - set(r))[:6]}")
        n_qa = len(base)
        print("=" * 78)
        print(f"[{name}] {len(dirs)} 个 seed × {n_qa} 题（qa_id 集合一致 → 同样本）")
        print("=" * 78)

        pt = per_task(runs)

        # ---- 2. 逐 seed 主表 ----
        print(f"{'task':24s} " + " ".join(f"{'s'+str(i):>7s}" for i in range(len(dirs)))
              + f" {'range':>8s}")
        tables = []
        for i, res in enumerate(runs):
            vals = [pt[t][i] for t in TASKS if pt[t][i] == pt[t][i]]
            tables.append(100.0 * sum(vals) / len(vals) if vals else float("nan"))
        for task in TASKS:
            vals = pt[task]
            cells = " ".join(f"{v:7.3f}" if v == v else f"{'n/a':>7s}" for v in vals)
            fin = [v for v in vals if v == v]
            rng = f"{max(fin) - min(fin):8.3f}" if fin else f"{'n/a':>8s}"
            print(f"{task:24s} {cells} {rng}")
        print(f"{'主表(×100)':24s} "
              + " ".join(f"{t:7.2f}" for t in tables)
              + f" {max(tables) - min(tables):8.2f}")

        # ---- 3. §18.3 噪声底 ----
        per_q: dict[str, list] = {}
        for q in base:
            task = runs[0][q].get("task") or ""
            vals = [score(r[q], task) for r in runs]
            per_q[q] = [(None if v is None else v >= 0.5) for v in vals]
        rep = noise_floor_report(per_q, per_task_scores=pt, n_seeds=len(dirs))
        ag = rep.agreement
        print(f"\n噪声底（§18.3）：seeds={rep.n_seeds} "
              f"ok_floor={rep.ok_noise_floor} ok_paper={rep.ok_paper}")
        print(f"  逐题一致率 = {rep.per_question_agreement_rate} "
              f"（全同 {ag.get('n_unanimous')} / 分裂 {ag.get('n_split')} / 计入 "
              f"{ag.get('n_questions_observed')}）")
        if ag.get("split_questions"):
            print(f"  分裂题（seed 间不一致）: {ag['split_questions']}")
        for n in rep.notes:
            print(f"  [note] {n}")
        fin_tab = [t for t in tables if t == t]
        if len(fin_tab) >= 2:
            print(f"  → 主表跨 seed 极差 = {max(fin_tab) - min(fin_tab):.2f} 分"
                  f"（小于此值的改动不得称增益）")
        else:
            print("  → 不足 2 个 seed：**没有**可报的波动（不得把 0 当成噪声底）")

        payload["arms"][name] = {
            "dirs": [str(d) for d in dirs], "n_qa": n_qa,
            "per_task": {t: v for t, v in pt.items()},
            "main_table_per_seed": tables,
            "main_table_range": (max(fin_tab) - min(fin_tab)) if fin_tab else None,
            "noise_floor": rep.as_dict(),
        }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"\n已写 {args.json_out}")

    print("\n[纪律] 本报告只给噪声门槛，不作显著性宣称（§18.3/§18.4）；"
          "inner 上的数只用于定策略（§18.1）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
