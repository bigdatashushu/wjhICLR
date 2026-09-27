#!/usr/bin/env python3
"""v7 inner 档主实验汇总：C1（工具+程序+yield）vs C0（直答 VLM），跨 3 seed。

用法：
    python scripts/report_v7_inner128.py [--root data/v7_inner128] [--ref-root data/v6_inner128]

做三件事，都只用落盘的 `evaluation_result.jsonl`（不重跑、不看代码内常量）：

1. **跨 seed 汇总**：每个臂的 coverage / MRA / Accuracy / 主表分（8 题型算术平均 ×100），
   并给出 seed 之间的极差（噪声底的粗略下界）。
2. **逐题配对**：C1 与 C0 在**同一批 qa_id** 上配对（不满足则拒绝比较），
   报主表差值、赢/输/平计数，以及按题型的分差。
3. **可选对照**：给 `--ref-root` 时把 v6 的同名臂一起列出，便于看"v7 相对 v6 改了多少"。

纪律（§16.1/§16.4）：本脚本**不做显著性宣称** —— 3 个 seed 且同 scene 内题目不独立，
主区间应按 scene 聚类重采样；这里只报描述统计，显著性留给 `evolution.paired_score`
与预登记的聚类 bootstrap。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

TASKS = ("object_counting", "object_abs_distance", "object_size_estimation",
         "room_size_estimation", "object_rel_distance", "object_rel_direction",
         "route_planning", "obj_appearance_order")
NA_TASKS = frozenset({"object_counting", "object_abs_distance",
                      "object_size_estimation", "room_size_estimation"})


def load(d: Path) -> dict[str, dict]:
    f = Path(d) / "traces" / "evaluation_result.jsonl"
    if not f.is_file():
        f = Path(d) / "evaluation_result.jsonl"
    if not f.is_file():
        raise SystemExit(f"缺 evaluation_result.jsonl: {d}")
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["qa_id"])] = r
    return out


def score_of(r: dict) -> float:
    """单题得分：MCA → 0/1；NA → MRA（无值按 0，与官方"无答案计 0"一致）。"""
    if r.get("is_mca"):
        return 1.0 if r.get("correct") else 0.0
    v = r.get("mra_value")
    return 0.0 if v is None else float(v)


def arm_summary(seed_dirs: list[Path]) -> dict:
    per_task_seed: dict[str, list[float]] = {t: [] for t in TASKS}
    cov, main = [], []
    for d in seed_dirs:
        rows = load(d)
        if not rows:
            continue
        for t in TASKS:
            rs = [r for r in rows.values() if r.get("task") == t]
            if rs:
                per_task_seed[t].append(statistics.mean(score_of(r) for r in rs))
        cov.append(statistics.mean(0.0 if r.get("abstained") else 1.0
                                   for r in rows.values()))
        if all(per_task_seed[t] for t in TASKS):
            main.append(statistics.mean(
                statistics.mean(per_task_seed[t]) for t in TASKS))
    task_mean = {t: (statistics.mean(v) if v else float("nan"))
                 for t, v in per_task_seed.items()}
    task_spread = {t: ((max(v) - min(v)) if len(v) > 1 else 0.0)
                   for t, v in per_task_seed.items()}
    return {"per_task": task_mean, "per_task_seed_spread": task_spread,
            "coverage": statistics.mean(cov) if cov else float("nan"),
            "main": statistics.mean(main) if main else float("nan"),
            "n_seeds": len(main),
            "seed_dirs": [str(d) for d in seed_dirs]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/v7_inner128")
    ap.add_argument("--ref-root", default="")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    root = Path(a.root)

    arms = {}
    for name in ("c1", "c0"):
        dirs = [root / f"{name}_seed{s}" for s in range(a.seeds)]
        dirs = [d for d in dirs if d.is_dir()]
        if dirs:
            arms[name] = arm_summary(dirs)
    if not arms:
        raise SystemExit(f"{root} 下没有 c1_seed*/c0_seed* 目录")

    print(f"=== v7 inner 档汇总（root={root}）===")
    for name, s in arms.items():
        print(f"\n[{name.upper()}] seeds={s['n_seeds']} coverage={s['coverage']:.4f} "
              f"主表={s['main'] * 100:.2f}")
        for t in TASKS:
            sp = s['per_task_seed_spread'].get(t, 0.0)
            print(f"   {t:<24} {s['per_task'][t]:.4f}  (seed 极差 {sp:.4f})")

    if "c1" in arms and "c0" in arms:
        print("\n=== 题型对照（C1 − C0；正 = 工具臂更好）===")
        for t in TASKS:
            d = arms["c1"]["per_task"][t] - arms["c0"]["per_task"][t]
            print(f"   {t:<24} C1={arms['c1']['per_task'][t]:.4f} "
                  f"C0={arms['c0']['per_task'][t]:.4f}  Δ={d:+.4f}")
        dm = arms["c1"]["main"] - arms["c0"]["main"]
        print(f"\n   主表：C1={arms['c1']['main'] * 100:.2f} "
              f"C0={arms['c0']['main'] * 100:.2f}  Δ={dm * 100:+.2f}")

        # 逐题配对（用 seed0 的两臂，qa_id 必须完全一致）
        d1 = [Path(p) for p in arms["c1"]["seed_dirs"]]
        d0 = [Path(p) for p in arms["c0"]["seed_dirs"]]
        if d1 and d0:
            r1, r0 = load(d1[0]), load(d0[0])
            common = sorted(set(r1) & set(r0))
            if set(r1) != set(r0):
                print(f"\n   [warn] 两臂 qa_id 集合不一致"
                      f"（C1={len(r1)} C0={len(r0)} 共同={len(common)}）"
                      "→ 配对统计只在共同题上进行")
            win = sum(1 for q in common if score_of(r1[q]) > score_of(r0[q]))
            lose = sum(1 for q in common if score_of(r1[q]) < score_of(r0[q]))
            tie = len(common) - win - lose
            print(f"\n   逐题配对（{len(common)} 题，seed0）：C1 赢 {win} / "
                  f"C0 赢 {lose} / 平 {tie}")

    if a.ref_root:
        ref = Path(a.ref_root)
        print(f"\n=== 参考：v6（root={ref}）===")
        for name in ("c1", "c0"):
            dirs = [ref / f"{name}_seed{s}" for s in range(a.seeds)]
            dirs = [d for d in dirs if d.is_dir()]
            if dirs:
                s = arm_summary(dirs)
                print(f"   {name.upper()}: seeds={s['n_seeds']} "
                      f"coverage={s['coverage']:.4f} 主表={s['main'] * 100:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
