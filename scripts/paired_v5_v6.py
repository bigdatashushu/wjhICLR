#!/usr/bin/env python3
"""v5 ↔ v6 逐题配对比较（§16.3 paired A/B + §18.4 统计）。

用法：
    python scripts/paired_v5_v6.py v5_dir v6_dir [--label-a v5 --label-b v6]

输入是两个 trace 目录（各自含 `evaluation_result.jsonl`）。两臂必须来自**同一
样本**（本脚本会核对 qa_id 集合完全一致，不一致即拒绝——§18.4 的"同帧/同输入"
是硬前提，不是口号）。

输出：
- 8 题型各自的 v5/v6 分（NA→MRA，MCA→Accuracy）与 95% CI；
- 主表口径（8 题型简单算术平均 ×100）；
- 逐题配对表（哪些题 v6 赢/输/平）；
- **不做显著性宣称**：McNemar 走 `evolution.paired_score`，退化样本自动判不显著。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
        if not line.strip():
            continue
        r = json.loads(line)
        out[str(r["qa_id"])] = r
    return out


def _score(r: dict, task: str):
    """单题得分：NA 题取 MRA，MCA 题取 0/1。无答案（unavailable）→ None（不进分母）。"""
    if task in NA_TASKS:
        return None if r.get("mra_value") is None else float(r["mra_value"])
    c = r.get("correct")
    return None if c is None else float(bool(c))


def _ci95(values: list[float]) -> tuple[float, float]:
    """Wilson 区间对二值；对 MRA（连续）用 1.96·SE（正态近似）。"""
    import math

    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    m = sum(values) / n
    if m in (0.0, 1.0) or all(v in (0.0, 1.0) for v in values):
        z = 1.96
        denom = 1 + z * z / n
        centre = (m + z * z / (2 * n)) / denom
        half = z * math.sqrt(m * (1 - m) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))
    sd = (sum((v - m) ** 2 for v in values) / max(1, n - 1)) ** 0.5
    half = 1.96 * sd / (n ** 0.5)
    return (max(0.0, m - half), min(1.0, m + half))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--label-a", default="v5")
    ap.add_argument("--label-b", default="v6")
    args = ap.parse_args()

    a, b = load_results(Path(args.dir_a)), load_results(Path(args.dir_b))
    if set(a) != set(b):
        raise SystemExit(
            f"两臂样本不一致（§16.3 硬前提）：仅 {args.label_a} 有 "
            f"{sorted(set(a) - set(b))[:6]}；仅 {args.label_b} 有 "
            f"{sorted(set(b) - set(a))[:6]}")
    print(f"配对样本 {len(a)} 题（qa_id 集合完全一致 → 同帧/同输入）\n")

    la, lb = args.label_a, args.label_b
    header = f"{'task':24s} {'n':>3s} {la:>18s} {lb:>18s}  {'Δ':>7s}"
    print(header)
    print("-" * len(header))
    per_task_b: dict[str, float] = {}
    per_task_a: dict[str, float] = {}
    for task in TASKS:
        qa = [q for q in a if a[q].get("task") == task]
        if not qa:
            continue
        va = [v for v in (_score(a[q], task) for q in qa) if v is not None]
        vb = [v for v in (_score(b[q], task) for q in qa) if v is not None]
        ma = sum(va) / len(va) if va else float("nan")
        mb = sum(vb) / len(vb) if vb else float("nan")
        ca, cb = _ci95(va), _ci95(vb)
        per_task_a[task], per_task_b[task] = ma, mb
        d = mb - ma if ma == ma and mb == mb else float("nan")
        print(f"{task:24s} {len(qa):3d} "
              f"{ma:7.3f}[{ca[0]:.2f},{ca[1]:.2f}] "
              f"{mb:7.3f}[{cb[0]:.2f},{cb[1]:.2f}]  {d:+7.3f}")

    def _main_table(pt: dict[str, float]) -> float:
        vals = [v for v in pt.values() if v == v]
        return 100.0 * sum(vals) / len(vals) if vals else float("nan")

    sa, sb = _main_table(per_task_a), _main_table(per_task_b)
    print(f"\n主表口径（8 题型简单算术平均 ×100）：{la}={sa:.2f}  {lb}={sb:.2f}  "
          f"Δ={sb - sa:+.2f}")

    # 逐题配对
    wins = losses = ties = 0
    rows = []
    for q in sorted(a, key=lambda x: int(x) if str(x).isdigit() else 0):
        task = a[q].get("task") or ""
        va, vb = _score(a[q], task), _score(b[q], task)
        if va is None or vb is None:
            continue
        tag = "=" if va == vb else ("B" if vb > va else "A")
        wins += tag == "B"
        losses += tag == "A"
        ties += tag == "="
        rows.append((q, task, va, vb, tag))
    print(f"\n逐题配对（可比 {len(rows)} 题）：{lb} 赢 {wins} / {la} 赢 {losses} / 平 {ties}")
    for q, task, va, vb, tag in rows:
        if tag != "=":
            mark = "→" if tag == "B" else "←"
            print(f"  {mark} qa={q:5s} {task:24s} {la}={va:.2f} {lb}={vb:.2f}")

    print("\n[纪律] 本表只报事实，不作显著性结论：inner_validation 上出的数只用于"
          "定策略（§18.1）；样本量 <32/题型时 SE 远大于效应量，不得当增益（§18.2）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
