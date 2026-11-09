"""Official per-task Accuracy/MRA summary for current runs."""
from typing import Optional
import numpy as np


def _slot_score(slot: dict) -> Optional[float]:
    """单个任务/档位的分数：MCA → accuracy；NA → mra；两者都有时按 n 取多数口径。"""
    acc = slot.get("accuracy")
    mra = slot.get("mra")
    n_mca = int(slot.get("n_mca", 0) or 0)
    n_na = int(slot.get("n_na", 0) or 0)
    if n_mca and acc is not None:
        return float(acc)
    if n_na and mra is not None:
        return float(mra)
    if acc is not None:
        return float(acc)
    return float(mra) if mra is not None else None


def macro_average_over_tasks(per_task: dict) -> dict:
    """§8.3 主表口径：4 NA(MRA) + 4 MCA(Acc) 共 8 任务**简单算术平均 × 100**，无加权。

    - `object_rel_direction` 的 easy/medium/hard 三档**先等权聚合**为该任务分
      （`per_task[task]["levels"]`），同时保留三档明细；
    - 缺失任务（该 run 没跑到）不参与平均，但会记进 `missing_tasks`（不臆造 0）；
    - 返回 `{"avg_x100", "per_task_x100": {...}, "scale": 100, "missing_tasks": [...]}`。
    """
    from skill3d.routing.task_classifier import TASK_TYPES

    per_task_x100: dict[str, float] = {}
    level_detail: dict[str, dict] = {}
    missing: list[str] = []
    for task in TASK_TYPES:
        slot = (per_task or {}).get(task)
        if not slot:
            missing.append(task)
            continue
        levels = {k: v for k, v in (slot.get("levels") or {}).items() if v}
        if len(levels) > 1:
            vals = [x for x in (_slot_score(v) for v in levels.values()) if x is not None]
            if not vals:
                missing.append(task)
                continue
            score = float(np.mean(vals))          # 三档等权（§8.3）
            level_detail[task] = {k: (None if _slot_score(v) is None
                                      else round(_slot_score(v) * 100, 4))
                                  for k, v in levels.items()}
        else:
            score = _slot_score(slot)
            if score is None:
                missing.append(task)
                continue
        per_task_x100[task] = round(score * 100, 4)
    # per_task_x100 已是 ×100 的量纲 → 直接取平均（不要再次 ×100）
    avg = round(float(np.mean(list(per_task_x100.values()))), 4) if per_task_x100 else None
    return {
        "avg_x100": avg,
        "per_task_x100": per_task_x100,
        "level_detail_x100": level_detail,
        "n_tasks": len(per_task_x100),
        "missing_tasks": missing,
        "scale": 100,
        "note": "§8.3：4 NA(MRA) + 4 MCA(Acc) 简单算术平均 ×100，无加权；"
                "rel_direction 三档先等权聚合",
    }


def format_main_table_row(name: str, per_task: dict) -> str:
    """按官方 Table 10 版式打印一行：`name | Avg | 4×NA(MRA) | 4×MCA(Acc)`（全 ×100）。"""
    m = macro_average_over_tasks(per_task)
    na = [("object_counting", "object_abs_distance", "object_size_estimation",
           "room_size_estimation")]
    mca = [("object_rel_distance", "object_rel_direction", "route_planning",
            "obj_appearance_order")]
    def _get(t: str) -> str:
        v = m["per_task_x100"].get(t)
        return "n/a" if v is None else f"{v:.2f}"
    cells = [f"{m['avg_x100']:.2f}" if m["avg_x100"] is not None else "n/a"]
    cells += [_get(t) for t in na[0]] + [_get(t) for t in mca[0]]
    return " | ".join([name] + cells)
