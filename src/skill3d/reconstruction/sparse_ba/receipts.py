"""`vggt_sparse_ba` 回执落盘与 §10.1 门槛判定（L0/L1/L2）。

**回执是止损纪律的唯一凭证**：任何失败都要写 receipt（含 `skip_reason` /
`rejected_reason`），正式主线继续用 `vggt`，不得"再试一次"扩范围（HC36）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

from skill3d.schemas.sparse_ba import SparseBAReceipt

from . import (
    L2_MIN_EPISODES,
    MAX_PEAK_GPU_GIB,
    MAX_SKIP_RATE,
    MIN_HEADROOM_GIB,
)


def write_receipt(receipt: SparseBAReceipt, path: Union[str, Path]) -> Path:
    """原子写回执（先写临时文件再 replace）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def read_receipt(path: Union[str, Path]) -> SparseBAReceipt:
    return SparseBAReceipt.model_validate_json(Path(path).read_text(encoding="utf-8"))


def l1_gate(receipt: SparseBAReceipt, *,
            peak_gpu_gib_limit: float = MAX_PEAK_GPU_GIB,
            headroom_gib: float = MIN_HEADROOM_GIB) -> tuple[bool, list[str]]:
    """L1 单 episode 硬门（§10.1）：不 OOM、峰值显存达标、BA 收敛、残差有限。"""
    problems: list[str] = []
    if receipt.status != "passed":
        problems.append(f"status={receipt.status}（skip_reason={receipt.skip_reason}）")
    if receipt.peak_gpu_gib > peak_gpu_gib_limit:
        problems.append(f"峰值显存 {receipt.peak_gpu_gib:.2f} GiB > 上限 "
                        f"{peak_gpu_gib_limit:.2f} GiB")
    # 24 GiB 卡上"至少留 headroom"：峰值 + headroom 必须仍装得下
    if receipt.peak_gpu_gib + headroom_gib > 24.0:
        problems.append(f"峰值 {receipt.peak_gpu_gib:.2f} GiB + 余量 {headroom_gib:.1f} GiB "
                        "超过 24 GiB 卡容量")
    if receipt.initial_cost is None or receipt.final_cost is None:
        problems.append("缺少 BA cost（initial/final）")
    elif not receipt.final_cost < receipt.initial_cost:
        problems.append(f"BA 未收敛：final_cost={receipt.final_cost} "
                        f">= initial_cost={receipt.initial_cost}")
    if receipt.n_tracks <= 0:
        problems.append("无有效 track")
    return (not problems), problems


def l2_gate(receipts: Sequence[SparseBAReceipt], *,
            max_skip_rate: float = MAX_SKIP_RATE,
            min_episodes: int = L2_MIN_EPISODES) -> tuple[bool, list[str], dict]:
    """L2 规模硬门（§10.1）：样本量、OOM 率、预注册 skip_rate。

    预注册门槛必须在看结果前冻结；本函数只做判定，**不做**任何事后放宽。
    """
    problems: list[str] = []
    items = list(receipts)
    n = len(items)
    n_oom = sum(1 for r in items if (r.skip_reason or "").lower().startswith("oom"))
    n_skipped = sum(1 for r in items if r.status in ("failed", "rejected"))
    n_passed = sum(1 for r in items if r.status == "passed")
    stats = {
        "n_episodes": n,
        "n_passed": n_passed,
        "n_oom": n_oom,
        "oom_rate": (n_oom / n) if n else 0.0,
        "skip_rate": (n_skipped / n) if n else 0.0,
        "success_rate": (n_passed / n) if n else 0.0,
        "peak_gpu_gib_max": max((r.peak_gpu_gib for r in items), default=0.0),
    }
    if n < int(min_episodes):
        problems.append(f"样本量 {n} < 预注册最小 {min_episodes}（N=10 仅诊断，无升级资格）")
    if stats["oom_rate"] > 0.0:
        problems.append(f"OOM_rate={stats['oom_rate']:.3f} ≠ 0")
    if stats["skip_rate"] > max_skip_rate:
        problems.append(f"skip_rate={stats['skip_rate']:.3f} > 预注册上限 {max_skip_rate}")
    return (not problems), problems, stats


def stop_loss(receipt: SparseBAReceipt, *, gate: str,
              problems: Sequence[str]) -> SparseBAReceipt:
    """统一止损：把 receipt 标为 `rejected` 并写明理由（HC36 一次性止损）。"""
    reason = ("rejected_on_24g_oom"
              if (receipt.skip_reason or "").lower().startswith("oom")
              else f"rejected_on_{gate}_gate")
    note = "; ".join(problems) if problems else "未通过门槛"
    return receipt.model_copy(update={
        "status": "rejected",
        "rejected_reason": reason,
        "skip_reason": (receipt.skip_reason or note),
    })


def receipts_report(receipts: Sequence[SparseBAReceipt]) -> dict:
    """L2 报告（§10.1：成功率/时延/峰值显存/有效 pair/内点/track 数完整报告）。"""
    items = list(receipts)
    if not items:
        return {"n_episodes": 0}
    _, problems, stats = l2_gate(items)
    return {
        **stats,
        "n_pairs_total": sum(r.n_pairs for r in items),
        "n_matches_total": sum(r.n_matches for r in items),
        "n_inliers_total": sum(r.n_inliers for r in items),
        "n_tracks_total": sum(r.n_tracks for r in items),
        "wallclock_s_total": sum(r.wallclock_s or 0.0 for r in items),
        "frontends": sorted({r.frontend for r in items}),
        "pair_graph_hashes": sorted({r.pair_graph_hash for r in items}),
        "statuses": {s: sum(1 for r in items if r.status == s)
                     for s in ("not_run", "passed", "failed", "rejected")},
        "gate_problems": problems,
        "paper_eligible": False,   # L2 通过也只进可选消融，永不升级为主线（HC36）
    }


def dump_json(obj: dict, path: Union[str, Path]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_receipts(paths: Sequence[Union[str, Path]]) -> list[SparseBAReceipt]:
    out: list[SparseBAReceipt] = []
    for p in paths:
        try:
            out.append(read_receipt(p))
        except Exception:  # noqa: BLE001 - 审计脚本：坏回执跳过并继续
            continue
    return out


def summarize_peak(receipts: Sequence[SparseBAReceipt]) -> Optional[float]:
    vals = [r.peak_gpu_gib for r in receipts if r.peak_gpu_gib > 0]
    return max(vals) if vals else None
