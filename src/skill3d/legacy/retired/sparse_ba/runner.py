"""`vggt_sparse_ba` L1 单 episode 驱动（§10.1 L1 / HC36）。

流程：32 帧 → SuperPoint/ALIKED 特征（流式）→ 预注册 pair graph → LightGlue 逐 pair
匹配 + RANSAC → track 合并（L0 不变量）→ PyCOLMAP 三角化 + BA → G5 → L1 门槛 →
写 `SparseBAReceipt`（**独立**于 `vggt` artifact，不覆盖、不复用名称）。

**止损纪律**：L1 任一硬门失败 → `rejected`，关闭生产接线，正式主线继续用 `vggt`；
不得通过降分辨率/减帧/双卡池化"救"它。

用法（真实运行，独占一张空闲卡）：

```bash
python -m skill3d.reconstruction.sparse_ba.runner \
    --scene 41069043 --frames data/v5_smoke/recon/vggt/41069043_frames \
    --c2w data/v5_smoke/recon/vggt/41069043_c2w.npy \
    --intrinsics data/v5_smoke/recon/vggt/41069043_intrinsics.npy \
    --depth data/v5_smoke/recon/vggt/41069043_depth.npy \
    --device cuda:0 --out-dir data/v5_sparse_ba
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import SparseBAReceipt

# 前端名（与 SparseBAReceipt.frontend 的受控枚举一致；HC36 只允许这两个）
FRONTENDS = ("superpoint_lightglue", "aliked_lightglue")


def _load_frames(frames_dir: str | Path, limit: Optional[int] = None) -> list[np.ndarray]:
    import cv2

    files = sorted(Path(frames_dir).glob("*.png")) + sorted(Path(frames_dir).glob("*.jpg"))
    if limit:
        files = files[:int(limit)]
    out = []
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法解码帧 {f}（输入合法性失败，不得静默跳过）")
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not out:
        raise RuntimeError(f"{frames_dir} 下没有可解码帧")
    return out


def run_l1(
    frames: Sequence[np.ndarray],
    *,
    c2w: np.ndarray,
    intrinsics: np.ndarray,
    src_hw: Optional[tuple[int, int]] = None,
    frontend: str = "superpoint_lightglue",
    device: str = "cuda",
    pair_graph_cfg=None,
    cache_dir: Optional[str | Path] = None,
) -> tuple[SparseBAReceipt, Optional[object]]:
    """跑一次 L1（返回回执 + BA 结果；任何失败都写回执，不抛给上层做静默回退）。"""
    import torch

    from .matching import MatchingError, match_pairs_streaming
    from .pair_graph import PairGraphConfig, build_pair_graph, l0_pair_graph_ok
    from .pycolmap_backend import (
        PycolmapUnavailable,
        build_ba_inputs,
        triangulate_and_ba,
    )

    t0 = time.time()
    graph = build_pair_graph(pair_graph_cfg or PairGraphConfig(n_frames=len(frames)))
    ok_graph, graph_problems = l0_pair_graph_ok(graph)
    if not ok_graph:
        return SparseBAReceipt(
            status="rejected", frontend=frontend,  # type: ignore[arg-type]
            pair_graph_hash=graph.pair_graph_hash, n_pairs=len(graph.pairs),
            n_matches=0, n_inliers=0, n_tracks=0, peak_gpu_gib=0.0,
            skip_reason="l0_pair_graph: " + "; ".join(graph_problems[:2]),
            rejected_reason="rejected_on_l0_gate"), None

    def _fail(reason: str, peak: float = 0.0, n_matches: int = 0,
              n_inliers: int = 0, n_tracks: int = 0,
              initial_cost: Optional[float] = None,
              final_cost: Optional[float] = None) -> tuple[SparseBAReceipt, None]:
        receipt = SparseBAReceipt(
            status="failed", frontend=frontend,  # type: ignore[arg-type]
            pair_graph_hash=graph.pair_graph_hash, n_pairs=len(graph.pairs),
            n_matches=int(n_matches), n_inliers=int(n_inliers), n_tracks=int(n_tracks),
            peak_gpu_gib=float(peak), skip_reason=reason,
            initial_cost=initial_cost, final_cost=final_cost,
            wallclock_s=float(time.time() - t0))
        return receipt, None

    # ---- 前端特征（逐帧流式）----
    try:
        from .features import extract_frames, scale_intrinsics

        feats, feat_hw = extract_frames(
            frames, frontend=frontend, device=device,  # type: ignore[arg-type]
            cache_path=(Path(cache_dir) / "features.npz") if cache_dir else None)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"features: {type(exc).__name__}: {exc}")
    peak = float(torch.cuda.max_memory_allocated() / (2 ** 30)) if device == "cuda" else 0.0

    # ---- 逐 pair 匹配（流式）----
    try:
        report = match_pairs_streaming(feats, graph.pairs, frontend=frontend, device=device)
    except MatchingError as exc:
        return _fail(f"matching: {exc}", peak=peak)
    peak = max(peak, float(report.peak_gpu_gib))

    # ---- track 合并 + 不变量 ----
    src = src_hw or (int(np.asarray(frames[0]).shape[0]), int(np.asarray(frames[0]).shape[1]))
    k_feat = scale_intrinsics(np.asarray(intrinsics)[0], src, feat_hw)
    k_all = np.broadcast_to(k_feat, (len(frames), 3, 3)).copy()
    try:
        inputs = build_ba_inputs(feats, report, c2w=np.asarray(c2w), intrinsics=k_all,
                                 min_track_length=3)
    except PycolmapUnavailable as exc:
        return _fail(f"tracks: {exc}", peak=peak, n_matches=report.n_matches,
                     n_inliers=report.n_inliers)

    # ---- 三角化 + BA ----
    try:
        ba = triangulate_and_ba(inputs, min_track_length=3)
    except PycolmapUnavailable as exc:
        return _fail(f"pycolmap: {exc}", peak=peak, n_matches=report.n_matches,
                     n_inliers=report.n_inliers, n_tracks=len(inputs.tracks))
    if not ba.ok:
        # oom 归类（用于 §10.1 的 oom_rate 统计与拒绝码）
        low = ba.reason.lower()
        reason = ("oom: " + ba.reason) if ("out of memory" in low or "oom" in low) else ba.reason
        return _fail(reason, peak=peak, n_matches=report.n_matches,
                     n_inliers=report.n_inliers, n_tracks=len(inputs.tracks),
                     initial_cost=ba.initial_cost, final_cost=ba.final_cost)

    receipt = SparseBAReceipt(
        status="passed", frontend=frontend,  # type: ignore[arg-type]
        pair_graph_hash=graph.pair_graph_hash, n_pairs=len(graph.pairs),
        n_matches=int(report.n_matches), n_inliers=int(report.n_inliers),
        n_tracks=int(len(inputs.tracks)), peak_gpu_gib=float(peak),
        initial_cost=ba.initial_cost, final_cost=ba.final_cost,
        wallclock_s=float(time.time() - t0))
    return receipt, ba


def main(argv: Optional[Sequence[str]] = None) -> int:
    from skill3d.reconstruction.sparse_ba.receipts import l1_gate, write_receipt

    p = argparse.ArgumentParser(
        prog="python -m skill3d.reconstruction.sparse_ba.runner",
        description="`vggt_sparse_ba` L1 单 episode PoC（§10.1；失败即止损，HC36）")
    p.add_argument("--scene", required=True)
    p.add_argument("--frames", required=True, help="帧目录（32 张 PNG/JPG）")
    p.add_argument("--c2w", required=True, help="VGGT c2w .npy（作为 BA 初值）")
    p.add_argument("--intrinsics", required=True, help="VGGT 内参 .npy")
    p.add_argument("--frontend", default="superpoint_lightglue", choices=list(FRONTENDS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="data/v5_sparse_ba")
    p.add_argument("--max-frames", type=int, default=0)
    args = p.parse_args(argv)

    frames = _load_frames(args.frames, limit=args.max_frames or None)
    c2w = np.load(args.c2w)
    k = np.load(args.intrinsics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    receipt, ba = run_l1(frames, c2w=c2w, intrinsics=k, frontend=args.frontend,
                         device=args.device, cache_dir=out_dir)
    ok, problems = l1_gate(receipt)
    if not ok:
        from skill3d.reconstruction.sparse_ba.receipts import stop_loss

        receipt = stop_loss(receipt, gate="l1", problems=problems)
    receipt_path = write_receipt(receipt, out_dir / f"{args.scene}_sparse_ba_receipt.json")
    print(f"receipt: {receipt_path}")
    print(f"  status={receipt.status} frontend={receipt.frontend} "
          f"peak_gpu={receipt.peak_gpu_gib:.2f}GiB")
    print(f"  pairs={receipt.n_pairs} matches={receipt.n_matches} "
          f"inliers={receipt.n_inliers} tracks={receipt.n_tracks}")
    print(f"  ba cost {receipt.initial_cost} → {receipt.final_cost}")
    if not ok:
        print("  L1 硬门未过 → rejected（止损：关闭生产接线，继续用 vggt）")
        for pr in problems:
            print(f"    - {pr}")
        return 1
    if ba is not None and getattr(ba, "c2w_refined", None) is not None:
        np.save(out_dir / f"{args.scene}_sparse_ba_c2w.npy", ba.c2w_refined)
        np.save(out_dir / f"{args.scene}_sparse_ba_points.npy", ba.points3D)
        np.save(out_dir / f"{args.scene}_sparse_ba_reproj.npy", ba.reproj_errors)
        meta = {"scene_name": args.scene, "recon_method": "vggt_sparse_ba",
                "frontend": receipt.frontend,
                "pair_graph_hash": receipt.pair_graph_hash,
                "g5_reproj_err_median": ba.g5_median, "g5_reproj_err_p95": ba.g5_p95,
                "peak_gpu_gib": receipt.peak_gpu_gib,
                "note": ("L1 通过不等于可进论文：仅 `paper_eligible` 后才可作为可选消融行"
                         "（HC36）；且 BA 不恢复米制尺度")}
        (out_dir / f"{args.scene}_sparse_ba_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  g5: median={ba.g5_median} p95={ba.g5_p95}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
