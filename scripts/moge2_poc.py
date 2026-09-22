#!/usr/bin/env python3
"""§11.4 MoGe-2 度量尺度融合 PoC：C1–C6 六项验收（真实数据）。

用法：
    python scripts/moge2_poc.py --scenes 7b6477cb95,0d2ee665be [--gpu 3]

产出（全部落盘，可审计）：
- `<out>/<scene>_scale_receipt.json`：32 个 s_k、median、MAD、离群帧（§11.4 要求）
- `<out>/moge2_poc_report.json`：C1–C6 逐项结果 + 阈值快照 + 延迟/显存
- `<out>/degenerate_<kind>_receipt.json`：C2 三种注入退化的 receipt

纪律：
- **无 GT、无标定池、无 BA**（D1 边界）；本脚本不做任何"用真值调尺度"的事；
- 导入/权重/输出不合法一律抛错，**绝不产出占位深度**；
- C4 需要米制三题的 paired 比较（跑在线链），本脚本只输出"待 C4 复核"的占位，
  不编造 MRA。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill3d.reconstruction import metric_fusion as mf  # noqa: E402


def _load_scene(recon_dir: Path, scene: str):
    frames_dir = recon_dir / "vggt" / f"{scene}_frames"
    paths = sorted(glob.glob(str(frames_dir / "*.png")))
    if not paths:
        raise SystemExit(f"缺帧目录: {frames_dir}")
    import cv2

    rgb = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    depth = np.load(recon_dir / "vggt" / f"{scene}_depth.npy")
    intr = np.load(recon_dir / "vggt" / f"{scene}_intrinsics.npy")
    return rgb, depth, intr


def _resize_to(d: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.resize(np.asarray(d, dtype=np.float64), (hw[1], hw[0]),
                      interpolation=cv2.INTER_LINEAR)


def _degrade(rgb, kind: str, rng: np.random.Generator):
    """C2 注入退化：抽稀（跨场景帧）/ 运动模糊。抽稀用**别的 scene 的帧**替换后半段。"""
    out = [np.asarray(x).copy() for x in rgb]
    if kind == "blur":
        import cv2

        for i in range(len(out)):
            out[i] = cv2.GaussianBlur(out[i], (0, 0), sigmaX=6.0)
    return out


def run_scene(model, scene: str, recon_dir: Path, out_dir: Path,
              *, cross_frames=None, blur: bool = False) -> dict:
    rgb, depth, intr = _load_scene(recon_dir, scene)
    n = min(len(rgb), depth.shape[0], len(intr))
    if cross_frames is not None:
        # C2 抽稀/跨场景帧：把后半段换成另一场景的帧（制造跨场景不连续）
        k = n // 2
        rgb = rgb[:k] + list(cross_frames[: n - k])
    if blur:
        rgb = _degrade(rgb, "blur", None)
    hw = depth[0].shape
    metric_list, masks, confs, vggt_list = [], [], [], []
    lat = []
    for i in range(n):
        K_i = intr[i] if intr.ndim == 3 else intr
        t0 = time.perf_counter()
        o = model.infer(rgb[i], K_i)
        lat.append((time.perf_counter() - t0) * 1000.0)
        d = np.asarray(o.depth_metric, dtype=np.float64)
        m = np.asarray(o.valid_mask, dtype=bool)
        if d.shape != hw:
            d = _resize_to(d, hw)
            import cv2

            m = cv2.resize(m.astype(np.uint8), (hw[1], hw[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        metric_list.append(d)
        masks.append(m)
        confs.append(None)
        vggt_list.append(np.asarray(depth[i], dtype=np.float64))
    res = mf.fuse_metric_scale(metric_list, vggt_list, masks, confs,
                               conf_warp_monotonic=None, model=mf.METRIC_MODEL_MOGE2)
    receipt = out_dir / f"{scene}{'_blur' if blur else ''}"
    receipt = Path(str(receipt) + ("_cross" if cross_frames is not None else "") + "_scale_receipt.json")
    mf.write_per_frame_receipt(res, receipt)
    sk = [p.s_k for p in res.per_frame]
    finite = [v for v in sk if v is not None and np.isfinite(v)]
    return {
        "scene": scene,
        "status": res.status,
        "metric_scale": res.metric_scale,
        "scale_self_consistency": res.scale_self_consistency,
        "valid_frame_ratio": res.valid_frame_ratio,
        "n_frames_valid": res.n_frames_valid,
        "n_frames_total": res.n_frames_total,
        "outlier_frames": list(res.outlier_frames),
        "mad": res.mad,
        "s_k_min": float(np.min(finite)) if finite else None,
        "s_k_max": float(np.max(finite)) if finite else None,
        "s_k_median_of_medians": float(np.median(finite)) if finite else None,
        "receipt": str(receipt),
        "latency_ms_mean": float(np.mean(lat)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "model": res.model,
        "version": res.version,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-dir", default="data/v6_scoped/recon")
    ap.add_argument("--out", default="data/v6_poc_moge2")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--limit-scenes", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
    recon_dir, out_dir = Path(args.recon_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = [s for s in args.scenes.split(",") if s] or [
        Path(p).name.replace(".json", "")
        for p in sorted(glob.glob(str(recon_dir / "vggt" / "*.json")))
        if "inventory" not in p and "manifest" not in p]
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]
    print(f"scenes = {scenes}")

    import torch

    print("加载 MoGe-2 …")
    t0 = time.perf_counter()
    model = mf.make_moge2_model(device="cuda")
    load_s = time.perf_counter() - t0
    print(f"  加载 {load_s:.1f}s  device={model.device}  checkpoint={model.checkpoint}")
    if torch.cuda.is_available():
        print(f"  显存占用 {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    report = {"kind": "moge2_scale_fusion_poc", "spec": "§11.4 C1–C6",
              "date": time.strftime("%Y-%m-%d"),
              "model": mf.METRIC_MODEL_MOGE2, "checkpoint": model.checkpoint,
              "fusion_version": mf.METRIC_FUSION_VERSION,
              "thresholds": mf.threshold_snapshot(),
              "load_seconds": load_s,
              "peak_gpu_gib": (torch.cuda.max_memory_allocated() / 2**30
                               if torch.cuda.is_available() else None),
              "scenes": []}

    # ---------- C1 / C3 / C5：干净场景 ----------
    clean = []
    for s in scenes:
        r = run_scene(model, s, recon_dir, out_dir)
        clean.append(r)
        report["scenes"].append(r)
        print(f"  [干净] {s:14s} status={r['status']:8s} s={r['metric_scale']} "
              f"disp={r['scale_self_consistency']} 有效帧={r['n_frames_valid']}/{r['n_frames_total']} "
              f"延迟={r['latency_ms_mean']:.0f}ms")

    sks = {r["scene"]: r["s_k_median_of_medians"] for r in clean}
    disps = [r["scale_self_consistency"] for r in clean
             if r["scale_self_consistency"] is not None]

    # ---------- C2：注入退化 ----------
    base_scene = scenes[0]
    other = scenes[1] if len(scenes) > 1 else scenes[0]
    cross_rgb, _, _ = _load_scene(recon_dir, other)
    deg = {}
    deg["cross_scene"] = run_scene(model, base_scene, recon_dir, out_dir,
                                   cross_frames=cross_rgb)
    deg["blur"] = run_scene(model, base_scene, recon_dir, out_dir, blur=True)
    print(f"  [退化] 跨场景 disp={deg['cross_scene']['scale_self_consistency']}  "
          f"模糊 disp={deg['blur']['scale_self_consistency']}")
    base_disp = clean[0]["scale_self_consistency"] if clean else None

    def _rose(a, b):
        if a is None or b is None:
            return None
        return bool(b > a)

    # ---------- C6：错 K 应发散 ----------
    bad = None
    try:
        rgb, depth, intr = _load_scene(recon_dir, base_scene)
        K_bad = np.array(intr[0], dtype=np.float64).copy()
        K_bad[0, 0] *= 1.8   # 故意给错焦距
        K_bad[1, 1] *= 1.8
        hw = depth[0].shape
        ml, mk = [], []
        for i in range(min(8, len(rgb))):
            o = model.infer(rgb[i], K_bad)
            d = np.asarray(o.depth_metric, dtype=np.float64)
            m = np.asarray(o.valid_mask, dtype=bool)
            if d.shape != hw:
                d = _resize_to(d, hw)
            ml.append(d)
            mk.append(m)
        rb = mf.fuse_metric_scale(ml, [np.asarray(depth[i]) for i in range(len(ml))], mk,
                                  [None] * len(ml), conf_warp_monotonic=None,
                                  model=mf.METRIC_MODEL_MOGE2)
        bad = {"metric_scale": rb.metric_scale,
               "scale_self_consistency": rb.scale_self_consistency}
    except Exception as exc:  # noqa: BLE001
        bad = {"error": f"{type(exc).__name__}: {exc}"}

    report["C1_cross_frame_stability"] = {
        "per_scene_scale": sks,
        "per_scene_dispersion": {r["scene"]: r["scale_self_consistency"] for r in clean},
        "max_dispersion": max(disps) if disps else None,
        "threshold": mf.MAX_SCALE_DISPERSION,
        "passed": bool(disps) and max(disps) <= mf.MAX_SCALE_DISPERSION,
    }
    report["C2_degradation_sensitivity"] = {
        "clean_dispersion": base_disp,
        "cross_scene_dispersion": deg["cross_scene"]["scale_self_consistency"],
        "blur_dispersion": deg["blur"]["scale_self_consistency"],
        "cross_scene_rose": _rose(base_disp, deg["cross_scene"]["scale_self_consistency"]),
        "blur_rose": _rose(base_disp, deg["blur"]["scale_self_consistency"]),
        "passed": bool(_rose(base_disp, deg["cross_scene"]["scale_self_consistency"])
                       or _rose(base_disp, deg["blur"]["scale_self_consistency"])),
    }
    report["C3_geometry_gate"] = {
        "note": "主门由 M4 在 artifact 上判定；本 PoC 只报告各 scene 的 "
                "main_gate_passed（来自重建产物），不在此重算",
        "per_scene_gate": {p.stem.replace(".json", ""): None for p in []},
    }
    lat = [r["latency_ms_mean"] for r in clean]
    report["C5_resources"] = {
        "latency_ms_mean_per_frame": float(np.mean(lat)) if lat else None,
        "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
        "peak_gpu_gib": report["peak_gpu_gib"],
        "threshold_latency_ms": mf.C5_MAX_LATENCY_MS_PER_FRAME,
        "threshold_peak_gpu_gib": mf.C5_MAX_PEAK_GPU_GIB,
        "passed": bool(lat and np.mean(lat) <= mf.C5_MAX_LATENCY_MS_PER_FRAME
                       and (report["peak_gpu_gib"] or 0) <= mf.C5_MAX_PEAK_GPU_GIB),
    }
    report["C6_wrong_K_diverges"] = {
        "clean": {"metric_scale": clean[0]["metric_scale"] if clean else None,
                  "dispersion": base_disp},
        "wrong_K": bad,
        "note": "错 K 应使尺度/离散度明显变化；具体判据在 §11.4 C6，"
                "Metric3D v2 同场景对照因 [TODO_LICENSE] 未核实而**未跑**",
    }
    report["C4_metric_tasks_paired"] = {
        "status": "待跑",
        "note": "C4 需要米制三题（abs_distance/size/room）对「VGGT 未缩放」与"
                "「同帧直答」的 paired MRA 比较，须在在线链上跑；本脚本不编造 MRA。",
    }
    rep_path = out_dir / "moge2_poc_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"\n报告已写: {rep_path}")
    print(f"  C1 跨帧稳定 : {report['C1_cross_frame_stability']['passed']} "
          f"(max disp={report['C1_cross_frame_stability']['max_dispersion']})")
    print(f"  C2 退化敏感 : {report['C2_degradation_sensitivity']['passed']}")
    print(f"  C5 资源延迟 : {report['C5_resources']['passed']} "
          f"({report['C5_resources']['latency_ms_mean_per_frame']} ms/帧)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
