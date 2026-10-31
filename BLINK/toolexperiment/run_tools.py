#!/usr/bin/env python3
"""Exercise frozen BLINK tools on published caches; no model inference or labels.

JSONL stdout contains a run manifest, one measured record per original problem,
and a completion receipt. Import this module only through main() or run().
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace

SCHEMA = "blink-real-tools-experiment/1"
PROTOCOL = {
    "schema": SCHEMA,
    "grid_fractions": [.1, .3, .5, .7, .9],
    "jitter_pixels": [[-2, 0], [2, 0], [0, -2], [0, 2]],
    "numeric_rtol": 1e-5,
    "numeric_atol": 1e-7,
    "sift_features": 1500,
    "sift_ratio": .7,
    "fundamental_ransac_px": 1.0,
    "min_inlier_matches": 12,
    "max_matches": 60,
    "max_cross_view_pairs": 80,
    "min_pair_pixel_separation": 30,
    "seed": 137,
    "model_requests": 0,
    "gpu_requests": 0,
    "benchmark_answer_access": False,
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, default=json_scalar,
                                     separators=(",", ":")).encode()).hexdigest()


def json_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(record):
    print(json.dumps(record, ensure_ascii=False, allow_nan=False, default=json_scalar,
                     separators=(",", ":")),
          flush=True)


def numeric_stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {"n": len(a), "median": float(np.median(a)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def reference_sample(prepared, view, xy):
    """Independent indexing + sorted order statistics, without calling tool helpers."""
    depth = np.asarray(prepared.depths[view], dtype=np.float64)
    h, w = depth.shape
    ix, iy = [int(round(float(v))) for v in xy]
    if not (0 <= xy[0] < w and 0 <= xy[1] < h):
        return None
    ix, iy = min(ix, w - 1), min(iy, h - 1)
    coords = [(y, x) for y in range(max(0, iy - 2), min(h, iy + 3))
              for x in range(max(0, ix - 2), min(w, ix + 3))]
    coords = [(y, x) for y, x in coords
              if prepared.geometry_masks[view][y, x] and
              np.isfinite(depth[y, x]) and depth[y, x] > 0]
    if len(coords) < 3:
        return None
    z = float(np.percentile([depth[y, x] for y, x in coords], 50))
    point = None
    if view in prepared.points:
        points = np.asarray([prepared.points[view][y, x] for y, x in coords],
                            dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) >= 3:
            point = np.percentile(points, 50, axis=0)
    return z, point


class Caller:
    def __init__(self, registry, scene, sample_id):
        self.registry, self.scene, self.sample_id = registry, scene, sample_id
        self.counts, self.failed = Counter(), Counter()
        self.hash_chain = hashlib.sha256()
        self.failure_examples = []

    def __call__(self, name, **args):
        self.counts[name] += 1
        try:
            response = self.registry.call_tool(
                name, args, self.scene, episode_id=self.sample_id, model_call=True)
            result = {"status": response.status, "payload": response.payload,
                      "error_code": response.error_code,
                      "authorization_allowed": response.authorization.get("allowed"),
                      "result_id": response.result_id}
        except Exception as exc:
            result = {"status": "rejected", "payload": None,
                      "error_type": type(exc).__name__}
        self.hash_chain.update(digest({"tool": name, "args": args, "result": result}).encode())
        if result["status"] != "ok":
            self.failed[name] += 1
            if len(self.failure_examples) < 15:
                self.failure_examples.append({"tool": name, "args": args, **result})
        return result


def distance(call, view, a, b):
    r = call("pair_point_distance", view_index=view, a_xy=list(a), b_xy=list(b))
    return r["payload"]["distance"] if r["status"] == "ok" else None


def cross_view(prepared, call, matching):
    """RGB-derived correspondences provide a consistency diagnostic, not GT."""
    base = {k: v for k, v in matching.items() if k not in ("xy0", "xy1", "image_sha256")}
    if matching["status"] != "matched":
        return base
    p0, p1 = np.asarray(matching["xy0"]), np.asarray(matching["xy1"])
    pairs = list(itertools.combinations(range(len(p0)), 2))
    np.random.default_rng(PROTOCOL["seed"]).shuffle(pairs)
    measurements = []
    for a, b in pairs:
        if any(np.linalg.norm(p[a] - p[b]) < PROTOCOL["min_pair_pixel_separation"]
               for p in (p0, p1)):
            continue
        v0 = distance(call, 0, p0[a].astype(float), p0[b].astype(float))
        v1 = distance(call, 1, p1[a].astype(float), p1[b].astype(float))
        if v0 is None or v1 is None or min(v0, v1) <= 1e-8:
            continue
        measurements.append({"a": a, "b": b, "distance0": v0, "distance1": v1})
        if len(measurements) >= PROTOCOL["max_cross_view_pairs"]:
            break
    if len(measurements) < 10:
        return {**base, "status": "insufficient_measurable_pairs",
                "measured_pairs": measurements}
    ratios = np.array([r["distance1"] / r["distance0"] for r in measurements])
    scale = float(np.median(ratios))
    relative_error = np.abs(ratios / scale - 1)
    for row, err in zip(measurements, relative_error):
        row["scale_normalized_error"] = float(err)
    return {
        **base, "status": "measured", "matches_used": len(p0),
        "correspondences0": p0.tolist(), "correspondences1": p1.tolist(),
        "scale_1_over_0": scale, "relative_error": numeric_stats(relative_error),
        "fraction_error_le_20pct": float(np.mean(relative_error <= .2)),
        "measured_pairs": measurements,
        "interpretation": "RGB matches may include mistakes; one global relative scale "
                          "is fitted. This is not a distance ground-truth metric.",
    }


def measure(problem_row, root, annotations, matching):
    from skill3d.small_image.storage import load_prepared
    from skill3d.small_image.tools import build_registry
    from skill3d.sandbox.kernel import RestrictedNamespaceKernel
    started = time.monotonic()
    cache = root / "prepared" / problem_row["input_sha256"]
    metadata_sha = file_sha(cache / "metadata.json")
    prepared = load_prepared(cache, problem=problem_row)
    if [file_sha(p) for p in problem_row["image_paths"]] != matching["image_sha256"]:
        raise ValueError("RGB matching inputs differ from frozen tool inputs")
    problem = SimpleNamespace(**problem_row)
    registry, scene = build_registry(prepared, problem)
    call = Caller(registry, scene, problem.sample_id)
    checks = Counter()
    evidence = {
        "sample_id": problem.sample_id, "source_split": problem.source_split,
        "input_sha256": problem.input_sha256, "metadata_sha256": metadata_sha,
        "artifact_sha256": prepared.artifact_sha256,
        "capabilities": dict(prepared.capabilities),
        "categories": list(prepared.categories),
        "n_candidates": len(prepared.candidates),
        "candidate_queries": [], "grid": [], "pairs": [], "invalid_inputs": [],
        "visual_ordinals": [], "view_geometry": [],
    }
    for view in (-1, 0, 1):
        r = call("view_candidates", view_index=view)
        expected = [c for c in prepared.candidates
                    if view == -1 or c["view_index"] == view]
        ok = r["status"] == "ok" and r["payload"]["count"] == len(expected)
        checks["candidate_query_checks"] += 1
        checks["candidate_query_passed"] += ok
        evidence["candidate_queries"].append({"view": view, "consistent": ok, **r})
    summary = call("view_depth_summary")
    evidence["depth_summary"] = summary
    first_valid = None
    for view, frame in enumerate(prepared.frames):
        h, w = frame.shape[:2]
        with Image.open(problem_row["image_paths"][view]) as image:
            checks["rgb_pixels_checked"] += 1
            checks["rgb_pixels_equal"] += np.array_equal(np.asarray(image.convert("RGB")), frame)
        z = np.asarray(prepared.depths[view])
        valid = np.asarray(prepared.geometry_masks[view]) & np.isfinite(z) & (z > 0)
        values = z[valid]
        expected_quantiles = np.percentile(values.astype(float), [10, 50, 90]).tolist()
        returned = summary["payload"]["views"][view] if summary["status"] == "ok" else {}
        summary_ok = returned.get("n_valid") == int(valid.sum()) and np.allclose(
            returned.get("quantiles", []), expected_quantiles, rtol=1e-5, atol=1e-7)
        checks["summary_views_checked"] += 1
        checks["summary_views_passed"] += bool(summary_ok)
        point_map = np.asarray(prepared.points[view])
        pvalid = valid & np.isfinite(point_map).all(axis=-1)
        z_error = np.abs(point_map[..., 2][pvalid] - z[pvalid]) / z[pvalid]
        evidence["view_geometry"].append({
            "view": view, "height": h, "width": w,
            "valid_fraction": float(valid.mean()), "summary_consistent": bool(summary_ok),
            "point_z_depth_relative_error": numeric_stats(z_error),
        })
        grid = [[round((w - 1) * fx), round((h - 1) * fy)]
                for fy in PROTOCOL["grid_fractions"] for fx in PROTOCOL["grid_fractions"]]
        valid_grid = []
        for xy in grid:
            r = call("point_depth", view_index=view, x=xy[0], y=xy[1])
            ref = reference_sample(prepared, view, xy)
            ok = (r["status"] != "ok") if ref is None else (
                r["status"] == "ok" and np.isclose(
                    r["payload"]["depth"], ref[0], rtol=1e-5, atol=1e-7))
            checks["point_numeric_checks"] += 1
            checks["point_numeric_passed"] += bool(ok)
            test = {"view": view, "xy": xy, "status": r["status"], "consistent": bool(ok),
                    "depth": r["payload"]["depth"] if r["status"] == "ok" else None,
                    "reference_depth": ref[0] if ref else None, "jitter": []}
            if r["status"] == "ok":
                valid_grid.append(xy)
                if first_valid is None and ref[1] is not None:
                    first_valid = view, xy
                for dx, dy in PROTOCOL["jitter_pixels"]:
                    jt = call("point_depth", view_index=view, x=xy[0]+dx, y=xy[1]+dy)
                    test["jitter"].append({
                        "offset": [dx, dy], "status": jt["status"],
                        "depth": jt["payload"]["depth"] if jt["status"] == "ok" else None,
                        "relative_change": abs(jt["payload"]["depth"]/test["depth"]-1)
                        if jt["status"] == "ok" else None,
                    })
            evidence["grid"].append(test)
        for a, b in zip(valid_grid[:12], reversed(valid_grid[-12:])):
            da = distance(call, view, a, b)
            ra, rb = reference_sample(prepared, view, a), reference_sample(prepared, view, b)
            expected = float(np.linalg.norm(ra[1] - rb[1])) if (
                ra and rb and ra[1] is not None and rb[1] is not None) else None
            ok = (da is None) if expected is None else (
                da is not None and bool(np.isclose(da, expected, rtol=1e-5, atol=1e-7)))
            reverse, same = distance(call, view, b, a), distance(call, view, a, a)
            checks["distance_numeric_checks"] += 1
            checks["distance_numeric_passed"] += ok
            checks["distance_symmetry_checks"] += 1
            checks["distance_symmetry_passed"] += da == reverse
            checks["same_point_checks"] += 1
            checks["same_point_passed"] += same is not None and abs(same) <= 1e-7
            row = {"view": view, "a": a, "b": b, "distance": da,
                   "reference_distance": expected, "reverse": reverse, "same_point": same,
                   "consistent": ok, "jitter": []}
            if da is not None and da > 1e-8:
                for dx, dy in PROTOCOL["jitter_pixels"]:
                    jd = distance(call, view, [a[0]+dx, a[1]+dy], b)
                    row["jitter"].append({"offset": [dx, dy], "distance": jd,
                                          "relative_change": abs(jd/da-1) if jd is not None else None})
            evidence["pairs"].append(row)
        invalid = [
            ("point_depth", {"view_index": view, "x": -1, "y": h//2}),
            ("point_depth", {"view_index": view, "x": w, "y": h//2}),
            ("point_depth", {"view_index": view, "x": 0, "y": h}),
            ("pair_point_distance", {"view_index": view, "a_xy": [-1, 0], "b_xy": [10, 10]}),
        ]
        # A naturally unsupported 5x5 area exercises missing geometry without fabricated data.
        counts = cv2.boxFilter(valid.astype("float32"), -1, (5, 5),
                              normalize=False, borderType=cv2.BORDER_CONSTANT)
        yy, xx = np.where(counts < 3)
        if len(xx):
            k = len(xx)//2
            invalid.append(("point_depth", {"view_index": view, "x": int(xx[k]), "y": int(yy[k])}))
        for name, args in invalid:
            response = call(name, **args)
            evidence["invalid_inputs"].append({"tool": name, "args": args, **response})
            checks["invalid_checks"] += 1
            checks["invalid_rejected"] += response["status"] != "ok"
    for name, args in [
        ("point_depth", {"view_index": 2, "x": 20, "y": 20}),
        ("view_candidates", {"view_index": 2}),
        ("pair_point_distance", {"view_index": 0, "a_xy": [20, 20], "b_xy": [40, 40],
                                 "other_view_index": 1}),
    ]:
        response = call(name, **args)
        checks["invalid_checks"] += 1
        checks["invalid_rejected"] += response["status"] != "ok"
        evidence["invalid_inputs"].append({"tool": name, "args": args, **response})
    if first_valid is not None:
        v, xy = first_valid
        # Invoke all four through the unchanged model-facing kernel; submit no benchmark answer.
        kernel = RestrictedNamespaceKernel(registry, scene, frames=list(prepared.frames),
                                           episode_id=problem.sample_id, cell_timeout_s=30)
        code = (
            "a = tools.view_candidates()\n"
            "b = tools.view_depth_summary()\n"
            f"c = tools.point_depth(view_index={v}, x={xy[0]}, y={xy[1]})\n"
            f"d = tools.pair_point_distance(view_index={v}, a_xy={xy!r}, b_xy={xy!r})\n"
            "YieldObservations([a['result_id'], b['result_id'], c['result_id'], d['result_id']])"
        )
        cell = kernel.run_cell(code)
        evidence["sandbox"] = {
            "program": code, "error_code": cell.error_code,
            "terminated": cell.terminated, "answer_untrusted": cell.answer_untrusted,
            "tool_results": [{"tool": r.tool, "status": r.status, "payload": r.payload}
                             for r in kernel.tool_results],
            "authorization_allowed": [r["allowed"] for r in kernel.authorization_receipts],
        }
        checks["sandbox_checked"] += 1
        checks["sandbox_passed"] += (
            not cell.error_code and not cell.answer_untrusted and cell.terminated == "yield"
            and len(kernel.tool_results) == 4 and all(r.status == "ok" for r in kernel.tool_results))
        del kernel
    for annotation in annotations:
        if annotation["sample_id"] != problem.sample_id:
            continue
        v = annotation["view_index"]
        samples = []
        for offset in [[0, 0], *PROTOCOL["jitter_pixels"]]:
            responses = [call("point_depth", view_index=v, x=annotation[k][0]+offset[0],
                              y=annotation[k][1]+offset[1]) for k in ("near", "far")]
            depths = [r["payload"]["depth"] if r["status"] == "ok" else None for r in responses]
            samples.append({"offset": offset, "near_depth": depths[0], "far_depth": depths[1],
                            "near_is_closer": depths[0] < depths[1]
                            if all(d is not None for d in depths) else None})
        evidence["visual_ordinals"].append({**annotation, "measurements": samples})
    evidence["cross_view"] = cross_view(prepared, call, matching)
    if any(a["sample_id"] == problem.sample_id for a in annotations):
        evidence["preview_depth"] = [
            np.where(prepared.geometry_masks[v][::4, ::4],
                     prepared.depths[v][::4, ::4], np.nan).tolist()
            for v in (0, 1)]
        # JSON null is explicit invalid geometry.
        evidence["preview_depth"] = [
            [[float(x) if np.isfinite(x) else None for x in row] for row in frame]
            for frame in evidence["preview_depth"]]
    evidence["checks"] = dict(checks)
    evidence["calls_by_tool"] = dict(call.counts)
    evidence["failed_calls_by_tool_including_expected_rejections"] = dict(call.failed)
    evidence["tool_call_chain_sha256"] = call.hash_chain.hexdigest()
    evidence["failure_examples"] = call.failure_examples
    evidence["elapsed_seconds"] = time.monotonic() - started
    evidence["record_sha256"] = digest(evidence)
    return evidence


def run(root, annotations, limit=None, correspondences=None):
    global np, cv2, Image
    root = Path(root).resolve()
    os.environ.clear()
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "code/src"))
    compat = Path("/home/wangy/wangjh/skill3d-lab/pilot-20261021-01/runtime/cpu_compat")
    sys.path.insert(0, str(compat))
    import numpy as np
    import cv2
    from PIL import Image
    cv2.setNumThreads(1)
    # Prevent accidental live service/model calls. The SSH transport runs in its parent.
    def deny_network(*args, **kwargs):
        raise RuntimeError("network_disabled_in_tool_experiment")
    socket.socket.connect = deny_network
    socket.create_connection = deny_network
    paths = sorted((root/"code/src/skill3d").rglob("*.py"))
    source_hashes = {str(p.relative_to(root/"code/src")): file_sha(p) for p in paths}
    rows = [r for r in json.loads((root/"manifest.json").read_text())["problems"]
            if r["benchmark"] == "blink"]
    if limit is not None:
        rows = rows[:limit]
    manifest = {
        "record_type": "manifest", "protocol": PROTOCOL, "root": str(root),
        "questions": len(rows), "sample_ids": [r["sample_id"] for r in rows],
        "input_manifest_sha256": file_sha(root/"manifest.json"),
        "source_files": source_hashes, "annotation_sha256": digest(annotations),
        "rgb_correspondences_sha256": digest(correspondences),
        "cpu_compat": str(compat),
        "python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__,
        "started_at_unix": time.time(), "pid": os.getpid(),
    }
    emit(manifest)
    started = time.monotonic()
    for i, row in enumerate(rows):
        result = measure(row, root, annotations["pairs"], correspondences["samples"][row["sample_id"]])
        emit({"record_type": "sample", "index": i, "result": result})
        gc.collect()
    assert all(file_sha(root/"code/src"/p) == h for p, h in source_hashes.items())
    emit({"record_type": "complete", "questions": len(rows),
          "source_unchanged": True, "elapsed_seconds": time.monotonic()-started,
          "model_requests": 0, "gpu_requests": 0, "finished_at_unix": time.time()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--rgb-correspondences", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.root, json.loads(args.annotations.read_text(encoding="utf-8")), args.limit,
        json.loads(args.rgb_correspondences.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
