"""Content-bound metric evidence; shared preparation happens before paired solves."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time

import numpy as np

from . import metric_fusion as mf
from .image_grid import GRID_VERSION, prepare_metric_frames

GEOMETRY_REFS = ("c2w_list", "intrinsics", "depth_maps", "point_map", "point_conf",
                 "depth_conf", "track_list")
METRIC_REFS = ("per_frame_scale_ref", "metric_depth_maps", "metric_valid_masks")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_identity(model_class, checkpoint: Path) -> dict:
    """Pin installed source bytes even when the package lacks VCS metadata."""
    from importlib.metadata import version

    package = Path(inspect.getfile(model_class)).resolve().parents[1]
    return {
        "name": "moge2", "api": "moge.model.v2.MoGeModel.infer/meters",
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_sha256": {str(p.relative_to(package)): file_sha256(p)
                          for p in sorted(package.rglob("*.py"))},
        "runtime_versions": {name: version(name) for name in ("torch", "utils3d", "moge")},
    }


def fusion_contract(model) -> dict:
    from importlib.metadata import version

    root = Path(__file__).parent
    return {
        "version": mf.METRIC_FUSION_VERSION, "grid": GRID_VERSION,
        "model": getattr(model, "identity", {"name": getattr(model, "name", "none")}),
        "thresholds": mf.threshold_snapshot(), "use_fp16": mf.MOGE2_USE_FP16,
        "black_luma_max": mf.MOGE2_BLACK_LUMA_MAX,
        "runtime_versions": {name: version(name) for name in ("numpy", "Pillow")},
        "depth_units": "meters", "conf_warp_monotonic": None,
        "source_sha256": {name: file_sha256(root / name) for name in
                          ("metric_fusion.py", "image_grid.py", "metric_evidence.py",
                           "vggt_runner.py")},
    }


def source_identity(frames, frame_set) -> str:
    from skill3d.adapters.frame_set import with_pixel_identity

    if frame_set is None:
        raise ValueError("Frozen FrameSet is required for cache identity")
    return with_pixel_identity(frame_set, frames).cache_identity()


def _array_hash(array) -> str:
    a = np.ascontiguousarray(array)
    return digest({"shape": list(a.shape), "dtype": str(a.dtype),
                   "bytes": hashlib.sha256(a.tobytes()).hexdigest()})


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def attach_metric_evidence(art, frames, frame_set, model, out_dir: Path):
    """Infer on VGGT's exact RGB grid; persist failures and all numerical evidence.

    The artifact geometry is never rescaled on disk. Tools apply the single scale
    once to lengths and squared to areas, using the validated metric gate.
    """
    contract = fusion_contract(model)
    input_id = source_identity(frames, frame_set)
    if (art.reconstruction_input_identity != input_id
            or art.frame_ids != list(frame_set.readable_frame_ids)
            or art.frame_set_hash != frame_set.frame_set_hash):
        raise ValueError("Frozen VGGT source identity must be verified before metric augmentation")
    sources = {key: file_sha256(getattr(art, key)) for key in GEOMETRY_REFS
               if getattr(art, key)}
    request = {"contract": contract, "input_identity": input_id, "sources": sources,
               "frame_ids": list(art.frame_ids), "frame_set_hash": art.frame_set_hash,
               "scene_name": art.scene_name, "source_frame_indices": art.source_frame_indices,
               "timestamps": art.timestamps}
    request_id = digest(request)
    prefix = f"{art.scene_name}__{request_id[:20]}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb, content_masks, layouts = prepare_metric_frames(frames)
    depth = np.load(art.depth_maps, allow_pickle=False)
    intrinsics = np.load(art.intrinsics, allow_pickle=False)
    if depth.shape != (len(rgb), *rgb[0].shape[:2]) or intrinsics.shape != (len(rgb), 3, 3):
        raise ValueError("Cached VGGT depth/K do not match the frozen crop grid")
    metric = np.full(depth.shape, np.nan, dtype=np.float32)
    valid = np.zeros(depth.shape, dtype=bool)
    diagnostics = []
    device = str(getattr(model, "device", "cpu"))
    torch = getattr(model, "_torch", None)
    cuda = torch is not None and device.startswith("cuda")
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for idx, image in enumerate(rgb):
        row = {"frame_idx": idx, "frame_id": art.frame_ids[idx],
               "rgb_sha256": _array_hash(image), "grid": layouts[idx],
               "fov_x_deg": None, "error": None}
        before = time.perf_counter()
        try:
            result = model.infer(image, intrinsics[idx])
            d = np.asarray(result.depth_metric, dtype=np.float32)
            mask = np.asarray(result.valid_mask)
            if d.shape != depth[idx].shape or mask.shape != d.shape:
                raise ValueError("Metric depth/mask must match VGGT grid; resize is forbidden")
            if mask.dtype != np.bool_:
                raise ValueError("Metric valid_mask must be boolean")
            metric[idx] = d
            valid[idx] = mask & content_masks[idx] & np.isfinite(d) & (d > 0)
            row["fov_x_deg"] = result.fov_x_deg
        except Exception as exc:  # A failed frame stays in the fixed denominator.
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if cuda:
            torch.cuda.synchronize(device)
        row["elapsed_seconds"] = time.perf_counter() - before
        diagnostics.append(row)
    elapsed = time.perf_counter() - started
    fusion = mf.fuse_metric_scale(list(metric), list(depth), list(valid),
                                  model=str(model.name), conf_warp_monotonic=None)
    receipt = mf.per_frame_receipt(fusion)
    receipt.update(request=request, request_sha256=request_id, diagnostics=diagnostics,
                   resource={"elapsed_seconds": elapsed, "device": device,
                             "latency_ms_per_frame": elapsed * 1000 / len(rgb),
                             "peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                                      if cuda else None)})
    refs = {}
    for key, value in (("metric_depth_maps", metric), ("metric_valid_masks", valid)):
        path = out_dir / f"{prefix}_{key}.npy"
        np.save(path, value, allow_pickle=False)
        refs[key] = str(path)
    receipt["evidence_sha256"] = {key: file_sha256(path) for key, path in refs.items()}
    path = out_dir / f"{prefix}_scale_receipt.json"
    _write(path, receipt)  # A write failure is reported, never silently called success.
    return art.model_copy(update={
        **refs, "per_frame_scale_ref": str(path), "metric_receipt_sha256": file_sha256(path),
        "metric_input_identity": input_id, "metric_request_sha256": request_id,
        "metric_scale": fusion.metric_scale, "scale_self_consistency": fusion.scale_self_consistency,
        "metric_model": fusion.model, "metric_fusion_version": fusion.version,
        "scale_fusion_status": fusion.status,
        "artifact_version": digest({"base": art.artifact_version,
                                    "request": request_id, "receipt": file_sha256(path)})[:16],
    })


def validated_receipt(art) -> dict:
    """Reject transplanted receipts, changed arrays, and inconsistent scalar fields."""
    path = art.per_frame_scale_ref
    if not path or file_sha256(path) != art.metric_receipt_sha256:
        raise ValueError("Metric receipt missing or content hash mismatch")
    rec = mf.read_per_frame_receipt(path)
    request = rec["request"]
    if (rec["request_sha256"] != digest(request)
            or rec["request_sha256"] != art.metric_request_sha256
            or request["input_identity"] != art.metric_input_identity
            or request["input_identity"] != art.reconstruction_input_identity
            or request["frame_ids"] != list(art.frame_ids)
            or request["frame_set_hash"] != art.frame_set_hash
            or request["scene_name"] != art.scene_name
            or request["source_frame_indices"] != art.source_frame_indices
            or request["timestamps"] != art.timestamps
            or rec["metric_fusion_version"] != mf.METRIC_FUSION_VERSION):
        raise ValueError("Metric input/request identity mismatch")
    for key, expected in request["sources"].items():
        if key not in GEOMETRY_REFS or file_sha256(getattr(art, key)) != expected:
            raise ValueError(f"Metric source changed: {key}")
    for key in ("metric_depth_maps", "metric_valid_masks"):
        if file_sha256(getattr(art, key)) != rec["evidence_sha256"][key]:
            raise ValueError(f"Metric evidence changed: {key}")
    for receipt_key, artifact_key in (
        ("status", "scale_fusion_status"), ("metric_scale", "metric_scale"),
        ("scale_self_consistency", "scale_self_consistency"),
        ("metric_model", "metric_model"), ("metric_fusion_version", "metric_fusion_version"),
    ):
        if rec[receipt_key] != getattr(art, artifact_key):
            raise ValueError(f"Metric scalar mismatch: {receipt_key}")
    n, total = rec["n_frames_valid"], rec["n_frames_total"]
    if (total != len(art.frame_ids) or total <= 0 or not 0 <= n <= total
            or rec["valid_frame_ratio"] != n / total):
        raise ValueError("Metric frame coverage mismatch")
    return rec


def copy_geometry(art, destination: Path):
    """Create a private geometry copy before adding/removing metric evidence."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    data = art.model_dump()
    for key in GEOMETRY_REFS:
        ref = data.get(key)
        if ref:
            target = destination / (key + Path(ref).suffix)
            shutil.copyfile(ref, target)
            data[key] = str(target)
    data["confidence"] = {"per_point_confidence": data["point_conf"],
                          "coverage_count_per_frame": ""}
    data.update({key: None for key in METRIC_REFS})
    data.update(metric_scale=None, scale_self_consistency=None, metric_model="none",
                metric_fusion_version=mf.METRIC_FUSION_VERSION, scale_fusion_status="not_run",
                metric_input_identity="", metric_receipt_sha256="", metric_request_sha256="")
    return type(art).model_validate(data)
