#!/usr/bin/env python3
"""Isolated full-scene precompute pilot with model-free preflight and consumers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    args = parser.parse_args()
    receipts = args.receipt_root
    receipts.mkdir(parents=True, exist_ok=True)
    if any((receipts / name).exists() for name in ["preflight.json", "result.json", "failure.json"]):
        raise ValueError("pilot receipts already exist")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("pilot output must be empty")
    method = json.loads(args.method_config.read_text())
    source = Path(__file__).resolve().parents[1]
    for relative, expected in method["code_sha256"].items():
        if hash_file(source / relative) != expected:
            raise ValueError("pilot frozen code identity changed")
    detector = method["detector"]
    if os.environ.get("SKILL3D_DETECTOR_ENDPOINT", "") != detector["endpoint"]:
        raise ValueError("pilot detector endpoint differs from frozen method")
    from skill3d.cluster.scene_compute import (
        compute_scene, validate_published_scene, validate_scene_input,
    )
    from skill3d.cluster.perception_client import RemoteBailianClient
    from skill3d.segmentation.open_vocab_detector import capture_detector_attempts
    summary = validate_scene_input(args.input_root)
    with urllib.request.urlopen(detector["endpoint"] + "/identity", timeout=30) as response:
        identity = json.load(response)
    model_pin = hashlib.sha256(json.dumps(
        identity["model"], sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    if (identity["enabled"] is not True or model_pin != detector["model_sha256"]
            or identity["model_sha256"] != model_pin):
        raise ValueError("pilot detector model pin mismatch")
    checkpoints = [
        (os.environ["SKILL3D_MOGE2_CHECKPOINT"],
         method["gpu_models"]["metric_model"]["checkpoint_sha256"]),
        (os.environ["SKILL3D_SAM2_CHECKPOINT"],
         method["gpu_models"]["sam2"]["checkpoint"]["sha256"]),
    ]
    for path, expected in checkpoints:
        if hash_file(path) != expected:
            raise ValueError("pilot checkpoint changed")
    preflight = {
        "schema": "harness3d-isolated-scene-pilot-preflight/1", "at": time.time(),
        "status": "validated", "input": summary, "code_files": len(method["code_sha256"]),
        "method_config_sha256": hash_file(args.method_config),
        "detector_model_sha256": model_pin,
        "checkpoint_hashes_verified": len(checkpoints),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "label_access": False, "solver_ran": False, "model_requests": 0,
    }
    (receipts / "preflight.json").write_text(json.dumps(preflight, sort_keys=True) + "\n")
    started = time.monotonic()
    try:
        with capture_detector_attempts() as attempts:
            result = compute_scene(
                args.input_root, args.output_root, method,
                client_factory=lambda *a, **kw: RemoteBailianClient(**kw))
        elapsed = time.monotonic() - started
        consumer = validate_published_scene(
            args.output_root, summary["qa_ids"], summary["scene_id"], method)
        success = [row for row in attempts if row["status"] in {"ok", "empty"}]
        errors = [row for row in attempts if row["status"] == "error"]
        request_hashes = {row["request_sha256"] for row in success}
        response = {
            "schema": "harness3d-isolated-scene-pilot-result/1", "at": time.time(),
            "status": "validated", "preflight_sha256": hash_file(receipts / "preflight.json"),
            "scene_id": summary["scene_id"], "qa_ids": summary["qa_ids"],
            "frozen_frames": summary["n_frames"], "elapsed_s": elapsed,
            "normal_consumers": {name: variant["n_questions"]
                                 for name, variant in consumer["variants"].items()},
            "detector": {"logical_successful_requests": len(success),
                         "unique_request_hashes": len(request_hashes),
                         "failed_attempts": len(errors),
                         "jpeg_cache_hits": sum(row.get("jpeg_cache_hit", False) for row in attempts),
                         "encode_ms_p50": statistics.median(
                             [row["encode_ms"] for row in attempts]) if attempts else None},
            "scene_result": result, "label_access": False, "solver_ran": False,
            "scoring_ran": False,
        }
        (receipts / "result.json").write_text(json.dumps(response, sort_keys=True) + "\n")
        print(json.dumps(response, sort_keys=True), flush=True)
    except Exception as error:
        (receipts / "failure.json").write_text(json.dumps({
            "schema": "harness3d-isolated-scene-pilot-failure/1", "at": time.time(),
            "error_type": type(error).__name__, "elapsed_s": time.monotonic() - started,
            "outputs_retained": True, "rerun_performed": False}) + "\n")
        raise


if __name__ == "__main__":
    main()
