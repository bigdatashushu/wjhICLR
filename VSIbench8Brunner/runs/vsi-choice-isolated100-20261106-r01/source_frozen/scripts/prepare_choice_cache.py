#!/usr/bin/env python3
"""Seal read-only cache links and their byte identities for the paired rerun."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from choice_cached100 import sha
import pilot40 as pilot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    spec = json.loads((root / "experiment_spec.json").read_text())
    source = Path(spec["paired_baseline_server_root"]) / "harness"
    target = root / "harness"
    if (root / "cached_preparation_manifest.json").exists():
        raise ValueError("cache already sealed; duplicate preparation disabled")
    if json.loads((source / "live_status.json").read_text())["status"] != "complete":
        raise ValueError("baseline preparation is not complete")
    files, started = [], time.time()
    for name in ("bindings", "scene_audits", "preparation", "sam2_receipts"):
        for path in sorted((source / name).rglob("*")):
            if path.is_symlink():
                raise ValueError("sealed cache must contain regular files")
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError("partial cache retained; duplicate linking disabled")
            os.link(path, destination)
            stat = path.stat()
            digest = sha(path)
            if path.stat().st_mtime_ns != stat.st_mtime_ns:
                raise ValueError("source preparation changed while sealing")
            files.append({
                "file": str(relative), "bytes": stat.st_size, "sha256": digest,
                "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino, "device": stat.st_dev,
            })
            if len(files) % 1000 == 0:
                print(json.dumps({"status": "cache_sealing", "files": len(files),
                                  "bytes": sum(r["bytes"] for r in files),
                                  "elapsed_s": time.time() - started}), flush=True)
    if (len(list((target / "bindings").glob("*.json"))) != 100
            or len(list((target / "scene_audits").glob("*.json"))) != 69):
        raise ValueError("cached preparation denominator changed")
    pilot.write(root / "cached_preparation_manifest.json", {
        "schema": "vsi-choice-frozen-preparation/1", "source_root": str(source),
        "transport": "hard_links_with_preparation_and_binding_writes_forbidden",
        "source_runtime_identity_sha256": sha(source / "runtime_identity.json"),
        "source_native_audit_sha256": sha(source / "native_trajectory_verified.json"),
        "source_preflight_sha256": sha(source / "preflight_verified.json"),
        "files": files, "total_bytes": sum(r["bytes"] for r in files),
        "reference_answer_visible": False, "new_model_requests": 0, "new_gpu_requests": 0,
    })
    print(json.dumps({"status": "cache_sealed", "files": len(files),
                      "bytes": sum(r["bytes"] for r in files),
                      "elapsed_s": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
