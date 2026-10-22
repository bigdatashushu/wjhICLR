#!/usr/bin/env python3
"""Package verified scene inputs from the archival node without opening labels."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time

from skill3d.cluster import FrozenIdentity, JobSpec, hash_file
from cluster_coordinator import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    exports = json.loads(args.export_manifest.read_text())
    mapping = {row["source"]: row for row in exports["files"]}
    if len(mapping) != len(exports["files"]):
        raise ValueError("source mapping is ambiguous")
    attachment = json.loads((root / "imports/attachment/geometry_attachment.json").read_text())
    if attachment["status"] != "verified" or attachment["verified_scenes"] != 219:
        raise ValueError("legacy geometry attachment is not fully verified")
    method = json.loads(args.method_config.read_text())
    scene_paths = sorted((root / "imports/frozen/scenes").glob("*.json"))
    if len(scene_paths) != 230:
        raise ValueError("authorized scene scope changed")
    public = {"schema": "harness3d-cluster-scope/1", "scenes": {}}
    scenes = {}
    for path in scene_paths:
        scene = json.loads(path.read_text())
        if scene["status"] != "frozen" or len(scene["frames"]) != 32:
            raise ValueError("frozen input is incomplete")
        if len(scene["qa_ids"]) > 8 or not scene["qa_ids"]:
            raise ValueError("scene question cap exceeded")
        if any("ground_truth" in row["episode"] for row in scene["questions"]):
            raise ValueError("public scene contains a forbidden label field")
        supplement = attachment["scenes"].get(scene["scene_id"])
        if supplement:
            if supplement["qa_ids"] != scene["qa_ids"]:
                raise ValueError("geometry attachment question group changed")
            scene["artifact_source_path"] = supplement["artifact_source_path"]
            for row in scene["questions"]:
                row["logical_input_identity"] = supplement["logical_input_identities"][row["episode"]["qa_id"]]
        public["scenes"][scene["scene_id"]] = {
            "qa_ids": scene["qa_ids"], "split": scene["split"],
            "frozen_scene_sha256": hash_file(path)[0],
            "legacy_artifact_sha256": (
                next(iter(supplement["logical_input_identities"].values()))["artifact_json_sha256"]
                if supplement else None),
        }
        scenes[scene["scene_id"]] = scene
    if sum(len(s["qa_ids"]) for s in scenes.values()) != 1154:
        raise ValueError("authorized question scope changed")
    identity = FrozenIdentity.freeze(public, method)
    jobs = []
    progress = {"schema": "harness3d-cluster-package-progress/1", "status": "running",
                "identity_hash": identity.identity_hash, "scenes": {}, "total_scenes": 230,
                "total_questions": 1154, "labels_accessed": False}
    began = time.monotonic()
    verified = set()
    for scene_id, original in sorted(scenes.items()):
        scene = copy.deepcopy(original)
        spec = JobSpec(
            f"precompute-20261022-r01/{scene_id}/gpu_aggregate",
            f"{identity.identity_hash}/{scene_id}/gpu_aggregate",
            identity, "gpu_aggregate", max_attempts=3,
            priority=10 if len(scene["qa_ids"]) == 2 and scene.get("artifact_source_path") else 0)
        jobs.append(spec.to_dict())
        key = hashlib.sha256(spec.job_key.encode()).hexdigest()
        archive = root / "inputs/jobs" / (key + ".tar")
        archive.parent.mkdir(parents=True, exist_ok=True)
        staging = root / ".packing" / key
        sources = {}
        for index, row in enumerate(scene["frames"]):
            original_path = row["source_path"]
            # The freeze step replayed all frames from source videos. Its
            # supplemental transfer preserves these exact newly frozen PNGs.
            relative = f"frozen/new_frames/{scene_id}/{index:04d}.png"
            local = root / "imports" / relative
            if hash_file(local) != (row["png_sha256"], row["size"]):
                raise ValueError("transferred frozen PNG differs from manifest")
            sources[original_path] = (local, f"data/frames/{index:04d}.png")
        geometry = scene.get("artifact_source_path")
        if geometry:
            logical = scene["questions"][0]["logical_input_identity"]
            for path in [geometry, *(v["path"] for v in logical["files"].values())]:
                row = mapping[path]
                local = root / "imports" / row["relative"]
                if path not in verified:
                    if hash_file(local) != (row["sha256"], row["size"]):
                        raise ValueError("archived geometry dependency differs from frozen export")
                    verified.add(path)
                sources[path] = (local, "data/" + row["relative"])
        scene["files"] = {source: value[1] for source, value in sources.items()}
        scene["file_identities"] = {
            value[1]: {"sha256": hash_file(value[0])[0], "size": value[0].stat().st_size}
            for value in sources.values()
        }
        scene["cluster_identity_hash"] = identity.identity_hash
        if not archive.is_file():
            staging.mkdir(parents=True, exist_ok=True)
            try:
                atomic_json(staging / "scene.json", scene)
                temp = archive.with_suffix(".part")
                with tarfile.open(temp, "w:") as stream:
                    stream.add(staging / "scene.json", arcname="scene.json", recursive=False)
                    for local, target in sources.values():
                        stream.add(local, arcname=target, recursive=False)
                with temp.open("rb") as file:
                    import os
                    os.fsync(file.fileno())
                temp.rename(archive)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        else:
            with tarfile.open(archive, "r:") as stream:
                existing = json.load(stream.extractfile("scene.json"))
            if existing != scene:
                raise ValueError("existing job input package has another identity")
        progress["scenes"][scene_id] = {
            "job_key": spec.job_key, "archive_sha256": hash_file(archive)[0],
            "archive_bytes": archive.stat().st_size, "questions": len(scene["qa_ids"]),
            "geometry": "verified_reuse" if geometry else "fresh_required"}
        progress["elapsed_s"] = time.monotonic() - began
        atomic_json(root / "receipts/package_progress.json", progress)
    atomic_json(root / "jobs.json", {"identity": identity.to_dict(), "jobs": jobs})
    progress["status"] = "packaged"
    atomic_json(root / "receipts/package_progress.json", progress)
    print(json.dumps({k: v for k, v in progress.items() if k != "scenes"}), flush=True)


if __name__ == "__main__":
    main()
