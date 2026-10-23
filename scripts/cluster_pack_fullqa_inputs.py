#!/usr/bin/env python3
"""Pack a complete, explicitly frozen public scope without reading labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time

from skill3d.cluster import FrozenIdentity, JobSpec, hash_file
from cluster_coordinator import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--method-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pilot-scene")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--expected-questions", type=int, default=5130)
    parser.add_argument("--expected-scenes", type=int, default=288)
    args = parser.parse_args()
    scope = json.loads(args.scope.read_text())
    grouped = {}
    for row in scope["questions"]:
        grouped.setdefault(row["scene_name"], []).append(str(row["qa_id"]))
    if (len(grouped) != args.expected_scenes or
            sum(map(len, grouped.values())) != args.expected_questions):
        raise ValueError("public scope count changed")
    method = json.loads(args.method_config.read_text())
    scenes, public = {}, {"schema": "harness3d-cluster-scope/1", "scenes": {}}
    for scene_id, qa_ids in sorted(grouped.items()):
        path = args.frozen_root / "scenes" / f"{scene_id}.json"
        scene = json.loads(path.read_text())
        if (scene["status"] != "frozen" or scene["qa_ids"] != sorted(set(qa_ids))
                or scene["real_labels_available"] is not False
                or len(qa_ids) > method["max_questions_per_scene"]):
            raise ValueError("frozen full scene scope changed")
        scenes[scene_id] = scene
        public["scenes"][scene_id] = {
            "qa_ids": scene["qa_ids"], "split": scene["split"],
            "frozen_scene_sha256": hash_file(path)[0],
        }
    identity = FrozenIdentity.freeze(public, method)
    specs = {}
    for scene_id, scene in scenes.items():
        specs[scene_id] = JobSpec(
            f"{args.run_id}/{scene_id}/gpu_aggregate",
            f"{identity.identity_hash}/{scene_id}/gpu_aggregate",
            identity, "gpu_aggregate", max_attempts=3,
            priority=1000 if scene_id == args.pilot_scene else len(scene["qa_ids"]),
        )
    payload = {"identity": identity.to_dict(), "jobs": [spec.to_dict() for spec in specs.values()]}
    atomic_json(args.root / "jobs.json", payload)
    atomic_json(args.root / "coordinator_jobs.json", payload)
    if args.plan_only:
        print(json.dumps({"status": "planned", "identity_hash": identity.identity_hash,
                          "scenes": len(specs), "questions": args.expected_questions}), flush=True)
        return
    progress = {
        "schema": "harness3d-cluster-package-progress/1", "status": "running",
        "identity_hash": identity.identity_hash, "scenes": {},
        "total_scenes": len(scenes), "total_questions": args.expected_questions,
        "labels_accessed": False,
    }
    started = time.monotonic()
    ordered = sorted(scenes.items(), key=lambda row: (-specs[row[0]].priority, row[0]))
    for scene_id, scene in ordered:
        spec = specs[scene_id]
        key = hashlib.sha256(spec.job_key.encode()).hexdigest()
        archive = args.root / "inputs/jobs" / f"{key}.tar"
        archive.parent.mkdir(parents=True, exist_ok=True)
        # The freezer records the actual local sources separately from the
        # original logical paths carried by relocation receipts.
        local_sources = scene.pop("local_sources")
        if set(local_sources) != set(scene["files"].values()):
            raise ValueError("local sources do not cover frozen transfer mapping")
        temporary_scene = args.root / ".packing" / f"{key}.json"
        atomic_json(temporary_scene, scene)
        if not archive.exists():
            temporary_archive = archive.with_suffix(".part")
            with tarfile.open(temporary_archive, "w:") as stream:
                stream.add(temporary_scene, arcname="scene.json", recursive=False)
                for relative, source in sorted(local_sources.items()):
                    expected = scene["file_identities"][relative]
                    if hash_file(source) != (expected["sha256"], expected["size"]):
                        raise ValueError("frozen source changed before transfer")
                    stream.add(source, arcname=relative, recursive=False)
            temporary_archive.replace(archive)
        else:
            with tarfile.open(archive, "r:") as stream:
                if json.load(stream.extractfile("scene.json")) != scene:
                    raise ValueError("existing archive has another identity")
        temporary_scene.unlink()
        sha, size = hash_file(archive)
        progress["scenes"][scene_id] = {
            "job_key": spec.job_key, "archive_sha256": sha, "archive_bytes": size,
            "questions": len(scene["qa_ids"]),
            "geometry": "verified_reuse" if scene.get("artifact_source_path") else "fresh_required",
        }
        progress["elapsed_s"] = time.monotonic() - started
        atomic_json(args.root / "receipts/package_progress.json", progress)
    progress["status"] = "packaged"
    atomic_json(args.root / "receipts/package_progress.json", progress)
    print(json.dumps({k: v for k, v in progress.items() if k != "scenes"}), flush=True)


if __name__ == "__main__":
    main()
