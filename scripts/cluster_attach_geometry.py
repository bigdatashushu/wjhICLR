#!/usr/bin/env python3
"""Attach verified legacy geometry to already frozen, label-free scene inputs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.adapters.frame_set import with_pixel_identity
from skill3d.evaluation.skill_ablation_v11 import input_identity
from skill3d.infra.m5_identity import content_sha256
from skill3d.schemas import ReconstructionArtifact, VSIBenchEpisode


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def registry(manifest: dict) -> dict[str, Path]:
    result = {}
    for row in manifest["files"]:
        source = str(row["source"])
        if "/reconstructions/vggt/" not in source or not source.endswith(".json"):
            continue
        path = Path(source)
        raw = path.read_bytes()
        if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("exported geometry artifact changed")
        art = ReconstructionArtifact.model_validate_json(raw)
        if art.metric_model not in (None, "none"):
            raise ValueError("legacy base registry contains metric augmentation")
        if art.scene_name in result:
            raise ValueError("multiple legacy reconstructions for one scene")
        result[art.scene_name] = path
    return result


def attach(scene: dict, artifact: Path) -> dict:
    if scene["status"] != "frozen" or len(scene["frames"]) != 32:
        raise ValueError("scene is not a complete frozen input")
    pixels = []
    for row in scene["frames"]:
        raw = Path(row["source_path"]).read_bytes()
        if len(raw) != row["size"] or hashlib.sha256(raw).hexdigest() != row["png_sha256"]:
            raise ValueError("frozen PNG bytes changed")
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("unreadable frozen PNG")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (list(image.shape) != row["shape"] or str(image.dtype) != row["dtype"]
                or hashlib.sha256(image.tobytes()).hexdigest() != row["rgb_sha256"]):
            raise ValueError("frozen RGB identity changed")
        pixels.append(image)
    art = ReconstructionArtifact.model_validate_json(artifact.read_bytes())
    base = None
    identities = {}
    for row in scene["questions"]:
        if "ground_truth" in row["episode"]:
            raise ValueError("public episode contains a label field")
        episode = VSIBenchEpisode.model_validate({**row["episode"], "ground_truth": ""})
        frame_set = with_pixel_identity(episode.frame_set, pixels)
        if (art.scene_name != scene["scene_id"]
                or frame_set.cache_identity() != art.reconstruction_input_identity
                or art.frame_ids != frame_set.readable_frame_ids
                or art.source_frame_indices != frame_set.source_frame_indices
                or art.timestamps != frame_set.timestamps):
            raise ValueError("legacy geometry and frozen source frames differ")
        item = EpisodeItem(episode, pixels, source=row["source"], video_path=row["video_path"])
        if base is None:
            base = input_identity(item, artifact)
        value = copy.deepcopy(base)
        value.pop("sha256")
        value["episode"] = episode.model_dump(mode="json")
        value["sha256"] = content_sha256(value)
        identities[episode.qa_id] = value
    if set(identities) != set(scene["qa_ids"]):
        raise ValueError("geometry attachment does not cover the complete QA group")
    return {
        "scene_id": scene["scene_id"],
        "qa_ids": scene["qa_ids"],
        "artifact_source_path": str(artifact),
        "logical_input_identities": identities,
        "status": "verified_reuse",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-manifest", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    available = registry(json.loads(args.export_manifest.read_text()))
    report = {
        "schema": "harness3d-geometry-attachment/1",
        "status": "running", "scenes": {}, "failures": [], "missing_scenes": [],
        "labels_accessed": False, "frozen_inputs_rewritten": False,
    }
    began = time.monotonic()
    for path in sorted((args.frozen_root / "scenes").glob("*.json")):
        scene = json.loads(path.read_text())
        artifact = available.get(scene["scene_id"])
        if artifact is None:
            report["missing_scenes"].append(scene["scene_id"])
        else:
            try:
                report["scenes"][scene["scene_id"]] = attach(scene, artifact)
            except Exception as exc:
                report["failures"].append({
                    "scene_id": scene["scene_id"], "error": type(exc).__name__,
                    "detail": str(exc),
                })
        report["elapsed_s"] = time.monotonic() - began
        write(args.output, report)
    report["status"] = "verified" if not report["failures"] else "incomplete"
    report["verified_scenes"] = len(report["scenes"])
    report["verified_questions"] = sum(len(v["qa_ids"]) for v in report["scenes"].values())
    write(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "scenes"}), flush=True)
    if report["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
