#!/usr/bin/env python3
"""Freeze the authorized public question pool before any model request."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from skill3d.adapters import vsibench_loader as vl
from skill3d.adapters.episode_source import EpisodeItem, _episode_from_row, _load_from_sources
from skill3d.adapters.frame_set import with_pixel_identity
from skill3d.evaluation.skill_ablation_v11 import input_identity
from skill3d.infra.m5_identity import content_sha256
from skill3d.online.config import load_yaml
from skill3d.schemas import ReconstructionArtifact


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def frozen_scene(rows, scene, split, geometry_row, root, video_root, fallbacks):
    qa_ids = sorted(row["qa_id"] for row in rows)
    path = None
    pixels = None
    loaded = None
    if geometry_row is not None:
        artifact = geometry_row["artifact"]
        path = Path(artifact if isinstance(artifact, str) else artifact["path"])
        art = ReconstructionArtifact.model_validate_json(path.read_bytes())
        frames = sorted(path.with_name(path.stem + "_frames").glob("*.png"))
        if len(frames) != 32 or len(art.frame_ids) != 32:
            raise ValueError("existing geometry does not have exactly 32 frozen images")
        pixels = []
        for frame in frames:
            image = cv2.imread(str(frame))
            if image is None:
                raise ValueError("existing frozen PNG is unreadable")
            pixels.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for source in vl.video_source_candidates(rows[0], video_root, fallbacks):
            if not source.path.is_file():
                continue
            cap = cv2.VideoCapture(str(source.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            cap.release()
            if total < 32:
                continue
            fs = with_pixel_identity(vl.build_frame_set(
                total, n_frames=32, fps=fps, episode_id=qa_ids[0],
                dataset_id=rows[0]["dataset"], video_id=vl.video_id_for(rows[0], source),
                scene_name=scene), pixels)
            if (fs.source_frame_indices == art.source_frame_indices and
                    fs.timestamps == art.timestamps and
                    fs.cache_identity() == art.reconstruction_input_identity):
                loaded = {"source": source, "total": total, "fps": fps, "pixels": pixels, "missing": []}
                break
        if loaded is None:
            raise ValueError("existing frozen geometry source identity does not replay")
    else:
        loaded, attempts = _load_from_sources(rows[0], video_root, fallbacks, 32)
        if loaded is None or loaded["missing"] or len(loaded["pixels"]) != 32:
            return {"scene_id": scene, "qa_ids": qa_ids, "split": split,
                    "status": "input_error", "source_attempts": attempts}
        pixels = loaded["pixels"]
        frames = []
        for slot, image in enumerate(pixels):
            frame = root / "new_frames" / scene / f"{slot:04d}.png"
            frame.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(frame), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise ValueError("cannot freeze new RGB image")
            frames.append(frame)
    questions = []
    logical_base = None
    for row in sorted(rows, key=lambda r: r["qa_id"]):
        fs = vl.build_frame_set(loaded["total"], n_frames=32, fps=loaded["fps"],
                                episode_id=row["qa_id"], dataset_id=row["dataset"],
                                video_id=vl.video_id_for(row, loaded["source"]), scene_name=scene)
        public_row = {**row, "ground_truth": ""}
        ep = _episode_from_row(public_row, split, pixels, 32, frame_set=fs)
        item = EpisodeItem(ep, pixels, source="vsi_bench", video_path=str(loaded["source"].path))
        serialized = ep.model_dump(mode="json")
        if serialized.pop("ground_truth") != "":
            raise ValueError("precompute episode unexpectedly contains a label")
        record = {"episode": serialized, "source": item.source, "video_path": item.video_path}
        if path is not None:
            if logical_base is None:
                logical_base = input_identity(item, path)
            identity = copy.deepcopy(logical_base)
            identity.pop("sha256")
            identity["episode"] = ep.model_dump(mode="json")
            identity["sha256"] = content_sha256(identity)
            # This compatibility identity contains a constant empty label field.
            # The public question/queue manifest above never contains labels.
            record["logical_input_identity"] = identity
        questions.append(record)
    frame_rows = []
    for frame, image in zip(frames, pixels):
        raw = frame.read_bytes()
        frame_rows.append({"source_path": str(frame), "png_sha256": hashlib.sha256(raw).hexdigest(),
                           "size": len(raw), "rgb_sha256": hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                           "shape": list(image.shape), "dtype": str(image.dtype)})
    return {"schema": "harness3d-frozen-scene/1", "scene_id": scene, "split": split,
            "qa_ids": qa_ids, "questions": questions, "frames": frame_rows,
            "artifact_source_path": str(path) if path is not None else None,
            "status": "frozen", "real_labels_available": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--fallback-root", action="append", default=[])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    scope = json.loads(args.scope.read_text())
    geometry = json.loads(args.geometry_manifest.read_text())["scenes"]
    mapping = load_yaml(args.split_config)
    splits = {}
    for split in ("induction", "inner_validation", "outer_holdout"):
        for scene in mapping[split + "_scene_ids"]:
            if str(scene) in splits:
                raise ValueError("split scene sets overlap")
            splits[str(scene)] = split
    grouped = {}
    for row in scope["questions"]:
        if any(key in row for key in ("ground_truth", "answer", "reference_answer")):
            raise ValueError("public scope contains a forbidden answer field")
        grouped.setdefault(row["scene_name"], []).append(row)
    if sum(map(len, grouped.values())) != 1154:
        raise ValueError("authorized scope changed")
    status = {"schema": "harness3d-input-freeze-progress/1", "planned_scenes": len(grouped),
              "planned_questions": 1154, "frozen_scenes": 0, "frozen_questions": 0,
              "failures": [], "status": "running"}
    began = time.monotonic()
    for scene, rows in sorted(grouped.items()):
        target = args.root / "scenes" / f"{scene}.json"
        if target.is_file():
            value = json.loads(target.read_text())
            if value["qa_ids"] != sorted(row["qa_id"] for row in rows):
                raise ValueError("frozen scene question group changed")
        else:
            try:
                value = frozen_scene(rows, scene, splits[scene], geometry.get(scene), args.root,
                                     args.video_root, args.fallback_root)
            except Exception as exc:
                value = {"scene_id": scene, "qa_ids": sorted(r["qa_id"] for r in rows),
                         "status": "identity_error", "error": type(exc).__name__,
                         "detail": str(exc)}
            write(target, value)
        if value["status"] == "frozen":
            status["frozen_scenes"] += 1
            status["frozen_questions"] += len(rows)
        else:
            status["failures"].append({"scene_id": scene, "status": value["status"]})
        status["elapsed_s"] = time.monotonic() - began
        write(args.root / "progress.json", status)
    status["status"] = "frozen" if not status["failures"] else "incomplete"
    write(args.root / "progress.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
