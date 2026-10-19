#!/usr/bin/env python3
"""Prepare a resumable scene-grouped v11 question pool within a disk budget."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_streaming_evolution_v11 import _base_config
from skill3d.adapters import vsibench_loader as vl
from skill3d.adapters.episode_source import (
    EpisodeItem, _episode_from_row, _load_from_sources, _row_id,
)
from skill3d.evaluation.skill_ablation_v11 import initial_state_for_item, input_identity
from skill3d.evolution.prepared_bindings_v11 import prepare_shared_bindings_v11
from skill3d.infra.m5_identity import content_sha256, file_identity, source_identity
from skill3d.online.config import load_config, load_yaml, resolve_runtime
from skill3d.reconstruction.run import plan_scene_jobs, run_jobs
from skill3d.routing.task_classifier import TASK_TYPES, canonical_task
from skill3d.segmentation.prepared_binding_store import PreparedBindingStore

VERSION = "v11-scene-question-pool/1.0"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def tree_bytes(root):
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def select_scene_rows(rows, split_config, splits, *, preferred_scenes=(), datasets=()):
    """One deterministic question per canonical task, without consulting labels."""
    scene_split = {}
    for split in ("induction", "inner_validation", "outer_holdout", "final_test"):
        for scene in split_config[split + "_scene_ids"]:
            if str(scene) in scene_split:
                raise ValueError("split scene sets overlap")
            scene_split[str(scene)] = split
    groups = defaultdict(dict)
    for row in rows:
        scene = str(row["scene_name"])
        split = scene_split.get(scene)
        if split not in splits or datasets and row["dataset"] not in datasets:
            continue
        task = canonical_task(str(row["question_type"]))
        old = groups[scene].get(task)
        if old is None or _row_id(row) < _row_id(old):
            groups[scene][task] = row
    preferred = {scene: index for index, scene in enumerate(preferred_scenes)}
    ordered = sorted(groups, key=lambda scene: (
        0 if scene in preferred else 1,
        preferred.get(scene, 0),
        splits.index(scene_split[scene]), scene))
    return [
        {"scene_id": scene, "split": scene_split[scene],
         "rows": [groups[scene][task] for task in TASK_TYPES if task in groups[scene]]}
        for scene in ordered]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--reuse-recon-dir", default="")
    parser.add_argument("--splits", default="induction,inner_validation")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--preferred-scenes", default="")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--min-free-gib", type=float, default=80)
    parser.add_argument("--scene-working-gib", type=float, default=8)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args(argv)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits or any(s not in {"induction", "inner_validation", "outer_holdout"} for s in splits):
        raise ValueError("only induction/inner_validation/outer_holdout may be prepared")
    if "outer_holdout" in splits and len(splits) > 1:
        raise ValueError("outer holdout preparation must run as a separate batch")
    if (not math.isfinite(args.min_free_gib) or not math.isfinite(args.scene_working_gib)
            or args.min_free_gib < 80 or args.scene_working_gib < 1 or args.max_scenes < 0):
        raise ValueError("reserve must be >=80 GiB and working allowance >=1 GiB")
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw = resolve_runtime(load_config(args.config), seed=args.seed)
    cfg = _base_config(SimpleNamespace(seed=args.seed), raw)
    rows = [json.loads(line) for line in Path(args.metadata).read_text().splitlines()
            if line.strip()]
    by_id = {_row_id(row): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate metadata question identities")
    grouped = select_scene_rows(
        rows, load_yaml(raw["split_config"]), splits,
        preferred_scenes=[s for s in args.preferred_scenes.split(",") if s],
        datasets=[s for s in args.datasets.split(",") if s])
    if args.max_scenes:
        grouped = grouped[:args.max_scenes]
    selection = {
        "schema_version": VERSION, "seed": args.seed,
        "metadata": file_identity(args.metadata),
        "split_config": file_identity(raw["split_config"]),
        "config": file_identity(args.config),
        "video_root": str(Path(args.video_root).resolve()),
        "sources": source_identity(Path(__file__).resolve().parents[1] / "src/skill3d"),
        "script_sha256": file_identity(__file__)["sha256"],
        "splits": splits, "one_question_per_task_per_scene": True,
        "scene_groups": [
            {"scene_id": group["scene_id"], "split": group["split"],
             "questions": [{"qa_id": _row_id(row), "question_type": canonical_task(row["question_type"])}
                           for row in group["rows"]]}
            for group in grouped],
        "min_free_gib": args.min_free_gib, "scene_working_gib": args.scene_working_gib,
        "reuse_recon_dir": str(Path(args.reuse_recon_dir).resolve()) if args.reuse_recon_dir else "",
        "pool_role": "precomputed-input-pool; exact-slot panels selected separately",
        "reference_answers_sent_to_models": False,
    }
    selection["sha256"] = content_sha256(selection)
    selection_path = root / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("frozen precomputation selection/config/source changed")
    else:
        write(selection_path, selection)
    if args.freeze_only:
        print(json.dumps({"status": "frozen", "n_scenes": len(grouped),
                          "n_questions": sum(len(group["rows"]) for group in grouped)}))
        return 0
    if not os.environ.get(cfg.online_api_key_env, "").strip():
        raise ValueError("online parser credential is not configured")
    store_root = root / "binding_store"
    recon_root = root / "reconstructions"
    manifest_path = root / "prepared_bindings.json"
    combined = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": "v11-prepared-bindings/1.0", "store_root": str(store_root),
        "n_questions": 0, "n_scenes": 0, "max_questions_per_scene": 8,
        "min_free_gib": args.min_free_gib, "scene_groups": {}, "cache_hit_qa_ids": [],
        "bindings": {}}
    report_path = root / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "schema_version": VERSION, "selection_sha256": selection["sha256"],
        "status": "running", "scenes": [], "failures": [],
        "planned_scenes": len(grouped),
        "planned_questions": sum(len(group["rows"]) for group in grouped)}
    completed = {scene["scene_id"] for scene in report["scenes"]}
    input_errors = {row["scene_id"] for row in report["failures"] if row.get("status") == "input_error"}
    for group in grouped:
        scene = group["scene_id"]
        if scene in completed or scene in input_errors:
            continue
        free_gib = shutil.disk_usage(root).free / 1024 ** 3
        allowance = max(args.scene_working_gib, max(
            (r.get("peak_working_bytes", 0) / 1024 ** 3 * 1.5 for r in report["scenes"]),
            default=0))
        if free_gib < args.min_free_gib + allowance:
            report.update(status="capacity_reached", free_gib=free_gib,
                          next_scene_id=scene, required_working_gib=allowance)
            write(report_path, report)
            print(json.dumps({"status": report["status"], "free_gib": free_gib}), flush=True)
            return 0
        started = time.perf_counter()
        write(root / "progress.json", {"status": "loading", "scene_id": scene,
              "completed_scenes": len(completed), "completed_questions": combined["n_questions"]})
        loaded, attempts = _load_from_sources(
            group["rows"][0], args.video_root, None, 32)
        if loaded is None:
            report["failures"].append({"scene_id": scene, "status": "input_error",
                                      "attempts": attempts})
            write(report_path, report)
            continue
        readable = [index for index in loaded["planned"] if index not in set(loaded["missing"])]
        items = []
        for row in group["rows"]:
            frame_set = vl.build_frame_set(
                loaded["total"], n_frames=32, fps=loaded["fps"],
                episode_id=_row_id(row), dataset_id=str(row["dataset"]),
                video_id=vl.video_id_for(row, loaded["source"]),
                scene_name=scene, readable_frame_ids=readable)
            episode = _episode_from_row(row, group["split"], loaded["pixels"], 32,
                                        frame_set=frame_set)
            items.append(EpisodeItem(episode=episode, pixels=loaded["pixels"],
                source="vsi_bench", video_path=str(loaded["source"].path)))
        # This records frozen frame and question identities before reconstruction
        # or semantic requests. Reference answers never enter this receipt.
        write(root / "input_receipts" / f"{scene}.json", {
            "scene_id": scene, "qa_ids": [_row_id(row) for row in group["rows"]],
            "frame_set_hash": items[0].episode.frame_set.frame_set_hash,
            "pixel_hashes": items[0].episode.frame_set.pixel_sha256,
            "source_frame_indices": items[0].episode.frame_set.source_frame_indices,
        })
        try:
            jobs = plan_scene_jobs(items, recon_root, "vggt")
            if jobs[0].status == "pending" and args.reuse_recon_dir:
                jobs = plan_scene_jobs(items, args.reuse_recon_dir, "vggt")
            if any(job.status == "failed" for job in jobs):
                raise ValueError([job.note for job in jobs])
            if jobs[0].status != "skipped":
                jobs = run_jobs(jobs, {scene: items}, recon_root, "vggt")
            if jobs[0].status not in {"done", "skipped"}:
                raise ValueError([job.note for job in jobs])
            artifact = jobs[0].artifact_ref
            artifacts = {item.episode.qa_id: artifact for item in items}
            states = {item.episode.qa_id: initial_state_for_item(item, artifact) for item in items}
            write(root / "progress.json", {"status": "binding", "scene_id": scene,
                  "qa_ids": list(artifacts), "completed_scenes": len(completed),
                  "completed_questions": combined["n_questions"]})
            manifest = prepare_shared_bindings_v11(
                items, artifact_paths=artifacts, base_cfg=cfg, store_root=store_root,
                min_free_gib=args.min_free_gib)
            store = PreparedBindingStore(store_root)
            results = []
            for item in items:
                qa = item.episode.qa_id
                ref = manifest["bindings"][qa]
                binding = store.load(ref["key"], expected_request_identity=ref["request_identity"],
                    scene_artifact=artifact, scene_input_identity=input_identity(item, artifact))
                results.append({"qa_id": qa, "question_type": canonical_task(item.episode.question_type),
                    "evidence_state": states[qa], "status": binding.stats.get("status"),
                    "n_objects": len(binding.objects), "stage_timings": binding.stats.get("stage_timings"),
                    "entry_bytes": tree_bytes(store.entry_path(ref["key"])),
                    "model_requests": binding.stats.get("actual_model_requests"),
                    "detector_requests": binding.stats.get("actual_detector_requests")})
            combined["bindings"].update(manifest["bindings"])
            combined["scene_groups"].update(manifest["scene_groups"])
            combined["n_questions"] = len(combined["bindings"])
            combined["n_scenes"] = len(combined["scene_groups"])
            combined["cache_hit_qa_ids"] = sorted(set(
                combined["cache_hit_qa_ids"] + manifest["cache_hit_qa_ids"]))
            write(manifest_path, combined)
            report["scenes"].append({"scene_id": scene, "split": group["split"],
                "artifact": artifact, "elapsed_s": time.perf_counter() - started,
                "questions": results, "binding_bytes": sum(r["entry_bytes"] for r in results),
                "peak_working_bytes": max(0, int(
                    (free_gib - shutil.disk_usage(root).free / 1024 ** 3) * 1024 ** 3))})
            completed.add(scene)
            report.update(status="running", completed_scenes=len(completed),
                          completed_questions=combined["n_questions"],
                          free_gib=shutil.disk_usage(root).free / 1024 ** 3)
            write(report_path, report)
            print(json.dumps({"scene_id": scene, "completed_questions": combined["n_questions"],
                              "free_gib": report["free_gib"]}), flush=True)
        except Exception as exc:
            report["failures"].append({"scene_id": scene, "error": f"{type(exc).__name__}: {exc}"})
            report["status"] = "failed"
            write(report_path, report)
            raise
    report.update(status="completed_with_input_errors" if report["failures"] else "completed",
                  completed_questions=combined["n_questions"], completed_scenes=len(completed),
                  free_gib=shutil.disk_usage(root).free / 1024 ** 3)
    write(report_path, report)
    write(root / "progress.json", {"status": report["status"],
          "completed_questions": combined["n_questions"], "completed_scenes": len(completed)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
