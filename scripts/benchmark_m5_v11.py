#!/usr/bin/env python3
"""Measure one real question's prepared binding using existing scene geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_streaming_evolution_v11 import _base_config
from skill3d.adapters import vsibench_loader as vl
from skill3d.adapters.episode_source import (
    EpisodeItem, _episode_from_row, _load_from_sources, _row_id,
)
from skill3d.evolution.prepared_bindings_v11 import prepare_shared_bindings_v11
from skill3d.infra.m5_identity import _file_digest
from skill3d.online.config import load_config, resolve_runtime
from skill3d.segmentation.prepared_binding_store import PreparedBindingStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--qa-id", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=137)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

    try:
        rows = [json.loads(line) for line in Path(args.metadata).read_text().splitlines()
                if line.strip()]
        matches = [row for row in rows if _row_id(row) == args.qa_id]
        if len(matches) != 1:
            raise ValueError("benchmark qa_id is missing or duplicated")
        row = matches[0]
        loaded, attempts = _load_from_sources(row, args.video_root, None, 32)
        if loaded is None:
            raise ValueError(f"benchmark input is unreadable: {attempts}")
        readable = [frame for frame in loaded["planned"]
                    if frame not in set(loaded["missing"])]
        frame_set = vl.build_frame_set(
            loaded["total"], n_frames=32, fps=loaded["fps"],
            episode_id=args.qa_id, dataset_id=str(row.get("dataset") or ""),
            video_id=vl.video_id_for(row, loaded["source"]),
            scene_name=str(row["scene_name"]), readable_frame_ids=readable)
        episode = _episode_from_row(
            row, "induction", loaded["pixels"], 32, frame_set=frame_set)
        item = EpisodeItem(
            episode=episode, pixels=loaded["pixels"], source="vsi_bench",
            video_path=str(loaded["source"].path))
        raw = resolve_runtime(load_config(args.config), seed=args.seed)
        cfg = _base_config(SimpleNamespace(seed=args.seed), raw)
        write("selection.json", {
            "qa_id": args.qa_id, "scene_name": episode.scene_name,
            "question_type": episode.question_type, "seed": args.seed,
            "artifact": str(Path(args.artifact).resolve()),
        })
        began = time.perf_counter()
        prepared = prepare_shared_bindings_v11(
            [item], artifact_paths={args.qa_id: args.artifact}, base_cfg=cfg,
            store_root=output / "binding_store", min_free_gib=80)
        prepare_s = time.perf_counter() - began
        write("prepared_bindings.json", prepared)
        reference = prepared["bindings"][args.qa_id]
        store = PreparedBindingStore(prepared["store_root"])
        binding = store.load(
            reference["key"], expected_request_identity=reference["request_identity"],
            scene_artifact=args.artifact,
            scene_input_identity={"sha256": reference["scene_input_identity_sha256"]})
        stats = binding.stats
        began = time.perf_counter()
        replay = prepare_shared_bindings_v11(
            [item], artifact_paths={args.qa_id: args.artifact}, base_cfg=cfg,
            store_root=output / "binding_store", min_free_gib=80)
        report = {
            "status": "completed", "qa_id": args.qa_id,
            "preparation_s": prepare_s, "replay_s": time.perf_counter() - began,
            "cache_hit_qa_ids": replay["cache_hit_qa_ids"],
            "binding_status": stats.get("status"),
            "stage_timings": stats.get("stage_timings"),
            "detector_requests": stats.get("actual_detector_requests"),
            "semantic_requests": (stats.get("semantic_review") or {}).get("n_visual_requests"),
            "review_selection": stats.get("review_selection"),
            "n_objects": len(binding.objects),
            "validated_objects": sum(
                obj.binding_validation is not None
                and obj.binding_validation.status == "validated" for obj in binding.objects),
            "hash_cache": _file_digest.cache_info()._asdict(),
        }
        write("benchmark.json", report)
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        write("failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
