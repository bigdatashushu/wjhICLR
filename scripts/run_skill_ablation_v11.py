#!/usr/bin/env python3
"""Run frozen v11 S0 B01/B11 pairs; see docs/skill_ablation_v11.md."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from skill3d.adapters.episode_source import load_jsonl_items, load_vsi_bench_items
from skill3d.evaluation.skill_ablation_v11 import (
    DEFAULT_LIBRARY,
    PairingError,
    quality_contract,
    run_skill_ablation_v11,
)
from skill3d.online.config import (
    active_vision_from, load_config, load_yaml, retrieval_policy_from, sandbox_from,
)
from skill3d.online.runner import OnlineRunConfig
from skill3d.reconstruction.run import resolve_artifact_path
from skill3d.routing.task_classifier import canonical_task


def _load_quality_confirmations(path: Path, items):
    value = json.loads(path.read_text(encoding="utf-8"))
    if "scenes" not in value:
        return value, None
    scene_paths = {
        str(row["scene_name"]): Path(str(row["path"]))
        for row in value["scenes"]
    }
    confirmations = {}
    missing = []
    for item in items:
        scene = str(item.episode.scene_name)
        confirmation_path = scene_paths.get(scene)
        if confirmation_path is not None and not confirmation_path.is_absolute():
            confirmation_path = path.parent / confirmation_path
        if confirmation_path is None or not confirmation_path.is_file():
            missing.append(scene)
            continue
        confirmations[item.episode.qa_id] = json.loads(
            confirmation_path.read_text(encoding="utf-8"))
    if missing:
        raise PairingError(
            f"missing scene quality confirmations: {sorted(set(missing))}")
    return None, confirmations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="v11 B01(no Skill) / B11(frozen S0), quality always on")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source", choices=["jsonl", "vsi_bench"], default="vsi_bench")
    parser.add_argument("--episodes-jsonl")
    parser.add_argument("--split", default="inner_validation",
                        choices=["induction", "inner_validation", "outer_holdout", "final_test"])
    parser.add_argument("--question-types", default="object_rel_distance", help="comma-separated canonical tasks")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video-root")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--sampling-per-task", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recon-dir", help="existing P1 reconstruction directory")
    parser.add_argument("--artifacts-json", help="explicit JSON {qa_id: artifact_path}, alternative to --recon-dir")
    parser.add_argument(
        "--reconstruction-costs-json",
        help="optional JSON {qa_id: elapsed_seconds} from the frozen reconstruction stage",
    )
    parser.add_argument("--expected-qa-ids", help="JSON list of preregistered qa_id values")
    parser.add_argument("--library-root", default=str(DEFAULT_LIBRARY))
    parser.add_argument("--vllm-endpoint", action="append", default=[])
    parser.add_argument("--vllm-model")
    parser.add_argument("--max-solver-rounds", type=int)
    parser.add_argument("--max-retries-per-operation", type=int)
    parser.add_argument("--finalization-rounds", type=int)
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--quality-confirmation", help="confirmed current quality contract with evidence_ref")
    parser.add_argument("--print-quality-contract", action="store_true")
    parser.add_argument("--output-dir", help="new directory; existing directories are rejected")
    args = parser.parse_args(argv)
    if args.print_quality_contract:
        print(json.dumps(quality_contract(), ensure_ascii=False, indent=2))
        return 0
    if not args.output_dir:
        parser.error("--output-dir is required")
    root = Path(args.output_dir)
    if root.exists():
        parser.error("--output-dir must not already exist")
    sampling, exclusions = {}, []
    try:
        if not Path(args.config).is_file():
            raise PairingError(f"config not found: {args.config}")
        raw = load_config(args.config)
        seed = int(args.seed if args.seed is not None else raw["seed"])
        paths = raw.get("paths", {})
        model = raw.get("vllm", {})
        vision, sandbox = active_vision_from(raw), sandbox_from(raw)
        tasks = [canonical_task(t.strip()) for t in args.question_types.split(",") if t.strip()]
        if not tasks:
            raise PairingError("empty question-types")
        detector = raw.get("detection", {}).get("endpoint")
        if detector:
            os.environ.setdefault("SKILL3D_DETECTOR_ENDPOINT", str(detector))
        from skill3d.env_preflight import assert_runtime_dependencies

        assert_runtime_dependencies(context="skill_ablation_v11")
        if args.source == "jsonl":
            if not args.episodes_jsonl:
                raise PairingError("--episodes-jsonl required for jsonl source")
            items = load_jsonl_items(
                args.episodes_jsonl,
                args.split,
                exclusions=exclusions,
                include_input_errors=True,
            )
            items = [it for it in items if canonical_task(it.episode.question_type) in tasks]
            if args.limit:
                items = items[:args.limit]
        else:
            items = load_vsi_bench_items(
                args.split, load_yaml(raw["split_config"]),
                video_root=args.video_root or paths.get("raw_videos", "data/raw_videos"),
                video_fallback_roots=paths.get("raw_video_fallbacks"),
                cache_dir=paths.get("vsi_bench_meta"),
                question_types=tasks, datasets=args.datasets.split(",") if args.datasets else None,
                n_frames=int(raw.get("frame_sampling", {}).get("n_frames", 32)),
                seed=seed, sampling_seed=seed, stratified_per_task=args.sampling_per_task,
                limit=args.limit or None, sampling_receipt=sampling, exclusions=exclusions,
                include_input_errors=True)
        expected = (json.loads(Path(args.expected_qa_ids).read_text()) if args.expected_qa_ids
                    else sampling.get("qa_ids"))
        cfg = OnlineRunConfig(
            mode="real", seed=seed, allow_final_test=args.allow_final_test,
            vllm_endpoints=args.vllm_endpoint or model.get("endpoints", []),
            vllm_model=args.vllm_model or model["model"],
            max_images=int(model.get("n_frames", 32)), max_pixels=int(model["max_pixels"]),
            max_model_len=int(model["max_model_len"]), max_tokens=int(model.get("max_tokens", 4096)),
            cell_timeout_s=sandbox.cell_timeout_s, max_regen=sandbox.max_regenerate,
            max_solver_rounds=args.max_solver_rounds if args.max_solver_rounds is not None else raw["max_solver_rounds"],
            max_retries_per_operation=(args.max_retries_per_operation
                                       if args.max_retries_per_operation is not None else raw["max_retries_per_operation"]),
            finalization_rounds=args.finalization_rounds if args.finalization_rounds is not None else raw["finalization_rounds"],
            image_layout=vision.layout, max_derived_images=vision.max_derived_images,
            retrieval_policy=retrieval_policy_from(raw),
        )
        if not cfg.vllm_endpoints:
            raise PairingError("no vLLM endpoint; supply --vllm-endpoint")
        if args.artifacts_json:
            artifacts = json.loads(Path(args.artifacts_json).read_text())
        else:
            recon = args.recon_dir or paths.get("reconstructions", "data/reconstructions")
            artifacts = {
                it.episode.qa_id: str(resolve_artifact_path(
                    recon, it.episode.scene_name, "vggt",
                    frame_set=it.episode.frame_set)[0])
                for it in items if it.input_error is None}
        confirmation, confirmations = (None, None)
        if args.quality_confirmation:
            confirmation, confirmations = _load_quality_confirmations(
                Path(args.quality_confirmation), items)
        reconstruction_costs = (
            json.loads(Path(args.reconstruction_costs_json).read_text())
            if args.reconstruction_costs_json
            else {}
        )
        result = run_skill_ablation_v11(
            items, artifact_paths=artifacts, output_dir=root, base_cfg=cfg, seed=seed,
            question_types=tasks, expected_qa_ids=expected, library_root=args.library_root,
            quality_confirmation=confirmation,
            quality_confirmations=confirmations,
            reconstruction_costs=reconstruction_costs)
        (root / "source_receipt.json").write_text(
            json.dumps({"sampling": sampling, "exclusions": exclusions}, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 1
    except Exception as exc:
        # The driver writes its own failure receipt. Loading can fail before it
        # starts; retain the preregistered panel and exclusions in that case too.
        if not root.exists():
            root.mkdir(parents=True)
            (root / "source_failure.json").write_text(json.dumps({
                "status": "incomplete", "error": f"{type(exc).__name__}: {exc}",
                "sampling": sampling, "exclusions": exclusions,
            }, ensure_ascii=False, indent=2))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
