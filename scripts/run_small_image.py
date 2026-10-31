#!/usr/bin/env python3
"""No-request small-image preflight by default; --execute is an explicit opt-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from skill3d.small_image.datasets import load_manifest, load_problems
from skill3d.small_image.runner import (
    SmallImageRunnerConfig, SmallImageSolver, preflight_small_image,
    validate_categories,
)
from skill3d.synthesis.bailian_client import BailianClient


def build_precomputer(config, *, seed=137):
    """Exact existing adapter APIs, imported only after explicit execution opt-in."""
    from skill3d.small_image.backends import (
        GroundingDINOAdapter, InternCategoryAdapter, MoGe2Adapter, SAM2ImageAdapter,
    )
    from skill3d.small_image.precompute import SmallImagePrecomputer

    for key, upper in (("max_categories", 12), ("max_candidates_per_view", 4)):
        if type(config.get(key)) is not int or not 1 <= config[key] <= upper:
            raise ValueError(f"invalid bounded precompute {key}")
    detector_config = dict(config["detector"])
    if detector_config.get("max_attempts") != 1:
        raise ValueError("small-image detector must use exactly one service attempt")
    if not detector_config.get("endpoint") or not config["review"].get("endpoint"):
        raise ValueError("verified execution endpoints must be supplied explicitly")
    sam2_config = dict(config["sam2"])
    moge_config = dict(config["moge2"])
    # Prevent SAM2's existing HF-ID path from downloading a missing checkpoint.
    if (not isinstance(sam2_config.get("checkpoint"), str)
            or not Path(sam2_config["checkpoint"]).is_file()
            or not moge_config.get("checkpoint")):
        raise ValueError("explicit locally available SAM2 and MoGe2 checkpoints required")
    reviewer = InternCategoryAdapter(config["review"], seed=seed)
    detector = GroundingDINOAdapter(**detector_config)
    segmenter = SAM2ImageAdapter(**sam2_config)
    depth = MoGe2Adapter(**moge_config, device=config["device"])
    return SmallImagePrecomputer(
        detector=detector, segmenter=segmenter, reviewer=reviewer, depth_estimator=depth,
        model_identities={
            "detector": detector.identity, "segmenter": segmenter.identity,
            "reviewer": reviewer.identity, "depth_estimator": depth.identity,
        },
        expected_review_identity_sha256=reviewer.expected_identity_sha256,
        cache_dir=config.get("cache_dir"), yes_threshold=reviewer.client.yes_threshold,
        resize_to=None, max_categories=config["max_categories"],
        max_candidates_per_view=config["max_candidates_per_view"],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=(
        Path(__file__).resolve().parents[1] / "configs" / "small_image.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--benchmark", choices=["blink", "cv3d", "mmsi"])
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--manifest", type=Path,
                        help="verified public replay manifest, never an answer file")
    parser.add_argument("--categories", nargs="*",
                        help="explicit category nouns; an empty list skips extraction")
    parser.add_argument("--execute", action="store_true",
                        help="allow model requests and explicitly configured GPU preparation")
    args = parser.parse_args(argv)
    if args.limit < 1 or (args.manifest is None and args.benchmark is None):
        parser.error("positive --limit and either --manifest or --benchmark are required")
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        runner_config = SmallImageRunnerConfig(**config.get("solver", {}))
        categories = (validate_categories(args.categories, max_categories=runner_config.max_categories)
                      if args.categories is not None else None)
        client = BailianClient(**config["model"])
        root = args.dataset_root or config["dataset_root"]
        if args.manifest is not None:
            problems = load_manifest(args.manifest, root)
            if args.benchmark is not None:
                problems = [p for p in problems if p.benchmark == args.benchmark]
            problems = [p for p in problems if p.source_split == args.split][:args.limit]
        else:
            problems = load_problems(root, args.benchmark, args.split, args.limit)
        if not problems:
            raise ValueError("no selected public inputs")
    except Exception:
        print(json.dumps({"status": "configuration_or_input_error", "request_count": 0}))
        return 2

    # CPU preflight always precedes backend construction; no credentials are read here.
    preflights = [preflight_small_image(p, client=client, config=runner_config) for p in problems]
    if not args.execute or any(p.status != "preflight" for p in preflights):
        for result in preflights:
            print(json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=True))
        return 0 if all(p.status == "preflight" for p in preflights) else 2

    try:
        precomputer = build_precomputer(config["precompute"], seed=runner_config.seed)
    except Exception:
        print(json.dumps({"status": "precompute_configuration_error", "request_count": 0}))
        return 2
    solver = SmallImageSolver(
        client, config=runner_config, prepare=precomputer.process)
    failed = False
    for problem in problems:
        result = solver(problem, categories=categories)
        print(json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=True))
        if result.status != "answered":
            # Never retry a failed service, or silently proceed with a broken backend.
            failed = True
            break
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
