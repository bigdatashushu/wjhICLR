#!/usr/bin/env python3
"""No-request small-image preflight by default; --execute is an explicit opt-in."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

from skill3d.small_image.datasets import load_manifest, load_problems
from skill3d.small_image.runner import (
    SmallImageRunnerConfig, SmallImageSolver, preflight_small_image,
    validate_categories, load_problem_frames, _validate_prepared, build_solver_prompt,
)
from skill3d.small_image.cv3d_protocol import (
    VERSION as CV3D_PROFILE, FrozenCV3DSkills, OfficialCV3DPrompts, task_tool_names,
)
from skill3d.small_image.storage import index_prepared, load_prepared
from skill3d.small_image.tools import build_registry, TOOL_NAMES
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig


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
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--benchmark", choices=["blink", "cv3d", "mmsi"])
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--manifest", type=Path,
                        help="verified public replay manifest, never an answer file")
    parser.add_argument("--categories", nargs="*",
                        help="explicit category nouns; an empty list skips extraction")
    parser.add_argument("--official-annotations", type=Path,
                        help="official CV-Bench 3D JSONL or verified public-only copy")
    parser.add_argument("--prepared-root", type=Path,
                        help="reuse existing prepared caches without online preprocessing")
    parser.add_argument("--skills-manifest", type=Path,
                        help="frozen CV3D task Skills with content hashes")
    parser.add_argument("--output-dir", type=Path,
                        help="new independent output directory for compact CV3D execution")
    parser.add_argument("--max-requests", type=int,
                        help="explicit run-wide request budget for compact execution")
    parser.add_argument("--execute", action="store_true",
                        help="allow model requests and explicitly configured GPU preparation")
    args = parser.parse_args(argv)
    if args.limit < 1 or (args.manifest is None and args.benchmark is None):
        parser.error("positive --limit and either --manifest or --benchmark are required")
    try:
        repository = Path(__file__).resolve().parents[1]
        config_path = args.config or repository / "configs" / (
            "cv3d_program.yaml" if args.benchmark == "cv3d" else "small_image.yaml")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
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
        compact = runner_config.prompt_profile == CV3D_PROFILE
        if compact:
            if args.official_annotations is None or args.prepared_root is None:
                raise ValueError("compact CV3D needs official annotations and prepared cache root")
            skill_path = args.skills_manifest or Path(config["skills_manifest"])
            if not skill_path.is_absolute():
                skill_path = repository / skill_path
            frozen_skills = FrozenCV3DSkills(skill_path)
            official = OfficialCV3DPrompts(args.official_annotations)
            prepared_index = index_prepared(args.prepared_root)
            public_prompts = {p.sample_id: official.for_problem(p) for p in problems}
            if any(p.input_sha256 not in prepared_index for p in problems):
                raise ValueError("missing existing prepared cache")
            if args.execute and (
                    args.output_dir is None or args.max_requests is None
                    or args.max_requests < 1 or args.output_dir.exists()):
                raise ValueError("compact execution needs new output and explicit total budget")
    except Exception:
        print(json.dumps({"status": "configuration_or_input_error", "request_count": 0}))
        return 2

    # CPU preflight always precedes backend construction; no credentials are read here.
    preflights = [preflight_small_image(p, client=client, config=runner_config) for p in problems]
    compact_identity = None
    if compact and all(p.status == "preflight" for p in preflights):
        try:
            previews = {}
            for problem in problems:
                prepared = load_prepared(prepared_index[problem.input_sha256], problem=problem)
                frames = load_problem_frames(problem)
                if _validate_prepared(prepared, problem, frames) is not None:
                    raise ValueError("prepared identity mismatch")
                if 0 not in prepared.geometries or (
                        problem.task_type == "Distance"
                        and prepared.geometries[0].get("points") is None):
                    raise ValueError("required predicted geometry missing")
                registry, scene = build_registry(prepared, problem)
                allowed = set(registry.names_for_scope(
                    scene.question_tool_scope, scene.available_artifacts,
                    evidence_profile=scene.evidence_profile,
                    question_type=scene.question_type, local_facts=scene.local_facts))
                allowed &= task_tool_names(problem.task_type)
                if set(registry.names()) - TOOL_NAMES:
                    raise ValueError("unexpected tool registry")
                prompt = build_solver_prompt(
                    problem, frames, registry, allowed,
                    skill_text=frozen_skills.texts[problem.task_type],
                    profile=CV3D_PROFILE, official_prompt=public_prompts[problem.sample_id],
                    remaining_requests=6)
                from skill3d.small_image.blink_protocol import sha_value, original_messages
                previews[problem.sample_id] = {
                    "input_sha256": problem.input_sha256,
                    "artifact_sha256": prepared.artifact_sha256,
                    "initial_messages_sha256": sha_value(original_messages(problem, prompt)),
                    "official_prompt_sha256": hashlib.sha256(
                        public_prompts[problem.sample_id].encode("utf-8")).hexdigest(),
                    "skill_sha256": frozen_skills.identities[problem.task_type],
                }
            compact_identity = {
                "protocol": CV3D_PROFILE, "evaluation_role": "test_development",
                "official_annotations_sha256": official.source_sha256,
                "skills_manifest_sha256": frozen_skills.manifest_sha256,
                "solver": config["solver"], "inputs": previews,
                "source_sha256": {
                    str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        repository / "scripts/run_small_image.py",
                        repository / "src/skill3d/small_image/runner.py",
                        repository / "src/skill3d/small_image/cv3d_protocol.py",
                        repository / "src/skill3d/small_image/tools.py",
                        repository / "src/skill3d/small_image/storage.py",
                        repository / "src/skill3d/small_image/blink_protocol.py",
                    )
                },
                "run_request_budget": args.max_requests,
                "model_requests": 0, "gpu_requests": 0,
            }
        except Exception:
            print(json.dumps({"status": "cv3d_preflight_error", "request_count": 0}))
            return 2
    if not args.execute or any(p.status != "preflight" for p in preflights):
        for result in preflights:
            print(json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=True))
        return 0 if all(p.status == "preflight" for p in preflights) else 2

    if compact:
        # All original bytes, cache identities, prompts and Skills were checked
        # before the first governed model request. No GPU backend is constructed.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / "manifest.json").open("x") as stream:
            json.dump(compact_identity, stream, sort_keys=True, ensure_ascii=False)
        governor = RequestGovernor(
            args.output_dir / "budget", RequestGovernorConfig(args.max_requests, 1))
        failed = False
        for problem in problems:
            if (FrozenCV3DSkills(skill_path).manifest_sha256 != frozen_skills.manifest_sha256
                    or any(hashlib.sha256((repository / relative).read_bytes()).hexdigest() != expected
                           for relative, expected in compact_identity["source_sha256"].items())):
                print(json.dumps({"status": "frozen_source_or_skill_changed", "request_count": 0}))
                return 2
            prepared = load_prepared(prepared_index[problem.input_sha256], problem=problem)
            # Revalidate each input after preflight and before admitting a request.
            solver = SmallImageSolver(
                client, config=runner_config, governor=governor,
                skill_text=frozen_skills.texts[problem.task_type],
                official_prompt=public_prompts[problem.sample_id])
            result = solver(problem, prepared=prepared)
            with (args.output_dir / (problem.sample_id + ".json")).open("x") as stream:
                json.dump(result.to_dict(), stream, sort_keys=True, ensure_ascii=True)
            print(json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=True), flush=True)
            if result.status != "answered":
                failed = True
                break
        return 2 if failed else 0

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
