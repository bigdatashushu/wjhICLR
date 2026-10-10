#!/usr/bin/env python3
"""Run the only v11 complete-SKILL.md evolution campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from skill3d.adapters.episode_source import (
    EpisodeItem,
    load_jsonl_items,
    load_vsi_bench_items,
)
from skill3d.evaluation.skill_ablation_v11 import quality_contract
from skill3d.evolution.adapters_v11 import (
    REVISION_PROMPT_VERSION,
    V11OfflineReviser,
    V11PairedEvaluator,
    V11PostPublishVerifier,
    model_config_identity_v11,
    solver_config_identity_v11,
)
from skill3d.evolution.campaign_v11 import (
    V11CampaignConfig,
    V11CampaignRunner,
)
from skill3d.evolution.experience_v11 import V11TraceCollector
from skill3d.governance.deepseek_client import (
    BASE_URL,
    MODEL_ID,
    DeepSeekClient,
)
from skill3d.online.config import (
    active_vision_from,
    load_config,
    load_yaml,
    retrieval_policy_from,
    sandbox_from,
)
from skill3d.online.runner import OnlineRunConfig
from skill3d.reconstruction.run import resolve_artifact_path
from skill3d.routing.task_classifier import canonical_task


class V11CampaignCliError(ValueError):
    """CLI inputs cannot define an auditable campaign."""


def _read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parent_trace_ids(trace_dir: str | Path) -> set[str]:
    path = Path(trace_dir) / "episode_input.jsonl"
    if not path.is_file():
        raise V11CampaignCliError(
            f"parent trace lacks episode_input.jsonl: {trace_dir}")
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qa_id = str(json.loads(line).get("qa_id") or "")
        if not qa_id or qa_id in result:
            raise V11CampaignCliError(
                "parent trace contains empty or duplicate qa_id")
        result.add(qa_id)
    if not result:
        raise V11CampaignCliError("parent trace contains no episodes")
    return result


def _load_items(
    args,
    raw: dict,
    *,
    split: str,
    limit: int,
    decode_limit: int | None = None,
) -> list[EpisodeItem]:
    paths = raw.get("paths", {})
    if args.source == "jsonl":
        if not args.episodes_jsonl:
            raise V11CampaignCliError(
                "--episodes-jsonl is required for --source=jsonl")
        items = load_jsonl_items(
            args.episodes_jsonl,
            split,
            limit=decode_limit if decode_limit is not None else limit or None,
            include_input_errors=True,
        )
    else:
        split_cfg = load_yaml(raw["split_config"])
        items = load_vsi_bench_items(
            split,
            split_cfg,
            video_root=args.video_root or paths.get(
                "raw_videos", "data/raw_videos"),
            video_fallback_roots=paths.get("raw_video_fallbacks"),
            cache_dir=paths.get("vsi_bench_meta"),
            question_types=[args.question_type],
            datasets=(
                [value for value in args.datasets.split(",") if value]
                if args.datasets else None
            ),
            n_frames=int(raw.get("frame_sampling", {}).get("n_frames", 32)),
            seed=args.seed,
            sampling_seed=args.seed,
            limit=decode_limit if decode_limit is not None else limit or None,
            include_input_errors=True,
        )
    selected = [
        item for item in items
        if canonical_task(item.episode.question_type) == args.question_type
    ]
    return selected[:limit] if limit else selected


def _artifact_mapping(
    items: list[EpisodeItem],
    *,
    explicit_json: str,
    recon_dir: str,
) -> dict[str, str]:
    if explicit_json:
        value = _read_json(explicit_json)
        if not isinstance(value, dict):
            raise V11CampaignCliError(
                f"artifact mapping must be a JSON object: {explicit_json}")
        mapping = {str(key): str(path) for key, path in value.items()}
    else:
        mapping = {
            item.episode.qa_id: str(resolve_artifact_path(
                recon_dir,
                item.episode.scene_name,
                "vggt",
                frame_set=item.episode.frame_set,
            )[0])
            for item in items
            if item.input_error is None
        }
    missing = sorted(
        item.episode.qa_id
        for item in items
        if item.input_error is None and item.episode.qa_id not in mapping
    )
    if missing:
        raise V11CampaignCliError(f"missing artifact mappings: {missing}")
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "v11 four-stage evolution: induction trace -> complete SKILL.md "
            "revision -> fixed E11 -> atomic publish -> fresh learning"
        )
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument(
        "--question-type",
        default="object_rel_distance",
        help="one canonical question type",
    )
    parser.add_argument("--parent-trace-dir", required=True)
    parser.add_argument("--source", choices=["jsonl", "vsi_bench"],
                        default="vsi_bench")
    parser.add_argument("--episodes-jsonl")
    parser.add_argument("--video-root")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inner-limit", type=int, default=16)
    parser.add_argument("--post-limit", type=int, default=4)
    parser.add_argument("--inner-artifacts-json", default="")
    parser.add_argument("--post-artifacts-json", default="")
    parser.add_argument("--recon-dir", default="")
    parser.add_argument("--expected-inner-qa-ids", default="")
    parser.add_argument("--quality-confirmation", required=True)
    parser.add_argument("--library-root", default="")
    parser.add_argument("--run-root", default="data/evolution/runs_v11")
    parser.add_argument("--max-revision-attempts", type=int, default=3)
    parser.add_argument("--vllm-endpoint", action="append", default=[])
    parser.add_argument("--vllm-model")
    parser.add_argument("--max-solver-rounds", type=int)
    parser.add_argument("--max-retries-per-operation", type=int)
    parser.add_argument("--finalization-rounds", type=int)
    parser.add_argument("--eval-visual-fallback", action=argparse.BooleanOptionalAction,
                        default=None, help="extra visual answer in evaluation splits only")
    parser.add_argument("--deepseek-base-url", default=BASE_URL)
    parser.add_argument("--deepseek-model", default=MODEL_ID)
    parser.add_argument("--deepseek-timeout-s", type=float, default=300.0)
    parser.add_argument("--deepseek-max-retries", type=int, default=3)
    parser.add_argument("--deepseek-max-tokens", type=int, default=32768)
    parser.add_argument(
        "--deepseek-reasoning-effort",
        choices=["none", "low", "high", "max"],
        default="high",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not Path(args.config).is_file():
            raise V11CampaignCliError(f"config not found: {args.config}")
        raw = load_config(args.config)
        args.question_type = canonical_task(args.question_type)
        if args.inner_limit <= 0 or args.post_limit <= 0:
            raise V11CampaignCliError("inner-limit and post-limit must be positive")
        args.seed = int(raw["seed"] if args.seed is None else args.seed)
        paths = raw.get("paths", {})
        model = raw.get("vllm", {})
        sandbox = sandbox_from(raw)
        vision = active_vision_from(raw)
        endpoints = args.vllm_endpoint or list(model.get("endpoints") or [])
        if not endpoints:
            raise V11CampaignCliError(
                "no vLLM endpoint; pass at least one --vllm-endpoint")
        detector = raw.get("detection", {}).get("endpoint")
        if detector:
            os.environ.setdefault("SKILL3D_DETECTOR_ENDPOINT", str(detector))

        from skill3d.env_preflight import assert_runtime_dependencies

        assert_runtime_dependencies(context="evolution_campaign_v11")
        parent_ids = _parent_trace_ids(args.parent_trace_dir)
        inner = _load_items(
            args, raw, split="inner_validation", limit=args.inner_limit)
        if not inner:
            raise V11CampaignCliError("inner panel is empty")
        decode_limit = len(parent_ids) + args.post_limit
        post_candidates = _load_items(
            args,
            raw,
            split="induction",
            limit=decode_limit,
            decode_limit=decode_limit,
        )
        post = [
            item for item in post_candidates
            if item.episode.qa_id not in parent_ids
        ][:args.post_limit]
        if len(post) < args.post_limit:
            raise V11CampaignCliError(
                "not enough fresh post-publish induction episodes after "
                "excluding parent learning qa_ids")
        if any(item.input_error is not None for item in post):
            raise V11CampaignCliError(
                "post-publish panel contains unreadable input; preregister "
                "another fresh induction panel")

        expected_inner = (
            list(_read_json(args.expected_inner_qa_ids))
            if args.expected_inner_qa_ids
            else [item.episode.qa_id for item in inner]
        )
        if expected_inner != [item.episode.qa_id for item in inner]:
            raise V11CampaignCliError(
                "expected inner qa_ids differ from the loaded frozen panel")
        quality_confirmation = _read_json(args.quality_confirmation)
        contract = quality_contract()
        if not (
            isinstance(quality_confirmation, dict)
            and quality_confirmation.get("confirmed") is True
            and quality_confirmation.get("sha256") == contract["sha256"]
            and quality_confirmation.get("evidence_ref")
        ):
            raise V11CampaignCliError(
                "quality confirmation must confirm the current contract and "
                "include evidence_ref")

        recon_dir = args.recon_dir or paths.get(
            "reconstructions", "data/reconstructions")
        inner_artifacts = _artifact_mapping(
            inner,
            explicit_json=args.inner_artifacts_json,
            recon_dir=recon_dir,
        )
        post_artifacts = _artifact_mapping(
            post,
            explicit_json=args.post_artifacts_json,
            recon_dir=recon_dir,
        )
        base_cfg = OnlineRunConfig(
            mode="real",
            baseline="C1_tools_program",
            seed=args.seed,
            skills=[],
            vllm_endpoints=endpoints,
            vllm_model=args.vllm_model or model["model"],
            max_images=int(model.get("n_frames", 32)),
            max_pixels=int(model["max_pixels"]),
            max_model_len=int(model["max_model_len"]),
            max_tokens=int(model.get("max_tokens", 4096)),
            cell_timeout_s=int(sandbox.cell_timeout_s),
            max_regen=int(sandbox.max_regenerate),
            max_solver_rounds=(
                args.max_solver_rounds
                if args.max_solver_rounds is not None
                else int(raw["max_solver_rounds"])
            ),
            max_retries_per_operation=(
                args.max_retries_per_operation
                if args.max_retries_per_operation is not None
                else int(raw["max_retries_per_operation"])
            ),
            finalization_rounds=(
                args.finalization_rounds
                if args.finalization_rounds is not None
                else int(raw["finalization_rounds"])
            ),
            eval_visual_fallback=(
                args.eval_visual_fallback if args.eval_visual_fallback is not None
                else bool(raw["eval_visual_fallback"])
            ),
            image_layout=vision.layout,
            max_derived_images=int(vision.max_derived_images),
            retrieval_policy=retrieval_policy_from(raw),
        )
        library_root = args.library_root or paths.get(
            "skill_library", "skill_library")
        model_identity = model_config_identity_v11(base_cfg)
        solver_identity = solver_config_identity_v11(
            base_cfg, seed=args.seed)
        campaign_cfg = V11CampaignConfig(
            campaign_id=args.campaign_id,
            question_type=args.question_type,
            library_root=library_root,
            run_root=args.run_root,
            max_revision_attempts=args.max_revision_attempts,
            method_context_max_chars=(
                base_cfg.retrieval_policy.method_context_max_chars),
            seed=args.seed,
            model_id=base_cfg.vllm_model,
            model_config_sha256=model_identity["sha256"],
            quality_contract_sha256=contract["sha256"],
            solver_config_sha256=solver_identity["sha256"],
            resume=True,
        )
        offline_client = DeepSeekClient(
            base_url=args.deepseek_base_url,
            model=args.deepseek_model,
            timeout_s=args.deepseek_timeout_s,
            max_retries=args.deepseek_max_retries,
            reasoning_effort=args.deepseek_reasoning_effort,
            thinking=args.deepseek_reasoning_effort != "none",
            prompt_version=REVISION_PROMPT_VERSION,
            default_max_tokens=args.deepseek_max_tokens,
        )
        collector = V11TraceCollector(args.parent_trace_dir, require_real=True)
        reviser = V11OfflineReviser(
            offline_client,
            audit_root=args.run_root,
            max_tokens=args.deepseek_max_tokens,
            require_health_check=True,
        )
        evaluator = V11PairedEvaluator(
            items=inner,
            artifact_paths=inner_artifacts,
            base_cfg=base_cfg,
            output_root=args.run_root,
            library_root=library_root,
            expected_qa_ids=expected_inner,
            quality_confirmation=quality_confirmation,
        )
        verifier = V11PostPublishVerifier(
            items=post,
            artifact_paths=post_artifacts,
            base_cfg=base_cfg,
            output_root=args.run_root,
            library_root=library_root,
        )
        checkpoint = V11CampaignRunner(
            campaign_cfg,
            collector=collector,
            reviser=reviser,
            evaluator=evaluator,
            post_publish_verifier=verifier,
        ).run()
        print(json.dumps(
            checkpoint.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ))
        return 0 if checkpoint.status in {"promoted", "rejected"} else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
