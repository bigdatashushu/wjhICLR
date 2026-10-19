#!/usr/bin/env python3
"""Run production v11 evolution as a continuous per-episode stream."""

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
)
from skill3d.evolution.prepared_bindings_v11 import (
    prepare_shared_bindings_v11,
)
from skill3d.evolution.live_v11 import (
    REACHABLE_V11_SLOTS,
    V11LiveEvolutionScheduler,
    V11LiveSchedulerConfig,
    V11ProductionAttempt,
    partition_items_by_slot,
    prepare_slot_panels,
    prepare_validation_panels,
    select_scene_round_robin,
    slot_key,
    validation_panel_manifest,
)
from skill3d.evolution.outer_v11 import V11OuterHoldoutEvaluator
from skill3d.governance.deepseek_client import (
    BASE_URL,
    MODEL_ID,
    DeepSeekClient,
)
from skill3d.online.config import (
    active_vision_from,
    config_receipt,
    load_config,
    load_yaml,
    online_model_from,
    resolve_runtime,
    retrieval_policy_from,
    sandbox_from,
)
from skill3d.online.runner import OnlineRunConfig
from skill3d.reconstruction.run import resolve_artifact_path
from skill3d.routing.task_classifier import canonical_task
from skill3d.skills.registry import active_snapshot_provenance


class V11StreamingCliError(ValueError):
    """CLI inputs cannot define a complete streaming experiment."""


def _read_json(path: str | Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    data = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if path.is_file():
        if path.read_text(encoding="utf-8") != data:
            raise V11StreamingCliError(
                f"frozen streaming input changed during resume: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)


def _csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _target_slots(args) -> list[tuple[str, str]]:
    if args.slot_manifest:
        value = _read_json(args.slot_manifest)
        rows = value.get("slots") if isinstance(value, dict) else value
        if not isinstance(rows, list):
            raise V11StreamingCliError(
                "slot manifest must be a list or an object with slots"
            )
        targets = []
        for row in rows:
            if isinstance(row, str) and "/" in row:
                question_type, evidence_state = row.split("/", 1)
            elif isinstance(row, dict):
                question_type = row.get("question_type")
                evidence_state = row.get("evidence_state")
            else:
                raise V11StreamingCliError(
                    f"invalid slot manifest row: {row!r}"
                )
            targets.append((
                canonical_task(str(question_type)),
                str(evidence_state),
            ))
    else:
        question_types = tuple(
            canonical_task(value)
            for value in _csv(args.question_types)
        )
        evidence_states = tuple(_csv(args.evidence_states))
        if not question_types and not evidence_states:
            targets = list(REACHABLE_V11_SLOTS)
        elif not question_types or not evidence_states:
            raise V11StreamingCliError(
                "question-types and evidence-states must be supplied together"
            )
        else:
            targets = [
                (question_type, evidence_state)
                for question_type in question_types
                for evidence_state in evidence_states
            ]
    if len(set(targets)) != len(targets):
        raise V11StreamingCliError("target slots contain duplicates")
    invalid = sorted(set(targets) - set(REACHABLE_V11_SLOTS))
    if invalid:
        raise V11StreamingCliError(
            f"target slots are unreachable under current evidence policy: {invalid}"
        )
    return targets


def _load_items(
    args,
    raw: dict,
    *,
    split: str,
    per_task_limit: int,
) -> list[EpisodeItem]:
    paths = raw.get("paths", {})
    question_types = list(args.question_types)
    if args.source == "jsonl":
        if not args.episodes_jsonl:
            raise V11StreamingCliError(
                "--episodes-jsonl is required for --source=jsonl"
            )
        items = load_jsonl_items(
            args.episodes_jsonl,
            split,
            limit=per_task_limit * len(question_types),
            include_input_errors=True,
            split_cfg=load_yaml(raw["split_config"]),
            n_frames=raw["frame_sampling"]["n_frames"],
        )
    else:
        split_cfg = load_yaml(raw["split_config"])
        items = load_vsi_bench_items(
            split,
            split_cfg,
            video_root=args.video_root
            or paths.get("raw_videos", "data/raw_videos"),
            video_fallback_roots=paths.get("raw_video_fallbacks"),
            cache_dir=paths.get("vsi_bench_meta"),
            local_meta_path=paths["vsi_bench_meta"],
            question_types=question_types,
            datasets=_csv(args.datasets) or None,
            n_frames=int(raw.get("frame_sampling", {}).get("n_frames", 32)),
            seed=args.seed,
            sampling_seed=args.seed,
            stratified_per_task=per_task_limit,
            include_input_errors=True,
        )
    wanted = set(question_types)
    return [
        item
        for item in items
        if canonical_task(item.episode.question_type) in wanted
    ]


def _artifact_mapping(
    items: list[EpisodeItem],
    *,
    explicit_json: str,
    recon_dir: str,
) -> dict[str, str]:
    if explicit_json:
        value = _read_json(explicit_json)
        if not isinstance(value, dict):
            raise V11StreamingCliError(
                f"artifact mapping must be an object: {explicit_json}"
            )
        mapping = {
            str(key): str(Path(path).expanduser().resolve())
            for key, path in value.items()
        }
        missing_files = sorted(
            qa_id for qa_id, path in mapping.items()
            if not Path(path).is_file()
        )
        if missing_files:
            raise V11StreamingCliError(
                f"explicit artifact mappings do not exist: {missing_files[:10]}"
            )
    else:
        mapping = {
            item.episode.qa_id: str(
                resolve_artifact_path(
                    recon_dir,
                    item.episode.scene_name,
                    "vggt",
                    frame_set=item.episode.frame_set,
                )[0]
            )
            for item in items
            if item.input_error is None
        }
        mapping = {
            qa_id: path
            for qa_id, path in mapping.items()
            if Path(path).is_file()
        }
    return mapping


def _base_config(args, raw: dict) -> OnlineRunConfig:
    model = online_model_from(raw)
    sandbox = sandbox_from(raw)
    vision = active_vision_from(raw)
    return OnlineRunConfig(
        mode="real",
        baseline="C1_tools_program",
        seed=args.seed,
        skills=[],
        online_model_enabled=model.enabled,
        online_model_provider=model.provider,
        online_model=model.model,
        online_model_revision=model.model_revision,
        online_base_url=model.base_url,
        online_api_key_env=model.api_key_env,
        online_timeout_s=model.timeout_s,
        semantic_min_pixels=model.audit_min_pixels,
        semantic_max_pixels=model.audit_max_pixels,
        semantic_relation_batch_size=model.relation_batch_size,
        semantic_review_config=model.mask_review,
        max_images=model.n_frames,
        max_pixels=model.max_pixels,
        max_model_len=model.max_model_len,
        max_tokens=model.max_tokens,
        temperature=model.temperature,
        cell_timeout_s=int(sandbox.cell_timeout_s),
        max_regen=int(sandbox.max_regenerate),
        max_solver_rounds=int(raw["max_solver_rounds"]),
        max_retries_per_operation=int(raw["max_retries_per_operation"]),
        finalization_rounds=int(raw["finalization_rounds"]),
        eval_visual_fallback=bool(raw["eval_visual_fallback"]),
        image_layout=vision.layout,
        max_derived_images=int(vision.max_derived_images),
        retrieval_policy=retrieval_policy_from(raw),
    )


def _quality_confirmation(path: str) -> dict:
    value = _read_json(path)
    contract = quality_contract()
    if not (
        isinstance(value, dict)
        and value.get("confirmed") is True
        and value.get("sha256") == contract["sha256"]
        and value.get("evidence_ref")
    ):
        raise V11StreamingCliError(
            "quality confirmation must match the current contract"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run isolated induction episodes, buffer exact v11 slots, and "
            "trigger a strict campaign every 12 eligible cases"
        )
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stream-id", required=True)
    parser.add_argument(
        "--question-types",
        default="",
        help="comma-separated canonical tasks; empty with states selects 19 slots",
    )
    parser.add_argument(
        "--evidence-states",
        default="",
        help="comma-separated states; must be paired with --question-types",
    )
    parser.add_argument(
        "--slot-manifest",
        default="",
        help="explicit JSON slot list; overrides question/state arguments",
    )
    parser.add_argument(
        "--source",
        choices=["jsonl", "vsi_bench"],
        default="vsi_bench",
    )
    parser.add_argument("--episodes-jsonl")
    parser.add_argument("--video-root")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--induction-limit-per-slot", type=int, default=12)
    parser.add_argument("--induction-candidates-per-task", type=int, default=64)
    parser.add_argument("--inner-candidates-per-task", type=int, default=32)
    parser.add_argument("--outer-candidates-per-task", type=int, default=32)
    parser.add_argument("--post-limit", type=int, default=4)
    parser.add_argument("--min-evaluation-pairs", type=int, default=8)
    parser.add_argument("--outer-limit-per-slot", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--inner-artifacts-json", default="")
    parser.add_argument("--outer-artifacts-json", default="")
    parser.add_argument("--induction-artifacts-json", default="")
    parser.add_argument("--recon-dir", default="")
    parser.add_argument(
        "--quality-confirmation",
        default="",
        help="required for campaign execution; not read by --prepare-only",
    )
    parser.add_argument("--library-root", default="")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--max-revision-attempts", type=int, default=3)
    parser.add_argument("--binding-store", default="")
    parser.add_argument("--prepared-bindings-json", default="")
    parser.add_argument(
        "--prepare-bindings",
        action="store_true",
        help="precompute audited bindings for all frozen inputs",
    )
    parser.add_argument(
        "--require-prepared-bindings",
        action="store_true",
        help="fail execution if any selected question lacks a binding",
    )
    parser.add_argument(
        "--run-outer",
        action="store_true",
        help="after streaming freezes, compare S0 with final active on outer",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze exact-slot stream/post inputs and validation panels only",
    )
    parser.add_argument("--retry-deferred", action="store_true")
    parser.add_argument(
        "--flush-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
        raw = resolve_runtime(load_config(args.config), seed=args.seed)
        args.seed = int(raw["seed"])
        targets = _target_slots(args)
        args.question_types = tuple(dict.fromkeys(
            question_type for question_type, _ in targets
        ))
        for name in (
            "induction_limit_per_slot",
            "induction_candidates_per_task",
            "inner_candidates_per_task",
            "outer_candidates_per_task",
            "post_limit",
            "min_evaluation_pairs",
            "outer_limit_per_slot",
            "batch_size",
        ):
            if int(getattr(args, name)) <= 0:
                raise V11StreamingCliError(f"{name} must be positive")
        if args.run_outer and set(targets) != set(REACHABLE_V11_SLOTS):
            raise V11StreamingCliError(
                "--run-outer requires the complete 19-slot manifest"
            )
        paths = raw.get("paths", {})
        library_root = Path(
            args.library_root
            or paths.get("skill_library", "skill_library")
        ).resolve()
        run_root = Path(args.run_root).resolve()
        recon_dir = args.recon_dir or paths.get(
            "reconstructions", "data/reconstructions"
        )
        from skill3d.env_preflight import assert_runtime_dependencies

        assert_runtime_dependencies(context="streaming_evolution_v11")
        induction_candidates = _load_items(
            args,
            raw,
            split="induction",
            per_task_limit=args.induction_candidates_per_task,
        )
        inner_candidates = _load_items(
            args,
            raw,
            split="inner_validation",
            per_task_limit=args.inner_candidates_per_task,
        )
        outer_candidates = _load_items(
            args,
            raw,
            split="outer_holdout",
            per_task_limit=args.outer_candidates_per_task,
        )
        induction_artifacts = _artifact_mapping(
            induction_candidates,
            explicit_json=args.induction_artifacts_json,
            recon_dir=recon_dir,
        )
        inner_artifacts = _artifact_mapping(
            inner_candidates,
            explicit_json=args.inner_artifacts_json,
            recon_dir=recon_dir,
        )
        outer_artifacts = _artifact_mapping(
            outer_candidates,
            explicit_json=args.outer_artifacts_json,
            recon_dir=recon_dir,
        )
        grouped_induction = partition_items_by_slot(
            induction_candidates,
            induction_artifacts,
            target_slots=targets,
        )
        selected_stream_items: list[EpisodeItem] = []
        selected_post_items: list[EpisodeItem] = []
        shortages: dict[str, dict[str, int]] = {}
        for question_type, evidence_state in targets:
            key = slot_key(question_type, evidence_state)
            candidates = grouped_induction.get(key, [])
            selected_stream = select_scene_round_robin(
                candidates,
                limit=int(args.induction_limit_per_slot),
                max_per_scene=1,
            )
            stream_scenes = {
                str(item.episode.scene_name)
                for item in selected_stream
            }
            post_candidates = [
                item
                for item in candidates
                if item.episode.qa_id
                not in {
                    selected.episode.qa_id
                    for selected in selected_stream
                }
                and str(item.episode.scene_name) not in stream_scenes
            ]
            selected_post = select_scene_round_robin(
                post_candidates,
                limit=int(args.post_limit),
                max_per_scene=1,
            )
            if (
                len(selected_stream) < int(args.induction_limit_per_slot)
                or len(selected_post) < int(args.post_limit)
            ):
                shortages[key] = {
                    "candidate_rows": len(candidates),
                    "candidate_scenes": len({
                        str(item.episode.scene_name)
                        for item in candidates
                    }),
                    "stream_rows": len(selected_stream),
                    "post_rows": len(selected_post),
                }
                continue
            selected_stream_items.extend(selected_stream)
            selected_post_items.extend(selected_post)
        if shortages:
            raise V11StreamingCliError(
                "induction exact-slot pools cannot cover scene-isolated "
                f"stream/post quotas: {shortages}"
            )
        stream_ids = {
            item.episode.qa_id for item in selected_stream_items
        }
        post_ids = {
            item.episode.qa_id for item in selected_post_items
        }
        stream_items = [
            item
            for item in induction_candidates
            if item.episode.qa_id in stream_ids
        ]
        post_items = [
            item
            for item in induction_candidates
            if item.episode.qa_id in post_ids
        ]
        induction_scenes = {
            item.episode.scene_name
            for item in stream_items + post_items
        }
        panels = prepare_validation_panels(
            inner_candidates,
            inner_artifacts,
            required_slots=targets,
            min_pairs=args.min_evaluation_pairs,
            excluded_scene_ids=sorted(induction_scenes),
        )
        inner_scenes = {
            scene_id
            for panel in panels.values()
            for scene_id in panel.scene_ids
        }
        outer_panels = prepare_slot_panels(
            outer_candidates,
            outer_artifacts,
            required_slots=targets,
            source_split="outer_holdout",
            min_pairs=args.outer_limit_per_slot,
            excluded_scene_ids=sorted(
                induction_scenes | inner_scenes
            ),
        )
        grouped_post = partition_items_by_slot(
            post_items,
            induction_artifacts,
            target_slots=targets,
        )
        post_artifacts = {
            qa_id: path
            for qa_id, path in induction_artifacts.items()
            if qa_id in post_ids
        }
        run_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_root / "validation_panels.json",
            validation_panel_manifest(panels),
        )
        _write_json(
            run_root / "outer_panels.json",
            validation_panel_manifest(outer_panels),
        )
        _write_json(
            run_root / "slot_manifest.json",
            {
                "schema_version": "v11-reachable-slots/1.0",
                "n_slots": len(targets),
                "slots": [
                    {
                        "question_type": question_type,
                        "evidence_state": evidence_state,
                        "slot_key": slot_key(
                            question_type,
                            evidence_state,
                        ),
                    }
                    for question_type, evidence_state in targets
                ],
            },
        )
        _write_json(
            run_root / "stream_inputs.json",
            {
                "schema_version": "v11-live-stream-inputs/1.1",
                "stream_id": args.stream_id,
                "target_slots": [
                    slot_key(*target) for target in targets
                ],
                "selection_contract": {
                    "algorithm": "stable-scene-round-robin",
                    "stream_per_slot": args.induction_limit_per_slot,
                    "post_per_slot": args.post_limit,
                    "inner_per_slot": args.min_evaluation_pairs,
                    "outer_per_slot": args.outer_limit_per_slot,
                    "max_same_slot_questions_per_scene": 1,
                    "post_excludes_stream_scenes": True,
                },
                "stream_qa_ids": [
                    item.episode.qa_id for item in stream_items
                ],
                "stream_scene_ids": [
                    item.episode.scene_name for item in stream_items
                ],
                "post_qa_ids": [
                    item.episode.qa_id for item in post_items
                ],
                "post_scene_ids": [
                    item.episode.scene_name for item in post_items
                ],
            },
        )
        _write_json(
            run_root / "runtime_configuration.json",
            config_receipt(raw),
        )
        detector = raw.get("detection", {}).get("endpoint")
        if detector:
            os.environ.setdefault(
                "SKILL3D_DETECTOR_ENDPOINT",
                str(detector),
            )
        base_cfg = _base_config(args, raw)
        selected_items = [
            *stream_items,
            *post_items,
            *(
                item
                for panel in panels.values()
                for item in panel.items
            ),
            *(
                item
                for panel in outer_panels.values()
                for item in panel.items
            ),
        ]
        selected_by_id = {
            item.episode.qa_id: item for item in selected_items
        }
        if len(selected_by_id) != len(selected_items):
            raise V11StreamingCliError(
                "frozen stream/post/inner/outer inputs reuse a qa_id"
            )
        selected_scene_counts: dict[tuple[str, str], int] = {}
        for item in selected_items:
            scene_key = (
                item.episode.split,
                item.episode.scene_name,
            )
            selected_scene_counts[scene_key] = (
                selected_scene_counts.get(scene_key, 0) + 1
            )
        oversized_scenes = {
            f"{split}/{scene_id}": count
            for (split, scene_id), count in selected_scene_counts.items()
            if count > 8
        }
        if oversized_scenes:
            raise V11StreamingCliError(
                "selected inputs exceed eight questions per scene: "
                f"{oversized_scenes}"
            )
        selected_artifacts = {
            **{
                qa_id: induction_artifacts[qa_id]
                for qa_id in stream_ids | post_ids
            },
            **{
                qa_id: path
                for panel in panels.values()
                for qa_id, path in panel.artifact_paths.items()
            },
            **{
                qa_id: path
                for panel in outer_panels.values()
                for qa_id, path in panel.artifact_paths.items()
            },
        }
        prepared_manifest = None
        if args.prepared_bindings_json:
            value = _read_json(args.prepared_bindings_json)
            if not isinstance(value, dict):
                raise V11StreamingCliError(
                    "prepared bindings manifest must be an object"
                )
            prepared_manifest = value
        if args.prepare_bindings:
            if prepared_manifest is not None:
                raise V11StreamingCliError(
                    "--prepare-bindings cannot be combined with "
                    "--prepared-bindings-json"
                )
            model = online_model_from(raw)
            if model.enabled and not os.environ.get(
                model.api_key_env,
                "",
            ).strip():
                raise V11StreamingCliError(
                    f"{model.api_key_env} is not configured"
                )
            binding_store = Path(
                args.binding_store
                or run_root / "prepared_binding_store"
            ).resolve()
            prepared_manifest = prepare_shared_bindings_v11(
                list(selected_by_id.values()),
                artifact_paths=selected_artifacts,
                base_cfg=base_cfg,
                store_root=binding_store,
                max_questions_per_scene=8,
            )
            _write_json(
                run_root / "prepared_bindings.json",
                prepared_manifest,
            )
        prepared_refs = (
            prepared_manifest.get("bindings")
            if prepared_manifest is not None
            else None
        )
        prepared_store = (
            prepared_manifest.get("store_root")
            if prepared_manifest is not None
            else None
        )
        if prepared_manifest is not None and (
            not isinstance(prepared_refs, dict)
            or set(prepared_refs) != set(selected_by_id)
            or not prepared_store
        ):
            raise V11StreamingCliError(
                "prepared bindings must exactly cover all frozen inputs"
            )
        if args.require_prepared_bindings and prepared_manifest is None:
            raise V11StreamingCliError(
                "--require-prepared-bindings needs a prepared manifest"
            )
        if args.prepare_only:
            print(json.dumps({
                "status": "prepared",
                "stream_id": args.stream_id,
                "target_slots": [
                    slot_key(*target) for target in targets
                ],
                "stream_episodes": len(stream_items),
                "post_publish_episodes": len(post_items),
                "validation_pairs": {
                    key: len(panel.items)
                    for key, panel in sorted(panels.items())
                },
                "outer_pairs": {
                    key: len(panel.items)
                    for key, panel in sorted(outer_panels.items())
                },
                "prepared_bindings": (
                    len(prepared_refs) if prepared_refs is not None else 0
                ),
                "run_root": str(run_root),
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if not args.quality_confirmation:
            raise V11StreamingCliError(
                "--quality-confirmation is required unless --prepare-only is used"
            )
        quality_confirmation = _quality_confirmation(
            args.quality_confirmation
        )
        model = online_model_from(raw)
        if model.enabled and not os.environ.get(
            model.api_key_env, ""
        ).strip():
            raise V11StreamingCliError(
                f"{model.api_key_env} is not configured"
            )
        if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
            raise V11StreamingCliError(
                "DEEPSEEK_API_KEY is not configured"
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
        reviser = V11OfflineReviser(
            offline_client,
            audit_root=run_root / "campaigns",
            max_tokens=args.deepseek_max_tokens,
            require_health_check=True,
        )
        production_attempt = V11ProductionAttempt(
            library_root=library_root,
            campaign_root=run_root / "campaigns",
            base_cfg=base_cfg,
            validation_panels=panels,
            post_items_by_slot=grouped_post,
            post_artifact_paths=post_artifacts,
            reviser=reviser,
            quality_confirmation=quality_confirmation,
            post_limit=args.post_limit,
            min_evaluation_pairs=args.min_evaluation_pairs,
            max_revision_attempts=args.max_revision_attempts,
            prepared_binding_store=prepared_store,
            prepared_binding_refs=prepared_refs,
        )
        summary = V11LiveEvolutionScheduler(
            V11LiveSchedulerConfig(
                stream_id=args.stream_id,
                root=run_root / "stream",
                library_root=library_root,
                batch_size=args.batch_size,
                flush_tail=args.flush_tail,
                retry_deferred=args.retry_deferred,
                require_prepared_bindings=args.require_prepared_bindings,
            ),
            items=stream_items,
            artifact_paths=induction_artifacts,
            base_cfg=base_cfg,
            attempt=production_attempt,
            prepared_binding_refs=prepared_refs,
        ).run()
        if args.run_outer:
            final_snapshot_id, _ = active_snapshot_provenance(
                library_root / "snapshots"
            )
            outer_report = V11OuterHoldoutEvaluator(
                panels=outer_panels,
                base_cfg=base_cfg,
                output_root=run_root / "outer",
                library_root=library_root,
                final_snapshot_id=final_snapshot_id,
                quality_confirmation=quality_confirmation,
                prepared_binding_store=prepared_store,
                prepared_binding_refs=prepared_refs,
            ).run()
            summary["outer_report"] = str(
                run_root / "outer" / "outer_report.json"
            )
            summary["outer_status_by_slot"] = {
                key: row["status"]
                for key, row in outer_report["slots"].items()
            }
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
