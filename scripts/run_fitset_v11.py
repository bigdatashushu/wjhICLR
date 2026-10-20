#!/usr/bin/env python3
"""Run and score the exploratory v11 same-question fit-set experiment."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill3d.adapters import vsibench_loader as vl
from skill3d.adapters.episode_source import (
    EpisodeItem,
    _episode_from_row,
    _load_from_sources,
)
from skill3d.evaluation.direct_qwen_v11 import (
    DIRECT_REQUEST_SCHEMA,
    DIRECT_RUN_SCHEMA,
    file_identity,
    load_safe_metadata,
)
from skill3d.evaluation.fitset_v11 import (
    PANEL_NAMES,
    FitSetError,
    SystemVariantRunner,
    audit_fitset_pilot,
    build_fitset_comparison,
    direct_rows_for_variants,
    load_fit_skill_bundle,
    load_fitset_panel,
    load_label_metadata,
    load_variant_bindings,
    migrate_empty_m4_bindings,
    select_panel_variants,
)
from skill3d.online.config import (
    active_vision_from,
    load_config,
    online_model_from,
    paths_from,
    resolve_runtime,
    retrieval_policy_from,
    sandbox_from,
)
from skill3d.routing.task_classifier import canonical_task
from skill3d.skills.registry import (
    active_snapshot_provenance,
    load_active_skills,
)


def _base_config(raw: dict):
    from skill3d.online.runner import OnlineRunConfig

    model = online_model_from(raw)
    sandbox = sandbox_from(raw)
    vision = active_vision_from(raw)
    return OnlineRunConfig(
        mode="real",
        baseline="C1_tools_program",
        seed=int(raw["seed"]),
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


class _FitSetItemProvider:
    """Decode selected scenes; labels remain framework-only for identity/M12."""

    def __init__(
        self,
        *,
        metadata_path: str | Path,
        direct_run_root: str | Path,
        video_root: str | Path,
        fallback_roots: list[str],
        max_scenes: int,
    ) -> None:
        if max_scenes <= 0:
            raise FitSetError("scene cache size must be positive")
        self.rows = load_safe_metadata(metadata_path)
        labels = load_label_metadata(metadata_path)
        if set(self.rows) != set(labels):
            raise FitSetError("safe metadata and label metadata identities differ")
        for qa_id, row in self.rows.items():
            label = labels[qa_id]
            if (
                row["scene_name"] != label["scene_id"]
                or canonical_task(row["question_type"])
                != label["question_type"]
            ):
                raise FitSetError(f"{qa_id}: metadata projections differ")
            row["ground_truth"] = label["reference_answer"]
        self.direct_root = Path(direct_run_root).resolve()
        direct_manifest_path = self.direct_root / "manifest.json"
        if not direct_manifest_path.is_file():
            raise FitSetError("direct run root lacks manifest.json")
        direct_manifest = json.loads(
            direct_manifest_path.read_text(encoding="utf-8"))
        if direct_manifest.get("schema_version") != DIRECT_RUN_SCHEMA:
            raise FitSetError("direct input manifest schema changed")
        metadata_identity = file_identity(metadata_path)
        if (
            (direct_manifest.get("metadata") or {}).get("sha256")
            != metadata_identity["sha256"]
        ):
            raise FitSetError("system/direct metadata file differs")
        declared_video_root = str(Path(video_root).resolve())
        declared_fallbacks = [
            str(Path(path).resolve()) for path in fallback_roots]
        if (
            declared_video_root != direct_manifest.get("video_root")
            or declared_fallbacks
            != list(direct_manifest.get("video_fallback_roots") or [])
        ):
            raise FitSetError("system/direct video source roots differ")
        if not set(self.rows).issuperset(
            direct_manifest.get("selected_qa_ids") or []
        ):
            raise FitSetError("direct manifest contains unknown metadata IDs")
        self.input_contract_identity = {
            "direct_run_root": str(self.direct_root),
            "direct_manifest": file_identity(direct_manifest_path),
            "metadata": metadata_identity,
            "video_root": declared_video_root,
            "video_fallback_roots": declared_fallbacks,
        }
        self.video_root = str(video_root)
        self.fallback_roots = list(fallback_roots)
        self.max_scenes = int(max_scenes)
        self.cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self.cache_lock = threading.RLock()

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["cache"] = OrderedDict()
        state.pop("cache_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.cache = OrderedDict()
        self.cache_lock = threading.RLock()

    def _scene(self, row: dict) -> dict:
        key = (row["dataset"], row["scene_name"])
        with self.cache_lock:
            if key in self.cache:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            loaded, attempts = _load_from_sources(
                row,
                self.video_root,
                self.fallback_roots,
                32,
            )
            if loaded is None:
                raise FitSetError(
                    f"{row['qa_id']}: frozen video input unavailable: {attempts}")
            if (
                len(loaded["planned"]) != 32
                or len(loaded["pixels"]) != 32
                or loaded["missing"]
            ):
                raise FitSetError(
                    f"{row['qa_id']}: fit-set requires exactly 32 readable frames")
            self.cache[key] = loaded
            while len(self.cache) > self.max_scenes:
                self.cache.popitem(last=False)
            return loaded

    def __call__(self, variant: dict) -> EpisodeItem:
        qa_id = str(variant["qa_id"])
        row = self.rows.get(qa_id)
        if row is None:
            raise FitSetError(f"{qa_id}: metadata row is missing")
        if (
            row["scene_name"] != variant["scene_id"]
            or canonical_task(row["question_type"])
            != variant["question_type"]
        ):
            raise FitSetError(f"{variant['variant_id']}: metadata identity changed")
        loaded = self._scene(row)
        source = loaded["source"]
        frame_set = vl.build_frame_set(
            int(loaded["total"]),
            n_frames=32,
            fps=float(loaded["fps"]),
            episode_id=qa_id,
            dataset_id=row["dataset"],
            video_id=vl.video_id_for(row, source),
            scene_name=row["scene_name"],
            readable_frame_ids=loaded["planned"],
        )
        episode = _episode_from_row(
            row,
            variant["split"],
            loaded["pixels"],
            32,
            frame_set=frame_set,
        )
        return EpisodeItem(
            episode=episode,
            pixels=loaded["pixels"],
            source="vsi_bench",
            video_path=str(source.path),
        )

    def receipt(self, variant: dict, item: EpisodeItem) -> dict:
        qa_id = str(variant["qa_id"])
        path = self.direct_root / "preflight" / f"{qa_id}.json"
        if not path.is_file():
            raise FitSetError(f"{qa_id}: direct preflight receipt is missing")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        frame_set = item.episode.frame_set
        if (
            receipt.get("schema_version") != DIRECT_REQUEST_SCHEMA
            or receipt.get("qa_id") != qa_id
            or receipt.get("scene_id") != variant["scene_id"]
            or receipt.get("question_type") != variant["question_type"]
            or variant["variant_id"]
            not in (receipt.get("system_variant_ids") or [])
            or receipt.get("n_images") != 32
            or receipt.get("frame_set_hash") != frame_set.frame_set_hash
            or receipt.get("source_frame_indices")
            != frame_set.source_frame_indices
            or receipt.get("pixel_sha256") != frame_set.pixel_sha256
            or receipt.get("reference_answer_visible") is not False
        ):
            raise FitSetError(
                f"{variant['variant_id']}: system/direct frozen input differs")
        return {
            "path": str(path),
            "file_sha256": file_identity(path)["sha256"],
            "request_sha256": receipt["request_sha256"],
            "messages_sha256": receipt["messages_sha256"],
            "frame_set_hash": receipt["frame_set_hash"],
        }


class _ActiveSkillProvider:
    def __init__(
        self,
        *,
        skills,
        snapshot_id: str,
        manifest_hash: str,
    ) -> None:
        self.skills = list(skills)
        self.snapshot_id = snapshot_id
        self.manifest_hash = manifest_hash

    def __call__(self, _variant):
        return self.skills, {
            "mode": "active_accumulation",
            "snapshot_ref": self.snapshot_id,
            "manifest_sha256": self.manifest_hash,
        }


class _FitSkillProvider:
    def __init__(self, bundle) -> None:
        self.bundle = bundle

    def __call__(self, variant):
        slot = self.bundle.skill_for(
            variant["question_type"],
            variant["evidence_state"],
        )
        spec = slot.as_spec()
        return [spec], {
            "mode": "fit_exact_slot_skill",
            "bundle_id": self.bundle.bundle_id,
            "snapshot_ref": self.bundle.bundle_id,
            "manifest_sha256": self.bundle.manifest_sha256,
            "slot_key": slot.slot_key,
            "slot_role": slot.role,
            "skill_version": f"{spec.skill_id}@{spec.version}",
            "content_sha256": spec.content_sha256,
            "evidence_state": spec.evidence_state,
        }


@dataclass(frozen=True)
class _RevisionClientFactory:
    config: dict
    max_tokens: int

    def __call__(self):
        from skill3d.synthesis.bailian_client import BailianClient

        model = online_model_from(self.config)
        return BailianClient(
            model=model.model, model_revision=model.model_revision,
            base_url=model.base_url, api_key_env=model.api_key_env,
            timeout_s=model.timeout_s, temperature=0.0,
            max_output_tokens=self.max_tokens,
        )


@dataclass(frozen=True)
class _GateClientFactory:
    config: dict

    def __call__(self, seed: int, output: Path):
        return _SeededClientFactory(self.config, seed, output)


@dataclass(frozen=True)
class _SeededClientFactory:
    config: dict
    seed: int
    output: Path

    def __call__(self, variant_id: str):
        from skill3d.evaluation.fitset_v11 import _variant_key
        from skill3d.evolution.fitset_stream_v11 import SeededGateClient

        client = _RevisionClientFactory(
            self.config, online_model_from(self.config).max_tokens)()
        return SeededGateClient(
            client, self.seed,
            self.output / "request_seeds" / (_variant_key(variant_id) + ".json"),
        )


def _active_skill_provider(library_root: Path):
    store = library_root / "snapshots"
    skills, warnings, snapshot_id = load_active_skills(store)
    active_id, manifest_hash = active_snapshot_provenance(store)
    if warnings or active_id != snapshot_id or not manifest_hash:
        raise FitSetError(f"active Skill snapshot is not trustworthy: {warnings}")
    return _ActiveSkillProvider(
        skills=skills,
        snapshot_id=snapshot_id,
        manifest_hash=manifest_hash,
    )


def _fit_skill_provider(bundle_path: Path):
    bundle = load_fit_skill_bundle(bundle_path)
    return _FitSkillProvider(bundle)


def _run_system(args) -> int:
    from skill3d.evolution.fitset_stream_v11 import FitSetEvolutionConfig

    panel = load_fitset_panel(args.panel)
    variants = select_panel_variants(panel, args.panel_name)
    bindings = load_variant_bindings(args.bindings, variants)
    raw = resolve_runtime(load_config(args.config), seed=args.seed)
    model = online_model_from(raw)
    if (
        not model.enabled
        or model.provider != "bailian"
        or model.model != "qwen3.8-flash"
    ):
        raise FitSetError(
            "fit-set system execution requires enabled bailian/qwen3.8-flash")
    if (
        not args.preflight_only
        and not os.environ.get(model.api_key_env, "").strip()
    ):
        raise FitSetError(f"{model.api_key_env} is not configured")
    detector = raw.get("detection", {}).get("endpoint")
    if detector:
        os.environ.setdefault("SKILL3D_DETECTOR_ENDPOINT", str(detector))
    paths = paths_from(raw)
    fallback_roots = (
        list(args.video_fallback_root)
        if args.video_fallback_root
        else list(paths.raw_video_fallbacks)
    )
    item_provider = _FitSetItemProvider(
        metadata_path=args.metadata,
        direct_run_root=args.direct_run_root,
        video_root=args.video_root or paths.raw_videos,
        fallback_roots=fallback_roots,
        max_scenes=args.scene_cache_size,
    )
    if args.phase == "accumulation":
        if args.skill_bundle:
            raise FitSetError("--skill-bundle is only valid for final_system")
        library_root = Path(
            args.library_root or paths.skill_library).resolve()
        skill_provider = _active_skill_provider(library_root)
    else:
        if not args.skill_bundle:
            raise FitSetError("final_system requires --skill-bundle")
        skill_provider = _fit_skill_provider(Path(args.skill_bundle).resolve())
    selected_ids = [
        value for value in args.variant_ids.split(",") if value
    ]
    runner = SystemVariantRunner(
        output_root=args.output_dir,
        panel=panel,
        panel_name=args.panel_name,
        variants=variants,
        item_provider=item_provider,
        input_receipt_provider=item_provider.receipt,
        input_contract_identity={
            **item_provider.input_contract_identity,
            "prepared_bindings_manifest": file_identity(args.bindings),
        },
        binding_refs=bindings,
        base_cfg=_base_config(raw),
        phase=args.phase,
        skills_for_variant=skill_provider,
        min_free_gib=args.min_free_gib,
        workers=args.workers,
        evolution_config=(
            FitSetEvolutionConfig(
                gate_size=args.gate_size,
                gate_seeds=(args.seed, args.seed + 1),
                max_revision_attempts=args.max_revision_attempts,
                max_tokens=args.revision_max_tokens,
            ) if args.phase == "accumulation" else None
        ),
        label_metadata=(
            load_label_metadata(args.metadata)
            if args.phase == "accumulation" else None
        ),
        revision_client_factory=_RevisionClientFactory(raw, args.revision_max_tokens),
        gate_client_factory=_GateClientFactory(raw),
    )
    action = runner.preflight if args.preflight_only else runner.run
    summary = action(
        resume=args.resume,
        max_variants=args.max_variants,
        variant_ids=selected_ids,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _validate(args) -> int:
    panel = load_fitset_panel(args.panel)
    variants = select_panel_variants(panel, args.panel_name)
    bindings = load_variant_bindings(args.bindings, variants)
    direct = direct_rows_for_variants(panel, variants)
    value = {
        "status": "validated",
        "panel_name": args.panel_name,
        "system_variants": len(variants),
        "direct_qa_ids": len(direct),
        "binding_references": len(bindings),
        "evidence_states": {
            state: sum(row["evidence_state"] == state for row in variants)
            for state in ("g0m0", "g1m0", "g1m1")
        },
    }
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


def _migrate_bindings(args) -> int:
    panel = load_fitset_panel(args.panel)
    variants = select_panel_variants(panel, args.panel_name)
    manifest = migrate_empty_m4_bindings(
        panel=panel,
        variants=variants,
        source_manifest_path=args.bindings,
        output_dir=args.output_dir,
        min_free_gib=args.min_free_gib,
    )
    print(json.dumps({
        "status": "completed",
        "schema_version": manifest["schema_version"],
        "n_variants": manifest["n_variants"],
        "n_migrated": manifest["n_migrated"],
        "n_reused": manifest["n_reused"],
        "n_migrated_scenes": manifest["n_migrated_scenes"],
        "identity_sha256": manifest["identity_sha256"],
        "prepared_bindings": str(
            Path(args.output_dir).resolve() / "prepared_bindings.json"),
    }, ensure_ascii=False, sort_keys=True))
    return 0


def _score(args) -> int:
    panel = load_fitset_panel(args.panel)
    report = build_fitset_comparison(
        panel=panel,
        panel_name=args.panel_name,
        system_root=args.system_root,
        direct_root=args.direct_root,
        label_metadata=load_label_metadata(args.metadata),
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def _audit_pilot(args) -> int:
    report = audit_fitset_pilot(
        output_root=args.system_root,
        expected_variants=args.expected_variants,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["expand_to_full"] else 2


def _upgrade_panel(args) -> int:
    from skill3d.evaluation.fitset_panel_v11 import upgrade_fitset_panel

    panel = upgrade_fitset_panel(args.source, args.output)
    print(json.dumps({
        "schema_version": panel["schema_version"],
        "proposal_sha256": panel["proposal_sha256"],
        "panel_slot_counts": {
            name: row["skill_slot_count"] for name, row in panel["panels"].items()
        },
    }, ensure_ascii=False))
    return 0


def _freeze(args) -> int:
    from skill3d.evolution.fitset_stream_v11 import freeze_fitset_stream

    summary = freeze_fitset_stream(args.accumulation_root, args.output_dir)
    print(json.dumps({
        "bundle_id": summary["bundle_id"],
        "revised_slot_count": summary["revised_slot_count"],
        "manifest_sha256": summary["manifest_sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate panel and prepared binding coverage")
    validate.add_argument("--panel", required=True)
    validate.add_argument("--panel-name", choices=(
        PANEL_NAMES), required=True)
    validate.add_argument("--bindings", required=True)
    validate.set_defaults(handler=_validate)

    migrate = subparsers.add_parser(
        "migrate-bindings",
        help="republish reviewed empty M4 bindings under current source identity",
    )
    migrate.add_argument("--panel", required=True)
    migrate.add_argument("--panel-name", choices=(
        PANEL_NAMES), required=True)
    migrate.add_argument("--bindings", required=True)
    migrate.add_argument("--output-dir", required=True)
    migrate.add_argument("--min-free-gib", type=float, default=80.0)
    migrate.set_defaults(handler=_migrate_bindings)

    run = subparsers.add_parser(
        "run-system", help="run accumulation or final Harness3D variants")
    run.add_argument("--config", default="configs/config.yaml")
    run.add_argument("--seed", type=int, default=137)
    run.add_argument("--panel", required=True)
    run.add_argument("--panel-name", choices=(
        PANEL_NAMES), required=True)
    run.add_argument("--bindings", required=True)
    run.add_argument("--metadata", required=True)
    run.add_argument("--direct-run-root", required=True)
    run.add_argument("--video-root")
    run.add_argument("--video-fallback-root", action="append", default=[])
    run.add_argument("--scene-cache-size", type=int, default=2)
    run.add_argument("--min-free-gib", type=float, default=80.0)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--gate-size", type=int, default=8)
    run.add_argument("--max-revision-attempts", type=int, default=3)
    run.add_argument("--revision-max-tokens", type=int, default=8192)
    run.add_argument("--library-root", default="")
    run.add_argument(
        "--phase", choices=("accumulation", "final_system"), required=True)
    run.add_argument("--skill-bundle", default="")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--max-variants", type=int, default=0)
    run.add_argument("--variant-ids", default="")
    run.add_argument("--preflight-only", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.set_defaults(handler=_run_system)

    freeze = subparsers.add_parser(
        "freeze", help="freeze each slot's last successfully published version")
    freeze.add_argument("--accumulation-root", required=True)
    freeze.add_argument("--output-dir", required=True)
    freeze.set_defaults(handler=_freeze)

    score = subparsers.add_parser(
        "score", help="join and rescore final system/direct results")
    score.add_argument("--panel", required=True)
    score.add_argument("--panel-name", choices=(
        PANEL_NAMES), required=True)
    score.add_argument("--metadata", required=True)
    score.add_argument("--system-root", required=True)
    score.add_argument("--direct-root", required=True)
    score.add_argument("--output-dir", required=True)
    score.set_defaults(handler=_score)

    audit = subparsers.add_parser(
        "audit-pilot",
        help="reject a pilot with sandbox, fallback, service, or count failures",
    )
    audit.add_argument("--system-root", required=True)
    audit.add_argument("--expected-variants", type=int, required=True)
    audit.set_defaults(handler=_audit_pilot)
    upgrade = subparsers.add_parser(
        "upgrade-panel", help="create a new reachable-slot v2 panel from v1")
    upgrade.add_argument("--source", required=True)
    upgrade.add_argument("--output", required=True)
    upgrade.set_defaults(handler=_upgrade_panel)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FitSetError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
