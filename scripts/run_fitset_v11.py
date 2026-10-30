#!/usr/bin/env python3
"""Run v11 fit-set learning with a frozen gate (scene-disjoint by default)."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
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
from skill3d.evolution.fitset_revision_v11 import RevisionOperationalError
from skill3d.evaluation.qa_partition_v11 import (
    load_qa_partition_manifest,
    partition_qa_variants,
    select_qa_execution_variants,
)
from skill3d.schemas.data_access import normalize_experiment_split
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
    perception_endpoint: str = ""

    def __call__(self):
        from skill3d.synthesis.bailian_client import BailianClient

        model = online_model_from(self.config)
        client_type = BailianClient
        options = {}
        if self.perception_endpoint:
            from skill3d.cluster.perception_client import RemoteBailianClient
            client_type = RemoteBailianClient
            options["endpoint"] = self.perception_endpoint
        return client_type(
            model=model.model, model_revision=model.model_revision,
            base_url=model.base_url, api_key_env=model.api_key_env,
            timeout_s=model.timeout_s, temperature=0.0,
            # RPC identity pins the service limit; chat carries its own budget.
            max_output_tokens=(
                model.max_tokens if self.perception_endpoint else self.max_tokens),
            **options,
        )


@dataclass(frozen=True)
class _GateClientFactory:
    config: dict
    perception_endpoint: str = ""
    framework_state_io: object = None

    def __call__(self, seed: int, output: Path):
        return _SeededClientFactory(self.config, seed, output, self.perception_endpoint,
                                    self.framework_state_io)


@dataclass(frozen=True)
class _SeededClientFactory:
    config: dict
    seed: int
    output: Path
    perception_endpoint: str = ""
    framework_state_io: object = None
    perception_capacity: int = 0
    deferred_admission: bool = False

    def __call__(self, variant_id: str):
        from skill3d.evaluation.fitset_v11 import _variant_key
        from skill3d.evolution.fitset_stream_v11 import SeededGateClient

        client = _RevisionClientFactory(
            self.config, online_model_from(self.config).max_tokens,
            self.perception_endpoint)()
        if self.deferred_admission:
            _admit_solver_client(client, self.perception_endpoint, self.perception_capacity)
        return SeededGateClient(
            client, self.seed,
            self.output / "request_seeds" / (_variant_key(variant_id) + ".json"),
            framework_state_io=self.framework_state_io,
        )

    def resume_after_cached_prefix(self, variant_id: str, *, calls):
        from skill3d.evaluation.cached_episode_recovery_v11 import (
            RecoveryStop, resumed_seeded_gate_client,
        )
        from skill3d.evaluation.fitset_v11 import _variant_key

        audit_path = self.output / "request_seeds" / (_variant_key(variant_id) + ".json")
        index = len(calls) + 1
        if (not calls or any(call.seed_record["request_seed"] != self.seed for call in calls)
                or any(path.exists() or path.is_symlink() for path in (
                    audit_path.with_name(f"{audit_path.stem}-{index:03d}.json"),
                    audit_path.with_name(f"{audit_path.stem}-response-{index:03d}.json")))):
            raise RecoveryStop("continuation_seed_already_exists_or_changed")
        client = _RevisionClientFactory(
            self.config, online_model_from(self.config).max_tokens,
            self.perception_endpoint)()
        _admit_solver_client(client, self.perception_endpoint, self.perception_capacity)
        return resumed_seeded_gate_client(
            client, seed=self.seed, audit_path=audit_path, calls=calls,
            framework_state_io=self.framework_state_io)


def _admit_solver_client(client, endpoint, capacity):
    if not client.configured:
        raise FitSetError("solver transport is not configured")
    if endpoint:
        import urllib.request
        from skill3d.cluster.perception_client import RPC_SCHEMA, json_object
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(endpoint.rstrip("/") + "/health", timeout=10) as response:
                health = json_object(response.read(65537))
        except Exception:
            raise FitSetError("RPC gateway health unavailable") from None
        if (health.get("schema_version") != RPC_SCHEMA or health.get("status") != "ready"
                or health.get("identity") != client.audit_identity()
                or health.get("max_concurrency") != capacity):
            raise FitSetError("RPC gateway identity or capacity differs from frozen run")

def _cached_prefix_admission_deferred(args):
    if getattr(args, "cached_prefix_recovery", None):
        return True
    sidecar = getattr(args, "deployment_continuation", None)
    if not sidecar:
        return False
    value = json.loads(Path(sidecar).read_bytes())
    return any(Path(ref).name == "cached_prefix_resume.json"
               for ref in value["preserved_files"])


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


def _select_gate_variants(panel: dict, args) -> list[dict]:
    if args.phase != "accumulation":
        if args.gate_variant_ids or args.gate_panel_name:
            raise FitSetError("gate selectors are only valid for accumulation")
        return []
    if args.gate_mode == "same_question":
        if args.gate_variant_ids or args.gate_panel_name:
            raise FitSetError("same_question cannot accept separate gate selectors")
        return []
    if args.gate_panel_name:
        return select_panel_variants(panel, args.gate_panel_name)
    ids = [value.strip() for value in args.gate_variant_ids.split(",") if value.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise FitSetError("scene_disjoint requires unique explicit gate variant IDs or --gate-panel-name")
    rows = {row["variant_id"]: row for row in panel["system_variants"]}
    unknown = set(ids) - set(rows)
    if unknown:
        raise FitSetError(f"gate variant IDs are outside the normalized panel: {sorted(unknown)}")
    return [rows[variant_id] for variant_id in ids]


def _select_experiment_variants(panel: dict, args) -> tuple[list[dict], list[dict], dict | None]:
    manifest_path = args.experiment_split_manifest
    if manifest_path or args.execution_role or args.gate_mode == "qa_disjoint":
        if args.phase == "final_system":
            if not manifest_path or args.execution_role != "validation":
                raise FitSetError("final_system manifest requires --execution-role validation and --experiment-split-manifest")
            if args.gate_mode == "qa_disjoint":
                raise FitSetError("final_system execution role does not use qa_disjoint or evolution")
        else:
            if args.execution_role:
                raise FitSetError("--execution-role is only valid for final_system")
            if not manifest_path or args.gate_mode != "qa_disjoint":
                raise FitSetError("--experiment-split-manifest and --gate-mode qa_disjoint require each other")
        if args.gate_variant_ids or args.gate_panel_name:
            raise FitSetError("experiment split manifest cannot be mixed with legacy gate selectors")
        partition = load_qa_partition_manifest(manifest_path, metadata_path=args.metadata)
        selected = select_panel_variants(panel, args.panel_name)
        if args.phase == "final_system":
            variants = select_qa_execution_variants(selected, partition, role=args.execution_role)
            gate_variants = []
            overlay = {key: value for key, value in partition.experiment_split.items()
                       if key != "identity_sha256"}
            overlay["execution_role"] = args.execution_role
            experiment_split = normalize_experiment_split(overlay)
            role = args.execution_role
        else:
            if args.seed != partition.experiment_split["seed"]:
                raise FitSetError("accumulation seed differs from frozen experiment split")
            variants, gate_variants = partition_qa_variants(selected, partition)
            experiment_split = partition.experiment_split
            role = "learning"
        selected_ids = [value for value in args.variant_ids.split(",") if value]
        if set(selected_ids) - {row["variant_id"] for row in variants}:
            raise FitSetError(f"--variant-ids must contain only manifest-authorized {role} variants")
        return variants, gate_variants, experiment_split
    variants = select_panel_variants(panel, args.panel_name)
    gate_variants = _select_gate_variants(panel, args)
    gate_ids = {row["variant_id"] for row in gate_variants}
    return [row for row in variants if row["variant_id"] not in gate_ids], gate_variants, None


def _run_system(args) -> int:
    from skill3d.evolution.fitset_stream_v11 import FitSetEvolutionConfig

    panel = load_fitset_panel(args.panel)
    variants, gate_variants, experiment_split = _select_experiment_variants(panel, args)
    if not variants:
        raise FitSetError("gate selection leaves no learning/execution variants")
    frozen_union = [*variants, *gate_variants]
    if (args.phase == "accumulation"
            and experiment_split is None
            and (args.learning_input_mode != "aggregate_v1" or args.gate_mode == "scene_disjoint")
            and any(row["split"] != "induction" for row in variants)):
        raise FitSetError("rich/disjoint learning requires induction-only variants")
    hot_runtime = None
    if getattr(args, "transport_closure", ""):
        from skill3d.storage.scene_runtime import SceneHotRuntime
        if not args.published_inputs or not all((
            args.transport_cache_root, args.transport_endpoint, args.transport_token_file,
        )):
            raise FitSetError("scene transport requires published inputs and complete storage locators")
        hot_runtime = SceneHotRuntime(
            closure_path=args.transport_closure, cache_root=args.transport_cache_root,
            endpoint=args.transport_endpoint, token_file=args.transport_token_file)
    framework_state_io = None
    if getattr(args, "state_workspace_key", ""):
        from skill3d.storage.framework_state_io import FrameworkStateIO
        from skill3d.storage.state_workspace import StateWorkspace
        if hot_runtime is None or not args.state_archive_job:
            raise FitSetError("state workspace requires scene transport and --state-archive-job")
        if args.state_workspace_key != hot_runtime.closure["shards"].get("state", {}).get("key"):
            raise FitSetError("state workspace key differs from frozen transport state")
        workspace = StateWorkspace(
            hot_runtime.cache(), args.state_workspace_key, bounded=True,
            lock_timeout_s=args.state_lock_timeout,
            transfer=hot_runtime.archive_transfer)
        framework_state_io = FrameworkStateIO(
            workspace, job=args.state_archive_job, attempt=args.state_archive_attempt,
            max_file_bytes=args.state_max_file_bytes)
    published_inputs = None
    if args.published_inputs:
        from skill3d.evaluation.published_fitset_v11 import PublishedFitSetInputProvider
        published_inputs = PublishedFitSetInputProvider(
            args.published_inputs, max_scenes=args.scene_cache_size,
            **({"transport_locator": hot_runtime.locator} if hot_runtime else {}))
        if args.bindings or args.direct_run_root or args.video_root or args.video_fallback_root:
            raise FitSetError("published inputs cannot be mixed with legacy input locators")
        if (args.phase == "accumulation" and experiment_split is None
                and any(row["split"] != "induction" for row in variants)):
            raise FitSetError("published live learning requires induction-only variants")
        entries = published_inputs.entries
        for row in frozen_union:
            entry = entries.get(row["qa_id"])
            if entry is None:
                raise FitSetError(f"{row['variant_id']}: published manifest lacks frozen union entry")
            if (entry["scene_id"] != row["scene_id"] or entry["split"] != row["split"]
                    or row.get("source_split", row["split"]) != entry["split"]
                    or entry["reference"]["scene_artifact_ref"] != row["artifact_ref"]):
                raise FitSetError(f"{row['variant_id']}: panel/published frozen union identity changed")
        bindings = {
            row["variant_id"]: entries[row["qa_id"]]["reference"]
            for row in frozen_union
        }
    else:
        if not args.bindings or not args.direct_run_root:
            raise FitSetError("legacy inputs require --bindings and --direct-run-root")
        bindings = load_variant_bindings(args.bindings, frozen_union)
    raw = resolve_runtime(load_config(args.config), seed=args.seed)
    model = online_model_from(raw)
    if (
        not model.enabled
        or model.provider != "bailian"
        or model.model != "qwen3.8-flash"
    ):
        raise FitSetError(
            "fit-set system execution requires enabled bailian/qwen3.8-flash")
    client_config = {"online_model": asdict(model)}
    if args.perception_endpoint:
        from skill3d.cluster.perception_client import gateway_endpoint
        gateway_endpoint(args.perception_endpoint)
        if not 1 <= args.perception_capacity <= 64:
            raise FitSetError("RPC gateway capacity must be frozen between 1 and 64")
        request_concurrency = (
            args.request_concurrency or args.workers
            if args.max_requests else max(args.workers, args.episode_concurrency)
        )
        if request_concurrency > args.perception_capacity:
            raise FitSetError("request concurrency exceeds frozen RPC gateway capacity")
        if args.phase == "accumulation" and not 1 <= args.revision_max_tokens <= model.max_tokens:
            raise FitSetError("revision token budget must be 1..frozen RPC model max_tokens")
        if (args.phase == "accumulation" and args.learning_input_mode in {
                "diagnosis_v1", "diagnosis_multimodal_v1"}
                and not 1 <= args.diagnosis_max_output_tokens <= model.max_tokens):
            raise FitSetError("diagnosis token budget must be 1..frozen RPC model max_tokens")
    elif args.perception_capacity:
        raise FitSetError("--perception-capacity requires --perception-endpoint")
    if getattr(args, "cached_prefix_recovery", None) and (
            not args.resume or not args.deployment_continuation
            or args.preflight_only):
        raise FitSetError("cached-prefix recovery requires resume/deployment, not preflight")
    deferred_admission = _cached_prefix_admission_deferred(args)
    if (not args.preflight_only and not getattr(args, "initialize_only", False)
            and not deferred_admission):
        client = _RevisionClientFactory(
            client_config, model.max_tokens, args.perception_endpoint)()
        if not client.configured:
            raise FitSetError(
                "RPC endpoint/token is not configured" if args.perception_endpoint
                else f"{model.api_key_env} is not configured")
        if args.perception_endpoint:
            import urllib.request
            from skill3d.cluster.perception_client import RPC_SCHEMA, json_object
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(args.perception_endpoint.rstrip("/") + "/health", timeout=10) as response:
                    health = json_object(response.read(65537))
            except Exception:
                raise FitSetError("RPC gateway health unavailable") from None
            if (health.get("schema_version") != RPC_SCHEMA or health.get("status") != "ready"
                    or health.get("identity") != client.audit_identity()
                    or health.get("max_concurrency") != args.perception_capacity):
                raise FitSetError("RPC gateway identity or capacity differs from frozen run")
    detector = raw.get("detection", {}).get("endpoint")
    if detector:
        os.environ.setdefault("SKILL3D_DETECTOR_ENDPOINT", str(detector))
    paths = paths_from(raw)
    if args.derived_binding_root:
        if published_inputs is None:
            raise FitSetError("--derived-binding-root requires --published-inputs")
        from skill3d.evaluation.derived_published_fitset_v11 import (
            DerivedPublishedFitSetInputProvider,
        )
        published_inputs = DerivedPublishedFitSetInputProvider(
            args.published_inputs, derivation_root=args.derived_binding_root,
            binding_cfg=_base_config(raw), max_scenes=args.scene_cache_size,
            **({"transport_locator": hot_runtime.locator,
                "derived_replay": hot_runtime.closure["derived_replay"]} if hot_runtime else {}))
    if hot_runtime:
        from skill3d.storage.scene_runtime import SceneHotInputProvider
        published_inputs = SceneHotInputProvider(published_inputs, hot_runtime)
    fallback_roots = (
        list(args.video_fallback_root)
        if args.video_fallback_root
        else list(paths.raw_video_fallbacks)
    )
    item_provider = published_inputs or _FitSetItemProvider(
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
        gate_variants=gate_variants,
        item_provider=item_provider,
        input_receipt_provider=item_provider.receipt,
        input_contract_identity={
            **item_provider.input_contract_identity,
            **({"offline_metadata": file_identity(args.metadata)}
               if published_inputs is not None else
               {"prepared_bindings_manifest": file_identity(args.bindings)}),
            "model_transport": {
                "kind": "perception_rpc" if args.perception_endpoint else "direct_bailian",
                "endpoint": args.perception_endpoint,
                "upstream_capacity": args.perception_capacity or None,
            },
            **({"shared_request_governor_root": str(
                Path(args.request_governor_root).resolve())}
               if args.request_governor_root else {}),
        },
        binding_refs=bindings,
        base_cfg=_base_config(raw),
        phase=args.phase,
        published_inputs=published_inputs,
        **({"deployment_continuation": args.deployment_continuation}
           if getattr(args, "deployment_continuation", None) else {}),
        **({"cached_prefix_recovery": args.cached_prefix_recovery}
           if getattr(args, "cached_prefix_recovery", None) else {}),
        **({"experiment_split": experiment_split} if experiment_split is not None else {}),
        skills_for_variant=skill_provider,
        min_free_gib=args.min_free_gib,
        workers=args.workers,
        scheduler=args.scheduler,
        skill_owners=args.skill_owners,
        reviser_concurrency=getattr(args, "reviser_concurrency", 1),
        request_governor_config=(
            _request_policy(args) if args.max_requests else None),
        client_factory=_SeededClientFactory(
            client_config, args.seed, Path(args.output_dir), args.perception_endpoint,
            framework_state_io, args.perception_capacity, deferred_admission),
        episode_concurrency=(
            args.episode_concurrency if args.episode_concurrency else None),
        evolution_config=(
            FitSetEvolutionConfig(
                gate_size=args.gate_size,
                gate_seeds=(args.seed, args.seed + 1),
                max_revision_attempts=args.max_revision_attempts,
                max_tokens=args.revision_max_tokens,
                scheduling_mode=args.scheduling_mode,
                max_revision_requests_per_slot=args.max_revision_requests_per_slot,
                max_gate_candidates_per_slot=args.max_gate_candidates_per_slot,
                learning_input_mode=args.learning_input_mode,
                gate_mode=args.gate_mode,
                diagnosis_max_cases=args.diagnosis_max_cases,
                diagnosis_max_input_chars=args.diagnosis_max_input_chars,
                diagnosis_max_images=args.diagnosis_max_images,
                diagnosis_max_output_tokens=args.diagnosis_max_output_tokens,
            ) if args.phase == "accumulation" else None
        ),
        label_metadata=(
            {qa: label for qa, label in load_label_metadata(args.metadata).items()
             if qa in {row["qa_id"] for row in frozen_union}}
            if args.phase == "accumulation" else None
        ),
        revision_client_factory=_RevisionClientFactory(
            client_config, args.revision_max_tokens, args.perception_endpoint),
        gate_client_factory=_GateClientFactory(
            client_config, args.perception_endpoint, framework_state_io),
        framework_state_io=framework_state_io,
    )
    if args.request_governor_root:
        if not args.max_requests:
            raise FitSetError("--request-governor-root requires --max-requests")
        runner._request_governor_root = Path(args.request_governor_root).resolve()
    if framework_state_io is not None and args.max_requests:
        framework_state_io.validate_root(runner.request_governor_root)
    if framework_state_io is not None and not getattr(args, "initialize_only", False):
        destinations = [runner.root / "summary.json"]
        if args.preflight_only:
            destinations.extend((runner.root / "preflight", runner.root / "preflight_summary.json"))
        else:
            destinations.extend(runner.root / name for name in (
                "variants", "results", "request_seeds"))
            if args.phase == "accumulation":
                destinations.append(runner.root / "workers")
            if getattr(args, "cached_prefix_recovery", None):
                destinations.append(runner.root / "solver_recoveries")
            elif getattr(args, "deployment_continuation", None):
                # Auto-pinned specs are loaded by initialize, after this scope check.
                destinations.append(runner.root / "solver_recoveries")
        if args.scheduler == "episode_pool":
            destinations.extend(runner.root / name for name in (
                "episode_pool.lock", "episode_pool_receipt.json", "scheduler_receipts"))
        if args.max_requests:
            destinations.append(runner.request_governor_root)
        framework_state_io.validate_destinations(destinations)
    if getattr(args, "initialize_only", False):
        manifest = runner.initialize(resume=args.resume)
        print(json.dumps({
            "status": "paused_initialized", "run_identity": runner.run_identity,
            "deployment_identity": runner.deployment_identity,
            "completed_variants": len(runner.completed()), "model_requests_initiated": 0,
            "manifest_identity_sha256": manifest["identity_sha256"],
        }, sort_keys=True))
        return 0
    action = runner.preflight if args.preflight_only else runner.run
    summary = action(
        resume=args.resume,
        max_variants=args.max_variants,
        variant_ids=selected_ids,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _request_policy(args):
    from skill3d.synthesis.request_governor import RequestGovernorConfig
    return RequestGovernorConfig(
        max_requests=args.max_requests,
        max_inflight=args.request_concurrency or args.workers,
        requests_per_second=args.requests_per_second,
    )


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
    run.add_argument("--bindings", default="")
    run.add_argument("--published-inputs", default="",
                     help="hash-pinned label-free r04 catalog and frozen scene inputs")
    run.add_argument("--derived-binding-root", default="",
                     help="explicit separate store for audited v3-to-current M5 derivation")
    run.add_argument("--metadata", required=True)
    run.add_argument("--transport-closure", default="")
    run.add_argument("--transport-cache-root", default="")
    run.add_argument("--transport-endpoint", default="")
    run.add_argument("--transport-token-file", default="")
    run.add_argument("--state-workspace-key", default="",
                     help="opt-in existing adopted state workspace; never adopts or expands quota")
    run.add_argument("--state-archive-job", default="",
                     help="trusted saved checkpoint archive job identity")
    run.add_argument("--state-archive-attempt", type=int, default=0)
    run.add_argument("--state-max-file-bytes", type=int, default=64 * 1024 * 1024)
    run.add_argument("--state-lock-timeout", type=float, default=120.0)
    run.add_argument("--deployment-continuation", default=None)
    run.add_argument("--cached-prefix-recovery", default=None,
                     help="owner-only deployment-sealed spec for one interrupted solver prefix")
    run.add_argument("--initialize-only", action="store_true",
                     help="validate preserved run identity without dispatching requests or preflight")
    run.add_argument("--direct-run-root", default="")
    run.add_argument("--video-root")
    run.add_argument("--video-fallback-root", action="append", default=[])
    run.add_argument("--scene-cache-size", type=int, default=2)
    run.add_argument("--min-free-gib", type=float, default=80.0)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--scheduler", choices=("slot_shards", "episode_pool"),
                     default="slot_shards")
    run.add_argument("--skill-owners", type=int, default=4,
                     help="fixed Skill library owners, independent of episode executors")
    run.add_argument("--reviser-concurrency", type=int, default=1,
                     help="independent owner revision processes; 1 preserves serial scheduling")
    run.add_argument("--max-requests", type=int, default=0,
                     help="shared solver/diagnosis/reviser/gate request budget; required for episode_pool")
    run.add_argument("--request-concurrency", type=int, default=0)
    run.add_argument("--request-governor-root", default="",
                     help="one frozen request ledger shared across authorized experiment phases")
    run.add_argument("--requests-per-second", type=float, default=0)
    run.add_argument("--perception-endpoint", default="",
                     help="public RPC origin; upstream API key stays on the service node")
    run.add_argument("--perception-capacity", type=int, default=0,
                     help="frozen gateway API capacity, checked before online execution")
    run.add_argument("--episode-concurrency", type=int, default=0,
                     help="global cap on concurrently executing episodes "
                          "across every worker and gate; 0 uses the scheduler default")
    run.add_argument("--gate-size", type=int, default=8)
    run.add_argument("--learning-input-mode", choices=(
        "aggregate_v1", "full_cases_v1", "diagnosis_v1", "diagnosis_multimodal_v1"),
        default="full_cases_v1")
    run.add_argument("--experiment-split-manifest", default="",
                     help="opt-in hash-anchored QA learning/validation allocation; "
                          "accumulation requires qa_disjoint; final_system requires "
                          "--execution-role validation; source split stays unchanged")
    run.add_argument("--execution-role", choices=("validation",), default="",
                     help="final_system only: execute manifest-authorized validation QAs "
                          "with frozen Skills, no learning/gate evolution; "
                          "same-scene question-disjoint from the manifest's learning role")
    run.add_argument("--gate-mode", choices=("same_question", "scene_disjoint", "qa_disjoint"),
                     default="scene_disjoint",
                     help="qa_disjoint requires --experiment-split-manifest: "
                          "same-scene question-disjoint, not scene-disjoint; "
                          "all variants of a QA inherit one role; "
                          "validation QAs cannot enter learning")
    gate_selector = run.add_mutually_exclusive_group()
    gate_selector.add_argument("--gate-variant-ids", default="",
                               help="comma-separated held-out IDs from the normalized panel")
    gate_selector.add_argument("--gate-panel-name", choices=PANEL_NAMES, default="",
                               help="freeze a separate panel selection as the held-out gate")
    run.add_argument("--diagnosis-max-cases", type=int, default=4)
    run.add_argument("--diagnosis-max-input-chars", type=int, default=64000)
    run.add_argument("--diagnosis-max-images", type=int, default=8)
    run.add_argument("--diagnosis-max-output-tokens", type=int, default=4096)
    run.add_argument("--max-revision-attempts", type=int, default=3)
    run.add_argument("--revision-max-tokens", type=int, default=8192)
    run.add_argument("--scheduling-mode", choices=("novel_failure_v1", "batch_v1"),
                     default="batch_v1")
    run.add_argument("--max-revision-requests-per-slot", type=int, default=18)
    run.add_argument("--max-gate-candidates-per-slot", type=int, default=6)
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
        if (args.command == "run-system" and args.scheduler == "episode_pool"
                and args.max_requests < 1):
            raise FitSetError("episode_pool requires an explicit --max-requests budget")
        return int(args.handler(args))
    except (FitSetError, RevisionOperationalError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except BaseException as exc:
        if args.command == "run-system" and getattr(args, "state_workspace_key", ""):
            from skill3d.storage.framework_state_io import FrameworkStorageRefused
            if isinstance(exc, FrameworkStorageRefused):
                print("FrameworkStorageRefused: storage paused; no automatic retry", file=sys.stderr)
                return 75
        raise


if __name__ == "__main__":
    raise SystemExit(main())
