"""Batch preparation of audited question bindings for v11 streams."""

from __future__ import annotations

import copy
import shutil
import uuid
from pathlib import Path
from typing import Callable, Mapping, Sequence

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evaluation.skill_ablation_v11 import input_identity
from skill3d.legacy.readers import load_artifact_v5
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig, PreparedObjectBinding
from skill3d.segmentation.binding_validation import validate_binding
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore,
    PreparedBindingStoreError,
    prepared_binding_request_identity,
)
from skill3d.segmentation.question_binding import refresh_coverage
from skill3d.segmentation.sam2_tracker import audit_semantic_mask_conflicts

PREPARATION_MANIFEST_VERSION = "v11-prepared-bindings/1.0"


class V11BindingPreparationError(RuntimeError):
    """A binding batch cannot be sealed without changing its identity."""


def _scene_groups(items: Sequence[EpisodeItem]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for item in items:
        qa_id = str(item.episode.qa_id)
        scene_id = str(item.episode.scene_name or "")
        if not qa_id or not scene_id:
            raise V11BindingPreparationError(
                "binding preparation requires question and scene identities"
            )
        groups.setdefault(scene_id, []).append(qa_id)
    return {
        scene_id: sorted(qa_ids)
        for scene_id, qa_ids in groups.items()
    }


def prepare_shared_bindings_v11(
    items: Sequence[EpisodeItem],
    *,
    artifact_paths: Mapping[str, str | Path],
    base_cfg: OnlineRunConfig,
    store_root: str | Path,
    client_factory: Callable[[str], object] | None = None,
    max_questions_per_scene: int = 8,
    min_free_gib: float = 80.0,
) -> dict:
    """Prepare, cross-audit and publish one immutable binding per question."""
    if int(max_questions_per_scene) <= 0:
        raise ValueError("max_questions_per_scene must be positive")
    if float(min_free_gib) < 0:
        raise ValueError("min_free_gib must be non-negative")
    if base_cfg.mode != "real":
        raise V11BindingPreparationError(
            "prepared bindings require the real reconstruction/M5 path"
        )
    indexed: dict[str, EpisodeItem] = {}
    for item in items:
        qa_id = str(item.episode.qa_id)
        if qa_id in indexed:
            raise V11BindingPreparationError(
                f"duplicate binding question identity: {qa_id}"
            )
        if item.input_error is not None:
            raise V11BindingPreparationError(
                f"{qa_id}: input_error cannot produce a prepared binding"
            )
        if qa_id not in artifact_paths:
            raise V11BindingPreparationError(
                f"{qa_id}: missing scene artifact"
            )
        indexed[qa_id] = item
    groups = _scene_groups(list(indexed.values()))
    oversized = {
        scene_id: len(qa_ids)
        for scene_id, qa_ids in groups.items()
        if len(qa_ids) > int(max_questions_per_scene)
    }
    if oversized:
        raise V11BindingPreparationError(
            f"scene question cap {max_questions_per_scene} exceeded: {oversized}"
        )

    store = PreparedBindingStore(store_root)
    identities = {
        qa_id: input_identity(item, artifact_paths[qa_id])
        for qa_id, item in indexed.items()
    }
    requests = {
        qa_id: prepared_binding_request_identity(
            input_identity=identities[qa_id],
            cfg=base_cfg,
            scene_audit_qa_ids=groups[item.episode.scene_name],
        )
        for qa_id, item in indexed.items()
    }
    work_root = store.root / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    cache_hits: set[str] = set()
    references = {}
    for scene_id, qa_ids in sorted(groups.items()):
        free_gib = shutil.disk_usage(store.root).free / (1024 ** 3)
        if free_gib < float(min_free_gib):
            raise V11BindingPreparationError(
                f"free disk {free_gib:.2f} GiB is below the "
                f"{float(min_free_gib):.2f} GiB runtime reserve"
            )
        bindings: dict[str, PreparedObjectBinding] = {}
        artifacts = {}
        work_dirs: dict[str, Path] = {}
        scene_cache_hits: set[str] = set()
        try:
            for qa_id in qa_ids:
                item = indexed[qa_id]
                artifact_path = Path(
                    artifact_paths[qa_id]
                ).expanduser().resolve()
                request = requests[qa_id]
                key = store.key_for(request)
                if store.entry_path(key).is_dir():
                    bindings[qa_id] = store.load(
                        key,
                        expected_request_identity=request,
                        scene_artifact=artifact_path,
                        scene_input_identity=identities[qa_id],
                    )
                    scene_cache_hits.add(qa_id)
                    cache_hits.add(qa_id)
                    artifacts[qa_id] = load_artifact_v5(
                        artifact_path
                    )
                    continue
                art = load_artifact_v5(artifact_path)
                art = runner._prepare_m4_quality(
                    art,
                    item.pixels,
                    persist_quality=False,
                )
                work = (
                    work_root
                    / f"{key}.{uuid.uuid4().hex}.tmp"
                )
                work.mkdir()
                work_dirs[qa_id] = work
                llm = (
                    client_factory(qa_id)
                    if client_factory is not None
                    else None
                )
                values = runner._bind_objects_best_effort(
                    art,
                    [pixels.copy() for pixels in item.pixels],
                    None,
                    item.episode.model_copy(deep=True),
                    copy.deepcopy(base_cfg),
                    llm,
                    m5_out_dir=str(work),
                )
                binding = PreparedObjectBinding(*values)
                if binding.stats.get("binding_error"):
                    raise V11BindingPreparationError(
                        f"{qa_id}: {binding.stats['binding_error']}"
                    )
                bindings[qa_id] = binding
                artifacts[qa_id] = art

            reconstruction_ids = {
                str(
                    artifacts[qa_id].reconstruction_input_identity
                    or artifacts[qa_id].frame_set_hash
                    or artifacts[qa_id].artifact_id
                )
                for qa_id in qa_ids
            }
            if len(reconstruction_ids) != 1:
                raise V11BindingPreparationError(
                    f"{scene_id}: one scene group has multiple reconstruction identities"
                )
            scene_identity = next(iter(reconstruction_ids))
            before = {
                qa_id: [
                    obj.model_dump(mode="json")
                    for obj in bindings[qa_id].objects
                ]
                for qa_id in qa_ids
                if qa_id in scene_cache_hits
            }
            audited, audits = audit_semantic_mask_conflicts(
                [
                    (qa_id, bindings[qa_id].objects)
                    for qa_id in qa_ids
                ],
                scene_identity=scene_identity,
            )
            for qa_id in qa_ids:
                binding = bindings[qa_id]
                generation_sha256 = str(
                    (binding.stats.get("generation") or {}).get("sha256") or ""
                )
                binding.objects = [
                    obj.model_copy(update={
                        "binding_validation": validate_binding(
                            obj,
                            generation_sha256=generation_sha256,
                        )
                    })
                    if obj.grounding_status == "question_driven"
                    else obj
                    for obj in audited[qa_id]
                ]
                binding.stats["semantic_conflict_audits"] = [
                    audit for audit in audits if qa_id in audit["qa_ids"]
                ]
                if binding.stats.get("grounding"):
                    binding.stats["grounding"] = refresh_coverage(
                        binding.stats["grounding"],
                        binding.objects,
                    )
                if qa_id in scene_cache_hits:
                    after = [
                        obj.model_dump(mode="json")
                        for obj in binding.objects
                    ]
                    if after != before[qa_id]:
                        raise PreparedBindingStoreError(
                            f"{qa_id}: sealed cross-question audit changed"
                        )
            for qa_id, binding in bindings.items():
                request = requests[qa_id]
                key = store.key_for(request)
                if qa_id in scene_cache_hits:
                    references[qa_id] = store.reference(key)
                else:
                    references[qa_id] = store.publish(
                        request_identity=request,
                        scene_artifact=artifact_paths[qa_id],
                        scene_input_identity=identities[qa_id],
                        binding=binding,
                    )
        finally:
            for work in work_dirs.values():
                shutil.rmtree(work, ignore_errors=True)
    return {
        "schema_version": PREPARATION_MANIFEST_VERSION,
        "store_root": str(store.root),
        "n_questions": len(indexed),
        "n_scenes": len(groups),
        "max_questions_per_scene": int(max_questions_per_scene),
        "min_free_gib": float(min_free_gib),
        "scene_groups": groups,
        "cache_hit_qa_ids": sorted(cache_hits),
        "bindings": {
            qa_id: references[qa_id]
            for qa_id in sorted(references)
        },
    }


__all__ = [
    "PREPARATION_MANIFEST_VERSION",
    "V11BindingPreparationError",
    "prepare_shared_bindings_v11",
]
