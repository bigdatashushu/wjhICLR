"""Explicit v3 publication replay followed by a separate current M5 derivation."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

from skill3d.evaluation.published_fitset_v11 import (
    PublishedFitSetInputProvider, _require,
)
from skill3d.evolution.prepared_binding_derivation_v11 import (
    PreparedBindingDerivationContext, publish_derived_binding_scene_v11,
)
from skill3d.routing.task_classifier import canonical_task
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore, load_prepared_binding_reference, prepared_binding_request_identity,
)


class DerivedPublishedFitSetInputProvider(PublishedFitSetInputProvider):
    """Keep original catalog anchors; derive only into the caller's new store."""

    def __init__(self, manifest_path, *, derivation_root, binding_cfg, max_scenes=2):
        super().__init__(manifest_path, max_scenes=max_scenes)
        self.derivation_root = Path(derivation_root).expanduser().absolute()
        self.binding_cfg = copy.deepcopy(binding_cfg)
        self._derived_scenes = {}
        _require(bool(self.scene_scopes), "derivation requires bounded fullQA scene scopes")
        _require(all(
            self.derivation_root != Path(entry["reference"]["store_root"])
            for entry in self._entries.values()
        ), "derivation destination must be separate from source stores")
        self._input_contract_identity["binding_derivation"] = {
            "mode": "explicit_v3_category_only_to_current_v4",
            "destination_root": str(self.derivation_root),
            "source_catalog_preserved": True,
            "new_model_requests": 0,
        }

    def __getstate__(self):
        state = super().__getstate__()
        state["_derived_scenes"] = {}
        return state

    def _derive_scene(self, entry):
        scene_id = entry["scene_id"]
        scope = self.scene_scopes[scene_id]
        scene = self._scene(entry)
        ids = sorted(scope["qa_ids"])
        indexed = {str(item.episode.qa_id): item for item in scene.items}
        _require(set(ids) == set(indexed), "derivation must include the full original scene QA group")
        sources, inputs, relocations, paths = {}, {}, {}, {}
        for qa_id in ids:
            original = self._entries[qa_id]
            variant = {
                "qa_id": qa_id, "scene_id": scene_id, "split": original["split"],
                "source_split": original["split"],
                "artifact_ref": original["reference"]["scene_artifact_ref"],
                "question_type": canonical_task(indexed[qa_id].episode.question_type),
            }
            transport = self._transport(variant, indexed[qa_id])
            sources[qa_id] = {
                "reference": original["reference"], "identity": transport["identity"],
                "artifact_path": original["scene_artifact"],
                "migration_receipt": transport["migration"],
                "migration_destination_root": original["publication_root"],
            }
            inputs[qa_id] = transport["identity"]
            relocations[qa_id] = transport["resolver"]
            paths[qa_id] = original["scene_artifact"]
        request = entry["reference"]["request_identity"]
        cfg = replace(
            copy.deepcopy(self.binding_cfg), seed=request["seed"],
            prepared_binding_seed=request["seed"],
            prepared_binding_source_compatibility=None,
            semantic_review_config=copy.deepcopy(
                request["parameters"]["experimental_semantic_review"]),
        )
        requests = {
            qa_id: prepared_binding_request_identity(
                input_identity=inputs[qa_id], cfg=cfg, scene_audit_qa_ids=ids)
            for qa_id in ids
        }
        context = PreparedBindingDerivationContext(
            sources=sources, items=[indexed[qa_id] for qa_id in ids],
            target_inputs=inputs, target_requests=requests,
            target_artifact_paths=paths, cfg=cfg, target_relocations=relocations,
        )
        references = publish_derived_binding_scene_v11(
            context, store_root=self.derivation_root / scene_id)
        store = PreparedBindingStore(self.derivation_root / scene_id)
        migrations = {}
        for qa_id in ids:
            original_migration = sources[qa_id]["migration_receipt"]
            migrations[qa_id] = store.create_migration_receipt(
                references[qa_id]["key"], scene_input_identity=inputs[qa_id],
                destination_root=sources[qa_id]["migration_destination_root"],
                scene_artifact_ref=original_migration["artifact"]["target_ref"],
                dependency_refs={
                    name: row["target_ref"]
                    for name, row in original_migration["dependencies"].items()},
            )
        return context, references, migrations

    def prepare_published(self, variant, item):
        transport = self._transport(variant, item)
        entry, resolver = transport["entry"], transport["resolver"]
        scene_id = entry["scene_id"]
        with self.cache_lock:
            if scene_id not in self._derived_scenes:
                self._derived_scenes[scene_id] = self._derive_scene(entry)
            context, references, migrations = self._derived_scenes[scene_id]
        reference = references[entry["qa_id"]]
        prepared = load_prepared_binding_reference(
            reference, scene_input_identity=transport["identity"],
            scene_artifact=entry["scene_artifact"],
            migration_receipt=migrations[entry["qa_id"]],
            migration_destination_root=entry["publication_root"])
        return {
            "artifact_path": resolver.artifact_path,
            "artifact": resolver.resolved_artifact(),
            "identity": transport["identity"],
            "reference": copy.deepcopy(entry["reference"]),
            "derived_reference": copy.deepcopy(reference),
            "derived_migration_receipt": copy.deepcopy(migrations[entry["qa_id"]]),
            "binding_derivation_context": context,
            "prepared": prepared, "artifact_relocation": resolver,
            "transport_receipt": transport["receipt"],
        }
