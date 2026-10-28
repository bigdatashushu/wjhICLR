"""Explicit v3 derivation, or byte-preserving replay of an approved derived store."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import threading

from skill3d.evaluation.published_fitset_v11 import (
    PublishedFitSetInputProvider, _require,
)
from skill3d.evolution.prepared_binding_derivation_v11 import (
    DERIVATION_SCHEMA_VERSION, LEGACY_SOURCE_COMMIT,
    PreparedBindingDerivationContext, _check_cache,
    _content, _context_fingerprint, _stamp, publish_derived_binding_scene_v11,
)
from skill3d.infra.m5_identity import content_sha256
from skill3d.routing.task_classifier import canonical_task
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore, load_prepared_binding_reference, prepared_binding_request_identity,
)


class DerivedPublishedFitSetInputProvider(PublishedFitSetInputProvider):
    """Keep original catalog anchors; derive only into the caller's new store."""

    def __init__(
        self, manifest_path, *, derivation_root, binding_cfg, max_scenes=2,
        transport_locator=None, derived_replay=None,
    ):
        """``derived_replay`` pins existing references/receipts, never rederives.

        Schema ``published-derived-replay-v1`` has ``scenes[scene_id]`` rows with
        ``references`` and ``migration_receipts`` keyed by the complete QA scope.
        References keep their original logical store roots; the transport locator
        declares both source and derived physical copies. Receipts may be full
        mappings or exact pinned file locators. Keep ``derivation_root`` logical.
        """
        super().__init__(
            manifest_path, max_scenes=max_scenes, transport_locator=transport_locator)
        self.derivation_root = Path(derivation_root).expanduser().absolute()
        self.binding_cfg = copy.deepcopy(binding_cfg)
        self._approved_derived_replay = copy.deepcopy(derived_replay)
        self._derived_scenes = {}
        self._derivation_locks = {}
        _require(bool(self.scene_scopes), "derivation requires bounded fullQA scene scopes")
        _require(transport_locator is None or derived_replay is not None,
                 "relocated derived inputs require approved existing derived replay")
        _require(all(
            self.derivation_root != Path(entry["reference"]["store_root"])
            for entry in self._entries.values()
        ), "derivation destination must be separate from source stores")
        if derived_replay is not None:
            _require(self.transport_locator is not None
                     and isinstance(derived_replay, dict)
                     and set(derived_replay) == {"schema_version", "scenes"}
                     and derived_replay["schema_version"] == "published-derived-replay-v1"
                     and isinstance(derived_replay["scenes"], dict)
                     and set(derived_replay["scenes"]) == set(self.scene_scopes),
                     "derived replay requires approved complete scene references")
            for scene_id, row in derived_replay["scenes"].items():
                ids = set(self.scene_scopes[scene_id]["qa_ids"])
                _require(isinstance(row, dict)
                         and set(row) == {"references", "migration_receipts"}
                         and all(isinstance(row[name], dict) and set(row[name]) == ids
                                 for name in row),
                         "derived replay must pin the complete scene QA group")
                _require(all(
                    Path(reference["store_root"]) == self.derivation_root / scene_id
                    for reference in row["references"].values()),
                    "derived replay original store differs from derivation destination")
        self._input_contract_identity["binding_derivation"] = {
            "mode": "explicit_v3_category_only_to_current_v4",
            "destination_root": str(self.derivation_root),
            "source_catalog_preserved": True,
            "new_model_requests": 0,
        }

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_derivation_locks", None)
        scene_id = state.get("_spawn_replay_scene")
        if scene_id:
            _require(set(state["_derived_scenes"]) == {scene_id},
                     "spawn replay must retain exactly the requested scene")
            _require(_check_cache(state["_derived_scenes"][scene_id][0]),
                     "spawn replay requires a validated scene context")
        else:
            state["_derived_scenes"] = {}
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._derivation_locks = {}

    def for_spawned_episode(self, variant):
        """Carry this parent's validated replay for one episode, never other scenes."""
        entry = self._entry(variant)
        scene_id = entry["scene_id"]
        with self.cache_lock:
            state = super().__getstate__()
            state["_derived_scenes"] = {}
            state["_spawn_replay_scene"] = None
            if scene_id in self._derived_scenes:
                replay = self._derived_scenes[scene_id]
                _require(_check_cache(replay[0]),
                         "spawn replay requires a validated scene context")
                state["_derived_scenes"] = {scene_id: replay}
                state["_spawn_replay_scene"] = scene_id
            clone = type(self).__new__(type(self))
            clone.__setstate__(state)
            return clone

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

    def _replay_derived_scene(self, entry):
        """Revalidate old source/output seals and build only a physical context."""
        _require(self.binding_cfg.mode == "real", "only real sealed inputs may be replayed")
        scene_id = entry["scene_id"]
        approved = self._approved_derived_replay["scenes"][scene_id]
        ids = sorted(self.scene_scopes[scene_id]["qa_ids"])
        scene = self._scene(entry)
        indexed = {str(item.episode.qa_id): item for item in scene.items}
        _require(set(indexed) == set(ids), "derived replay scene QA group changed")
        sources, inputs, requests, paths, relocations = {}, {}, {}, {}, {}
        migrations, anchors, expected, closure = {}, {}, {}, {}
        references = copy.deepcopy(approved["references"])
        source_stores, derived_stores = {}, {}

        def seal(value):
            _require(isinstance(value, dict) and value.get("sha256") == content_sha256({
                k: v for k, v in value.items() if k != "sha256"}),
                "derived replay sealed provenance hash changed")

        def track_store(store, key):
            root = store.entry_path(key)
            for path in (root, *root.rglob("*")):
                closure.setdefault(str(path), _stamp(path))

        for qa_id in ids:
            original = self._entries[qa_id]
            item = indexed[qa_id]
            variant = {
                "qa_id": qa_id, "scene_id": scene_id, "split": original["split"],
                "artifact_ref": original["reference"]["scene_artifact_ref"],
                "question_type": canonical_task(item.episode.question_type),
            }
            transport = self._transport(variant, item)
            identity = transport["identity"]
            source_store = source_stores.setdefault(str(transport["store_root"]),
                                                    PreparedBindingStore(transport["store_root"]))
            track_store(source_store, original["reference"]["key"])
            source_manifest = source_store.validate_reference(original["reference"])
            source_binding = load_prepared_binding_reference(
                original["reference"], scene_input_identity=identity,
                physical_store_root=source_store.root,
                scene_artifact=transport["resolver"].artifact_path,
                migration_receipt=transport["migration"],
                migration_destination_root=transport["destination_root"])
            generation = source_binding.stats.get("generation")
            _require(isinstance(generation, dict), "derived replay source generation missing")
            _require(generation.get("sha256") == content_sha256(generation.get("identity"))
                     == source_manifest["binding_generation_sha256"],
                     "derived replay initial v3 generation changed")
            anchors[qa_id] = {
                "reference": copy.deepcopy(original["reference"]),
                "input_sha256": identity["sha256"],
                "request_identity": copy.deepcopy(source_manifest["request_identity"]),
                "manifest_sha256": source_manifest["manifest_sha256"],
                "payload_sha256": source_manifest["payload"]["sha256"],
                "generation_sha256": generation["sha256"],
                "generation_stats_sha256": content_sha256(generation),
                "binding_content_sha256": content_sha256(_content(source_binding)),
                "relocation_sha256": content_sha256(transport["original_migration"]),
            }
            reference = references[qa_id]
            physical_root = self._physical_root(reference["store_root"])
            _require(physical_root != source_store.root,
                     "derived replay destination must be separate from source stores")
            derived_store = derived_stores.setdefault(str(physical_root),
                                                      PreparedBindingStore(physical_root))
            track_store(derived_store, reference["key"])
            derived_store.validate_reference(reference)
            old_migration = approved["migration_receipts"][qa_id]
            if isinstance(old_migration, (str, Path)):
                path = self._physical_file(old_migration, None)
                closure.setdefault(str(path), _stamp(path))
            migrations[qa_id], destination = self._rebase_migration(
                derived_store, reference["key"], old_migration, identity,
                original["publication_root"])
            resolver = derived_store.validate_migration_receipt(
                reference["key"], migrations[qa_id], scene_input_identity=identity,
                destination_root=destination)
            binding = load_prepared_binding_reference(
                reference, scene_input_identity=identity, physical_store_root=physical_root,
                migration_receipt=migrations[qa_id], scene_artifact=resolver.artifact_path,
                migration_destination_root=destination)
            receipt = binding.stats.get("binding_derivation")
            seal(receipt)
            _require(receipt.get("schema_version") == DERIVATION_SCHEMA_VERSION
                     and receipt.get("scene_audit_qa_ids") == ids
                     and receipt.get("new_model_requests") == 0
                     and receipt.get("new_detector_requests") == 0
                     and receipt.get("new_sam2_tracks") == 0
                     and receipt.get("generation_resealed") is False
                     and receipt.get("quality_assessed") is False
                     and receipt.get("source_commit") == LEGACY_SOURCE_COMMIT
                     and receipt.get("semantic_policy") == "category-only"
                     and receipt.get("output_binding_sha256") == content_sha256(_content(binding))
                     and binding.stats.get("generation") == generation,
                     "derived replay output or initial generation seal changed")
            requests[qa_id] = reference["request_identity"]
            inputs[qa_id] = identity
            paths[qa_id] = resolver.artifact_path
            relocations[qa_id] = resolver
            sources[qa_id] = {
                "reference": copy.deepcopy(original["reference"]), "identity": identity,
                "artifact_path": transport["resolver"].artifact_path,
                "migration_receipt": transport["migration"],
                "migration_destination_root": transport["destination_root"],
            }
            expected[qa_id] = copy.deepcopy(receipt)
            for path in (resolver.artifact_path, *resolver.dependency_paths.values()):
                closure.setdefault(str(path), _stamp(path))

        destinations = {
            qa: {"input_sha256": inputs[qa]["sha256"], "request_identity": requests[qa]}
            for qa in ids}
        for qa_id, receipt in expected.items():
            _require(receipt["source"] == anchors[qa_id]
                     and receipt["scene_sources"] == anchors
                     and receipt["destination"] == destinations[qa_id]
                     and receipt["scene_destinations"] == destinations,
                     "derived replay original source/reference/destination provenance changed")
        request = entry["reference"]["request_identity"]
        cfg = replace(
            copy.deepcopy(self.binding_cfg), seed=request["seed"],
            prepared_binding_seed=request["seed"], prepared_binding_source_compatibility=None,
            semantic_review_config=copy.deepcopy(
                request["parameters"]["experimental_semantic_review"]))
        context = PreparedBindingDerivationContext(
            sources=sources, items=[indexed[qa] for qa in ids],
            target_inputs=inputs, target_requests=requests,
            target_artifact_paths=paths, cfg=cfg, target_relocations=relocations,
            target_migration_receipts=migrations)
        # The approved immutable output replaces re-derivation as the expected
        # seal. Existing consumer validation still checks arrays, objects, and
        # every output field, with a stamp guard over actual physical files.
        context._replay_cache.update(
            fingerprint=_context_fingerprint(context), closure=closure, expected=expected)
        _require(_check_cache(context), "derived physical replay closure changed")
        return context, references, migrations

    def _derived_scene(self, entry):
        scene_id = entry["scene_id"]
        with self.cache_lock:
            if scene_id in self._derived_scenes:
                return self._derived_scenes[scene_id]
            lock = self._derivation_locks.setdefault(scene_id, threading.Lock())
        # A scene publishes once, while independent scenes may derive together.
        # Never retain the shared input-cache lock across expensive derivation.
        with lock:
            with self.cache_lock:
                if scene_id in self._derived_scenes:
                    return self._derived_scenes[scene_id]
            replay = (self._replay_derived_scene(entry)
                      if getattr(self, "_approved_derived_replay", None) is not None
                      else self._derive_scene(entry))
            with self.cache_lock:
                self._derived_scenes[scene_id] = replay
            return replay

    def prepare_published(self, variant, item):
        transport = self._transport(variant, item)
        entry, resolver = transport["entry"], transport["resolver"]
        if self._reuse_verified_publications:
            _require(self._approved_derived_replay is not None,
                     "verified derived replay requires existing published outputs")
            from skill3d.evaluation.verified_publication_v11 import ProducerVerifiedBindingContext
            from skill3d.storage.relocation import ProducerVerifiedRelocation
            approved = self._approved_derived_replay["scenes"][entry["scene_id"]]
            reference = approved["references"][entry["qa_id"]]
            store = PreparedBindingStore(self._physical_root(reference["store_root"]))
            migration, _ = self._rebase_migration(
                store, reference["key"], approved["migration_receipts"][entry["qa_id"]],
                transport["identity"], entry["publication_root"])
            resolver = ProducerVerifiedRelocation(migration)
            prepared = store.load_verified(reference["key"])
            context = ProducerVerifiedBindingContext(
                prepared, copy.deepcopy(reference["request_identity"]))
            return {
                "artifact_path": resolver.artifact_path,
                "artifact": resolver.resolved_artifact(),
                "identity": transport["identity"],
                "reference": copy.deepcopy(entry["reference"]),
                "derived_reference": copy.deepcopy(reference),
                "derived_migration_receipt": copy.deepcopy(migration),
                "binding_derivation_context": context,
                "prepared": prepared, "artifact_relocation": resolver,
                "transport_receipt": transport["receipt"],
            }
        context, references, migrations = self._derived_scene(entry)
        replaying = getattr(self, "_approved_derived_replay", None) is not None
        if replaying:
            _require(_check_cache(context), "derived physical replay closure changed")
        reference = references[entry["qa_id"]]
        prepared = load_prepared_binding_reference(
            reference, scene_input_identity=transport["identity"],
            scene_artifact=resolver.artifact_path,
            migration_receipt=migrations[entry["qa_id"]],
            migration_destination_root=transport["destination_root"],
            physical_store_root=(self._physical_root(reference["store_root"])
                                 if replaying else None))
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
