"""Prepare the original scene audit groups with the new pre-review description."""
from __future__ import annotations

import copy
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import time

from skill3d.evaluation.published_fitset_v11 import PublishedFitSetInputProvider, _require
from skill3d.evaluation.qwen8b_published_fitset import SCHEMA, read_manifest
from skill3d.evolution.prepared_bindings_v11 import prepare_shared_bindings_v11
from skill3d.infra.m5_identity import content_sha256, file_identity, source_identity
from skill3d.infra.vsi_preparation_budget import ENV, reserve
from skill3d.online import runner
from skill3d.routing.task_classifier import canonical_task
from skill3d.segmentation.prepared_binding_store import PreparedBindingStore


def write_once(path, value):
    path = Path(path)
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        _require(json.loads(path.read_text()) == value, "preparation identity changed")
        return
    with path.open("x") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


class PreparationClient:
    """Count the original object-parser calls without changing their requests."""
    def __init__(self, client):
        if client is None:
            raise ValueError("original M5 parser service is not configured")
        self.client = client

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, messages, **kwargs):
        reserve("original_m5_parser", {"messages": messages, "kwargs": kwargs})
        return self.client.chat(messages, **kwargs)


def validate_outcome(stats):
    if stats.get("binding_error") or stats.get("detector_fault"):
        raise RuntimeError("M5 preparation operational failure")
    if (stats.get("object_parse") or {}).get("status") == "unavailable":
        raise RuntimeError("M5 object parser unavailable")
    for row in stats.get("candidate_receipts", []):
        visual = row.get("semantic_visual") or {}
        if visual and visual.get("status") != "observed":
            raise RuntimeError("Qwen description or binary reviewer unavailable")
        # SAM failures precede semantic review. Do not silently certify a partial scene.
        if row.get("error") and not visual:
            raise RuntimeError("M5 candidate preparation unavailable")
    return True


def prepare(published_inputs, cache_root, cfg, *, max_requests, min_free_gib=80):
    provider = PublishedFitSetInputProvider(published_inputs, max_scenes=1)
    root = Path(cache_root).absolute()
    _require(not root.is_relative_to(Path(published_inputs).parent.resolve()),
             "new M5 cache must be outside the original dataset directory")
    root.mkdir(parents=True, exist_ok=True)
    _require("target_description" in (cfg.semantic_review_config or {}),
             "Qwen description configuration is required")
    _require(type(max_requests) is int and max_requests > 0,
             "preparation requires a positive request cap")
    identity = {
        "schema": SCHEMA, "published_inputs_sha256": file_identity(published_inputs)["sha256"],
        "selected_qa_ids": sorted(provider.entries),
        "scene_scopes_sha256": content_sha256(provider.scene_scopes),
        "semantic_review_config": cfg.semantic_review_config,
        "m5_sources_sha256": content_sha256(source_identity(Path(__file__).resolve().parents[1])),
        "m5_runtime_sha256": content_sha256({name: getattr(cfg, name) for name in (
            "online_model_enabled", "online_model_provider", "online_model",
            "online_model_revision", "online_base_url", "temperature", "max_tokens",
            "semantic_min_pixels", "semantic_max_pixels", "semantic_relation_batch_size")}),
        "max_requests": max_requests,
    }
    final = root / "refresh_manifest.json"
    if final.exists():
        existing = read_manifest(final)
        _require(all(existing[k] == v for k, v in identity.items()),
                 "completed cache preparation identity changed")
        return existing
    with (root / "prepare.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_once(root / "preparation_identity.json", identity)
        budget_path = root / "request_budget.json"
        if not budget_path.exists():
            write_once(budget_path, {"max_requests": max_requests, "used": 0, "requests": []})
        (root / "stores").mkdir(exist_ok=True)
        (root / "scenes").mkdir(exist_ok=True)
        (root / "trajectories").mkdir(exist_ok=True)
        previous_budget = os.environ.get(ENV)
        os.environ[ENV] = str(budget_path)
        bindings = {}
        try:
            for scene_id, scope in sorted(provider.scene_scopes.items()):
                scene_receipt = root / "scenes" / f"{scene_id}.json"
                if scene_receipt.exists():
                    cached = json.loads(scene_receipt.read_text())
                    _require(cached["preparation_identity_sha256"] == content_sha256(identity),
                             "scene cache preparation identity differs")
                    bindings.update(cached["bindings"])
                    continue
                first_id = next(qa for qa in scope["qa_ids"] if qa in provider.entries)
                scene = provider._scene(provider.entries[first_id])
                items = {str(item.episode.qa_id): item for item in scene.items}
                _require(set(items) == set(scope["qa_ids"]), "full original scene group differs")
                transports, paths = {}, {}
                for qa_id in sorted(items):
                    source = provider._source_entries[qa_id]
                    item = items[qa_id]
                    _require(item.episode.ground_truth == "", "preparation input contains an answer")
                    variant = {
                        "qa_id": qa_id, "scene_id": scene_id, "split": source["split"],
                        "source_split": source["split"],
                        "artifact_ref": source["reference"]["scene_artifact_ref"],
                        "question_type": canonical_task(item.episode.question_type),
                    }
                    transports[qa_id] = provider._transport_entry(variant, item, source)
                    paths[qa_id] = transports[qa_id]["resolver"].artifact_path
                seeds = {provider._source_entries[qa]["reference"]["request_identity"]["seed"]
                         for qa in items}
                _require(len(seeds) == 1, "original scene M5 seeds differ")
                frozen = replace(copy.deepcopy(cfg), seed=next(iter(seeds)))
                def identify(item, path):
                    return copy.deepcopy(transports[str(item.episode.qa_id)]["identity"])
                art_by_path = {str(path): transports[qa]["resolver"].resolved_artifact()
                               for qa, path in paths.items()}
                manifest = prepare_shared_bindings_v11(
                    [items[qa] for qa in sorted(items)], artifact_paths=paths,
                    base_cfg=frozen, store_root=root / "stores" / scene_id,
                    client_factory=lambda _: PreparationClient(runner._make_online_client(frozen)),
                    max_questions_per_scene=128, min_free_gib=min_free_gib,
                    identity_factory=identify,
                    artifact_loader=lambda path: copy.deepcopy(art_by_path[str(path)]),
                    outcome_validator=validate_outcome,
                    scene_migration_factory=lambda item: transports[
                        str(item.episode.qa_id)]["migration"],
                    trace_root=root / "trajectories" / "raw")
                store = PreparedBindingStore(root / "stores" / scene_id)
                scene_bindings = {}
                for qa_id, reference in manifest["bindings"].items():
                    transport = transports[qa_id]
                    migration = transport["migration"]
                    new_migration = store.create_migration_receipt(
                        reference["key"], scene_input_identity=transport["identity"],
                        destination_root=transport["destination_root"],
                        scene_artifact_ref=migration["artifact"]["target_ref"],
                        dependency_refs={name: row["target_ref"]
                                         for name, row in migration["dependencies"].items()})
                    scene_bindings[qa_id] = {
                        "reference": reference, "migration_receipt": new_migration,
                        "source_reference_sha256": content_sha256(
                            provider._source_entries[qa_id]["reference"]),
                    }
                    # Preserve all candidate receipts, including rejection and Qwen raw responses.
                    prepared = store.load(
                        reference["key"], expected_request_identity=reference["request_identity"],
                        scene_artifact=paths[qa_id],
                        scene_input_identity=transport["identity"], migration_receipt=new_migration,
                        migration_destination_root=transport["destination_root"])
                    write_once(root / "trajectories" / f"{qa_id}.json", {
                        "qa_id": qa_id, "scene_id": scene_id,
                        "reference_answer_visible": False,
                        "object_parse": prepared.stats.get("object_parse"),
                        "candidate_receipts": prepared.stats.get("candidate_receipts", []),
                        "semantic_review": prepared.stats.get("semantic_review", {}),
                        "generation_identity": prepared.stats.get("generation"),
                    })
                receipt = {
                    "scene_id": scene_id, "preparation_identity_sha256": content_sha256(identity),
                    "bindings": scene_bindings, "full_audit_qa_ids": sorted(items),
                }
                write_once(scene_receipt, receipt)
                bindings.update(scene_bindings)
                print(json.dumps({"scene_id": scene_id, "status": "prepared",
                                  "completed_bindings": len(bindings)}), flush=True)
            body = {**identity, "status": "complete", "bindings": bindings,
                    "n_questions": len(provider.entries), "n_audit_questions": len(bindings),
                    "n_scenes": len(provider.scene_scopes), "completed_at": time.time()}
            write_once(final, {**body, "sha256": content_sha256(body)})
            return read_manifest(final)
        finally:
            if previous_budget is None:
                os.environ.pop(ENV, None)
            else:
                os.environ[ENV] = previous_budget
