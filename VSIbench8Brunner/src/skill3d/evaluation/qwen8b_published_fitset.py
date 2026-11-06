"""Replay freshly prepared M5 bindings while retaining all original M4 anchors."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from skill3d.evaluation.published_fitset_v11 import PublishedFitSetInputProvider, _require
from skill3d.infra.m5_identity import content_sha256, file_identity
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore, load_prepared_binding_reference,
)

SCHEMA = "vsi-qwen8b-refreshed-bindings/1"


def read_manifest(path):
    value = json.loads(Path(path).read_text())
    _require(value.get("schema") == SCHEMA and value.get("status") == "complete",
             "Qwen8B M5 cache is incomplete; run prepare first")
    _require(value.get("sha256") == content_sha256({
        k: v for k, v in value.items() if k != "sha256"}),
        "Qwen8B M5 preparation manifest changed")
    return value


class Qwen8BPublishedFitSetInputProvider(PublishedFitSetInputProvider):
    """Read-only cache consumer; preflight never describes or resegments images."""
    def __init__(self, manifest_path, *, refresh_root, binding_cfg, max_scenes=2):
        super().__init__(manifest_path, max_scenes=max_scenes)
        self.refresh_root = Path(refresh_root).absolute()
        self.refresh_manifest_path = self.refresh_root / "refresh_manifest.json"
        self.refresh_identity = file_identity(self.refresh_manifest_path)
        self.refresh = read_manifest(self.refresh_manifest_path)
        _require(self.refresh["published_inputs_sha256"] == self._manifest_identity["sha256"],
                 "M5 cache belongs to different original inputs")
        _require(self.refresh["selected_qa_ids"] == sorted(self.entries),
                 "M5 cache experiment subset differs")
        _require(self.refresh["scene_scopes_sha256"] == content_sha256(self.scene_scopes),
                 "M5 cache scene audit scope differs")
        _require(self.refresh["semantic_review_config"] == binding_cfg.semantic_review_config,
                 "M5 cache semantic configuration differs")
        self._input_contract_identity["qwen8b_refresh"] = {
            "manifest": self.refresh_identity, "schema": SCHEMA,
            "new_model_requests": 0, "original_m4_preserved": True,
        }

    def prepare_published(self, variant, item):
        _require(file_identity(self.refresh_manifest_path) == self.refresh_identity,
                 "M5 refresh manifest changed after provider construction")
        transport = self._transport(variant, item)
        original = transport["entry"]
        qa_id = str(variant["qa_id"])
        row = self.refresh["bindings"][qa_id]
        reference = row["reference"]
        _require(row["source_reference_sha256"] == content_sha256(original["reference"])
                 and reference["scene_input_identity_sha256"] == transport["identity"]["sha256"]
                 and reference["request_identity"]["seed"] == original["reference"]["request_identity"]["seed"],
                 "refreshed binding source/input/seed changed")
        _require(Path(reference["store_root"]) == self.refresh_root / "stores" / original["scene_id"],
                 "refreshed binding store escapes its scene cache")
        resolver = transport["resolver"]
        prepared = load_prepared_binding_reference(
            reference, scene_input_identity=transport["identity"],
            migration_receipt=row["migration_receipt"],
            scene_artifact=resolver.artifact_path,
            migration_destination_root=transport["destination_root"])
        return {
            "artifact_path": resolver.artifact_path, "artifact": resolver.resolved_artifact(),
            "identity": transport["identity"], "reference": copy.deepcopy(original["reference"]),
            "prepared": prepared, "artifact_relocation": resolver,
            "transport_receipt": transport["receipt"],
            "derived_reference": copy.deepcopy(reference),
            "derived_migration_receipt": copy.deepcopy(row["migration_receipt"]),
            "binding_refresh_identity": {
                "manifest": self.refresh_identity, "reference_sha256": content_sha256(reference),
                "original_reference_sha256": content_sha256(original["reference"]),
            },
        }
