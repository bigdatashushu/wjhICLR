#!/usr/bin/env python3
"""Read scalar eligibility from pinned publications while payload audit runs."""

import argparse
import json
from pathlib import Path
import sys

from audit_metric_coverage import checked, checked_identity, load_json, make_report

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--inventory", type=Path, required=True)
p.add_argument("--catalog", type=Path, required=True)
p.add_argument("--source-root", type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.source_root))
from skill3d.reconstruction.metric_evidence import digest
from skill3d.reconstruction_gate.evidence_profile import evaluate_metric_gate
from skill3d.reconstruction_gate.scene_state import scene_route_from_quality
from skill3d.schemas.evidence import metric_scale_state_from_gate
from skill3d.schemas.reconstruction import ReconstructionArtifact

inventory, _ = load_json(a.inventory)
catalog, _ = load_json(a.catalog, inventory["anchors"]["catalog_sha256"])
checked(catalog["status"] == "verified" and catalog["no_labels"] is True
        and catalog["identity_hash"] == inventory["anchors"]["catalog_identity_hash"],
        "catalog_identity_changed")
entries = catalog["variants"]["metric"]
rows, scenes = [], []
for scene in inventory["scenes"]:
    if not scene["m4_passed"]:
        continue
    first = entries[scene["qa_ids"][0]]
    validation, _ = load_json(first["migration_receipt_file"],
                              first["migration_receipt_file_sha256"])
    identity, migration, _ = checked_identity(first, validation)
    artifact_data = Path(first["scene_artifact"]).read_bytes()
    import hashlib
    checked(hashlib.sha256(artifact_data).hexdigest() == identity["artifact_json_sha256"],
            "artifact_hash_changed")
    art = ReconstructionArtifact.model_validate_json(artifact_data)
    recpath = Path(first["publication_root"]) / migration["dependencies"][
        "per_frame_scale_ref"]["target_ref"]
    rec, _ = load_json(recpath, art.metric_receipt_sha256)
    checked(rec["request_sha256"] == digest(rec["request"]) == art.metric_request_sha256
            and rec["request"]["input_identity"] == art.metric_input_identity
            == art.reconstruction_input_identity, "metric_request_identity_changed")
    for receipt_key, artifact_key in (
        ("status", "scale_fusion_status"), ("metric_scale", "metric_scale"),
        ("scale_self_consistency", "scale_self_consistency"),
        ("metric_model", "metric_model"), ("metric_fusion_version", "metric_fusion_version")):
        checked(rec[receipt_key] == getattr(art, artifact_key), "metric_scalar_changed")
    checked(rec["n_frames_total"] == len(art.frame_ids)
            and rec["valid_frame_ratio"] == rec["n_frames_valid"] / rec["n_frames_total"],
            "metric_frame_ratio_changed")
    gate = evaluate_metric_gate(
        scene_route=scene_route_from_quality(
            art.quality, quality_status=art.quality_status,
            quality_metric_version=art.quality_metric_version),
        main_gate_passed=bool(getattr(art.quality, "main_gate_passed", False)),
        fusion_status=art.scale_fusion_status,
        scale_self_consistency=art.scale_self_consistency,
        valid_frame_ratio=rec["valid_frame_ratio"], metric_scale=art.metric_scale)
    status = "pass" if metric_scale_state_from_gate(gate) == "available" else "fail"
    scenes.append({"scene_id": scene["scene_id"], "status": status})
    for qa in scene["qa_ids"]:
        _, _, public = checked_identity(entries[qa], validation)
        rows.append({**public, "status": status})
report = make_report(inventory, rows, scenes, scanned=True)
report["status"] = "published_receipt_preview"
report["payload_hashes_recomputed"] = False
print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
