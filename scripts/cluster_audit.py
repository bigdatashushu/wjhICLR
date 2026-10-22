#!/usr/bin/env python3
"""Independently replay immutable publications without touching the live queue."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from skill3d.cluster.publication import inspect_receipt, validate_artifacts
from skill3d.cluster.scene_compute import validate_published_scene
from cluster_coordinator import atomic_json


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def completed_rows(root: Path) -> dict:
    """Open the coordinator database only in SQLite read/query-only mode."""
    uri = (root / "state/jobs.sqlite3").resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    try:
        connection.execute("PRAGMA query_only=ON")
        return {
            key: {"attempt": attempt, "receipt_sha256": receipt_hash}
            for key, attempt, receipt_hash in connection.execute(
                "SELECT j.job_key,j.attempts,c.receipt_hash FROM jobs j "
                "JOIN completions c ON c.job_key=j.job_key WHERE j.status='complete'"
            )
        }
    finally:
        connection.close()


def audit_scene(root: Path, spec: dict, row: dict, db_row: dict) -> dict:
    key = spec["job_key"]
    scene = key.split("/")[-2]
    digest = hashlib.sha256(key.encode()).hexdigest()
    scope = spec["identity"]["manifest"]["scenes"][scene]
    output = root / "results" / digest / f"attempt-{db_row['attempt']}"
    receipt_hash = db_row["receipt_sha256"]
    if (
        row["model_attempts"] != db_row["attempt"]
        or row["receipt_sha256"] != receipt_hash
        or Path(row["output_root"]) != output
        or row["qa_ids"] != scope["qa_ids"]
        or row["split"] != scope["split"]
    ):
        raise ValueError("coverage differs from the committed queue publication")
    publication = inspect_receipt(
        output / "publication.json", expected_receipt_hash=receipt_hash
    )
    receipt = publication.receipt
    if (
        receipt.job_key != key
        or receipt.completed_key != spec["completed_key"]
        or receipt.identity_hash != spec["identity"]["identity_hash"]
        or receipt.attempt != db_row["attempt"]
        or receipt.stage != spec["stage"]
        or receipt.validator != "harness3d-scene-qa-validator-v1"
    ):
        raise ValueError("publication receipt differs from the frozen job")
    validate_artifacts(publication, output)
    validation_path = root / "receipts/scene_validation" / (digest + ".json")
    if Path(row["validation_receipt"]) != validation_path:
        raise ValueError("coverage validation receipt names an unexpected destination")
    validation = validate_published_scene(
        output,
        expected_qa_ids=scope["qa_ids"],
        expected_scene_id=scene,
        expected_method_config=spec["identity"]["method_config"],
    )
    if validation != read_json(validation_path):
        raise ValueError("independent loader replay differs from the publication validation")
    provenance = read_json(output / "source_provenance.json")
    variants = {}
    for name, variant in validation["variants"].items():
        artifact = read_json(output / variant["artifact"])
        variants[name] = {
            "outcomes": variant["outcomes"],
            "metric_status": variant["metric_status"],
            "world_frame_status": artifact.get("world_frame_status"),
            "quality_main_gate_passed": (artifact.get("quality") or {}).get("main_gate_passed"),
        }
    return {
        "schema": "harness3d-independent-scene-audit/1",
        "status": "verified",
        "checked_at": time.time(),
        "job_key": key,
        "identity_hash": spec["identity"]["identity_hash"],
        "scene_id": scene,
        "split": scope["split"],
        "qa_ids": scope["qa_ids"],
        "attempt": db_row["attempt"],
        "publication_receipt_sha256": receipt_hash,
        "loader_validation_receipt_sha256": file_sha256(validation_path),
        "artifact_files": len(receipt.artifacts),
        "artifact_bytes": sum(item.size_bytes for item in receipt.artifacts),
        "geometry_source": provenance["geometry_source"],
        "variants": variants,
        "solver_ran": False,
        "scoring_ran": False,
        "real_labels_available": False,
    }


def audit(root: Path, *, recheck: bool = False) -> dict:
    jobs = read_json(root / "coordinator_jobs.json")
    coverage = read_json(root / "deliverables/coverage.json")
    identity_hash = jobs["identity"]["identity_hash"]
    if coverage["identity_hash"] != identity_hash:
        raise ValueError("coverage run identity changed")
    database = completed_rows(root)
    destination = root / "audits"
    rows = {}
    errors = []
    names = ("base", "metric") if jobs["identity"]["method_config"].get("metric_enabled") else ("base",)
    planned = {name: set() for name in names}
    observed = {name: set() for name in names}
    outcomes = {name: Counter() for name in names}
    metric = Counter()
    geometry = Counter()
    splits = {}
    for spec in jobs["jobs"]:
        key = spec["job_key"]
        scene = key.split("/")[-2]
        scope = spec["identity"]["manifest"]["scenes"][scene]
        for name in planned:
            if planned[name].intersection(scope["qa_ids"]):
                raise ValueError("frozen jobs contain duplicate question IDs")
            planned[name].update(scope["qa_ids"])
        splits.setdefault(scope["split"], {"planned_scenes": 0, "verified_scenes": 0,
                                          "planned_questions": 0, "verified_questions": 0})
        splits[scope["split"]]["planned_scenes"] += 1
        splits[scope["split"]]["planned_questions"] += len(scope["qa_ids"])
        row = coverage["scenes"][scene]
        if row["status"] != "complete":
            continue
        try:
            db_row = database[key]
            path = destination / "scenes" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
            verified = read_json(path) if path.is_file() and not recheck else None
            if not (
                verified
                and verified.get("status") == "verified"
                and verified.get("identity_hash") == identity_hash
                and verified.get("publication_receipt_sha256") == db_row["receipt_sha256"]
                and verified.get("attempt") == db_row["attempt"]
                and verified.get("qa_ids") == scope["qa_ids"]
                and verified.get("loader_validation_receipt_sha256")
                == file_sha256(Path(row["validation_receipt"]))
            ):
                verified = audit_scene(root, spec, row, db_row)
                atomic_json(path, verified)
            rows[scene] = verified
            splits[scope["split"]]["verified_scenes"] += 1
            splits[scope["split"]]["verified_questions"] += len(scope["qa_ids"])
            geometry[verified["geometry_source"]] += 1
            for name, variant in verified["variants"].items():
                if name not in observed or set(variant["outcomes"]) != set(scope["qa_ids"]):
                    raise ValueError("audited QA coverage differs from the frozen variant scope")
                observed[name].update(variant["outcomes"])
                outcomes[name].update(value["status"] for value in variant["outcomes"].values())
                if name == "metric":
                    metric[variant["metric_status"]] += 1
        except Exception as exc:
            errors.append({"scene_id": scene, "error_type": type(exc).__name__,
                           "detail": str(exc)})
    complete = (
        coverage["status"] == "completed"
        and len(rows) == len(jobs["jobs"])
        and all(observed[name] == planned[name] for name in observed)
        and not errors
        and not coverage["integrity_errors"]
    )
    result = {
        "schema": "harness3d-independent-run-audit/1",
        "updated_at": time.time(),
        "coverage_updated_at": coverage["updated_at"],
        "identity_hash": identity_hash,
        "status": "verified" if complete else "audit_errors" if errors else "running",
        "planned_scenes": len(jobs["jobs"]),
        "planned_questions": len(planned["base"]),
        "published_scenes": coverage["queue"]["counts"]["complete"],
        "verified_scenes": len(rows),
        "verified_questions": {name: len(values) for name, values in observed.items()},
        "question_outcomes": {name: dict(values) for name, values in outcomes.items()},
        "metric_status_scenes": dict(metric),
        "geometry_source_scenes": dict(geometry),
        "artifact_bytes": sum(row["artifact_bytes"] for row in rows.values()),
        "artifact_files": sum(row["artifact_files"] for row in rows.values()),
        "splits": splits,
        "queue": coverage["queue"],
        "errors": errors,
        "coverage_integrity_errors": coverage["integrity_errors"],
        "solver_ran": False,
        "scoring_ran": False,
        "real_labels_available": False,
    }
    atomic_json(destination / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--recheck", action="store_true")
    args = parser.parse_args()
    while True:
        result = audit(args.root.resolve(), recheck=args.recheck)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        if not args.watch or result["status"] == "verified":
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
