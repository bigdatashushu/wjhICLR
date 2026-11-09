#!/usr/bin/env python3
"""Read-only coverage catalog and SQLite online backups for an active run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

from skill3d.cluster import SQLiteJobQueue
from cluster_coordinator import atomic_json


def report(root: Path) -> dict:
    jobs = json.loads((root / "coordinator_jobs.json").read_text())
    queue = SQLiteJobQueue(root / "state/jobs.sqlite3")
    rows = {}
    errors = []
    questions = 0
    outcomes = {"base": {}, "metric": {}}
    metric = {}
    for spec in jobs["jobs"]:
        key = spec["job_key"]
        job = queue.get_job(key)
        scene = key.split("/")[-2]
        scope = spec["identity"]["manifest"]["scenes"][scene]
        questions += len(scope["qa_ids"])
        row = {
            "status": job["status"], "qa_ids": scope["qa_ids"], "split": scope["split"],
            "model_attempts": job["attempts"], "prefetch_failures": job["prefetch_failures"],
            "return_failures": job["return_failures"], "error_code": job["error_code"],
            "error_detail": job["error_detail"],
        }
        if job["status"] == "complete":
            digest = hashlib.sha256(key.encode()).hexdigest()
            path = root / "results" / digest / f"attempt-{job['attempts']}"
            validation = root / "receipts/scene_validation" / (digest + ".json")
            if not path.is_dir() or not validation.is_file():
                errors.append({"scene_id": scene, "error": "completed_output_missing"})
            else:
                valid = json.loads(validation.read_text())
                if valid["status"] != "validated" or valid["qa_ids"] != scope["qa_ids"]:
                    errors.append({"scene_id": scene, "error": "published_scope_mismatch"})
                row["output_root"] = str(path)
                row["validation_receipt"] = str(validation)
                row["receipt_sha256"] = job["completion"]["receipt_hash"]
                row["variants"] = {}
                for name, variant in valid["variants"].items():
                    row["variants"][name] = {
                        "artifact": str(path / variant["artifact"]),
                        "binding_manifest": str(path / variant["binding_manifest"]),
                        "outcomes": variant["outcomes"],
                        "metric_status": variant["metric_status"],
                    }
                    for qa_id, outcome in variant["outcomes"].items():
                        if qa_id in outcomes[name]:
                            errors.append({"qa_id": qa_id, "error": "duplicate_question"})
                        outcomes[name][qa_id] = outcome
                    if name == "metric":
                        metric[scene] = variant["metric_status"]
        rows[scene] = row
    progress = queue.progress_summary()
    counts = progress["counts"]
    terminal = sum(counts[name] for name in ("complete", "failed", "skipped"))
    result = {
        "schema": "harness3d-cluster-coverage/1", "updated_at": time.time(),
        "identity_hash": jobs["identity"]["identity_hash"],
        "planned_scenes": len(rows), "planned_questions": questions,
        "queue": progress, "terminal_scenes": terminal,
        "completed_questions": len(outcomes["base"]),
        "metric_completed_questions": len(outcomes["metric"]),
        "metric_success_scenes": sum(value == "success" for value in metric.values()),
        "metric_failed_scenes": sum(value == "failed" for value in metric.values()),
        "scenes": rows, "integrity_errors": errors,
        "status": (
            "completed" if counts["complete"] == len(rows) and not errors else
            "terminal_incomplete" if terminal == len(rows) else "running"),
        "solver_ran": False, "scoring_ran": False, "real_labels_available": False,
        "disk_free_bytes": shutil.disk_usage(root).free,
    }
    atomic_json(root / "deliverables/coverage.json", result)
    return result


def backup(root: Path) -> None:
    destination = root / "backups"
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / "queue-online.partial"
    source = sqlite3.connect(root / "state/jobs.sqlite3")
    target = sqlite3.connect(temporary)
    try:
        source.backup(target)
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        target.execute("PRAGMA journal_mode=DELETE")
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite online backup integrity failed")
    finally:
        target.close()
        source.close()
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(destination / "queue-online.sqlite3")
    atomic_json(destination / "queue-online-receipt.json", {
        "schema": "harness3d-sqlite-backup/1", "created_at": time.time(),
        "method": "SQLite backup API", "integrity_check": "ok",
        "database_bytes": (destination / "queue-online.sqlite3").stat().st_size,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    last_backup = -1000
    while True:
        result = report(args.root)
        completed = result["queue"]["counts"]["complete"]
        if completed - last_backup >= 10 or result["status"] != "running":
            backup(args.root)
            last_backup = completed
        print(json.dumps({k: v for k, v in result.items() if k not in {"scenes"}}), flush=True)
        if not args.watch or result["status"] != "running":
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
