#!/usr/bin/env python3
"""Retain a pre-model source-cache failure after the owned pipeline has drained."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def archive(root, base, name, qa):
    launch = read(base / f"{name}.pid.json")
    proc = Path("/proc") / str(launch["pid"])
    if proc.exists():
        stat = proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        if stat[19] != launch["process_identity"]["start_ticks"] or stat[0] != "Z":
            raise ValueError("previous owned pipeline has not exited")
    status = read(root / "live_status.json")
    if status["status"] != "stopped" or status.get("active_qwen_requests"):
        raise ValueError("source recovery requires a stopped pipeline")
    for path in (root / "solver_requests").glob("*/*.json"):
        if read(path)["status"] != "returned":
            raise ValueError("an issued solver request is unfinished")
    if list((root / "solver_requests" / qa).glob("*.json")):
        raise ValueError("source recovery cannot repeat an issued solver episode")
    errors = list((root / "host_errors").glob("*.json"))
    if [path.stem for path in errors] != [qa]:
        raise ValueError("source recovery has unexpected host failures")
    error = read(errors[0])
    failed = read(root / "bindings" / f"{qa}.json")
    consumer_failure = (
        error["stage"] == "solve" and error["error_type"] == "ValueError"
        and (error.get("stack") or [{}])[-1].get("function") == "validate_prepared_binding"
        and not failed["materialized"] and not failed["objects"]
        and (failed["stats"].get("object_parse") or {}).get("status") == "unavailable"
        and not failed["stats"].get("candidate_receipts")
        and not failed["stats"].get("detection_receipts")
        and not failed["stats"].get("binding_error")
        and not failed["stats"].get("detector_fault"))
    if not (consumer_failure or (
            error["stage"] == "prepare" and failed["stats"].get("binding_error")
            == "ValueError: cached parser was not successful")):
        raise ValueError("source recovery failure reason differs")
    if read(root / "sam2_receipts" / f"{qa}.json"):
        raise ValueError("source recovery cannot repeat issued SAM2 calls")
    preflight = read(root / "reuse_preflight.json")
    source = next(row for row in preflight["questions"] if row["qa_id"] == qa)
    source_path = Path(source["binding_file"])
    if digest(source_path) != source["binding_file_sha256"]:
        raise ValueError("source binding changed after preflight")
    original = read(source_path)
    if (original["materialized"] or original["objects"]
            or (original["stats"].get("object_parse") or {}).get("status") != "unavailable"
            or original["stats"].get("candidate_receipts")
            or original["stats"].get("detection_receipts")):
        raise ValueError("source recovery requires original missing object evidence")
    if consumer_failure:
        if failed["stats"]["generation"] != original["stats"]["generation"]:
            raise ValueError("consumer failure changed original missing-evidence identity")
        scene_audit = root / "scene_audits" / f"{source['scene_id']}.json"
        if read(scene_audit)["refreshed_qa_ids"] != [qa]:
            raise ValueError("consumer recovery requires an isolated scene audit")
    settings = read(root / "parallel_execution.json")
    history_path = root / "execution_resume_history.json"
    history = read(history_path) if history_path.exists() else {
        "schema": "pilot-execution-resumes/1", "epochs": [],
        "reference_answers_read": False,
    }
    if any(row["name"] == name for row in history["epochs"]):
        raise ValueError("source failure has already been archived")
    destination = root / "attempt_history" / f"source-parser-unavailable-{qa}-20261105"
    if consumer_failure:
        destination = root / "attempt_history" / f"source-parser-unavailable-{qa}-{name}"
    if destination.exists():
        raise ValueError("source archive destination already exists")
    paths = [
        root / "bindings" / f"{qa}.json",
        root / "host_errors" / f"{qa}.json",
        root / "sam2_receipts" / f"{qa}.json",
        root / "trajectories" / f"{qa}.json",
        *sorted((root / "preparation" / qa).glob("*.json")),
    ]
    if consumer_failure:
        paths.extend([scene_audit, *sorted((root / "solver_trace" / qa).glob("*.jsonl"))])
    paths = [path for path in paths if path.exists()]
    identities = [(path, digest(path), path.stat().st_size) for path in paths]
    destination.mkdir(parents=True)
    records = []
    for path, sha, size in identities:
        relative = path.relative_to(root)
        archived = destination / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        path.rename(archived)
        if archived.stat().st_size != size or digest(archived) != sha:
            raise ValueError("archived source failure identity differs")
        records.append({"original_file": str(relative),
                        "file": str(archived.relative_to(root)), "sha256": sha, "bytes": size})
    epoch = {
        "name": name, "captured_at": time.time(), "final_status": status,
        "settings": settings, "process_identity": launch["process_identity"],
        "reason": ("source_parser_unavailable_legacy_consumer_identity"
                   if consumer_failure else "source_parser_unavailable_with_zero_candidates"),
        "failed_qa_id": qa, "archived_files": records,
        "preserved_outcomes": sorted(path.stem for path in (root / "outcomes").glob("*.json")),
        "preserved_outcome_sha256": {
            path.stem: digest(path) for path in sorted((root / "outcomes").glob("*.json"))},
        "issued_solver_requests_repeated": 0,
    }
    history["epochs"].append(epoch)
    temporary = history_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(history, indent=2) + "\n")
    temporary.replace(history_path)
    print(json.dumps({"status": "source_failure_archived", "qa_id": qa,
                      "preserved_outcomes": len(epoch["preserved_outcomes"]),
                      "archived_files": len(records), "issued_solver_requests_repeated": 0}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--base-dir", required=True, type=Path)
    parser.add_argument("--name", default="pilot40-parallel01")
    parser.add_argument("--qa-id", default="5060")
    args = parser.parse_args()
    archive(args.run_dir, args.base_dir, args.name, args.qa_id)


if __name__ == "__main__":
    main()
