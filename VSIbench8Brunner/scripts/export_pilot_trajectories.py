#!/usr/bin/env python3
"""Export label-free per-question JSON snapshots, including incomplete attempts."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"$nonfinite": "nan" if math.isnan(value) else
                "positive_infinity" if value > 0 else "negative_infinity"}
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for name in ("DASHSCOPE_API_KEY", "QWEN3VL8B_API_KEY", "VSI_SAM2_TOKEN"):
        secret = os.environ.get(name, "")
        if secret and secret in text:
            raise ValueError("trajectory contains credentials")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def recorded_path(root, value):
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    elif not path.is_relative_to(root):
        marker = "/" + root.name + "/"
        if marker not in str(path):
            raise ValueError("preparation request is outside this run")
        path = root / str(path).split(marker, 1)[1]
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("preparation request escapes this run")
    return path


def records(paths, root):
    return [{"file": str(path.relative_to(root)), "record": read_json(path)}
            for path in sorted(paths)]


def export_question(root, qa_id, *, status=None, error=None, incremental=False):
    """Journal callbacks stay small; completed/failed snapshots are self-contained."""
    root = Path(root).absolute()
    qa_id = str(qa_id)
    if not qa_id or any(char not in "0123456789" for char in qa_id):
        raise ValueError("invalid pilot question identifier")
    lock_dir = root / ".trajectory-locks"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f"{qa_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if incremental:
            paths = [p for kind in ("solver_requests", "parser_requests", "program_executions")
                     for p in (root / kind / qa_id).glob("*.json")]
            destination = root / "trajectories" / "live" / f"{qa_id}.json"
            write_json(destination, {
                "schema": "vsi8b-live-trajectory-journal/1", "qa_id": qa_id,
                "saved_at": time.time(), "reference_answer_visible": False,
                "automatic_retry": False, "status": status or "in_progress", "error": error,
                "journal_files": [
                    {"file": str(p.relative_to(root)), "bytes": p.stat().st_size}
                    for p in sorted(paths)],
                "binding_file": f"bindings/{qa_id}.json",
                "preparation_directory": f"preparation/{qa_id}",
                "trace_directory": f"solver_trace/{qa_id}",
                "full_snapshot_file": f"trajectories/{qa_id}.json",
            })
            return destination
        return _export_question(root, qa_id, status=status, error=error)


def _export_question(root, qa_id, *, status=None, error=None):
    root = Path(root).absolute()
    qa_id = str(qa_id)
    sampling = read_json(root / "sampling_manifest.json")
    question = next(row for row in sampling["questions"] if str(row["qa_id"]) == qa_id)
    if any(question.get(key) not in (None, "") for key in (
            "ground_truth", "reference_answer", "label", "answer")):
        raise ValueError("online trajectory input contains a reference answer")
    paths = {
        "solver_requests": sorted((root / "solver_requests" / qa_id).glob("*.json")),
        "parser_requests": sorted((root / "parser_requests" / qa_id).glob("*.json")),
        "program_executions": sorted((root / "program_executions" / qa_id).glob("*.json")),
    }
    outcome_path = root / "outcomes" / f"{qa_id}.json"
    result = read_json(outcome_path) if outcome_path.exists() else None
    binding_path = root / "bindings" / f"{qa_id}.json"
    binding = read_json(binding_path) if binding_path.exists() else None
    sam_path = root / "sam2_receipts" / f"{qa_id}.json"
    runtime_path = root / "runtime_identity.json"
    host_error_path = root / "host_errors" / f"{qa_id}.json"
    if error is None and host_error_path.exists():
        error = read_json(host_error_path)
    reviewer_requests, missing_requests = [], []
    reviewer_paths = []
    for candidate in (binding or {}).get("stats", {}).get("candidate_receipts", []):
        visual = candidate.get("semantic_visual") or {}
        if not visual.get("request_file"):
            continue
        path = recorded_path(root, visual["request_file"])
        if not path.exists():
            missing_requests.append(str(path.relative_to(root)))
            continue
        reviewer_paths.append(path)
        reviewer_requests.append({
            "candidate_id": candidate["candidate_id"],
            "file": str(path.relative_to(root)), "request": read_json(path),
        })
    topics, topic_paths, incomplete_lines = {}, [], []
    trace_paths = [*(root / "solver_trace").glob("*.jsonl"),
                   *(root / "solver_trace" / qa_id).glob("*.jsonl")]
    for path in sorted(trace_paths):
        rows = []
        for index, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                incomplete_lines.append({"file": str(path.relative_to(root)), "line": index})
                continue
            if str(row.get("qa_id", row.get("episode_id", ""))) == qa_id:
                rows.append(row)
        if rows:
            topics.setdefault(path.stem, []).extend(rows)
            topic_paths.append(path)
    solver = records(paths["solver_requests"], root)
    executions = records(paths["program_executions"], root)
    previous_attempt_paths = sorted({
        path for directory in (root / "attempt_history").glob("*")
        if qa_id in directory.stem.split("-")
        for path in ([directory] if directory.is_file() and directory.suffix == ".json"
                     else directory.rglob("*.json"))
    })
    outcome = (result or {}).get("outcome") or {}
    latest_ledger = (executions[-1]["record"].get("episode_ledger") or {}) if executions else {}
    authorizations = outcome.get("authorization_receipts") or latest_ledger.get(
        "authorization_receipts") or [
            receipt for execution in executions
            for receipt in (execution["record"].get("authorization_receipts") or [
                result["authorization"] for result in (
                    execution["record"].get("program_trace") or {}).get("results", [])
                if result.get("authorization")])
        ]
    observation_reads = latest_ledger.get("observation_reads") or [
        page for round_ in outcome.get("rounds", []) for page in round_.get("observation_reads", [])
    ]
    operational_errors = [
        {key: row["record"].get(key) for key in (
            "request_index", "error_type", "http_status", "elapsed_s", "transport_timeout_s")}
        for row in solver if row["record"].get("status") == "unavailable"
    ]
    if status is None:
        if error or operational_errors:
            status = "failed"
        elif result:
            status = "finished"
        else:
            status = "solving" if solver else "prepared" if binding else "preparing"
    source_paths = [
        root / "sampling_manifest.json", runtime_path, binding_path, outcome_path, sam_path,
        host_error_path,
        *[path for group in paths.values() for path in group], *reviewer_paths, *topic_paths,
        *previous_attempt_paths,
    ]
    source_paths = sorted({path for path in source_paths if path.exists()})
    asset_paths = sorted((root / "preparation" / qa_id).rglob("*.npz"))
    fingerprint = hashlib.sha256(json.dumps({
        "status": status, "error": error, "missing_requests": missing_requests,
        "incomplete_lines": incomplete_lines,
        "sources": [(str(path.relative_to(root)), path.stat().st_size, path.stat().st_mtime_ns)
                    for path in [*source_paths, *asset_paths]],
    }, sort_keys=True).encode()).hexdigest()
    destination = root / "trajectories" / f"{qa_id}.json"
    index_path = root / "trajectory_index.json"
    # Only index read/modify/write is shared; large exports use their own QA lock.
    with (root / ".trajectory-export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = read_json(index_path) if index_path.exists() else {"questions": {}}
        previous = index["questions"].get(qa_id, {})
    if previous.get("source_fingerprint") == fingerprint and destination.exists():
        return destination
    data = {
        "schema": "vsi8b-full-question-trajectory/1",
        "qa_id": qa_id, "run_id": root.name, "status": status, "saved_at": time.time(),
        "reference_answer_visible": False, "automatic_retry": False,
        "nonfinite_encoding": {"$nonfinite": "nan|positive_infinity|negative_infinity"},
        "question": question, "runtime_identity": read_json(runtime_path) if runtime_path.exists() else None,
        "native_solver_requests": solver,
        "native_parser_requests": records(paths["parser_requests"], root),
        "program_executions": executions,
        "rounds": outcome.get("rounds", []),
        "authorization_receipts": authorizations,
        "observation_reads": observation_reads,
        "trace_topics": topics,
        "previous_attempts": records(previous_attempt_paths, root),
        "preparation": {
            "binding": binding, "sam2_receipts": read_json(sam_path) if sam_path.exists() else None,
            "reviewer_requests": reviewer_requests,
            "mask_assets": [{"file": str(path.relative_to(root)), "bytes": path.stat().st_size,
                             "sha256": digest(path)} for path in asset_paths],
        },
        "result": result, "errors": {"host_error": error, "operational": operational_errors},
        "completeness": {
            "outcome_written": result is not None,
            "all_model_requests_returned": bool(solver) and all(
                row["record"].get("status") == "returned" for row in solver),
            "missing_reviewer_request_files": missing_requests,
            "incomplete_jsonl_lines": incomplete_lines,
            "tool_calls": sum(len((row["record"].get("program_trace") or {}).get("calls", []))
                              for row in executions),
            "tool_results": sum(len((row["record"].get("program_trace") or {}).get("results", []))
                                for row in executions),
        },
        "source_files": [{"file": str(path.relative_to(root)), "bytes": path.stat().st_size,
                          "sha256": digest(path)} for path in source_paths],
    }
    write_json(destination, data)
    question_index = {
        "file": str(destination.relative_to(root)), "status": status,
        "source_fingerprint": fingerprint, "sha256": digest(destination),
        "saved_at": data["saved_at"], "completeness": data["completeness"],
    }
    with (root / ".trajectory-export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = read_json(index_path) if index_path.exists() else {
            "schema": "vsi8b-trajectory-index/1", "reference_answer_visible": False,
            "selected_questions": sampling["n_questions"], "questions": {},
        }
        index["questions"][qa_id] = question_index
        index["saved_at"] = time.time()
        write_json(index_path, index)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--qa-id")
    args = parser.parse_args()
    sampling = read_json(args.run_dir / "sampling_manifest.json")
    ids = [args.qa_id] if args.qa_id else [str(row["qa_id"]) for row in sampling["questions"]]
    for qa_id in ids:
        path = export_question(args.run_dir, qa_id)
        print(json.dumps({"qa_id": qa_id, "trajectory": str(path)}), flush=True)


if __name__ == "__main__":
    main()
