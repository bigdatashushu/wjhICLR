"""Recovery retains failed bytes and refuses to repeat issued model calls."""
import importlib
import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
archiver = importlib.import_module("archive_pilot_source_failure")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def stopped_source_case(tmp_path):
    base, root = tmp_path / "base", tmp_path / "run"
    put(base / "previous.pid.json", {
        "pid": 999999999, "process_identity": {"start_ticks": "synthetic"}})
    put(root / "live_status.json", {"status": "stopped", "elapsed_s": 10})
    put(root / "parallel_execution.json", {"api_limit": 32})
    put(root / "host_errors/5060.json", {"stage": "prepare", "error_type": "RuntimeError"})
    put(root / "bindings/5060.json", {
        "stats": {"binding_error": "ValueError: cached parser was not successful"}})
    put(root / "sam2_receipts/5060.json", [])
    put(root / "trajectories/5060.json", {"status": "failed"})
    put(root / "preparation/5060/binding_progress.json", {"stage": "parse"})
    put(root / "outcomes/44.json", {"answer": "retained"})
    put(root / "solver_requests/44/001.json", {"status": "returned"})
    source = tmp_path / "source.json"
    put(source, {"materialized": False, "objects": [], "stats": {
        "object_parse": {"status": "unavailable"}, "candidate_receipts": [],
        "detection_receipts": []}})
    put(root / "reuse_preflight.json", {"questions": [{
        "qa_id": "5060", "binding_file": str(source),
        "binding_file_sha256": archiver.digest(source)}]})
    return base, root, source


def test_archives_failed_bytes_and_preserves_completed_outcomes(tmp_path):
    base, root, source = stopped_source_case(tmp_path)
    failed_hash = archiver.digest(root / "bindings/5060.json")
    outcome = (root / "outcomes/44.json").read_bytes()
    source_bytes = source.read_bytes()
    archiver.archive(root, base, "previous", "5060")
    assert not (root / "bindings/5060.json").exists()
    assert (root / "outcomes/44.json").read_bytes() == outcome
    assert source.read_bytes() == source_bytes
    history = archiver.read(root / "execution_resume_history.json")
    epoch = history["epochs"][0]
    assert epoch["preserved_outcomes"] == ["44"]
    assert epoch["preserved_outcome_sha256"]["44"] == archiver.digest(root / "outcomes/44.json")
    assert epoch["issued_solver_requests_repeated"] == 0
    archived = next(row for row in epoch["archived_files"]
                    if row["original_file"] == "bindings/5060.json")
    assert archived["sha256"] == failed_hash == archiver.digest(root / archived["file"])


@pytest.mark.parametrize("status", ["started", "queued", "failed"])
def test_refuses_unreturned_requests_before_moving_files(tmp_path, status):
    base, root, _ = stopped_source_case(tmp_path)
    put(root / "solver_requests/44/001.json", {"status": status})
    with pytest.raises(ValueError, match="unfinished"):
        archiver.archive(root, base, "previous", "5060")
    assert (root / "bindings/5060.json").exists()
    assert not (root / "attempt_history").exists()


def test_refuses_to_repeat_previously_issued_episode(tmp_path):
    base, root, _ = stopped_source_case(tmp_path)
    put(root / "solver_requests/5060/001.json", {"status": "returned"})
    with pytest.raises(ValueError, match="cannot repeat"):
        archiver.archive(root, base, "previous", "5060")
    assert (root / "bindings/5060.json").exists()


def test_refuses_source_identity_change(tmp_path):
    base, root, source = stopped_source_case(tmp_path)
    source.write_text("{}")
    with pytest.raises(ValueError, match="changed after preflight"):
        archiver.archive(root, base, "previous", "5060")
    assert (root / "bindings/5060.json").exists()


def test_archives_pre_model_legacy_consumer_failure_and_scene_audit(tmp_path):
    base, root, source = stopped_source_case(tmp_path)
    original = archiver.read(source)
    original["stats"]["generation"] = {"identity": {"version": "legacy"}, "sha256": "fixed"}
    put(source, original)
    put(root / "bindings/5060.json", original)
    put(root / "host_errors/5060.json", {
        "stage": "solve", "error_type": "ValueError",
        "stack": [{"function": "validate_prepared_binding"}]})
    put(root / "reuse_preflight.json", {"questions": [{
        "qa_id": "5060", "binding_file": str(source),
        "binding_file_sha256": archiver.digest(source), "scene_id": "scene"}]})
    put(root / "scene_audits/scene.json", {"refreshed_qa_ids": ["5060"]})
    trace = root / "solver_trace/5060/state.jsonl"
    put(trace, {"state": "before_model"})
    original_trace = trace.read_bytes()
    archiver.archive(root, base, "previous", "5060")
    assert not (root / "scene_audits/scene.json").exists()
    assert not trace.exists()
    epoch = archiver.read(root / "execution_resume_history.json")["epochs"][0]
    assert epoch["reason"] == "source_parser_unavailable_legacy_consumer_identity"
    archived = next(row for row in epoch["archived_files"]
                    if row["original_file"] == "solver_trace/5060/state.jsonl")
    assert (root / archived["file"]).read_bytes() == original_trace
