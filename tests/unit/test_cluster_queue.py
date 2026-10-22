"""CPU-only durability, fencing, identity and publication tests for the coordinator."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest

from skill3d.cluster import (
    ArtifactDigest, EnqueueConflict, FrozenIdentity, IdentityMismatch, InvalidTransition,
    JobSpec, Lease, LeaseLost, PublicationConflict, PublicationError, PublicationReceipt,
    QueueError, SQLiteJobQueue, UnsafeDatabasePath, UnsafePayload, WorkerIdentity, hash_file,
)
from skill3d.cluster.identity import canonical_json


class Clock:
    def __init__(self, now: float = 1000):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def identity() -> FrozenIdentity:
    return FrozenIdentity.freeze(
        {"scope": "sealed-r03", "scene_ids": ["scene-a"], "qa_ids": ["qa-a", "qa-b"]},
        {"method": "cpu-test", "revision": "frozen-code", "seed": 7},
    )


def job(key: str = "job-a", *, attempts: int = 3) -> JobSpec:
    return JobSpec(key, f"completed/{key}", identity(), "reconstruction", max_attempts=attempts)


def worker(index: int = 0) -> WorkerIdentity:
    return WorkerIdentity(f"worker-{index}", f"host-{index // 8}", str(index % 8))


@pytest.fixture
def queue(tmp_path):
    clock = Clock()
    return SQLiteJobQueue(tmp_path / "queue.sqlite3", clock=clock), clock


def claim_running(q: SQLiteJobQueue, index: int = 0) -> Lease:
    lease = q.claim(worker(index), lease_seconds=10)
    assert lease is not None
    return q.start_attempt(lease)


def make_publication(root: Path, lease: Lease, content: bytes = b"validated output"):
    root.mkdir(parents=True, exist_ok=True)
    artifact_path = root / "scene.bin"
    artifact_path.write_bytes(content)
    digest, size = hash_file(artifact_path)
    receipt = PublicationReceipt(
        lease.job_key, lease.spec.completed_key, lease.spec.identity.identity_hash,
        lease.spec.stage, lease.attempt, "cpu-test-validator-v1",
        (ArtifactDigest("scene.bin", digest, size),),
    )
    path = root / "receipt.json"
    path.write_text(canonical_json(receipt.to_dict()), encoding="ascii")
    receipt_hash = hash_file(path)[0]
    return path, receipt_hash, root


def commit(q: SQLiteJobQueue, lease: Lease, publication):
    path, digest, root = publication
    return q.complete(lease, receipt_path=path, expected_receipt_hash=digest, artifact_root=root)


def rewrite_receipt(path: Path, **changes) -> str:
    body = json.loads(path.read_text())
    body.update(changes)
    path.write_text(canonical_json(body), encoding="ascii")
    return hash_file(path)[0]


def _process_claim(db_path: str, barrier, output, index: int):
    q = SQLiteJobQueue(db_path)
    barrier.wait(timeout=20)
    lease = q.claim(worker(index), lease_seconds=60)
    output.put(None if lease is None else lease.to_dict())


def _crash_after_start(db_path: str, output):
    q = SQLiteJobQueue(db_path, clock=lambda: 1000)
    lease = q.start_attempt(q.claim(worker(0), lease_seconds=10))
    output.send(lease.to_dict())
    output.close()
    os._exit(17)


def _crash_uncommitted_transaction(db_path: str):
    db = sqlite3.connect(db_path, isolation_level=None)
    db.execute("BEGIN IMMEDIATE")
    db.execute("UPDATE jobs SET attempts=1,status='running' WHERE job_key='job-a'")
    os._exit(19)


def test_frozen_identity_and_wire_roundtrip():
    manifest = {"scene_ids": ["scene-a"]}
    method = {"model": "test"}
    frozen = FrozenIdentity.freeze(manifest, method)
    manifest["scene_ids"].append("scene-b")
    method["model"] = "different"
    assert frozen.to_dict()["manifest"] == {"scene_ids": ["scene-a"]}
    assert frozen.to_dict()["method_config"] == {"model": "test"}
    assert FrozenIdentity.from_dict(frozen.to_dict()) == frozen
    assert JobSpec.from_dict(job().to_dict()) == job()
    assert frozen == FrozenIdentity.freeze({"scene_ids": ["scene-a"]}, {"model": "test"})


@pytest.mark.parametrize("field", ["manifest", "method_config", "manifest_hash",
                                   "method_config_hash", "identity_hash"])
def test_tampered_identity_rejected(field):
    body = identity().to_dict()
    body[field] = {"modified": True} if field in ("manifest", "method_config") else "0" * 64
    with pytest.raises(IdentityMismatch):
        FrozenIdentity.from_dict(body)


def test_expected_hash_and_noncanonical_snapshot_rejected():
    with pytest.raises(IdentityMismatch):
        FrozenIdentity.freeze({"scope": "r03"}, {}, expected_manifest_hash="0" * 64)
    with pytest.raises(IdentityMismatch):
        replace(identity(), manifest_json=json.dumps(identity().to_dict()["manifest"]))


@pytest.mark.parametrize("data", [
    {"api_key": "not-a-real-secret"}, {"nested": {"credentials": "never-store"}},
    {"reference_answers": ["never-store"]}, {"ground_truth": "never-store"},
    {"uri": "https://example.invalid/path?X-Amz-Signature=not-a-real-secret"},
    {"uri": "https://user:password@example.invalid/path"},
    {"extra": "Bearer not-a-real-secret"}, {"value": float("nan")},
])
def test_sensitive_or_nonfinite_payload_rejected(data):
    with pytest.raises(UnsafePayload):
        FrozenIdentity.freeze(data, {})


def test_enqueue_idempotent_conflicts_and_durability(queue):
    q, clock = queue
    assert q.enqueue(job()).created
    assert not q.enqueue(job()).created
    assert len(q.events()) == 1
    with pytest.raises(EnqueueConflict):
        q.enqueue(replace(job(), max_attempts=4))
    with pytest.raises(EnqueueConflict):
        q.enqueue(replace(job(), identity=FrozenIdentity.freeze({"scope": "other"}, {})))
    with pytest.raises(EnqueueConflict):
        q.enqueue(replace(job(), job_key="other"))
    reopened = SQLiteJobQueue(q.db_path, clock=clock)
    assert reopened.get_job("job-a")["spec"] == job().to_dict()
    assert not reopened.enqueue(job()).created
    with sqlite3.connect(q.db_path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_atomic_batch_rolls_back_and_dependencies_enforce_identity(queue):
    q, _ = queue
    bad_scale = JobSpec("scale", "done-scale", identity(), "scale", ("missing",))
    with pytest.raises(QueueError):
        q.enqueue_many([job(), bad_scale])
    assert q.progress_summary()["total"] == 0
    q.enqueue(job())
    wrong_identity = FrozenIdentity.freeze({"scope": "different"}, {})
    with pytest.raises(IdentityMismatch):
        q.enqueue(JobSpec("scale", "done-scale", wrong_identity, "scale", ("job-a",)))
    with pytest.raises(QueueError):
        JobSpec("scale", "done-scale", identity(), "scale")
    with pytest.raises(QueueError):
        JobSpec("root", "done-root", identity(), "reconstruction", ("job-a",))


def test_scene_aggregate_can_be_root_or_follow_dag(queue):
    q, _ = queue
    root = JobSpec("scene-a", "done-scene-a", identity(), "gpu_aggregate")
    next_job = JobSpec("scene-b", "done-scene-b", identity(), "gpu_aggregate", ("scene-a",))
    q.enqueue_many([root, next_job])
    lease = q.claim(worker(), stages=["gpu_aggregate"])
    assert lease.job_key == "scene-a"
    assert q.claim(worker(1), stages=["gpu_aggregate"]) is None
    q.skip(lease, reason="scene not applicable")
    assert q.get_job("scene-b")["status"] == "skipped"


def test_thread_competing_claims_are_exclusive(queue):
    q, _ = queue
    q.enqueue_many([job(f"job-{i:02}") for i in range(12)])
    barrier = Barrier(24)

    def claim_one(i):
        barrier.wait(timeout=20)
        return q.claim(worker(i), lease_seconds=60)

    with ThreadPoolExecutor(max_workers=24) as pool:
        leases = list(pool.map(claim_one, range(24)))
    claimed = [lease for lease in leases if lease is not None]
    assert len(claimed) == 12
    assert len({lease.job_key for lease in claimed}) == 12
    assert len({lease.token for lease in claimed}) == 12
    assert q.progress_summary()["model_attempts"] == 0


def test_process_competing_claims_are_exclusive(tmp_path):
    db_path = str(tmp_path / "queue.sqlite3")
    q = SQLiteJobQueue(db_path)
    q.enqueue(job())
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(4)
    output = ctx.Queue()
    processes = [ctx.Process(target=_process_claim, args=(db_path, barrier, output, i))
                 for i in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("claim process did not finish")
        assert process.exitcode == 0
    leases = [output.get(timeout=5) for _ in processes]
    output.close()
    output.join_thread()
    assert sum(lease is not None for lease in leases) == 1


def test_crashed_worker_expiry_reclaims_and_fences_old_lease(tmp_path):
    q = SQLiteJobQueue(tmp_path / "queue.sqlite3", clock=Clock())
    q.enqueue(job(attempts=2))
    ctx = multiprocessing.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_crash_after_start, args=(str(q.db_path), sender))
    process.start()
    assert receiver.poll(20)
    old = Lease.from_dict(receiver.recv())
    process.join(20)
    assert process.exitcode == 17
    receiver.close()
    sender.close()
    assert q.get_job("job-a")["attempts"] == 1
    later = SQLiteJobQueue(q.db_path, clock=Clock(1011))
    lease = later.claim(worker(1), lease_seconds=10)
    assert lease is not None and lease.token != old.token
    assert later.start_attempt(lease).attempt == 2
    for action in (later.heartbeat, later.start_attempt):
        with pytest.raises(LeaseLost):
            action(old)
    with pytest.raises(LeaseLost):
        later.fail(old, error_code="stale", detail="old crashed worker")


def test_crashed_uncommitted_transaction_rolls_back(queue):
    q, clock = queue
    q.enqueue(job())
    lease = q.claim(worker(), lease_seconds=10)
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=_crash_uncommitted_transaction, args=(str(q.db_path),))
    process.start()
    process.join(20)
    assert process.exitcode == 19
    reopened = SQLiteJobQueue(q.db_path, clock=clock)
    assert reopened.get_job("job-a")["attempts"] == 0
    assert reopened.get_job("job-a")["status"] == "leased"
    assert reopened.start_attempt(lease).attempt == 1


def test_heartbeat_and_tampered_worker_are_fenced(queue):
    q, clock = queue
    q.enqueue(job())
    lease = q.claim(worker(), lease_seconds=10)
    assert Lease.from_dict(lease.to_dict()) == lease
    clock.advance(8)
    renewed = q.heartbeat(lease, lease_seconds=20)
    assert renewed.expires_at == 1028
    assert q.heartbeat(lease, lease_seconds=1).expires_at == 1028
    clock.advance(3)
    assert q.claim(worker(1)) is None
    with pytest.raises(LeaseLost):
        q.heartbeat(replace(lease, worker=worker(1)))
    with pytest.raises(LeaseLost):
        q.heartbeat(replace(lease, token="0" * 64))
    with pytest.raises(IdentityMismatch):
        q.heartbeat(replace(lease, spec=replace(lease.spec, completed_key="other")))
    clock.advance(17)
    with pytest.raises(LeaseLost):
        q.heartbeat(lease)
    assert q.claim(worker(1)) is not None


def test_model_budget_expiry_and_explicit_error_terminal(queue):
    q, clock = queue
    q.enqueue(job(attempts=1))
    lease = claim_running(q)
    assert q.start_attempt(lease).attempt == 1
    clock.advance(10)
    assert q.claim(worker(1)) is None
    result = q.get_job("job-a")
    assert result["status"] == "failed"
    assert result["error_code"] == "attempt_budget_exhausted"
    assert result["error_detail"]
    assert result["completion"] is None
    assert not q.enqueue(job(attempts=1)).created
    assert q.progress_summary()["counts"]["complete"] == 0


def test_model_retry_budget_and_delay(queue):
    q, clock = queue
    q.enqueue(job(attempts=2))
    lease = claim_running(q)
    assert q.fail(lease, error_code="model_error", detail="sanitized failure",
                  retryable=True, retry_delay=5) == "queued"
    assert q.claim(worker(1)) is None
    clock.advance(5)
    lease = claim_running(q, 1)
    assert lease.attempt == 2
    assert q.fail(lease, error_code="model_error", detail="second failure",
                  retryable=True) == "failed"
    assert q.get_job("job-a")["error_code"] == "model_error"
    assert q.claim(worker(2)) is None


def test_prefetch_and_return_failures_do_not_consume_model_budget(queue, tmp_path):
    q, clock = queue
    q.enqueue(job(attempts=1))
    lease = q.claim(worker(), lease_seconds=10)
    q.transfer_failed(lease, phase="prefetch", error_code="fetch_failed",
                      detail="remote source unavailable", retry_delay=5)
    assert q.get_job("job-a")["attempts"] == 0
    assert q.get_job("job-a")["prefetch_failures"] == 1
    assert q.claim(worker(1)) is None
    clock.advance(5)
    lease = claim_running(q, 1)
    metadata = {"output_locator": "local-scene-a-attempt-1", "content_hash": "a" * 64}
    lease = q.finish_attempt(lease, output_metadata=metadata)
    assert q.finish_attempt(lease, output_metadata=metadata) == lease
    with pytest.raises(PublicationConflict):
        q.finish_attempt(lease, output_metadata={"output_locator": "different"})
    publication = make_publication(tmp_path / "publish", lease)
    q.transfer_failed(lease, phase="return", error_code="upload_failed",
                      detail="Titan unavailable")
    resumed = q.claim(worker(2), lease_seconds=10)
    assert resumed.phase == "publish" and resumed.attempt == 1
    assert json.loads(resumed.output_metadata_json) == metadata
    with pytest.raises(InvalidTransition):
        q.start_attempt(resumed)
    assert commit(q, resumed, publication).status == "complete"
    summary = q.progress_summary()
    assert summary["model_attempts"] == 1
    assert summary["prefetch_failures"] == summary["return_failures"] == 1


def test_publication_expiry_resumes_without_rerun(queue, tmp_path):
    q, clock = queue
    q.enqueue(job(attempts=1))
    lease = q.finish_attempt(claim_running(q), output_metadata={"output_locator": "retained"})
    publication = make_publication(tmp_path / "publish", lease)
    clock.advance(11)
    resumed = SQLiteJobQueue(q.db_path, clock=clock).claim(worker(1))
    assert resumed.phase == "publish" and resumed.attempt == 1
    with pytest.raises(LeaseLost):
        commit(q, lease, publication)
    assert commit(q, resumed, publication).status == "complete"


def test_complete_commit_ack_lost_is_idempotent_and_conflicts_rejected(queue, tmp_path):
    q, clock = queue
    q.enqueue(job())
    lease = q.finish_attempt(claim_running(q), output_metadata={"output_locator": "retained"})
    publication = make_publication(tmp_path / "publish", lease)
    first = commit(q, lease, publication)
    assert not first.duplicate
    clock.advance(1000)
    reopened = SQLiteJobQueue(q.db_path, clock=clock)
    # A committed ACK retry uses the stored hashes even if artifact bytes are unavailable.
    (publication[2] / "scene.bin").unlink()
    second = commit(reopened, Lease.from_dict(lease.to_dict()), publication)
    assert second.duplicate and second.fingerprint == first.fingerprint
    assert reopened.claim(worker(1)) is None
    assert len([e for e in reopened.events() if e["event"] == "completed"]) == 1
    conflict = make_publication(tmp_path / "different", lease, b"different validated bytes")
    with pytest.raises(PublicationConflict):
        commit(reopened, lease, conflict)
    with pytest.raises(LeaseLost):
        commit(reopened, replace(lease, worker=worker(2)), publication)
    assert reopened.get_job("job-a")["completion"]["fingerprint"] == first.fingerprint


def test_concurrent_duplicate_completion_commits_once(queue, tmp_path):
    q, _ = queue
    q.enqueue(job())
    lease = q.finish_attempt(claim_running(q), output_metadata={})
    publication = make_publication(tmp_path / "publish", lease)
    barrier = Barrier(8)

    def complete_one(_):
        barrier.wait(timeout=20)
        return commit(q, lease, publication)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(complete_one, range(8)))
    assert sum(not result.duplicate for result in results) == 1
    assert len({result.fingerprint for result in results}) == 1


@pytest.mark.parametrize("tamper", ["artifact", "receipt_hash", "identity", "validation",
                                   "missing_artifact", "attempt", "path_escape"])
def test_publication_tampering_never_marks_complete(queue, tmp_path, tamper):
    q, _ = queue
    q.enqueue(job())
    lease = q.finish_attempt(claim_running(q), output_metadata={})
    path, digest, root = make_publication(tmp_path / "publish", lease)
    expected_exception = QueueError
    if tamper == "artifact":
        (root / "scene.bin").write_bytes(b"tampered content")
    elif tamper == "receipt_hash":
        digest = "0" * 64
    elif tamper == "identity":
        digest = rewrite_receipt(path, identity_hash="0" * 64)
        expected_exception = IdentityMismatch
    elif tamper == "validation":
        digest = rewrite_receipt(path, validation={"status": "pending", "validator": "cpu-test"})
    elif tamper == "missing_artifact":
        (root / "scene.bin").unlink()
    elif tamper == "attempt":
        digest = rewrite_receipt(path, attempt=2)
        expected_exception = InvalidTransition
    elif tamper == "path_escape":
        body = json.loads(path.read_text())
        body["artifacts"][0]["name"] = "../scene.bin"
        digest = rewrite_receipt(path, artifacts=body["artifacts"])
    with pytest.raises(expected_exception):
        commit(q, lease, (path, digest, root))
    result = q.get_job("job-a")
    assert result["status"] == "publishing"
    assert result["completion"] is None


def test_missing_receipt_uncheckpointed_and_symlink_escape_rejected(queue, tmp_path):
    q, _ = queue
    q.enqueue(job())
    lease = claim_running(q)
    publication = make_publication(tmp_path / "publish", lease)
    with pytest.raises(InvalidTransition):
        commit(q, lease, publication)
    lease = q.finish_attempt(lease, output_metadata={})
    with pytest.raises(PublicationError):
        q.complete(lease, receipt_path=tmp_path / "missing", expected_receipt_hash="0" * 64,
                   artifact_root=publication[2])
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"validated output")
    (publication[2] / "scene.bin").unlink()
    (publication[2] / "scene.bin").symlink_to(outside)
    with pytest.raises(PublicationError):
        commit(q, lease, publication)
    assert q.get_job("job-a")["completion"] is None


def test_dag_stage_dependencies_and_completion_order(queue, tmp_path):
    q, _ = queue
    assert all(result.created for result in q.enqueue_pipeline("scene-a", identity()))
    assert not any(result.created for result in q.enqueue_pipeline("scene-a", identity()))
    assert q.claim(worker(), stages=["scale", "binding"]) is None
    for stage in ("reconstruction", "scale", "binding"):
        lease = claim_running(q)
        assert lease.spec.stage == stage
        assert q.claim(worker(1)) is None
        lease = q.finish_attempt(lease, output_metadata={})
        commit(q, lease, make_publication(tmp_path / stage, lease))
    summary = q.progress_summary()
    assert summary["counts"]["complete"] == 3 and summary["total"] == 3


def test_skips_and_failed_dependencies_remain_distinct(queue):
    q, _ = queue
    q.enqueue_pipeline("skip-scene", identity())
    lease = q.claim(worker())
    with pytest.raises(UnsafePayload):
        q.skip(lease, reason="")
    q.skip(lease, reason="Scale not applicable for this scene")
    for stage in ("reconstruction", "scale", "binding"):
        result = q.get_job(f"skip-scene/{stage}")
        assert result["status"] == "skipped" and result["skip_reason"]
        assert result["completion"] is None
    q.enqueue_pipeline("failed-scene", identity())
    lease = claim_running(q)
    with pytest.raises(UnsafePayload):
        q.fail(lease, error_code="", detail="must specify error")
    q.fail(lease, error_code="invalid_geometry", detail="Validator rejected output")
    q.reap_expired()
    for stage in ("reconstruction", "scale", "binding"):
        result = q.get_job(f"failed-scene/{stage}")
        assert result["status"] == "failed" and result["error_code"] and result["error_detail"]
        assert result["completion"] is None
    assert q.progress_summary()["counts"]["complete"] == 0


def test_explicit_skipped_dependency_policy(queue):
    q, _ = queue
    root = job()
    scale = JobSpec("scale", "done-scale", identity(), "scale", (root.job_key,),
                    dependency_policy="complete_or_skipped")
    q.enqueue_many([root, scale])
    q.skip(q.claim(worker()), reason="Preexisting geometry handled by wrapper")
    assert q.claim(worker(1)).job_key == "scale"


def test_read_only_summaries_and_host_gpu_events(queue):
    q, clock = queue
    q.enqueue(job())
    lease = claim_running(q, 9)
    clock.advance(11)
    with sqlite3.connect(q.db_path) as db:
        before = db.total_changes
        rows_before = db.execute("SELECT * FROM jobs").fetchall()
        events_before = db.execute("SELECT * FROM events").fetchall()
        summary = q.progress_summary()
        q.events()
        result = q.get_job("job-a")
        assert db.total_changes == before
        assert db.execute("SELECT * FROM jobs").fetchall() == rows_before
        assert db.execute("SELECT * FROM events").fetchall() == events_before
    assert result["status"] == "running" and "lease_token" not in result
    assert summary["expired_leases"] == 1 and summary["counts"]["running"] == 1
    assert "token" not in json.dumps(q.events())
    owned_events = [e for e in q.events() if e["event"] in ("claimed", "attempt_started")]
    assert all((e["worker_id"], e["host"], e["gpu_id"]) == ("worker-9", "host-1", "1")
               for e in owned_events)
    assert owned_events and lease.token not in json.dumps(result)


def test_sql_constraints_prevent_fake_terminal_and_identity_overwrite(queue, tmp_path):
    q, _ = queue
    q.enqueue(job())
    with sqlite3.connect(q.db_path) as db:
        for sql in (
            "UPDATE jobs SET status='complete'",
            "UPDATE jobs SET status='failed'",
            "UPDATE jobs SET status='skipped'",
            "UPDATE jobs SET identity_hash='tampered'",
            "UPDATE jobs SET max_attempts=100",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(sql)
    lease = q.finish_attempt(claim_running(q), output_metadata={})
    commit(q, lease, make_publication(tmp_path / "publish", lease))
    with sqlite3.connect(q.db_path) as db:
        for sql in ("UPDATE completions SET fingerprint='tampered'",
                    "DELETE FROM completions", "UPDATE jobs SET status='queued'"):
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(sql)


def test_sensitive_errors_and_output_metadata_are_not_logged(queue):
    q, _ = queue
    q.enqueue(job())
    lease = claim_running(q)
    before = q.events()
    with pytest.raises(UnsafePayload):
        q.fail(lease, error_code="model_error", detail="Bearer not-a-real-secret")
    with pytest.raises(UnsafePayload):
        q.finish_attempt(lease, output_metadata={"reference_answers": ["never-store"]})
    assert q.events() == before
    assert q.get_job("job-a")["status"] == "running"


def test_network_db_rejected_before_creation(monkeypatch, tmp_path):
    import skill3d.cluster.queue as module

    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "nfs\n")
    with pytest.raises(UnsafeDatabasePath):
        SQLiteJobQueue(tmp_path / "network" / "queue.sqlite3")
    assert not (tmp_path / "network").exists()
    with pytest.raises(UnsafeDatabasePath):
        SQLiteJobQueue(":memory:")


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan"), True])
def test_invalid_lease_duration_rejected(queue, seconds):
    q, _ = queue
    q.enqueue(job())
    with pytest.raises(QueueError):
        q.claim(worker(), lease_seconds=seconds)


def test_tampered_stored_identity_and_dag_fail_closed(queue):
    q, _ = queue
    q.enqueue_pipeline("scene", identity())
    with sqlite3.connect(q.db_path) as db:
        db.execute("DROP TRIGGER immutable_job_identity")
        db.execute("UPDATE jobs SET spec_hash=? WHERE job_key='scene/reconstruction'", ("0" * 64,))
    with pytest.raises(IdentityMismatch):
        q.claim(worker())
    assert not any(e["event"] == "claimed" for e in q.events())


def test_tampered_dependencies_are_detected_even_if_sql_guard_removed(queue):
    q, _ = queue
    q.enqueue_pipeline("scene", identity())
    with sqlite3.connect(q.db_path) as db:
        db.execute("DROP TRIGGER immutable_dependency_delete")
        db.execute("DELETE FROM dependencies WHERE job_key='scene/binding'")
    with pytest.raises(IdentityMismatch):
        q.get_job("scene/binding")
    with pytest.raises(IdentityMismatch):
        q.claim(worker(), stages=["binding"])


def test_duplicate_commit_rejects_modified_lease_spec(queue, tmp_path):
    q, _ = queue
    q.enqueue(job())
    lease = q.finish_attempt(claim_running(q), output_metadata={})
    publication = make_publication(tmp_path / "publish", lease)
    commit(q, lease, publication)
    changed = replace(lease, spec=replace(lease.spec, max_attempts=100))
    with pytest.raises(IdentityMismatch):
        commit(q, changed, publication)


def test_clock_expiry_during_hashing_does_not_commit(queue, tmp_path, monkeypatch):
    import skill3d.cluster.queue as module

    q, clock = queue
    q.enqueue(job())
    lease = q.finish_attempt(claim_running(q), output_metadata={})
    publication = make_publication(tmp_path / "publish", lease)
    original = module.validate_artifacts

    def slow_validate(*args, **kwargs):
        original(*args, **kwargs)
        clock.advance(11)

    monkeypatch.setattr(module, "validate_artifacts", slow_validate)
    with pytest.raises(LeaseLost):
        commit(q, lease, publication)
    assert q.get_job("job-a")["completion"] is None
    assert q.get_job("job-a")["status"] == "publishing"


def test_duplicate_receipt_json_keys_rejected(queue, tmp_path):
    q, _ = queue
    q.enqueue(job())
    lease = q.finish_attempt(claim_running(q), output_metadata={})
    path = tmp_path / "receipt.json"
    raw = b'{"job_key":"job-a","job_key":"other"}'
    path.write_bytes(raw)
    with pytest.raises(PublicationError):
        q.complete(lease, receipt_path=path, expected_receipt_hash=hashlib.sha256(raw).hexdigest(),
                   artifact_root=tmp_path)
    assert q.get_job("job-a")["completion"] is None


def test_foreign_database_not_used(tmp_path):
    path = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated(value TEXT)")
    with pytest.raises(QueueError):
        SQLiteJobQueue(path)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=1")
    with pytest.raises(QueueError):
        SQLiteJobQueue(path)


def test_failed_completion_transaction_rolls_back_receipt_and_status(queue, tmp_path, monkeypatch):
    q, _ = queue
    q.enqueue(job())
    lease = q.finish_attempt(claim_running(q), output_metadata={})
    publication = make_publication(tmp_path / "publish", lease)
    original = q._event

    def reject_completion_event(db, row, event, now, detail=None):
        if event == "completed":
            raise sqlite3.OperationalError("simulated commit failure")
        return original(db, row, event, now, detail)

    monkeypatch.setattr(q, "_event", reject_completion_event)
    with pytest.raises(sqlite3.OperationalError):
        commit(q, lease, publication)
    assert q.get_job("job-a")["status"] == "publishing"
    assert q.get_job("job-a")["completion"] is None
    with sqlite3.connect(q.db_path) as db:
        assert db.execute("SELECT COUNT(*) FROM completions").fetchone()[0] == 0
    monkeypatch.setattr(q, "_event", original)
    assert not commit(q, lease, publication).duplicate


def test_output_validation_failure_can_retry_model_but_never_fake_complete(queue):
    q, _ = queue
    q.enqueue(job(attempts=2))
    lease = q.finish_attempt(claim_running(q), output_metadata={"output_locator": "invalid"})
    assert q.fail(lease, error_code="qa_group_invalid", detail="QA group validation failed",
                  retryable=True) == "queued"
    result = q.get_job("job-a")
    assert result["resume_phase"] == "prefetch"
    assert result["output_metadata"] == {} and result["completion"] is None
    lease = claim_running(q, 1)
    assert lease.attempt == 2
    lease = q.finish_attempt(lease, output_metadata={"output_locator": "invalid-again"})
    assert q.fail(lease, error_code="qa_group_invalid", detail="Final validation failed",
                  retryable=True) == "failed"
    assert q.get_job("job-a")["completion"] is None


def test_prefetch_expired_final_budget_still_has_no_model_attempt(queue):
    q, clock = queue
    q.enqueue(job(attempts=1))
    first = q.claim(worker(), lease_seconds=10)
    clock.advance(11)
    replacement = q.claim(worker(1))
    assert replacement is not None and replacement.attempt == 0
    assert first.token != replacement.token
    assert q.start_attempt(replacement).attempt == 1


@pytest.mark.parametrize("decoder", [FrozenIdentity.from_dict, JobSpec.from_dict, Lease.from_dict])
@pytest.mark.parametrize("body", [None, [], {}, {"unexpected": True}])
def test_invalid_wire_envelopes_raise_queue_error(decoder, body):
    with pytest.raises(QueueError):
        decoder(body)
