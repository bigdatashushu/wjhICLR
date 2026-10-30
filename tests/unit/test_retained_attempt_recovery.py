"""Exercise the real queue's continuation, rollback and stale-owner fencing."""
import importlib.util
from pathlib import Path
import pytest

from skill3d.cluster import FrozenIdentity, JobSpec, LeaseLost, SQLiteJobQueue, WorkerIdentity
from skill3d.cluster.identity import canonical_json

path = Path(__file__).parents[2] / "scripts/retained_attempt_recovery.py"
spec = importlib.util.spec_from_file_location("retained_attempt_recovery", path)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


@pytest.fixture
def retained(tmp_path):
    queue = SQLiteJobQueue(tmp_path / "queue.sqlite3", clock=lambda: 1000)
    identity = FrozenIdentity.freeze({"scene_ids": ["scene"]}, {"method": "frozen"})
    queue.enqueue(JobSpec("retained", "complete-retained", identity, "gpu_aggregate"))
    queue.enqueue(JobSpec("untouched", "complete-other", identity, "gpu_aggregate"))
    old = queue.start_attempt(queue.claim(WorkerIdentity("worker", "3090", "0")))
    queue.fail(old, error_code="scene_compute_failed", detail="PermissionError", retryable=True)
    with queue._connect(readonly=True) as db:
        expected = ops.expectation(db, "retained")
    proof = {
        "status": "verified", "job_key": "retained", "gpu": 0, "attempt": 1,
        "model_requests": 0, "complete_geometry": ["base", "metric"],
        "verified_at": 999, "method_config_canonical_sha256": ops.digest({"method": "frozen"}),
    }
    return queue, old, expected, proof


def test_commit_preserves_attempt_and_native_start_is_idempotent(retained):
    queue, old, expected, proof = retained
    lease, receipt = ops.recover(queue, expected, proof, commit=True, now=1000)
    assert lease.token != old.token and lease.phase == "running"
    assert lease.attempt == 1 and receipt["claims"] == 1
    assert queue.start_attempt(lease) == lease
    job = queue.get_job("retained")
    assert job["error_detail"] == "PermissionError"
    assert job["claims"] == 1 and job["attempts"] == 1
    assert queue.get_job("untouched")["claims"] == 0
    assert [e["event"] for e in queue.events(job_key="retained")] == [
        "enqueued", "claimed", "attempt_started", "model_retry", ops.EVENT]
    with pytest.raises(LeaseLost):
        queue.start_attempt(old)
    with pytest.raises(ValueError):
        ops.recover(queue, expected, proof, commit=True, now=1000)


def test_dry_run_restores_all_queue_tables(retained):
    queue, _, expected, proof = retained
    with queue._connect(readonly=True) as db:
        before = ops.snapshot(db)
    _, receipt = ops.recover(queue, expected, proof, now=1000)
    assert receipt["status"] == "rolled_back_preflight"
    with queue._connect(readonly=True) as db:
        assert ops.snapshot(db) == before


@pytest.mark.parametrize("change", [
    {"verified_at": -4000}, {"method_config_canonical_sha256": "0" * 64},
    {"gpu": 5}, {"complete_geometry": ["base"]}, {"model_requests": 1},
])
def test_changed_checkpoint_proof_refuses_without_writes(retained, change):
    queue, _, expected, proof = retained
    with queue._connect(readonly=True) as db:
        before = ops.snapshot(db)
    with pytest.raises(ValueError):
        ops.recover(queue, expected, proof | change, commit=True, now=1000)
    with queue._connect(readonly=True) as db:
        assert ops.snapshot(db) == before


def test_changed_history_refuses_without_writes(retained):
    queue, _, expected, proof = retained
    with queue._write() as db:
        row = db.execute("SELECT * FROM jobs WHERE job_key='retained'").fetchone()
        queue._event(db, row, "external_operator_event", 1000)
    with queue._connect(readonly=True) as db:
        before = ops.snapshot(db)
    with pytest.raises(ValueError):
        ops.recover(queue, expected, proof, commit=True, now=1000)
    with queue._connect(readonly=True) as db:
        assert ops.snapshot(db) == before
