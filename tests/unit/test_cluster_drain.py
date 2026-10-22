"""Drain live work and retain publication recovery without claiming model jobs."""
import hashlib
import importlib.util
from pathlib import Path
import sys

from skill3d.cluster import (
    ArtifactDigest, FrozenIdentity, JobSpec, PublicationReceipt, SQLiteJobQueue,
    WorkerIdentity, hash_file,
)
from skill3d.cluster.identity import canonical_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "cluster_drain_coordinator_test", ROOT / "scripts/cluster_drain_coordinator.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_drain_finishes_live_leases_and_reclaims_failed_return_without_recompute(tmp_path):
    now = [1000.0]
    clock = lambda: now[0]
    identity = FrozenIdentity.freeze({"scope": "test"}, {"method": "frozen"})
    first = JobSpec("a", "done/a", identity, "reconstruction")
    queued = JobSpec("b", "done/b", identity, "reconstruction", priority=100)
    worker = WorkerIdentity("w", "host", "0")
    original = SQLiteJobQueue(tmp_path / "queue.sqlite3", clock=clock)
    original.enqueue(first)
    lease = original.start_attempt(original.claim(worker, lease_seconds=10))
    original.enqueue(queued)
    draining = module.PublicationDrainQueue(original.db_path, clock=clock)
    before = draining.get_job("b")
    assert draining.claim(worker, lease_seconds=10) is None
    assert draining.get_job("b") == before
    lease = draining.heartbeat(lease, lease_seconds=10)
    output = tmp_path / "output"
    output.mkdir()
    data = output / "scene.bin"
    data.write_bytes(b"retained model output")
    receipt = PublicationReceipt(
        first.job_key, first.completed_key, identity.identity_hash, first.stage,
        lease.attempt, "drain-model-free-test",
        (ArtifactDigest("scene.bin", *hash_file(data)),))
    receipt_file = output / "publication.json"
    receipt_file.write_text(canonical_json(receipt.to_dict()))
    lease = draining.finish_attempt(lease, output_metadata={
        "receipt_sha256": hash_file(receipt_file)[0],
        "archive_sha256": hashlib.sha256(b"test archive").hexdigest(),
        "archive_bytes": 12,
    })
    draining.transfer_failed(lease, phase="return", error_code="test_return",
                             detail="temporary return failure", retry_delay=1)
    now[0] += 2
    recovered = draining.claim(WorkerIdentity("w2", "host", "1"), lease_seconds=10)
    assert recovered.job_key == first.job_key
    assert recovered.phase == "publish" and recovered.attempt == 1
    result = draining.complete(
        recovered, receipt_path=receipt_file,
        expected_receipt_hash=hash_file(receipt_file)[0], artifact_root=output)
    assert result.status == "complete"
    assert draining.claim(worker) is None
    assert draining.get_job("b") == before
    assert draining.progress_summary()["model_attempts"] == 1
    # Restoring the unchanged normal control plane can resume the remaining job.
    resumed = SQLiteJobQueue(original.db_path, clock=clock).claim(worker)
    assert resumed.job_key == queued.job_key and resumed.attempt == 0


def test_drain_reaps_expired_compute_but_never_admits_its_replacement(tmp_path):
    now = [1000.0]
    clock = lambda: now[0]
    queue = SQLiteJobQueue(tmp_path / "queue.sqlite3", clock=clock)
    identity = FrozenIdentity.freeze({"scope": "test"}, {"method": "frozen"})
    job = JobSpec("a", "done/a", identity, "reconstruction")
    queue.enqueue(job)
    worker = WorkerIdentity("w", "host", "0")
    queue.start_attempt(queue.claim(worker, lease_seconds=10))
    now[0] += 11
    draining = module.PublicationDrainQueue(queue.db_path, clock=clock)
    assert draining.claim(worker) is None
    state = draining.get_job(job.job_key)
    assert state["status"] == "queued" and state["attempts"] == 1
    assert sum(row["event"] == "lease_expired" for row in draining.events()) == 1
