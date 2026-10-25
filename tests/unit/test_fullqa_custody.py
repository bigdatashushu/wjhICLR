"""Operational crash and admission contracts; no GPU or model requests."""
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import fullqa_custody as c
from fullqa_supervisor import complete_ack
from skill3d.cluster import FrozenIdentity, JobSpec, SQLiteJobQueue, WorkerIdentity


def fixture_queue(tmp_path):
    root = tmp_path / "run"
    q = SQLiteJobQueue(root / "state/jobs.sqlite3")
    identity = FrozenIdentity.freeze({"scene": "fixture"}, {"method": "fixture"})
    q.enqueue_many([JobSpec(f"job-{i}", f"done-{i}", identity, "gpu_aggregate") for i in range(3)])
    return root, q


def test_claim_fences_settlement_and_keeps_attempt_zero(tmp_path):
    root, q = fixture_queue(tmp_path)
    worker = {"worker_id": "slot", "host": "fixture", "gpu_id": "3"}
    lease = c.native_claim(root, worker)
    assert lease.attempt == 0 and lease.phase == "leased"
    with pytest.raises(c.Refused, match="worker_already_owned"):
        c.native_claim(root, worker)
    with c.ro_database(root) as db:
        assert list(db.execute("SELECT event FROM events WHERE event='attempt_started'")) == []
        assert dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status")) == {
            "queued": 2, "leased": 1}


def test_expiring_other_worker_cannot_be_reaped_by_admission(tmp_path):
    root, q = fixture_queue(tmp_path)
    original = q.claim(WorkerIdentity("other", "fixture", "4"), lease_seconds=10)
    with pytest.raises(c.Refused, match="expiring"):
        c.native_claim(root, {"worker_id": "slot", "host": "fixture", "gpu_id": "3"})
    with c.ro_database(root) as db:
        row = db.execute("SELECT * FROM jobs WHERE job_key=?", (original.job_key,)).fetchone()
        assert row["lease_token"] == original.token and row["claims"] == 1
        assert not db.execute("SELECT 1 FROM events WHERE event='lease_expired'").fetchone()


def test_next_claim_recovers_after_reply_loss_with_previous_delivery_retained(tmp_path):
    root, q = fixture_queue(tmp_path)
    worker = {"worker_id": "slot", "host": "fixture", "gpu_id": "3"}
    lease = c.native_claim(root, worker)
    with c.ro_database(root) as db:
        row = db.execute("SELECT * FROM jobs WHERE job_key=?", (lease.job_key,)).fetchone()
    old = {"lease": {"token": "previous"}}
    intent = {"worker": worker, "phase": "claiming", "at": row["updated_at"] - 1}
    assert c.recover_delivery(row, lease, old, intent, worker) is None
    with pytest.raises(c.Refused):
        c.recover_delivery(row, lease, old, {**intent, "at": row["updated_at"] + 1}, worker)


@pytest.mark.parametrize("phase,pending,ack,expected", [
    ("running", False, False, "quarantine"),
    ("starting", False, False, "quarantine"),
    ("publishing", True, False, "resume_publication"),
    ("running", True, False, "resume_publication"),
    ("delivered", False, False, "launch"),
    ("publishing", False, True, "copy_ack"),
])
def test_recovery_never_restarts_computed_attempt(phase, pending, ack, expected):
    assert c.recovery_action(phase, False, pending, ack) == expected
    if not ack:
        assert c.recovery_action(phase, True, pending, ack) == "observe"


def test_immutable_delivery_survives_replay_and_rejects_drift(tmp_path):
    path = tmp_path / "delivery.json"
    value = {"job_key": "fixture", "attempt": 0}
    first = c.durable(path, value, immutable=True)
    assert first == c.durable(path, value, immutable=True)
    with pytest.raises(c.Refused):
        c.durable(path, {**value, "attempt": 1}, immutable=True)
    assert json.loads(path.read_bytes()) == value
    assert not list(tmp_path.glob("*.part"))


def test_ack_requires_complete_exact_identity_and_boolean_duplicate():
    expected = {"status": "complete", "job_key": "fixture", "receipt_hash": "a" * 64}
    complete_ack(expected, {**expected, "duplicate": True})
    for value in [{**expected, "duplicate": 1},
                  {**expected, "duplicate": False, "status": "running"},
                  {**expected, "duplicate": False, "receipt_hash": "b" * 64}]:
        with pytest.raises(c.Refused):
            complete_ack(expected, value)
