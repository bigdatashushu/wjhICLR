"""Operator-only continuation of a verified retained attempt, without a new claim."""
from __future__ import annotations

import hashlib
import json
import math
import secrets
import time

from skill3d.cluster.queue import SQLiteJobQueue

EVENT = "ops_retained_attempt_recovered"


def digest(value):
    # Internal snapshots include SQL newlines and fencing tokens. Only their
    # digest leaves this function; native _event still validates its public body.
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def snapshot(db):
    return {
        table: [list(row) for row in db.execute("SELECT * FROM " + table + " ORDER BY 1")]
        for table in ("jobs", "dependencies", "completions", "events")
    } | {
        "schema": [list(row) for row in db.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name")]
    }


def expectation(db, job_key):
    row = db.execute("SELECT * FROM jobs WHERE job_key=?", (job_key,)).fetchone()
    if row is None:
        raise ValueError("unknown retained job")
    history = [dict(item) for item in db.execute(
        "SELECT * FROM events WHERE job_key=? ORDER BY id", (job_key,))]
    return {"job_key": job_key, "row_sha256": digest(dict(row)),
            "history_sha256": digest(history)}


def recover(queue, expected, proof, *, commit=False, now=None, lease_seconds=21600):
    """Return a new, explicitly recorded custody token for the existing attempt.

    A successful precheck must come from the frozen input/checkpoint validator.
    The caller separately fences the exited worker and verifies these proof bytes.
    This entry point never claims, starts a new model attempt, or erases failure history.
    """
    if type(commit) is not bool:
        raise ValueError("commit must be explicit")
    now = time.time() if now is None else now
    if (type(now) not in (int, float) or not math.isfinite(now) or now < 0
            or type(lease_seconds) not in (int, float)
            or not math.isfinite(lease_seconds) or lease_seconds <= 60):
        raise ValueError("invalid recovery lifetime")
    with queue._connect() as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            before = snapshot(db)
            if expectation(db, expected["job_key"]) != expected:
                raise ValueError("retained queue row or history changed")
            row = db.execute("SELECT * FROM jobs WHERE job_key=?",
                             (expected["job_key"],)).fetchone()
            queue._verify_dependencies(db, row)
            if not (row["status"] == "queued" and row["attempts"] == 1
                    and row["resume_phase"] == "prefetch"
                    and row["output_metadata_json"] == "{}"
                    and row["lease_token"] is None and row["lease_until"] is None
                    and row["error_code"] == "scene_compute_failed"
                    and row["error_detail"] == "PermissionError"):
                raise ValueError("job is not the inspected retained filesystem failure")
            history = db.execute("SELECT * FROM events WHERE job_key=? ORDER BY id",
                                 (row["job_key"],)).fetchall()
            if not (history[-1]["event"] == "model_retry"
                    and history[-1]["attempt"] == 1
                    and sum(item["event"] == "attempt_started" for item in history) == 1):
                raise ValueError("retained attempt history differs")
            if db.execute("SELECT 1 FROM completions WHERE job_key=?",
                          (row["job_key"],)).fetchone():
                raise ValueError("job already has a completion")
            if db.execute(
                    "SELECT 1 FROM jobs WHERE worker_id=? AND status IN "
                    "('leased','running','publishing')", (row["worker_id"],)).fetchone():
                raise ValueError("retained worker already owns active custody")
            if not (proof["status"] == "verified" and proof["job_key"] == row["job_key"]
                    and proof["attempt"] == row["attempts"]
                    and str(proof["gpu"]) == row["gpu_id"]
                    and proof["model_requests"] == 0
                    and proof["complete_geometry"] == ["base", "metric"]
                    and 0 <= now - proof["verified_at"] < 3600):
                raise ValueError("fresh frozen checkpoint proof required")
            spec = queue._spec(row)
            if proof["method_config_canonical_sha256"] != digest(
                    json.loads(spec.identity.method_config_json)):
                raise ValueError("precheck method identity differs")
            token = secrets.token_hex(32)
            db.execute(
                "UPDATE jobs SET status='running',lease_token=?,lease_until=?,updated_at=? "
                "WHERE job_key=?", (token, now + lease_seconds, now, row["job_key"]))
            current = db.execute("SELECT * FROM jobs WHERE job_key=?",
                                 (row["job_key"],)).fetchone()
            detail = {
                "reason_code": "operator_continue_verified_retained_attempt",
                "previous_row_sha256": expected["row_sha256"],
                "previous_history_sha256": expected["history_sha256"],
                "checkpoint_proof_sha256": digest(proof),
                "new_custody_token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                "attempts_preserved": True, "claims_preserved": True,
                "failure_history_preserved": True,
            }
            queue._event(db, current, EVENT, now, detail)
            after = snapshot(db)
            for key in ("schema", "dependencies", "completions"):
                if before[key] != after[key]:
                    raise ValueError("immutable queue data changed")
            if (after["events"][:-1] != before["events"]
                    or len(after["events"]) != len(before["events"]) + 1):
                raise ValueError("queue history changed")
            for item in db.execute("SELECT * FROM jobs ORDER BY job_key"):
                expected_row = dict(zip(item.keys(), next(
                    old for old in before["jobs"] if old[0] == item["job_key"])))
                if item["job_key"] == row["job_key"]:
                    expected_row.update(status="running", lease_token=token,
                                        lease_until=now + lease_seconds, updated_at=now)
                if dict(item) != expected_row:
                    raise ValueError("unexpected queue mutation")
            lease = queue._lease(current)
            receipt = {
                "status": "committed" if commit else "rolled_back_preflight",
                "at": now, "job_key": row["job_key"], "attempt": row["attempts"],
                "claims": row["claims"], "before_sha256": digest(before),
                "after_sha256": digest(after), **detail,
            }
            if commit:
                db.commit()
            else:
                db.rollback()
                if snapshot(db) != before:
                    raise ValueError("rollback did not restore exact snapshot")
            return lease, receipt
        except BaseException:
            db.rollback()
            raise
