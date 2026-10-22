#!/usr/bin/env python3
"""Drain an existing queue through separate, compute-free control-plane code.

Keep the frozen coordinator, model code and artifacts intact. This server uses
the normal handlers and only restricts claim admission to publication recovery.
Deploy only after stopping the receipt-identified coordinator on the same port.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import secrets
import time

from cluster_coordinator import Coordinator, atomic_json, serve
from skill3d.cluster.identity import QueueError
from skill3d.cluster.queue import STAGES, SQLiteJobQueue, WorkerIdentity


class PublicationDrainQueue(SQLiteJobQueue):
    """Keep leases alive and admit retained outputs; never claim model work."""

    def claim(self, worker: WorkerIdentity, *, lease_seconds: float = 300,
              stages=None):
        if not isinstance(worker, WorkerIdentity):
            raise QueueError("claim requires WorkerIdentity")
        worker.__post_init__()
        duration = self._positive(lease_seconds, "lease_seconds")
        permitted = tuple(STAGES if stages is None else stages)
        if not permitted or any(stage not in STAGES for stage in permitted):
            raise QueueError("claim stages must be a nonempty supported stage list")
        with self._write() as db:
            now = self._now()
            self._settle(db, now)
            placeholders = ",".join("?" for _ in permitted)
            row = db.execute(
                "SELECT j.* FROM jobs j WHERE j.status='queued' "
                "AND j.resume_phase='publish' AND j.available_at<=? "
                f"AND j.stage IN ({placeholders}) AND NOT EXISTS ("
                "SELECT 1 FROM dependencies d JOIN jobs u ON u.job_key=d.depends_on "
                "WHERE d.job_key=j.job_key AND NOT (u.status='complete' OR "
                "(u.status='skipped' AND j.dependency_policy='complete_or_skipped'))) "
                "ORDER BY j.priority DESC,j.created_at,j.job_key LIMIT 1",
                (now, *permitted),
            ).fetchone()
            if row is None:
                return None
            self._verify_dependencies(db, row)
            db.execute(
                "UPDATE jobs SET status='publishing',lease_token=?,lease_until=?,"
                "worker_id=?,host=?,gpu_id=?,claims=claims+1,updated_at=? WHERE job_key=?",
                (secrets.token_hex(32), now + duration, worker.worker_id, worker.host,
                 worker.gpu_id, now, row["job_key"]),
            )
            current = db.execute("SELECT * FROM jobs WHERE job_key=?",
                                 (row["job_key"],)).fetchone()
            self._event(db, current, "claimed", now,
                        {"phase": "publishing", "admission": "publication-drain"})
            return self._lease(current)


def draining_coordinator(root: Path, jobs: Path, *, reserve_gib: int = 160):
    coordinator = Coordinator(root, jobs, reserve_gib=reserve_gib)
    coordinator.queue = PublicationDrainQueue(coordinator.root / "state/jobs.sqlite3")
    return coordinator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=27033)
    args = parser.parse_args()
    coordinator = draining_coordinator(args.root, args.jobs)
    stat = Path("/proc/self/stat")
    atomic_json(coordinator.root / "receipts/drain_launch.json", {
        "schema": "harness3d-publication-drain-launch/1",
        "pid": os.getpid(), "started_at": time.time(),
        "start_ticks": stat.read_text().split()[21] if stat.exists() else None,
        "host": args.host, "port": args.port,
        "admission": "publication-only",
        "control_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_handler_sha256": hashlib.sha256(
            Path(serve.__code__.co_filename).read_bytes()).hexdigest(),
        "queue_before": coordinator.queue.progress_summary(),
        "model_code_modified": False,
        "phase": "launching; verify coordinator_startup.json and authenticated progress",
    })
    serve(coordinator, args.host, args.port)


if __name__ == "__main__":
    main()
