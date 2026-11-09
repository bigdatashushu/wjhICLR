#!/usr/bin/env python3
"""Titan-only admission, original-probe acceptance, backup/archive binding."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.error

import fullqa_custody as c


def complete_ack(expected, actual):
    c.require(set(actual) == set(expected) | {"duplicate"}
              and type(actual["duplicate"]) is bool
              and all(actual[k] == v for k, v in expected.items()), "ack_mismatch")


def collect(root, home, endpoint, final):
    from fullqa_native_acceptance import collect as run_original
    return run_original(root, home, endpoint, final)


def finalize(root, home, endpoint):
    """Executed once in a dedicated process; preserve native archive locking."""
    receipt_path = root / "receipts/fullqa_final_publication_ack_verification.json"
    if not receipt_path.exists():
        collect(root, home, endpoint, True)
    ack_sha = c.digest(receipt_path.read_bytes())
    result_path = root / "receipts/final_archive_with_binding_capacity_v2_result.json"
    if not result_path.exists():
        archive = root.parent / "archive-fullqa-20261024-r01"
        c.require(not archive.exists(), "archive_attempt_requires_review")
        parent = root / "ops/archive_parent_fullqa_capacity_v2.py"
        c.require(c.digest(parent.read_bytes()) == 
                  "38a9271c3b6cbb07289586390f395ed495ff02036c340ebd93a843510433dc73",
                  "capacity_parent_pin_changed")
        # The parent and binder retain the original source checks, success
        # callback, partial preservation, failure receipts and archive lock.
        result = subprocess.run([sys.executable, "-B", str(parent),
                                 "--ack-sha256", ack_sha], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        c.require(result.returncode == 0 and result_path.exists(), "native_archive_not_successful")
    parent_result = c.read(result_path)
    c.require(parent_result["status"] == "verified" and parent_result["ready"] is True
              and parent_result["parent_ack_sha256"] == ack_sha, "archive_result_mismatch")
    binder = c.load_pinned("fullqa_supervisor_binder", root / "ops/final_bind_fullqa_v1.py",
                          "76e115d911c621c8b9671828e5dd31388a778de1201918e223a26a82983b10d1")
    native = binder.load_archive(root / "ops/final_archive_fullqa_v2.py")
    adapter = c.load_pinned("fullqa_supervisor_capacity", root / "ops/archive_receipt_capacity_v1.py",
                           "655b916545067097427717883b3fe9be83503aa4429010ec8f2824c986e3bf50")
    capacity = adapter.ArchiveReceiptCapacity(native)
    try:
        archive = root.parent / "archive-fullqa-20261024-r01"
        evidence, request = binder.validate_local(native, ack=receipt_path,
            manifest=archive / "final.manifest.json", verified=archive / "final.verified.json",
            database=Path(c.read(receipt_path)["titan"]["backup"]["local_file"]),
            method=root / "method_config.json")
        anchor_sha = c.digest(binder.ANCHOR.read_bytes())
        challenge = secrets.token_hex(32)
        current = binder.observe_current(native, anchor_sha, request, challenge)
        result = binder.finalize(native, evidence, request, current, anchor_sha, challenge)
        result.update(execution_host="Titan", exact_database_copy_host="Titan",
                      binder_sha256="76e115d911c621c8b9671828e5dd31388a778de1201918e223a26a82983b10d1")
        binder.install_receipt(native, root / "receipts/fullqa_unattended_final_binding.json", result)
    finally:
        capacity.close()


def slot_tick(root, home, endpoint, config):
    state = c.http(endpoint, "/state", {})
    c.require(state["method_sha256"] == c.METHOD and state["worker_sha256"] == c.WORKER,
              "gateway_pin_changed")
    c.require((home / "handoff_complete.json").exists(), "waiting_owned_controller_handoff")
    with c.ro_database(root) as db:
        rows = db.execute("SELECT * FROM jobs WHERE worker_id=? AND "
                          "status IN ('leased','running','publishing')",
                          (config["worker"]["worker_id"],)).fetchall()
        c.require(len(rows) <= 1, "duplicate_worker_custody")
        audit = c.read(root / "audits/summary.json")
        counts = dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
        if state["legacy"]["live"]:
            return {"status": "waiting_legacy_boundary", "counts": counts}
        if rows:
            from skill3d.cluster.queue import SQLiteJobQueue
            row = rows[0]
            delivery_path = home / "current_delivery.json"
            lease = SQLiteJobQueue._lease(row)
            intent_path = home / "admission_intent.json"
            stored_body = c.recover_delivery(row, lease,
                c.read(delivery_path) if delivery_path.exists() else None,
                c.read(intent_path) if intent_path.exists() else None, config["worker"])
        else:
            for delivery in state["deliveries"]:
                if delivery["phase"] == "quarantined":
                    raise c.Refused("quarantined_slot")
                c.require(delivery["phase"] == "complete", "delivery_not_finished")
                scene = delivery["job_key"].split("/")[-2]
                p = root / "audits/scenes" / (c.digest(delivery["job_key"].encode()) + ".json")
                c.require(p.exists() and c.read(p)["status"] == "verified", "waiting_delivery_audit")
            legacy_key = state["legacy"]["job_key"]
            p = root / "audits/scenes" / (c.digest(legacy_key.encode()) + ".json")
            c.require(p.exists() and c.read(p)["status"] == "verified", "waiting_legacy_audit")
            c.require(not audit["errors"], "audit_errors")
            resources = state["resources"]
            c.require(resources["MemAvailable"] >= 172 * c.GIB
                      and resources["cache_free_bytes"] >= 92 * c.GIB, "memory_reserve")
            intent_path = home / "admission_intent.json"
            c.durable(intent_path, {"worker": config["worker"], "at": time.time(),
                                    "phase": "claiming"})
            lease = c.native_claim(root, config["worker"])
            if lease is None:
                return {"status": "queue_draining", "counts": counts}
            stored_body = None
    c.require(lease.phase == "leased" or (home / "current_delivery.json").exists(),
              "running_attempt_without_delivery")
    delivery_id = c.digest(c.canon({"job_key": lease.job_key, "token": lease.token}))
    body = stored_body or {"delivery_id": delivery_id, "lease": lease.to_dict()}
    c.durable(home / "current_delivery.json", body)
    result = c.http(endpoint, "/deliver", body)
    c.require(result["status"] == "retained" and result["delivery_id"] == delivery_id,
              "delivery_ack_mismatch")
    return {"status": "server_custody_active", "job_key": lease.job_key,
            "delivery_id": delivery_id, "counts": counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    config = c.read(args.config)
    root, home = Path(config["root"]), Path(config["home"])
    sys.path[:0] = [str(root / "source_frozen/src"), str(root / "source_frozen/scripts")]
    c.pins(root)
    if args.finalize:
        finalize(root, home, config["gateway"])
        return
    import fcntl
    with (home / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        c.durable(home / "supervisor_startup.json",
                  {"at": time.time(), "process": c.process(os.getpid())})
        final_process = None
        while True:
            try:
                with c.ro_database(root) as db:
                    counts = dict(db.execute("SELECT status,COUNT(*) FROM jobs GROUP BY status"))
                audit = c.read(root / "audits/summary.json")
                if counts == {"complete": 288} and audit["status"] == "verified":
                    marker = home / "finalization_intent.json"
                    if not marker.exists():
                        backup = root / "backups/queue-online.sqlite3"
                        if not backup.exists():
                            native_report = c.load_pinned("fullqa_final_report",
                                root / "source_frozen/scripts/cluster_report.py",
                                "32fb2ec1b87ca20fb0fdb3ffafac40d43d9c5f55b28225db4d00403d70e34fa2")
                            native_report.backup(root)
                        if not (root / "receipts/fullqa_final_publication_ack_verification.json").exists():
                            collect(root, home, config["gateway"], True)
                        # Persistent before spawning: an uncertain archive
                        # attempt is inspected, never restarted by this loop.
                        c.durable(marker, {"at": time.time(), "status": "requested"}, immutable=True)
                        final_process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                            "--config", str(args.config), "--finalize"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                        c.durable(home / "finalization_process.json",
                                  {"at": time.time(), "process": c.process(final_process.pid)})
                    success = root / "receipts/fullqa_unattended_final_binding.json"
                    status = {"status": "complete" if success.exists() else "finalization_running",
                              "counts": counts, "final_binding": str(success) if success.exists() else None}
                    identity_path = home / "finalization_process.json"
                    if not success.exists() and identity_path.exists():
                        c.require(c.live(c.read(identity_path)["process"]), "finalization_requires_review")
                else:
                    status = slot_tick(root, home, config["gateway"], config)
                status.update(at=time.time(), verified_scenes=audit["verified_scenes"],
                              verified_questions=audit["verified_questions"])
                c.durable(home / "current.json", status)
                if status["status"] == "complete":
                    return
            except Exception as exc:
                c.durable(home / "current.json", {"status": "waiting_or_quarantined", "at": time.time(),
                    "error_type": type(exc).__name__,
                    "reason_code": str(exc) if isinstance(exc, c.Refused) else "transport_unavailable"})
            time.sleep(10)


if __name__ == "__main__":
    main()
