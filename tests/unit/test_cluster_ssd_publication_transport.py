"""Local real pinned Handler/Client/Queue tests; no models or remote actions."""
from concurrent.futures import ThreadPoolExecutor
import base64
from dataclasses import replace
import hashlib
import http.client
from http.server import ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import cluster_coordinator as frozen
import cluster_worker as worker
import cluster_ssd_publication_transport as transport
from skill3d.cluster.identity import FrozenIdentity
from skill3d.cluster.publication import ArtifactDigest, PublicationReceipt
from skill3d.cluster.queue import JobSpec, SQLiteJobQueue, WorkerIdentity

BASE_PATH = ROOT / "scripts" / "cluster_publication_backpressure_v2.py"
BASE = transport.load_base_adapter(BASE_PATH)
# Pin the repository modules exercised by these local tests. Production still
# validates every dependency against the existing task's source_frozen hashes.
PINS = {
    "scripts/cluster_coordinator.py": "7ccc8bcee8a7807b4e295d4b7f11eb937e9bc64e626d64cbee97e2f2d6f5ae94",
    "scripts/cluster_worker.py": "fc0f4c7390c8e33fef0f27af2e1d15203e576f44c65d43b71ae57c9e80e3d716",
    "src/skill3d/cluster/queue.py": "49a3fe57227bc3c836039445c5b54aadd939f09361d068271530f86afabb3b9b",
    "src/skill3d/cluster/publication.py": "b578062d0824a84b2c02ace4d03ab32a5be4b391d9d176ed4908db4b327e67c2",
}


def snapshot(queue):
    with queue._connect(readonly=True) as db:
        db.execute("BEGIN")
        result = {name: [tuple(row) for row in db.execute("SELECT * FROM " + name + " ORDER BY 1")]
                  for name in ("jobs", "dependencies", "events", "completions", "sqlite_sequence")}
        db.rollback()
        return result


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="harness3d-ssd-transport-test-")
        self.root = Path(self.temp.name).resolve()
        self.ssd_root = self.root / "ssd"
        self.identity = FrozenIdentity.freeze({"scene": "test"}, {"method": "fixed"})
        jobs = [JobSpec(f"test-{n}/reconstruction", f"test-{n}/done",
                        self.identity, "reconstruction") for n in range(12)]
        self.job_file = self.root / "jobs.json"
        self.job_file.write_text(json.dumps({"jobs": [j.to_dict() for j in jobs]}))
        self.coordinator = frozen.Coordinator(self.root, self.job_file, reserve_gib=0)
        captured = []

        class Capture:
            def __init__(self, address, handler):
                captured.append(handler)

            def serve_forever(self, **kwargs):
                pass

        with patch.dict(os.environ, {"HARNESS3D_CLUSTER_TOKEN": "fixture-token"}), \
             patch.object(frozen, "ThreadingHTTPServer", Capture), \
             patch.object(frozen, "atomic_json"):
            frozen.serve(self.coordinator, "127.0.0.1", 0)
        self.original_handler = captured[0]
        self.original_handler.date_time_string = lambda self, *args: "Thu, 22 Oct 2026 00:00:00 GMT"
        self.original_publish = self.coordinator.publish
        self.publish_hook = None
        self.publish_calls = []

        def publish(lease, archive, sha, size):
            self.publish_calls.append((lease, archive, sha, size))
            if self.publish_hook is not None:
                self.publish_hook(lease, archive, sha, size)
            return self.original_publish(lease, archive, sha, size)

        self.coordinator.publish = publish
        self.evidence = transport.Evidence(self.root / "evidence.jsonl")
        self.handler = transport.adapt_handler(
            self.original_handler, self.coordinator, self.evidence,
            base_adapter=BASE, ssd_root=self.ssd_root, interim_seconds=0.02)
        self.manager = self.handler.spool_manager
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = worker.Client(f"http://127.0.0.1:{self.server.server_port}", "fixture-token")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.manager.close()
        self.temp.cleanup()

    def rows(self):
        path = self.root / "evidence.jsonl"
        return [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]

    def wait_finished(self, event="publication_finished", count=1):
        deadline = time.monotonic() + 5
        while True:
            rows = [r for r in self.rows() if r["event"] == event]
            if len(rows) >= count:
                return rows
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.005)

    def assert_clean(self):
        self.assertEqual(self.manager.jobs, 0)
        self.assertEqual(self.manager.root_reserved, 0)
        self.assertEqual(self.manager.ssd_reserved, 0)
        self.assertEqual(self.coordinator.publish_slots.active, 0)
        self.assertEqual(list((self.root / transport.PATHS["spool"]).iterdir()), [])
        self.assertEqual(list(self.ssd_root.iterdir()), [])
        self.assertFalse((self.root / ".uploads").exists())

    def publication(self, n=0, *, member=None, metadata_change=None, bad_artifact=False):
        lease = self.coordinator.queue.claim(WorkerIdentity(f"cpu-{n}", "local", "cpu"))
        lease = self.coordinator.queue.start_attempt(lease)
        output = self.root / f"sender-{n}"
        output.mkdir()
        data = b"frozen fixture payload"
        (output / "data.bin").write_bytes(data)
        receipt = PublicationReceipt(
            lease.job_key, lease.spec.completed_key, self.identity.identity_hash,
            "reconstruction", 1, "fixture-validator",
            (ArtifactDigest("data.bin", "0" * 64 if bad_artifact else hashlib.sha256(data).hexdigest(),
                            len(data)),))
        (output / "publication.json").write_text(json.dumps(receipt.to_dict(), sort_keys=True))
        archive = self.root / f"sender-{n}.tar"
        with tarfile.open(archive, "w:") as stream:
            for path in sorted(output.iterdir()):
                stream.add(path, arcname=path.name, recursive=False)
            if member is not None:
                stream.addfile(member, io.BytesIO(b"x" * member.size) if member.isfile() else None)
        metadata = {
            "output_locator": archive.name, "archive_bytes": archive.stat().st_size,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "receipt_sha256": hashlib.sha256((output / "publication.json").read_bytes()).hexdigest(),
        }
        stored = {**metadata, **(metadata_change or {})}
        lease = self.coordinator.queue.finish_attempt(lease, output_metadata=stored)
        return lease, archive, metadata

    def input(self):
        lease = self.coordinator.queue.claim(WorkerIdentity("input-cpu", "local", "cpu"))
        path = self.root / "inputs/jobs" / (hashlib.sha256(lease.job_key.encode()).hexdigest() + ".tar")
        path.parent.mkdir(parents=True)
        path.write_bytes(bytes(range(256)) * 4096)
        return lease, path, path.read_bytes()

    def raw_request(self, route, data, *, headers=None, token="fixture-token", truncate=False, port=None):
        with socket.create_connection(("127.0.0.1", port or self.server.server_port), timeout=5) as connection:
            supplied = {"Host": "localhost", "Authorization": "Bearer " + token,
                        "Content-Length": str(len(data)), "Connection": "close", **(headers or {})}
            request = (f"POST {route} HTTP/1.1\r\n" +
                       "".join(f"{k}: {v}\r\n" for k, v in supplied.items()) + "\r\n").encode()
            connection.sendall(request + data)
            if truncate:
                connection.shutdown(socket.SHUT_WR)
            blocks = []
            try:
                while block := connection.recv(1024 * 1024):
                    blocks.append(block)
            except ConnectionResetError:
                # The frozen HTTP/1.0 handler closes an unread rejected body.
                pass
        raw, count = b"".join(blocks), 0
        while raw.startswith(b"HTTP/1.1 100 Continue\r\n\r\n"):
            raw = raw[len(b"HTTP/1.1 100 Continue\r\n\r\n"):]
            count += 1
        self.assertNotIn(b"HTTP/1.1 100 Continue\r\n\r\n", raw)
        self.assertEqual(raw.count(b"HTTP/1.0 "), 1)
        return count, raw

    def headers(self, lease, metadata, *, full=False):
        wire = lease.to_dict() if full else {
            "schema": "harness3d-lease-header/1", "job_key": lease.job_key,
            "worker": lease.worker.to_dict(), "token": lease.token,
            "expires_at": lease.expires_at, "attempt": lease.attempt,
        }
        return {"X-Lease-JSON": base64.b64encode(json.dumps(wire).encode()).decode(),
                "X-Archive-SHA256": metadata["archive_sha256"]}

    def test_pins_and_no_original_method_replacement(self):
        for name, expected in PINS.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256(BASE_PATH.read_bytes()).hexdigest(), transport.BASE_ADAPTER_SHA256)
        self.assertIs(self.original_publish.__func__, frozen.Coordinator.publish)
        self.assertIs(self.handler.__bases__[0].__bases__[0], self.original_handler)
        self.assertIs(self.handler.authenticated, self.original_handler.authenticated)
        self.assertIs(self.handler.do_GET, self.original_handler.do_GET)
        self.assertTrue(all("_v3" in value for value in transport.PATHS.values()))

    def test_real_client_single_ssd_spool_original_complete_and_expired_idempotent_ack(self):
        args = self.publication()
        lease, archive, metadata = args
        writes = []
        original_open, original_path_open = os.open, Path.open

        def guarded_open(path, flags, *a, **kw):
            if flags & (os.O_WRONLY | os.O_RDWR):
                self.assertNotIn(".uploads", Path(path).parts)
                writes.append(str(path))
            return original_open(path, flags, *a, **kw)

        def guarded_path_open(path, mode="r", *a, **kw):
            if any(c in mode for c in "wax+"):
                self.assertNotIn(".uploads", path.parts)
            return original_path_open(path, mode, *a, **kw)

        def proof(lease, path, sha, size):
            self.assertEqual(path.parent, self.ssd_root)
            self.assertEqual(path.read_bytes(), archive.read_bytes())
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)

        self.publish_hook = proof
        with patch.object(os, "open", guarded_open), patch.object(Path, "open", guarded_path_open):
            first = self.client.publish(*args)
            with patch.object(self.coordinator.queue, "_clock", return_value=lease.expires_at + 10):
                repeated = self.client.publish(*args)
        self.assertFalse(first["duplicate"])
        self.assertTrue(repeated["duplicate"])
        self.assertEqual(first["receipt_hash"], repeated["receipt_hash"])
        self.assertEqual(self.coordinator.queue.progress_summary()["model_attempts"], 1)
        rows = self.wait_finished(count=2)
        for row in rows:
            self.assertEqual(row["schema"], "harness3d-publication-fifo-event/3")
            self.assertEqual(row["bytes_write_count"], metadata["archive_bytes"])
            self.assertEqual(row["archive_sha256"], row["second_archive_sha256"])
            self.assertEqual(row["backend"], "ssd")
            self.assertEqual(row["final_status"], 200)
            self.assertLessEqual(row["receive_finished_at"], row["queue_admitted_at"])
            self.assertLessEqual(row["queue_admitted_at"], row["publish_started_at"])
            self.assertLessEqual(row["publish_finished_at"], row["final_response_at"])
        self.assertEqual(sum(p.endswith(".tar") for p in writes), 2)
        raw = (self.root / "evidence.jsonl").read_text()
        self.assertNotIn("fixture-token", raw)
        self.assertNotIn(lease.token, raw)
        self.assert_clean()

    def test_full_original_lease_header(self):
        lease, archive, metadata = self.publication()
        _, wire = self.raw_request("/publish", archive.read_bytes(),
                                   headers=self.headers(lease, metadata, full=True))
        self.assertTrue(wire.startswith(b"HTTP/1.0 200 OK"))
        self.wait_finished()
        self.assert_clean()

    def test_bad_sha_truncated_and_noncanonical_sha_never_publish(self):
        lease, archive, metadata = self.publication()
        cases = [
            (archive.read_bytes(), {**self.headers(lease, metadata), "X-Archive-SHA256": "0" * 64}, False),
            (archive.read_bytes()[:-1], {**self.headers(lease, metadata),
                                       "Content-Length": str(archive.stat().st_size)}, True),
            (archive.read_bytes(), {**self.headers(lease, metadata),
                                   "X-Archive-SHA256": metadata["archive_sha256"].upper()}, False),
        ]
        for n, (data, headers, truncate) in enumerate(cases, 1):
            _, wire = self.raw_request("/publish", data, headers=headers, truncate=truncate)
            self.assertTrue(wire.startswith(b"HTTP/1.0 409 Conflict"))
            row = self.wait_finished(count=n)[-1]
            if n <= 2:
                self.assertGreaterEqual(row["receive_seconds"], 0)
                self.assertIn("receive_finished_at", row)
                self.assertLessEqual(row["bytes_write_count"], metadata["archive_bytes"])
            self.assert_clean()
        self.assertEqual(self.publish_calls, [])

    def test_malformed_headers_bad_sizes_and_auth_never_allocate_or_read_body(self):
        with patch.object(self.manager, "allocate", side_effect=AssertionError("body allocation")):
            for header in ("!", base64.b64encode(b"[]").decode(), base64.b64encode(b"{}").decode()):
                _, wire = self.raw_request("/publish", b"x", headers={"X-Lease-JSON": header})
                self.assertTrue(wire.startswith(b"HTTP/1.0 409 Conflict"))
            lease, archive, metadata = self.publication()
            for size in ("0", "-1", str(transport.MAX_ARCHIVE_BYTES + 1), "invalid"):
                _, wire = self.raw_request("/publish", b"", headers={
                    **self.headers(lease, metadata), "Content-Length": size})
                self.assertTrue(wire.startswith(b"HTTP/1.0 409 Conflict"))
            count, wire = self.raw_request("/publish", b"x", token="wrong")
            self.assertEqual(count, 0)
            self.assertTrue(wire.startswith(b"HTTP/1.0 403 Forbidden"))
        self.wait_finished(count=7)
        self.assert_clean()

    def test_checkpoint_binding_and_original_artifact_validation(self):
        for n, kwargs in enumerate((
                {"metadata_change": {"archive_sha256": "0" * 64}},
                {"metadata_change": {"archive_bytes": 1}},
                {"metadata_change": {"receipt_sha256": "0" * 64}},
                {"bad_artifact": True})):
            with self.assertRaises(worker.CoordinatorRejected):
                self.client.publish(*self.publication(n, **kwargs))
            self.wait_finished(count=n + 1)
            self.assert_clean()
        self.assertEqual(self.coordinator.queue.progress_summary()["counts"]["complete"], 0)

    def test_original_unsafe_tar_validators(self):
        for n, kind in enumerate(("traversal", "absolute", "symlink", "hardlink", "duplicate")):
            member = tarfile.TarInfo({"traversal": "../escape", "absolute": "/escape",
                                      "duplicate": "data.bin"}.get(kind, kind))
            if kind in ("symlink", "hardlink"):
                member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                member.linkname = "data.bin"
            else:
                member.size = 1
            with self.assertRaises(worker.CoordinatorRejected):
                self.client.publish(*self.publication(n, member=member))
            self.wait_finished(count=n + 1)
            self.assert_clean()
        self.assertFalse((self.root / ".staging/escape").exists())
        self.assertFalse((self.root / "escape").exists())
        self.assertEqual(self.coordinator.queue.progress_summary()["counts"]["complete"], 0)

    def test_stale_lease_and_duplicate_wrong_owner_are_fenced_by_original_queue(self):
        lease, archive, metadata = self.publication()
        with patch.object(self.coordinator.queue, "_clock", return_value=lease.expires_at + 10):
            with self.assertRaises(worker.CoordinatorRejected):
                self.client.publish(lease, archive, metadata)
        first = self.client.publish(lease, archive, metadata)
        self.assertFalse(first["duplicate"])
        for wrong in (replace(lease, token="0" * 64),
                      replace(lease, worker=WorkerIdentity("another", "local", "cpu"))):
            with self.assertRaises(worker.CoordinatorRejected):
                self.client.publish(wrong, archive, metadata)
        self.wait_finished(count=4)
        self.assert_clean()

    def test_ssd_budget_and_reserve_fallback_before_read_keeps_one_hdd_spool(self):
        args = self.publication()
        for n, (name, value, reason) in enumerate((
                ("SSD_BUDGET_BYTES", 1, "ssd_declared_budget"),
                ("SYSTEM_RESERVE_BYTES", 1 << 100, "ssd_system_reserve")), 1):
            with patch.object(transport, name, value):
                result = self.client.publish(*args)
            self.assertEqual(result["status"], "complete")
            row = self.wait_finished(count=n)[-1]
            self.assertEqual(row["backend"], "hdd")
            self.assertEqual(row["fallback_reason"], reason)
            self.assertEqual(Path(row["archive_path"]).parent, self.root / transport.PATHS["spool"])
            self.assertEqual(row["bytes_write_count"], args[2]["archive_bytes"])
            self.assert_clean()

    def test_file_mutation_in_fifo_rejected_before_original_publish(self):
        args = self.publication()
        original_verify = transport.Spool.verify_full

        def mutate(spool):
            os.pwrite(spool.fd, b"X", 0)
            return original_verify(spool)

        with patch.object(transport.Spool, "verify_full", mutate):
            with self.assertRaises(worker.CoordinatorRejected):
                self.client.publish(*args)
        row = self.wait_finished()[0]
        self.assertEqual(row["final_status"], 409)
        self.assertEqual(self.publish_calls, [])
        self.assert_clean()

    def test_second_hash_rejects_corruption_even_if_metadata_check_bypassed(self):
        args = self.publication()
        original_verify = transport.Spool.verify_full

        def mutate(spool):
            os.pwrite(spool.fd, b"X", 0)
            with patch.object(spool, "check_identity"):
                spool.sealed = spool.identity(os.fstat(spool.fd))
                return original_verify(spool)

        with patch.object(transport.Spool, "verify_full", mutate):
            with self.assertRaises(worker.CoordinatorRejected):
                self.client.publish(*args)
        self.wait_finished()
        self.assertEqual(self.publish_calls, [])
        self.assert_clean()

    def test_mutation_after_second_hash_read_rejected_by_final_identity_check(self):
        args = self.publication()
        original_verify, original_read = transport.Spool.verify_full, os.read

        def verify(spool):
            changed = False

            def read(fd, size):
                nonlocal changed
                block = original_read(fd, size)
                if block and not changed:
                    changed = True
                    os.pwrite(spool.fd, b"X", spool.size - 1)
                return block

            with patch.object(os, "read", read):
                return original_verify(spool)

        with patch.object(transport.Spool, "verify_full", verify):
            with self.assertRaises(worker.CoordinatorRejected):
                self.client.publish(*args)
        self.assertEqual(self.wait_finished()[0]["final_status"], 409)
        self.assertEqual(self.publish_calls, [])
        self.assert_clean()

    def test_mutation_during_original_publish_cannot_emit_final_200(self):
        args = self.publication()

        def mutate(lease, path, sha, size):
            # Alter unused tar padding; original artifact validation still
            # succeeds, but the transport's sealed identity must reject ACK.
            path.chmod(0o600)
            with path.open("r+b") as stream:
                stream.seek(size - 1)
                stream.write(b"X")
            path.chmod(0o400)

        self.publish_hook = mutate
        with self.assertRaises(worker.CoordinatorRejected):
            self.client.publish(*args)
        row = self.wait_finished()[0]
        self.assertEqual(row["final_status"], 409)
        self.assertEqual(self.coordinator.queue.progress_summary()["model_attempts"], 1)
        self.assert_clean()

    def test_cleanup_failure_preserves_evidence_and_reservation_before_single_final_409(self):
        args = self.publication()
        original_cleanup, original_unlink = transport.Spool.cleanup, os.unlink
        retained = []

        def cleanup(spool):
            retained.append(spool)

            def unlink(path, **kwargs):
                if path == spool.name:
                    raise OSError("injected cleanup failure")
                return original_unlink(path, **kwargs)

            with patch.object(os, "unlink", unlink):
                return original_cleanup(spool)

        self.publish_hook = lambda *args: time.sleep(0.08)
        with patch.object(transport.Spool, "cleanup", cleanup):
            count, wire = self.raw_request("/publish", args[1].read_bytes(),
                                           headers=self.headers(args[0], args[2]))
        self.assertGreater(count, 0)
        self.assertTrue(wire.startswith(b"HTTP/1.0 409 Conflict"))
        row = self.wait_finished()[0]
        self.assertEqual(row["final_status"], 409)
        self.assertEqual(row["cleanup_error_type"], "OSError")
        self.assertNotIn("spool_cleaned", row)
        spool = retained[0]
        self.assertEqual(spool.path.read_bytes(), args[1].read_bytes())
        self.assertEqual(self.manager.ssd_reserved, spool.size)
        self.assertEqual(self.manager.jobs, 1)
        self.assertEqual(self.coordinator.publish_slots.active, 0)
        self.assertEqual(self.coordinator.queue.progress_summary()["counts"]["complete"], 1)
        self.assertTrue(self.client.publish(*args)["duplicate"])
        self.wait_finished(count=2)
        self.assertEqual(self.coordinator.queue.progress_summary()["model_attempts"], 1)
        spool.path.unlink()
        self.manager.release(spool)
        self.assert_clean()

    def test_hdd_reserve_rejection_before_receive_has_one_final_409(self):
        lease, archive, metadata = self.publication()
        with patch.object(transport.shutil, "disk_usage", return_value=shutil_usage(0)), \
             patch.object(transport.Spool, "receive", side_effect=AssertionError("body read")):
            _, wire = self.raw_request("/publish", archive.read_bytes(),
                                       headers=self.headers(lease, metadata))
        self.assertTrue(wire.startswith(b"HTTP/1.0 409 Conflict"))
        self.wait_finished()
        self.assertEqual(self.publish_calls, [])
        self.assert_clean()

    def test_partial_write_fsync_evidence_and_publish_failures_cleanup(self):
        args = self.publication()
        original_write = os.write
        seen = 0

        def fail_write(fd, data):
            nonlocal seen
            seen += 1
            if seen == 1:
                return original_write(fd, data[:7])
            raise OSError("injected spool write failure")

        with patch.object(os, "write", fail_write):
            with self.assertRaises(worker.CoordinatorRejected):
                self.client.publish(*args)
        self.assertEqual(self.wait_finished()[0]["bytes_write_count"], 7)
        self.assert_clean()
        for n, target in enumerate(("fsync", "evidence", "publish"), 2):
            if target == "fsync":
                original_sync = os.fsync
                fired = False

                def sync(fd):
                    nonlocal fired
                    if not fired and os.fstat(fd).st_mode & 0o777 == 0o400:
                        fired = True
                        raise OSError("injected fsync failure")
                    return original_sync(fd)

                context = patch.object(os, "fsync", sync)
            elif target == "evidence":
                context = patch.object(self.evidence, "write", side_effect=OSError("evidence failure"))
            else:
                def reject(*args):
                    raise frozen.QueueError("original publish rejection")
                self.publish_hook = reject
                context = patch.object(transport, "BLOCK_BYTES", transport.BLOCK_BYTES)
            with context:
                with self.assertRaises(worker.CoordinatorRejected):
                    self.client.publish(*args)
            if target != "evidence":
                self.wait_finished(count=n if target == "fsync" else n - 1)
            self.assert_clean()
        self.assertEqual(self.coordinator.queue.progress_summary()["counts"]["complete"], 0)

    def test_eight_real_publishers_four_fifo_slots_short_idle_deadline_and_single_writes(self):
        inputs = [self.publication(n) for n in range(8)]
        active, peak = 0, 0
        lock = threading.Lock()

        def delay(*args):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
            try:
                time.sleep(0.25)
            finally:
                with lock:
                    active -= 1

        self.publish_hook = delay
        original_connection = http.client.HTTPConnection

        class ShortDeadline(original_connection):
            def __init__(self, host, port, **kwargs):
                super().__init__(host, port, timeout=0.15)

        with patch.object(worker.http.client, "HTTPConnection", ShortDeadline):
            with ThreadPoolExecutor(max_workers=8) as pool:
                acks = list(pool.map(lambda args: self.client.publish(*args), inputs))
        self.assertEqual(peak, 4)
        self.assertTrue(all(ack["status"] == "complete" for ack in acks))
        rows = self.wait_finished(count=8)
        self.assertTrue(all(r["interim_count"] > 0 and
                            r["last_interim_seconds"] < r["pre_header_seconds"] for r in rows))
        self.assertTrue(all(r["bytes_write_count"] == r["archive_bytes"] for r in rows))
        self.assertEqual(self.coordinator.queue.progress_summary()["model_attempts"], 8)
        self.assert_clean()

    def test_input_original_hash_hooks_final_wire_and_auth_unchanged(self):
        lease, path, data = self.input()
        before = snapshot(self.coordinator.queue)
        body = json.dumps({"lease": lease.to_dict()}).encode()
        server = ThreadingHTTPServer(("127.0.0.1", 0), self.original_handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        original_hash = frozen.hash_file

        def delayed(path):
            time.sleep(0.25)
            return original_hash(path)

        try:
            _, original_wire = self.raw_request("/input", body, port=server.server_port)
            with patch.object(frozen, "hash_file", delayed):
                count, wire = self.raw_request("/input", body)
                self.client.download(lease, self.root / "download.tar")
            self.assertGreater(count, 0)
            self.assertEqual(wire, original_wire)
            self.assertEqual((self.root / "download.tar").read_bytes(), data)
            count, wire = self.raw_request("/input", body, token="wrong")
            self.assertEqual(count, 0)
            self.assertTrue(wire.startswith(b"HTTP/1.0 403 Forbidden"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        rows = self.wait_finished("input_request_finished", 3)
        self.assertEqual(rows[0]["body_bytes_written"], len(data))
        self.assertEqual(rows[0]["content_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(before, snapshot(self.coordinator.queue))
        self.assert_clean()

    def test_input_corrupt_body_and_evidence_failure_preserve_existing_behavior(self):
        lease, path, data = self.input()
        before = snapshot(self.coordinator.queue)

        def corrupt(stream, output, length):
            block = stream.read()
            output.write(bytes([block[0] ^ 1]) + block[1:])

        with patch.object(frozen.shutil, "copyfileobj", corrupt):
            with self.assertRaisesRegex(worker.TransportError, "input transport digest mismatch"):
                self.client.download(lease, self.root / "bad.tar")
        with patch.object(self.evidence, "write", side_effect=OSError("evidence failure")):
            self.client.download(lease, self.root / "good.tar")
        self.assertEqual((self.root / "good.tar").read_bytes(), data)
        self.assertFalse((self.root / "bad.part").exists())
        self.assertEqual(before, snapshot(self.coordinator.queue))


class ReservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="harness3d-spool-budget-test-")
        self.root = Path(self.temp.name).resolve()
        self.manager = transport.SpoolManager(self.root, self.root / "ssd", 0)

    def tearDown(self):
        self.manager.close()
        self.temp.cleanup()

    def test_concurrent_declared_reservations_capacity_and_hdd_fallback(self):
        with patch.object(transport, "SSD_BUDGET_BYTES", 300), \
             patch.object(transport, "SYSTEM_RESERVE_BYTES", 0):
            with ThreadPoolExecutor(max_workers=8) as pool:
                spools = list(pool.map(self.manager.allocate, [100] * 8))
        self.assertEqual(sum(s.backend == "ssd" for s in spools), 3)
        self.assertEqual(self.manager.ssd_reserved, 300)
        self.assertEqual(self.manager.root_reserved, 1600)
        self.assertEqual(self.manager.jobs, 8)
        with self.assertRaisesRegex(RuntimeError, "job capacity"):
            self.manager.allocate(1)
        for spool in spools:
            self.assertEqual(spool.bytes_write_count, 0)
            spool.cleanup()
        self.assertEqual(self.manager.jobs, self.manager.ssd_reserved)

    def test_parallel_reserve_does_not_double_admit_unallocated_bytes(self):
        usage = shutil_usage(transport.SYSTEM_RESERVE_BYTES + 150)
        with patch.object(transport.shutil, "disk_usage", return_value=usage):
            first = self.manager.allocate(100)
            second = self.manager.allocate(100)
        self.assertEqual(first.backend, "ssd")
        self.assertEqual(second.backend, "hdd")
        first.cleanup()
        second.cleanup()

    def test_original_hdd_reserve_failure_before_body_or_file(self):
        with patch.object(transport.shutil, "disk_usage", return_value=shutil_usage(199)):
            with self.assertRaisesRegex(ValueError, "reserve"):
                self.manager.allocate(100)
        self.assertEqual(self.manager.jobs, 0)
        self.assertEqual(list(self.manager.hdd.path.iterdir()), [])
        self.assertEqual(list(self.manager.ssd.path.iterdir()), [])

    def test_linux_fallocate_enospc_falls_back_before_body(self):
        allocations = []

        def no_space(fd, offset, size):
            allocations.append((offset, size))
            raise OSError(28, "no space")

        with patch.object(transport.sys, "platform", "linux"), \
             patch.object(os, "posix_fallocate", no_space, create=True):
            spool = self.manager.allocate(100)
        self.assertEqual(allocations, [(0, 100)])
        self.assertEqual(spool.backend, "hdd")
        self.assertEqual(spool.bytes_write_count, 0)
        self.assertEqual(list(self.manager.ssd.path.iterdir()), [])
        spool.cleanup()
        self.assertEqual(self.manager.ssd_reserved, 0)

    def test_linux_fallocate_success_and_unsupported_keep_single_spool(self):
        for error in (None, OSError(transport.errno.EOPNOTSUPP, "unsupported")):
            with patch.object(transport.sys, "platform", "linux"), \
                 patch.object(os, "posix_fallocate", side_effect=error, create=True) as allocation:
                spool = self.manager.allocate(100)
            allocation.assert_called_once_with(spool.fd, 0, 100)
            self.assertEqual(spool.backend, "ssd")
            spool.receive(io.BytesIO(b"x" * 100), hashlib.sha256(b"x" * 100).hexdigest())
            self.assertEqual(spool.verify_full(), spool.archive_sha256)
            spool.cleanup()

    def test_allocation_cleanup_failure_does_not_fallback_or_drop_reservation(self):
        original_unlink = os.unlink

        def unlink(name, **kwargs):
            if kwargs.get("dir_fd") == self.manager.ssd.fd:
                raise OSError("retain failed SSD allocation")
            return original_unlink(name, **kwargs)

        with patch.object(transport.sys, "platform", "linux"), \
             patch.object(os, "posix_fallocate", side_effect=OSError(28, "no space"), create=True), \
             patch.object(os, "unlink", unlink):
            with self.assertRaisesRegex(RuntimeError, "allocation cleanup failed"):
                self.manager.allocate(100)
        self.assertEqual(list(self.manager.hdd.path.iterdir()), [])
        self.assertEqual(self.manager.jobs, 1)
        self.assertEqual(self.manager.root_reserved, 200)
        self.assertEqual(self.manager.ssd_reserved, 100)
        retained = list(self.manager.ssd.path.iterdir())
        self.assertEqual(len(retained), 1)
        # Explicit fixture-owner cleanup; production retains this reservation.
        retained[0].unlink()
        self.manager.jobs = self.manager.root_reserved = self.manager.ssd_reserved = 0

    def test_unavailable_ssd_selects_hdd(self):
        other = self.root / "other"
        other.mkdir()
        # Only the SSD directory creation fails; the HDD remains real.
        original_init = transport.PrivateDirectory.__init__

        def create(instance, path):
            if Path(path).name == "missing":
                raise PermissionError("SSD not writable")
            original_init(instance, path)

        with patch.object(transport.PrivateDirectory, "__init__", create):
            manager = transport.SpoolManager(other, other / "missing", 0)
        try:
            spool = manager.allocate(100)
            self.assertEqual(spool.backend, "hdd")
            self.assertEqual(spool.fallback_reason, "PermissionError")
            spool.cleanup()
        finally:
            manager.close()

    def test_nonempty_private_link_and_owner_violations_fail_closed_without_deletion(self):
        for kind in ("nonempty", "permissions", "symlink", "file"):
            root = self.root / kind
            root.mkdir()
            ssd = root / "ssd"
            if kind == "symlink":
                ssd.symlink_to(self.manager.ssd.path, target_is_directory=True)
            elif kind == "file":
                ssd.write_bytes(b"existing non-directory evidence")
            else:
                ssd.mkdir(mode=0o700)
                if kind == "nonempty":
                    (ssd / "prior.tar").write_bytes(b"prior evidence")
                else:
                    ssd.chmod(0o755)
            with self.assertRaises(RuntimeError):
                transport.SpoolManager(root, ssd, 0)
            if kind == "nonempty":
                self.assertEqual((ssd / "prior.tar").read_bytes(), b"prior evidence")
            elif kind == "file":
                self.assertEqual(ssd.read_bytes(), b"existing non-directory evidence")
        root = self.root / "hdd"
        root.mkdir()
        hdd = root / transport.PATHS["spool"]
        hdd.mkdir(mode=0o700)
        (hdd / "prior.tar").write_bytes(b"preserve")
        with self.assertRaises(RuntimeError):
            transport.SpoolManager(root, root / "ssd", 0)
        self.assertEqual((hdd / "prior.tar").read_bytes(), b"preserve")

    def test_replacement_hardlink_and_trailing_file_bytes_rejected(self):
        for kind in ("replace", "hardlink", "append"):
            spool = self.manager.allocate(100)
            spool.receive(io.BytesIO(b"x" * 100), hashlib.sha256(b"x" * 100).hexdigest())
            if kind == "replace":
                spool.path.unlink()
                spool.path.write_bytes(b"x" * 100)
            elif kind == "hardlink":
                os.link(spool.path, spool.path.with_suffix(".link"))
            else:
                os.pwrite(spool.fd, b"x", 100)
            with self.assertRaises(RuntimeError):
                spool.verify_full()
            if kind == "replace":
                with self.assertRaisesRegex(RuntimeError, "refusing to clean"):
                    spool.cleanup()
                self.assertEqual(spool.path.read_bytes(), b"x" * 100)
                self.assertEqual(self.manager.jobs, 1)
                # Test owns the replacement; explicitly remove it and reset the
                # fixture reservation, never let production clean prior evidence.
                spool.path.unlink()
                self.manager.release(spool)
            else:
                if kind == "hardlink":
                    spool.path.with_suffix(".link").unlink()
                spool.cleanup()


def shutil_usage(free):
    return transport.shutil._ntuple_diskusage(free * 2, free, free)


class WarmAndActivationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="harness3d-warm-transport-test-")
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_230_full_readonly_constructor_during_original_server_and_write_barrier(self):
        identity = FrozenIdentity.freeze({"scene": "warm"}, {"method": "fixed"})
        specs = [JobSpec(f"warm-{n}/reconstruction", f"warm-{n}/done", identity, "reconstruction")
                 for n in range(230)]
        jobs = self.root / "jobs.json"
        jobs.write_text(json.dumps({"jobs": [j.to_dict() for j in specs]}))
        original = frozen.Coordinator(self.root, jobs, reserve_gib=0)
        before = snapshot(original.queue)
        captured = []

        class Capture:
            def __init__(self, address, handler):
                captured.append(handler)

            def serve_forever(self, **kwargs):
                pass

        with patch.dict(os.environ, {"HARNESS3D_CLUSTER_TOKEN": "fixture-token"}), \
             patch.object(frozen, "ThreadingHTTPServer", Capture), patch.object(frozen, "atomic_json"):
            frozen.serve(original, "127.0.0.1", 0)
        server = ThreadingHTTPServer(("127.0.0.1", 0), captured[0])
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sqlite3.connect(original.queue.db_path, isolation_level=None) as writer:
                writer.execute("BEGIN IMMEDIATE")
                with patch.object(SQLiteJobQueue, "_write", side_effect=AssertionError("warm write")), \
                     patch.object(SQLiteJobQueue, "__init__", side_effect=AssertionError("schema write")), \
                     patch.object(original.queue, "_clock", return_value=time.time()):
                    warm = BASE.build_warm_coordinator(frozen, self.root, jobs, reserve_gib=0)
                    connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
                    try:
                        connection.request("GET", "/progress", headers={"Authorization": "Bearer fixture-token"})
                        response = connection.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(json.loads(response.read()), original.queue.progress_summary())
                    finally:
                        connection.close()
                writer.rollback()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.assertEqual(before, snapshot(original.queue))
        self.assertEqual(warm.warm_admission["existing_jobs_verified"], 230)
        self.assertEqual(warm.warm_admission["write_transactions"], 0)
        self.assertIs(warm.queue.enqueue_many.__func__, SQLiteJobQueue.enqueue_many)
        self.assertIs(frozen.SQLiteJobQueue, SQLiteJobQueue)

    def command(self):
        return {"status": "old_coordinator_exited", "method_config_sha256": transport.METHOD_SHA256,
                "adapter_sha256": "a" * 64, "base_adapter_sha256": transport.BASE_ADAPTER_SHA256,
                "warm_pid": 99991, "old_pid": 99992, "ssd_root": str(self.root / "ssd")}

    def validate(self, command=None):
        transport.validate_activation(command or self.command(), self.root,
                                      adapter_sha256="a" * 64, warm_pid=99991,
                                      ssd_root=self.root / "ssd")

    def test_activation_exact_pids_all_old_and_new_empty_dirs_and_pins(self):
        with patch.object(transport.Path, "is_file", return_value=True):
            self.validate()
            for key, value in (("old_pid", True), ("old_pid", 99991), ("warm_pid", 99993),
                               ("method_config_sha256", "wrong"), ("adapter_sha256", "wrong"),
                               ("base_adapter_sha256", "wrong"), ("status", "prepared"),
                               ("ssd_root", "elsewhere")):
                with self.assertRaises(RuntimeError):
                    self.validate({**self.command(), key: value})
            with patch.object(transport.os.path, "lexists", side_effect=lambda p: str(p) == "/proc/99992"):
                with self.assertRaisesRegex(RuntimeError, "exact old PID"):
                    self.validate()
            for name in (*transport.TEMP_NAMES, "ssd"):
                path = self.root / name
                path.mkdir(mode=0o700)
                marker = path / "prior"
                marker.write_bytes(b"preserve")
                with self.assertRaisesRegex(RuntimeError, "empty"):
                    self.validate()
                self.assertEqual(marker.read_bytes(), b"preserve")
                marker.unlink()
        with patch.object(transport.Path, "is_file", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Linux"):
                self.validate()

    def test_frozen_imports_all_method_pins_and_loaded_dependencies_are_bound(self):
        source = self.root / "source_frozen"
        modules, pins = {}, {}
        for name, loaded in list(sys.modules.items()):
            if name == "cluster_coordinator" or name == "skill3d" or name.startswith("skill3d."):
                path = Path(loaded.__file__).resolve()
                relative = path.relative_to(ROOT.resolve())
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                raw = path.read_bytes()
                target.write_bytes(raw)
                pins[str(relative)] = hashlib.sha256(raw).hexdigest()
                proxy = transport.types.ModuleType(name)
                proxy.__file__ = str(target)
                modules[name] = proxy
        coordinator = modules["cluster_coordinator"]
        coordinator.SQLiteJobQueue = SQLiteJobQueue
        with patch.dict(sys.modules, modules), patch.object(sys, "path", list(sys.path)):
            self.assertIs(transport.verify_frozen_imports(self.root, {"code_sha256": pins}), coordinator)
            self.assertEqual(sys.path[:2], [str(source / "scripts"), str(source / "src")])
            for bad in ({}, {**pins, "../escape.py": "0" * 64},
                        {**pins, "scripts/cluster_coordinator.py": "0" * 64}):
                with self.assertRaises(RuntimeError):
                    transport.verify_frozen_imports(self.root, {"code_sha256": bad})
            unbound = {name: sha for name, sha in pins.items()
                       if name != "src/skill3d/cluster/publication.py"}
            with self.assertRaisesRegex(RuntimeError, "unbound frozen dependency"):
                transport.verify_frozen_imports(self.root, {"code_sha256": unbound})
            with patch.object(modules["skill3d"], "__file__", str(ROOT / "src/skill3d/__init__.py")):
                with self.assertRaisesRegex(RuntimeError, "outside source_frozen"):
                    transport.verify_frozen_imports(self.root, {"code_sha256": pins})
            with patch.object(coordinator, "__file__", frozen.__file__):
                with self.assertRaisesRegex(RuntimeError, "coordinator import"):
                    transport.verify_frozen_imports(self.root, {"code_sha256": pins})

    def test_explicit_base_pin_and_cli_method_ssd_constraints(self):
        wrong = self.root / "wrong.py"
        wrong.write_text("raise AssertionError('must not execute')")
        with self.assertRaisesRegex(RuntimeError, "base adapter SHA256"):
            transport.load_base_adapter(wrong)
        with self.assertRaises(SystemExit):
            transport.main(["--root", str(self.root), "--base-adapter", str(BASE_PATH),
                            "--ssd-root", str(transport.SSD_ROOT),
                            "--expected-method-sha256", "0" * 64])
        with self.assertRaisesRegex(RuntimeError, "explicit v3"):
            transport.main(["--root", str(self.root), "--base-adapter", str(BASE_PATH),
                            "--ssd-root", str(self.root / "wrong"),
                            "--expected-method-sha256", transport.METHOD_SHA256])


if __name__ == "__main__":
    unittest.main(verbosity=2)
