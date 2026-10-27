"""Isolated engineering tests using only synthetic temporary payloads.

Run without collecting the application's online tests:
PYTHONPATH=src python3 -m unittest discover -s tests -p test_hot_cache.py -v
"""

from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from skill3d.storage.hot_cache import (
    ArchiveAck, ArchiveIdentity, ArchiveReceiver, DEFAULT_QUOTA_BYTES, GB,
    HotCache, IdentityConflict, IntegrityError, MAX_QUOTA_BYTES, MIN_FREE_BYTES,
    EntryBusy, QuotaExceeded, StorageError,
)

MiB = 1024 * 1024


def free_disk(_path):
    return shutil._ntuple_diskusage(2_000 * GB, 500 * GB, 1_500 * GB)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def bind_archive(manifest, path):
    """Rebind outer hashes so tests reach the member/parser verification layer."""
    data = deepcopy(manifest)
    data["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    data["archive_bytes"] = path.stat().st_size
    body = {k: v for k, v in data.items() if k != "manifest_sha256"}
    data["manifest_sha256"] = digest(body)
    return data


class HotCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3d-hot-cache-test-")
        self.root = Path(self.temp.name)
        self.cache = HotCache(self.root / "cache", quota_bytes=4 * MiB,
                              disk_usage=free_disk)
        self.receiver = ArchiveReceiver(self.root / "titan", disk_usage=free_disk)

    def tearDown(self):
        self.temp.cleanup()

    def put(self, key="result", data=b"synthetic-output", kind="result", slots=()):
        return self.cache.populate(
            key, lambda r: r.write_bytes("payload/data.bin", data),
            kind=kind, output_bytes=len(data), slots=slots,
        )

    def package(self, key="result", **kwargs):
        options = {"job": "engineering-job", "shard": "synthetic-shard", "attempt": 0}
        options.update(kwargs)
        return self.cache.queue_archive(key, **options)

    def other_handle(self):
        return HotCache(self.cache.root, quota_bytes=self.cache.quota_bytes,
                        disk_usage=free_disk)

    def child(self, code):
        result = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def child_prelude(self):
        return (
            "import os, shutil\n"
            "from pathlib import Path\n"
            "from skill3d.storage.hot_cache import HotCache, ArchiveReceiver, GB\n"
            "def disk(path): return shutil._ntuple_diskusage(2000*GB,500*GB,1500*GB)\n"
            f"root = Path({str(self.cache.root)!r})\n"
            f"cache = HotCache(root, quota_bytes={self.cache.quota_bytes}, disk_usage=disk)\n"
        )

    def malicious_tar(self, members, *, raw_tail=b""):
        path = self.root / ("malicious-" + str(len(list(self.root.glob("malicious-*")))) + ".tar.gz")
        with tarfile.open(path, "w:gz", format=tarfile.USTAR_FORMAT) as tar:
            for info, data in members:
                tar.addfile(info, io.BytesIO(data) if info.isreg() else None)
        if raw_tail:
            with path.open("ab") as stream:
                stream.write(gzip.compress(raw_tail, mtime=0))
        return path

    def assert_rejected(self, path, manifest):
        with self.assertRaises(IntegrityError):
            self.receiver.receive(path, manifest)
        self.assertEqual(list((self.receiver.root / ".incoming").iterdir()), [])
        self.assertEqual(list((self.receiver.root / "objects").iterdir()), [])

    def test_defaults_and_hard_limits(self):
        self.assertEqual(DEFAULT_QUOTA_BYTES, 35 * GB)
        self.assertLessEqual(DEFAULT_QUOTA_BYTES, MAX_QUOTA_BYTES)
        self.assertEqual(MIN_FREE_BYTES, 25 * GB)
        for quota in (0, MAX_QUOTA_BYTES + 1, True, -1):
            with self.subTest(quota=quota), self.assertRaises(StorageError):
                HotCache(self.root / "invalid", quota_bytes=quota, disk_usage=free_disk)
        with self.assertRaises(StorageError):
            HotCache(self.root / "low-floor", min_free_bytes=MIN_FREE_BYTES - 1)
        with self.assertRaises(StorageError):
            ArchiveReceiver(self.root / "low-receiver-floor", min_free_bytes=0)
        with self.assertRaises(StorageError):
            HotCache(self.cache.root, quota_bytes=3 * MiB, disk_usage=free_disk)

    def test_sum_of_download_extract_output_and_compression_reservations(self):
        with self.cache.reserve(download_bytes=100, extract_bytes=200,
                                output_bytes=300, compression_bytes=400) as r:
            self.assertEqual(r.bytes, 1000)
            self.assertGreaterEqual(self.cache.stats()["reserved_bytes"], 1000)
            charged_before = self.cache.stats()["charged_bytes"]
            r.write_bytes("download.bin", b"x" * 100)
            self.assertLess(self.cache.stats()["reserved_bytes"], r.charged_bytes)
            self.assertEqual(self.cache.stats()["charged_bytes"], charged_before)
            with self.cache.reserve(output_bytes=1000):
                self.assertGreater(self.cache.stats()["charged_bytes"], charged_before)
        self.assertEqual(self.cache.stats()["reservations"], 0)

    def test_bounded_writer_refuses_over_budget(self):
        with self.cache.reserve(output_bytes=3) as r:
            with self.assertRaises(QuotaExceeded):
                r.write_bytes("too-large.bin", b"four")
            self.assertEqual((r.path / "too-large.bin").stat().st_size, 0)
        self.assertEqual(self.cache.stats()["reservations"], 0)

    def test_shared_quota_denies_second_reservation(self):
        with self.cache.reserve(output_bytes=3 * MiB):
            with self.assertRaises(QuotaExceeded):
                self.other_handle().reserve(download_bytes=MiB)
        self.assertEqual(self.cache.stats()["reservations"], 0)

    def test_filesystem_floor_includes_unwritten_reservations(self):
        high = self.cache.disk_usage
        self.cache.disk_usage = lambda p: shutil._ntuple_diskusage(
            2000 * GB, 500 * GB, MIN_FREE_BYTES + 300_000)
        try:
            with self.cache.reserve(output_bytes=100_000):
                with self.assertRaises(QuotaExceeded):
                    self.cache.reserve(extract_bytes=100_000)
        finally:
            self.cache.disk_usage = high

    def test_write_checks_actual_free_space(self):
        with self.cache.reserve(output_bytes=10) as r:
            self.cache.disk_usage = lambda p: shutil._ntuple_diskusage(
                2000 * GB, 500 * GB, MIN_FREE_BYTES)
            try:
                with self.assertRaises(QuotaExceeded):
                    r.write_bytes("blocked.bin", b"x")
            finally:
                self.cache.disk_usage = free_disk

    def test_current_next_share_quota_and_are_not_evicted(self):
        current = self.put("current", b"c" * MiB, "input", ("current",))
        next_path = self.put("next", b"n" * MiB, "input", ("next",))
        with self.assertRaises(QuotaExceeded):
            self.cache.reserve(output_bytes=2 * MiB)
        self.assertTrue(current.exists())
        self.assertTrue(next_path.exists())
        self.assertEqual(self.cache.stats()["slots"], {"current": "current", "next": "next"})

    def test_lru_evicts_only_inactive_unpinned_inputs(self):
        old = self.put("old", b"x" * MiB, "input")
        retained = self.put("keep", b"y" * MiB, "input", ("current",))
        with self.cache.reserve(output_bytes=2 * MiB):
            self.assertFalse(old.exists())
            self.assertTrue(retained.exists())

    def test_scene_lease_blocks_eviction_across_handles(self):
        active = self.put("active", b"x" * (2 * MiB), "input")
        with self.cache.lease("active"):
            with self.assertRaises(QuotaExceeded):
                self.other_handle().reserve(output_bytes=2 * MiB)
            self.assertTrue(active.exists())
            with self.assertRaises(EntryBusy):
                with self.other_handle().lease("active", exclusive=True):
                    pass
        with self.other_handle().reserve(output_bytes=2 * MiB):
            self.assertFalse(active.exists())

    def test_scene_lease_blocks_eviction_across_processes(self):
        active = self.put("active", b"x" * (2 * MiB), "input")
        code = self.child_prelude() + (
            "from skill3d.storage.hot_cache import QuotaExceeded\n"
            "try:\n"
            f"    cache.reserve(output_bytes={2 * MiB})\n"
            "except QuotaExceeded:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('active entry evicted')\n"
        )
        with self.cache.lease("active"):
            self.child(code)
        self.assertTrue(active.exists())

    def test_pending_results_cannot_be_evicted_for_space(self):
        path = self.put(data=b"r" * (2 * MiB))
        with self.assertRaises(QuotaExceeded):
            self.cache.reserve(output_bytes=2 * MiB)
        with self.assertRaises(StorageError):
            self.cache.collect_acknowledged("result")
        self.assertTrue(path.exists())

    def test_no_ack_and_wrong_ack_never_delete(self):
        path = self.put()
        package = self.package()
        self.assertIsNone(self.cache.transfer_one("result", lambda p, m: None,
                                                 delete_acked=True))
        wrong = ArchiveAck(ArchiveIdentity("wrong-job", "synthetic-shard", 0,
                                          package.identity.sha256),
                           package.manifest["manifest_sha256"])
        with self.assertRaises(IdentityConflict):
            self.cache.transfer_one("result", lambda p, m: wrong, delete_acked=True)
        self.assertTrue(path.exists())
        self.assertTrue(package.path.exists())

    def test_real_transfer_then_matching_ack_allows_delete(self):
        path = self.put()
        package = self.package()
        ack = self.cache.transfer_one("result", self.receiver, delete_acked=True)
        self.assertEqual(ack.identity, package.identity)
        self.assertFalse(path.exists())
        self.assertFalse(package.path.exists())
        self.assertEqual(self.receiver.published_path(ack.identity).joinpath(
            "data.bin").read_bytes(), b"synthetic-output")
        self.assertFalse(self.cache.collect_acknowledged("result"))

    def test_duplicate_ack_is_idempotent_and_skips_retransfer(self):
        self.put()
        package = self.package()
        first = self.receiver(package.path, package.manifest)
        second = self.receiver(package.path, package.manifest)
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.receiver.root / "objects").iterdir())), 1)
        self.assertEqual(self.cache.transfer_one("result", lambda p, m: first.as_dict()), first)
        def unexpected(p, m):
            self.fail("ACKed identity was retransferred")
        self.assertEqual(self.other_handle().transfer_one("result", unexpected), first)

    def test_conflicting_publication_for_same_job_shard_attempt(self):
        self.put("a", b"one")
        self.put("b", b"two")
        first, second = self.package("a"), self.package("b")
        self.receiver(first.path, first.manifest)
        with self.assertRaises(IdentityConflict):
            self.receiver(second.path, second.manifest)
        with self.assertRaises(IdentityConflict):
            self.package("a", attempt=1)
        self.assertEqual(self.receiver.published_path(first.identity).joinpath(
            "data.bin").read_bytes(), b"one")

    def test_transport_failure_and_restart_reuse_identity_without_inference(self):
        producer_calls = []
        self.cache.populate(
            "result", lambda r: (producer_calls.append(1),
                                 r.write_bytes("payload/data.bin", b"saved-output")),
            kind="result", output_bytes=12,
        )
        package = self.package()
        seen = []
        def lost_response(path, manifest):
            seen.append(manifest)
            self.receiver(path, manifest)
            raise ConnectionError("synthetic lost ACK")
        with self.assertRaises(ConnectionError):
            self.cache.transfer_one("result", lost_response, delete_acked=True)
        resumed = self.other_handle()
        def retransmit(path, manifest):
            seen.append(manifest)
            return self.receiver(path, manifest)
        ack = resumed.transfer_one("result", retransmit, delete_acked=True)
        self.assertEqual(ack.identity, package.identity)
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(producer_calls, [1])

    def test_latest_best_retained_old_checkpoint_waits_for_ack(self):
        old = self.put("old", b"old-state", "checkpoint", ("latest", "best"))
        latest = self.put("new", b"new-state", "checkpoint", ("latest",))
        self.assertNotIn("old", self.cache.archive_candidates())
        self.cache.pin("best", "new")
        self.assertIn("old", self.cache.archive_candidates())
        with self.assertRaises(StorageError):
            self.cache.collect_acknowledged("old")
        self.assertTrue(old.exists())
        self.package("old")
        self.cache.transfer_one("old", self.receiver, delete_acked=True)
        self.assertFalse(old.exists())
        self.package("new", shard="new-checkpoint")
        self.cache.transfer_one("new", self.receiver, delete_acked=True)
        self.assertTrue(latest.exists())
        self.assertEqual(self.cache.stats()["slots"], {"latest": "new", "best": "new"})

    def test_active_acked_entry_cannot_be_deleted_or_evicted(self):
        path = self.put(data=b"r" * (2 * MiB))
        self.package(compression_bytes=128 * 1024)
        self.cache.transfer_one("result", self.receiver)
        with self.cache.lease("result"):
            self.assertFalse(self.cache.collect_acknowledged("result"))
            with self.assertRaises(QuotaExceeded):
                self.other_handle().reserve(output_bytes=2 * MiB)
        self.assertTrue(self.cache.collect_acknowledged("result"))
        self.assertFalse(path.exists())

    def test_mutated_local_payload_not_deleted_by_old_ack(self):
        path = self.put()
        self.package()
        self.cache.transfer_one("result", self.receiver)
        with self.cache.lease("result", exclusive=True):
            (path / "data.bin").write_bytes(b"changed-after-archive")
        with self.assertRaises(IntegrityError):
            self.cache.collect_acknowledged("result")
        self.assertTrue(path.exists())

    def test_publish_identity_validator_and_failed_validation(self):
        seen = []
        def validator(path):
            seen.append((path / "data.bin").read_bytes())
        self.cache.populate(
            "validated", lambda r: r.write_bytes("payload/data.bin", b"input"),
            output_bytes=5, validator=validator,
        )
        self.assertEqual(seen, [b"input"])
        def reject(path):
            raise IntegrityError("synthetic scene identity mismatch")
        with self.assertRaises(IntegrityError):
            self.cache.populate("invalid", lambda r: r.write_bytes(
                "payload/data.bin", b"bad"), output_bytes=3, validator=reject)
        self.assertNotIn("invalid", self.cache.stats()["entries"])
        self.assertEqual(self.cache.stats()["reservations"], 0)

    def test_unbudgeted_external_producer_and_bad_slot_are_rejected(self):
        with self.cache.reserve(output_bytes=1) as r:
            (r.path / "payload").mkdir()
            (r.path / "payload" / "data.bin").write_bytes(b"not-budgeted")
            with self.assertRaises(QuotaExceeded):
                r.commit("oversized")
        with self.assertRaises(StorageError):
            self.put("invalid", kind="result", slots=("current",))

    def test_compression_budget_fails_without_losing_result(self):
        path = self.put()
        with self.assertRaises(QuotaExceeded):
            self.package(compression_bytes=1)
        self.assertTrue(path.exists())
        self.assertEqual(self.cache.stats()["reservations"], 0)
        self.assertIn("result", self.cache.archive_candidates())

    def test_compression_temp_shares_quota_with_results(self):
        path = self.put(data=b"x" * (3 * MiB))
        with self.assertRaises(QuotaExceeded):
            self.package()
        self.assertTrue(path.exists())

    def test_archive_is_deterministic_and_includes_empty_directories(self):
        def producer(r):
            (r.path / "payload" / "empty").mkdir()
            r.write_bytes("payload/nested/data.bin", b"abc")
        for key in ("a", "b"):
            self.cache.populate(key, producer, kind="result", output_bytes=3)
        a, b = self.package("a"), self.package("b")
        self.assertEqual(a.identity, b.identity)
        self.assertEqual(a.manifest, b.manifest)
        ack = self.receiver(a.path, a.manifest)
        self.assertTrue((self.receiver.published_path(ack.identity) / "empty").is_dir())
        self.assertEqual(a.manifest["members_sha256"], digest(a.manifest["members"]))
        self.assertEqual(a.manifest["total_bytes"], 3)

    def test_archive_total_hash_corruption_is_rejected(self):
        self.put()
        package = self.package()
        contents = bytearray(package.path.read_bytes())
        contents[len(contents) // 2] ^= 0xFF
        package.path.write_bytes(contents)
        self.assert_rejected(package.path, package.manifest)
        with self.assertRaises(IntegrityError):
            self.cache.transfer_one("result", self.receiver)

    def test_member_hash_corruption_is_rejected_even_with_correct_total_hash(self):
        self.put(data=b"abc")
        package = self.package()
        info = tarfile.TarInfo("data.bin")
        info.size = 3
        path = self.malicious_tar([(info, b"xyz")])
        self.assert_rejected(path, bind_archive(package.manifest, path))

    def test_manifest_corruption_and_duplicate_fields_are_rejected(self):
        self.put()
        package = self.package()
        manifest = deepcopy(package.manifest)
        manifest["members"][0]["sha256"] = "0" * 64
        self.assert_rejected(package.path, manifest)
        self.cache.root.joinpath("state.json").write_text(
            '{"schema_version":"h3d-hot-cache-v1","schema_version":"duplicate"}')
        with self.assertRaises(IntegrityError):
            self.other_handle()

    def test_traversal_absolute_and_backslash_members_are_rejected(self):
        self.put(data=b"abc")
        package = self.package()
        for name in ("../escape", "/absolute", "a/../../escape", "a\\escape", "C:/escape"):
            with self.subTest(name=name):
                info = tarfile.TarInfo(name)
                info.size = 3
                path = self.malicious_tar([(info, b"abc")])
                self.assert_rejected(path, bind_archive(package.manifest, path))
        self.assertFalse((self.root / "escape").exists())

    def test_symlinks_hardlinks_special_and_duplicate_members_are_rejected(self):
        self.put(data=b"abc")
        package = self.package()
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE):
            with self.subTest(kind=kind):
                info = tarfile.TarInfo("data.bin")
                info.type, info.linkname = kind, "../escape"
                path = self.malicious_tar([(info, b"")])
                self.assert_rejected(path, bind_archive(package.manifest, path))
        info = tarfile.TarInfo("data.bin")
        info.size = 3
        path = self.malicious_tar([(info, b"abc"), (info, b"abc")])
        self.assert_rejected(path, bind_archive(package.manifest, path))

    def test_tar_declared_bomb_and_extended_headers_are_rejected(self):
        self.put(data=b"abc")
        package = self.package()
        for kind in (tarfile.REGTYPE, tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME):
            with self.subTest(kind=kind):
                info = tarfile.TarInfo("data.bin")
                info.type = kind
                info.size = GB
                path = self.root / "bomb.tar.gz"
                path.write_bytes(gzip.compress(info.tobuf(format=tarfile.USTAR_FORMAT), mtime=0))
                self.assert_rejected(path, bind_archive(package.manifest, path))

    def test_truncated_gzip_and_appended_payload_rejected(self):
        self.put()
        package = self.package()
        path = self.root / "truncated.tar.gz"
        path.write_bytes(package.path.read_bytes()[:-4])
        self.assert_rejected(path, bind_archive(package.manifest, path))
        path = self.root / "trailing.tar.gz"
        path.write_bytes(package.path.read_bytes() + gzip.compress(b"hidden-data"))
        self.assert_rejected(path, bind_archive(package.manifest, path))

    def test_receiver_limits_and_duplicate_ack_recheck_published_bytes(self):
        self.put()
        package = self.package()
        limited = ArchiveReceiver(self.root / "limited", max_extract_bytes=1,
                                  disk_usage=free_disk)
        with self.assertRaises(QuotaExceeded):
            limited(package.path, package.manifest)
        ack = self.receiver(package.path, package.manifest)
        self.receiver.published_path(ack.identity).joinpath("data.bin").write_bytes(b"corrupt")
        with self.assertRaises(IntegrityError):
            self.receiver(package.path, package.manifest)

    def test_source_symlinks_are_rejected_and_not_followed(self):
        outside = self.root / "outside"
        outside.write_bytes(b"synthetic-do-not-follow")
        with self.cache.reserve(output_bytes=100) as r:
            (r.path / "payload").mkdir()
            (r.path / "payload" / "link").symlink_to(outside)
            with self.assertRaises(IntegrityError):
                r.commit("linked", kind="result")
        self.assertEqual(outside.read_bytes(), b"synthetic-do-not-follow")

    def test_live_staging_is_not_removed_by_recovery(self):
        with self.cache.reserve(output_bytes=100) as r:
            r.write_bytes("partial.bin", b"partial")
            self.other_handle().recover()
            self.assertTrue((r.path / "partial.bin").exists())
            self.assertEqual(self.cache.stats()["reservations"], 1)

    def test_crash_scratch_is_removed_without_touching_pending_result(self):
        path = self.put()
        self.child(self.child_prelude() + (
            "r = cache.reserve(output_bytes=100)\n"
            "r.write_bytes('partial.tar.gz', b'incomplete')\n"
            "os._exit(0)\n"
        ))
        self.assertEqual(self.cache.stats()["reservations"], 1)
        resumed = self.other_handle()
        self.assertEqual(resumed.stats()["reservations"], 0)
        self.assertEqual(list((self.cache.root / "work").iterdir()), [])
        self.assertTrue(path.exists())

    def test_crash_before_payload_rename_finishes_publish_without_producer(self):
        self.child(self.child_prelude() + (
            "r = cache.reserve(output_bytes=3)\n"
            "r.write_bytes('payload/data.bin', b'abc')\n"
            "replace = os.replace\n"
            "def crash(source, dest):\n"
            "    if Path(dest).parent == root/'entries': os._exit(0)\n"
            "    return replace(source, dest)\n"
            "os.replace = crash\n"
            "r.commit('crash-result', kind='result')\n"
        ))
        resumed = self.other_handle()
        with resumed.lease("crash-result") as path:
            self.assertEqual((path / "data.bin").read_bytes(), b"abc")
        self.assertIn("crash-result", resumed.archive_candidates())
        self.assertEqual(resumed.stats()["reservations"], 0)

    def test_crash_before_archive_rename_resumes_same_identity(self):
        self.put()
        self.child(self.child_prelude() + (
            "replace = os.replace\n"
            "def crash(source, dest):\n"
            "    if Path(dest).parent == root/'archives': os._exit(0)\n"
            "    return replace(source, dest)\n"
            "os.replace = crash\n"
            "cache.queue_archive('result', job='engineering-job', shard='synthetic-shard', attempt=0)\n"
        ))
        state = json.loads((self.cache.root / "state.json").read_bytes())
        saved_identity = state["entries"]["result"]["archive"]["sha256"]
        resumed = self.other_handle()
        package = resumed.queue_archive("result", job="engineering-job",
                                        shard="synthetic-shard", attempt=0)
        self.assertEqual(package.identity.sha256, saved_identity)
        self.assertEqual(resumed.stats()["reservations"], 0)
        resumed.transfer_one("result", self.receiver, delete_acked=True)

    def test_receiver_crash_temp_is_cleaned_and_retry_publishes_once(self):
        self.put()
        package = self.package()
        code = self.child_prelude() + (
            f"receiver = ArchiveReceiver({str(self.receiver.root)!r}, disk_usage=disk)\n"
            "replace = os.replace\n"
            "def crash(source, dest):\n"
            "    if Path(dest).parent == receiver.root/'objects': os._exit(0)\n"
            "    return replace(source, dest)\n"
            "os.replace = crash\n"
            f"receiver.receive({str(package.path)!r}, {package.manifest!r})\n"
        )
        self.child(code)
        self.assertEqual(len(list((self.receiver.root / ".incoming").iterdir())), 1)
        resumed = ArchiveReceiver(self.receiver.root, disk_usage=free_disk)
        self.assertEqual(list((resumed.root / ".incoming").iterdir()), [])
        first = resumed(package.path, package.manifest)
        self.assertEqual(resumed(package.path, package.manifest), first)

    def test_crash_after_ack_before_delete_finishes_only_acked_cleanup(self):
        self.put()
        self.package()
        self.cache.transfer_one("result", self.receiver)
        self.child(self.child_prelude() + (
            "rmtree = shutil.rmtree\n"
            "def crash(path, *args, **kwargs):\n"
            "    if Path(path).parent == root/'entries': os._exit(0)\n"
            "    return rmtree(path, *args, **kwargs)\n"
            "shutil.rmtree = crash\n"
            "cache.collect_acknowledged('result')\n"
        ))
        resumed = self.other_handle()
        self.assertNotIn("result", resumed.stats()["entries"])
        self.assertEqual(list((resumed.root / "archives").iterdir()), [])

    def test_transfer_pending_calls_real_hook_with_saved_manifest(self):
        self.put("a", b"a")
        self.put("b", b"b")
        self.package("a", shard="a")
        self.package("b", shard="b")
        acks = self.cache.transfer_pending(self.receiver, delete_acked=True)
        self.assertEqual(len(acks), 2)
        self.assertEqual(self.cache.stats()["entries"], [])


if __name__ == "__main__":
    unittest.main()
