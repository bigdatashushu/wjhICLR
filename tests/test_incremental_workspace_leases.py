"""Concurrent fixed-ceiling writes and crash reconciliation, without services."""

import multiprocessing
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from skill3d.storage import hot_cache as hot
from skill3d.storage import state_workspace as state
from skill3d.storage.hot_cache import HotCache, IntegrityError, QuotaExceeded, _tree_bytes
from skill3d.storage.state_workspace import StateWorkspace, WorkspaceIO
from test_atomic_workspace_io import free_disk

MiB = 1024 * 1024


def write_child(root, name, crash=False):
    cache = HotCache(root, quota_bytes=64 * MiB, disk_usage=free_disk)
    workspace = StateWorkspace(
        cache, "state", bounded=True, atomic_writes=True, incremental_leases=True)
    try:
        with workspace._write_lease() as budget:
            WorkspaceIO(budget).write_bytes(name, b"x" * (2 * MiB))
            if crash:
                os._exit(77)
    except QuotaExceeded:
        os._exit(23)


class IncrementalWorkspaceLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3d-incremental-synthetic-")
        self.cache = HotCache(
            Path(self.temp.name) / "cache", quota_bytes=64 * MiB, disk_usage=free_disk)
        self.root = self.cache.populate(
            "state", lambda _: None, kind="workspace",
            workspace_bytes=3 * MiB, allocation_bytes=4 * MiB)

    def tearDown(self):
        self.temp.cleanup()

    def lease(self):
        return self.cache.workspace_lease("state", atomic_writes=True, incremental=True)

    def test_clean_sections_do_not_rescan_or_save_global_state(self):
        with self.lease() as budget:
            WorkspaceIO(budget).write_bytes("first", b"first")
        with patch.object(hot, "_tree_bytes", side_effect=AssertionError("repeat scan")), \
                patch.object(self.cache, "_save", side_effect=AssertionError("LRU save")):
            for index in range(32):
                with self.lease() as budget:
                    WorkspaceIO(budget).write_bytes(f"result/{index}", b"result")
        with self.lease() as budget:
            self.assertEqual(budget._logical_bytes, _tree_bytes(self.root, logical=True))
            self.assertEqual(budget._allocated_bytes, _tree_bytes(self.root))

    def test_process_death_reconciles_before_the_next_write(self):
        process = multiprocessing.get_context("fork").Process(
            target=write_child, args=(str(self.cache.root), "committed-before-crash", True))
        process.start()
        process.join(20)
        self.assertEqual(process.exitcode, 77)
        with patch.object(hot, "_tree_bytes", wraps=hot._tree_bytes) as scans:
            with self.lease() as budget:
                self.assertGreater(scans.call_count, 0)
                self.assertEqual(budget._logical_bytes, 2 * MiB)
                with self.assertRaises(QuotaExceeded):
                    WorkspaceIO(budget).write_bytes("too-large", b"x" * (2 * MiB))
        with patch.object(hot, "_tree_bytes", side_effect=AssertionError("repeat scan")):
            with self.lease() as budget:
                WorkspaceIO(budget).write_bytes("after-recovery", b"ok")

    def test_concurrent_processes_share_the_same_ceiling(self):
        ctx = multiprocessing.get_context("fork")
        processes = [ctx.Process(
            target=write_child, args=(str(self.cache.root), f"child-{index}"))
            for index in range(2)]
        for process in processes:
            process.start()
        for process in processes:
            process.join(20)
        self.assertEqual(sorted(process.exitcode for process in processes), [0, 23])
        with self.lease() as budget:
            self.assertEqual(budget._logical_bytes, _tree_bytes(self.root, logical=True))
            self.assertEqual(budget._allocated_bytes, _tree_bytes(self.root))
            self.assertLessEqual(budget._logical_bytes, budget.bytes)

    def test_raw_exclusive_lease_invalidates_persisted_counts(self):
        with self.lease() as budget:
            WorkspaceIO(budget).write_bytes("first", b"first")
        with self.cache.lease("state", exclusive=True) as root:
            (root / "external").write_bytes(b"external")
        with patch.object(hot, "_tree_bytes", wraps=hot._tree_bytes) as scans:
            with self.lease() as budget:
                self.assertGreater(scans.call_count, 0)
                self.assertEqual(budget._logical_bytes, len(b"firstexternal"))

    def test_crashed_raw_writer_is_rejected_on_reconciliation(self):
        with self.lease():
            pass
        with self.cache.lease("state", exclusive=True) as root:
            (root / "link").symlink_to(root / "outside")
        with self.assertRaises(IntegrityError):
            with self.lease():
                self.fail("producer must not run")

    def test_swallowed_directory_fsync_failure_keeps_dirty_intent(self):
        with self.lease() as budget:
            WorkspaceIO(budget).write_bytes("mutable/file", b"durable")
        with self.lease() as budget:
            with patch.object(state, "_fsync_dir", side_effect=OSError("sync failure")):
                with self.assertRaises(OSError):
                    WorkspaceIO(budget).unlink("mutable/file")
            self.assertFalse(budget._durability_ok)
        record = self.cache.root / "locks" / ("w-" + hot._digest("state") + ".json")
        self.assertFalse(hot._read_json(record)["clean"])
        with patch.object(hot, "_fsync_tree", wraps=hot._fsync_tree) as sync:
            with self.lease() as budget:
                self.assertEqual(sync.call_count, 1)
                self.assertEqual(budget._logical_bytes, _tree_bytes(self.root, logical=True))
        self.assertTrue(hot._read_json(record)["clean"])

    def test_accounting_metadata_has_its_own_persistent_quota_envelope(self):
        for key in ("state", "other"):
            if key != "state":
                self.cache.populate(
                    key, lambda _: None, kind="workspace",
                    workspace_bytes=3 * MiB, allocation_bytes=4 * MiB)
            with self.cache.workspace_lease(key, atomic_writes=True, incremental=True):
                pass
        with self.cache._locked() as saved:
            for key in ("state", "other"):
                record = self.cache.root / "locks" / ("w-" + hot._digest(key) + ".json")
                reserved = saved["entries"][key]["accounting_charged_bytes"]
                self.assertGreater(reserved, hot._charged(record.stat()))
        self.assertLessEqual(self.cache.stats()["charged_bytes"], self.cache.quota_bytes)

    def test_long_unicode_key_checks_actual_json_allocation_before_replacement(self):
        import shutil
        key = "\U0001f600" * 512
        self.cache.populate(
            key, lambda _: None, kind="workspace",
            workspace_bytes=MiB, allocation_bytes=MiB)
        with self.cache.workspace_lease(key, atomic_writes=True, incremental=True):
            pass
        unit = hot._allocation_unit(self.cache.root)
        self.cache.disk_usage = lambda _: shutil._ntuple_diskusage(
            2000 * 10**9, 500 * 10**9, self.cache.min_free_bytes + unit)
        with self.assertRaises(QuotaExceeded):
            with self.cache.workspace_lease(key, atomic_writes=True, incremental=True):
                self.fail("oversized accounting temporary must not be admitted")
        self.cache.disk_usage = free_disk


if __name__ == "__main__":
    unittest.main()
