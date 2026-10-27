"""Synthetic source tests only; no framework runtime or external services.

PYTHONPATH=src python3 -m unittest discover -s tests -p test_atomic_workspace_io.py -v
"""

import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch

from skill3d.storage import hot_cache as hot
from skill3d.storage import state_workspace as state
from skill3d.storage.hot_cache import (
    GB, HotCache, IntegrityError, MIN_FREE_BYTES, QuotaExceeded, StorageError,
    _allocation_unit, _tree_bytes,
)
from skill3d.storage.state_workspace import StateWorkspace, WorkspaceIO

MiB = 1024 * 1024


def free_disk(_path):
    return shutil._ntuple_diskusage(2000 * GB, 500 * GB, 1500 * GB)


class AtomicWorkspaceIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3d-atomic-io-synthetic-")
        self.cache = HotCache(Path(self.temp.name) / "cache",
                              quota_bytes=64 * MiB, disk_usage=free_disk)

    def tearDown(self):
        self.temp.cleanup()

    def workspace(self, *, seed=None, logical=8 * MiB, allocated=8 * MiB):
        seed = seed or {}

        def produce(reservation):
            (reservation.path / "payload").mkdir(exist_ok=True)
            # Trusted synthetic fixture, admitted in full before materialization.
            for relative, data in seed.items():
                path = reservation.path / "payload" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)

        return self.cache.populate(
            "state", produce, kind="workspace",
            output_bytes=sum(len(data) for data in seed.values()),
            workspace_bytes=logical, allocation_bytes=allocated)

    def assert_counts(self, budget):
        self.assertEqual(budget._logical_bytes, _tree_bytes(budget.path, logical=True))
        self.assertEqual(budget._allocated_bytes, _tree_bytes(budget.path))

    def test_multiblock_logical_peak_counts_old_plus_temporary(self):
        old = b"o" * (3 * MiB)
        root = self.workspace(seed={"mutable/old": old}, logical=5 * MiB)
        with self.cache.workspace_lease("state", atomic_writes=True) as budget:
            io = WorkspaceIO(budget)
            with self.assertRaises(QuotaExceeded):
                io.write_bytes("mutable/old", b"n" * (3 * MiB))
            self.assertEqual((root / "mutable/old").read_bytes(), old)
            self.assertEqual(list((root / ".state-workspace/tmp").iterdir()), [])
            self.assert_counts(budget)
            io.unlink("mutable/old")
            io.write_bytes("mutable/new", b"n" * (3 * MiB))
            self.assert_counts(budget)
        self.assertEqual((root / "mutable/new").stat().st_size, 3 * MiB)

    def test_multiblock_allocation_peak_is_independent_of_logical_ceiling(self):
        probe = Path(self.temp.name) / "allocation-probe"
        probe.write_bytes(b"x")
        tiny_charge = hot._charged(probe.stat())
        old = b"o" * MiB
        seed = {f"tiny/{index}": b"x" for index in range(MiB // tiny_charge)}
        seed["mutable/old"] = old
        root = self.workspace(seed=seed,
                              logical=4 * MiB, allocated=4 * MiB)
        with self.cache.workspace_lease("state", atomic_writes=True) as budget:
            io = WorkspaceIO(budget)
            self.assertLess(budget._logical_bytes + 5 * MiB // 2, budget.bytes)
            self.assertGreater(budget._allocated_bytes + 5 * MiB // 2,
                               budget.charged_bytes)
            with self.assertRaises(QuotaExceeded):
                io.write_bytes("mutable/old", b"n" * (5 * MiB // 2))
            self.assertEqual((root / "mutable/old").read_bytes(), old)
            self.assert_counts(budget)
            io.write_bytes("mutable/old", b"n" * MiB)
            io.write_bytes("mutable/second", b"s" * MiB)
            self.assert_counts(budget)

    def test_replace_abort_delete_and_directories_keep_exact_counters(self):
        root = self.workspace(seed={"mutable/old": b"o" * MiB})
        with self.cache.workspace_lease("state", atomic_writes=True) as budget:
            io = WorkspaceIO(budget)
            io.write_bytes("mutable/old", b"small")
            self.assert_counts(budget)
            io.mkdir("mutable/nested/empty")
            self.assert_counts(budget)
            with self.assertRaisesRegex(RuntimeError, "abort"):
                with io.open("mutable/old") as writer:
                    writer.write(b"a" * MiB)
                    writer.write(b"b" * MiB)
                    self.assert_counts(budget)
                    raise RuntimeError("abort")
            self.assertEqual((root / "mutable/old").read_bytes(), b"small")
            self.assert_counts(budget)
            io.unlink("mutable/old")
            io.rmdir("mutable/nested/empty")
            io.rmdir("mutable/nested")
            self.assert_counts(budget)
            io.write_bytes("mutable/after", b"a" * (3 * MiB))
            self.assert_counts(budget)

    def test_failed_and_short_stream_writes_account_actual_bytes(self):
        root = self.workspace(seed={"old": b"original"})
        with self.cache.workspace_lease("state", atomic_writes=True) as budget:
            io = WorkspaceIO(budget)
            with self.assertRaisesRegex(RuntimeError, "abort"):
                with io.open("old") as writer:
                    real = writer._stream

                    class PartialStream:
                        def fileno(self):
                            return real.fileno()

                        def write(self, data):
                            return real.write(data[:3])

                    writer._stream = PartialStream()
                    self.assertEqual(writer.write(b"longer"), 3)
                    self.assert_counts(budget)

                    class FailingStream(PartialStream):
                        def write(self, data):
                            real.write(data[:2])
                            raise OSError("partial write failure")

                    writer._stream = FailingStream()
                    with self.assertRaises(OSError):
                        writer.write(b"longer")
                    self.assert_counts(budget)
                    self.assertEqual(budget._logical_bytes, len(b"original") + 5)
                    raise RuntimeError("abort")
            self.assertEqual((root / "old").read_bytes(), b"original")
            self.assert_counts(budget)

    def test_failed_replace_cleans_temporary_and_preserves_original(self):
        root = self.workspace(seed={"old": b"original"})
        with self.cache.workspace_lease("state", atomic_writes=True) as budget:
            io = WorkspaceIO(budget)
            with patch.object(state.os, "replace", side_effect=OSError("replace failure")):
                with self.assertRaises(OSError):
                    io.write_bytes("old", b"replacement")
            self.assertEqual((root / "old").read_bytes(), b"original")
            self.assertEqual(list((root / ".state-workspace/tmp").iterdir()), [])
            self.assert_counts(budget)
            io.write_bytes("old", b"next")
            self.assert_counts(budget)

    def test_nested_open_writers_account_both_temporaries(self):
        self.workspace(logical=3 * MiB)
        with self.cache.workspace_lease("state", atomic_writes=True) as budget:
            io = WorkspaceIO(budget)
            with io.open("first") as first, io.open("second") as second:
                first.write(b"a" * MiB)
                second.write(b"b" * MiB)
                self.assert_counts(budget)
                with self.assertRaises(QuotaExceeded):
                    first.write(b"c" * (2 * MiB))
                self.assert_counts(budget)
            self.assert_counts(budget)
            self.assertEqual(budget._writers, 0)

    def test_free_floor_drop_between_blocks_refuses_before_more_bytes(self):
        root = self.workspace(seed={"old": b"original"})
        with self.cache.workspace_lease("state", atomic_writes=True) as budget:
            io = WorkspaceIO(budget)
            with self.assertRaises(QuotaExceeded):
                with io.open("old") as writer:
                    writer.write(b"a" * MiB)
                    unit = _allocation_unit(root)
                    self.cache.disk_usage = lambda p: shutil._ntuple_diskusage(
                        2000 * GB, 500 * GB, MIN_FREE_BYTES + unit - 1)
                    writer.write(b"b" * MiB)
            self.cache.disk_usage = free_disk
            self.assertEqual((root / "old").read_bytes(), b"original")
            self.assert_counts(budget)

    def test_free_floor_is_checked_even_without_new_allocation(self):
        self.workspace()
        with self.cache.workspace_lease("state", atomic_writes=True) as budget:
            io = WorkspaceIO(budget)
            with self.assertRaises(QuotaExceeded):
                with io.open("new") as writer:
                    writer.write(b"a")
                    self.cache.disk_usage = lambda p: shutil._ntuple_diskusage(
                        2000 * GB, 500 * GB, MIN_FREE_BYTES - 1)
                    writer.write(b"b")
            self.cache.disk_usage = free_disk
            self.assert_counts(budget)

    def test_directory_and_temporary_admission_precedes_creation(self):
        root = self.workspace()
        with self.cache.workspace_lease("state", atomic_writes=True) as budget:
            io = WorkspaceIO(budget)
            self.cache.disk_usage = lambda p: shutil._ntuple_diskusage(
                2000 * GB, 500 * GB, MIN_FREE_BYTES)
            try:
                with self.assertRaises(QuotaExceeded):
                    io.mkdir("missing/deep")
                with self.assertRaises(QuotaExceeded):
                    io.write_bytes("other/deep/file", b"data")
                self.assertFalse((root / "missing").exists())
                self.assertFalse((root / "other").exists())
                self.assertFalse((root / ".state-workspace").exists())
                self.assert_counts(budget)
            finally:
                self.cache.disk_usage = free_disk

    def test_directory_allocation_ceiling_precedes_creation(self):
        root = self.workspace(logical=0, allocated=0)
        with self.cache.workspace_lease("state", atomic_writes=True) as budget:
            depth = hot._HEADROOM // _allocation_unit(root) + 2
            relative = "missing/" + "/".join(f"dir{index}" for index in range(depth))
            with self.assertRaises(QuotaExceeded):
                WorkspaceIO(budget).mkdir(relative)
            self.assertFalse((root / "missing").exists())
            self.assert_counts(budget)

    def test_raw_symlink_tamper_is_rejected_at_exit(self):
        root = self.workspace()
        with self.assertRaises(IntegrityError):
            with self.cache.workspace_lease("state", atomic_writes=True):
                (root / "raw-link").symlink_to(root / "target")

    def test_raw_logical_overquota_tamper_is_rejected_at_exit(self):
        root = self.workspace(logical=MiB)
        with self.assertRaises(QuotaExceeded):
            with self.cache.workspace_lease("state", atomic_writes=True):
                (root / "raw").write_bytes(b"x" * (2 * MiB))

    def test_raw_allocated_overquota_tamper_is_rejected_at_exit(self):
        root = self.workspace(logical=MiB, allocated=MiB)
        probe = Path(self.temp.name) / "allocation-probe"
        probe.write_bytes(b"x")
        tiny_charge = hot._charged(probe.stat())
        with self.assertRaises(QuotaExceeded):
            with self.cache.workspace_lease("state", atomic_writes=True) as budget:
                for index in range(budget.charged_bytes // tiny_charge + 2):
                    (root / str(index)).write_bytes(b"x")
                self.assertLess(_tree_bytes(root, logical=True), budget.bytes)
                self.assertGreater(_tree_bytes(root), budget.charged_bytes)

    def test_raw_tamper_is_rejected_before_atomic_producer_entry(self):
        root = self.workspace()
        (root / "raw-link").symlink_to(root / "target")
        entered = False
        with self.assertRaises(IntegrityError):
            with self.cache.workspace_lease("state", atomic_writes=True):
                entered = True
        self.assertFalse(entered)

    def test_default_raw_lease_retains_per_check_scans_and_whole_tree_fsync(self):
        root = self.workspace()
        with patch.object(hot, "_tree_bytes", wraps=hot._tree_bytes) as scans, \
                patch.object(hot, "_fsync_tree", wraps=hot._fsync_tree) as sync:
            with self.cache.workspace_lease("state") as budget:
                before = scans.call_count
                budget.check_growth(logical_bytes=1, allocation_bytes=4096)
                self.assertGreaterEqual(scans.call_count - before, 4)
                (root / "raw").write_bytes(b"raw")
            self.assertEqual(sync.call_count, 1)

    def test_atomic_scan_count_is_bounded_by_boundaries_with_many_files(self):
        self.workspace(seed={f"seed/{index}": b"seed" for index in range(2048)},
                       logical=16 * MiB, allocated=16 * MiB)
        counts = []
        for writes in (1, 40):
            with patch.object(hot, "_tree_bytes", wraps=hot._tree_bytes) as scans, \
                    patch.object(self.cache, "_fits", wraps=self.cache._fits) as fits, \
                    patch.object(hot, "_fsync_tree", side_effect=AssertionError("whole sync")):
                with self.cache.workspace_lease("state", atomic_writes=True) as budget:
                    io = WorkspaceIO(budget)
                    before = scans.call_count
                    for index in range(writes):
                        io.write_bytes(f"mutable/{index}", b"x" * (2 * MiB // 40))
                        io.read_bytes(f"mutable/{index}")
                    self.assertEqual(scans.call_count, before)
                    self.assertEqual(fits.call_count, 1)
                self.assertEqual(fits.call_count, 2)
                counts.append(scans.call_count)
        self.assertEqual(counts[0], counts[1])
        self.assertLessEqual(counts[0], 10)

    def test_atomic_fsyncs_written_file_and_all_changed_ancestors(self):
        root = self.workspace(seed={"unchanged": b"seed"})
        synced_inodes = []
        real_fsync = os.fsync

        def sync(fd):
            info = os.fstat(fd)
            if stat.S_ISREG(info.st_mode):
                synced_inodes.append(info.st_ino)
            return real_fsync(fd)

        with patch.object(state.os, "fsync", side_effect=sync), \
                patch.object(state, "_fsync_dir", wraps=state._fsync_dir) as dirs, \
                patch.object(hot, "_fsync_tree", side_effect=AssertionError("whole sync")):
            with self.cache.workspace_lease("state", atomic_writes=True) as budget:
                WorkspaceIO(budget).write_bytes("mutable/nested/new", b"written")
        self.assertIn((root / "mutable/nested/new").stat().st_ino, synced_inodes)
        self.assertNotIn((root / "unchanged").stat().st_ino, synced_inodes)
        paths = {call.args[0] for call in dirs.call_args_list}
        self.assertTrue({root, root / "mutable", root / "mutable/nested",
                         root / ".state-workspace",
                         root / ".state-workspace/tmp"}.issubset(paths))

    def test_atomic_option_requires_fixed_envelope_and_closes_capability(self):
        self.workspace()
        with self.assertRaises(StorageError):
            StateWorkspace(self.cache, "state", atomic_writes=True)
        for estimates in ({"growth_bytes": 0}, {"allocation_bytes": 0}):
            with self.assertRaises(StorageError):
                with self.cache.workspace_lease("state", atomic_writes=True, **estimates):
                    self.fail("atomic lease must not extend reservations")
        workspace = StateWorkspace(self.cache, "state", bounded=True, atomic_writes=True)
        with workspace._write_lease(growth_bytes=MiB) as budget:
            self.assertTrue(budget.atomic_writes)
            io = WorkspaceIO(budget)
            io.write_bytes("file", b"written")
            self.assertFalse(hasattr(io, "path"))
            self.assertFalse(hasattr(io, "fileno"))
        with self.assertRaises(StorageError):
            io.write_bytes("file", b"late")

    def test_atomic_state_mutation_uses_declared_roots_and_preserves_fixed_ceiling(self):
        data = b"baseline"
        self.workspace(seed={"mutable/old": data})
        StateWorkspace.adopt(
            self.cache, "state", baseline_anchor="1" * 64,
            baseline_members={"mutable/old": {
                "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}},
            mutable_roots=("mutable",), workspace_bytes=8 * MiB,
            allocation_bytes=8 * MiB)
        workspace = StateWorkspace(self.cache, "state", bounded=True, atomic_writes=True)
        with self.cache._locked() as saved:
            ceiling = saved["entries"]["state"]["workspace_charged_bytes"]
        workspace.mutate(lambda io: io.write_bytes("mutable/new", b"new"),
                         estimated_growth_bytes=3)
        with self.assertRaises(StorageError):
            workspace.mutate(lambda io: io.write_bytes("undeclared/new", b"new"),
                             estimated_growth_bytes=3)
        with self.cache._locked() as saved:
            self.assertEqual(saved["entries"]["state"]["workspace_charged_bytes"], ceiling)


if __name__ == "__main__":
    unittest.main()
