"""Synthetic hot-cache admission tests, using only the standard library.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover \
    -s tests -p test_hot_cache_admission.py -v
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import unittest
from unittest.mock import patch

from skill3d.storage import hot_cache as hot
from skill3d.storage.hot_cache import (
    ArchiveReceiver, EntryBusy, GB, HotCache, QuotaExceeded, StorageError,
)

MiB = 1024 * 1024


def free_disk(_path):
    return shutil._ntuple_diskusage(2_000 * GB, 500 * GB, 1_500 * GB)


def collect_worker(root, quota, ready, start, attempting, finished, results):
    """Open an independent handle before racing a parent's lease admission."""
    try:
        cache = HotCache(root, quota_bytes=quota, disk_usage=free_disk)
        ready.set()
        if not start.wait(10):
            raise AssertionError("collector was not started")
        attempting.set()
        results.put(("ok", cache.collect_acknowledged("result")))
    except BaseException:
        results.put(("error", traceback.format_exc()))
    finally:
        ready.set()
        finished.set()


class HotCacheAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3d-admission-synthetic-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = HotCache(
            self.root / "cache", quota_bytes=4 * MiB, disk_usage=free_disk)

    def put(self, key="result", *, kind="result", slots=(), data=b"synthetic"):
        return self.cache.populate(
            key, lambda r: r.write_bytes("payload/data.bin", data),
            kind=kind, output_bytes=len(data), slots=slots)

    def other_handle(self):
        return HotCache(
            self.cache.root, quota_bytes=self.cache.quota_bytes, disk_usage=free_disk)

    def state_bytes(self):
        return (self.cache.root / "state.json").read_bytes()

    def acknowledge(self):
        path = self.put()
        self.cache.queue_archive(
            "result", job="engineering-job", shard="synthetic-shard", attempt=0)
        receiver = ArchiveReceiver(self.root / "receiver", disk_usage=free_disk)
        self.cache.transfer_one("result", receiver)
        return path

    def child(self, code):
        prelude = (
            "import shutil\n"
            "from skill3d.storage.hot_cache import "
            "HotCache, EntryBusy, GB, QuotaExceeded, StorageError\n"
            "def disk(_): return shutil._ntuple_diskusage(2000*GB, 500*GB, 1500*GB)\n"
            f"cache = HotCache({str(self.cache.root)!r}, "
            f"quota_bytes={self.cache.quota_bytes}, disk_usage=disk)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", prelude + code],
            capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_published_queries_do_not_scan_save_or_touch(self):
        for kind in sorted(self.cache._KINDS):
            self.put(kind, kind=kind)
        before = self.state_bytes()
        with patch.object(hot, "_tree_bytes", side_effect=AssertionError("scan")), \
                patch.object(self.cache, "_save", side_effect=AssertionError("save")), \
                patch.object(self.cache, "stats", side_effect=AssertionError("audit")), \
                patch.object(hot.time, "time_ns", side_effect=AssertionError("touch")):
            for kind in sorted(self.cache._KINDS):
                status = self.cache.entry_status(kind)
                self.assertEqual(set(status), {"kind", "ready", "last_used"})
                self.assertEqual(status["kind"], kind)
                self.assertTrue(status["ready"])
                self.assertTrue(self.cache.contains(kind))
                status["kind"] = "changed-snapshot"
                self.assertEqual(self.cache.entry_status(kind)["kind"], kind)
            self.assertIsNone(self.cache.entry_status("missing"))
            self.assertFalse(self.cache.contains("missing"))
        self.assertEqual(self.state_bytes(), before)

    def test_query_validates_keys(self):
        for key in ("", "bad\nkey", "x" * 513, None, 1):
            for query_method in (self.cache.entry_status, self.cache.contains):
                with self.subTest(key=key, method=query_method.__name__), \
                        self.assertRaises(StorageError):
                    query_method(key)

    def test_unpublished_or_deleting_entries_are_not_admitted(self):
        self.put("input", kind="input", slots=("current",))
        for changes in ({"ready": False}, {"ready": True, "deleting": True}):
            with self.subTest(changes=changes):
                with self.cache._locked() as state:
                    state["entries"]["input"].update(changes)
                    self.cache._save(state)
                before = self.state_bytes()
                with patch.object(hot, "_tree_bytes", side_effect=AssertionError("scan")), \
                        patch.object(self.cache, "_save", side_effect=AssertionError("save")):
                    self.assertIsNone(self.cache.entry_status("input"))
                    self.assertFalse(self.cache.contains("input"))
                    with self.assertRaises(StorageError):
                        self.cache.pin("current", "input")
                    for touch in (False, True):
                        with self.assertRaises(StorageError):
                            with self.cache.lease("input", touch=touch):
                                self.fail("unpublished entry was admitted")
                self.assertEqual(self.state_bytes(), before)
                # The existing audit API still reports the durable intent.
                self.assertIn("input", self.cache.stats()["entries"])

    def test_idempotent_pins_do_not_save_or_scan_for_all_compatible_kinds(self):
        cases = (
            ("current", "input"), ("next", "input"), ("global", "input"),
            ("latest", "checkpoint"), ("best", "checkpoint"),
            ("state", "result"), ("state", "workspace"),
        )
        for slot, kind in cases:
            with self.subTest(slot=slot, kind=kind):
                key = slot + "-" + kind
                self.put(key, kind=kind, slots=(slot,))
                before = self.state_bytes()
                with patch.object(hot, "_tree_bytes", side_effect=AssertionError("scan")), \
                        patch.object(self.cache, "_save", side_effect=AssertionError("save")):
                    for _ in range(3):
                        self.cache.pin(slot, key)
                self.assertEqual(self.state_bytes(), before)
                self.cache.pin(slot, None)
                before = self.state_bytes()
                with patch.object(hot, "_tree_bytes", side_effect=AssertionError("scan")), \
                        patch.object(self.cache, "_save", side_effect=AssertionError("save")):
                    self.cache.pin(slot, None)
                self.assertEqual(self.state_bytes(), before)

    def test_idempotent_pin_still_validates_kind_and_dangling_entry(self):
        self.put("input", kind="input")
        for kind in ("result", "checkpoint", "workspace", None):
            with self.subTest(kind=kind):
                with self.cache._locked() as state:
                    state["slots"]["current"] = "input"
                    if kind is None:
                        del state["entries"]["input"]
                    else:
                        state["entries"]["input"]["kind"] = kind
                    self.cache._save(state)
                before = self.state_bytes()
                with patch.object(hot, "_tree_bytes", side_effect=AssertionError("scan")), \
                        patch.object(self.cache, "_save", side_effect=AssertionError("save")):
                    with self.assertRaises(StorageError):
                        self.cache.pin("current", "input")
                self.assertEqual(self.state_bytes(), before)

    def test_pin_changes_keep_durable_saves_and_quota_checks(self):
        self.put("first", kind="input")
        self.put("second", kind="input")
        with patch.object(self.cache, "_save", wraps=self.cache._save) as saves, \
                patch.object(hot, "_tree_bytes", wraps=hot._tree_bytes) as scans:
            self.cache.pin("current", "first")
            self.cache.pin("current", "second")
            self.cache.pin("current", None)
            self.assertEqual(saves.call_count, 3)
            self.assertGreaterEqual(scans.call_count, 3)
        self.cache.disk_usage = lambda _: shutil._ntuple_diskusage(
            2_000 * GB, 500 * GB, self.cache.min_free_bytes)
        before = self.state_bytes()
        with self.assertRaises(QuotaExceeded):
            self.cache.pin("current", "first")
        self.assertEqual(self.state_bytes(), before)
        with self.assertRaises(StorageError):
            self.cache.pin("invalid-slot", None)

    def test_no_touch_lease_does_not_scan_save_or_update_lru(self):
        path = self.put("input", kind="input")
        other = self.other_handle()
        before = self.state_bytes()
        with patch.object(hot, "_tree_bytes", side_effect=AssertionError("scan")), \
                patch.object(self.cache, "_save", side_effect=AssertionError("save")), \
                patch.object(other, "_save", side_effect=AssertionError("save")), \
                patch.object(hot.time, "time_ns", side_effect=AssertionError("touch")):
            for _ in range(3):
                with self.cache.lease("input", touch=False) as leased:
                    self.assertEqual(leased, path)
                    self.assertEqual((leased / "data.bin").read_bytes(), b"synthetic")
                    with other.lease("input", touch=False):
                        pass
                    with self.assertRaises(EntryBusy):
                        with other.lease("input", exclusive=True, touch=False):
                            self.fail("shared OS lease was not held")
        self.assertEqual(self.state_bytes(), before)
        with other.lease("input", exclusive=True, touch=False):
            with self.assertRaises(EntryBusy):
                with self.cache.lease("input", touch=False):
                    self.fail("exclusive OS lease was not held")

    def test_default_and_explicit_touch_update_lru_and_save(self):
        self.put("input", kind="input")
        for kwargs in ({}, {"touch": True}):
            timestamp = self.cache.entry_status("input")["last_used"] + 1
            with self.subTest(kwargs=kwargs), \
                    patch.object(hot.time, "time_ns", return_value=timestamp), \
                    patch.object(self.cache, "_save", wraps=self.cache._save) as saves, \
                    patch.object(hot, "_tree_bytes", wraps=hot._tree_bytes) as scans:
                with self.cache.lease("input", **kwargs):
                    pass
                self.assertEqual(saves.call_count, 1)
                self.assertGreater(scans.call_count, 0)
            self.assertEqual(self.cache.entry_status("input")["last_used"], timestamp)

    def test_no_touch_lease_releases_os_lock_on_body_exception(self):
        self.put("input", kind="input")
        with self.assertRaisesRegex(RuntimeError, "reader failed"):
            with self.cache.lease("input", touch=False):
                raise RuntimeError("reader failed")
        with self.other_handle().lease("input", exclusive=True, touch=False):
            pass

    def test_default_touch_save_failure_releases_os_lock(self):
        self.put("input", kind="input")
        before = self.state_bytes()
        with patch.object(self.cache, "_save", side_effect=QuotaExceeded("save denied")):
            with self.assertRaises(QuotaExceeded):
                with self.cache.lease("input"):
                    self.fail("failed save admitted the reader")
        self.assertEqual(self.state_bytes(), before)
        with self.other_handle().lease("input", exclusive=True, touch=False):
            pass

    def test_no_touch_exclusive_workspace_lease_invalidates_accounting(self):
        path = self.put("workspace", kind="workspace")
        with self.cache.workspace_lease("workspace", atomic_writes=True, incremental=True):
            pass
        record = self.cache.root / "locks" / ("w-" + hot._digest("workspace") + ".json")
        self.assertTrue(record.exists())
        before = self.state_bytes()
        with patch.object(hot, "_tree_bytes", side_effect=AssertionError("scan")), \
                patch.object(self.cache, "_save", side_effect=AssertionError("save")):
            with self.cache.lease("workspace", exclusive=True, touch=False) as leased:
                self.assertEqual(leased, path)
                self.assertFalse(record.exists())
        self.assertEqual(self.state_bytes(), before)
        with patch.object(hot, "_tree_bytes", wraps=hot._tree_bytes) as scans:
            with self.cache.workspace_lease("workspace", atomic_writes=True, incremental=True):
                self.assertGreater(scans.call_count, 0)
        self.assertTrue(record.exists())

    def test_no_touch_active_input_prevents_eviction_across_handles_and_processes(self):
        path = self.put("input", kind="input", data=b"x" * (2 * MiB))
        other = self.other_handle()
        with self.cache.lease("input", touch=False):
            with self.assertRaises(QuotaExceeded):
                other.reserve(output_bytes=2 * MiB)
            self.child(
                "try:\n"
                f"    cache.reserve(output_bytes={2 * MiB})\n"
                "except QuotaExceeded:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('active entry evicted')\n"
                "try:\n"
                "    with cache.lease('input', exclusive=True, touch=False): pass\n"
                "except EntryBusy:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('active OS lease was not held')\n")
            self.assertTrue(path.exists())
        with other.reserve(output_bytes=2 * MiB):
            self.assertFalse(path.exists())

    def test_query_snapshot_does_not_bypass_deleted_entry_revalidation(self):
        for use_process in (False, True):
            with self.subTest(use_process=use_process):
                self.acknowledge()
                self.assertTrue(self.cache.contains("result"))
                if use_process:
                    self.child("assert cache.collect_acknowledged('result')\n")
                else:
                    self.assertTrue(self.other_handle().collect_acknowledged("result"))
                with patch.object(self.cache, "_save", side_effect=AssertionError("save")), \
                        patch.object(hot, "_tree_bytes", side_effect=AssertionError("scan")):
                    self.assertIsNone(self.cache.entry_status("result"))
                    self.assertFalse(self.cache.contains("result"))
                    with self.assertRaises(StorageError):
                        self.cache.pin("state", "result")
                    with self.assertRaises(StorageError):
                        with self.cache.lease("result", touch=False):
                            self.fail("stale snapshot admitted a deleted entry")

    def check_admission_delete_race(self, *, use_process):
        path = self.acknowledge()
        if use_process:
            ctx = multiprocessing.get_context("spawn")
            events = [ctx.Event() for _ in range(5)]
            results = ctx.Queue()
            worker_type = ctx.Process
        else:
            events = [threading.Event() for _ in range(5)]
            results = queue.Queue()
            worker_type = threading.Thread
        ready, start, attempting, finished, release_reader = events
        worker = worker_type(
            target=collect_worker,
            args=(str(self.cache.root), self.cache.quota_bytes,
                  ready, start, attempting, finished, results))
        admission_paused = threading.Event()
        resume_admission = threading.Event()
        leased = threading.Event()
        reader_errors = []
        original_lease_path = self.cache._lease_path

        def pause_before_os_lock(key):
            # lease() has validated metadata but has not acquired its OS lock.
            admission_paused.set()
            if not resume_admission.wait(10):
                raise AssertionError("lease admission was not resumed")
            return original_lease_path(key)

        def reader():
            try:
                with self.cache.lease("result", touch=False):
                    leased.set()
                    if not release_reader.wait(10):
                        raise AssertionError("reader was not released")
            except BaseException:
                reader_errors.append(traceback.format_exc())

        reader_thread = threading.Thread(target=reader)
        worker.start()
        try:
            self.assertTrue(ready.wait(10), "collector did not open its handle")
            self.assertFalse(finished.is_set(), "collector failed during initialization")
            with patch.object(self.cache, "_lease_path", side_effect=pause_before_os_lock):
                reader_thread.start()
                self.assertTrue(admission_paused.wait(10), "admission was not reached")
                start.set()
                self.assertTrue(attempting.wait(10), "collector did not attempt deletion")
                self.assertFalse(finished.wait(0.1), "deletion bypassed the metadata lock")
                resume_admission.set()
                self.assertTrue(leased.wait(10), "reader did not acquire its OS lease")
                self.assertTrue(finished.wait(10), "collector did not finish")
                self.assertEqual(results.get(timeout=10), ("ok", False))
                self.assertTrue(path.exists())
                self.assertTrue(self.cache.contains("result"))
        finally:
            start.set()
            resume_admission.set()
            release_reader.set()
            if reader_thread.ident is not None:
                reader_thread.join(20)
            worker.join(20)
            if use_process:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(20)
                results.close()
                results.join_thread()
        self.assertFalse(reader_thread.is_alive(), "reader did not exit")
        self.assertFalse(worker.is_alive(), "collector did not exit")
        self.assertEqual(reader_errors, [])
        if use_process:
            self.assertEqual(worker.exitcode, 0)
        self.assertTrue(self.other_handle().collect_acknowledged("result"))
        self.assertFalse(path.exists())

    def test_admission_delete_race_across_handles(self):
        self.check_admission_delete_race(use_process=False)

    def test_admission_delete_race_across_processes(self):
        self.check_admission_delete_race(use_process=True)

    def test_interrupted_deletion_stays_unpublished_until_recovery(self):
        path = self.acknowledge()
        with patch.object(hot.shutil, "rmtree", side_effect=OSError("delete interrupted")):
            with self.assertRaisesRegex(OSError, "delete interrupted"):
                self.cache.collect_acknowledged("result")
        self.assertTrue(path.exists())
        self.assertIn("result", self.cache.stats()["entries"])
        with patch.object(hot, "_tree_bytes", side_effect=AssertionError("scan")), \
                patch.object(self.cache, "_save", side_effect=AssertionError("save")):
            self.assertFalse(self.cache.contains("result"))
            self.assertIsNone(self.cache.entry_status("result"))
            with self.assertRaises(StorageError):
                self.cache.pin("state", "result")
            with self.assertRaises(StorageError):
                with self.cache.lease("result", touch=False):
                    self.fail("deleting entry was admitted")
        resumed = self.other_handle()
        self.assertFalse(resumed.contains("result"))
        self.assertFalse(path.exists())
        self.assertFalse(resumed.collect_acknowledged("result"))

    def test_no_touch_lease_does_not_enable_deleting_unacknowledged_results(self):
        path = self.put()
        with self.cache.lease("result", touch=False):
            with self.assertRaises(StorageError):
                self.other_handle().collect_acknowledged("result")
        with self.assertRaises(StorageError):
            self.cache.collect_acknowledged("result")
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
