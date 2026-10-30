"""Synthetic-only unittest suite; no application fixtures or online collection.

PYTHONPATH=src python3 -m unittest discover -s tests -p test_state_workspace.py -v
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from skill3d.storage.hot_cache import (
    ArchiveReceiver, EntryBusy, GB, HotCache, IntegrityError, MIN_FREE_BYTES,
    QuotaExceeded, StorageError, _tree_bytes,
)
from skill3d.storage.state_workspace import ArchiveRoute, BaselineAnchor, StateWorkspace

MiB = 1024 * 1024


def free_disk(_path):
    return shutil._ntuple_diskusage(2000 * GB, 500 * GB, 1500 * GB)


def route(key, kind):
    return ArchiveRoute("synthetic-job", kind + ":" + key, 0)


class StateWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3d-workspace-synthetic-")
        self.root = Path(self.temp.name)
        self.cache = HotCache(self.root / "cache", quota_bytes=35 * MiB, disk_usage=free_disk)
        self.receiver = ArchiveReceiver(self.root / "synthetic-receiver", disk_usage=free_disk)
        self.seed = {
            "offline_images/old.bin": b"baseline-image" * 100,
            "stream/state.json": b'{"completed":0}',
            "ledger/old.json": b'{"synthetic":true}',
        }
        self.baseline = BaselineAnchor(
            hashlib.sha256(b"verified-synthetic-titan-baseline").hexdigest(),
            {ref: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
             for ref, data in self.seed.items()},
        )
        self.stage()
        self.workspace = self.adopt()

    def tearDown(self):
        self.temp.cleanup()

    def stage(self):
        def produce(reservation):
            for ref, data in self.seed.items():
                reservation.write_bytes("payload/" + ref, data)
        self.raw = self.cache.populate(
            "migration-state", produce, kind="result",
            output_bytes=sum(map(len, self.seed.values())), slots=("state",))

    def adopt(self):
        return StateWorkspace.adopt(
            self.cache, "migration-state", baseline_anchor=self.baseline.sha256,
            baseline_members=self.baseline.files,
            baseline_directories=self.baseline.directories,
            mutable_roots=("stream", "ledger"), workspace_bytes=10 * MiB,
            allocation_bytes=10 * MiB, identity=route, transfer=self.receiver)

    def reopen(self):
        self.cache = HotCache(self.cache.root, quota_bytes=35 * MiB, disk_usage=free_disk)
        self.workspace = StateWorkspace(self.cache, "migration-state", identity=route,
                                        transfer=self.receiver)

    def state(self):
        return json.loads((self.cache.root / "state.json").read_bytes())

    def child(self, body):
        prelude = (
            "import os, json, shutil\n"
            "from pathlib import Path\n"
            "from skill3d.storage.hot_cache import HotCache, ArchiveReceiver, GB\n"
            "from skill3d.storage.state_workspace import StateWorkspace, ArchiveRoute\n"
            "def disk(p): return shutil._ntuple_diskusage(2000*GB,500*GB,1500*GB)\n"
            f"cache=HotCache(Path({str(self.cache.root)!r}),quota_bytes={35 * MiB},disk_usage=disk)\n"
            f"receiver=ArchiveReceiver(Path({str(self.receiver.root)!r}),disk_usage=disk)\n"
            "def route(k,t): return ArchiveRoute('synthetic-job',t+':'+k,0)\n"
            "ws=StateWorkspace(cache,'migration-state',identity=route,transfer=receiver)\n"
        )
        result = subprocess.run([sys.executable, "-c", prelude + body],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_adopt_preserves_path_inode_state_pin_and_seed_bytes(self):
        info = self.raw.stat()
        self.adopt()
        self.assertEqual(self.raw.stat().st_ino, info.st_ino)
        for ref, data in self.seed.items():
            self.assertEqual((self.raw / ref).read_bytes(), data)
        self.cache.pin("state", "migration-state")
        self.assertEqual(self.state()["entries"]["migration-state"]["kind"], "workspace")
        self.assertEqual(self.cache.stats()["slots"]["state"], "migration-state")
        self.assertGreaterEqual(self.cache.stats()["charged_bytes"], 10 * MiB)
        self.assertLess(self.cache.stats()["physical_bytes"], MiB)
        self.assertEqual(self.cache.stats()["reservations"], 0)

    def test_shared_quota_counts_growth_scene_and_scratch_without_double_charge(self):
        self.cache.populate("scene", lambda r: r.write_bytes("payload/s.bin", b"s" * (20 * MiB)),
                            kind="input", output_bytes=20 * MiB, slots=("current",))
        other = HotCache(self.cache.root, quota_bytes=35 * MiB, disk_usage=free_disk)
        self.assertGreaterEqual(other.stats()["charged_bytes"], 30 * MiB)
        with other.reserve(output_bytes=4 * MiB):
            with self.assertRaises(QuotaExceeded):
                self.cache.reserve(output_bytes=MiB)
        with self.assertRaises(QuotaExceeded):
            other.reserve(output_bytes=5 * MiB)
        charged = other.stats()["charged_bytes"]
        self.workspace.mutate(lambda io: io.write_bytes("stream/new.bin", b"x" * MiB),
                              estimated_growth_bytes=MiB)
        self.assertLess(abs(other.stats()["charged_bytes"] - charged), 128 * 1024)
        self.reopen()
        self.assertGreaterEqual(self.cache.stats()["charged_bytes"], 30 * MiB)

    def test_quota_refusal_invokes_zero_mutation_and_output_producers(self):
        mutation, output = Mock(), Mock()
        with self.assertRaises(QuotaExceeded):
            self.workspace.mutate(mutation, estimated_growth_bytes=40 * MiB)
        with self.assertRaises(QuotaExceeded):
            self.workspace.publish_output("too-big", output, estimated_bytes=30 * MiB)
        self.assertEqual(mutation.call_count, 0)
        self.assertEqual(output.call_count, 0)
        self.assertNotIn("too-big", self.state()["entries"])
        self.assertEqual(self.cache.stats()["reservations"], 0)

    def test_new_workspace_admits_full_future_envelope_before_producer(self):
        producer = Mock()
        with self.assertRaises(QuotaExceeded):
            self.cache.new_workspace("blocked", producer, initial_bytes=MiB,
                                     growth_bytes=30 * MiB)
        self.assertEqual(producer.call_count, 0)
        self.cache.new_workspace("new", lambda r: r.write_bytes("payload/a", b"a"),
                                 initial_bytes=1, growth_bytes=MiB)
        row = self.state()["entries"]["new"]
        self.assertEqual(row["workspace_bytes"], MiB + 1)
        self.assertGreaterEqual(self.cache.stats()["reserved_bytes"], 10 * MiB)
        self.assertEqual(self.cache.stats()["reservations"], 0)

    def test_direct_populate_workspace_ceiling_and_state_pin(self):
        path = self.cache.populate(
            "direct", lambda r: r.write_bytes("payload/a", b"abcd"),
            kind="workspace", output_bytes=4, workspace_bytes=MiB, slots=("state",))
        self.assertEqual((path / "a").read_bytes(), b"abcd")
        self.assertEqual(self.state()["entries"]["direct"]["workspace_bytes"], MiB)
        self.assertEqual(self.cache.stats()["slots"]["state"], "direct")
        with self.cache.reserve(output_bytes=4) as reservation:
            reservation.write_bytes("payload/a", b"abcd")
            with self.assertRaises(QuotaExceeded):
                reservation.commit("bad-ceiling", kind="workspace", workspace_bytes=3)
        self.assertNotIn("bad-ceiling", self.cache.stats()["entries"])

    def test_adopt_admission_failure_preserves_unarchived_result_kind_and_path(self):
        path = self.cache.populate("candidate", lambda r: r.write_bytes("payload/a", b"a"),
                                   kind="result", output_bytes=1)
        inode = path.stat().st_ino
        with self.assertRaises(QuotaExceeded):
            self.cache.adopt_workspace("candidate", workspace_bytes=30 * MiB)
        self.assertEqual(self.state()["entries"]["candidate"]["kind"], "result")
        self.assertEqual(path.stat().st_ino, inode)
        self.cache.queue_archive("candidate", job="synthetic", shard="candidate", attempt=0)
        with self.assertRaises(StorageError):
            self.cache.adopt_workspace("candidate", growth_bytes=MiB)

    def test_incomplete_baseline_injection_is_rejected_before_journal_creation(self):
        path = self.cache.populate("incomplete", lambda r: r.write_bytes("payload/a", b"a"),
                                   kind="result", output_bytes=1)
        with self.assertRaises(IntegrityError):
            StateWorkspace.adopt(
                self.cache, "incomplete", baseline_anchor="1" * 64,
                baseline_members={}, mutable_roots=(), workspace_bytes=MiB)
        self.assertEqual((path / "a").read_bytes(), b"a")
        self.assertFalse((path / ".state-workspace/journal.json").exists())

    def test_floor_refusal_accounts_for_persistent_unwritten_workspace_growth(self):
        self.cache.disk_usage = lambda p: shutil._ntuple_diskusage(
            2000 * GB, 500 * GB, MIN_FREE_BYTES + 9 * MiB)
        producer = Mock()
        with self.assertRaises(QuotaExceeded):
            self.workspace.mutate(producer, estimated_growth_bytes=1)
        self.assertEqual(producer.call_count, 0)
        self.cache.disk_usage = free_disk

    def test_crash_keeps_completed_mutable_and_immutable_outputs_not_scratch(self):
        self.child(
            "ws.mutate(lambda io: io.write_bytes('stream/completed.json',b'completed'),"
            "estimated_growth_bytes=1024)\n"
            "ws.publish_output('completed',lambda r:r.write_bytes('payload/out',b'done'),"
            "estimated_bytes=4)\n"
            "r=cache.reserve(output_bytes=1024)\n"
            "r.write_bytes('abandoned',b'scratch')\n"
            "os._exit(0)\n")
        self.reopen()
        self.assertEqual((self.raw / "stream/completed.json").read_bytes(), b"completed")
        self.assertIn("completed", self.cache.stats()["entries"])
        self.assertEqual(self.cache.stats()["reservations"], 0)
        self.assertEqual(list((self.cache.root / "work").iterdir()), [])
        self.assertGreaterEqual(self.cache.stats()["charged_bytes"], 10 * MiB)
        self.assertEqual(len(self.workspace.archive_pending()), 1)
        self.assertNotIn("completed", self.cache.stats()["entries"])
        self.assertIn("migration-state", self.cache.stats()["entries"])

    def test_crash_during_workspace_publish_recovers_persistent_growth(self):
        self.child(
            "import skill3d.storage.hot_cache as hot\n"
            "replace=hot.os.replace\n"
            "def stop(src,dst):\n"
            "    if Path(src).name=='payload' and Path(dst).parent.name=='entries': os._exit(0)\n"
            "    return replace(src,dst)\n"
            "hot.os.replace=stop\n"
            "cache.new_workspace('new-crashed',lambda r:r.write_bytes('payload/a',b'a'),"
            f"initial_bytes=1,growth_bytes={2 * MiB})\n")
        self.reopen()
        with self.cache.workspace_lease("new-crashed") as budget:
            self.assertEqual((budget.path / "a").read_bytes(), b"a")
            self.assertEqual(budget.bytes, 2 * MiB + 1)
        self.assertEqual(self.cache.stats()["reservations"], 0)
        self.assertGreaterEqual(self.cache.stats()["charged_bytes"], 12 * MiB)
        self.assertLess(self.cache.stats()["charged_bytes"], 13 * MiB)

    def test_adopt_crash_before_journal_can_finish_in_place_without_clone(self):
        self.child(
            "cache.populate('adopt-crash',lambda r:r.write_bytes('payload/a',b'a'),"
            "kind='result',output_bytes=1)\n"
            f"cache.adopt_workspace('adopt-crash',workspace_bytes={2 * MiB})\n"
            "os._exit(0)\n")
        self.reopen()
        with self.cache.lease("adopt-crash") as path:
            inode = path.stat().st_ino
            self.assertFalse((path / ".state-workspace/journal.json").exists())
        StateWorkspace.adopt(
            self.cache, "adopt-crash", baseline_anchor="1" * 64,
            baseline_members={"a": {"size": 1, "sha256": hashlib.sha256(b"a").hexdigest()}},
            mutable_roots=(), workspace_bytes=2 * MiB)
        self.assertEqual(path.stat().st_ino, inode)
        self.assertEqual((path / "a").read_bytes(), b"a")
        self.assertTrue((path / ".state-workspace/journal.json").exists())

    def test_unchanged_mutable_baseline_is_not_copied_into_checkpoints(self):
        path = self.workspace.checkpoint("unchanged", latest=True, best=True)
        data = StateWorkspace.validate_checkpoint(path, self.baseline)
        self.assertEqual(data["files"], {})
        self.assertEqual(data["directories"], [])
        self.assertEqual(data["deleted_files"], [])
        self.assertEqual(data["deleted_directories"], [])
        self.assertEqual(len(list(path.rglob("*"))), 1)

    def test_delta_snapshots_do_not_read_clone_or_omit_mutable_new_members(self):
        self.workspace.mutate(
            lambda io: (io.write_bytes("stream/state.json", b'{"completed":1}'),
                        io.unlink("ledger/old.json"),
                        io.write_bytes("ledger/new.json", b"new-ledger"),
                        io.write_bytes("new-output/done", b"done")),
            estimated_growth_bytes=1024)
        old = self.raw / "offline_images/old.bin"
        import skill3d.storage.state_workspace as module
        real_hash = module._file_hash
        def selective_hash(path):
            if Path(path) == old:
                self.fail("delta rehashed immutable baseline content")
            return real_hash(path)
        with patch.object(module, "_file_hash", selective_hash):
            path = self.workspace.checkpoint("delta", latest=True, best=True)
        data = StateWorkspace.validate_checkpoint(path, self.baseline)
        self.assertEqual(data["baseline_archive_sha256"], self.baseline.sha256)
        self.assertEqual(data["deleted_files"], ["ledger/old.json"])
        self.assertEqual(set(data["files"]),
                         {"stream/state.json", "ledger/new.json", "new-output/done"})
        self.assertFalse((path / "offline_images/old.bin").exists())
        self.assertFalse((path / ".state-workspace").exists())
        self.assertLess(_tree_bytes(path, logical=True), 4096)
        self.assertEqual(old.read_bytes(), self.seed["offline_images/old.bin"])
        with self.assertRaises(IntegrityError):
            StateWorkspace.validate_checkpoint(
                path, BaselineAnchor("0" * 64, self.baseline.files))

    def test_delta_empty_directories_and_deleted_refs_are_complete(self):
        self.workspace.mutate(
            lambda io: (io.unlink("ledger/old.json"), io.rmdir("ledger"),
                        io.mkdir("stream/empty"), io.mkdir("new-empty")),
            estimated_growth_bytes=1024, allocation_bytes=64 * 1024)
        path = self.workspace.checkpoint("directories")
        data = StateWorkspace.validate_checkpoint(path, self.baseline)
        self.assertEqual(data["deleted_files"], ["ledger/old.json"])
        self.assertEqual(data["deleted_directories"], ["ledger"])
        self.assertIn("new-empty", data["directories"])
        self.assertTrue((path / "stream/empty").is_dir())
        (path / "unlisted-directory").mkdir()
        with self.assertRaises(IntegrityError):
            StateWorkspace.validate_checkpoint(path, self.baseline)

    def test_completed_delta_crash_before_archive_keeps_latest_best_and_saved_intent(self):
        self.child(
            "ws.checkpoint('delta-crash',latest=True,best=True)\n"
            "os._exit(0)\n")
        self.reopen()
        self.assertEqual(self.cache.stats()["slots"]["latest"], "delta-crash")
        self.assertEqual(self.cache.stats()["slots"]["best"], "delta-crash")
        self.assertIsNone(self.state()["entries"]["delta-crash"]["archive"])
        self.assertEqual(len(self.workspace.archive_pending()), 1)
        self.assertIn("delta-crash", self.cache.stats()["entries"])
        self.assertIsNotNone(self.state()["entries"]["delta-crash"]["ack"])

    def test_delta_copy_uses_trusted_prebounded_materializer_not_per_block_tree_writer(self):
        self.workspace.mutate(
            lambda io: io.write_bytes("stream/new-large.bin", b"x" * (3 * MiB)),
            estimated_growth_bytes=3 * MiB)
        with patch("skill3d.storage.hot_cache.Reservation.open",
                   side_effect=AssertionError("delta used per-block cache-scanning writer")):
            path = self.workspace.checkpoint("bounded-copy")
        data = StateWorkspace.validate_checkpoint(path, self.baseline)
        self.assertEqual(data["files"]["stream/new-large.bin"]["size"], 3 * MiB)

    def test_latest_and_best_survive_restart_and_ack_only_old_outputs_delete(self):
        best = self.workspace.checkpoint("best", latest=True, best=True)
        old = self.workspace.checkpoint("old-latest")
        latest = self.workspace.checkpoint("latest")
        self.workspace.transfer = lambda path, manifest: None
        self.assertEqual(self.workspace.archive_pending(), [])
        self.assertTrue(all(path.exists() for path in (best, old, latest)))
        with self.assertRaises(StorageError):
            self.cache.collect_acknowledged("old-latest")
        self.reopen()
        self.assertEqual(self.cache.stats()["slots"]["latest"], "latest")
        self.assertEqual(self.cache.stats()["slots"]["best"], "best")
        self.assertEqual(len(self.workspace.archive_pending()), 3)
        self.assertTrue(best.exists())
        self.assertTrue(latest.exists())
        self.assertFalse(old.exists())
        self.assertTrue(self.raw.exists())
        self.assertEqual(len(self.workspace.archive_pending()), 2)

    def test_checkpoint_capacity_refusal_happens_before_any_copy(self):
        with patch("skill3d.storage.hot_cache.Reservation.open") as opened:
            with self.assertRaises(QuotaExceeded):
                self.workspace.checkpoint("too-small", estimated_bytes=1)
            self.assertEqual(opened.call_count, 0)
        self.assertNotIn("too-small", self.cache.stats()["entries"])

    def test_bounded_atomic_overrun_preserves_previous_completed_file(self):
        before = (self.raw / "stream/state.json").read_bytes()
        with self.assertRaises(QuotaExceeded):
            self.workspace.mutate(
                lambda io: io.write_bytes("stream/state.json", b"x" * (11 * MiB)),
                estimated_growth_bytes=1)
        self.assertEqual((self.raw / "stream/state.json").read_bytes(), before)
        self.assertEqual(list((self.raw / ".state-workspace/tmp").iterdir()), [])

    def test_immutable_baseline_protocol_paths_and_closed_capability_are_blocked(self):
        caps = []
        def produce(io):
            caps.append(io)
            self.assertFalse(hasattr(io, "path"))
            for name in ("offline_images/old.bin", ".state-workspace/journal.json", "../escape"):
                with self.assertRaises(StorageError):
                    io.write_bytes(name, b"changed")
        self.workspace.mutate(produce, estimated_growth_bytes=1024)
        with self.assertRaises(StorageError):
            caps[0].read_bytes("stream/state.json")

    def test_active_lease_blocks_mutation_adoption_and_checkpoint_across_handles(self):
        other = HotCache(self.cache.root, quota_bytes=35 * MiB, disk_usage=free_disk)
        with other.lease("migration-state"):
            with self.assertRaises(EntryBusy):
                self.workspace.mutate(Mock(), estimated_growth_bytes=1)
            with self.assertRaises(EntryBusy):
                self.workspace.checkpoint("busy")
            with self.assertRaises(EntryBusy):
                other.adopt_workspace("migration-state", growth_bytes=1)

    def test_workspace_never_evicts_or_enters_archive_protocol_even_unpinned(self):
        self.cache.pin("state", None)
        with self.assertRaises(QuotaExceeded):
            self.cache.reserve(output_bytes=30 * MiB)
        self.assertTrue(self.raw.exists())
        self.assertNotIn("migration-state", self.cache.archive_candidates())
        with self.assertRaises(StorageError):
            self.cache.queue_archive("migration-state", **route("state", "workspace").as_dict())
        with self.assertRaises(StorageError):
            self.cache.collect_acknowledged("migration-state")
        self.reopen()
        self.assertTrue(self.raw.exists())

    def test_archive_async_does_not_hold_mutation_lease_while_waiting_for_ack(self):
        self.workspace.publish_output("async-output", lambda r: r.write_bytes("payload/a", b"a"),
                                      estimated_bytes=1)
        entered, release = threading.Event(), threading.Event()
        def transfer(path, manifest):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("synthetic ACK wait timed out")
            return self.receiver(path, manifest)
        self.workspace.transfer = transfer
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = self.workspace.archive_async(executor)
            try:
                self.assertTrue(entered.wait(10))
                self.workspace.mutate(lambda io: io.write_bytes("stream/during-ack", b"done"),
                                      estimated_growth_bytes=1024)
                self.assertIn("async-output", self.cache.stats()["entries"])
            finally:
                release.set()
            self.assertEqual(len(future.result(timeout=10)), 1)
        self.assertNotIn("async-output", self.cache.stats()["entries"])
        self.assertEqual((self.raw / "stream/during-ack").read_bytes(), b"done")

    def test_receiver_commit_before_local_ack_crash_retries_same_saved_identity(self):
        self.workspace.publish_output("ack-crash", lambda r: r.write_bytes("payload/a", b"a"),
                                      estimated_bytes=1)
        self.child(
            "def transfer(path,manifest):\n"
            "    receiver(path,manifest)\n"
            "    os._exit(0)\n"
            "ws.transfer=transfer\n"
            "ws.archive_pending()\n")
        self.reopen()
        self.assertIn("ack-crash", self.cache.stats()["entries"])
        self.assertEqual(len(self.workspace.archive_pending()), 1)
        self.assertNotIn("ack-crash", self.cache.stats()["entries"])
        self.assertEqual(len(list((self.receiver.root / "objects").iterdir())), 1)

    def test_bad_ack_does_not_delete_completed_output(self):
        path = self.workspace.publish_output(
            "bad-ack", lambda r: r.write_bytes("payload/a", b"a"), estimated_bytes=1)
        self.workspace.transfer = lambda path, manifest: {"invalid": "ack"}
        with self.assertRaises(IntegrityError):
            self.workspace.archive_pending()
        self.assertTrue(path.exists())
        self.assertIsNone(self.state()["entries"]["bad-ack"]["ack"])


if __name__ == "__main__":
    unittest.main()
