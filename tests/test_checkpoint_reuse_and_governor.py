"""Synthetic checkpoint custody and process-local governor regression checks."""
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from skill3d.storage.framework_state_io import FrameworkStateIO, FrameworkStorageRefused
from skill3d.storage.hot_cache import ArchiveReceiver, HotCache, IntegrityError
from skill3d.storage.state_workspace import ArchiveRoute, StateWorkspace
from skill3d.synthesis.request_governor import RequestGovernorConfig

MiB = 1024 * 1024


def disk(_):
    return shutil._ntuple_diskusage(200 * 10**9, 0, 100 * 10**9)


class CheckpointReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3d-checkpoint-reuse-")
        self.root = Path(self.temp.name)
        self.cache = HotCache(self.root / "cache", quota_bytes=64 * MiB, disk_usage=disk)
        seed = b"immutable seed"
        self.cache.populate(
            "state", lambda r: r.write_bytes("payload/seed", seed),
            kind="result", output_bytes=len(seed))
        self.ws = StateWorkspace.adopt(
            self.cache, "state", baseline_anchor="1" * 64,
            baseline_members={"seed": {
                "size": len(seed), "sha256": hashlib.sha256(seed).hexdigest()}},
            mutable_roots=("mutable", "request_governor"),
            workspace_bytes=16 * MiB, allocation_bytes=16 * MiB)
        self.ws.bounded = True
        self.io = FrameworkStateIO(self.ws, job="synthetic")

    def tearDown(self):
        self.io.finish()
        self.temp.cleanup()

    def checkpoint(self, key, **kwargs):
        return self.ws.checkpoint(
            key, route=ArchiveRoute("synthetic", key, 0), **kwargs)

    def validate(self, path):
        with self.ws._write_lease() as budget:
            baseline = self.ws._baseline(budget, self.ws._journal(budget))
        return StateWorkspace.validate_checkpoint(path, baseline)

    def test_unchanged_file_reuses_only_immutable_checkpoint_inode(self):
        source = self.io.root / "mutable/large"
        self.io.write_bytes(source, b"x" * MiB)
        first = self.checkpoint("first")
        self.io.write_bytes(self.io.root / "mutable/new", b"new")
        second = self.checkpoint("second")
        self.assertNotEqual(source.stat().st_ino, (first / "mutable/large").stat().st_ino)
        self.assertEqual((first / "mutable/large").stat().st_ino,
                         (second / "mutable/large").stat().st_ino)
        self.assertEqual(set(self.validate(second)["files"]), {"mutable/large", "mutable/new"})
        # A later in-place mutation of the live source never corrupts snapshots.
        source.write_bytes(b"y" * MiB)
        self.assertEqual((first / "mutable/large").read_bytes(), b"x" * MiB)
        self.assertEqual((second / "mutable/large").read_bytes(), b"x" * MiB)

    def test_changed_same_size_and_deleted_files_preserve_full_delta(self):
        self.io.write_bytes(self.io.root / "mutable/same-size", b"old")
        self.io.write_bytes(self.io.root / "mutable/gone", b"gone")
        first = self.checkpoint("first")
        with self.io.section() as active:
            active.unlink("mutable/gone")
        self.io.write_bytes(self.io.root / "mutable/same-size", b"new")
        second = self.checkpoint("second")
        delta = self.validate(second)
        self.assertNotIn("mutable/gone", delta["files"])
        self.assertEqual((second / "mutable/same-size").read_bytes(), b"new")
        self.assertNotEqual((first / "mutable/same-size").stat().st_ino,
                            (second / "mutable/same-size").stat().st_ino)
        self.assertEqual((first / "mutable/gone").read_bytes(), b"gone")

    def test_archiving_and_deleting_previous_keeps_new_snapshot_independent(self):
        self.io.write_bytes(self.io.root / "mutable/value", b"exact")
        first = self.checkpoint("first")
        second = self.checkpoint("second")
        receiver = ArchiveReceiver(self.root / "receiver", disk_usage=disk)
        self.ws.transfer = receiver.receive
        self.ws.archive_pending()
        self.assertFalse(first.exists())
        self.assertTrue(second.exists())
        delta = self.validate(second)
        self.assertEqual((second / "mutable/value").read_bytes(), b"exact")
        with self.cache._locked() as state:
            saved = state["entries"]["second"]
        restored = receiver.published_path(
            self.cache.queue_archive("second", job="synthetic", shard="second",
                                     attempt=0).identity)
        self.assertEqual((restored / "mutable/value").read_bytes(), b"exact")
        self.assertEqual(json.loads((restored / ".workspace-delta.json").read_bytes()), delta)
        self.assertIsNotNone(saved["ack"])

    def test_unsupported_links_fall_back_to_independent_copy(self):
        self.io.write_bytes(self.io.root / "mutable/value", b"exact")
        first = self.checkpoint("first")
        with patch("skill3d.storage.state_workspace.os.link",
                   side_effect=OSError(errno.EOPNOTSUPP, "synthetic filesystem")):
            second = self.checkpoint("second")
        self.assertNotEqual((first / "mutable/value").stat().st_ino,
                            (second / "mutable/value").stat().st_ino)
        self.validate(second)

    def test_corrupt_previous_snapshot_cannot_publish_a_reused_file(self):
        self.io.write_bytes(self.io.root / "mutable/value", b"exact")
        first = self.checkpoint("first")
        (first / "mutable/value").write_bytes(b"wrong")
        with self.assertRaises(IntegrityError):
            self.checkpoint("second")
        with self.cache._locked() as state:
            self.assertNotIn("second", state["entries"])
        self.assertEqual((self.io.root / "mutable/value").read_bytes(), b"exact")

    def test_previous_deleted_between_metadata_query_and_lease_falls_back_to_copy(self):
        self.io.write_bytes(self.io.root / "mutable/value", b"exact")
        first = self.checkpoint("first")
        receiver = ArchiveReceiver(self.root / "receiver", disk_usage=disk)
        self.cache.queue_archive("first", job="synthetic", shard="first", attempt=0)
        self.cache.transfer_one("first", receiver)
        original_lease = self.cache.lease

        @contextmanager
        def collect_before_lease(key, **kwargs):
            if key == "first":
                self.cache.pin("latest", None)
                self.assertTrue(self.cache.collect_acknowledged(key))
            with original_lease(key, **kwargs) as path:
                yield path

        with patch.object(self.cache, "lease", side_effect=collect_before_lease):
            second = self.checkpoint("second")
        self.assertFalse(first.exists())
        self.assertEqual((second / "mutable/value").read_bytes(), b"exact")
        self.validate(second)

    def test_previous_busy_falls_back_to_copy_without_revoking_its_lock(self):
        self.io.write_bytes(self.io.root / "mutable/value", b"exact")
        first = self.checkpoint("first")
        with self.cache.lease("first", exclusive=True, touch=False):
            second = self.checkpoint("second")
        self.assertNotEqual((first / "mutable/value").stat().st_ino,
                            (second / "mutable/value").stat().st_ino)
        self.assertEqual((second / "mutable/value").read_bytes(), b"exact")
        self.validate(second)

    def test_bad_previous_descriptor_identity_still_fails_closed(self):
        self.io.write_bytes(self.io.root / "mutable/value", b"exact")
        first = self.checkpoint("first")
        path = first / ".workspace-delta.json"
        descriptor = json.loads(path.read_bytes())
        descriptor["baseline_archive_sha256"] = "2" * 64
        path.write_text(json.dumps(descriptor))
        with self.assertRaisesRegex(IntegrityError, "baseline identity changed"):
            self.checkpoint("second")
        self.assertFalse(self.cache.contains("second"))

    def test_generic_workspace_keeps_independent_copy_mode(self):
        self.io.write_bytes(self.io.root / "mutable/value", b"exact")
        self.ws.reuse_checkpoint_files = False
        first, second = self.checkpoint("first"), self.checkpoint("second")
        self.assertNotEqual((first / "mutable/value").stat().st_ino,
                            (second / "mutable/value").stat().st_ino)

    def test_governor_cached_per_process_root_and_frozen_policy(self):
        root = self.io.root / "request_governor"
        policy = RequestGovernorConfig(2, 1)
        governor = self.io.request_governor(root, policy)
        self.assertIs(self.io.request_governor(root, policy), governor)
        with governor.request("solver", {"variant_id": "synthetic"}):
            pass
        self.assertEqual(self.io.request_governor(root, policy).summary()["reserved_requests"], 1)
        with self.assertRaises(ValueError):
            self.io.request_governor(root, RequestGovernorConfig(3, 1))
        clone = pickle.loads(pickle.dumps(self.io))
        self.assertEqual(clone._governors, {})
        self.assertTrue(clone._workspace.reuse_checkpoint_files)
        recovered = clone.request_governor(root, policy)
        self.assertIsNot(recovered, governor)
        self.assertEqual(recovered.summary()["reserved_requests"], 1)

    def test_fork_process_reset_discards_inherited_governor_handle(self):
        root = self.io.root / "request_governor"
        policy = RequestGovernorConfig(2, 1)
        governor = self.io.request_governor(root, policy)
        self.io._pid = os.getpid() - 1
        self.assertIsNot(self.io.request_governor(root, policy), governor)

    def test_nested_requests_retain_outer_permit_after_invalid_and_valid_inner_calls(self):
        governor = self.io.request_governor(
            self.io.root / "request_governor", RequestGovernorConfig(5, 2))
        with governor.request("solver", {"variant_id": "synthetic"}):
            self.assertEqual(governor.summary()["active_requests"], 1)
            with self.assertRaises(ValueError):
                with governor.request("invalid", {}):
                    self.fail("invalid request admitted")
            self.assertEqual(governor.summary()["active_requests"], 1)
            with governor.request("gate", {}):
                self.assertEqual(governor.summary()["active_requests"], 2)
            self.assertEqual(governor.summary()["active_requests"], 1)
        summary = governor.summary()
        self.assertEqual(summary["active_requests"], 0)
        self.assertEqual(summary["completed_requests"], 2)
        self.assertEqual(summary["unknown_requests"], 0)
        self.assertEqual(summary["peak_inflight"], 2)

    def test_nested_admission_failure_releases_only_inner_permit(self):
        governor = self.io.request_governor(
            self.io.root / "request_governor", RequestGovernorConfig(5, 2))
        with governor.request("solver", {}):
            with patch.object(governor, "_atomic",
                              side_effect=FrameworkStorageRefused("synthetic refusal")):
                with self.assertRaises(FrameworkStorageRefused):
                    with governor.request("gate", {}):
                        self.fail("refused request admitted")
            self.assertEqual(governor.summary()["active_requests"], 1)
            with governor.request("gate", {}):
                self.assertEqual(governor.summary()["active_requests"], 2)
        self.assertEqual(governor.summary()["completed_requests"], 2)

    def test_runtime_cache_handle_is_reused_and_not_serialized(self):
        from skill3d.storage.scene_runtime import SceneHotRuntime
        runtime = SceneHotRuntime.__new__(SceneHotRuntime)
        runtime.cache_root = self.cache.root
        with patch("skill3d.storage.scene_runtime.HotCache", return_value=self.cache) as create:
            self.assertIs(runtime.cache(), self.cache)
            self.assertIs(runtime.cache(), self.cache)
            self.assertEqual(create.call_count, 1)
            clone = pickle.loads(pickle.dumps(runtime))
            self.assertIsNone(clone._cache_handle)
            self.assertIs(clone.cache(), self.cache)
            self.assertEqual(create.call_count, 2)

    def test_cached_runtime_recovers_child_crash_after_durable_publish_intent(self):
        from skill3d.storage import hot_cache as hot
        from skill3d.storage.scene_runtime import SceneHotRuntime
        runtime = SceneHotRuntime.__new__(SceneHotRuntime)
        runtime.cache_root = self.cache.root
        runtime.closure = {"shards": {"scene": {"key": "interrupted"}}}
        with patch("skill3d.storage.scene_runtime.HotCache", return_value=self.cache):
            self.assertIs(runtime.cache(), self.cache)
        code = f"""
import os,shutil
from unittest.mock import patch
from skill3d.storage.hot_cache import HotCache
def disk(_): return shutil._ntuple_diskusage(200*10**9,0,100*10**9)
cache=HotCache({str(self.cache.root)!r},quota_bytes={self.cache.quota_bytes},disk_usage=disk)
replace=os.replace
def crash(source,target):
    if target==cache._entry_path('interrupted'): os._exit(73)
    return replace(source,target)
with patch('skill3d.storage.hot_cache.os.replace',side_effect=crash):
    cache.populate('interrupted',lambda r:r.write_bytes('payload/value',b'exact'),
                   kind='input',output_bytes=5)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 73)
        with self.cache._locked() as state:
            self.assertFalse(state["entries"]["interrupted"]["ready"])
            self.assertIn("staging", state["entries"]["interrupted"])
        with patch.object(runtime, "_request", side_effect=AssertionError("network")), \
                patch.object(self.cache, "recover", wraps=self.cache.recover) as recovery:
            runtime._populate(runtime.cache(), "scene", slot="current")
            self.assertEqual(recovery.call_count, 1)
        self.assertEqual((self.cache._entry_path("interrupted") / "value").read_bytes(), b"exact")
        self.assertTrue(self.cache.contains("interrupted"))
        with patch.object(hot, "_tree_bytes", side_effect=AssertionError("scan")), \
                patch.object(self.cache, "recover", side_effect=AssertionError("recover")), \
                patch.object(self.cache, "_save", side_effect=AssertionError("save")):
            runtime._populate(runtime.cache(), "scene", slot="current")
        with self.cache._locked() as state:
            self.assertEqual(state["reservations"], {})

    def test_transport_callbacks_do_not_serialize_scene_closure_or_credentials(self):
        from skill3d.storage.scene_runtime import SceneArchiveTransport, SceneHotRuntime
        runtime = SceneHotRuntime.__new__(SceneHotRuntime)
        runtime.endpoint = "http://synthetic.invalid"
        runtime.token_file = self.root / "never-read.token"
        runtime.closure = {"marker": "never-serialize-this-scene" * 100000}
        self.ws.transfer = runtime.archive_transfer
        self.io.enable_cold_history(runtime.fetch_cold_member)
        encoded = pickle.dumps(self.io)
        self.assertLess(len(encoded), 20000)
        self.assertNotIn(b"never-serialize-this-scene", encoded)
        self.assertFalse(runtime.token_file.exists())
        clone = pickle.loads(encoded)
        self.assertIsInstance(clone._workspace.transfer.__self__, SceneArchiveTransport)
        self.assertIsInstance(clone._cold_fetch.__self__, SceneArchiveTransport)
        self.assertEqual(clone._workspace.transfer.__self__.endpoint, runtime.endpoint)
        self.assertEqual(clone._cold_fetch.__self__.token_file, runtime.token_file)


if __name__ == "__main__":
    unittest.main()
