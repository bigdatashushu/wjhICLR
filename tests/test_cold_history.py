"""Synthetic storage tests only; no application fixtures, network or solving.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover \
    -s tests -p test_cold_history.py -v
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

from skill3d.storage.cold_history import (
    COLD_HISTORY_PATH, MAX_COLD_INDEX_BYTES, MAX_COLD_RECORDS,
    MAX_COLD_ARCHIVE_RECORDS, ColdHistory,
    validate_cold_index,
)
from skill3d.storage.hot_cache import (
    ArchiveAck, ArchiveIdentity, ArchiveReceiver, EntryBusy, GB, HotCache,
    IdentityConflict, IntegrityError, MIN_FREE_BYTES, QuotaExceeded, StorageError,
    _digest, _json_bytes,
)
from skill3d.storage.state_workspace import (
    ArchiveRoute, StateWorkspace, WorkspaceIO,
)

MiB = 1024 * 1024
ROUTE = ArchiveRoute("synthetic-cold", "completed", 0)
KEY = "cold-completed-0"


def free_disk(_path):
    return shutil._ntuple_diskusage(2000 * GB, 500 * GB, 1500 * GB)


class ColdHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h3d-cold-synthetic-")
        self.root = Path(self.temp.name)
        self.cache = HotCache(self.root / "cache", quota_bytes=35 * MiB, disk_usage=free_disk)
        self.receiver = ArchiveReceiver(self.root / "receiver", disk_usage=free_disk)
        self.seed = {
            "completed/a.bin": b"synthetic-a" * 100,
            "completed/nested/b.bin": b"synthetic-b" * 150,
            "stream/state.bin": b"synthetic-active",
        }
        self.refs = tuple(sorted(ref for ref in self.seed if ref.startswith("completed/")))
        self.raw = self.cache.populate(
            "workspace", lambda r: [
                r.write_bytes("payload/" + ref, data) for ref, data in self.seed.items()],
            kind="result", output_bytes=sum(map(len, self.seed.values())), slots=("state",))
        self.workspace = StateWorkspace.adopt(
            self.cache, "workspace", baseline_anchor=hashlib.sha256(b"synthetic-baseline").hexdigest(),
            baseline_members={
                ref: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for ref, data in self.seed.items()},
            mutable_roots=("stream",), workspace_bytes=10 * MiB,
            allocation_bytes=10 * MiB, transfer=self.receiver)
        self.workspace.bounded = True
        self.fetch = Mock(side_effect=self.fetch_member)
        self.history = ColdHistory(self.workspace, self.fetch)

    def tearDown(self):
        self.temp.cleanup()

    def fetch_member(self, ack, member_key, writer):
        with (self.receiver.published_path(ack.identity) / member_key).open("rb") as stream:
            for block in iter(lambda: stream.read(257), b""):
                writer.write(block)

    def index(self):
        return json.loads((self.raw / COLD_HISTORY_PATH).read_bytes())

    def cache_state(self):
        return json.loads((self.cache.root / "state.json").read_bytes())

    def replace_index(self, value, *, rehash=False):
        if rehash:
            value["index_sha256"] = _digest(
                {key: row for key, row in value.items() if key != "index_sha256"})
        with self.workspace._write_lease() as budget:
            WorkspaceIO(budget, internal=True).write_bytes(COLD_HISTORY_PATH, _json_bytes(value))

    def collect(self):
        return self.history.collect(self.refs, ROUTE, KEY)

    def reopen(self):
        self.cache = HotCache(self.cache.root, quota_bytes=35 * MiB, disk_usage=free_disk)
        self.workspace = StateWorkspace(
            self.cache, "workspace", transfer=self.receiver, bounded=True)
        self.history = ColdHistory(self.workspace, self.fetch)

    def child(self, body):
        prelude = (
            "import os, json, shutil\n"
            "from pathlib import Path\n"
            "from skill3d.storage.hot_cache import HotCache, ArchiveReceiver, GB\n"
            "from skill3d.storage.state_workspace import StateWorkspace, ArchiveRoute, WorkspaceIO\n"
            "from skill3d.storage.cold_history import ColdHistory\n"
            "def disk(p): return shutil._ntuple_diskusage(2000*GB,500*GB,1500*GB)\n"
            f"cache=HotCache(Path({str(self.cache.root)!r}),quota_bytes={35 * MiB},disk_usage=disk)\n"
            f"receiver=ArchiveReceiver(Path({str(self.receiver.root)!r}),disk_usage=disk)\n"
            "ws=StateWorkspace(cache,'workspace',transfer=receiver,bounded=True)\n"
            "history=ColdHistory(ws,lambda *a:None)\n"
            f"refs={self.refs!r}\n"
            f"route=ArchiveRoute({ROUTE.job!r},{ROUTE.shard!r},{ROUTE.attempt!r})\n"
            f"key={KEY!r}\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", prelude + body], capture_output=True, text=True,
            timeout=30, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_exact_ack_backed_records_survive_unlink_and_fetch_without_workspace_growth(self):
        before = self.cache_state()["entries"]["workspace"]
        result = self.collect()
        self.assertEqual(set(result), set(self.refs))
        self.assertNotIn(KEY, self.cache.stats()["entries"])
        self.assertEqual(self.history.records(), result)
        for ref in self.refs:
            self.assertFalse((self.raw / ref).exists())
            record = result[ref]
            self.assertEqual(record.member_key, ref)
            self.assertEqual(record.ack.identity.job, ROUTE.job)
            self.assertEqual(self.history.identity_for(ref), {
                "size": len(self.seed[ref]), "sha256": hashlib.sha256(self.seed[ref]).hexdigest()})
            self.assertEqual(self.history.read_bytes(ref), self.seed[ref])
        after = self.cache_state()["entries"]["workspace"]
        self.assertEqual(after["workspace_bytes"], before["workspace_bytes"])
        self.assertEqual(after["workspace_charged_bytes"], before["workspace_charged_bytes"])
        self.assertEqual((self.raw / "stream/state.bin").read_bytes(), self.seed["stream/state.bin"])
        self.assertEqual(self.cache.stats()["reservations"], 0)
        self.assertEqual(json.loads((self.raw / ".state-workspace/journal.json").read_bytes())["outputs"], {})

    def test_no_ack_keeps_raw_and_unacknowledged_index_then_retry_never_reproduces(self):
        self.workspace.transfer = Mock(return_value=None)
        publish = self.workspace.publish_output
        with patch.object(self.workspace, "publish_output", wraps=publish) as producer:
            self.assertEqual(self.collect(), {})
            self.assertEqual(producer.call_count, 1)
            self.assertEqual(self.history.records(), {})
            self.assertIsNone(self.index()["archives"][KEY]["ack"])
            for ref in self.refs:
                self.assertEqual((self.raw / ref).read_bytes(), self.seed[ref])
            with self.assertRaises(StorageError):
                self.history.read_bytes(self.refs[0])
            first_manifest = self.index()["archives"][KEY]["manifest"]
            self.workspace.transfer = self.receiver
            self.assertEqual(len(self.collect()), len(self.refs))
            self.assertEqual(producer.call_count, 1)
            self.assertEqual(self.index()["archives"][KEY]["manifest"], first_manifest)

    def test_mismatched_ack_cannot_authorize_unlink(self):
        def mismatch(path, manifest):
            ack = self.receiver(path, manifest)
            return ArchiveAck(
                ArchiveIdentity(ack.identity.job, ack.identity.shard, ack.identity.attempt, "f" * 64),
                ack.manifest_sha256)
        self.workspace.transfer = mismatch
        with self.assertRaises(IdentityConflict):
            self.collect()
        self.assertIsNone(self.index()["archives"][KEY]["ack"])
        self.assertTrue(all((self.raw / ref).exists() for ref in self.refs))

    def test_ack_saved_before_transport_returns_and_before_any_raw_unlink(self):
        unlink = WorkspaceIO.unlink
        def checked_unlink(io, ref):
            self.assertIsNotNone(self.index()["archives"][KEY]["ack"])
            return unlink(io, ref)
        with patch.object(WorkspaceIO, "unlink", checked_unlink):
            self.collect()

    def test_changed_raw_after_upload_aborts_entire_selected_set_removal(self):
        def transfer(path, manifest):
            ack = self.receiver(path, manifest)
            with self.workspace._write_lease() as budget:
                WorkspaceIO(budget, internal=True).write_bytes(self.refs[-1], b"changed-after-upload")
            return ack
        self.workspace.transfer = transfer
        with self.assertRaises(IntegrityError):
            self.collect()
        self.assertIsNotNone(self.index()["archives"][KEY]["ack"])
        self.assertTrue(all((self.raw / ref).exists() for ref in self.refs))
        self.assertEqual(self.history.read_bytes(self.refs[-1]), self.seed[self.refs[-1]])
        with self.assertRaises(IntegrityError):
            self.collect()

    def test_active_reference_lease_prevents_collection_including_other_process(self):
        with self.history.lease(self.refs):
            with self.assertRaises(EntryBusy):
                self.collect()
            output = self.child(
                "from skill3d.storage.hot_cache import EntryBusy\n"
                "try: history.collect(refs,route,key)\n"
                "except EntryBusy: print('busy')\n"
                "else: raise AssertionError('active lease ignored')\n")
            self.assertEqual(output.strip(), "busy")
        self.assertFalse((self.raw / COLD_HISTORY_PATH).exists())
        self.collect()

    def test_global_retention_lock_serializes_collectors_during_upload(self):
        entered, release = threading.Event(), threading.Event()
        def transfer(path, manifest):
            entered.set()
            if not release.wait(10):
                raise AssertionError("synthetic retention test timed out")
            return self.receiver(path, manifest)
        self.workspace.transfer = transfer
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.collect)
            try:
                self.assertTrue(entered.wait(10))
                with self.assertRaises(EntryBusy):
                    self.history.collect(("stream/state.bin",), ArchiveRoute("other", "set", 0), "other")
            finally:
                release.set()
            self.assertEqual(len(future.result(timeout=10)), len(self.refs))

    def test_scoped_archive_does_not_transfer_or_drop_unrelated_output_intent(self):
        other_route = ArchiveRoute("synthetic-other", "unrelated", 0)
        self.workspace.publish_output(
            "unrelated", lambda r: r.write_bytes("payload/x", b"unrelated"),
            estimated_bytes=9, route=other_route)
        self.collect()
        self.assertIn("unrelated", self.cache.stats()["entries"])
        with self.workspace._write_lease() as budget:
            journal = self.workspace._journal(budget)
        self.assertEqual(set(journal["outputs"]), {"unrelated"})
        self.assertIsNone(journal["outputs"]["unrelated"]["ack"])
        self.assertIsNone(self.cache_state()["entries"]["unrelated"]["archive"])

    def test_materialized_hot_path_is_leased_and_reused_then_evictable(self):
        self.collect()
        with self.history.materialize(self.refs[0]) as path:
            self.assertEqual(path.read_bytes(), self.seed[self.refs[0]])
            self.assertFalse(path.is_relative_to(self.raw))
            key = next(k for k in self.cache.stats()["entries"] if k.startswith("cold-member:"))
            with self.assertRaises(EntryBusy):
                with self.cache.lease(key, exclusive=True):
                    pass
            with self.assertRaises(EntryBusy):
                self.collect()
            with self.history.materialize(self.refs[0]) as same:
                self.assertEqual(same, path)
        self.assertEqual(self.fetch.call_count, 1)
        with self.cache.lease(key, exclusive=True):
            pass
        with self.cache._locked() as state:
            self.assertTrue(self.cache._deletable(state, key, state["entries"][key]))
        self.assertFalse((self.raw / self.refs[0]).exists())

    def test_quota_pressure_cannot_evict_an_active_read_but_can_evict_after_release(self):
        ref = self.refs[0]
        payload = b"synthetic" * (250 * 1024)
        with self.workspace._write_lease() as budget:
            WorkspaceIO(budget, internal=True).write_bytes(ref, payload)
        self.collect()
        index = self.index()
        with self.history.materialize(ref) as path:
            key = next(k for k in self.cache.stats()["entries"] if k.startswith("cold-member:"))
            with self.assertRaises(QuotaExceeded):
                with self.cache.reserve(output_bytes=23 * MiB):
                    pass
            self.assertIn(key, self.cache.stats()["entries"])
            self.assertEqual(path.read_bytes(), payload)
        with self.cache.reserve(output_bytes=23 * MiB):
            self.assertNotIn(key, self.cache.stats()["entries"])
            self.assertFalse(path.exists())
        self.assertEqual(self.history.read_bytes(ref), payload)
        self.assertEqual(self.fetch.call_count, 2)
        self.assertEqual(self.index(), index)
        self.assertFalse((self.raw / ref).exists())

    def test_fetch_tamper_truncation_and_oversize_never_publish_a_hot_copy(self):
        self.collect()
        ref = self.refs[0]
        for payload in (b"x" * len(self.seed[ref]), self.seed[ref][:-1], self.seed[ref] + b"x"):
            with self.subTest(size=len(payload)):
                bad = ColdHistory(self.workspace, lambda ack, member, writer: writer.write(payload))
                with self.assertRaises(IntegrityError):
                    bad.read_bytes(ref)
                self.assertEqual(self.cache.stats()["entries"], ["workspace"])
                self.assertEqual(self.cache.stats()["reservations"], 0)
        self.assertEqual(self.history.read_bytes(ref), self.seed[ref])

    def test_cached_hot_copy_tamper_is_rejected_not_silently_refetched(self):
        self.collect()
        with self.history.materialize(self.refs[0]) as path:
            pass
        path.write_bytes(b"x" * len(self.seed[self.refs[0]]))
        with self.assertRaises(IntegrityError):
            self.history.read_bytes(self.refs[0])
        self.assertEqual(self.fetch.call_count, 1)

    def test_traversal_private_refs_duplicate_paths_and_symlinks_are_rejected(self):
        for ref in ("../escape", "/absolute", "a//b", "a/./b", "a\\b",
                    ".state-workspace/journal.json", ".workspace-delta.json"):
            with self.subTest(ref=ref):
                with self.assertRaises(StorageError):
                    self.history.collect((ref,), ROUTE, KEY)
                with self.assertRaises(StorageError):
                    self.history.read_bytes(ref)
        with self.assertRaises(IdentityConflict):
            self.history.collect((self.refs[0], self.refs[0]), ROUTE, KEY)
        (self.raw / "completed/link").symlink_to(self.raw / self.refs[0])
        with self.assertRaises(IntegrityError):
            self.history.collect(("completed/link",), ROUTE, KEY)
        (self.raw / "completed/link").unlink()
        self.assertTrue(all((self.raw / ref).exists() for ref in self.refs))

    def test_tampered_index_checksum_raw_hash_ack_and_member_key_are_rejected(self):
        self.collect()
        original = self.index()
        for field, value, rehash in (
            ("sha256", "f" * 64, False),
            ("sha256", "f" * 64, True),
            ("member_key", "../escape", True),
            ("ack", "f" * 64, True),
        ):
            with self.subTest(field=field, rehash=rehash):
                changed = json.loads(_json_bytes(original))
                if field == "ack":
                    changed["archives"][KEY]["ack"]["manifest_sha256"] = value
                else:
                    changed["archives"][KEY]["files"][self.refs[0]][field] = value
                self.replace_index(changed, rehash=rehash)
                with self.assertRaises((IntegrityError, IdentityConflict)):
                    self.history.records()
        self.replace_index(original)
        self.assertEqual(len(self.history.records()), len(self.refs))

    def test_duplicate_json_and_symlinked_index_are_rejected(self):
        self.collect()
        original = (self.raw / COLD_HISTORY_PATH).read_bytes()
        with self.workspace._write_lease() as budget:
            WorkspaceIO(budget, internal=True).write_bytes(
                COLD_HISTORY_PATH, b'{"schema_version":"a","schema_version":"b"}')
        with self.assertRaises(IntegrityError):
            self.history.records()
        with self.workspace._write_lease() as budget:
            WorkspaceIO(budget, internal=True).write_bytes(COLD_HISTORY_PATH, original)
        (self.root / "synthetic-index.json").write_bytes(original)
        (self.raw / COLD_HISTORY_PATH).unlink()
        (self.raw / COLD_HISTORY_PATH).symlink_to(self.root / "synthetic-index.json")
        with self.assertRaises(IntegrityError):
            self.history.records()
        (self.raw / COLD_HISTORY_PATH).unlink()

    def test_immutable_keys_routes_and_raw_refs_cannot_be_rebound(self):
        self.collect()
        with self.assertRaises(IdentityConflict):
            self.history.collect(self.refs, ArchiveRoute("changed", "completed", 0), KEY)
        with self.assertRaises(IdentityConflict):
            self.history.collect(self.refs[:1], ROUTE, KEY)
        with self.assertRaises(IdentityConflict):
            self.history.collect(("stream/state.bin",), ROUTE, "new-key")
        with self.assertRaises(IdentityConflict):
            self.history.collect((self.refs[0],), ArchiveRoute("new", "route", 0), "new-key")
        with self.workspace._write_lease() as budget:
            WorkspaceIO(budget, internal=True).write_bytes(self.refs[0], self.seed[self.refs[0]])
        with self.assertRaises(IdentityConflict):
            self.history.collect((self.refs[0],), ArchiveRoute("new", "route", 0), "new-key")
        self.assertTrue((self.raw / self.refs[0]).exists())

    def test_admission_count_and_index_cap_refuse_before_output_producer(self):
        with patch.object(self.workspace, "publish_output", wraps=self.workspace.publish_output) as produce:
            with self.assertRaises(QuotaExceeded):
                self.history.collect((f"synthetic/{i}" for i in range(MAX_COLD_RECORDS + 1)), ROUTE, KEY)
            with patch("skill3d.storage.cold_history.MAX_COLD_INDEX_BYTES", 1000):
                with self.assertRaises(QuotaExceeded):
                    self.collect()
            self.assertEqual(produce.call_count, 0)
        self.assertFalse((self.raw / COLD_HISTORY_PATH).exists())
        self.assertTrue(all((self.raw / ref).exists() for ref in self.refs))

    def test_index_workspace_peak_admission_does_not_enlarge_envelope(self):
        with self.cache._locked() as state:
            row = state["entries"]["workspace"]
            row["workspace_bytes"] = sum(p.stat().st_size for p in self.raw.rglob("*") if p.is_file()) + 100
            self.cache._save(state)
        before = self.cache_state()["entries"]["workspace"]["workspace_bytes"]
        with patch.object(self.workspace, "publish_output") as producer:
            with self.assertRaises(QuotaExceeded):
                self.collect()
            producer.assert_not_called()
        self.assertEqual(self.cache_state()["entries"]["workspace"]["workspace_bytes"], before)
        self.assertFalse((self.raw / COLD_HISTORY_PATH).exists())

    def test_fetch_quota_and_floor_refusal_call_zero_transport_and_keep_cold_index(self):
        self.collect()
        before = self.index()
        self.cache.disk_usage = lambda p: shutil._ntuple_diskusage(
            2000 * GB, 500 * GB, MIN_FREE_BYTES)
        with self.assertRaises(QuotaExceeded):
            self.history.read_bytes(self.refs[0])
        self.fetch.assert_not_called()
        self.cache.disk_usage = free_disk
        with self.assertRaises(QuotaExceeded):
            ColdHistory(self.workspace, self.fetch, max_read_bytes=1).read_bytes(self.refs[0])
        self.fetch.assert_not_called()
        self.assertEqual(self.index(), before)
        self.assertEqual(self.cache.stats()["reservations"], 0)

    def test_fetch_cache_quota_refusal_happens_before_transport(self):
        ref = self.refs[0]
        with self.workspace._write_lease() as budget:
            WorkspaceIO(budget, internal=True).write_bytes(ref, b"synthetic" * (250 * 1024))
        self.collect()
        with self.cache._locked() as state:
            # Synthetic policy saturation: persistent workspace admission takes
            # nearly the full quota, even though physical seed files are tiny.
            state["entries"]["workspace"]["workspace_charged_bytes"] = 34 * MiB
            self.cache._save(state)
        with self.assertRaises(QuotaExceeded):
            self.history.read_bytes(ref)
        self.assertEqual(self.fetch.call_count, 0)
        self.assertIn(ref, self.history.records())

    def test_ack_index_write_failure_preserves_raw_and_retry_reuses_published_output(self):
        with patch.object(self.history, "_save_ack", side_effect=QuotaExceeded("synthetic index full")):
            with self.assertRaises(QuotaExceeded):
                self.collect()
        self.assertTrue(all((self.raw / ref).exists() for ref in self.refs))
        self.assertIsNone(self.index()["archives"][KEY]["ack"])
        self.assertIsNone(self.cache_state()["entries"][KEY]["ack"])
        with patch.object(self.workspace, "publish_output", side_effect=AssertionError("reproduced")):
            self.collect()

    def test_archive_record_limit_has_a_bounded_manifest_and_index(self):
        files = {f"synthetic/{i:05d}.bin": {
            "size": 0, "sha256": hashlib.sha256(b"").hexdigest(),
            "member_key": f"synthetic/{i:05d}.bin"}
                 for i in range(MAX_COLD_ARCHIVE_RECORDS)}
        value = {"schema_version": "h3d-cold-history-v1", "workspace_key": "workspace",
                 "archives": {KEY: {"route": ROUTE.as_dict(), "files": files,
                                   "manifest": None, "ack": None}}}
        value["index_sha256"] = _digest(value)
        self.assertEqual(validate_cold_index(value, workspace_key="workspace"), {})
        self.assertLess(self.history._projected(value, KEY), MAX_COLD_INDEX_BYTES)

    def test_large_collection_requires_split_before_intent_or_raw_read(self):
        refs = [f"synthetic/{index:05d}.bin"
                for index in range(MAX_COLD_ARCHIVE_RECORDS + 1)]
        with self.assertRaises(QuotaExceeded):
            self.history.collect(refs, ROUTE, KEY)
        self.assertFalse((self.raw / COLD_HISTORY_PATH).exists())

    def test_atomic_workspace_io_contract_is_preserved(self):
        self.workspace.atomic_writes = True
        self.collect()
        self.assertEqual(self.history.read_bytes(self.refs[0]), self.seed[self.refs[0]])
        self.assertEqual(self.cache.stats()["reservations"], 0)

    def test_fixed_ceiling_workspace_is_required_without_implicit_growth(self):
        self.workspace.bounded = False
        with self.assertRaises(StorageError):
            ColdHistory(self.workspace, self.fetch)
        self.assertFalse((self.raw / COLD_HISTORY_PATH).exists())

    def test_validator_accepts_parent_held_budget_and_rejects_expected_identity_mismatch(self):
        self.collect()
        ref = self.refs[0]
        with self.workspace._write_lease() as budget:
            records = self.history.records(budget=budget)
            self.assertEqual(self.history.identity_for(ref, budget=budget), records[ref].identity)
            self.assertEqual(
                self.history.validate_reference(
                    ref, expected_identity=records[ref].identity, budget=budget), records[ref])
            with self.assertRaises(IntegrityError):
                self.history.validate_reference(
                    ref, expected_identity={"size": 0, "sha256": "0" * 64}, budget=budget)
        with self.assertRaises(StorageError):
            self.history.records(budget=budget)
        changed = self.index()
        changed["workspace_key"] = "other-workspace"
        with self.assertRaises(IntegrityError):
            validate_cold_index(changed, workspace_key="workspace")

    def test_crash_after_publication_resumes_saved_output_without_reproducing(self):
        self.child(
            "def stop(path,manifest): os._exit(0)\n"
            "ws.transfer=stop\n"
            "history.collect(refs,route,key)\n")
        manifest = self.index()["archives"][KEY]["manifest"]
        self.reopen()
        with patch.object(self.workspace, "publish_output", side_effect=AssertionError("reproduced")):
            self.collect()
        self.assertEqual(self.index()["archives"][KEY]["manifest"], manifest)
        self.assertFalse(any((self.raw / ref).exists() for ref in self.refs))

    def test_crash_after_ack_before_unlink_preserves_receipt_and_needs_no_transport(self):
        self.child(
            "history._remove_raw=lambda *args: os._exit(0)\n"
            "history.collect(refs,route,key)\n")
        self.assertIsNotNone(self.index()["archives"][KEY]["ack"])
        self.assertTrue(all((self.raw / ref).exists() for ref in self.refs))
        self.reopen()
        self.workspace.transfer = Mock(side_effect=AssertionError("retried saved ACK"))
        self.collect()
        self.workspace.transfer.assert_not_called()
        self.assertNotIn(KEY, self.cache.stats()["entries"])
        self.assertEqual(self.history.read_bytes(self.refs[0]), self.seed[self.refs[0]])

    def test_crash_between_cold_and_cache_ack_writes_recovers_without_transport(self):
        self.child(
            "save=history._save_ack\n"
            "def stop(key,ack):\n"
            "    save(key,ack)\n"
            "    os._exit(0)\n"
            "history._save_ack=stop\n"
            "history.collect(refs,route,key)\n")
        self.assertIsNotNone(self.index()["archives"][KEY]["ack"])
        self.assertIsNone(self.cache_state()["entries"][KEY]["ack"])
        self.reopen()
        self.workspace.transfer = Mock(side_effect=AssertionError("retried saved ACK"))
        self.collect()
        self.workspace.transfer.assert_not_called()
        self.assertNotIn(KEY, self.cache.stats()["entries"])
        self.assertEqual(self.history.read_bytes(self.refs[0]), self.seed[self.refs[0]])

    def test_crash_during_raw_unlink_keeps_all_immutable_records_and_resumes(self):
        self.child(
            "unlink=WorkspaceIO.unlink\n"
            "def stop(io,ref):\n"
            "    unlink(io,ref)\n"
            "    os._exit(0)\n"
            "WorkspaceIO.unlink=stop\n"
            "history.collect(refs,route,key)\n")
        self.assertEqual(sum((self.raw / ref).exists() for ref in self.refs), 1)
        self.reopen()
        self.assertEqual(set(self.history.records()), set(self.refs))
        self.collect()
        self.assertFalse(any((self.raw / ref).exists() for ref in self.refs))
        for ref in self.refs:
            self.assertEqual(self.history.read_bytes(ref), self.seed[ref])

    def test_uncommitted_intent_is_never_reproduced_on_resume(self):
        self.child(
            "ws.publish_output=lambda *a,**kw: os._exit(0)\n"
            "history.collect(refs,route,key)\n")
        self.reopen()
        with patch.object(self.workspace, "publish_output", side_effect=AssertionError("reproduced")):
            with self.assertRaises(StorageError):
                self.collect()
        self.assertTrue(all((self.raw / ref).exists() for ref in self.refs))
        self.assertEqual(self.history.records(), {})

    def test_absent_baseline_members_keep_exact_identities_for_parent_delta_validation(self):
        self.collect()
        with self.workspace._write_lease() as budget:
            baseline = self.workspace._baseline(budget, self.workspace._journal(budget))
            for ref in self.refs:
                self.assertFalse((budget.path / ref).exists())
                record = self.history.validate_reference(
                    ref, expected_identity=baseline.files[ref], budget=budget)
                self.assertEqual(record.identity, dict(baseline.files[ref]))
            self.assertTrue((budget.path / "completed/nested").is_dir())


if __name__ == "__main__":
    unittest.main()
