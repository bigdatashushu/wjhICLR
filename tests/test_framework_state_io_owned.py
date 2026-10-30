"""Synthetic-only storage integration checks; every temporary tree stays in WORK.

Run with PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -B
tests/test_framework_state_io_owned.py. No pytest fixtures, runtime initialization,
real data, network, environment credentials, or model backend is used.
"""

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import importlib.abc
import importlib.util
import io as bytes_io
import json
import multiprocessing
from pathlib import Path
import pickle
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class _NoRemoteImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.endswith(".remote"):
            raise AssertionError("remote backend imports are forbidden in synthetic checks")
        return None


sys.meta_path.insert(0, _NoRemoteImports())

from skill3d.storage.hot_cache import (
    ArchiveAck, ArchiveIdentity, EntryBusy, HotCache, IdentityConflict, IntegrityError,
    QuotaExceeded, StorageError,
)
from skill3d.storage.state_workspace import ArchiveRoute, StateWorkspace
from skill3d.storage.framework_state_io import (
    FrameworkStateIO, FrameworkStorageRefused, WorkspaceRequestGovernor,
)
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

WORK = Path(__file__).resolve().parents[1]


class _Disk:
    total = 200_000_000_000
    used = 0
    free = 100_000_000_000


def _disk_usage(_path):
    return _Disk()


def _ack(_path, manifest):
    return ArchiveAck(
        ArchiveIdentity(manifest["job"], manifest["shard"], manifest["attempt"],
                        manifest["sha256"]), manifest["manifest_sha256"])

def _fetch_forbidden(_ack, _ref, _writer):
    raise AssertionError("synthetic restore is not requested in this test")


def _spawn_write(io, connection):
    try:
        io.write_json_once(io.root / "accumulation" / "results" / "spawn.json",
                           {"synthetic": True})
        connection.send("ok")
    except BaseException as exc:
        connection.send(type(exc).__name__)
    finally:
        io.finish()
        connection.close()


def _skill(task, state, version="1.0.0", body="Inspect the supplied evidence."):
    from skill3d.schemas import SkillSpecV11
    name = task.replace("_", "-") + "-" + state
    return SkillSpecV11(
        skill_id="S-" + task + "-" + state, version=version,
        question_type=task, evidence_state=state,
        skill_md=f"---\nname: {name}\ndescription: Synthetic method.\n---\n{body}\n")


def _case(index, skill):
    from skill3d.evolution.fitset_stream_v11 import FitSetExperienceCase
    profile = {"geometry_3d": "unavailable", "metric_scale": "unavailable"}
    return FitSetExperienceCase(
        case_id=f"synthetic-case-{index}", qa_id=f"synthetic-qa-{index}",
        scene_id="synthetic-scene", question_type=skill.question_type,
        evidence_state=skill.evidence_state, source_split="induction",
        delivered_skill_version=f"{skill.skill_id}@{skill.version}",
        delivered_content_sha256=skill.content_sha256, delivered_skill_md=skill.skill_md,
        score=0.0, outcome="failure", failure_categories=["synthetic"],
        trace_ref="synthetic-trace", trace_sha256="b" * 64,
        initial_evidence_states=profile,
        evidence_history=[{"evidence_states": profile}])


def _cli_module():
    name = "run_fitset_v11_owned"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, WORK / "scripts" / "run_fitset_v11.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class _MockClient:
    def __init__(self, io, text='{"synthetic":true}'):
        self.io, self.text = io, text
        self.calls = 0
        self.last_usage = {"total_tokens": 1}

    def chat(self, *_args, **_kwargs):
        self.calls += 1
        if getattr(self.io._local, "io", None) is not None:
            raise AssertionError("mock callback holds workspace EX")
        with self.io._workspace._write_lease():
            pass
        return self.text


class FrameworkStateIOChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="owned-state-io-", dir=WORK)
        self.base = Path(self.temp.name)
        self.cache = HotCache(self.base / "cache", quota_bytes=64 * 1024 * 1024,
                              disk_usage=_disk_usage)
        self.seed = b"synthetic immutable seed"

        def populate(reservation):
            with reservation.open("payload/seed.bin") as stream:
                stream.write(self.seed)

        self.cache.populate("state", populate, kind="result", output_bytes=4096)
        StateWorkspace.adopt(
            self.cache, "state", baseline_anchor="a" * 64,
            baseline_members={"seed.bin": {
                "size": len(self.seed), "sha256": hashlib.sha256(self.seed).hexdigest()}},
            mutable_roots=("accumulation", "request_governor"),
            workspace_bytes=8 * 1024 * 1024)
        self.workspace = StateWorkspace(self.cache, "state", bounded=True, transfer=_ack)
        self.io = FrameworkStateIO(self.workspace, job="synthetic-job", attempt=7)

    def tearDown(self):
        try:
            self.io.finish()
        finally:
            self.temp.cleanup()

    def _envelope(self):
        with self.cache._locked() as state:
            row = state["entries"]["state"]
            return row["workspace_bytes"], row["workspace_charged_bytes"]

    def _journal(self):
        with self.workspace._write_lease() as budget:
            return self.workspace._journal(budget)

    def test_reviser_status_stays_inside_declared_scheduler_receipts(self):
        from skill3d.evaluation.reviser_pool_v11 import OwnerReviserPool
        self.io.mutable_roots = ("accumulation/scheduler_receipts",)
        pool = object.__new__(OwnerReviserPool)
        pool.runner = SimpleNamespace(
            root=self.io.root / "accumulation", run_identity="synthetic",
            _write_json_atomic=self.io.write_json)
        pool.cap, pool.peak = 4, 2
        pool.busy = {0: ("recover", []), 3: ("ingest", [(None, None)])}
        pool.processes = [SimpleNamespace(pid=123), SimpleNamespace(pid=456)]
        pool._status()
        path = pool.runner.root / "scheduler_receipts/reviser_pool_status.json"
        value = json.loads(path.read_bytes())
        self.assertEqual(value["reviser_concurrency"], 4)
        self.assertEqual(value["peak_active_owner_transactions"], 2)
        self.assertFalse((pool.runner.root / "reviser_pool_status.json").exists())

    def test_fixed_envelope_and_declared_roots(self):
        before = self._envelope()
        self.io.write_json(self.io.root / "accumulation" / "summary.json", {"n": 1})
        called = []
        with self.assertRaises(QuotaExceeded):
            self.workspace.mutate(lambda io: called.append(True),
                                  estimated_growth_bytes=9 * 1024 * 1024)
        self.assertFalse(called)
        self.assertEqual(self._envelope(), before)
        with self.assertRaises(StorageError):
            self.workspace.mutate(lambda io: io.write_bytes("other/new", b"x"),
                                  estimated_growth_bytes=0)
        with self.assertRaises(StorageError):
            self.io.write_bytes(self.io.root / "seed.bin", b"changed")
        with self.assertRaises(StorageError):
            self.io.write_bytes(self.io.root / "accumulation" / ".." / "escape", b"x")
        self.assertEqual((self.io.root / "seed.bin").read_bytes(), self.seed)

    def test_peak_replacement_and_per_file_refusal_leave_old_bytes(self):
        path = self.io.root / "accumulation" / "trace.jsonl"
        self.io.write_bytes(path, b"original\n")
        with self.assertRaises(FrameworkStorageRefused):
            self.io.write_bytes(path, b"x" * (8 * 1024 * 1024))
        self.assertEqual(path.read_bytes(), b"original\n")
        self.assertFalse(list((self.io.root / ".state-workspace" / "tmp").glob("*")))
        self.io.max_file_bytes = 10
        with self.assertRaises(FrameworkStorageRefused):
            self.io.append_bytes(path, b"longer\n")
        self.assertEqual(path.read_bytes(), b"original\n")

    def test_create_once_is_atomic_and_detects_conflicts(self):
        path = self.io.root / "accumulation" / "results" / "same.json"
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: self.io.write_json_once(path, {"same": True}), range(8)))
        with self.assertRaises(IntegrityError):
            self.io.write_json_once(path, {"same": False})
        self.assertEqual(json.loads(path.read_bytes()), {"same": True})
        self.assertFalse(list((self.io.root / ".state-workspace" / "tmp").glob("*")))

    def test_trace_encoding_claim_and_binding_match_legacy(self):
        from skill3d.trace.store import TraceStore
        bounded = self.io.trace_store(self.io.root / "accumulation" / "variants" / "x" / "trace")
        legacy = TraceStore(self.base / "legacy-trace")
        for store in (bounded, legacy):
            store.claim_run("synthetic-run")
            store.bind(run_id="synthetic-run", qa_id="synthetic-qa").append("rounds", {"n": 1})
            store.append_episode({"synthetic": True})
        for relative in (".runs/synthetic-run.json", "rounds.jsonl", "episode_trace.jsonl"):
            self.assertEqual((bounded.root / relative).read_bytes(),
                             (legacy.root / relative).read_bytes())
        with self.assertRaises(ValueError):
            bounded.claim_run("synthetic-run")
        with self.assertRaises(ValueError):
            bounded.bind(run_id="one").bind(run_id="two")

    def test_16_concurrent_mock_requests_do_not_hold_workspace_ex(self):
        policy = RequestGovernorConfig(max_requests=32, max_inflight=16)
        governor = WorkspaceRequestGovernor(self.io, self.io.root / "request_governor", policy)
        barrier = threading.Barrier(16)
        before = self._envelope()

        def episode(index):
            self.io.write_json_once(self.io.root / "accumulation" / "plans" / f"{index}.json",
                                    {"synthetic": index})
            with governor.request("solver", {"variant_id": f"synthetic-{index}"}) as receipt:
                self.assertIsNone(getattr(self.io._local, "io", None))
                barrier.wait(timeout=45)
                # A mock call's body can acquire EX, proving no episode-long
                # workspace SH or request-long EX is held by any of the 16.
                with self.workspace._write_lease():
                    pass
                receipt["usage"] = {"total_tokens": 1}
            self.io.write_json_once(self.io.root / "accumulation" / "results" / f"{index}.json",
                                    {"synthetic": index})

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(episode, index) for index in range(16)]
            for future in futures:
                future.result(timeout=90)
        summary = governor.summary()
        self.assertEqual(summary["completed_requests"], 16)
        self.assertEqual(summary["peak_inflight"], 16)
        self.assertEqual(summary["active_requests"], 0)
        self.assertEqual(summary["usage"], {"total_tokens": 16})
        self.assertEqual(self._envelope(), before)

    def test_request_refusal_is_not_model_failure_and_releases_permit(self):
        self.io.max_file_bytes = 512
        governor = self.io.request_governor(
            self.io.root / "request_governor", RequestGovernorConfig(2, 1))
        calls = []
        with self.assertRaises(FrameworkStorageRefused):
            with governor.request("solver", {"variant_id": "synthetic-" * 100}):
                calls.append(True)
        self.assertFalse(calls)
        self.assertEqual(governor.summary()["reserved_requests"], 0)
        with governor.request("solver", {"variant_id": "s"}):
            pass
        self.assertEqual(governor.summary()["completed_requests"], 1)

    def test_archives_saved_identities_latest_best_and_bounded_journal(self):
        before = self._envelope()
        self.io.write_bytes(
            self.io.root / "accumulation" / "workers" / "buffer" / "stream.json", b"{}")
        self.io.write_bytes(self.io.root / "accumulation" / "workers" / "buffer.lock", b"")
        for index in range(6):
            self.io.write_json(self.io.root / "accumulation" / "summary.json", {"n": index})
            self.io.after_batch(best=index in {0, 2})
            self.io.finish()
        stats = self.cache.stats()
        self.assertEqual(len(stats["entries"]), 3)
        self.assertNotEqual(stats["slots"]["latest"], stats["slots"]["best"])
        journal = self._journal()
        self.assertEqual(len(journal["outputs"]), 2)
        for key, record in journal["outputs"].items():
            self.assertEqual(record["route"]["job"], "synthetic-job")
            self.assertEqual(record["route"]["attempt"], 7)
            self.assertIsNotNone(record["ack"])
            with self.cache.lease(key) as path:
                from skill3d.storage.state_workspace import BaselineAnchor
                StateWorkspace.validate_checkpoint(path, BaselineAnchor(
                    "a" * 64, {"seed.bin": {"size": len(self.seed),
                                           "sha256": hashlib.sha256(self.seed).hexdigest()}}))
        self.assertEqual(self._envelope(), before)
        self.assertTrue((self.io.root / "accumulation" / "workers" / "buffer" / "stream.json").exists())

    def test_failed_ack_retains_bytes_and_resumes_saved_route(self):
        seen = []

        def wrong_ack(path, manifest):
            seen.append((manifest["job"], manifest["shard"], manifest["attempt"]))
            ack = _ack(path, manifest).as_dict()
            ack["attempt"] += 1
            return ack

        self.workspace.transfer = wrong_ack
        self.io.write_bytes(self.io.root / "accumulation" / "summary.json", b"{}")
        self.io.after_batch()
        with self.assertRaises(FrameworkStorageRefused):
            self.io.finish()
        key, record = next(iter(self._journal()["outputs"].items()))
        self.assertIsNone(record["ack"])
        self.assertIn(key, self.cache.stats()["entries"])

        def resumed_ack(path, manifest):
            self.assertEqual((manifest["job"], manifest["shard"], manifest["attempt"]), seen[0])
            return _ack(path, manifest)

        self.workspace.transfer = resumed_ack
        self.workspace.archive_pending()
        self.assertIsNotNone(self._journal()["outputs"][key]["ack"])
        self.assertTrue((self.io.root / "accumulation" / "summary.json").exists())

    def test_spawn_reconstructs_adapter_without_executor_or_module_patch(self):
        self.io.after_batch()
        self.io.finish()
        cloned = pickle.loads(pickle.dumps(self.io))
        self.assertIsNone(cloned._executor)
        self.assertIsNone(cloned._future)
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        process = context.Process(target=_spawn_write, args=(cloned, child))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(30))
            self.assertEqual(parent.recv(), "ok")
        finally:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join()
            parent.close()
        self.assertEqual(process.exitcode, 0)
        self.assertTrue((self.io.root / "accumulation" / "results" / "spawn.json").exists())

    def test_stream_finalized_before_archive_and_callback_has_no_ex(self):
        from skill3d.evolution.fitset_stream_v11 import FitSetStreamingBuffer
        from skill3d.evolution.streaming_v11 import (
            StreamingAttemptResult, StreamingEvolutionConfig, StreamingEvolutionRunner,
        )
        original_writer = StreamingEvolutionRunner._save
        skill = _skill("object_counting", "g0m0")
        root = self.io.root / "accumulation" / "workers" / "buffer"
        stream = FitSetStreamingBuffer(
            root, stream_id="synthetic-stream", config=StreamingEvolutionConfig(batch_size=2),
            framework_state_io=self.io)
        seen = []

        def callback(_request):
            self.assertIsNone(getattr(self.io._local, "io", None))
            with self.workspace._write_lease():
                pass
            return StreamingAttemptResult(outcome="rejected", campaign_id="synthetic-campaign")

        def checkpoint(*, best=False):
            value = json.loads((root / "stream.json").read_bytes())
            seen.append((best, value))

        with patch.object(self.io, "after_batch", side_effect=checkpoint):
            self.assertIsNone(stream.ingest(_case(0, skill), parent_snapshot_id="S", attempt=callback))
            result = stream.ingest(_case(1, skill), parent_snapshot_id="S", attempt=callback)
        self.assertEqual(result.outcome, "rejected")
        self.assertEqual(len(seen), 1)
        self.assertFalse(seen[0][0])
        self.assertEqual(len(stream.attempts), 1)
        restored = FitSetStreamingBuffer(root, stream_id="synthetic-stream",
                                        config=StreamingEvolutionConfig(batch_size=2),
                                        framework_state_io=self.io)
        self.assertEqual(restored.slots[skill.slot_key].attempt_count, 1)
        self.assertEqual(len(restored.slots[skill.slot_key].case_refs), 2)
        self.assertIs(StreamingEvolutionRunner._save, original_writer)

    def test_library_initialization_and_real_publication_are_admitted(self):
        from skill3d.evaluation.fitset_v11 import ALL_FIT_SLOTS
        from skill3d.evolution.fitset_stream_v11 import initialize_worker_library
        from skill3d.schemas import SkillCandidateV11
        from skill3d.skills import v11_library, promote_atomic
        original_writer = v11_library._write_create_once
        original_lock = promote_atomic.publish_lock
        skills = [_skill(task, state) for task, state in ALL_FIT_SLOTS]
        root = self.io.root / "accumulation" / "workers" / "00" / "library"
        initialize_worker_library(
            root, seed_skills=skills, owned_slots=[skills[0].slot_key],
            run_identity="c" * 64, framework_state_io=self.io)
        before = promote_atomic.read_active_snapshot(root / "snapshots")
        parent = skills[0]
        candidate = _skill(parent.question_type, parent.evidence_state, "1.0.1",
                           "Inspect the supplied evidence, then verify the count.")
        snapshot, receipt = self.io.publish_candidate(root, SkillCandidateV11(
            candidate_id="synthetic-candidate", parent_snapshot_id=before["snapshot_id"],
            parent_skill_version=f"{parent.skill_id}@{parent.version}",
            full_skill_spec=candidate))
        loaded = v11_library.validate_v11_snapshot(snapshot, library_root=root)
        self.assertIn(candidate, loaded)
        self.assertNotIn(parent, loaded)
        self.assertEqual(receipt["snapshot_before"], before["snapshot_id"])
        self.assertFalse((root / "snapshots" / ".publish.lock").exists())
        self.assertIs(v11_library._write_create_once, original_writer)
        self.assertIs(promote_atomic.publish_lock, original_lock)

    def test_paused_scope_extension_cas_preserves_anchor_and_quota(self):
        before = self._envelope()
        journal = self._journal()
        self.workspace.extend_mutable_roots(
            ["logs"], expected_roots=journal["mutable_roots"],
            baseline_sha256=journal["baseline_archive_sha256"],
            baseline_members_sha256=journal["baseline_members_sha256"])
        with self.assertRaises(IdentityConflict):
            self.workspace.extend_mutable_roots(
                ["receipts"], expected_roots=journal["mutable_roots"],
                baseline_sha256=journal["baseline_archive_sha256"],
                baseline_members_sha256=journal["baseline_members_sha256"])
        reopened = FrameworkStateIO(self.workspace, job="synthetic-job")
        reopened.write_bytes(reopened.root / "logs" / "synthetic.txt", b"synthetic")
        self.assertEqual(self._envelope(), before)

    def test_disabled_runner_helpers_remain_original(self):
        from skill3d.evaluation.fitset_v11 import SystemVariantRunner
        runner = object.__new__(SystemVariantRunner)
        runner.framework_state_io = None
        once = self.base / "legacy" / "once.json"
        atomic = self.base / "legacy" / "atomic.json"
        runner._write_create_once(once, {"n": 1})
        runner._write_json_atomic(atomic, {"n": 2})
        self.assertEqual(once.read_bytes(), b'{"n":1}\n')
        self.assertEqual(atomic.read_bytes(), b'{"n":2}\n')
        runner.framework_state_io = self.io
        runner._write_create_once(self.io.root / "accumulation" / "result.json", {"n": 1})
        self.assertEqual((self.io.root / "accumulation" / "result.json").read_bytes(), once.read_bytes())

    def test_free_space_floor_refuses_before_mutating_destination(self):
        path = self.io.root / "accumulation" / "summary.json"
        self.io.write_bytes(path, b"original")
        with patch.object(_Disk, "free", 25_000_000_000):
            with self.assertRaises(FrameworkStorageRefused):
                self.io.write_bytes(path, b"changed")
        self.assertEqual(path.read_bytes(), b"original")

    def test_input_contract_preserves_logical_identity_and_rejects_changed_sha(self):
        from skill3d.storage.published_transport import (
            PUBLISHED_TRANSPORT_SCHEMA, PublishedTransportLocator, RelocationError,
        )
        from skill3d.storage.scene_runtime import SceneHotInputProvider
        from skill3d.infra.m5_identity import file_identity
        physical = self.io.root / "accumulation" / "synthetic-manifest.json"
        self.io.write_json_once(physical, {"synthetic": True})
        identity = file_identity(physical)
        original = str(self.base / "logical-manifest.json")
        descriptor = {
            "schema_version": PUBLISHED_TRANSPORT_SCHEMA, "roots": {},
            "files": {original: {"path": str(physical), "identity": identity}},
        }
        provider = SimpleNamespace(input_contract_identity={
            "manifest": {"path": str(physical), "sha256": identity["sha256"]},
            "source_sha256": "c" * 64,
        })
        runtime = SimpleNamespace(
            closure={"original_published_manifest": original, "descriptor": descriptor},
            locator=PublishedTransportLocator(descriptor))
        wrapped = SceneHotInputProvider(provider, runtime)
        expected = copy.deepcopy(provider.input_contract_identity)
        expected["manifest"]["path"] = original
        self.assertEqual(wrapped.input_contract_identity, expected)
        self.assertEqual(provider.input_contract_identity["manifest"]["path"], str(physical))
        provider.input_contract_identity["manifest"]["sha256"] = "d" * 64
        with self.assertRaises(RelocationError):
            _ = wrapped.input_contract_identity
        provider.input_contract_identity["manifest"]["sha256"] = identity["sha256"]
        provider.input_contract_identity["manifest"]["path"] = original
        with self.assertRaises(IntegrityError):
            _ = wrapped.input_contract_identity
        descriptor["files"].clear()
        with self.assertRaises(IntegrityError):
            _ = wrapped.input_contract_identity

    def test_scene_lease_skips_only_adopted_workspace_state_sh(self):
        from skill3d.storage.scene_runtime import SceneHotRuntime
        for key, kind in (("global", "input"), ("scene", "input"), ("old-state", "result")):
            def populate(reservation):
                with reservation.open("payload/synthetic.txt") as stream:
                    stream.write(b"synthetic")
            self.cache.populate(key, populate, kind=kind, output_bytes=4096)
        runtime = object.__new__(SceneHotRuntime)
        runtime.scenes = {"synthetic-scene"}
        runtime.cache = lambda: self.cache
        runtime.closure = {"shards": {
            "global": {"key": "global"}, "state": {"key": "state"},
            "synthetic-scene": {"key": "scene"},
        }}
        with runtime.scene_lease("synthetic-scene"):
            with self.workspace._write_lease():
                pass
            for key in ("global", "scene"):
                with self.assertRaises(EntryBusy):
                    with self.cache.lease(key, exclusive=True):
                        pass
        runtime.closure["shards"]["state"]["key"] = "old-state"
        with runtime.scene_lease("synthetic-scene"):
            with self.assertRaises(EntryBusy):
                with self.cache.lease("old-state", exclusive=True):
                    pass

    def test_actual_32_offline_png_writes_keep_original_layout_and_identity(self):
        import numpy as np
        from PIL import Image
        from skill3d.evaluation.fitset_v11 import SystemVariantRunner, _variant_key
        runner = object.__new__(SystemVariantRunner)
        runner.root = self.io.root / "accumulation"
        runner.framework_state_io = self.io
        runner._continuation = None
        runner.published_inputs = None
        variant = {"variant_id": "synthetic-png", "qa_id": "synthetic-qa",
                   "scene_id": "synthetic-scene", "split": "induction"}
        fs = SimpleNamespace(frame_ids=list(range(32)), source_frame_indices=list(range(32)),
                             frame_set_hash="b" * 64)
        pixels = [np.full((4, 5, 3), index, dtype=np.uint8) for index in range(32)]
        item = SimpleNamespace(
            episode=SimpleNamespace(frame_set=fs, question="Synthetic public input.", options=[]),
            pixels=pixels)
        before = self._envelope()
        runner._freeze_offline_case_input(variant, item)
        frozen = runner.offline_case_input(variant)
        parent = runner.root / "variants" / _variant_key(variant["variant_id"]) / "offline_images"
        self.assertEqual(len(frozen["image_refs"]), 32)
        for index, ref in enumerate(frozen["image_refs"]):
            buffer = bytes_io.BytesIO()
            Image.fromarray(pixels[index]).save(buffer, format="PNG")
            path = Path(ref["path"])
            self.assertEqual(path.parent, parent)
            self.assertEqual(path.name, f"{index:02d}.png")
            self.assertEqual(path.read_bytes(), buffer.getvalue())
            self.assertEqual(ref["file_sha256"], hashlib.sha256(buffer.getvalue()).hexdigest())
        runner._freeze_offline_case_input(variant, item)
        self.assertEqual(self._envelope(), before)
        refused = {**variant, "variant_id": "synthetic-refused-png"}
        self.io.max_file_bytes = 32
        calls = []
        with self.assertRaises(FrameworkStorageRefused):
            runner._freeze_offline_case_input(refused, item)
            calls.append("mock-paid-call")
        self.assertFalse(calls)
        self.assertFalse((runner.root / "variants" / _variant_key(refused["variant_id"])
                          / "offline_images").exists())

    def test_revision_and_diagnosis_adapters_write_without_locking_mock_calls(self):
        from skill3d.evolution.fitset_revision_v11 import FitSetSlotReviser
        from skill3d.evolution.diagnosis_v11 import V11CaseDiagnoser, OfflineDiagnosisConfig
        import skill3d.evolution.diagnosis_v11 as diagnosis
        parent = _skill("object_counting", "g0m0")
        original = FitSetSlotReviser._request
        client = _MockClient(self.io)
        reviser = FitSetSlotReviser(
            output_root=self.io.root / "accumulation" / "workers" / "00" / "revision",
            panel={}, panel_name="synthetic", variants=[],
            accumulation_root=self.io.root / "accumulation", label_metadata={},
            library_root=self.io.root / "accumulation" / "workers" / "00" / "library",
            client=client)
        self.io.adapt_reviser(reviser)
        exchange = reviser._request(slot=("object_counting", "g0m0"), attempt=1,
                                    prompt="Synthetic method request.")
        self.assertEqual(client.calls, 1)
        self.assertEqual(exchange["response_text"], '{"synthetic":true}')
        self.assertEqual(reviser._request(slot=("object_counting", "g0m0"), attempt=1,
                                         prompt="Synthetic method request."), exchange)
        self.assertEqual(client.calls, 1)
        self.assertIs(FitSetSlotReviser._request, original)
        case = {"case_id": "synthetic-case", "score": 0, "source_split": "induction",
                "delivered_content_sha256": parent.content_sha256}
        full = {"campaign_id": "synthetic-campaign", "cases": [case],
                "source_split": "induction", "label_access": True,
                "parent_content_sha256": parent.content_sha256,
                "parent_skill_version": f"{parent.skill_id}@{parent.version}"}
        client = _MockClient(self.io, '{"observations":[],"hypotheses":[],"unknowns":["Synthetic."]}')
        context = {"campaign_id": "synthetic-campaign"}
        diagnoser = V11CaseDiagnoser(
            client, reviser.root / "diagnosis", OfflineDiagnosisConfig(mode="text"))
        self.io.adapt_diagnoser(diagnoser, context)
        receipt = diagnoser.diagnose(parent, full, context=context)
        self.assertEqual(diagnoser.diagnose(parent, full, context=context), receipt)
        self.assertEqual(client.calls, 1)
        self.assertEqual(receipt["diagnosis"]["unknowns"], ["Synthetic."])
        audit = diagnoser.audit_root / diagnosis._sha(context)[:24]
        self.assertTrue((audit / "request_intent.json").is_file())
        self.assertTrue((audit / "diagnosis.json").is_file())
        self.assertEqual((audit / "request.lock").read_bytes(), b"")

    def test_background_failure_surfaces_before_next_write_and_finalizers_drain(self):
        from skill3d.evaluation.fitset_v11 import SystemVariantRunner
        self.workspace.transfer = lambda *_args: (_ for _ in ()).throw(RuntimeError("synthetic"))
        self.io.write_json(self.io.root / "accumulation" / "summary.json", {"n": 1})
        self.io.after_batch()
        with self.assertRaises(FrameworkStorageRefused):
            self.io._future.result(timeout=30)
        path = self.io.root / "accumulation" / "never-written.json"
        with self.assertRaises(FrameworkStorageRefused):
            self.io.write_json(path, {"n": 2})
        self.assertFalse(path.exists())
        runner = object.__new__(SystemVariantRunner)
        runner.framework_state_io = self.io
        seen = []
        def body(*_args, **_kwargs):
            self.assertIsNone(getattr(self.io._local, "io", None))
            seen.append("body")
            return {"synthetic": True}
        with patch.object(runner, "_run_body", side_effect=body, create=True):
            with self.assertRaises(FrameworkStorageRefused):
                runner.run(resume=True)
        self.assertEqual(seen, ["body"])
        self.assertIsNone(self.io._executor)
        self.workspace.transfer = _ack
        self.workspace.archive_pending()
        with patch.object(runner, "_run_shard_body", side_effect=body, create=True):
            self.assertEqual(runner._run_shard(0, [], [], None, lambda _: None), {"synthetic": True})
        self.assertEqual(seen, ["body", "body"])

    def test_cli_defaults_factory_pickle_and_sanitized_storage_stop(self):
        cli = _cli_module()
        command = ["run-system", "--panel", "synthetic-panel", "--panel-name", "full",
                   "--metadata", "synthetic-metadata", "--phase", "accumulation",
                   "--output-dir", str(self.io.root / "accumulation")]
        # Select a real parser choice without loading any panel.
        command[command.index("full")] = cli.PANEL_NAMES[0]
        args = cli.build_parser().parse_args(command)
        self.assertEqual(args.state_workspace_key, "")
        self.assertEqual(args.state_max_file_bytes, 64 * 1024 * 1024)
        self.assertEqual(args.min_free_gib, 80.0)
        configured = cli.build_parser().parse_args(command + [
            "--state-workspace-key", "state", "--state-archive-job", "synthetic-job",
            "--state-archive-attempt", "7", "--state-lock-timeout", "2"])
        self.assertEqual(configured.state_archive_attempt, 7)
        self.assertEqual(configured.state_lock_timeout, 2)
        gate = cli._GateClientFactory({}, framework_state_io=self.io)
        seeded = gate(137, self.io.root / "accumulation")
        cloned = pickle.loads(pickle.dumps(seeded))
        self.assertEqual(cloned.seed, 137)
        self.assertEqual(cloned.framework_state_io.job, "synthetic-job")
        self.assertIsNone(cloned.framework_state_io._executor)
        configured.handler = lambda _args: (_ for _ in ()).throw(
            FrameworkStorageRefused("private diagnostic must not print"))
        with patch.object(cli, "build_parser") as parser, patch("sys.stderr", new=bytes_io.StringIO()) as out:
            parser.return_value.parse_args.return_value = configured
            self.assertEqual(cli.main([]), 75)
        self.assertEqual(out.getvalue(),
                         "FrameworkStorageRefused: storage paused; no automatic retry\n")

    def test_narrow_scope_refuses_destination_before_dispatch(self):
        journal = self._journal()
        # Reopen against a deliberately narrow synthetic operator scope.
        with self.workspace._write_lease() as budget:
            journal["mutable_roots"] = ["accumulation/summary.json", "accumulation/workers"]
            self.workspace._save_journal(budget, journal)
        reopened = FrameworkStateIO(self.workspace, job="synthetic-job")
        calls = []
        with self.assertRaises(FrameworkStorageRefused):
            reopened.validate_destinations(
                [reopened.root / "accumulation" / name for name in ("variants", "results")])
            calls.append("dispatch")
        self.assertFalse(calls)
        self.assertEqual(self._journal()["mutable_roots"], journal["mutable_roots"])

    def test_completed_history_retains_active_episodes_and_restores_exact_ack_bytes(self):
        from skill3d.storage.hot_cache import ArchiveReceiver
        receiver = ArchiveReceiver(self.base / "receiver", disk_usage=_disk_usage)
        self.workspace.transfer = receiver.receive
        def fetch(ack, ref, writer):
            writer.write((receiver.published_path(ack.identity) / ref).read_bytes())
        self.io.enable_cold_history(fetch)
        base = self.io.root / "accumulation"
        retired = {
            base / "variants/done/trace/topic.jsonl": b"synthetic finished trace\n",
            base / "variants/done/offline_images/00.png": b"synthetic frozen image",
            base / "workers/00/campaigns/finished/gate/0/parent/variants/g/outcome.json":
                b'{"synthetic":"finished gate"}',
        }
        protected = {
            base / "variants/active/trace/topic.jsonl": b"synthetic active trace\n",
            base / "variants/done/outcome.json": b'{"synthetic":"learning evidence"}',
            base / "workers/00/campaigns/active/gate/0/parent/variants/g/outcome.json":
                b'{"synthetic":"active gate"}',
        }
        for path, raw in {**retired, **protected}.items():
            self.io.write_bytes(path, raw)
        self.io.write_json(base / "results/done.json", {"synthetic": True})
        self.io.write_json(base / "workers/00/campaigns/finished/result.json",
                           {"synthetic": True})
        envelope = self._envelope()
        self.io._archive_completed_history()
        for path, raw in retired.items():
            self.assertFalse(path.exists())
            self.assertEqual(self.io.read_history_bytes(path), raw)
            self.assertEqual(self.io.cold_reference(path).sha256,
                             hashlib.sha256(raw).hexdigest())
        for path, raw in protected.items():
            self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(self._envelope(), envelope)
        self.io._archive_completed_history()
        self.assertEqual(len(self.io._cold_history.records()), len(retired))

    def test_checkpoint_carries_cold_baseline_custody_instead_of_deletion(self):
        from skill3d.storage.hot_cache import ArchiveReceiver
        receiver = ArchiveReceiver(self.base / "receiver", disk_usage=_disk_usage)
        self.workspace.transfer = receiver.receive
        self.io.enable_cold_history(_fetch_forbidden)
        self.io._cold_history.collect(("seed.bin",), ArchiveRoute(
            "synthetic-job", "immutable-cold-seed", 1), "immutable-cold-seed")
        self.assertFalse((self.io.root / "seed.bin").exists())
        path = self.workspace.checkpoint(
            "cold-checkpoint", route=ArchiveRoute("synthetic-job", "cold-checkpoint", 1))
        with self.workspace._write_lease() as budget:
            journal = self.workspace._journal(budget)
            baseline = self.workspace._baseline(budget, journal)
        delta = self.workspace.validate_checkpoint(path, baseline)
        self.assertNotIn("seed.bin", delta["deleted_files"])
        self.assertIn("cold_history", delta)
        record = self.io.cold_reference(self.io.root / "seed.bin")
        self.assertEqual(record.identity, dict(baseline.files["seed.bin"]))

    def test_shared_scene_png_retention_and_cold_diagnosis_preserve_bytes(self):
        import numpy as np
        from skill3d.evaluation.fitset_v11 import SystemVariantRunner, _variant_key
        from skill3d.evolution.diagnosis_v11 import FrozenCaseImageProvider
        from skill3d.infra.m5_identity import content_sha256
        from skill3d.storage.hot_cache import ArchiveReceiver
        from skill3d.storage.offline_frames import scene_key
        receiver = ArchiveReceiver(self.base / "shared-receiver", disk_usage=_disk_usage)
        self.workspace.transfer = receiver.receive
        def fetch(ack, ref, writer):
            writer.write((receiver.published_path(ack.identity) / ref).read_bytes())
        self.io.enable_cold_history(fetch)
        runner = object.__new__(SystemVariantRunner)
        runner.root = self.io.root / "accumulation"
        runner.framework_state_io = self.io
        runner._continuation = None
        runner._cached_prefix_spec = None
        runner.published_inputs = None
        runner.offline_frame_storage = "shared_scene_v2"
        variants = [
            {"variant_id": "shared-qa-" + str(index), "qa_id": str(index),
             "scene_id": "shared-scene", "split": "induction"} for index in range(2)]
        fs = SimpleNamespace(frame_ids=list(range(32)), source_frame_indices=list(range(32)),
                             frame_set_hash="a" * 64)
        item = SimpleNamespace(episode=SimpleNamespace(frame_set=fs, question="Synthetic.", options=[]),
                               pixels=[np.full((4, 5, 3), i, np.uint8) for i in range(32)])
        manifest = {"shared_frame_consumers": {
            scene_key("shared-scene"): sorted(_variant_key(row["variant_id"]) for row in variants)}}
        self.io.write_json_once(runner.root / "manifest.json",
                               {**manifest, "identity_sha256": content_sha256(manifest)})
        for variant in variants:
            runner._freeze_offline_case_input(variant, item)
        source = runner.offline_case_input(variants[0])
        originals = {Path(ref["path"]): Path(ref["path"]).read_bytes()
                     for ref in source["image_refs"]}
        envelope = self._envelope()
        self.io.write_json_once(runner._result_path(variants[0]["variant_id"]),
                                {**variants[0], "offline_input_sha256": hashlib.sha256(
                                    (runner.root / "variants" / _variant_key(variants[0]["variant_id"])
                                     / "offline_input.json").read_bytes()).hexdigest()})
        self.io._archive_completed_history()
        self.assertTrue(all(path.exists() for path in originals))
        self.io.write_json_once(runner._result_path(variants[1]["variant_id"]),
                                {**variants[1], "offline_input_sha256": hashlib.sha256(
                                    (runner.root / "variants" / _variant_key(variants[1]["variant_id"])
                                     / "offline_input.json").read_bytes()).hexdigest()})
        self.io._archive_completed_history()
        self.assertTrue(all(not path.exists() for path in originals))
        self.assertEqual(runner.offline_case_input(variants[0]), source)
        provider = FrozenCaseImageProvider([runner.root], framework_state_io=self.io)
        import base64
        delivered = provider({}, source["image_refs"][0])
        self.assertEqual(base64.b64decode(delivered["data_url"].split(",", 1)[1]),
                         next(iter(originals.values())))
        self.io._archive_completed_history()
        self.assertEqual(len(self.io._cold_history.records()), 32)
        self.assertEqual(self._envelope(), envelope)

    def test_cold_transport_reconstructs_with_spawned_framework_io(self):
        self.io.enable_cold_history(_fetch_forbidden)
        cloned = pickle.loads(pickle.dumps(self.io))
        self.assertTrue(cloned._workspace.atomic_writes)
        self.assertIsNotNone(cloned._cold_history)
        self.assertIs(cloned._cold_fetch, _fetch_forbidden)
        self.assertEqual(cloned._cold_history.records(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
