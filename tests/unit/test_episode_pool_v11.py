"""Concurrency, version windows and interruption recovery for the episode pool."""

import json
import os
import threading
import time
from dataclasses import dataclass

import pytest

from skill3d.evaluation.fitset_v11 import _variant_key, _write_create_once, FitSetError
from skill3d.evolution.fitset_stream_v11 import freeze_fitset_stream
from test_fitset_stream_v11 import ProbeRunner, make_runner


class ConcurrentProbeRunner(ProbeRunner):
    def _run_one(self, variant):
        with self._episode_permit():
            started = time.time()
            result = self._probe_one(variant)
            interval_root = getattr(self, "_global_interval_root", None)
            if interval_root is not None:
                key = _variant_key(str(self.root) + "/" + variant["variant_id"])
                _write_create_once(interval_root / f"{key}.json", {
                    "started": started, "ended": time.time()})
            return result

    def _probe_one(self, variant):
        assert threading.current_thread() is threading.main_thread()
        # Only learning episodes participate; gate arms use their own roots.
        if getattr(self, "_request_category", "solver") == "solver":
            assert self.label_metadata == {}
            started = time.time()
            marker = self.root / "overlap" / f"{_variant_key(variant['variant_id'])}.json"
            _write_create_once(marker, {"pid": os.getpid(), "started": started})
            if variant["wave_index"] < 12:
                deadline = time.monotonic() + 40
                while len(list(marker.parent.glob("*.json"))) < self.workers:
                    assert time.monotonic() < deadline, "pool never reached requested concurrency"
                    time.sleep(0.05)
            row = super()._run_one(variant)
            _write_create_once(self.root / "intervals" / marker.name, {
                "pid": os.getpid(), "started": started, "ended": time.time(),
                "variant_id": variant["variant_id"],
            })
            return row
        return super()._run_one(variant)


def pool_runner(tmp_path, workers):
    runner = make_runner(tmp_path, workers=1, n=13)
    runner.__class__ = ConcurrentProbeRunner
    runner.workers = workers
    runner.scheduler = "episode_pool"
    runner.skill_owners = 2
    runner.episode_concurrency = workers
    tasks = ["object_counting", "object_abs_distance", "object_size_estimation"]
    variants = []
    labels = {}
    for task in tasks:
        rows = [{
            **row, "variant_id": row["variant_id"] + "-" + task,
            "qa_id": row["qa_id"] + "-" + task,
            "question_type": task, "wave_index": index, "split": "induction",
        } for index, row in enumerate(runner.variants)]
        variants.extend(rows)
        runner.gate_panels[task + "/g0m0"] = [row["variant_id"] for row in rows[:8]]
        labels.update({row["qa_id"]: {
            "question_type": task, "scene_id": row["scene_id"],
            "options": [], "reference_answer": "2",
        } for row in rows})
    runner.variants = variants
    runner.execution_variants = sorted(variants, key=lambda row: (
        row["scene_id"], row["variant_id"]))
    runner.label_metadata = labels
    return runner


@pytest.mark.parametrize("workers", [16, 32])
def test_pool_reaches_concurrency_and_preserves_three_lineage_boundaries(
        tmp_path, monkeypatch, workers):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    runner = pool_runner(tmp_path, workers)
    result = runner.run(resume=False)
    assert result["completed_variants"] == 39
    events = []
    intervals = [json.loads(p.read_text()) for p in (runner.root / "intervals").glob("*.json")]
    for row in intervals:
        events.extend([(row["started"], 1), (row["ended"], -1)])
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    assert peak == workers
    assert len({row["pid"] for row in intervals}) == workers
    for task in ["object_counting", "object_abs_distance", "object_size_estimation"]:
        rows = sorted([row for row in runner.completed().values()
                       if row["question_type"] == task], key=lambda row: row["qa_id"])
        assert all(row["initial_skill_selection"]["skill_version"].endswith("@1.0.0")
                   for row in rows[:12])
        assert rows[12]["initial_skill_selection"]["skill_version"].endswith("@1.1.0")
    bundle = freeze_fitset_stream(runner.root, tmp_path / "bundle")
    assert bundle["revised_slot_count"] == 3
    assert pool_runner(tmp_path, workers).run(resume=True)["status"] == "complete"
    with pytest.raises(FitSetError, match="manifest changed"):
        pool_runner(tmp_path, 16 if workers == 32 else 32).run(resume=True)


class InterruptedPoolRunner(ConcurrentProbeRunner):
    def _write_summary(self, manifest, completed):
        raise KeyboardInterrupt()


def test_pool_drains_interruption_then_recovers_once(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    runner = pool_runner(tmp_path, 16)
    runner.__class__ = InterruptedPoolRunner
    with pytest.raises(KeyboardInterrupt):
        runner.run(resume=False)
    before = runner.completed()
    assert before
    resumed = pool_runner(tmp_path, 16)
    assert resumed.run(resume=True)["status"] == "complete"
    assert all(resumed.completed()[key] == row for key, row in before.items())
    bundle = freeze_fitset_stream(runner.root, tmp_path / "bundle")
    assert bundle["revised_slot_count"] == 3


class FailedPoolRunner(ProbeRunner):
    def _run_one(self, variant):
        raise RuntimeError("fixture failure")


def test_failed_executor_stops_queued_work_and_preserves_failure_receipt(tmp_path):
    runner = pool_runner(tmp_path, 16)
    runner.__class__ = FailedPoolRunner
    with pytest.raises(FitSetError, match="failed"):
        runner.run(resume=False, max_variants=4)
    receipt = json.loads((runner.root / "episode_pool_receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert not runner.completed()
    assert list((runner.root / "scheduler_receipts").glob("*.json"))


def test_learning_selection_rejects_skipping_a_lineage_prefix(tmp_path):
    runner = pool_runner(tmp_path, 16)
    with pytest.raises(FitSetError, match="prefix"):
        runner.run(resume=False, variant_ids=[runner.variants[12]["variant_id"]])


def test_bounded_resume_can_fill_one_gap_before_buffered_results(tmp_path):
    import copy
    from skill3d.evolution.fitset_stream_v11 import FitSetWorkerEvolution

    runner = make_runner(tmp_path, n=4)
    runner.scheduler = "episode_pool"
    runner.skill_owners = 1
    runner.initialize(resume=False)
    owner = FitSetWorkerEvolution(runner, 0, runner._planned_slot_shards()[0])
    worker = copy.copy(runner)
    worker.skills_for_variant = owner.provider
    # Simulate two later episodes completing before the first two results.
    before = {row["variant_id"]: worker._run_one(row) for row in runner.variants[2:]}
    first = runner.run(resume=True, max_variants=1)
    assert first["completed_variants"] == 3
    stream_path = runner.root / "workers/00/buffer/stream.json"
    stream = json.loads(stream_path.read_text())
    assert stream["slots"]["object_counting/g0m0"]["new_since_attempt"] == 1
    assert runner.run(resume=True)["completed_variants"] == 4
    stream = json.loads(stream_path.read_text())
    assert stream["slots"]["object_counting/g0m0"]["new_since_attempt"] == 4
    assert all(runner.completed()[key] == row for key, row in before.items())


@dataclass
class ParallelRevisionClient:
    root: object

    def chat(self, messages, **kwargs):
        from test_fitset_stream_v11 import RevisionClient
        assert threading.current_thread() is threading.main_thread()
        marker = self.root / "reviser_calls" / f"{os.getpid()}.json"
        if not marker.exists():
            _write_create_once(marker, {"started": time.time(), "pid": os.getpid()})
        deadline = time.monotonic() + 40
        while len(list(marker.parent.glob("*.json"))) < 2:
            assert time.monotonic() < deadline, "independent revisers never overlapped"
            time.sleep(0.05)
        return RevisionClient().chat(messages, **kwargs)


@dataclass
class ParallelRevisionFactory:
    root: object

    def __call__(self):
        return ParallelRevisionClient(self.root)


def test_parallel_owners_overlap_and_preserve_publication_and_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    runner = pool_runner(tmp_path, 16)
    runner.reviser_concurrency = 2
    runner._global_interval_root = runner.root / "global_episode_intervals"
    runner.revision_client_factory = ParallelRevisionFactory(runner.root)
    assert runner.run(resume=False)["completed_variants"] == 39
    assert len(list((runner.root / "reviser_calls").glob("*.json"))) == 2
    receipt = json.loads((runner.root / "episode_pool_receipt.json").read_text())
    assert receipt["peak_active_reviser_owners"] == 2
    assert receipt["reviser_executor_count"] == 2
    intervals = [json.loads(path.read_text()) for path in runner._global_interval_root.glob("*.json")]
    assert len(intervals) == 39 + 3 * 8 * 2 * 2
    events = sorted(
        [(row["started"], 1) for row in intervals] + [(row["ended"], -1) for row in intervals])
    active_episodes = peak_episodes = 0
    for _, delta in events:
        active_episodes += delta
        peak_episodes = max(peak_episodes, active_episodes)
    assert peak_episodes <= 16
    active = set()
    for event in receipt["events"]:
        if event["kind"] == "reviser_started":
            assert event["owner"] not in active
            active.add(event["owner"])
        elif event["kind"] == "reviser_done":
            active.remove(event["owner"])
    assert not active
    for task in ["object_counting", "object_abs_distance", "object_size_estimation"]:
        rows = sorted([row for row in runner.completed().values()
                       if row["question_type"] == task], key=lambda row: row["qa_id"])
        assert all(row["initial_skill_selection"]["skill_version"].endswith("@1.0.0")
                   for row in rows[:12])
        assert rows[12]["initial_skill_selection"]["skill_version"].endswith("@1.1.0")
    before = runner.completed()
    resumed = pool_runner(tmp_path, 16)
    resumed.reviser_concurrency = 2
    resumed.revision_client_factory = ParallelRevisionFactory(resumed.root)
    assert resumed.run(resume=True)["status"] == "complete"
    assert resumed.completed() == before
    assert freeze_fitset_stream(runner.root, tmp_path / "parallel-bundle")["revised_slot_count"] == 3


def test_parallel_reviser_failure_drains_and_keeps_completed_results(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    runner = pool_runner(tmp_path, 16)
    runner.reviser_concurrency = 2
    runner.__class__ = InterruptedPoolRunner
    with pytest.raises(KeyboardInterrupt):
        runner.run(resume=False)
    before = runner.completed()
    resumed = pool_runner(tmp_path, 16)
    resumed.reviser_concurrency = 2
    assert resumed.run(resume=True)["status"] == "complete"
    assert all(resumed.completed()[key] == row for key, row in before.items())


class FailingRevisionClient:
    def chat(self, *args, **kwargs):
        raise RuntimeError("synthetic provider body must stay private")


def failing_revision_factory():
    return FailingRevisionClient()


def test_owner_service_error_stops_dispatch_and_retains_immutable_results(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    runner = pool_runner(tmp_path, 16)
    runner.reviser_concurrency = 2
    runner.revision_client_factory = failing_revision_factory
    with pytest.raises(FitSetError, match="reviser owner"):
        runner.run(resume=False)
    assert runner.completed()
    receipt = json.loads((runner.root / "episode_pool_receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert any(row["kind"] == "reviser_error" for row in receipt["events"])
    assert "provider body" not in json.dumps(receipt)
    assert not any(row["initial_skill_selection"]["skill_version"].endswith("@1.1.0")
                   for row in runner.completed().values())


def test_busy_owner_blocks_only_transaction_slots_after_recovery():
    from types import SimpleNamespace
    from skill3d.evaluation.reviser_pool_v11 import OwnerReviserPool
    pool = OwnerReviserPool.__new__(OwnerReviserPool)
    pool.runner = SimpleNamespace(overlap_owner_slots=True)
    row = {"question_type": "object_counting", "evidence_state": "g0m0"}
    pool.busy = {0: ("ingest", [(row, {})])}
    pool.recovered = set()
    assert pool.blocks_slot(0, "object_size_estimation/g0m0")
    pool.recovered.add(0)
    assert pool.blocks_slot(0, "object_counting/g0m0")
    assert not pool.blocks_slot(0, "object_size_estimation/g0m0")
    pool.runner.overlap_owner_slots = False
    assert pool.blocks_slot(0, "object_size_estimation/g0m0")
    pool.runner.overlap_owner_slots = True
    pool.busy[0] = ("recover", [])
    assert pool.blocks_slot(0, "object_size_estimation/g0m0")


class OverlapOwnerProbeRunner(ProbeRunner):
    def _run_one(self, variant):
        if (getattr(self, "_request_category", "solver") == "solver"
                and variant["question_type"] == "object_size_estimation"
                and variant["wave_index"] == 0):
            deadline = time.monotonic() + 40
            marker = self.root / "owner_overlap_started.json"
            while not marker.exists():
                assert time.monotonic() < deadline, "first owner revision never began"
                time.sleep(0.03)
        return super()._run_one(variant)


@dataclass
class OverlapOwnerClient:
    root: object
    def chat(self, messages, **kwargs):
        from test_fitset_stream_v11 import RevisionClient
        if "object_abs_distance" in messages[-1]["content"]:
            marker = self.root / "owner_overlap_started.json"
            if not marker.exists():
                _write_create_once(marker, {"started": time.time()})
            deadline = time.monotonic() + 40
            while True:
                rows = [json.loads(p.read_text()) for p in (self.root / "results").glob("*.json")]
                count = sum(row["question_type"] == "object_size_estimation" for row in rows)
                if count >= 12:
                    break
                assert time.monotonic() < deadline, "unrelated frozen window did not progress"
                time.sleep(0.03)
        return RevisionClient().chat(messages, **kwargs)


@dataclass
class OverlapOwnerFactory:
    root: object
    def __call__(self):
        return OverlapOwnerClient(self.root)


def test_unrelated_window_finishes_during_revision_without_crossing_publication(tmp_path):
    runner = pool_runner(tmp_path, 2)
    runner.__class__ = OverlapOwnerProbeRunner
    runner.reviser_concurrency = 2
    runner.overlap_owner_slots = True
    runner.revision_client_factory = OverlapOwnerFactory(runner.root)
    assert runner.run(resume=False)["completed_variants"] == 39
    receipt = json.loads((runner.root / "episode_pool_receipt.json").read_text())
    active, overlapped, ingestion = set(), False, {}
    ownership = json.loads((runner.root / "manifest.json").read_text())["execution"]["slot_ownership"]
    for event in receipt["events"]:
        if event["kind"] == "reviser_started":
            active.add(event["owner"])
        elif event["kind"] == "reviser_done":
            active.remove(event["owner"])
        elif event["kind"] == "dispatched":
            key = event["slot_key"]
            if key == "object_size_estimation/g0m0" and ownership[key] in active:
                overlapped = True
        elif event["kind"] == "ingested":
            ingestion.setdefault(event["slot_key"], []).append(event["variant_id"])
    assert overlapped
    for task in ["object_counting", "object_abs_distance", "object_size_estimation"]:
        rows = sorted([row for row in runner.completed().values()
                       if row["question_type"] == task], key=lambda row: row["qa_id"])
        assert all(row["initial_skill_selection"]["skill_version"].endswith("@1.0.0")
                   for row in rows[:12])
        assert rows[12]["initial_skill_selection"]["skill_version"].endswith("@1.1.0")
        assert ingestion[task + "/g0m0"] == [row["variant_id"] for row in rows]
