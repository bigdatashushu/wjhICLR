"""Concurrency, version windows and interruption recovery for the episode pool."""

import json
import os
import threading
import time

import pytest

from skill3d.evaluation.fitset_v11 import _variant_key, _write_create_once, FitSetError
from skill3d.evolution.fitset_stream_v11 import freeze_fitset_stream
from test_fitset_stream_v11 import ProbeRunner, make_runner


class ConcurrentProbeRunner(ProbeRunner):
    def _run_one(self, variant):
        with self._episode_permit():
            return self._probe_one(variant)

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
