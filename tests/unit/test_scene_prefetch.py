import threading
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path

import pytest

from skill3d.storage.hot_cache import IntegrityError
from skill3d.storage.scene_prefetch import ScenePrefetch


def test_one_eligible_lookahead_never_launches_an_episode_or_duplicate_transfer():
    entered, release = threading.Event(), threading.Event()
    calls, events = [], []
    def prefetch(scene):
        calls.append(scene)
        entered.set()
        assert release.wait(5)
        return {"scene": scene, "status": "ready"}
    manager = ScenePrefetch(SimpleNamespace(prefetch=prefetch), events)
    row = {"variant_id": "eligible-window-qa", "scene_id": "next"}
    try:
        manager.consider(row, active_scenes={"current"}, completed_count=0)
        assert entered.wait(2)
        manager.consider({**row, "scene_id": "later"}, active_scenes=set(), completed_count=0)
        assert calls == ["next"]
    finally:
        release.set()
        manager.close()
    assert events[-1]["status"] == "ready"
    assert events[0]["variant_id"] == row["variant_id"]


def test_quota_defers_without_busy_retry_and_integrity_failure_is_observed():
    events = []
    manager = ScenePrefetch(SimpleNamespace(prefetch=lambda scene: {
        "scene": scene, "status": "deferred_storage_quota"}), events)
    row = {"variant_id": "eligible", "scene_id": "next"}
    manager.consider(row, active_scenes=set(), completed_count=0)
    manager.future.result(timeout=2)
    manager.consider(row, active_scenes=set(), completed_count=0)
    assert len([event for event in events if event["kind"] == "prefetch_started"]) == 1
    manager.close()
    def fail(_):
        raise IntegrityError("synthetic")
    manager = ScenePrefetch(SimpleNamespace(prefetch=fail), events)
    manager.consider(row, active_scenes=set(), completed_count=1)
    with pytest.raises(IntegrityError):
        manager.close()
    assert events[-1] == {"kind": "prefetch_failed", "scene": "next",
                          "error_type": "IntegrityError"}


@dataclass
class TraceRuntime:
    root: Path

    def prefetch(self, scene):
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / (scene + ".txt")).write_text("public input bytes only")
        return {"scene": scene, "status": "ready"}


def test_real_spawned_scheduler_prefetch_preserves_windows_versions_and_ingest_order(tmp_path):
    import json
    from test_fitset_stream_v11 import make_runner
    results, sequences = [], []
    for enabled in (False, True):
        runner = make_runner(tmp_path / str(enabled), workers=2)
        runner.scheduler = "episode_pool"
        runner.skill_owners = 1
        runner.prefetch_scenes = enabled
        runner.published_inputs = SimpleNamespace(runtime=TraceRuntime(
            tmp_path / "prefetch-input-bytes"))
        runner.binding_refs = {row["variant_id"]: {"request_identity": {
            "seed": runner.base_cfg.seed}} for row in runner.variants}
        runner.run(resume=False)
        results.append([(row["variant_id"], row["answer"],
                         row["initial_skill_selection"]["content_sha256"],
                         row["initial_skill_selection"]["skill_version"])
                        for row in sorted(runner.completed().values(),
                                          key=lambda row: row["variant_id"])])
        receipt = json.loads((runner.root / "episode_pool_receipt.json").read_bytes())
        sequences.append([event["variant_id"] for event in receipt["events"]
                          if event["kind"] == "ingested"])
        if enabled:
            starts = [event for event in receipt["events"] if event["kind"] == "prefetch_started"]
            assert starts
            for event in starts:
                position = receipt["events"].index(event)
                dispatched = [value["variant_id"] for value in receipt["events"][:position]
                              if value["kind"] == "dispatched"]
                assert event["variant_id"] not in dispatched
                # Every lookahead is in the first frozen 12-case window.
                assert int(event["variant_id"].split("@")[0].split("-")[1]) < 12
    assert results[0] == results[1]
    assert sequences[0] == sequences[1]
    assert all(row[-1].endswith("@1.0.0") for row in results[1][:12])
    assert results[1][12][-1].endswith("@1.1.0")
