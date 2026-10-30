"""Exercise real scene-lock contention without model or dataset requests."""

from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import pickle
import threading

import pytest

from skill3d.evaluation.derived_published_fitset_v11 import (
    DerivedPublishedFitSetInputProvider,
)


def _provider():
    provider = DerivedPublishedFitSetInputProvider.__new__(
        DerivedPublishedFitSetInputProvider)
    provider.cache = OrderedDict()
    provider._scene_checks = {}
    provider.cache_lock = threading.RLock()
    provider._derived_scenes = {}
    provider._derivation_locks = {}
    return provider


def test_independent_scenes_overlap_and_shared_scene_publishes_once(monkeypatch):
    provider = _provider()
    arrived = threading.Barrier(2, timeout=5)
    counts = Counter()
    counts_lock = threading.Lock()

    def derive(self, entry):
        # A retained global cache lock would deadlock at this barrier.
        with self.cache_lock:
            with counts_lock:
                counts[entry["scene_id"]] += 1
        arrived.wait()
        return (entry["scene_id"], {}, {})

    monkeypatch.setattr(DerivedPublishedFitSetInputProvider, "_derive_scene", derive)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [
            pool.submit(provider._derived_scene, {"scene_id": name})
            for name in ("first", "second") * 8
        ]
        values = [future.result(timeout=10) for future in futures]
    assert counts == {"first": 1, "second": 1}
    assert all(value is provider._derived_scenes[value[0]] for value in values)


def test_failed_scene_releases_lock_and_never_caches_partial_replay(monkeypatch):
    provider = _provider()
    attempts = []

    def derive(self, entry):
        attempts.append(entry["scene_id"])
        if len(attempts) == 1:
            raise ValueError("replay rejected")
        return (entry["scene_id"], {}, {})

    monkeypatch.setattr(DerivedPublishedFitSetInputProvider, "_derive_scene", derive)
    with pytest.raises(ValueError, match="replay rejected"):
        provider._derived_scene({"scene_id": "first"})
    assert not provider._derived_scenes
    with ThreadPoolExecutor(max_workers=1) as pool:
        value = pool.submit(
            provider._derived_scene, {"scene_id": "first"}).result(timeout=5)
    assert value == ("first", {}, {}) and len(attempts) == 2


def test_pickle_recreates_empty_locks_and_leaves_parent_unchanged():
    provider = _provider()
    provider._derivation_locks["first"] = threading.Lock()
    provider._derived_scenes["first"] = ("cold-replay", {}, {})
    clone = pickle.loads(pickle.dumps(provider))
    assert not clone._derived_scenes and not clone._derivation_locks
    assert not clone.cache and clone.cache_lock is not provider.cache_lock
    assert set(provider._derivation_locks) == {"first"}
