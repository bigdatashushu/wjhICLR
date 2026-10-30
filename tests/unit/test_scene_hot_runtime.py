import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import pytest

from skill3d.infra.m5_identity import content_sha256
from skill3d.storage.hot_cache import ArchiveReceiver, HotCache, IntegrityError, StorageError
from skill3d.storage.scene_runtime import SceneHotRuntime


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "hot"
    cache = HotCache(root)
    shards, files, archives = {}, {}, {}
    for scene in ("global", "state", "a", "b", "c"):
        data = (scene * 1024).encode()
        import hashlib
        identity = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        key = "scene-" + scene
        source = "/original/" + scene
        members = {"data.bin": {"source": source, "identity": identity}}
        shards[scene] = {
            "key": key, "members": members, "bytes": len(data),
            "allocation_bytes": len(data) + 65536,
            "sha256": content_sha256(members),
        }
        files[source] = {"path": str(cache._entry_path(key) / "data.bin"),
                         "identity": identity}
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as archive:
            info = tarfile.TarInfo("data.bin")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        archives[scene] = raw.getvalue()
    closure = tmp_path / "closure.json"
    closure.write_text(json.dumps({
        "schema_version": "h3d-scene-closure-v1", "shards": shards,
        "descriptor": {"schema_version": "published-physical-transport-v1",
                       "files": files, "roots": {}},
    }))
    result = SceneHotRuntime(
        closure_path=closure, cache_root=root, endpoint="http://unused",
        token_file=tmp_path / "no-token",
    )
    result._request = lambda path, *_args: io.BytesIO(archives[path.split("/")[-1]])
    return result


def test_scene_lease_survives_next_current_pin(runtime):
    cache = runtime.cache()
    with runtime.scene_lease("a") as path:
        assert path.is_dir()
        with runtime.scene_lease("b"):
            # Changing current must not release an earlier active episode.
            with cache._locked() as state:
                assert not cache._pinned(state, "scene-a")
                with pytest.raises(BlockingIOError):
                    from skill3d.storage.hot_cache import _flock
                    with _flock(cache._lease_path("scene-a"), blocking=False):
                        pass
        assert (path / "data.bin").is_file()
    assert cache.stats()["slots"]["global"] == "scene-global"
    assert cache.stats()["slots"]["state"] == "scene-state"


def test_duplicate_parallel_scene_downloads_publish_once(runtime):
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(r["status"] == "ready" for r in pool.map(runtime.prefetch, ["b", "b"]))
    assert runtime.cache().stats()["entries"].count("scene-b") == 1


def test_prefetch_really_overlaps_current_work(runtime):
    with runtime.scene_lease("a"):
        with runtime.overlap_next("a") as future:
            assert future.result()["scene"] == "b"
            assert runtime.cache().stats()["slots"]["next"] == "scene-b"
            # The episode can still consume its currently leased input.
            assert runtime.locator.file("/original/a").read_bytes() == b"a" * 1024


def test_corrupt_download_never_publishes(runtime):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        info = tarfile.TarInfo("data.bin")
        info.size = 1024
        archive.addfile(info, io.BytesIO(b"x" * 1024))
    runtime._request = lambda *_args: io.BytesIO(raw.getvalue())
    with pytest.raises(IntegrityError):
        runtime.prefetch("a")
    assert "scene-a" not in runtime.cache().stats()["entries"]
    assert runtime.cache().stats()["reservations"] == 0


def test_outputs_wait_for_ack_and_keep_latest_best(runtime, tmp_path):
    receiver = ArchiveReceiver(tmp_path / "durable")
    runtime.archive_transfer = receiver
    runtime.publish_output(
        "checkpoint-one", lambda r: r.write_bytes("payload/state.json", b"{}"),
        estimated_bytes=1024, job="job", shard="cp1", attempt=1,
        checkpoint=True, latest=True, best=True,
    )
    runtime.publish_output(
        "result-one", lambda r: r.write_bytes("payload/result.json", b"{}"),
        estimated_bytes=1024, job="job", shard="result1", attempt=1,
    )
    assert set(runtime.cache().stats()["entries"]) == {"checkpoint-one", "result-one"}
    acks = runtime.archive_pending()
    assert len(acks) == 2
    assert runtime.cache().stats()["entries"] == ["checkpoint-one"]


def test_wrong_layout_and_unknown_scene_fail_before_network(runtime):
    with pytest.raises(StorageError):
        runtime.prefetch("not-in-scope")
    value = json.loads(runtime.closure_path.read_bytes())
    value["descriptor"]["files"]["/original/a"]["path"] = "/wrong-layout/data.bin"
    runtime.closure_path.write_text(json.dumps(value))
    with pytest.raises(IntegrityError):
        SceneHotRuntime(
            closure_path=runtime.closure_path, cache_root=runtime.cache_root,
            endpoint="http://unused", token_file=Path("/unused"),
        )


def test_cold_transfer_does_not_block_another_hot_episode(runtime):
    import threading
    with runtime.scene_lease("b"):
        pass
    entered, release = threading.Event(), threading.Event()
    original = runtime._request
    calls = []
    def blocked(path, *args):
        calls.append(path)
        if path == "/scene/a":
            entered.set()
            assert release.wait(10)
        return original(path, *args)
    runtime._request = blocked
    def read_hot():
        with runtime.scene_lease("b") as path:
            return (path / "data.bin").read_bytes()
    with ThreadPoolExecutor(max_workers=3) as pool:
        cold = pool.submit(runtime.prefetch, "a")
        try:
            assert entered.wait(5)
            duplicate = pool.submit(runtime.prefetch, "a")
            assert pool.submit(read_hot).result(timeout=3) == b"b" * 1024
        finally:
            release.set()
        assert cold.result(timeout=5)["status"] == "ready"
        assert duplicate.result(timeout=5)["status"] == "ready"
    assert calls.count("/scene/a") == 1


def test_publish_to_reader_is_guarded_before_current_pin_moves(runtime):
    import threading
    cache = runtime.cache()
    acquired, release = threading.Event(), threading.Event()
    lease = cache.lease
    from contextlib import contextmanager
    @contextmanager
    def blocked_lease(key, **kwargs):
        if key == "scene-a":
            acquired.set()
            assert release.wait(10)
        with lease(key, **kwargs) as path:
            yield path
    cache.lease = blocked_lease
    def consume(scene):
        with runtime.scene_lease(scene) as path:
            return (path / "data.bin").read_bytes()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(consume, "a")
        try:
            assert acquired.wait(5)
            second = pool.submit(consume, "b")
            # B cannot reassign A's current pin before A's SH has been taken.
            with pytest.raises(TimeoutError):
                second.result(timeout=0.2)
            assert cache.stats()["slots"]["current"] == "scene-a"
        finally:
            release.set()
        assert first.result(timeout=5) == b"a" * 1024
        assert second.result(timeout=5) == b"b" * 1024
