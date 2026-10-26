"""Cache correctness regressions using small real arrays and files."""

import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.infra import m5_identity as identity
from skill3d.segmentation import sam2_tracker as st


def _stat_from_stamp(stamp):
    return SimpleNamespace(st_dev=stamp[0], st_ino=stamp[1], st_size=stamp[2],
                           st_mtime_ns=stamp[3], st_ctime_ns=stamp[4])


def _mutate_after_hash_block(monkeypatch, mutate):
    original_sha256 = identity.hashlib.sha256

    def sha256():
        digest = original_sha256()
        mutated = False

        def update(block):
            nonlocal mutated
            digest.update(block)
            if not mutated:
                mutated = True
                mutate()

        return SimpleNamespace(update=update, hexdigest=digest.hexdigest)

    monkeypatch.setattr(identity.hashlib, "sha256", sha256)


def _inputs():
    return {
        "scene_name": "scene", "frame_set_hash": "frozen",
        "frames": [np.zeros((3, 4, 3), dtype=np.uint8)],
        "depth_maps": np.ones((1, 3, 4)),
        "c2w_list": np.eye(4)[None], "intrinsics": np.eye(3),
        "point_conf": np.ones((1, 3, 4)), "depth_conf": np.ones((1, 3, 4)),
        "frame_provenance": [{"frame_slot": 0, "source_frame_index": 27}],
        "grid_transform": None, "model": {"weights": "a" * 64, "seed": 137},
        "sam2": {"checkpoint": "b" * 64, "config": {"fill_hole_area": 8}},
        "detector": {"weights": "c" * 64},
        "sources": {"coords.py": {"sha256": "d" * 64}},
    }


@pytest.mark.parametrize("name", [
    "depth_maps", "c2w_list", "intrinsics", "point_conf", "depth_conf", "frames",
    "model", "sam2", "detector", "sources", "frame_provenance", "grid_transform",
])
def test_same_shape_content_changes_invalidate_identity(name):
    inputs = _inputs()
    original = identity.generation_identity(**inputs)["sha256"]
    changed = copy.deepcopy(inputs)
    if isinstance(changed[name], np.ndarray):
        changed[name].flat[0] += .25
    elif name == "frames":
        changed[name][0].flat[0] += 1
    elif name == "frame_provenance":
        changed[name][0]["source_frame_index"] += 1
    else:
        changed[name] = {"changed": True}
    assert identity.generation_identity(**changed)["sha256"] != original


@pytest.mark.parametrize("shape", [(3, 4), (0, 4), (), (1,)])
def test_streaming_hash_matches_logical_bytes(shape):
    array = np.arange(np.prod(shape, dtype=int), dtype=np.int32).reshape(shape)
    got = identity.array_identity(array, block_bytes=7)
    assert got["sha256"] == hashlib.sha256(array.tobytes()).hexdigest()
    assert got["shape"] == list(shape)


def test_noncontiguous_array_hash_is_content_equivalent():
    array = np.arange(120, dtype=np.float64).reshape(10, 12)[::2, ::-2]
    assert not array.flags.c_contiguous
    got = identity.array_identity(array, block_bytes=24)
    assert got == identity.array_identity(np.ascontiguousarray(array), block_bytes=24)


def test_relocation_does_not_change_file_or_generation_identity(tmp_path):
    first = tmp_path / "one.pt"
    second = tmp_path / "elsewhere.pt"
    first.write_bytes(b"weights")
    second.write_bytes(first.read_bytes())
    assert identity.file_identity(first) == identity.file_identity(second)
    inputs = _inputs()
    inputs["sam2"] = {"checkpoint": identity.file_identity(first)}
    before = identity.generation_identity(**inputs)
    inputs["sam2"] = {"checkpoint": identity.file_identity(second)}
    assert identity.generation_identity(**inputs) == before


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("spelling", ["actual/asset", "alias", "link/../asset"],
                         ids=["plain", "symlink", "symlink-parent"])
def test_file_identity_preserves_absolute_spelling_and_os_path_semantics(
        tmp_path, monkeypatch, relative, spelling):
    nested = tmp_path / "actual" / "nested"
    nested.mkdir(parents=True)
    asset = nested.parent / "asset"
    asset.write_bytes(b"OS target")
    (tmp_path / "asset").write_bytes(b"wrong one")
    (tmp_path / "alias").symlink_to(asset)
    (tmp_path / "link").symlink_to(nested, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    expected_path = tmp_path / spelling
    path = spelling if relative else expected_path
    stamp = identity._file_stamp(asset)
    monkeypatch.setattr(identity.time, "time_ns", lambda: (
        max(stamp[3:]) + identity.FILE_HASH_SETTLE_NS + 1))
    original_open = Path.open
    opened_paths = []

    def record_open(open_path, *args, **kwargs):
        opened_paths.append(open_path)
        return original_open(open_path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)
    identity._file_digest.cache_clear()
    expected = {"size": 9, "sha256": hashlib.sha256(b"OS target").hexdigest()}
    assert identity.file_identity(path) == expected
    assert identity.file_identity(path) == expected
    assert opened_paths == [expected_path]
    assert identity._file_digest.cache_info().hits == 1


def test_symlink_retarget_invalidates_cached_digest(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"before")
    second.write_bytes(b"after!")
    previous_mtime = first.stat().st_mtime_ns
    identity.os.utime(second, ns=(previous_mtime, previous_mtime))
    path = tmp_path / "alias"
    path.symlink_to(first)
    first_stamp = identity._file_stamp(first)
    second_stamp = identity._file_stamp(second)
    assert first_stamp[2:4] == second_stamp[2:4]
    monkeypatch.setattr(identity.time, "time_ns", lambda: (
        max(first_stamp[3:] + second_stamp[3:]) + identity.FILE_HASH_SETTLE_NS + 1))
    identity._file_digest.cache_clear()
    before = identity.file_identity(path)
    assert identity.file_identity(path) == before
    path.unlink()
    path.symlink_to(second)
    after = identity.file_identity(path)
    assert after == {"size": 6, "sha256": hashlib.sha256(b"after!").hexdigest()}
    assert after != before
    assert identity.file_identity(path) == after
    assert identity._file_stamp(first) == first_stamp
    assert identity._file_stamp(second) == second_stamp
    assert identity._file_digest.cache_info().hits == 2
    assert identity._file_digest.cache_info().misses == 2


def test_file_hash_cache_reuses_reads_and_detects_same_size_edit(tmp_path, monkeypatch):
    path = tmp_path / "weights.bin"
    path.write_bytes(b"original")
    # Advancing a mocked clock does not advance the filesystem's ctime tick.
    identity.time.sleep(identity.FILE_HASH_SETTLE_NS / 1_000_000_000 + 0.01)
    future = identity.time.time_ns() + 3_000_000_000
    monkeypatch.setattr(identity.time, "time_ns", lambda: future)
    identity._file_digest.cache_clear()
    first = identity.file_identity(path)
    assert identity.file_identity(path) == first
    assert identity._file_digest.cache_info().hits == 1
    previous_mtime = path.stat().st_mtime_ns
    previous_ctime = path.stat().st_ctime_ns
    path.write_bytes(b"modified")
    identity.os.utime(path, ns=(previous_mtime, previous_mtime))
    assert path.stat().st_ctime_ns != previous_ctime
    future = path.stat().st_ctime_ns + 3_000_000_000
    changed = identity.file_identity(path)
    assert changed != first
    assert changed["sha256"] == hashlib.sha256(b"modified").hexdigest()
    assert identity._file_digest.cache_info().misses == 2


@pytest.mark.parametrize("size", [0, 7, 8, 9])
@pytest.mark.parametrize("age_ns", [0, identity.FILE_HASH_SETTLE_NS,
                                  identity.FILE_HASH_SETTLE_NS + 1])
def test_file_hash_reuse_requires_settling_regardless_of_size(
        tmp_path, monkeypatch, size, age_ns):
    path = tmp_path / "small"
    path.write_bytes(b"x" * size)
    monkeypatch.setattr(identity, "HASH_BLOCK_BYTES", 8)
    stamp = identity._file_stamp(path)
    monkeypatch.setattr(identity.time, "time_ns", lambda: max(stamp[3:]) + age_ns)
    identity._file_digest.cache_clear()
    first = identity.file_identity(path)
    assert first == {"size": size, "sha256": hashlib.sha256(b"x" * size).hexdigest()}
    assert identity.file_identity(path) == first
    info = identity._file_digest.cache_info()
    settled = age_ns > identity.FILE_HASH_SETTLE_NS
    assert info.hits == int(settled)
    assert info.currsize == int(settled)


def test_replaced_inode_invalidates_file_hash_cache(tmp_path, monkeypatch):
    path = tmp_path / "asset"
    path.write_bytes(b"before")
    monkeypatch.setattr(identity.time, "time_ns", lambda: (
        max(identity._file_stamp(path)[3:]) + identity.FILE_HASH_SETTLE_NS + 1))
    identity._file_digest.cache_clear()
    first = identity.file_identity(path)
    previous_mtime = path.stat().st_mtime_ns
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"after!")
    identity.os.utime(replacement, ns=(previous_mtime, previous_mtime))
    replacement.replace(path)
    assert identity.file_identity(path) != first
    assert identity._file_digest.cache_info().misses == 2


@pytest.mark.parametrize("stamp_index", range(5), ids=["device", "inode", "size", "mtime", "ctime"])
def test_every_file_stamp_component_is_part_of_hash_cache_key(
        tmp_path, monkeypatch, stamp_index):
    path = tmp_path / "asset"
    path.write_bytes(b"content")
    stamp = list(identity._file_stamp(path))
    monkeypatch.setattr(identity, "_file_stamp", lambda _: tuple(stamp))
    monkeypatch.setattr(identity.os, "fstat", lambda _: _stat_from_stamp(stamp))
    monkeypatch.setattr(identity.time, "time_ns", lambda: (
        max(stamp[3:]) + identity.FILE_HASH_SETTLE_NS + 1))
    identity._file_digest.cache_clear()
    identity.file_identity(path)
    identity.file_identity(path)
    stamp[stamp_index] += 1
    identity.file_identity(path)
    assert identity._file_digest.cache_info().hits == 1
    assert identity._file_digest.cache_info().misses == 2


@pytest.mark.parametrize("settled", [False, True])
@pytest.mark.parametrize("checkpoint", ["before", "after"])
@pytest.mark.parametrize("stamp_index", range(5),
                         ids=["device", "inode", "size", "mtime", "ctime"])
def test_every_open_fd_stamp_component_is_checked_before_and_after_hashing(
        tmp_path, monkeypatch, settled, checkpoint, stamp_index):
    path = tmp_path / "asset"
    path.write_bytes(b"content")
    stamp = identity._file_stamp(path)
    changed = list(stamp)
    changed[stamp_index] += 1
    original_fstat = identity.os.fstat
    observed = []

    def fstat(fd):
        stat = original_fstat(fd)
        assert identity._stat_stamp(stat) == stamp
        observed.append(fd)
        if checkpoint == "before" or len(observed) == 2:
            return _stat_from_stamp(changed)
        return stat

    monkeypatch.setattr(identity.os, "fstat", fstat)
    monkeypatch.setattr(identity.time, "time_ns", lambda: (
        max(stamp[3:]) + (identity.FILE_HASH_SETTLE_NS + 1 if settled else 0)))
    identity._file_digest.cache_clear()
    with pytest.raises(ValueError, match="changed during hashing"):
        identity.file_identity(path)
    assert len(observed) == (1 if checkpoint == "before" else 2)
    assert identity._file_stamp(path) == stamp
    assert identity._file_digest.cache_info().currsize == 0


@pytest.mark.parametrize("settled", [False, True])
def test_symlink_aba_opening_wrong_file_is_rejected_before_reading(
        tmp_path, monkeypatch, settled):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"content")
    second.write_bytes(b"WRONG!!")
    path = tmp_path / "alias"
    path.symlink_to(first)
    stamp = identity._file_stamp(path)
    identity.os.utime(second, ns=(stamp[3], stamp[3]))
    assert identity._file_stamp(second)[2:4] == stamp[2:4]
    monkeypatch.setattr(identity.time, "time_ns", lambda: (
        max(stamp[3:]) + (identity.FILE_HASH_SETTLE_NS + 1 if settled else 0)))
    original_open = Path.open
    opened_stamps = []
    reads = []

    @contextmanager
    def switched_open(open_path, *args, **kwargs):
        if open_path != path:
            with original_open(open_path, *args, **kwargs) as stream:
                yield stream
            return
        path.unlink()
        path.symlink_to(second)
        with original_open(open_path, *args, **kwargs) as stream:
            opened_stamps.append(identity._stat_stamp(identity.os.fstat(stream.fileno())))
            path.unlink()
            path.symlink_to(first)
            assert identity._file_stamp(path) == stamp

            def read(size):
                reads.append(size)
                return stream.read(size)

            yield SimpleNamespace(fileno=stream.fileno, read=read)

    identity._file_digest.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", switched_open)
        with pytest.raises(ValueError, match="changed during hashing"):
            identity.file_identity(path)
    assert opened_stamps == [identity._file_stamp(second)]
    assert opened_stamps[0] != stamp
    assert reads == []
    assert identity._file_stamp(path) == stamp
    assert identity._file_digest.cache_info().currsize == 0
    assert identity.file_identity(path) == {
        "size": 7, "sha256": hashlib.sha256(b"content").hexdigest()}
    assert identity._file_digest.cache_info().currsize == int(settled)


@pytest.mark.parametrize("settled", [False, True])
def test_real_open_file_mutation_during_hashing_is_rejected_with_stable_path_stamp(
        tmp_path, monkeypatch, settled):
    path = tmp_path / "asset"
    path.write_bytes(b"content")
    original_file_stamp = identity._file_stamp
    stamp = original_file_stamp(path)
    monkeypatch.setattr(identity.time, "time_ns", lambda: (
        max(original_file_stamp(path)[3:])
        + (identity.FILE_HASH_SETTLE_NS + 1 if settled else 0)))
    monkeypatch.setattr(identity, "HASH_BLOCK_BYTES", 3)
    mutations = []

    def mutate():
        with path.open("ab") as writer:
            writer.write(b"!")
        mutations.append(original_file_stamp(path))

    identity._file_digest.cache_clear()
    with monkeypatch.context() as patch:
        # Keep pathname observations stable to isolate the real descriptor check.
        patch.setattr(identity, "_file_stamp", lambda _: stamp)
        _mutate_after_hash_block(patch, mutate)
        with pytest.raises(ValueError, match="changed during hashing"):
            identity.file_identity(path)
    assert len(mutations) == 1
    assert mutations[0][2] == stamp[2] + 1
    assert identity._file_digest.cache_info().currsize == 0
    assert identity.file_identity(path) == {
        "size": 8, "sha256": hashlib.sha256(b"content!").hexdigest()}
    assert identity._file_digest.cache_info().currsize == int(settled)


@pytest.mark.parametrize("settled", [False, True])
def test_symlink_retarget_during_hashing_is_rejected_with_unchanged_open_fd(
        tmp_path, monkeypatch, settled):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"content")
    second.write_bytes(b"changed")
    path = tmp_path / "alias"
    path.symlink_to(first)
    stamp = identity._file_stamp(path)
    monkeypatch.setattr(identity.time, "time_ns", lambda: (
        max(identity._file_stamp(path)[3:])
        + (identity.FILE_HASH_SETTLE_NS + 1 if settled else 0)))
    monkeypatch.setattr(identity, "HASH_BLOCK_BYTES", 3)
    original_fstat = identity.os.fstat
    fd_stamps = []

    def fstat(fd):
        stat = original_fstat(fd)
        fd_stamps.append(identity._stat_stamp(stat))
        return stat

    def retarget():
        path.unlink()
        path.symlink_to(second)

    identity._file_digest.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(identity.os, "fstat", fstat)
        _mutate_after_hash_block(patch, retarget)
        with pytest.raises(ValueError, match="changed during hashing"):
            identity.file_identity(path)
    assert fd_stamps == [stamp, stamp]
    assert identity._file_stamp(path) != stamp
    assert identity._file_digest.cache_info().currsize == 0
    assert identity.file_identity(path) == {
        "size": 7, "sha256": hashlib.sha256(b"changed").hexdigest()}
    assert identity._file_digest.cache_info().currsize == int(settled)


@pytest.mark.parametrize("settled", [False, True])
def test_file_change_during_hashing_is_rejected_and_not_cached(
        tmp_path, monkeypatch, settled):
    path = tmp_path / "asset"
    path.write_bytes(b"content")
    stamp = identity._file_stamp(path)
    changed = (*stamp[:4], stamp[4] + 1)
    stamps = iter([stamp, changed])
    monkeypatch.setattr(identity, "_file_stamp", lambda _: next(stamps))
    monkeypatch.setattr(identity.time, "time_ns", lambda: (
        max(stamp[3:]) + (identity.FILE_HASH_SETTLE_NS + 1 if settled else 0)))
    identity._file_digest.cache_clear()
    with pytest.raises(ValueError, match="changed during hashing"):
        identity.file_identity(path)
    assert identity._file_digest.cache_info().currsize == 0


def test_file_change_during_cached_lookup_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "asset"
    path.write_bytes(b"content")
    stamp = identity._file_stamp(path)
    monkeypatch.setattr(identity.time, "time_ns", lambda: (
        max(stamp[3:]) + identity.FILE_HASH_SETTLE_NS + 1))
    identity._file_digest.cache_clear()
    identity.file_identity(path)
    stamps = iter([stamp, (*stamp[:4], stamp[4] + 1)])
    monkeypatch.setattr(identity, "_file_stamp", lambda _: next(stamps))
    with pytest.raises(ValueError, match="changed during lookup"):
        identity.file_identity(path)
    assert identity._file_digest.cache_info().hits == 1


@pytest.mark.parametrize("name", ["model", "sam2", "detector"])
def test_incomplete_model_identity_is_diagnostic_only(name):
    inputs = _inputs()
    inputs[name] = None
    generation = identity.generation_identity(**inputs)
    assert not generation["cache_eligible"]
    assert generation["missing"] == [{"model": "vlm"}.get(name, name)]


def test_detector_receipt_requires_matching_lifetime_port_and_files(tmp_path, monkeypatch):
    root = tmp_path / "service"
    root.mkdir()
    asset = root / "checkpoint"
    asset.write_bytes(b"weights")
    process = {"pid": 123, "start_ticks": "45", "boot_id": "boot"}
    monkeypatch.setattr(identity, "process_identity", lambda pid: process)
    monkeypatch.setattr(identity, "_listens_on", lambda pid, port: port == 20024)
    monkeypatch.setattr(identity.os, "readlink", lambda path: str(root))
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda path: (
        b"python\0scripts/serve_groundingdino.py\0" if str(path).endswith("/cmdline")
        else original_read(path)))
    receipt = {
        "process": process, "port": 20024, "status": "starting_service",
        "service_root": str(root), "generation_identity": {"weights": "a" * 64},
        "identity_files": [{"path": str(asset), "identity": identity.file_identity(asset)}],
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    assert identity.detector_identity("http://127.0.0.1:20024", path)
    assert identity.detector_identity("http://127.0.0.1:20022", path) is None
    asset.write_bytes(b"changed")
    assert identity.detector_identity("http://127.0.0.1:20024", path) is None


def test_final_scope_uses_only_verified_question_binding_denominators(tmp_path):
    from skill3d.online.runner import _m5_evidence_summary
    from skill3d.schemas import ObjectRecord
    from skill3d.segmentation.binding_validation import validate_binding

    final = [ObjectRecord(obj_id="old", category_name="chair", track_id="old")]
    for i in range(3):
        observation = {"observed_object": "chair", "alternate_object": "none",
                       "visibility": "clear", "mask_purity": "clean",
                       "mask_completeness": "whole"}
        request_identity = {"candidate_id": f"c{i}", "model": {"name": "synthetic"}}
        obj = st.bind_masks_to_world(
            [{0: np.ones((2, 2), bool)}], np.ones((2, 2, 2)),
            np.repeat(np.eye(4)[None], 2, axis=0), np.eye(3),
            class_hints=["chair"], out_dir=tmp_path / str(i),
            grounding_status="question_driven")[0].model_copy(update={
                "obj_id": f"q{i}", "track_id": f"q{i}", "candidate_ids": [f"c{i}"],
                "question_roles": [{"category": "chair", "role": "target",
                                    "source_text": "chair"}],
                "binding_provenance": [{
                    "candidate_id": f"c{i}", "category": "chair",
                    "semantic_provider": "bailian",
                    "semantic_policy": "visual-quality-and-relation",
                    "semantic_relation": "match",
                    "semantic_visual_request_sha256": identity.content_sha256(request_identity),
                    "semantic_relation_request_sha256": "b" * 64,
                    "semantic_observation": observation,
                    "semantic_observation_sha256": identity.content_sha256(observation),
                    "semantic_request_identity": request_identity,
                    "semantic_model_identity_sha256": identity.content_sha256(request_identity["model"]),
                }]})
        obj = obj.model_copy(update={
            "binding_validation": validate_binding(obj, generation_sha256="e" * 64)})
        final.append(obj)
    stats = {"track_consensus": {"track_fragmentation_ratio": .25,
                                 "duplicate_suspect_ratio": 0}}
    episode = SimpleNamespace(frame_set=SimpleNamespace(readable_frame_ids=[0, 1]))
    summary = _m5_evidence_summary(stats, final, episode)
    assert summary.n_objects == 3 and summary.n_tracks == 3
    assert summary.track_scope == "question_binding" and summary.base_inventory == {}
    assert summary.duplicate_suspect_ratio == 0
    assert summary.track_fragmentation_ratio == .25
    assert summary.question_binding["track_fragmentation_ratio"] == .25
    assert summary.question_inventory == {}
