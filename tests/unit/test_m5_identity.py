"""Cache correctness regressions using small real arrays and files."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.infra import m5_identity as identity
from skill3d.segmentation import sam2_tracker as st


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


def test_final_scope_keeps_base_gate_denominators():
    from skill3d.online.runner import _m5_evidence_summary
    from skill3d.schemas import ObjectRecord

    def obj(name, source, suspect):
        return ObjectRecord(obj_id=name, category_name="chair", track_id=name,
                            grounding_status=source, duplicate_suspect=suspect,
                            visible_frames=[0], mask_per_frame="", pointcloud_world="")
    base = [obj("base", "base_list", True)]
    final = base + [obj(f"q{i}", "question_targeted_fill", False) for i in range(3)]
    consensus = st.track_consensus_metrics(base, n_propagations=2, n_frames=2)
    stats = {"base_inventory": st.inventory_scope_statistics(
        base, n_frames=2, scope="base_inventory", n_propagations=2,
        n_dedup_groups=1, n_dedup_absorbed=1, consensus=consensus)}
    episode = SimpleNamespace(frame_set=SimpleNamespace(readable_frame_ids=[0, 1]))
    summary = _m5_evidence_summary(stats, final, episode)
    assert summary.n_objects == 4 and summary.n_tracks == 1
    assert summary.duplicate_suspect_ratio == 1
    assert summary.track_fragmentation_ratio == .5
    assert summary.question_inventory["duplicate_suspect_ratio"] == .25
    assert summary.question_inventory["track_fragmentation_ratio"] is None
