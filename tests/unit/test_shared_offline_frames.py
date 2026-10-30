import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.evaluation.fitset_v11 import SystemVariantRunner, _variant_key, FitSetError
from skill3d.storage.hot_cache import IntegrityError
from skill3d.storage.offline_frames import publish_frame, shared_frame_path


def capture(root, policy):
    root.mkdir()
    runner = object.__new__(SystemVariantRunner)
    runner.root = root
    runner.framework_state_io = None
    runner._continuation = None
    runner._cached_prefix_spec = None
    runner.published_inputs = None
    runner.offline_frame_storage = policy
    return runner


def inputs():
    variant = {"variant_id": "qa-one", "qa_id": "one", "scene_id": "scene-A",
               "split": "induction"}
    frames = SimpleNamespace(frame_ids=list(range(32)),
                             source_frame_indices=list(range(0, 64, 2)),
                             frame_set_hash="b" * 64)
    pixels = [np.full((4, 6, 3), index, dtype=np.uint8) for index in range(32)]
    item = SimpleNamespace(episode=SimpleNamespace(frame_set=frames,
                            question="Synthetic question.", options=["A", "B"]), pixels=pixels)
    return variant, item


def test_two_qa_shared_input_preserves_every_original_png_and_provenance(tmp_path):
    legacy = capture(tmp_path / "legacy", "per_variant_v1")
    shared = capture(tmp_path / "shared", "shared_scene_v2")
    variant, item = inputs()
    legacy._freeze_offline_case_input(variant, item)
    shared._freeze_offline_case_input(variant, item)
    other = {**variant, "variant_id": "qa-two", "qa_id": "two"}
    shared._freeze_offline_case_input(other, item)
    old, new, second = (runner.offline_case_input(row) for runner, row in
                        ((legacy, variant), (shared, variant), (shared, other)))
    assert len(list((shared.root / "variants" / "shared_frames").rglob("*.png"))) == 32
    assert not list((shared.root / "variants").glob("*/offline_images"))
    for first, ref, another in zip(old["image_refs"], new["image_refs"], second["image_refs"]):
        assert {k: v for k, v in first.items() if k != "path"} == {
            k: v for k, v in ref.items() if k != "path"}
        assert ref["path"] == another["path"]
        assert first["image_id"] != another["image_id"]
        assert Path(first["path"]).read_bytes() == Path(ref["path"]).read_bytes()
        assert Path(ref["path"]).stat().st_nlink == 1
    # Legacy capture remains readable by the same reader.
    assert old["schema_version"].endswith("/1.0")
    assert new["schema_version"].endswith("/2.0")


def test_shared_publication_races_are_atomic_and_existing_tamper_is_refused(tmp_path):
    data = b"frozen public bytes"
    path = shared_frame_path(tmp_path, "scene", "a" * 64, hashlib.sha256(data).hexdigest())
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: publish_frame(tmp_path, path, data), range(16)))
    assert path.read_bytes() == data
    assert not list(path.parent.glob(".frame-*"))
    path.write_bytes(b"tampered")
    with pytest.raises(IntegrityError):
        publish_frame(tmp_path, path, data)


def test_shared_ref_cannot_point_to_another_scene_or_symlink(tmp_path):
    runner = capture(tmp_path / "run", "shared_scene_v2")
    variant, item = inputs()
    runner._freeze_offline_case_input(variant, item)
    frozen = runner.offline_case_input(variant)
    source = Path(frozen["image_refs"][0]["path"])
    outside = tmp_path / "outside.png"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(FitSetError):
        runner.offline_case_input(variant)
    source.unlink()
    source.write_bytes(outside.read_bytes())
    manifest = runner.root / "variants" / _variant_key(variant["variant_id"]) / "offline_input.json"
    frozen["image_refs"][0]["path"] = str(shared_frame_path(
        runner.root, "wrong-scene", item.episode.frame_set.frame_set_hash,
        frozen["image_refs"][0]["file_sha256"]))
    manifest.write_text(json.dumps(frozen))
    with pytest.raises(FitSetError):
        runner.offline_case_input(variant)


def test_new_output_refs_do_not_need_sealed_input_transport_capabilities(tmp_path):
    runner = capture(tmp_path / "run", "shared_scene_v2")
    def input_only(_):
        raise AssertionError("new outputs are not sealed input members")
    runner.published_inputs = SimpleNamespace(logical_path=input_only, logical_file=input_only)
    variant, item = inputs()
    runner._freeze_offline_case_input(variant, item)
    assert len(runner.offline_case_input(variant)["image_refs"]) == 32
