"""Overview identity validation uses synthetic PNGs and no model services."""
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from skill3d.online.scene_overview import SceneOverviewStore, read_overview_image


def sha(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def overview_fixture(tmp_path):
    root = tmp_path / "overviews"
    root.mkdir()
    pixels = np.full((480, 640, 3), [34, 89, 144], np.uint8)
    Image.fromarray(pixels).save(root / "scene-a.png")
    spec = {"resolution_wh": [640, 480], "renderer_version": "synthetic/1"}
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"quality":{"main_gate_passed":true}}')
    row = {
        "scene_id": "scene-a", "file": "scene-a.png", "n_frozen_frames": 32,
        "png_sha256": sha((root / "scene-a.png").read_bytes()),
        "rgb_sha256": sha(pixels.tobytes()), "scene_artifact": str(artifact),
        "scene_artifact_sha256": sha(artifact.read_bytes()), "frozen_scene_sha256": "b" * 64,
        "world_frame_status": "available", "resolution_wh": [640, 480],
        "selected_azimuth_degrees": 90, "elevation_degrees": 52, "render_only_cutaway": True,
    }
    manifest = {
        "schema": "harness3d-vsi1000-scene-overviews/1", "status": "complete",
        "render_spec": spec,
        "render_spec_sha256": sha(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()),
        "n_scenes": 1, "images": [row],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    entry = {"scene_id": "scene-a", "scene_artifact": str(artifact),
             "frozen_scene_sha256": "b" * 64}
    return path, manifest, entry, pixels


def test_exact_pixels_and_immutable_cached_array(overview_fixture):
    path, _, entry, pixels = overview_fixture
    store = SceneOverviewStore(path)
    descriptor = store.descriptor("scene-a", entry=entry)
    loaded = read_overview_image(descriptor, scene_id="scene-a")
    assert np.array_equal(loaded, pixels)
    with pytest.raises(ValueError):
        loaded.setflags(write=True)
    assert store.validate_entries({"q1": entry, "q2": entry})["n_questions"] == 2
    assert "image_path" not in store.identity


@pytest.mark.parametrize("changed", ["png", "manifest", "artifact", "entry", "scene"])
def test_changes_fail_even_after_cache_warmup(overview_fixture, changed):
    path, _, entry, _ = overview_fixture
    store = SceneOverviewStore(path)
    store.descriptor("scene-a", entry=entry)
    if changed == "png":
        with (path.parent / "scene-a.png").open("ab") as stream:
            stream.write(b"tampered")
    elif changed == "manifest":
        path.write_bytes(path.read_bytes() + b" ")
    elif changed == "artifact":
        from pathlib import Path
        Path(entry["scene_artifact"]).write_text("changed")
    elif changed == "entry":
        entry["frozen_scene_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        store.descriptor("scene-b" if changed == "scene" else "scene-a", entry=entry)


def test_manifest_pin_and_exact_scene_scope(overview_fixture):
    path, _, entry, _ = overview_fixture
    with pytest.raises(ValueError, match="hash"):
        SceneOverviewStore(path, "0" * 64)
    store = SceneOverviewStore(path)
    with pytest.raises(ValueError, match="scene set"):
        store.validate_entries({"q1": entry, "q2": {**entry, "scene_id": "scene-b"}})


def test_path_escape_and_symlink_rejected(overview_fixture):
    path, manifest, _, _ = overview_fixture
    manifest["images"][0]["file"] = "../scene-a.png"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filename"):
        SceneOverviewStore(path)
    link = path.parent / "linked.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symbolic"):
        SceneOverviewStore(link)
