"""Frozen-panel loading rejects label and pixel changes before any GPU call."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from skill3d.adapters.frame_set import build_frame_set, with_pixel_identity
from skill3d.evaluation import skill_ablation_v11 as ab
from skill3d.online.synthetic import make_synthetic_episode

spec = importlib.util.spec_from_file_location(
    "metric_validation", Path(__file__).resolve().parents[2] / "scripts/run_metric_validation_v11.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_frozen_input_validates_hashes_and_never_accepts_labels(tmp_path):
    pixels = [np.full((12, 16, 3), 80, np.uint8)]
    fs = with_pixel_identity(build_frame_set(
        1, scene_name="fixture", episode_id="1", dataset_id="scannet"), pixels)
    ep = make_synthetic_episode("room_size_estimation").episode.model_copy(update={
        "qa_id": "1", "scene_name": "fixture", "dataset": "scannet",
        "split": "inner_validation", "ground_truth": "", "frame_set": fs})
    image = tmp_path / "frame.npy"
    np.save(image, pixels[0])
    value = {"episode": ep.model_dump(mode="json"), "input_error": None, "source": "vsi_bench",
             "video_path": "/frozen/video.mp4", "frame_files": [str(image)],
             "frame_file_hashes": {image.name: ab._file_hash(image)},
             "pixels": ab._pixels_identity(pixels), "cache_identity": fs.cache_identity()}
    path = tmp_path / "input.json"

    def save():
        value.pop("sha256", None)
        value["sha256"] = ab._hash(value)
        ab._write(path, value)
        ab._write(tmp_path / "input_manifest.json",
                  {"1": {"path": str(path), "sha256": ab._file_hash(path)}})

    row = {"qa_id": "1", "scene_name": "fixture", "task": "room_size_estimation"}
    save()
    assert validation.load_input(tmp_path, row)[0].episode.ground_truth == ""
    value["episode"]["ground_truth"] = "12.5"
    save()
    with pytest.raises(ValueError, match="label"):
        validation.load_input(tmp_path, row)
    value["episode"]["ground_truth"] = ""
    save()
    np.save(image, pixels[0] + 1)
    with pytest.raises(ValueError, match="frame file changed"):
        validation.load_input(tmp_path, row)
