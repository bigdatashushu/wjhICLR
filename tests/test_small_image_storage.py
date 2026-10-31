import json

import numpy as np
import pytest
from PIL import Image

from skill3d.small_image.datasets import SmallImageProblem
from skill3d.small_image.precompute import SmallImagePrecomputer
from skill3d.small_image.storage import load_prepared, save_prepared


def test_persisted_geometry_replays_without_backend_and_rejects_changed_bytes(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.zeros((9, 11, 3), dtype=np.uint8)).save(path)
    problem = SmallImageProblem("cv3d", "example", "train", "Which is closer?",
                                ("red", "blue"), (str(path),), "Depth")
    depth = np.arange(99, dtype=float).reshape(9, 11) + 1
    depth[0, 0] = np.nan
    computer = SmallImagePrecomputer(
        depth_estimator=lambda rgb: {"depth": depth, "valid_mask": np.isfinite(depth)},
        model_identities={"depth_estimator": {"model": "fixture"}})
    prepared = computer.process(problem, categories=())
    directory = tmp_path / "prepared"
    save_prepared(prepared, directory)
    replay = load_prepared(directory, problem=problem)
    assert replay.artifact_sha256 == prepared.artifact_sha256
    np.testing.assert_equal(replay.depths[0], prepared.depths[0])
    with pytest.raises(ValueError):
        replay.depths[0].setflags(write=True)
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["input_sha256"] = "0" * 64
    (directory / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_prepared(directory)
    save_bytes = (directory / "arrays.npz").read_bytes()
    (directory / "arrays.npz").write_bytes(save_bytes[:-1] + bytes([save_bytes[-1] ^ 1]))
    with pytest.raises(ValueError, match="array file digest mismatch"):
        load_prepared(directory)
