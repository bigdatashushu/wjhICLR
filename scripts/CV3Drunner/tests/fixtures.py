from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from skill3d.small_image.datasets import SmallImageProblem
from skill3d.small_image.precompute import SmallImagePrecomputer
from skill3d.small_image.storage import save_prepared
from CV3D.data import PublicInput
from CV3Drunner.protocol import FrozenSkills

RECTS = {"red": (20, 20, 60, 58), "blue": (85, 15, 150, 58),
         "green": (25, 75, 110, 110)}


def rgb_image():
    image = Image.new("RGB", (180, 120), (60, 65, 70))
    draw = ImageDraw.Draw(image)
    for color, rect in RECTS.items():
        draw.rectangle(rect, fill=(100, 105, 110), outline={
            "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 255, 0)}[color], width=3)
    return image


def fixture(tmp_path, *, task="Depth", split="train", sample_id="fixture", image=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    image_path = tmp_path / "original.png"
    (image or rgb_image()).save(image_path)
    question = ("Which is closer to the camera, the red box or the blue box?" if task == "Depth"
                else "Is the red box closer to the blue box or the green box?")
    options = ("blue object", "red object") if task == "Depth" else ("blue object", "green object")
    problem = SmallImageProblem("cv3d", sample_id, split, question, options,
                               (str(image_path),), task)
    h, w = np.asarray(Image.open(image_path)).shape[:2]
    yy, xx = np.indices((h, w))
    z = 1 + xx / 100 + yy / 200
    computer = SmallImagePrecomputer(
        depth_estimator=lambda _: {
            "depth": z, "valid_mask": np.ones_like(z, dtype=bool),
            "points": np.stack([xx / 30, yy / 30, z], axis=-1)},
        model_identities={"depth_estimator": {"model": "offline-fixture"}})
    prepared = computer.process(problem, categories=())
    directory = tmp_path / "prepared"
    save_prepared(prepared, directory)
    text = question + "\n" + "\n".join(f"({label}) {value}" for label, value in zip("AB", options))
    return PublicInput(problem, text), directory, prepared


def frozen():
    return FrozenSkills(Path(__file__).resolve().parents[1] / "skills/frozen_skills.json")
