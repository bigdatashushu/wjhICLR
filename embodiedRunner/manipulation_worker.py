"""Credential-free official manipulation wrapper; no agent or scorer in public data."""
from __future__ import annotations

import base64
import importlib.metadata
import importlib.util
import io
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types

from manipulation_protocol import ACTION_SPEC, INPUT_SOURCE, SUBSETS, source, vector, verify_sources
from protocol import ContractError, OperationalError, ROOT, canonical, json_object, sha256

RUNTIME = ROOT / "manipulation-runtime"


def file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_catalog():
    catalog = json_object((ROOT / "manipulation_assets.json").read_bytes())
    if (catalog["schema"] != "embodied-manipulation-assets/1"
            or catalog["repository"] != "EmbodiedBench/EB-Manipulation"
            or catalog["reference_answer_visible"] is not False):
        raise ContractError("manipulation_asset_catalog")
    return catalog


def verify_assets():
    catalog = asset_catalog()
    for row in catalog["files"]:
        path = (RUNTIME / "dataset" / row["path"]).resolve()
        if (not path.is_relative_to((RUNTIME / "dataset").resolve())
                or path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]):
            raise ContractError("manipulation_asset_changed")
    return sha256(canonical(catalog))


def runtime():
    from action_bridge import v2_environment
    environment = v2_environment(ROOT / "tmp")
    environment.update(
        COPPELIASIM_ROOT=str(RUNTIME / "coppeliasim"),
        LD_LIBRARY_PATH=str(RUNTIME / "coppeliasim"),
        QT_QPA_PLATFORM_PLUGIN_PATH=str(RUNTIME / "coppeliasim"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe", CUDA_VISIBLE_DEVICES="",
        PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    os.environ.clear()
    os.environ.update(environment)
    main = types.ModuleType("embodiedbench.main")
    main.logger = logging.getLogger("embodiedrunner-manipulation")
    main.logger.disabled = True
    sys.modules["embodiedbench.main"] = main
    official = ROOT / "upstream-manipulation"
    sys.path[:0] = [str(official), str(official / "embodiedbench/envs/eb_manipulation")]


def probe():
    versions = {}
    for name in ("numpy", "scipy", "Pillow", "cffi", "gymnasium", "ultralytics"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    installed = {name: importlib.util.find_spec(name) is not None for name in ("pyrep", "amsolver")}
    receipt = json_object((RUNTIME / "runtime.json").read_bytes())
    if (receipt.get("schema") != "embodied-manipulation-runtime/1"
            or receipt.get("coppeliasim_version") != "4.1.0"
            or receipt.get("pyrep_revision") != "8f420be8064b1970aae18a9cfbc978dfb15747ef"
            or not receipt.get("pinned_files") or not receipt.get("packages")):
        raise ContractError("manipulation_runtime_receipt")
    source_hash, assets_hash = verify_sources(), verify_assets()
    for row in receipt["pinned_files"]:
        path = (RUNTIME / row["path"]).resolve()
        if (not path.is_relative_to(RUNTIME.resolve()) or path.stat().st_size != row["bytes"]
                or file_hash(path) != row["sha256"]):
            raise ContractError("manipulation_runtime_changed")
    installed_versions = {name: importlib.metadata.version(name) for name in receipt["packages"]}
    ready = (all(installed.values()) and versions["numpy"] == "1.26.4"
             and all(versions.values()) and source_hash == receipt["official_sources_sha256"]
             and assets_hash == receipt["assets_sha256"] and installed_versions == receipt["packages"])
    return {
        "versions": installed_versions, "installed": installed, "runtime_ready": bool(ready),
        "official_sources_sha256": source_hash, "assets_sha256": assets_hash,
        "runtime_sha256": sha256(canonical(receipt)),
        "render_profile": {"front_rgb": True, "official_anonymous_coordinates": True,
                           "official_xyz_axes": True, "official_yolo_boxes": True,
                           "raw_private_channels_exposed": False, "yolo_device": "cpu"},
        "model_requests": 0, "simulator_launches": 0,
        "source_sha256": {n: sha256((ROOT / n).read_bytes()) for n in
                          ("manipulation_worker.py", "manipulation_protocol.py",
                           "manipulation_sources.json", "manipulation_assets.json")},
    }


class OfficialManipulation:
    environment = "eb_manipulation"
    from action_adapters import validate_packet
    validate_packet = staticmethod(validate_packet)

    def __init__(self, subset, work):
        self.subset, self.work = subset, Path(work)
        self.env = self.display = None
        self.index = self.entry = self.last_packet = None
        self.done, self.success, self.action_success = False, False, True
        self.feedback, self.input_sha256 = "", ""

    def validate_open(self, parameters):
        if (parameters["subset"] != self.subset or self.subset not in SUBSETS
                or not any(r["subset"] == self.subset and r["index"] == parameters["index"]
                           for r in asset_catalog()["selection"])):
            raise ContractError("manipulation_task_not_installed")

    def _layout(self):
        layout = self.work / "official-layout"
        task = layout / "embodiedbench/envs/eb_manipulation"
        task.mkdir(parents=True, exist_ok=False)
        official = ROOT / "upstream-manipulation/embodiedbench/envs/eb_manipulation"
        for child in official.iterdir():
            (task / child.name).symlink_to(child)
        (task / "data").symlink_to(RUNTIME / "dataset/data")
        (layout / "yolo11n.pt").symlink_to(RUNTIME / "yolo11n.pt")
        sys.path.insert(0, str(task))
        os.chdir(layout)

    def _display(self):
        # This worker owns only this Xvfb child; no shared X server is modified.
        display_id = next(n for n in range(111, 130)
                          if not Path(f"/tmp/.X11-unix/X{n}").exists()
                          and not Path(f"/tmp/.X{n}-lock").exists())
        self.display = subprocess.Popen([
            str(ROOT / "runtime/xvfb/usr/bin/Xvfb"), ":" + str(display_id),
            "-screen", "0", "1024x768x24", "-nolisten", "tcp", "-ac", "+extension", "GLX"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 15
        while not Path(f"/tmp/.X11-unix/X{display_id}").exists():
            if self.display.poll() is not None or time.monotonic() >= deadline:
                raise OperationalError("manipulation_display_failed")
            time.sleep(.05)
        os.environ["DISPLAY"] = ":" + str(display_id)

    def open(self, parameters):
        self.validate_open(parameters)
        catalog = asset_catalog()
        if sha256(canonical(catalog)) != parameters["source_sha256"]:
            raise ContractError("manipulation_dataset_identity")
        verify_sources()
        verify_assets()
        self.index = parameters["index"]
        self.entry = next(r for r in catalog["selection"]
                          if r["subset"] == self.subset and r["index"] == self.index)
        self._layout()
        self._display()
        import numpy as np
        random.seed(137)
        np.random.seed(137)
        from embodiedbench.envs.eb_manipulation import eb_man_utils as utils
        original_predict = utils.object_detection_model.predict
        utils.object_detection_model.predict = lambda *a, **kw: original_predict(*a, device="cpu", **kw)
        from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
        self.env = EBManEnv(eval_set=self.subset, img_size=(500, 500), render_mode=None,
                            selected_indexes=[self.entry["runtime_index"]], log_path=str(self.work / "images"))
        if (self.env._max_episode_steps != 15 or self.env.number_of_episodes != 1
                or self.env.dataset[0][-1] != self.entry["task_variation"]):
            raise ContractError("manipulation_official_selection")
        expected = self.entry["episode_directory"]
        if any(not str(self.env.dataset[0][i]).endswith(expected + "/" + name)
               for i, name in ((1, "task_base.ttm"), (2, "waypoint_sets.ttm"), (3, "configs.pkl"))):
            raise ContractError("manipulation_official_episode_identity")
        instruction, _obs = self.env.reset()
        self.input_sha256 = sha256(canonical({
            "environment": self.environment, "subset": self.subset, "index": self.index,
            "dataset_sha256": parameters["source_sha256"], "instruction": instruction}))
        return self.packet()

    def packet(self):
        import numpy as np
        from PIL import Image
        from embodiedbench.envs.eb_manipulation.eb_man_utils import (
            form_object_coord_for_input, draw_xyz_coordinate, draw_bounding_boxes)
        obs = self.env.last_frame_obs
        rgb = np.asarray(obs["front_rgb"])
        if rgb.dtype != np.uint8 or rgb.shape != (500, 500, 3):
            raise ContractError("manipulation_official_rgb_shape")
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        png = stream.getvalue()
        coords, points, extrinsics, intrinsics = form_object_coord_for_input(obs, self.env.task_class, ["front_rgb"])
        objects = {name: [int(x) for x in value] for name, value in coords.items()}
        # Work-owned visual annotations follow the official evaluator, while the
        # original image remains the source of prediction tools.
        path = self.work / f"front-step-{self.env._current_step:02d}.png"
        Image.fromarray(rgb).save(path)
        draw_xyz_coordinate(str(path), 500)
        annotated = draw_bounding_boxes([str(path)], points, extrinsics, intrinsics)[0]
        planner_png = Path(annotated).read_bytes()
        public = {"action_spec": ACTION_SPEC, "task_variation": self.env.current_task_variation,
                  "objects": objects, "source": INPUT_SOURCE}
        public["sha256"] = sha256(canonical(public))
        self.last_packet = {
            "environment": self.environment, "official_task_id": f"{self.subset}:{self.index:03d}",
            "subset": self.subset, "input_sha256": self.input_sha256,
            "image_base64": base64.b64encode(png).decode(), "image_sha256": sha256(png),
            "planner_image_base64": base64.b64encode(planner_png).decode(),
            "planner_image_sha256": sha256(planner_png), "manipulation_input": public,
            "step": self.env._current_step, "instruction": self.env.episode_language_instruction,
            "actions": [], "env_feedback": self.feedback, "last_action_success": self.action_success,
            "official": {"done": self.done, "task_success": self.success,
                         "task_progress": float(self.success), "invalid_actions": 0},
        }
        return self.last_packet

    def step(self, action, first_in_plan):
        _obs, _reward, done, info = self.env.step(list(vector(action)))
        self.done = bool(done)
        self.success = info["task_success"] == 1.0
        self.action_success = info["action_success"] == 1.0
        self.feedback = info["env_feedback"]
        return self.packet()

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            if self.display is not None and self.display.poll() is None:
                self.display.terminate()
                try:
                    self.display.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.display.kill()
                    self.display.wait(timeout=3)
