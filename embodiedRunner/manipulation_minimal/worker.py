"""Credential-free simulator process. Compatible with server Python 3.9."""
import argparse
import base64
import contextlib
import importlib.metadata
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import types

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "manipulation_minimal"

from . import VERSION
from .protocol import COORD_SOURCE, Task, validate_packet, vector
from .sources import (ContractError, LIMITS, OperationalError, Sources, TREES,
                      canonical, digest, file_digest, implementation_hashes)


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def selected_assets(sources, task):
    """Read manifests and hash selected assets; private pickle is never decoded here."""
    root, kind = sources.root, task.environment
    if kind == "navigation":
        path = f"embodiedbench/envs/eb_navigation/datasets/{task.subset}.json"
        document = json.loads(sources.read(kind, path))
        if task.index >= len(document["tasks"]):
            raise ContractError("task_index_out_of_range")
        return {"dataset_sha256": digest(sources.read(kind, path))}
    if kind == "habitat":
        path = f"embodiedbench/envs/eb_habitat/datasets/{task.subset}.pickle"
        return {"dataset_sha256": digest(sources.read(kind, path))}
    if kind == "alfred":
        path = "embodiedbench/envs/eb_alfred/data/splits/splits.json"
        split = json.loads(sources.read(kind, path))
        if task.subset not in split or task.index >= len(split[task.subset]):
            raise ContractError("task_index_out_of_range")
        entry = split[task.subset][task.index]
        relative = entry["task"] + f"/pp/ann_{entry['repeat_idx']}.json"
        for manifest, folder in (
                ("action_alfred_data.json", "action-alfred-data"),
                ("alfred_pilot_data.json", "upstream-household/embodiedbench/envs/eb_alfred/data/json_2.1.0")):
            catalog = json.loads((root / manifest).read_bytes())
            row = next((r for r in catalog["files"] if r["path"] == relative), None)
            if row is not None:
                asset = root / folder / relative
                if asset.stat().st_size != row["bytes"] or file_digest(asset) != row["sha256"]:
                    raise ContractError("alfred_asset_changed")
                return {"dataset_sha256": digest(sources.read(kind, path)),
                        "asset_path": str(asset), "asset_sha256": row["sha256"],
                        "task_relative_path": relative, "entry": entry}
        raise ContractError("alfred_task_not_installed")
    catalog_raw = sources.manipulation_catalog.read_bytes()
    catalog = json.loads(catalog_raw)
    entry = next((r for r in catalog["selection"] if r["subset"] == task.subset and r["index"] == task.index), None)
    if entry is None or catalog["reference_answer_visible"] is not False:
        raise ContractError("manipulation_task_not_installed")
    runtime = (root / "manipulation-runtime").resolve()
    selected = [row for row in catalog["files"] if row["path"].startswith(entry["episode_directory"] + "/")]
    if {Path(row["path"]).name for row in selected} != {"configs.pkl", "task_base.ttm", "waypoint_sets.ttm"}:
        raise ContractError("manipulation_episode_files_missing")
    for row in selected:
        if row["path"].startswith(entry["episode_directory"] + "/"):
            asset = (sources.manipulation_dataset / row["path"]).resolve()
            if sources.manipulation_dataset not in asset.parents or asset.stat().st_size != row["bytes"] or file_digest(asset) != row["sha256"]:
                raise ContractError("manipulation_asset_changed")
    receipt = json.loads((runtime / "runtime.json").read_bytes())
    if (receipt["coppeliasim_version"] != "4.1.0"
            or receipt["pyrep_revision"] != "8f420be8064b1970aae18a9cfbc978dfb15747ef"
            or not isinstance(receipt["packages"], dict)
            or receipt["packages"].get("numpy") != "1.26.4"):
        raise ContractError("coppeliasim_version_mismatch")
    for row in receipt["pinned_files"]:
        asset = (runtime / row["path"]).resolve()
        if runtime not in asset.parents or asset.stat().st_size != row["bytes"] or file_digest(asset) != row["sha256"]:
            raise ContractError("manipulation_runtime_changed")
    return {"dataset_sha256": digest(catalog_raw), "entry": entry,
            "runtime_sha256": digest(canonical(receipt)),
            "expected_packages": receipt["packages"]}


def gpu_probe(index, minimum_free=0):
    if index is None:
        return {"required": False, "idle": True}
    if index not in (0, 1, 3, 7):
        raise ContractError("gpu_outside_authorized_set")
    try:
        rows = subprocess.check_output([
            "nvidia-smi", f"--id={index}", "--query-gpu=uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], timeout=10, stderr=subprocess.DEVNULL).decode().strip()
        uuid, memory, free, utilization = [x.strip() for x in rows.split(",")]
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"], timeout=10, stderr=subprocess.DEVNULL).decode()
        occupied = any(line.split(",")[0].strip() == uuid for line in processes.splitlines())
        # Also catch graphics users that are absent from compute-apps.
        idle = not occupied and int(memory) <= 64
        return {"required": True, "physical_index": index, "memory_used_mb": int(memory),
                "memory_free_mb": int(free), "utilization_percent": int(utilization),
                "idle": idle, "minimum_free_mb": minimum_free,
                "admissible": int(free) >= minimum_free if minimum_free else idle}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"required": True, "physical_index": index, "idle": False, "status": "unavailable"}


def probe(sources, task, gpu, minimum_free=0):
    counts = sources.verify_all()
    assets = selected_assets(sources, task)
    expected = {
        "navigation": {"ai2thor": "5.0.0", "gym": "0.26.2", "numpy": "1.26.4"},
        "alfred": {"ai2thor": "2.1.0", "gym": "0.23.0", "numpy": "1.23.5"},
        "habitat": {"habitat-sim": "0.3.0", "habitat-lab": "0.3.0", "numpy": "1.23.5"},
        "manipulation": assets.get("expected_packages", {}),
    }[task.environment]
    versions = {name: package_version(name) for name in expected}
    modules = {"navigation": ("ai2thor", "cv2", "PIL"), "alfred": ("ai2thor", "PIL"),
               "habitat": ("habitat", "habitat_sim"),
               "manipulation": ("pyrep", "amsolver", "ultralytics")}[task.environment]
    installed = {name: importlib.util.find_spec(name) is not None for name in modules}
    gpu_state = gpu_probe(gpu, minimum_free)
    return {"runtime_ready": platform.system() == "Linux"
            and versions == expected and all(installed.values()),
            "versions": versions, "installed": installed, "gpu": gpu_state,
            "official_file_counts": counts, "source_hashes": sources.hashes,
            "implementation_hashes": implementation_hashes(),
            "dataset_sha256": assets["dataset_sha256"],
            "simulator_launches": 0, "model_requests": 0}


@contextlib.contextmanager
def native_log(log):
    saved = [os.dup(fd) for fd in (1, 2)]
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            yield
    finally:
        log.flush()
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def runtime_environment(work, sources, task, gpu):
    original_path = os.environ.get("PATH", "/usr/bin:/bin")
    environment = {"PATH": original_path, "LANG": "C.UTF-8",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    for name, relative in {
        "HOME": "home", "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache",
        "HF_HOME": "cache/huggingface", "TORCH_HOME": "cache/torch",
        "MPLCONFIGDIR": "cache/matplotlib", "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data", "XDG_STATE_HOME": "state",
    }.items():
        path = work / relative
        path.mkdir(parents=True)
        environment[name] = str(path)
    environment["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    if task.environment == "navigation":
        # AI2-THOR gpu_device is a physical CUDA index in the existing runtime.
        environment.pop("CUDA_VISIBLE_DEVICES")
        environment["PATH"] = str(sources.root / "runtime/vulkan-tools/usr/bin") + os.pathsep + original_path
    if task.environment in ("alfred", "manipulation"):
        environment.update(LIBGL_ALWAYS_SOFTWARE="1", GALLIUM_DRIVER="llvmpipe", LP_NUM_THREADS="2")
    if task.environment == "manipulation":
        sim = str(sources.root / "manipulation-runtime/coppeliasim")
        environment.update(COPPELIASIM_ROOT=sim, LD_LIBRARY_PATH=sim, QT_QPA_PLATFORM_PLUGIN_PATH=sim)
    os.environ.clear()
    os.environ.update(environment)
    main = types.ModuleType("embodiedbench.main")
    main.logger = logging.getLogger("official-direct-worker")
    main.logger.disabled = True
    sys.modules["embodiedbench.main"] = main
    sys.path.insert(0, str(sources.root / TREES[task.environment][0]))


class Environment:
    def __init__(self, sources, task, work, gpu, seed):
        self.sources, self.task, self.work = sources, task, work
        self.gpu, self.seed = gpu, seed
        self.env, self.display, self.display_log, self.last_packet = None, None, None, None
        self.obs, self.info, self.reward, self.done = None, {}, 0, False
        self.assets = selected_assets(sources, task)

    def display_start(self):
        # Xorg allocates an unused display atomically, including other users.
        read_fd, write_fd = os.pipe()
        self.display_log = (self.work / "xvfb.log").open("x")
        self.display = subprocess.Popen([
            str(self.sources.root / "runtime/xvfb/usr/bin/Xvfb"), "-displayfd", str(write_fd),
            "-screen", "0", "1024x768x24", "-nolisten", "tcp", "-ac", "+extension", "GLX"],
            stdout=self.display_log, stderr=self.display_log, pass_fds=(write_fd,))
        os.close(write_fd)
        import select
        try:
            if not select.select([read_fd], [], [], 15)[0]:
                raise OperationalError("owned_display_start_failed")
            value = os.read(read_fd, 32).decode().strip()
            if not value.isdigit() or self.display.poll() is not None:
                raise OperationalError("owned_display_start_failed")
            index = int(value)
        finally:
            os.close(read_fd)
        os.environ["DISPLAY"] = f":{index}"
        return index

    def thor(self, navigation):
        import ai2thor.controller
        original = ai2thor.controller.Controller
        root, work = self.sources.root, self.work
        cache = work / ("thor5" if navigation else "thor21")
        # Real copies prevent lock files or runtime updates touching old caches.
        shutil.copytree(root / ("cache/ai2thor" if navigation else "cache/ai2thor-2.1"), cache)
        if navigation:
            import ai2thor.build
            import ai2thor.downloader
            ai2thor.build.base_url = ai2thor.build.base_url.replace("http://", "https://", 1)
            def cached_download(url, sha256_digest, include_private_scenes=False):
                archive = root / "cache/unity-download" / url.rsplit("/", 1)[-1]
                if not include_private_scenes and archive.is_file() and file_digest(archive) == sha256_digest:
                    return archive.read_bytes()
                raise OperationalError("uncached_unity_build")
            ai2thor.downloader.download = cached_download
            class Controller(original):
                @property
                def base_dir(self):
                    return str(cache)
                @property
                def log_dir(self):
                    return str(work)
                def __init__(self, *args, **kwargs):
                    kwargs["gpu_device"] = self_outer.gpu
                    super().__init__(*args, **kwargs)
                def unity_command(self, *args, **kwargs):
                    return super().unity_command(*args, **kwargs) + ["-logFile", str(work / "unity.log")]
            self_outer = self
        else:
            class Controller(original):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.log_path = str(work / "unity.log")
                def base_dir(self):
                    return str(cache)
                def unity_command(self, *args, **kwargs):
                    return super().unity_command(*args, **kwargs) + ["-logFile", str(work / "unity.log")]
            from PIL import ImageFont
            original_font = ImageFont.truetype
            def font(name, *args, **kwargs):
                if name == "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf":
                    name = str(root / "runtime/xvfb/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf")
                return original_font(name, *args, **kwargs)
            ImageFont.truetype = font
        ai2thor.controller.Controller = Controller

    def open(self):
        kind, root, task = self.task.environment, self.sources.root, self.task
        runtime_environment(self.work, self.sources, task, self.gpu)
        random.seed(self.seed)
        import numpy as np
        np.random.seed(self.seed)
        os.chdir(self.work)
        if kind == "navigation":
            self.thor(True)
            from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv
            self.env = EBNavigationEnv(eval_set=task.subset, selected_indexes=[task.index],
                                      resolution=500, multiview=False, multistep=False, boundingbox=False)
            self.env.log_path = str(self.work / "official")
            self.obs = self.env.reset()
        elif kind == "alfred":
            self.thor(False)
            display = self.display_start()
            from embodiedbench.envs.eb_alfred import EBAlfEnv as module
            module.X_DISPLAY = str(display)
            def load_task(entry):
                name = entry["task"] + f"/pp/ann_{entry['repeat_idx']}.json"
                if name != self.assets["task_relative_path"]:
                    raise ContractError("alfred_episode_identity_mismatch")
                return json.loads(Path(self.assets["asset_path"]).read_bytes())
            module.utils.load_task_json = load_task
            self.env = module.EBAlfEnv(eval_set=task.subset, selected_indexes=[task.index],
                                      detection_box=False, resolution=500)
            self.env.log_path = str(self.work / "official")
            self.obs = self.env.reset()
            if self.env.episode_language_instruction != self.assets["entry"]["instruction"]:
                raise ContractError("alfred_instruction_identity_mismatch")
        elif kind == "habitat":
            import habitat
            from omegaconf import OmegaConf
            original_config = habitat.get_config
            def get_config(*args, **kwargs):
                config = original_config(*args, **kwargs)
                OmegaConf.set_readonly(config, False)
                config.habitat.seed = self.seed
                config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
                config.habitat.environment.iterator_options.shuffle = False
                config.habitat.environment.iterator_options.group_by_scene = False
                return config
            habitat.get_config = get_config
            (self.work / "data").rmdir()
            (self.work / "data").symlink_to(root / "upstream-household/embodiedbench/envs/eb_habitat/data")
            from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
            self.env = EBHabEnv(eval_set=task.subset, start_epi_index=task.index,
                               resolution=500, recording=False)
            self.env.log_path = str(self.work / "official")
            self.obs = self.env.reset()
            expected = self.env.dataset.episodes[task.index].episode_id
            actual = self.env.env.env.env._env.current_episode.episode_id
            if str(actual) != str(expected) or self.env._current_episode_num != task.index + 1:
                raise ContractError("habitat_episode_identity_mismatch")
        else:
            self.display_start()
            layout = self.work / "layout"
            directory = layout / "embodiedbench/envs/eb_manipulation"
            directory.mkdir(parents=True)
            for child in (root / "upstream-manipulation/embodiedbench/envs/eb_manipulation").iterdir():
                if child.name != "data":
                    (directory / child.name).symlink_to(child)
            (directory / "data").symlink_to(self.sources.manipulation_dataset / "data")
            (layout / "yolo11n.pt").symlink_to(root / "manipulation-runtime/yolo11n.pt")
            sys.path.insert(0, str(directory))
            os.chdir(layout)
            from embodiedbench.envs.eb_manipulation import eb_man_utils as utils
            predict = utils.object_detection_model.predict
            utils.object_detection_model.predict = lambda *args, **kwargs: predict(*args, device="cpu", **kwargs)
            from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
            entry = self.assets["entry"]
            self.env = EBManEnv(eval_set=task.subset, render_mode=None, img_size=(500, 500),
                                selected_indexes=[entry["runtime_index"]], log_path=str(self.work / "official"))
            if self.env.dataset[0][-1] != entry["task_variation"] or any(
                    not str(self.env.dataset[0][i]).endswith(entry["episode_directory"] + "/" + name)
                    for i, name in ((1, "task_base.ttm"), (2, "waypoint_sets.ttm"), (3, "configs.pkl"))):
                raise ContractError("manipulation_episode_identity_mismatch")
            self.env.reset()
            self.obs = self.env.last_frame_obs
        if self.env._max_episode_steps != LIMITS[kind]:
            raise ContractError("official_action_limit_changed")
        if kind in ("alfred", "habitat") and self.env._max_invalid_actions != 10:
            raise ContractError("official_invalid_limit_changed")
        return self.packet(annotate=kind == "manipulation")

    def packet(self, annotate=False):
        import numpy as np
        from PIL import Image
        kind = self.task.environment
        if kind == "habitat":
            from embodiedbench.envs.eb_habitat.utils import observations_to_image
            rgb = observations_to_image(self.obs, {}, key="head_rgb")
        else:
            rgb = self.obs["front_rgb" if kind == "manipulation" else "head_rgb"]
        if rgb.dtype != np.uint8 or rgb.shape != (500, 500, 3):
            raise ContractError("invalid_official_rgb")
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        png, manip, image_source = stream.getvalue(), None, "official current RGB"
        if kind == "manipulation":
            from embodiedbench.envs.eb_manipulation.eb_man_utils import (
                form_object_coord_for_input, draw_xyz_coordinate, draw_bounding_boxes)
            coords, points, extrinsics, intrinsics = form_object_coord_for_input(
                self.obs, self.env.task_class, ["front_rgb"])
            manip = {"objects": {name: [int(x) for x in point] for name, point in coords.items()},
                     "task_variation": self.env.current_task_variation, "source": COORD_SOURCE}
            if annotate:
                path = self.work / f"step-{self.env._current_step:03d}-front.png"
                path.write_bytes(png)
                draw_xyz_coordinate(str(path), 500)
                png = Path(draw_bounding_boxes([str(path)], points, extrinsics, intrinsics)[0]).read_bytes()
                image_source = "official front_rgb + draw_xyz_coordinate + draw_bounding_boxes"
        info = self.info
        public = {"instruction": self.env.episode_language_instruction,
                  "images": [{"view": "front_rgb" if kind == "manipulation" else "head_rgb",
                              "base64": base64.b64encode(png).decode(), "sha256": digest(png),
                              "source": image_source}],
                  "actions": [] if kind == "manipulation" else list(self.env.language_skill_set),
                  "step": self.env._current_step, "env_feedback": info.get("env_feedback", ""),
                  "last_action_success": bool(info.get("action_success" if kind == "manipulation"
                                                      else "last_action_success", True)),
                  "manipulation": manip}
        result = {"done": bool(self.done), "task_success": float(info.get("task_success", 0)),
                  "invalid_actions": int(getattr(self.env, "_cur_invalid_actions", 0)),
                  "reward": float(self.reward), "termination_source": "official_environment"}
        for key in ("task_progress", "distance", "episode_elapsed_seconds"):
            if key in info:
                value = info[key]
                # Habitat's official code accidentally wraps elapsed time in a tuple.
                if key == "episode_elapsed_seconds" and isinstance(value, tuple) and len(value) == 1:
                    value = value[0]
                result[key] = float(value)
        packet = {"task": self.task.as_dict(), "public": public, "official": result,
                  "input_sha256": digest(canonical({"task": self.task.as_dict(),
                                                   "instruction": public["instruction"]}))}
        self.last_packet = validate_packet(packet, self.task)
        return self.last_packet

    def step(self, action, first_in_plan, reasoning):
        if self.done or self.env is None:
            raise ContractError("step_without_active_episode")
        kind = self.task.environment
        if kind == "manipulation":
            action = vector(action)
            self.obs, self.reward, self.done, self.info = self.env.step(action)
        else:
            if type(action) is not int or not 0 <= action < len(self.env.language_skill_set):
                raise ContractError("invalid_action_id")
            if kind == "navigation":
                self.obs, self.reward, self.done, self.info = self.env.step(action, reasoning, int(first_in_plan))
            else:
                self.obs, self.reward, self.done, self.info = self.env.step(action, reasoning=reasoning)
        return self.packet()

    def invalid(self, reasoning):
        if self.task.environment not in ("alfred", "habitat") or self.done:
            raise ContractError("invalid_plan_operation_not_supported")
        self.env._cur_invalid_actions += 1
        self.env.episode_log.append({"last_action_success": 0.0, "action_id": -1,
                                     "action_description": "invalid action", "reasoning": reasoning})
        self.done = self.env._cur_invalid_actions >= 10
        self.reward = -1
        packet = self.packet()
        packet["official"]["termination_source"] = "official_evaluator_invalid_plan"
        return packet

    def close(self):
        try:
            if self.env is not None:
                env, self.env = self.env, None
                if self.task.environment in ("alfred", "habitat"):
                    try:
                        env.save_episode_log()
                    finally:
                        env.close()
                else:
                    env.close()
        finally:
            if self.display is not None and self.display.poll() is None:
                self.display.terminate()
                try:
                    self.display.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.display.kill()
                    self.display.wait(timeout=3)
            if self.display_log is not None:
                self.display_log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--manipulation-catalog")
    parser.add_argument("--manipulation-dataset")
    parser.add_argument("--gpu-min-free-mb", type=int, default=0)
    parser.add_argument("--work-dir")
    parser.add_argument("--environment", choices=tuple(LIMITS), required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--seed", type=int, default=137)
    args = parser.parse_args()
    task = Task(args.environment, args.subset, args.index)
    sources = Sources(args.source_root, args.manipulation_catalog, args.manipulation_dataset)
    if args.probe:
        try:
            result = probe(sources, task, args.gpu, args.gpu_min_free_mb)
            if task.environment == "manipulation":
                import pyrep.backend.sim
                result["native_import_ready"] = True
        except Exception as exc:
            result = {"runtime_ready": False, "error_code": "runtime_preflight_failed",
                      "exception_type": type(exc).__name__, "model_requests": 0, "simulator_launches": 0}
        print(canonical(result).decode())
        return 0
    work = Path(args.work_dir).resolve()
    if Path(__file__).resolve().parent not in work.parents:
        raise ContractError("worker_work_dir_outside_independent_package")
    work.mkdir(parents=True, exist_ok=False)
    session = Environment(sources, task, work, args.gpu, args.seed)
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as wire, \
            (work / "runtime.log").open("x") as log, native_log(log):
        try:
            for line in sys.stdin:
                command, operation = {}, None
                try:
                    command = json.loads(line)
                    operation = command["op"]
                    if operation == "open":
                        if command["task"] != task.as_dict() or session.env is not None:
                            raise ContractError("invalid_open_identity")
                        # Verify a fresh snapshot/runtime/assets/GPU state before launch.
                        report = probe(sources, task, args.gpu, args.gpu_min_free_mb)
                        if (not report["runtime_ready"] or not report["gpu"].get("admissible", report["gpu"]["idle"])
                                or report["source_hashes"] != command["source_hashes"]
                                or report["implementation_hashes"] != command["implementation_hashes"]):
                            raise OperationalError("runtime_preflight_failed")
                        data = session.open()
                    elif operation == "step":
                        data = session.step(command["action"], command["first_in_plan"], command["reasoning"])
                    elif operation == "observe":
                        if session.env is None or session.done or task.environment != "manipulation":
                            raise ContractError("invalid_observe_operation")
                        data = session.packet(annotate=True)
                    elif operation == "invalid":
                        data = session.invalid(command["reasoning"])
                    elif operation == "close":
                        session.close()
                        session.env, session.display = None, None
                        data = {}
                    else:
                        raise ContractError("unknown_worker_operation")
                    response = {"id": command["id"], "version": VERSION, "ok": True, "data": data}
                except Exception as exc:
                    response = {"id": command.get("id"), "version": VERSION, "ok": False,
                                "error_code": "environment_error", "exception_type": type(exc).__name__}
                    (work / "failure.json").write_bytes(canonical({
                        "operation": operation, "exception_type": type(exc).__name__,
                        "stack": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                                  for f in traceback.extract_tb(exc.__traceback__)],
                        "automatic_retry": False}) + b"\n")
                wire.write(canonical(response).decode() + "\n")
                wire.flush()
                if operation == "close" or not response["ok"]:
                    break
        finally:
            session.close()


if __name__ == "__main__":
    main()
