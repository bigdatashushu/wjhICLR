"""Official EB-Navigation simulator worker; compatible with Python 3.9."""
from __future__ import annotations

import argparse
import base64
import contextlib
import importlib.metadata
import importlib.util
import io
import logging
from pathlib import Path
import platform
import sys
import types

from protocol import (ACTIONS, ROOT, SUBSETS, VERSION, ContractError,
                      action_id, canonical, json_object, sha256,
                      verified_sources, within_root)


def probe():
    packages = {"ai2thor": "ai2thor", "gym": "gym", "numpy": "numpy",
                "PIL": "Pillow", "cv2": "opencv-python"}
    installed, versions = {}, {}
    for module, distribution in packages.items():
        installed[module] = importlib.util.find_spec(module) is not None
        try:
            versions[module] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[module] = None
    pinned = {"ai2thor": "5.0.0", "gym": "0.26.2", "numpy": "1.26.4"}
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "installed": installed, "versions": versions,
        "runtime_ready": platform.system() == "Linux" and all(installed.values())
                         and all(versions[key] == value for key, value in pinned.items()),
        "simulator_launches": 0, "model_requests": 0,
    }


def environment_factory(upstream, work_dir, gpu_device=None):
    import ai2thor.build
    import ai2thor.controller
    import ai2thor.downloader

    # 5.0.0 uses an old HTTP S3 endpoint; fetch the same hash-checked build via TLS.
    ai2thor.build.base_url = ai2thor.build.base_url.replace("http://", "https://", 1)
    original_download = ai2thor.downloader.download
    build_name = "thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917.zip"
    archive = ROOT / "cache/unity-download" / build_name

    def cached_download(url, sha256_digest, include_private_scenes=False):
        if url.endswith("/" + build_name) and archive.is_file() and not include_private_scenes:
            data = archive.read_bytes()
            if sha256(data) != sha256_digest:
                raise ContractError("unity_archive_hash_mismatch")
            return data
        return original_download(url, sha256_digest, include_private_scenes)

    ai2thor.downloader.download = cached_download
    original_controller = ai2thor.controller.Controller
    cache = str(ROOT / "cache/ai2thor")

    class TaskController(original_controller):
        def __init__(self, *args, **kwargs):
            if gpu_device is not None:
                kwargs["gpu_device"] = gpu_device
            super().__init__(*args, **kwargs)

        @property
        def log_dir(self):
            return str(work_dir)

        def unity_command(self, width, height, headless):
            return super().unity_command(width, height, headless) + [
                "-logFile", str(work_dir / "player.log"),
            ]

        # AI2-THOR 5.0.0 exposes this as a property, not a constructor argument.
        @property
        def base_dir(self):
            return cache

    ai2thor.controller.Controller = TaskController
    # EBNavEnv only imports logger from main; importing main normally creates a
    # data symlink and imports Hydra. Supply that dependency without side effects.
    main_module = types.ModuleType("embodiedbench.main")
    main_module.logger = logging.getLogger("embodiedrunner-simulator")
    main_module.logger.disabled = True
    sys.modules["embodiedbench.main"] = main_module
    sys.path.insert(0, str(upstream))
    from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv
    return EBNavigationEnv


class OfficialNavigation:
    def __init__(self, upstream, work_dir, *, resolution=500, factory=None, gpu_device=None):
        self.upstream, self.work_dir = Path(upstream).resolve(), Path(work_dir).resolve()
        self.resolution = resolution
        self.factory = factory
        self.gpu_device = gpu_device
        self.env, self.subset, self.instruction = None, None, ""
        self.active = False

    def _packet(self, observation, *, done=False, success=False, action_success=True,
                feedback=""):
        from PIL import Image
        import numpy as np

        rgb = observation["head_rgb"]
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.shape != (self.resolution, self.resolution, 3)):
            raise ContractError("invalid_official_rgb")
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        png = stream.getvalue()
        return {
            "image_base64": base64.b64encode(png).decode("ascii"),
            "image_sha256": sha256(png), "step": self.env._current_step,
            "instruction": self.instruction, "done": bool(done),
            "task_success": bool(success), "last_action_success": bool(action_success),
            "env_feedback": feedback,
        }

    def open(self, subset, index, input_sha256):
        if subset not in SUBSETS or type(index) is not int or not 0 <= index < 60:
            raise ContractError("invalid_command")
        verified_sources(self.upstream, [subset])
        source = self.upstream / f"embodiedbench/envs/eb_navigation/datasets/{subset}.json"
        rows = json_object(source.read_text(encoding="utf-8"))["tasks"]
        if sha256(canonical(rows[index])) != input_sha256:
            raise ContractError("official_source_changed")
        if self.env is None or self.subset != subset:
            self.close()
            factory = self.factory or environment_factory(self.upstream, self.work_dir, self.gpu_device)
            self.env = factory(
                eval_set=subset, exp_name="embodiedrunner", down_sample_ratio=1.0,
                multiview=False, boundingbox=False, multistep=False,
                resolution=self.resolution,
            )
            self.subset = subset
            self.env.log_path = str(self.work_dir / "simulator")
            # The upstream logger writes instruction, reasoning and scorer distance.
            # Keep our own sanitized trace instead of persisting that raw stream.
            self.env.save_episode_log_per_step = lambda _flag: None
        if (tuple(self.env.language_skill_set) != ACTIONS
                or self.env._max_episode_steps != 20 or self.env.feedback_verbosity != 0):
            raise ContractError("official_source_changed")
        self.env._current_episode_num = index
        observation = self.env.reset()
        self.instruction = self.env.episode_language_instruction
        self.active = True
        return self._packet(observation)

    def step(self, action, first_in_plan):
        number = action_id(action)
        if not self.active or type(first_in_plan) is not bool:
            raise ContractError("invalid_command")
        # No model text is passed to upstream logging or simulator methods.
        observation, _reward, done, info = self.env.step(number, {}, int(first_in_plan))
        if info["task_success"] not in (0, 1) or info["last_action_success"] not in (False, True):
            raise ContractError("invalid_official_feedback")
        self.active = not done
        return self._packet(
            observation, done=done, success=info["task_success"],
            action_success=info["last_action_success"], feedback=info["env_feedback"],
        )

    def close(self):
        if self.env is not None:
            self.env.close()
        self.env, self.subset, self.active = None, None, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--upstream", type=Path, default=ROOT / "upstream")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument("--gpu-device", type=int)
    args = parser.parse_args()
    if args.probe:
        print(canonical(probe()).decode("utf-8"))
        return
    work_dir = within_root(args.work_dir) if args.work_dir else None
    if work_dir is None or args.resolution not in (300, 500):
        parser.error("work-dir and supported resolution are required")
    wire = sys.stdout
    session = OfficialNavigation(args.upstream, work_dir, resolution=args.resolution,
                                 gpu_device=args.gpu_device)
    # Simulator and dependency output cannot corrupt the NDJSON channel.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            for line in sys.stdin:
                command_id, op = None, None
                try:
                    command = json_object(line)
                    command_id, op = command["id"], command["op"]
                    if type(command_id) is not int or command_id < 1:
                        raise ContractError("invalid_command")
                    if op == "open" and set(command) == {"id", "op", "subset", "index", "input_sha256"}:
                        if not probe()["runtime_ready"]:
                            raise ContractError("missing_simulator_dependencies")
                        data = session.open(command["subset"], command["index"], command["input_sha256"])
                    elif op == "step" and set(command) == {"id", "op", "action", "first_in_plan"}:
                        data = session.step(command["action"], command["first_in_plan"])
                    elif op == "close" and set(command) == {"id", "op"}:
                        session.close()
                        data = {}
                    else:
                        raise ContractError("invalid_command")
                    response = {"id": command_id, "version": VERSION, "ok": True, "data": data}
                except Exception as exc:
                    code = str(exc) if isinstance(exc, ContractError) else "environment_error"
                    if code not in {"invalid_command", "invalid_action_id", "official_source_changed",
                                    "missing_simulator_dependencies"}:
                        code = "environment_error"
                    response = {"id": command_id, "version": VERSION, "ok": False, "error_code": code}
                wire.write(canonical(response).decode("utf-8") + "\n")
                wire.flush()
                if op == "close" or not response["ok"]:
                    break
        finally:
            session.close()


if __name__ == "__main__":
    main()
