"""Credential-free official ALFRED/Habitat worker, compatible with Python 3.9."""
from __future__ import annotations

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
import socket
import subprocess
import sys
import time
import traceback
import types

from bridge import simulator_environment
from household_protocol import ROOT, SUBSETS, VERSION, source, verify_environment
from protocol import ContractError, canonical, sha256, within_root


@contextlib.contextmanager
def native_log(log):
    """Capture native Habitat/Magnum output without corrupting the JSON wire."""
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


def runtime():
    environment = simulator_environment(ROOT)
    for key in tuple(environment):
        if "PROXY" in key.upper():
            environment.pop(key)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       CUDA_VISIBLE_DEVICES="7", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ.clear()
    os.environ.update(environment)
    addresses = socket.getaddrinfo
    socket.getaddrinfo = lambda host, port, family=0, *a, **kw: addresses(
        host, port, socket.AF_INET if family == 0 else family, *a, **kw)
    main = types.ModuleType("embodiedbench.main")
    main.logger = logging.getLogger("embodiedrunner-household")
    main.logger.disabled = True
    sys.modules["embodiedbench.main"] = main
    sys.path.insert(0, str(ROOT / "upstream-household"))


def probe(environment):
    expected = {"ai2thor": "2.1.0", "gym": "0.23.0", "numpy": "1.23.5"}
    versions = {}
    for name in expected:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    modules = ("ai2thor",) if environment == "alfred" else ("habitat", "habitat_sim")
    return {"versions": versions, "installed": {n: importlib.util.find_spec(n) is not None for n in modules},
            "runtime_ready": all(versions[n] == v for n, v in expected.items()) and all(
                importlib.util.find_spec(n) is not None for n in modules),
            "model_requests": 0, "simulator_launches": 0,
            "source_sha256": {name: sha256((ROOT / name).read_bytes()) for name in
                              ("household_worker.py", "household_protocol.py", "household_sources.json")}}


class OfficialHousehold:
    def __init__(self, environment, subset, work):
        self.kind, self.subset, self.work = environment, subset, work
        self.env, self.display, self.display_log = None, None, None
        self.observation, self.last_packet = None, None
        self.input_sha256 = None

    def init_alfred(self):
        import ai2thor.controller
        from PIL import ImageFont
        original_font = ImageFont.truetype
        def task_font(font, *a, **kw):
            if font == "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf":
                font = str(ROOT / "runtime/xvfb/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf")
            return original_font(font, *a, **kw)
        ImageFont.truetype = task_font
        original = ai2thor.controller.Controller
        work = self.work
        class TaskController(original):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.log_path = str(work / "unity.log")

            def base_dir(self):
                return str(ROOT / "cache/ai2thor-2.1")

            def unity_command(self, width, height, headless):
                return super().unity_command(width, height, headless) + ["-logFile", str(work / "unity.log")]

        ai2thor.controller.Controller = TaskController
        display_id = next(n for n in range(91, 110) if not Path(f"/tmp/.X11-unix/X{n}").exists()
                          and not Path(f"/tmp/.X{n}-lock").exists())
        self.display_log = (work / "xvfb.log").open("a")
        self.display = subprocess.Popen([
            str(ROOT / "runtime/xvfb/usr/bin/Xvfb"), ":" + str(display_id),
            "-screen", "0", "1024x768x24", "-nolisten", "tcp", "-ac", "+extension", "GLX"],
            stdout=self.display_log, stderr=self.display_log)
        deadline = time.monotonic() + 15
        while not Path(f"/tmp/.X11-unix/X{display_id}").exists():
            if self.display.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("task_display_start_failed")
            time.sleep(0.05)
        os.environ.update(DISPLAY=":" + str(display_id), LIBGL_ALWAYS_SOFTWARE="1",
                          GALLIUM_DRIVER="llvmpipe", LP_NUM_THREADS="2")
        from embodiedbench.envs.eb_alfred import EBAlfEnv as module
        module.X_DISPLAY = str(display_id)
        self.env = module.EBAlfEnv(eval_set=self.subset, exp_name="embodiedrunner",
                                  down_sample_ratio=1.0, detection_box=False, resolution=500)
        self.env.log_path = str(work / "simulator")

    def init_habitat(self):
        import habitat
        from omegaconf import OmegaConf
        original = habitat.get_config
        def config(*a, **kw):
            result = original(*a, **kw)
            OmegaConf.set_readonly(result, False)
            result.habitat.seed = 137
            result.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
            result.habitat.environment.iterator_options.shuffle = False
            result.habitat.environment.iterator_options.group_by_scene = False
            return result
        habitat.get_config = config
        os.chdir(ROOT / "upstream-household/embodiedbench/envs/eb_habitat")
        from embodiedbench.envs.eb_habitat.EBHabEnv import EBHabEnv
        self.env = EBHabEnv(eval_set=self.subset, exp_name="embodiedrunner",
                            down_sample_ratio=1.0, start_epi_index=0, resolution=500, recording=False)
        self.env.log_path = str(self.work / "simulator")

    def packet(self, obs, info=None, reward=0, done=False):
        import numpy as np
        from PIL import Image
        info = info or {}
        if self.kind == "habitat":
            from embodiedbench.envs.eb_habitat.utils import observations_to_image
            rgb = observations_to_image(obs, {}, key="head_rgb")
        else:
            rgb = obs["head_rgb"]
        if rgb.shape != (500, 500, 3) or rgb.dtype != np.uint8:
            raise ContractError("unexpected_rgb_shape")
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        png = stream.getvalue()
        result = {
            "image_base64": base64.b64encode(png).decode(), "image_sha256": sha256(png),
            "step": self.env._current_step, "instruction": self.env.episode_language_instruction,
            "actions": list(self.env.language_skill_set), "input_sha256": self.input_sha256,
            "done": bool(done), "task_success": bool(info.get("task_success", False)),
            "last_action_success": bool(info.get("last_action_success", True)),
            "task_progress": float(info.get("task_progress", 0)),
            "env_feedback": info.get("env_feedback", ""),
            "invalid_actions": self.env._cur_invalid_actions, "reward": float(reward),
        }
        self.observation, self.last_packet = obs, result
        return result

    def open(self, index, dataset_sha256):
        if self.subset not in SUBSETS[self.kind] or type(index) is not int or index < 0:
            raise ContractError("invalid_command")
        verify_environment(self.kind)
        path = ("embodiedbench/envs/eb_alfred/data/splits/splits.json" if self.kind == "alfred"
                else f"embodiedbench/envs/eb_habitat/datasets/{self.subset}.pickle")
        if sha256(source(path)) != dataset_sha256:
            raise ContractError("official_source_changed")
        if self.env is None:
            self.init_alfred() if self.kind == "alfred" else self.init_habitat()
        if self.env._max_episode_steps != 30 or self.env._max_invalid_actions != 10:
            raise ContractError("official_source_changed")
        if self.kind == "alfred":
            self.env._current_episode_num = index
            task = self.env.dataset[index]
            manifest = json.loads((ROOT / "alfred_pilot_data.json").read_text())
            data_path = task["task"] + f"/pp/ann_{task['repeat_idx']}.json"
            row = next(x for x in manifest["files"] if x["path"] == data_path)
            data = (ROOT / "upstream-household/embodiedbench/envs/eb_alfred/data/json_2.1.0" / data_path).read_bytes()
            if sha256(data) != row["sha256"]:
                raise ContractError("official_source_changed")
        elif self.env._current_episode_num != index:
            raise ContractError("habitat_nonsequential_episode")
        obs = self.env.reset()
        if self.kind == "habitat":
            # With shuffle disabled, the real episode matches this dataset index.
            actual = self.env.env.env.env._env.current_episode.episode_id
            if str(actual) != str(index):
                raise ContractError("habitat_episode_identity_mismatch")
        self.input_sha256 = sha256(canonical({
            "environment": self.kind, "subset": self.subset, "index": index,
            "dataset_sha256": dataset_sha256, "instruction": self.env.episode_language_instruction}))
        return self.packet(obs)

    def step(self, action):
        if type(action) is not int or not 0 <= action < len(self.env.language_skill_set):
            raise ContractError("invalid_action_id")
        obs, reward, done, info = self.env.step(action, reasoning="")
        return self.packet(obs, info, reward, done)

    def invalid(self):
        self.env._cur_invalid_actions += 1
        packet = dict(self.last_packet)
        packet.update(invalid_actions=self.env._cur_invalid_actions,
                      done=self.env._cur_invalid_actions >= 10, reward=-1)
        self.last_packet = packet
        return packet

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            if self.display is not None:
                self.display.terminate()
                self.display.wait(timeout=10)
                self.display = None
            if self.display_log:
                self.display_log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("probe", "worker", "smoke"))
    parser.add_argument("--environment", choices=tuple(SUBSETS), required=True)
    parser.add_argument("--subset", default="base")
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()
    runtime()
    if args.action == "probe":
        print(canonical(probe(args.environment)).decode())
        return
    work = within_root(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    session = OfficialHousehold(args.environment, args.subset, work)
    if args.action == "smoke":
        receipt = {"model_requests": 0, "check": "real_reset_step", "environment": args.environment}
        try:
            with (work / "runtime.log").open("a") as log, native_log(log):
                path = "embodiedbench/envs/eb_alfred/data/splits/splits.json" if args.environment == "alfred" else \
                    f"embodiedbench/envs/eb_habitat/datasets/{args.subset}.pickle"
                packet = session.open(0, sha256(source(path)))
                (work / "initial.png").write_bytes(base64.b64decode(packet["image_base64"]))
                second = session.step(0)
            receipt.update(status="ready", initial_step=packet["step"], final_step=second["step"],
                           input_sha256=packet["input_sha256"], action_count=len(packet["actions"]),
                           initial_image_sha256=packet["image_sha256"], final_image_sha256=second["image_sha256"])
        except Exception as exc:
            receipt.update(status="failed", exception_type=type(exc).__name__, exception_module=type(exc).__module__,
                           stack=[{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                                  for f in traceback.extract_tb(exc.__traceback__)])
        finally:
            session.close()
        (work / "receipt.json").write_bytes(canonical(receipt) + b"\n")
        print(canonical(receipt).decode())
        return 0 if receipt["status"] == "ready" else 2
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as wire, \
            (work / "runtime.log").open("a") as log, native_log(log):
        try:
            for line in sys.stdin:
                op, identity = None, None
                try:
                    cmd = json.loads(line)
                    op, identity = cmd["op"], cmd["id"]
                    if op == "open":
                        data = session.open(cmd["index"], cmd["dataset_sha256"])
                    elif op == "step":
                        data = session.step(cmd["action"])
                    elif op == "invalid":
                        data = session.invalid()
                    elif op == "close":
                        data = {}
                    else:
                        raise ContractError("invalid_command")
                    response = {"id": identity, "version": VERSION, "ok": True, "data": data}
                except Exception as exc:
                    code = str(exc) if isinstance(exc, ContractError) else "environment_error"
                    response = {"id": identity, "version": VERSION, "ok": False, "error_code": code}
                    diagnostic = {"exception_type": type(exc).__name__,
                                  "stack": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                                            for f in traceback.extract_tb(exc.__traceback__)]}
                    (work / "error.json").write_bytes(canonical(diagnostic) + b"\n")
                wire.write(canonical(response).decode() + "\n")
                wire.flush()
                if op == "close" or not response["ok"]:
                    break
        finally:
            session.close()


if __name__ == "__main__":
    raise SystemExit(main())
