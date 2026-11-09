from __future__ import annotations

import ast
from importlib import metadata
import os
from pathlib import Path
import random
import subprocess
import sys

from .config import Config, ROOT, RunnerError
from .observation import public_observation
from .storage import file_digest, read_json, verify_files, write_json

ORIGIN = "github.com/Lifelong-Robot-Learning/LIBERO"
SOURCE_FILES = (
    "libero/lifelong/metric.py",
    "libero/libero/benchmark/__init__.py",
    "libero/libero/envs/env_wrapper.py",
    "libero/libero/envs/bddl_base_domain.py",
    "libero/configs/eval/default.yaml",
    "libero/configs/data/default.yaml",
)


def git(root: Path, *args) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL,
            timeout=15,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        raise RunnerError("Cannot verify the official LIBERO Git checkout.") from None


def upstream_identity(root: Path) -> dict:
    root = root.resolve()
    if not all((root / name).is_file() for name in SOURCE_FILES):
        raise RunnerError(
            f"Original LIBERO source is missing at {root}. "
            "Run LIBERORunner/scripts/install.sh or set --libero-root."
        )
    receipt_path = root / ".official-archive.json"
    if receipt_path.exists() and not (root / ".git").exists():
        from .scripts.stage_official import ARCHIVE_SHA256, COMMIT
        receipt = read_json(receipt_path)
        if (receipt.get("origin") != ORIGIN or receipt.get("commit") != COMMIT
                or receipt.get("archive_sha256") != ARCHIVE_SHA256):
            raise RunnerError("The archive receipt is not the verified official source.")
        archive = Path(receipt["archive"])
        identity = {str(archive): ARCHIVE_SHA256,
                    str(receipt_path): file_digest(receipt_path)}
        for name, value in receipt["files"].items():
            if (root / name).is_symlink():
                raise RunnerError("Unsafe link in official source receipt.")
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise RunnerError("Unsafe path in official source receipt.")
            identity[str(path)] = value
        if not all(str(root / name) in identity for name in SOURCE_FILES):
            raise RunnerError("Official archive receipt has missing source members.")
        verify_files(identity)
        return {"origin": ORIGIN, "commit": COMMIT, "source_type": "verified_archive",
                "archive_sha256": ARCHIVE_SHA256, "files": identity}
    origin = git(root, "remote", "get-url", "origin")
    normalized = origin.removesuffix(".git").rstrip("/")
    if normalized not in ("https://" + ORIGIN, "git@" + ORIGIN.replace("/", ":", 1)):
        raise RunnerError("Use the original Lifelong-Robot-Learning/LIBERO checkout.")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RunnerError("The official LIBERO checkout has tracked modifications.")
    identity = {str(root / name): file_digest(root / name) for name in SOURCE_FILES}
    return {"origin": ORIGIN, "commit": git(root, "rev-parse", "HEAD"), "files": identity}


def read_settling_steps(source: str) -> int:
    tree = ast.parse(source)
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "evaluate_one_task_success"), None)
    if function is None:
        raise RunnerError("Official evaluate_one_task_success was not found.")
    # Anchor on the initializer loop containing env.step(dummy).
    waits = []
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        call = node.iter
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "range" and len(call.args) == 1
                and isinstance(call.args[0], ast.Constant)
                and type(call.args[0].value) is int):
            continue
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "step" and n.args
               and isinstance(n.args[0], ast.Name) and n.args[0].id == "dummy"
               for statement in node.body for n in ast.walk(statement)):
            waits.append(call.args[0].value)
    if waits != [5]:
        raise RunnerError("Official settling loop changed; review the adapter before running.")
    return waits[0]


def read_defaults(root: Path) -> dict:
    import yaml
    try:
        evaluation = yaml.safe_load((root / SOURCE_FILES[4]).read_text())
        data = yaml.safe_load((root / SOURCE_FILES[5]).read_text())
        values = {"episodes_per_task": evaluation["n_eval"],
                  "max_steps": evaluation["max_steps"],
                  "image_height": data["img_h"], "image_width": data["img_w"]}
        if any(type(v) is not int or v <= 0 for v in values.values()):
            raise ValueError
    except (KeyError, ValueError, TypeError, OSError):
        raise RunnerError("Cannot read the official evaluation/image defaults.") from None
    values["settling_steps"] = read_settling_steps((root / SOURCE_FILES[0]).read_text())
    return values


def controller_spec(env) -> dict:
    import numpy as np

    inner = env.env
    low, high = inner.action_spec
    low, high = np.asarray(low), np.asarray(high)
    if low.shape != (7,) or high.shape != (7,) or not np.isfinite([low, high]).all():
        raise RunnerError("Expected a finite 7-dimensional LIBERO action space.")
    controller = inner.robots[0].controller
    if getattr(controller, "name", None) != "OSC_POSE":
        raise RunnerError("This adapter requires the official OSC_POSE controller.")
    if not getattr(controller, "use_delta", False) or inner.control_freq != 20:
        raise RunnerError("Expected delta OSC_POSE at the official 20 Hz frequency.")
    spec = {"name": "OSC_POSE", "control_freq": 20, "use_delta": True,
            "action_low": low.tolist(), "action_high": high.tolist()}
    for key in ("input_min", "input_max", "output_min", "output_max"):
        value = np.asarray(getattr(controller, key), dtype=float)
        if not np.isfinite(value).all():
            raise RunnerError("Controller has invalid scaling.")
        spec[key] = value.tolist()
    return spec


class Runtime:
    def __init__(self, config: Config):
        self.config = config
        self.root = config.upstream_root
        self.identity = upstream_identity(self.root)
        try:
            self.defaults = read_defaults(self.root)
            import numpy as np
            import torch
        except ImportError:
            raise RunnerError("Install the isolated LIBERO environment first.") from None
        self.settings = {key: getattr(config, key) or value
                         for key, value in self.defaults.items() if key != "settling_steps"}
        self.settings["settling_steps"] = self.defaults["settling_steps"]
        # Use a project-local LIBERO config; do not change ~/.libero or other runs.
        module_root = self.root / "libero" / "libero"
        resource_paths = {
            "benchmark_root": str(module_root),
            "bddl_files": str(module_root / "bddl_files"),
            "init_states": str(module_root / "init_files"),
            "datasets": str(self.root / "datasets"),
            "assets": str(module_root / "assets"),
        }
        if not (module_root / "assets").is_dir():
            raise RunnerError("Original LIBERO simulation assets are missing.")
        config_dir = ROOT / ".runtime" / self.identity["commit"]
        write_json(config_dir / "config.yaml", resource_paths)  # JSON is valid YAML.
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
        sys.path.insert(0, str(self.root))
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        from .rendering import configure
        configure()
        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError:
            raise RunnerError("Cannot import the official LIBERO simulator dependencies.") from None
        import libero.libero
        if not Path(libero.libero.__file__).resolve().is_relative_to(self.root):
            raise RunnerError("A different installed LIBERO package shadowed the official checkout.")
        self.env_class = OffScreenRenderEnv
        self.suites = {
            name: benchmark.get_benchmark_dict()[name](task_order_index=config.task_order_index)
            for name in config.suites
        }
        self.versions = {}
        for name in ("numpy", "Pillow", "torch", "robosuite", "mujoco", "bddl", "PyYAML"):
            try:
                self.versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                self.versions[name] = None
        self.versions["render_backend"] = os.environ.get("MUJOCO_GL", "glfw")
        self.versions["osmesa_color_buffer"] = os.environ.get("LIBERO_OSMESA_BUFFER")
        render_library = os.environ.get("LIBERO_RENDER_LIBRARY")
        self.versions["render_library_sha256"] = (
            file_digest(Path(render_library)) if render_library else None
        )
        if not str(self.versions["robosuite"]).startswith("1.4."):
            raise RunnerError("Use robosuite 1.4.x; newer controller APIs require a new adapter.")

    def create_env(self, task_record: dict):
        env = self.env_class(
            bddl_file_name=task_record["bddl_file"],
            camera_heights=self.settings["image_height"],
            camera_widths=self.settings["image_width"],
            render_gpu_device_id=self.config.render_gpu_device_id,
        )
        env.seed(self.config.seed)
        return env

    def initial_states(self, suite: str, task_id: int):
        return self.suites[suite].get_task_init_states(task_id)

    def tasks(self) -> list[dict]:
        result = []
        for name, suite in self.suites.items():
            ids = self.config.task_ids or tuple(range(suite.n_tasks))
            if any(i >= suite.n_tasks for i in ids):
                raise RunnerError(f"task_ids exceed the task count for {name}.")
            for task_id in ids:
                task = suite.get_task(task_id)
                bddl = self.root / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
                init = self.root / "libero/libero/init_files" / task.problem_folder / task.init_states_file
                if not bddl.is_file() or not init.is_file():
                    raise RunnerError(f"Task resources are missing for {name}, task {task_id}.")
                states = self.initial_states(name, task_id)
                if self.settings["episodes_per_task"] > len(states):
                    raise RunnerError("Episode count exceeds the available fixed initial states.")
                result.append({
                    "suite": name, "task_id": task_id, "task_name": task.name,
                    "language": task.language, "bddl_file": str(bddl),
                    "init_states_file": str(init), "available_initial_states": len(states),
                    "files": {str(bddl): file_digest(bddl), str(init): file_digest(init)},
                })
        return result

    def probe(self, task_record: dict) -> dict:
        """Real simulator preflight with zero model requests."""
        env = self.create_env(task_record)
        try:
            env.reset()
            obs = env.set_init_state(self.initial_states(
                task_record["suite"], task_record["task_id"])[0])
            for _ in range(self.settings["settling_steps"]):
                obs, _, done, _ = env.step([0.0] * 7)
                if bool(done) != bool(env.check_success()):
                    raise RunnerError("LIBERO success/termination contract differs from the original.")
            state, images = public_observation(obs, self.config.image_orientation)
            spec = controller_spec(env)
            return {"controller": spec, "image_keys": list(images), "state_keys": list(state),
                    "model_requests": 0, "settling_actions": self.settings["settling_steps"]}
        finally:
            env.close()
