"""Read-only, sanitized runtime discovery; no model or simulator launches."""
from __future__ import annotations

import getpass
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess

user = getpass.getuser()
home_directory = Path("/home") / user
result = {
    "hostname": platform.node(), "user": user,
    "project_parent_exists": (home_directory / "wangjh").is_dir(),
    "python_version": platform.python_version(),
    "commands": {name: shutil.which(name) for name in
                 ("python3", "conda", "uv", "vulkaninfo", "Xvfb", "nvidia-smi")},
    "python_candidates": [
        str(path) for directory in
        (home_directory / "anaconda3/envs", home_directory / "miniconda3/envs")
        if directory.is_dir() for path in directory.glob("*/bin/python")
    ],
    "current_packages": {name: importlib.util.find_spec(name) is not None
                         for name in ("ai2thor", "gym", "numpy", "PIL", "cv2")},
    "model_requests": 0, "simulator_launches": 0,
}
for command, key in [
    (["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
      "--format=csv,noheader,nounits"], "gpus"),
    (["ldconfig", "-p"], "graphics_libraries"),
]:
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=10)
        result[key] = (process.stdout.splitlines() if key == "gpus" else
                       [line.strip() for line in process.stdout.splitlines()
                        if any(name in line for name in ("libvulkan", "libGLX", "libEGL", "libX11"))])
    except (OSError, subprocess.TimeoutExpired):
        result[key] = []
print(json.dumps(result, sort_keys=True))
