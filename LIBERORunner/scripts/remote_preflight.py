"""Run a fresh, zero-request LIBERO simulator preflight on the configured host."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import uuid


def render_environment(task_root, settings):
    backend = settings.get("render_backend", "egl")
    if backend == "egl":
        return "MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1", 1
    if backend != "osmesa":
        raise ValueError("Use egl or osmesa in the local settings.")
    library_root = task_root / "runtime/osmesa/usr/lib/x86_64-linux-gnu"
    return (
        "MUJOCO_GL=osmesa LIBERO_OSMESA_BUFFER=uint8 "
        f"LIBERO_RENDER_LIBRARY={shlex.quote(str(library_root / 'libOSMesa.so.8'))} "
        f"LD_LIBRARY_PATH={shlex.quote(str(library_root))}:\"${{LD_LIBRARY_PATH:-}}\"",
        -1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    task_root = PurePosixPath(settings["remote_root"])
    if not task_root.is_absolute():
        raise ValueError("remote_root must be absolute.")
    source = task_root / "source"
    output = task_root / "runs" / ("preflight-" + uuid.uuid4().hex)
    quote = shlex.quote
    render_env, device = render_environment(task_root, settings)
    command = (
        f"env {render_env} OMP_NUM_THREADS=2 "
        "OPENBLAS_NUM_THREADS=2 "
        f"PYTHONPATH={quote(str(source))} "
        f"{quote(str(task_root / 'runtime/venv/bin/python'))} -m LIBERORunner prepare "
        f"--config {quote(str(source / 'LIBERORunner/configs/gpt6.1terra.json'))} "
        f"--libero-root {quote(str(task_root / 'official'))} "
        "--suites libero_spatial --task-ids 0 --episodes 1 "
        f"--max-steps 3 --max-requests 3 --render-gpu-device-id {device} "
        f"--output {quote(str(output))}"
    )
    ssh = ["ssh", "-o", "BatchMode=yes"]
    if settings.get("ssh_config"):
        ssh += ["-F", str(Path(settings["ssh_config"]).expanduser().resolve())]
    ssh += [settings["host"], command]
    return subprocess.call(ssh)


if __name__ == "__main__":
    raise SystemExit(main())
