"""Credential-free remote simulator entry and real, model-free rendering check."""
from __future__ import annotations

import argparse
import base64
import contextlib
import os
from pathlib import Path
import socket
import sys
import traceback

sys.dont_write_bytecode = True
from bridge import simulator_environment
from protocol import ROOT, canonical, select_tasks, sha256, within_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("probe", "smoke", "worker"))
    parser.add_argument("--work-dir", default="checks/render-smoke")
    parser.add_argument("--gpu-device", type=int, default=3)
    args = parser.parse_args()
    environment = simulator_environment(ROOT)
    for name in tuple(environment):
        if "PROXY" in name.upper():
            environment.pop(name)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    environment["PATH"] = str(ROOT / "runtime/vulkan-tools/usr/bin") + os.pathsep + environment.get("PATH", "")
    # gpu_device uses physical CUDA numbering when CUDA_VISIBLE_DEVICES is absent.
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.clear()
    os.environ.update(environment)
    original_addresses = socket.getaddrinfo

    def ipv4_addresses(host, port, family=0, *parameters, **options):
        return original_addresses(host, port, socket.AF_INET if family == 0 else family,
                                  *parameters, **options)

    socket.getaddrinfo = ipv4_addresses
    from simulator_worker import OfficialNavigation, probe
    if args.action == "probe":
        receipt = probe()
        receipt["source_sha256"] = {
            name: sha256((ROOT / name).read_bytes()) for name in
            ("bridge.py", "protocol.py", "simulator_worker.py", "remote_runtime.py", "official_sources.json")
        }
        print(canonical(receipt).decode())
        return 0
    work_dir = within_root(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "worker":
        os.execve(str(ROOT / "venv-nav/bin/python"), [
            str(ROOT / "venv-nav/bin/python"), "-B", str(ROOT / "simulator_worker.py"),
            "--work-dir", str(work_dir), "--gpu-device", str(args.gpu_device),
            "--resolution", "500",
        ], environment)
    session = OfficialNavigation(ROOT / "upstream", work_dir, gpu_device=args.gpu_device)
    receipt = {"check": "real_ai2thor_reset_step", "model_requests": 0,
               "gpu_device": args.gpu_device, "reference_answer_visible": False}
    try:
        if not probe()["runtime_ready"]:
            raise RuntimeError("simulator dependencies unavailable")
        task = select_tasks(ROOT / "upstream", ["base"], 1)[0]
        with (work_dir / "runtime.log").open("a") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                first = session.open(task.subset, task.index, task.input_sha256)
                (work_dir / "initial.png").write_bytes(base64.b64decode(first["image_base64"]))
                second = session.step(4, True)
        receipt.update(
            status="ready", initial_step=first["step"], final_step=second["step"],
            initial_image_sha256=first["image_sha256"],
            final_image_sha256=second["image_sha256"],
            action_success=second["last_action_success"],
            image_changed=first["image_sha256"] != second["image_sha256"],
        )
    except Exception as exc:
        receipt.update(
            status="failed", exception_type=type(exc).__name__,
            exception_module=type(exc).__module__,
            stack=[{"file": Path(frame.filename).name, "line": frame.lineno,
                    "function": frame.name}
                   for frame in traceback.extract_tb(exc.__traceback__)],
        )
    finally:
        session.close()
    (work_dir / "receipt.json").write_bytes(canonical(receipt) + b"\n")
    print(canonical(receipt).decode())
    return 0 if receipt["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
