#!/usr/bin/env python3
"""Run one label-free Skill episode on a fresh process's main thread."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

if not __package__:
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"),
                   str(Path(__file__).resolve().parent.parent)]
    __package__ = "CV3D"

from skill3d.small_image.cv3d_protocol import FrozenCV3DSkills
from skill3d.small_image.datasets import SmallImageProblem
from skill3d.small_image.runner import SmallImageRunnerConfig
from skill3d.synthesis.request_governor import RequestGovernor, RequestGovernorConfig

from .common import InferenceConfig, digest, file_sha256, json_bytes, read_json, write_json
from .data import PublicInput
from .governor import SharedSlots
from .program import ProgramClient, solve
from .transport import ProxyOpener

SCHEMA = "cv3d-main-thread-episode/1"


class RunStop:
    """Combine scheduler notification with stop admission across all child processes."""

    def __init__(self, path):
        self.path = Path(path)
        self.event = threading.Event()

    def is_set(self):
        return self.event.is_set() or self.path.exists()

    def set(self):
        self.event.set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab"):
            pass


def validate_source(source):
    root = Path(__file__).resolve().parents[2]
    if "tests/CV3D/program_worker.py" not in source:
        raise ValueError("worker source pin missing")
    for name, expected in source.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or file_sha256(root / path) != expected:
            raise ValueError("frozen worker source changed")


class ProcessSolver:
    def __init__(self, config, manifest, runner, frozen):
        self.config = config
        self.manifest = manifest
        self.runner = runner
        self.frozen = frozen

    def __call__(self, public, prepared_directory, *, arm, client, governor,
                 output_directory, stop_event, expected_identity):
        if arm == "direct":
            return solve(
                public, prepared_directory, arm=arm, client=client, governor=governor,
                output_directory=output_directory, stop_event=stop_event,
                expected_identity=expected_identity, frozen=self.frozen, config=self.runner)
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        job = {
            "schema": SCHEMA, "public": public.problem.public_dict(),
            "official_prompt": public.official_prompt, "expected_identity": expected_identity,
            "prepared_directory": str(prepared_directory), "output_directory": str(directory),
            "runner": asdict(self.runner),
            "model": InferenceConfig(timeout_s=self.config["timeout_s"]).public_dict(),
            "skills_manifest": self.config["skills_manifest"],
            "transport": {k: self.config[k] for k in ("gateway", "auth_module_root")},
            "run_identity": digest(self.manifest), "source": self.manifest["source"],
            "budget_root": str(governor.governor.root),
            "budget_policy": asdict(governor.governor.config),
            "reserve_direct": governor.reserve_direct, "stop_file": str(stop_event.path),
        }
        # No clients, credentials, answer map, or host scoring paths cross this boundary.
        command = [sys.executable, "-B", str(Path(__file__).resolve())]
        environment = dict(os.environ)
        for name in tuple(environment):
            if any(word in name.upper() for word in ("API_KEY", "TOKEN", "ENDPOINT")):
                environment.pop(name)
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env=environment)
        write_json(directory / "process.json", {
            "pid": process.pid, "job_sha256": digest(job),
            "run_identity": job["run_identity"], "source_sha256": digest(job["source"]),
            "main_thread_kernel": True, "automatic_retry": False,
        }, exclusive=True)
        process.communicate(json_bytes(job))
        if process.returncode:
            stop_event.set()
            raise RuntimeError("Skill episode process failed")
        row = read_json(directory / "result.json")
        if (row["input_identity"] != expected_identity or row["receipt_sha256"] != digest(
                {k: v for k, v in row.items() if k != "receipt_sha256"})):
            stop_event.set()
            raise ValueError("worker episode receipt changed")
        return row


def execute(job):
    if job["schema"] != SCHEMA:
        raise ValueError("unexpected worker schema")
    validate_source(job["source"])
    raw = job["public"]
    fields = {"benchmark", "sample_id", "source_split", "question", "options",
              "image_paths", "task_type", "input_sha256"}
    if set(raw) != fields:
        raise ValueError("worker public fields are not label-free")
    problem = SmallImageProblem(
        **{k: tuple(v) if k in {"options", "image_paths"} else v
           for k, v in raw.items() if k != "input_sha256"})
    if problem.public_dict() != raw:
        raise ValueError("worker public input identity changed")
    public = PublicInput(problem, job["official_prompt"])
    stop = RunStop(job["stop_file"])
    governor = SharedSlots(
        RequestGovernor(job["budget_root"], RequestGovernorConfig(**job["budget_policy"])),
        reserve_direct=job["reserve_direct"], max_slots=8,
        stop_event=stop, stop_on_request_error=True)
    sys.path.insert(0, job["transport"]["auth_module_root"])
    from transport_auth import load_transport_token
    os.environ["CV3D_PROXY_PLACEHOLDER"] = "local-proxy-configuration"
    client = ProgramClient(
        InferenceConfig(**job["model"]), api_key_env="CV3D_PROXY_PLACEHOLDER",
        opener=ProxyOpener(
            gateway=job["transport"]["gateway"], secret_loader=load_transport_token,
            operation_id=digest({"arm": "tools", "input": problem.input_sha256}),
            run_identity=job["run_identity"]))
    return solve(
        public, job["prepared_directory"], arm="tools", client=client, governor=governor,
        output_directory=job["output_directory"],
        frozen=FrozenCV3DSkills(job["skills_manifest"]),
        config=SmallImageRunnerConfig(**job["runner"]), stop_event=stop,
        expected_identity=job["expected_identity"])


def main():
    job = json.load(sys.stdin)
    try:
        execute(job)
        return 0
    except BaseException as exc:
        RunStop(job["stop_file"]).set()
        write_json(Path(job["output_directory"]) / "worker_failure.json", {
            "error_type": type(exc).__name__, "automatic_retry": False,
            "locations": [{"file": Path(frame.filename).name, "line": frame.lineno}
                          for frame in traceback.extract_tb(exc.__traceback__)],
        })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
