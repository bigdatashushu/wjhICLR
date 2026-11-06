#!/usr/bin/env python3
"""Run a bounded official-protocol Qwen-only pilot with a remote simulator."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
from bridge import SimulatorBridge, simulator_environment
from official_bare import BareConfig, OfficialBarePlanner, DEVIATIONS
from protocol import ROOT, REVISION, ContractError, canonical, json_object, select_tasks, sha256, within_root
from run import model_client
from runner import evaluate, write_json


def ssh_command(config, action, output):
    remote_root = config["remote_root"]
    if remote_root != "/home/cvailab/wangjh/embodiedRunner":
        raise ContractError("unexpected_remote_root")
    if config["remote_host"] != "cvailab@10.10.0.198" or config["gpu_device"] != 3:
        raise ContractError("unexpected_remote_resource")
    remote = [
        remote_root + "/venv-nav/bin/python", "-B",
        remote_root + "/remote_runtime.py", action,
        "--work-dir", str(output.relative_to(ROOT)),
        "--gpu-device", str(config["gpu_device"]),
    ]
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ControlMaster=no",
            "-o", "ConnectTimeout=10", "-S", str(ROOT / "checks/ssh4090"),
            config["remote_host"], shlex.join(remote)]


def load_authorized_key():
    """Read the already-authorized local key only into this process environment."""
    if os.environ.get("DASHSCOPE_API_KEY", "").strip():
        return
    source = ROOT.parent / "qwenkey.md"
    if source.stat().st_mode & 0o077:
        raise ContractError("existing_key_file_permissions_not_private")
    text = source.read_text()
    keys = re.findall(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_.+/-]{16,}", text)
    if len(set(keys)) != 1:
        raise ContractError("existing_model_key_not_unique")
    os.environ["DASHSCOPE_API_KEY"] = keys[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--config", type=Path, default=ROOT / "bare_navigation.json")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    try:
        config = json_object(args.config.read_text())
        if args.output_dir:
            config["output_dir"] = args.output_dir
        expected = set(BareConfig.__dataclass_fields__) | {
            "model", "max_requests", "eval_sets", "limit_per_subset", "resolution",
            "code_root", "remote_host", "remote_root", "simulator_timeout_s",
            "gpu_device", "output_dir",
        }
        if set(config) != expected or config["model"] != "qwen3.8-flash" or config["resolution"] != 500:
            raise ContractError("unexpected_bare_configuration")
        if type(config["max_requests"]) is not int or not 1 <= config["max_requests"] <= 60:
            raise ContractError("invalid_pilot_request_budget")
        if type(config["simulator_timeout_s"]) not in (int, float) or not 1 <= config["simulator_timeout_s"] <= 600:
            raise ContractError("invalid_simulator_timeout")
        planner_config = BareConfig(**{key: config[key] for key in BareConfig.__dataclass_fields__})
        tasks = select_tasks(ROOT / "upstream", config["eval_sets"], config["limit_per_subset"])
        if len(tasks) > 3:
            raise ContractError("pilot_task_limit_exceeded")
        output = within_root(config["output_dir"])
        command = ssh_command(config, "probe", output)
        checked = subprocess.run(command, cwd=ROOT, env=simulator_environment(ROOT / "checks"),
                                 capture_output=True, timeout=30, check=True)
        runtime = json_object(checked.stdout.decode())
        if any(sha256((ROOT / name).read_bytes()) != digest
               for name, digest in runtime["source_sha256"].items()):
            raise ContractError("remote_simulator_source_mismatch")
        load_authorized_key()
        client = model_client(config)
        planner = OfficialBarePlanner(client, planner_config)
        source_names = [
            "bare_run.py", "official_bare.py", "bare_navigation.json",
            "protocol.py", "runner.py", "bridge.py", "simulator_worker.py", "remote_runtime.py",
        ]
        receipt = {
            "status": "ready" if runtime["runtime_ready"] and client.configured else "prerequisites_missing",
            "upstream_revision": REVISION, "tasks": [task.public_dict() for task in tasks],
            "planned_tasks": len(tasks), "simulator": runtime,
            "model_identity": client.audit_identity(seed=planner_config.seed),
            "model_credentials_configured": client.configured,
            "planner_config": asdict(planner_config),
            "official_prompt_hashes": planner.official_prompt_hashes,
            "source_sha256": {name: sha256((ROOT / name).read_bytes()) for name in source_names},
            "existing_client_sha256": sha256((ROOT.parent / "src/skill3d/synthesis/bailian_client.py").read_bytes()),
            "model_requests": 0, "simulator_launches": 0, "skill_sha256": None,
            "geometry_tools_enabled": False, "skill_evolution_enabled": False,
            "model_tool_calls": 0, "reference_answer_visible": False,
            "automatic_retry": False, "uses_official_prompt": True,
            "uses_official_messages": True, "protocol_deviations": DEVIATIONS,
        }
        write_json(ROOT / "checks/bare-startup.json", receipt)
        if args.action == "check":
            print(canonical({key: receipt[key] for key in (
                "status", "planned_tasks", "simulator", "model_credentials_configured",
                "model_requests", "simulator_launches")}).decode())
            return 0
        if receipt["status"] != "ready":
            raise ContractError("bare_prerequisites_missing")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "startup.json", receipt)
        write_json(output / "config.json", config)
        environment = SimulatorBridge(
            ssh_command(config, "worker", output), work_dir=output,
            resolution=500, timeout_s=config["simulator_timeout_s"],
        )
        try:
            summary = evaluate(tasks, planner, environment, output, max_requests=config["max_requests"])
        finally:
            environment.close()
        summary.update(
            requested_model=config["model"], n_shots=planner_config.n_shots,
            max_tokens=planner_config.max_tokens, temperature=0,
            enable_thinking=False, model_tool_calls=0, upstream_revision=REVISION,
        )
        write_json(output / "summary.json", summary)
        print(canonical(summary).decode())
        return 0 if summary["status"] == "complete" else 2
    except Exception as exc:
        code = str(exc) if isinstance(exc, ContractError) else type(exc).__name__
        print(canonical({"status": "startup_error", "reason_code": code}).decode(), file=sys.stderr)
        return 2
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
