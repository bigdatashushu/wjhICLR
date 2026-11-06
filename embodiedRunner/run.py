#!/usr/bin/env python3
"""Initial EB-Navigation entry: check prerequisites, then run an isolated pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
from bridge import SimulatorBridge, simulator_environment
from planner import NavigationPlanner, PlannerConfig
from protocol import (ROOT, REVISION, VERSION, ContractError, canonical,
                      json_object, select_tasks, sha256, verified_sources, within_root)
from runner import evaluate, write_json


def load_config(path):
    value = json_object(Path(path).read_text(encoding="utf-8"))
    required = {
        "model", "mode", "seed", "max_tokens", "max_requests",
        "max_requests_per_episode", "max_actions_per_plan", "history_steps",
        "enable_thinking", "eval_sets", "limit_per_subset", "resolution",
        "simulator_python", "simulator_timeout_s", "upstream_root",
        "code_root", "skill_path", "output_dir",
    }
    if set(value) != required:
        raise ContractError("unexpected_configuration_fields")
    for name in ("model", "simulator_python", "upstream_root", "code_root",
                 "skill_path", "output_dir"):
        if not isinstance(value[name], str) or not value[name]:
            raise ContractError("invalid_configuration_path_or_model")
    if value["resolution"] not in (300, 500) or type(value["resolution"]) is not int:
        raise ContractError("unsupported_resolution")
    if type(value["max_requests"]) is not int or not 1 <= value["max_requests"] <= 100000:
        raise ContractError("invalid_run_request_budget")
    if (type(value["simulator_timeout_s"]) not in (int, float)
            or not 1 <= value["simulator_timeout_s"] <= 600):
        raise ContractError("invalid_simulator_timeout")
    if not isinstance(value["eval_sets"], list):
        raise ContractError("invalid_eval_sets")
    planner_config = PlannerConfig(**{
        name: value[name] for name in PlannerConfig.__dataclass_fields__
    })
    return value, planner_config


def resolve_read_path(path):
    value = Path(path)
    return (value if value.is_absolute() else ROOT / value).resolve()


def model_client(config):
    code_root = resolve_read_path(config["code_root"])
    if not (code_root / "skill3d/synthesis/bailian_client.py").is_file():
        raise ContractError("existing_model_client_missing")
    # Reuse the existing project or VSIbench8Brunner code through a source path.
    sys.path.insert(0, str(code_root))
    from skill3d.synthesis.bailian_client import BailianClient
    import inspect
    if Path(inspect.getfile(BailianClient)).resolve() != (
            code_root / "skill3d/synthesis/bailian_client.py").resolve():
        raise ContractError("loaded_model_client_source_mismatch")
    client = BailianClient(model=config["model"], temperature=0,
                           max_output_tokens=config["max_tokens"])
    return client


def startup(config, planner_config, *, simulator_python=None):
    upstream = resolve_read_path(config["upstream_root"])
    tasks = select_tasks(upstream, config["eval_sets"], config["limit_per_subset"])
    output = within_root(config["output_dir"])
    skill_path = resolve_read_path(config["skill_path"])
    if planner_config.mode == "skill" and not skill_path.is_file():
        raise ContractError("navigation_skill_missing")
    executable = simulator_python or config["simulator_python"]
    resolved_executable = shutil.which(executable) if not Path(executable).is_absolute() else executable
    runtime = {"runtime_ready": False, "reason_code": "simulator_python_missing"}
    if resolved_executable and Path(resolved_executable).is_file():
        try:
            checked = subprocess.run(
                [resolved_executable, "-B", str(ROOT / "simulator_worker.py"), "--probe"],
                env=simulator_environment(ROOT / "checks"),
                cwd=ROOT, capture_output=True, timeout=20, check=True,
            )
            runtime = json_object(checked.stdout.decode("utf-8"))
        except (subprocess.SubprocessError, ValueError, OSError):
            runtime = {"runtime_ready": False, "reason_code": "simulator_probe_failed"}
    client = model_client(config)
    receipt = {
        "schema": VERSION, "upstream_revision": REVISION,
        "status": "ready" if runtime["runtime_ready"] and client.configured else "prerequisites_missing",
        "model_credentials_configured": client.configured,
        "model_identity": client.audit_identity(seed=planner_config.seed),
        "simulator": runtime, "planned_tasks": len(tasks),
        "tasks": [task.public_dict() for task in tasks],
        "official_sources": verified_sources(upstream, config["eval_sets"]),
        "planner_config": asdict(planner_config),
        "skill_sha256": sha256(skill_path.read_bytes()) if planner_config.mode == "skill" else None,
        "source_sha256": {
            name: sha256((ROOT / name).read_bytes()) for name in
            ("run.py", "protocol.py", "planner.py", "runner.py", "bridge.py", "simulator_worker.py")
        },
        "existing_client_sha256": sha256((
            resolve_read_path(config["code_root"]) / "skill3d/synthesis/bailian_client.py").read_bytes()),
        "model_requests": 0, "simulator_launches": 0,
        "geometry_tools_enabled": False, "skill_evolution_enabled": False,
        "automatic_retry": False, "reference_answer_visible": False,
    }
    write_json(ROOT / "checks/startup.json", receipt)
    return tasks, output, resolved_executable, client, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--config", type=Path, default=ROOT / "navigation.json")
    parser.add_argument("--simulator-python", help="Python executable in the official navigation environment")
    parser.add_argument("--output-dir", help="New run directory inside embodiedRunner")
    args = parser.parse_args(argv)
    try:
        config, planner_config = load_config(args.config)
        if args.output_dir:
            config["output_dir"] = args.output_dir
        tasks, output, executable, client, receipt = startup(
            config, planner_config, simulator_python=args.simulator_python)
        if args.action == "check":
            print(canonical({
                key: receipt[key] for key in ("status", "planned_tasks", "simulator",
                                             "model_credentials_configured", "model_requests",
                                             "simulator_launches")
            }).decode("utf-8"))
            return 0
        if receipt["status"] != "ready":
            print("Runtime prerequisites are missing; see checks/startup.json.", file=sys.stderr)
            return 2
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "startup.json", receipt)
        write_json(output / "config.json", config)
        planner = NavigationPlanner(client, planner_config,
                                    skill_path=resolve_read_path(config["skill_path"]))
        if planner.skill_sha256 != receipt["skill_sha256"]:
            raise ContractError("skill_changed_since_startup")
        environment = SimulatorBridge(
            [executable, "-B", str(ROOT / "simulator_worker.py"),
             "--upstream", str(resolve_read_path(config["upstream_root"])),
             "--work-dir", str(output), "--resolution", str(config["resolution"])],
            work_dir=output, resolution=config["resolution"],
            timeout_s=config["simulator_timeout_s"],
        )
        try:
            summary = evaluate(tasks, planner, environment, output,
                               max_requests=config["max_requests"])
        finally:
            environment.close()
        print(canonical(summary).decode("utf-8"))
        return 0 if summary["status"] == "complete" else 2
    except Exception as exc:
        code = str(exc) if isinstance(exc, ContractError) else type(exc).__name__
        print(canonical({"status": "startup_error", "reason_code": code}).decode("utf-8"),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
