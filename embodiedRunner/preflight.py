"""Offline compatibility checks using actual official sources and local interfaces.

This does not launch simulators, call models, or estimate benchmark performance.
Run with ../.venv/bin/python -B preflight.py from this directory.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import socket
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.dont_write_bytecode = True

# Remove inherited service authentication/configuration before importing local code.
for name in tuple(os.environ):
    if any(part in name.upper() for part in
           ("TOKEN", "API_KEY", "PASSWORD", "SECRET", "CREDENTIAL", "ENDPOINT")):
        os.environ.pop(name, None)
for name in ("remote_url", "server_url", "OPENAI_BASE_URL", "DASHSCOPE_BASE_URL"):
    os.environ.pop(name, None)

connection_attempts = []


def no_connection(*_args, **_kwargs):
    connection_attempts.append("blocked")
    raise RuntimeError("Network is disabled during compatibility preflight")


socket.socket.connect = no_connection
socket.socket.connect_ex = no_connection
socket.create_connection = no_connection
sys.path.insert(0, str(PROJECT / "src"))

from skill3d.routing.task_classifier import TASK_TYPES, canonical_task
from skill3d.small_image.datasets import canonical_benchmark
from skill3d.synthesis.bailian_client import BailianClient


def syntax(path: str) -> ast.Module:
    return ast.parse((ROOT / "upstream" / path).read_text(encoding="utf-8"))


def assignment(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any((isinstance(target, ast.Name) and target.id == name)
                        or (isinstance(target, ast.Attribute) and target.attr == name)
                        for target in node.targets)):
            return ast.literal_eval(node.value)
    raise ValueError(f"Constant not found: {name}")


def method_args(tree: ast.AST, name: str) -> list[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [arg.arg for arg in node.args.args]
    raise ValueError(f"Method not found: {name}")


def rejected(function, value: str) -> bool:
    try:
        function(value)
    except ValueError:
        return True
    return False


def main() -> None:
    manifest = json.loads((ROOT / "official_sources.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        data = (ROOT / "upstream" / record["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert hashlib.sha1(
            b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        ).hexdigest() == record["git_blob_sha"]
        if record["path"].endswith(".py"):
            ast.parse(data.decode("utf-8"))

    main_tree = syntax("embodiedbench/main.py")
    official_environments = list(assignment(main_tree, "class_names"))
    environment_files = {
        "eb-alf": "eb_alfred/EBAlfEnv.py",
        "eb-hab": "eb_habitat/EBHabEnv.py",
        "eb-nav": "eb_navigation/EBNavEnv.py",
        "eb-man": "eb_manipulation/EBManEnv.py",
    }
    interfaces = {}
    for environment, suffix in environment_files.items():
        tree = syntax("embodiedbench/envs/" + suffix)
        interfaces[environment] = {
            "max_environment_steps": assignment(tree, "_max_episode_steps"),
            "reset_arguments": method_args(tree, "reset"),
            "step_arguments": method_args(tree, "step"),
            "vsi_task_type_rejected": rejected(canonical_task, environment),
            "small_image_benchmark_rejected": rejected(canonical_benchmark, environment),
        }
    assert all(item["vsi_task_type_rejected"]
               and item["small_image_benchmark_rejected"]
               for item in interfaces.values())
    navigation_tree = syntax("embodiedbench/envs/eb_navigation/EBNavEnv.py")
    navigation_actions = assignment(navigation_tree, "DISCRETE_SKILLSET")
    assert len(navigation_actions) == 8
    navigation_counts = {}
    for subset in assignment(navigation_tree, "ValidEvalSets"):
        path = ROOT / "upstream/embodiedbench/envs/eb_navigation/datasets" / f"{subset}.json"
        # No task/target/pose content is included in the resulting receipt.
        tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
        navigation_counts[subset] = len(tasks)
    assert sum(navigation_counts.values()) == 300

    standard = assignment(
        syntax("embodiedbench/planner/planner_config/generation_guide.py"),
        "vlm_generation_guide",
    )
    manipulation = assignment(
        syntax("embodiedbench/planner/planner_config/generation_guide_manip.py"),
        "vlm_generation_guide_manip",
    )
    source_paths = [
        "pyproject.toml", "src/skill3d/schemas/episode.py",
        "src/skill3d/routing/task_classifier.py",
        "src/skill3d/small_image/datasets.py", "src/skill3d/small_image/tools.py",
        "src/skill3d/sandbox/action_check.py", "src/skill3d/synthesis/bailian_client.py",
    ]
    receipt = {
        "schema": "embodiedbench-compatibility-preflight/1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "offline_contract_checks_passed",
        "upstream_revision": manifest["revision"],
        "verified_official_files": len(manifest["files"]),
        "official_source_bytes": sum(record["bytes"] for record in manifest["files"]),
        "local_source_sha256": {
            path: hashlib.sha256((PROJECT / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "python": platform.python_version(),
        "host": {"system": platform.system(), "machine": platform.machine()},
        "simulator_packages_installed": {
            name: importlib.util.find_spec(name) is not None
            for name in ("ai2thor", "habitat", "habitat_sim", "pyrep", "rlbench", "amsolver")
        },
        "official_environments": official_environments,
        "environment_interfaces": interfaces,
        "navigation_action_count": len(navigation_actions),
        "navigation_actions": navigation_actions,
        "navigation_success_threshold": assignment(navigation_tree, "SUCCESS_THRESHOLD"),
        "navigation_task_counts": navigation_counts,
        "official_plan_required_fields": standard["required"],
        "standard_action_fields": standard["properties"]["executable_plan"]["items"]["required"],
        "manipulation_action_fields":
            manipulation["properties"]["executable_plan"]["items"]["required"],
        "local_vsi_task_types": list(TASK_TYPES),
        "local_model_chat_parameters": list(inspect.signature(BailianClient.chat).parameters),
        "model_requests": 0,
        "simulator_launches": 0,
        "network_connection_attempts": len(connection_attempts),
        "benchmark_episodes_executed": 0,
        "benchmark_success_rate": None,
    }
    assert not connection_attempts
    (ROOT / "preflight.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: receipt[key] for key in (
            "status", "verified_official_files", "python", "host",
            "simulator_packages_installed", "navigation_task_counts",
            "environment_interfaces", "model_requests", "simulator_launches",
            "network_connection_attempts", "benchmark_success_rate")
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
