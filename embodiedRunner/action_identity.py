"""Freeze explicit code/config identities without loading tasks or credentials."""
from pathlib import Path
import platform
import sys

import numpy as np

from action_contract import plain, public_hash
from action_runtime.execution_contract import ActionExecutionContract
from action_solver import MODEL_CONFIG
from budget import policy_for
from evidence import POLICY, PARSER_SHA256
from protocol import json_object, sha256
from mutation_protocol import VERSIONS
from source_inventory import SHARED_SOURCES

ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = ("eb_navigation", "eb_alfred", "eb_habitat", "eb_manipulation")
ARMS = ("Direct", "B01", "B11")
COMPARE_ARMS = ("Direct", "Full")
LOCAL_SOURCES = (
    "action_adapters.py", "action_audit.py", "action_bridge.py", "action_contract.py",
    "action_host.py", "action_identity.py", "action_run.py", "action_solver.py", "action_live.py",
    "action_trajectory.py", "source_inventory.py",
    "action_household_v2.py", "action_assets.py", "action_alfred_data.json",
    "action_tools.py", "action_worker_v2.py", "budget.py", "evidence.py", "methods.py",
    "mutation_protocol.py", "protocol.py", "state.py", "sandbox_process.py",
    "sandbox_wire.py", "sandbox_worker.py", "sandbox_linux.py", "official_bare.py", "household_planner.py",
    "household_protocol.py", "household_worker.py", "simulator_worker.py",
    "official_sources.json", "household_sources.json", "planner.py",
    "action_runtime/__init__.py", "action_runtime/action_check.py",
    "action_runtime/ast_guard.py", "action_runtime/delivery.py",
    "action_runtime/execution_contract.py", "action_runtime/frozen_method.py",
    "action_runtime/kernel.py", "action_runtime/segment.py", "action_runtime/control_profile.py",
    "methods/manifest.json", "methods/navigation.md", "methods/alfred.md", "methods/habitat.md",
    "bridge.py", "action_live_launch.py", "action_deploy.py",
    "manipulation_protocol.py", "manipulation_planner.py", "manipulation_sources.json",
    "manipulation_worker.py", "manipulation_assets.json", "methods/manipulation.md",
    "setup_manipulation.py", "manipulation_smoke.py", "audit_action_trajectory.py",
)
HOST_CONFIGS = (
    "config/action-compare-4090.json", "config/action-compare-extra-4090.json",
    "config/action-live-4090.json", "config/action-manipulation-4090.json",
)


def source_identity():
    """Hash the shipped explicit code inventory; live config is pinned separately."""
    paths = {"embodiedRunner/" + name for name in LOCAL_SOURCES}
    paths.update(SHARED_SOURCES)
    for manifest_name, directory in (("official_sources.json", "upstream"),
                                     ("household_sources.json", "upstream-household"),
                                     ("manipulation_sources.json", "upstream-manipulation")):
        manifest = json_object((ROOT / manifest_name).read_bytes())
        paths.update("embodiedRunner/" + directory + "/" + row["path"]
                     for row in manifest["files"] if row["path"].endswith(".py"))
    hashes = {name: sha256((ROOT.parent / name).read_bytes()) for name in sorted(paths)}
    return {"schema": "embodied-source-freeze/1", "files_sha256": hashes,
            "sha256": public_hash(hashes)}


def protocol_identity():
    config = {
        "schema": "embodied-protocol-freeze/1", "versions": VERSIONS,
        "execution_contract_sha256": ActionExecutionContract().sha256,
        "manipulation_execution_contract_sha256": ActionExecutionContract(environment="eb_manipulation").sha256,
        "input_policy": "current_only", "execution_policy": "single_action",
        "automatic_retry": False, "reference_answer_visible": False,
    }
    return {**config, "sha256": public_hash(config)}


def configuration_identity(methods, evidence_config_sha256):
    config = {
        "schema": "embodied-config-freeze/1", "model": plain(MODEL_CONFIG),
        "budgets": {e: plain(policy_for(e)) for e in ENVIRONMENTS},
        "evidence_policy": plain(POLICY), "parser_sha256": PARSER_SHA256,
        "evidence_config_sha256": evidence_config_sha256,
        "methods": {e: {"content_sha256": m.content_sha256,
                        "skill_version": m.skill_id + "@" + m.version,
                        "complete_body_chars": len(m.skill_md)}
                    for e, m in methods.items()},
        "method_library_sha256": next(iter(methods.values())).manifest_sha256,
        "runtime": {"python": list(sys.version_info[:3]), "platform": platform.system(),
                    "numpy": np.__version__},
    }
    return {**config, "sha256": public_hash(config)}
