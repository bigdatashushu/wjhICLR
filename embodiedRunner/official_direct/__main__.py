"""python -m official_direct check|probe|run"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "official_direct"

from .client import ModelClient
from .config import load_config
from .planner import OfficialPlanner
from .runner import suite
from .sources import ContractError, LIMITS, OperationalError, ROOT, Sources, canonical
from .transport import Session, preflight


def check(config, tasks, sources):
    counts = sources.verify_all()
    for task in tasks:
        # Real action spaces are read after official reset. Offline compilation
        # uses placeholders solely to exercise all four prompt constructors.
        actions = ["offline placeholder"] * (8 if task.environment == "navigation" else 2)
        OfficialPlanner(sources, task, config["model"]["name"], config["n_shots"][task.environment], actions,
                        config["model"]["response_format"])
        source_kind = task.environment
        if source_kind == "navigation":
            name = f"embodiedbench/envs/eb_navigation/datasets/{task.subset}.json"
            data = json.loads(sources.read(source_kind, name))
            if task.index >= len(data["tasks"]):
                raise ContractError("task_index_out_of_range")
        elif source_kind == "alfred":
            data = json.loads(sources.read(source_kind, "embodiedbench/envs/eb_alfred/data/splits/splits.json"))
            if task.subset not in data or task.index >= len(data[task.subset]):
                raise ContractError("task_index_out_of_range")
        elif source_kind == "habitat":
            sources.read(source_kind, f"embodiedbench/envs/eb_habitat/datasets/{task.subset}.pickle")
        else:
            catalog = json.loads(sources.manipulation_catalog.read_text())
            if not any(r["subset"] == task.subset and r["index"] == task.index for r in catalog["selection"]):
                raise ContractError("manipulation_task_not_in_asset_manifest")
    return {"status": "offline_ready", "official_revision": "9be4e980e9cd6bcb38373cd4aab7c32724bdd401",
            "verified_official_files": counts, "selected_tasks": [t.as_dict() for t in tasks],
            "official_action_limits": LIMITS, "simulator_launches": 0, "model_requests": 0,
            "network_requests": 0, "remote_runtime_verified": False}


def main():
    parser = argparse.ArgumentParser(description="Four-environment official direct evaluation")
    parser.add_argument("action", choices=("check", "probe", "run"))
    parser.add_argument("--config", type=Path, default=ROOT / "official_direct/default.json")
    parser.add_argument("--environment", choices=tuple(LIMITS), help="Filter configured episodes")
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Local verified official snapshots")
    parser.add_argument("--manipulation-catalog", type=Path, help="Independent verified task asset catalog")
    parser.add_argument("--run-id", help="Unique run directory name; existing runs cannot be overwritten")
    args = parser.parse_args()
    try:
        config, tasks = load_config(args.config)
        if args.environment:
            tasks = [t for t in tasks if t.environment == args.environment]
            if not tasks:
                raise ContractError("no_selected_episode_for_environment")
        sources = Sources(args.source_root, args.manipulation_catalog)
        report = check(config, tasks, sources)
        if args.action == "check":
            print(canonical(report).decode())
            return 0
        if args.action == "probe":
            reports = {}
            for task in tasks:
                reports[task.identity] = preflight(config, task, sources)
            print(canonical({"status": "runtime_ready", "reports": reports,
                             "simulator_launches": 0, "model_requests": 0}).decode())
            return 0
        client = ModelClient(config["model"])
        client.assert_available()
        run_id = args.run_id or ("direct-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                + "-" + uuid.uuid4().hex[:8])
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id):
            raise ContractError("invalid_run_identity")
        directory = ROOT / "official_direct/runs" / run_id
        def factory(task, local_dir, journal):
            return Session(config, task, local_dir, journal, sources, run_id)
        result = suite(tasks, config, sources, client, factory, directory)
        print(canonical({"run_directory": str(directory), **result}).decode())
        return 2 if result["status"] != "finished" else 0
    except (ContractError, OperationalError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, OperationalError) else "local_configuration_or_source_error"
        print(canonical({"status": "failed", "error_code": code, "exception_type": type(exc).__name__}).decode())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
