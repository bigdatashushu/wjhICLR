"""Opt-in V2 runner with offline checks and explicitly configured real execution."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

import numpy as np

from action_adapters import FakeAdapter
from action_audit import audit_run, seal_record
from action_contract import plain, public_hash
from action_identity import configuration_identity, protocol_identity, source_identity
from action_bridge import ActionBridge, LocalTransport, NDJSONTransport
from action_host import ActionHost
from action_solver import ActionSolver, MODEL_CONFIG
from budget import RunBudget, policy_for
from evidence import CurrentEvidencePreparer, PerceptionBackends, POLICY, PARSER_SHA256
from methods import load_methods
from mutation_protocol import MutationSession
from protocol import ContractError, OperationalError, canonical, json_object, sha256

ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = ("eb_navigation", "eb_alfred", "eb_habitat", "eb_manipulation")
ARMS = ("Direct", "B01", "B11")
FAKE_SOURCE = """def solve(ctx):
    r = point_depth(0, 250, 250)
    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,
        ctx.snapshot_sha256, ctx.action_table_sha256, [0, 1], [r["result_id"]]))
"""


class FakeClient:
    """Exercises real prompt/assembly with no network or model loading."""
    offline_fake = True
    def __init__(self, arm):
        self.arm, self.calls = arm, 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        self.last_usage = {"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16}
        self.last_response_identity = {"model": "offline-fake"}
        text = canonical(messages).decode()
        manip = "Franka Panda" in text or "eb_manipulation" in text
        actions = [[50, 50, 30, 120, 60, 90, 1], [50, 50, 12, 120, 60, 90, 0]]
        plan = [{"action": a} for a in actions] if manip else [{"action_id": 0}, {"action_id": 1}]
        program = FAKE_SOURCE.replace("[0, 1]", repr(actions)) if manip else FAKE_SOURCE
        if self.arm != "Direct" and "embodied-task-card/" in text:
            card = json_object(messages[1]["content"][0]["text"])
            goal = card["subgoal"] or {"id": "main", "goal": "Advance the public task",
                                       "object": "", "destination": "", "complete_when": "visual"}
            result = {"subgoal": goal,
                      "transition": {"kind": "keep", "source": "none",
                                     "observation_revision": card["identity"]["observation_revision"],
                                     "detail": ""}}
            if card["evidence"]["status"] == "not_requested":
                result["perception_request"] = {"targets": [], "depth": True}
            else:
                result["program_source"] = program
            return canonical(result).decode()
        return canonical({"executable_plan": plan} if self.arm == "Direct" else
                         {"program_source": program}).decode()


class FakeDepth:
    identity = {"backend": "offline-fake-depth", "version": "1"}
    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        return {"depth": np.ones((h, w), dtype=np.float32),
                "valid_mask": np.ones((h, w), dtype=bool)}


def check():
    methods = load_methods()
    return {
        "schema": "embodied-offline-check/1", "status": "ready_for_fake",
        "implementation_root": str(ROOT), "model_config": plain(MODEL_CONFIG),
        "model_config_sha256": public_hash(MODEL_CONFIG),
        "evidence_policy": plain(POLICY), "parser_sha256": PARSER_SHA256,
        "methods": {environment: {
            "method_id": method.skill_id, "version": method.version,
            "content_sha256": method.content_sha256, "manifest_sha256": method.manifest_sha256,
            "complete_body_chars": len(method.skill_md),
        } for environment, method in methods.items()},
        "budgets": {environment: plain(policy_for(environment)) for environment in ENVIRONMENTS},
        "live_requests": 0, "simulator_launches": 0,
        "input_policy": "current_only", "execution_policy": "single_action",
        "official_execution_enabled": False,
        "implementation_source": source_identity(), "protocol": protocol_identity(),
    }


def fake_run(*, process_worker=False, run_id=None, environments=ENVIRONMENTS,
             client_factory=FakeClient, transport_factory=None, preparer_factory=None,
             task_cards=False, trajectory_dir=None):
    methods = load_methods()
    run_id = run_id or "offline-" + uuid.uuid4().hex
    selection = [{
        "episode_id": environment + ":base:000", "environment": environment,
        "subset": "base", "index": 0,
        "source_sha256": public_hash({"offline_fake": True, "environment": environment,
                                      "subset": "base", "index": 0}),
    } for environment in environments]
    identity_preparer = (preparer_factory(environments[0]) if preparer_factory else
                         CurrentEvidencePreparer(PerceptionBackends(depth_estimator=FakeDepth())))
    manifest = {
        "schema": "embodied-paired-run/1", "run_id": run_id,
        "arms": list(ARMS), "selection": selection, "denominator_per_arm": len(selection),
        "model_config_sha256": public_hash(MODEL_CONFIG),
        "budget_sha256_by_environment": {e: public_hash(policy_for(e)) for e in environments},
        "method_sha256_by_environment": {e: methods[e].content_sha256 for e in environments},
        "method_library_sha256": next(iter(methods.values())).manifest_sha256,
        "evidence_config_sha256": identity_preparer.config_sha256,
        "input_policy": "current_only", "execution_policy": "single_action",
        "offline_fake": True, "automatic_retry": False, "reference_answer_visible": False,
        "pairing": "same public task/source and policy; action trajectories may diverge",
        "task_cards": task_cards,
    }
    manifest.update(
        implementation_source=source_identity(), protocol=protocol_identity(),
        configuration=configuration_identity(methods, manifest["evidence_config_sha256"]))
    manifest["sha256"] = public_hash(manifest)
    trajectories = []
    if trajectory_dir is not None:
        from action_live import write_new
        trajectory_dir = Path(trajectory_dir)
        trajectory_dir.mkdir(mode=0o700, exist_ok=False)
        write_new(trajectory_dir / "manifest.json", manifest)
    reports, budgets = [], {}
    for arm in ARMS:
        budget = RunBudget(run_id + ":" + arm, environments)
        for item in selection:
            environment = item["environment"]
            episode_budget = budget.episode(item["episode_id"], environment)
            transport, solver = None, None
            method = methods[environment] if arm == "B11" else None
            trajectory = None
            if trajectory_dir is not None:
                from action_trajectory import Trajectory
                trajectory = Trajectory(
                    trajectory_dir / f"trajectory-{selection.index(item):03d}-{arm}",
                    episode_id=item["episode_id"], arm=arm)
            try:
                if transport_factory:
                    transport = transport_factory(environment)
                elif process_worker:
                    transport = NDJSONTransport([
                        sys.executable, "-B", str(ROOT / "action_worker_v2.py"),
                        "--environment", environment, "--fake", "--work-dir", str(ROOT / "tmp"),
                    ], work_dir=ROOT / "tmp", environment=environment)
                else:
                    transport = LocalTransport(MutationSession(FakeAdapter(environment)))
                bridge = ActionBridge(transport, environment=environment, episode_id=item["episode_id"],
                                      budget=episode_budget)
                solver = ActionSolver(client_factory(arm), arm=arm, environment=environment, method=method,
                                      task_cards=task_cards, trajectory=trajectory)
                preparer = (None if arm == "Direct" else
                            preparer_factory(environment) if preparer_factory else
                            CurrentEvidencePreparer(PerceptionBackends(depth_estimator=FakeDepth())))
                host = ActionHost(bridge, solver, run_id=budget.run_id, arm=arm,
                                  evidence_preparer=preparer,
                                  method_sha256=method.content_sha256 if method else "", trajectory=trajectory)
                if preparer is not None:
                    preparer.trajectory = trajectory
                report = host.run(subset=item["subset"], index=item["index"],
                                  source_sha256=item["source_sha256"])
            except Exception:
                report = startup_failure_report(budget.run_id, item, arm, episode_budget,
                                                method.content_sha256 if method else "")
            finally:
                if transport:
                    transport.close()
            report.update(model_requests=solver.receipts if solver else [],
                          paired_manifest_sha256=manifest["sha256"],
                          source_sha256=item["source_sha256"], offline_fake=True,
                          implementation_source_sha256=manifest["implementation_source"]["sha256"],
                          protocol_sha256=manifest["protocol"]["sha256"],
                          prepared_config_sha256=manifest["configuration"]["sha256"])
            reports.append(seal_record(report))
            if trajectory is not None:
                trajectories.append({"episode_id": item["episode_id"], "arm": arm,
                                     "directory": trajectory.directory.name,
                                     "trajectory": trajectory.finish(report)})
        budgets[arm] = budget.summary()
    result = seal_record({
        "schema": "embodied-offline-run/1", "manifest": manifest,
        "attempts": reports, "run_budgets": budgets,
        "live_requests": 0, "official_simulator_launches": 0, "deployments": 0,
        "fake_transport": "ndjson_process" if process_worker else "local",
        "status": "passed" if all(r["terminal_reason"] == "official_done" for r in reports) else "failed",
    })
    result["audit"] = audit_run(result)
    if trajectory_dir is not None:
        write_new(trajectory_dir / "trajectories.json", {
            "schema": "embodied-comparison-trajectories/1", "run_sha256": result["sha256"],
            "manifest": manifest, "episodes": trajectories})
    return result


def startup_failure_report(run_id, item, arm, budget, method_sha256):
    return {
        "schema": "embodied-action-attempt/1", "run_id": run_id,
        "episode_id": item["episode_id"], "environment": item["environment"],
        "arm": arm, "split_role": "development", "status": "aborted",
        "terminal_reason": "endpoint_startup_failure", "task_success": False,
        "task_progress": 0.0, "environment_steps": 0, "mutation_revision": 0,
        "invalid_actions": 0, "observation_revision": -1, "method_sha256": method_sha256,
        "budget": budget.summary(), "transitions": [], "mutations": [],
        "decisions": [], "executions": [], "errors": [], "observations": [],
        "revoked_decisions": [], "reference_answer_visible": False,
        "input_policy": "current_only", "execution_policy": "single_action", "automatic_retry": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--fake", action="store_true")
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--compare", action="store_true")
    mode.add_argument("--live-check", action="store_true")
    parser.add_argument("--live-config", type=Path)
    parser.add_argument("--process-worker", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task-cards", action="store_true",
                        help="exercise the bounded task-card protocol in offline fake mode")
    args = parser.parse_args()
    if args.task_cards and not args.fake:
        parser.error("--task-cards is for offline --fake; live mode selects it in its config")
    if args.output:
        from action_live import private_output
        try:
            private_output(args.output)
        except ContractError as error:
            parser.error(str(error))
    if args.live or args.live_check or args.compare:
        if not args.live_config or args.process_worker or ((args.live or args.compare) and not args.output):
            parser.error("live modes require --live-config; --live also requires --output")
        from action_live import live_check, live_run
        try:
            result = (live_run(args.live_config, args.output, compare=args.compare) if args.live or args.compare else
                      live_check(args.live_config, ROOT / "tmp"))
        except Exception as error:
            import re
            code = str(error) if isinstance(error, (ContractError, OperationalError)) else ""
            if not re.fullmatch(r"[a-z_]{1,100}", code):
                code = "live_preflight_" + type(error).__name__.lower()
            print(canonical({"schema": "embodied-live-preflight/1", "status": "rejected",
                             "reason": code, "live_requests": 0}).decode())
            raise SystemExit(1) from None
    elif args.check:
        result = check()
    else:
        result = fake_run(process_worker=args.process_worker, task_cards=args.task_cards,
                          trajectory_dir=args.output.with_name(args.output.stem + "-episodes")
                          if args.task_cards and args.output else None)
    if args.output and not (args.live or args.compare):
        path = args.output.resolve()
        if (not path.is_relative_to(ROOT) or not path.parent.is_dir() or path.exists()):
            parser.error("output must be a new file in an existing embodiedRunner directory")
        with path.open("xb") as stream:
            stream.write(canonical(result) + b"\n")
    print(canonical({
        "schema": result["schema"], "status": result["status"],
        "live_requests": result.get("live_requests", 0),
        "official_simulator_launches": result.get("official_simulator_launches", 0),
        "attempts": len(result.get("attempts", ())),
        "audit_status": result.get("audit", {}).get("status"),
        "output": str(args.output.resolve()) if args.output else None,
    }).decode())
    if result["status"] == "failed" or result.get("audit", {}).get("status", "passed") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
