"""Run three real Qwen-only episodes using pinned official household protocols."""
from __future__ import annotations

import argparse
import base64
import os
import shlex
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from bare_run import load_authorized_key
from bridge import SimulatorBridge, simulator_environment
from household_planner import DEVIATIONS, OfficialHouseholdPlanner
from household_protocol import VERSION, SUBSETS, decode, source
from protocol import ROOT, REVISION, ContractError, OperationalError, canonical, json_object, sha256, within_root
from run import model_client
from runner import write_json


def ssh(action, environment, subset, output):
    remote = ["/home/cvailab/wangjh/embodiedRunner/venv-household/bin/python", "-B",
              "/home/cvailab/wangjh/embodiedRunner/household_worker.py", action,
              "--environment", environment, "--subset", subset,
              "--work-dir", str(output.relative_to(ROOT))]
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ControlMaster=no",
            "-o", "ConnectTimeout=10", "-S", str(ROOT / "checks/ssh4090"),
            "cvailab@10.10.0.198", shlex.join(remote)]


class HouseholdBridge(SimulatorBridge):
    def request(self, op, **parameters):
        if self.closed:
            raise OperationalError("simulator_closed")
        self.sequence += 1
        try:
            self.process.stdin.write(canonical({"id": self.sequence, "op": op, **parameters}) + b"\n")
            self.process.stdin.flush()
            value = json_object(self._line())
        except (OSError, ValueError):
            raise OperationalError("simulator_protocol_error") from None
        if value.get("id") != self.sequence or value.get("version") != VERSION:
            raise OperationalError("simulator_response_identity_mismatch")
        if value.get("ok") is not True:
            code = value.get("error_code")
            allowed = {"environment_error", "official_source_changed", "habitat_episode_identity_mismatch",
                       "habitat_nonsequential_episode", "invalid_action_id", "invalid_command"}
            raise OperationalError(code if code in allowed else "simulator_error")
        return value["data"]


def evaluate(environment, subset, output, client, bridge, dataset_digest):
    planner = None
    results = []
    summary = {
        "status": "running", "environment": environment, "subset": subset,
        "planned_tasks": 3, "completed_tasks": 0, "successful_tasks": 0,
        "model_requests": 0, "model_tool_calls": 0, "n_shots": 10,
        "geometry_tools_enabled": False, "skill_evolution_enabled": False,
        "reference_answer_visible": False, "automatic_retry": False,
        "requested_model": "qwen3.8-flash", "upstream_revision": REVISION,
    }
    write_json(output / "summary.json", summary)
    try:
        for index in range(3):
            record = {"index": index, "task_id": f"{environment}:{subset}:{index:03d}",
                      "status": "running", "model_requests": 0, "num_steps": 0,
                      "task_success": False, "task_progress": 0, "num_invalid_actions": 0,
                      "empty_plan": False}
            results.append(record)
            started = time.monotonic()
            packet = bridge.request("open", index=index, dataset_sha256=dataset_digest)
            obs = decode(packet)
            if planner is None:
                planner = OfficialHouseholdPlanner(client, environment, subset, obs.actions)
                write_json(output / "official_prompt_hashes.json", planner.adapter.prompt_hashes)
            planner.reset(obs.actions)
            folder = output / f"episode-{index:03d}"
            folder.mkdir()
            write_json(folder / "input.json", {
                "input_sha256": obs.input_sha256, "image_sha256": obs.image_sha256,
                "action_space_sha256": sha256(canonical(obs.actions)),
                "action_count": len(obs.actions), "reference_answer_visible": False})
            (folder / "step-000.png").write_bytes(obs.png)
            with (folder / "trace.jsonl").open("w") as trace:
                while not packet["done"]:
                    if planner.total_requests >= 120:
                        raise ContractError("run_request_budget_exhausted")
                    try:
                        action = planner.act(obs)
                    finally:
                        trace.write(canonical({"kind": "model_request", **planner.last_receipt}).decode() + "\n")
                        trace.flush()
                        record["model_requests"] = planner.adapter.planner_steps
                        summary["model_requests"] = planner.total_requests
                        write_json(output / "summary.json", summary)
                    if action == -2:
                        record["empty_plan"] = True
                        break
                    if action == -1:
                        packet = bridge.request("invalid")
                        trace.write(canonical({"kind": "invalid_plan", "invalid_actions": packet["invalid_actions"]}).decode() + "\n")
                        continue
                    for number in action[:30 - obs.step]:
                        packet = bridge.request("step", action=number)
                        next_obs = decode(packet)
                        if next_obs.step != obs.step + 1 or next_obs.input_sha256 != obs.input_sha256 \
                                or next_obs.actions != obs.actions:
                            raise ContractError("household_transition_identity_mismatch")
                        planner.update(number, packet["env_feedback"])
                        trace.write(canonical({
                            "kind": "environment_step", "step": next_obs.step, "action_id": number,
                            "image_sha256": next_obs.image_sha256,
                            "last_action_success": packet["last_action_success"],
                            "task_success": packet["task_success"], "task_progress": packet["task_progress"],
                            "invalid_actions": packet["invalid_actions"],
                            "feedback_sha256": sha256(packet["env_feedback"].encode()),
                        }).decode() + "\n")
                        trace.flush()
                        (folder / f"step-{next_obs.step:03d}.png").write_bytes(next_obs.png)
                        obs = next_obs
                        if packet["done"] or not packet["last_action_success"]:
                            break
            record.update(status="complete", num_steps=packet["step"],
                          task_success=packet["task_success"], task_progress=packet["task_progress"],
                          num_invalid_actions=packet["invalid_actions"],
                          planner_output_error=planner.adapter.output_json_error,
                          input_sha256=obs.input_sha256,
                          elapsed_seconds=round(time.monotonic() - started, 3))
            write_json(folder / "result.json", record)
            summary["completed_tasks"] += 1
            summary["successful_tasks"] += int(record["task_success"])
            summary["results"] = results
            write_json(output / "summary.json", summary)
            print(canonical(record).decode(), flush=True)
        summary["status"] = "complete"
    except Exception as exc:
        summary["status"] = "stopped"
        summary["reason_code"] = str(exc) if isinstance(exc, (ContractError, OperationalError)) else type(exc).__name__
        if results and results[-1]["status"] == "running":
            results[-1]["status"] = "operational_failure"
    summary.update(results=results, success_rate=summary["successful_tasks"] / 3,
                   num_steps=sum(x["num_steps"] for x in results))
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--environment", choices=tuple(SUBSETS), required=True)
    parser.add_argument("--subset", default="base")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        if args.subset not in SUBSETS[args.environment]:
            raise ContractError("invalid_eval_subset")
        output = within_root(args.output_dir)
        config = {"model": "qwen3.8-flash", "max_tokens": 4096, "code_root": "../src",
                  "environment": args.environment, "subset": args.subset,
                  "indexes": [0, 1, 2], "n_shots": 10, "seed": 137, "temperature": 0,
                  "enable_thinking": False, "resolution": 500, "max_environment_steps": 30,
                  "max_invalid_actions": 10, "max_requests_per_episode": 40,
                  "max_requests": 120, "chat_history": False, "env_feedback": True,
                  "multistep": False, "detection_box": False,
                  "habitat_episode_shuffle": False, "habitat_seed": 137,
                  "habitat_physical_gpu": 7,
                  "alfred_rendering": "task-local-Xvfb-Mesa-software"}
        checked = subprocess.run(ssh("probe", args.environment, args.subset, output),
                                 env=simulator_environment(ROOT / "checks"), capture_output=True, timeout=30, check=True)
        runtime = json_object(checked.stdout.decode())
        if any(sha256((ROOT / name).read_bytes()) != digest
               for name, digest in runtime["source_sha256"].items()):
            raise ContractError("remote_simulator_source_mismatch")
        load_authorized_key()
        client = model_client(config)
        dataset_path = "embodiedbench/envs/eb_alfred/data/splits/splits.json" if args.environment == "alfred" else \
            f"embodiedbench/envs/eb_habitat/datasets/{args.subset}.pickle"
        digest = sha256(source(dataset_path))
        receipt = {
            "status": "ready" if runtime["runtime_ready"] and client.configured else "prerequisites_missing",
            "simulator": runtime, "model_identity": client.audit_identity(seed=137),
            "dataset_sha256": digest, "model_requests": 0, "planned_tasks": 3,
            "reference_answer_visible": False, "model_tool_calls": 0, "skill_sha256": None,
            "automatic_retry": False, "uses_official_prompt": True, "uses_official_messages": True,
            "protocol_deviations": DEVIATIONS + ["Habitat pilot selects dataset indexes 0,1,2 with shuffle=false and seed=137"],
            "source_sha256": {name: sha256((ROOT / name).read_bytes()) for name in
                              ("household_run.py", "household_planner.py", "household_protocol.py",
                               "household_worker.py", "household_sources.json", "bridge.py", "protocol.py")},
            "existing_client_sha256": sha256((ROOT.parent / "src/skill3d/synthesis/bailian_client.py").read_bytes()),
        }
        if args.action == "check":
            write_json(ROOT / f"checks/{args.environment}-startup.json", receipt)
            print(canonical(receipt).decode())
            return 0 if receipt["status"] == "ready" else 2
        if receipt["status"] != "ready":
            raise ContractError("household_prerequisites_missing")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "startup.json", receipt)
        write_json(output / "config.json", config)
        bridge = HouseholdBridge(ssh("worker", args.environment, args.subset, output),
                                 work_dir=output, timeout_s=600)
        try:
            summary = evaluate(args.environment, args.subset, output, client, bridge, digest)
        finally:
            bridge.close()
        print(canonical(summary).decode())
        return 0 if summary["status"] == "complete" else 2
    except Exception as exc:
        print(canonical({"status": "startup_error", "reason_code": str(exc) if isinstance(exc, ContractError)
                         else type(exc).__name__}).decode(), file=sys.stderr)
        return 2
    finally:
        os.environ.pop("DASHSCOPE_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(main())
