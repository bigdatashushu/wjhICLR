"""Host-side navigation loop, separate from simulator scoring and model prompts."""
from __future__ import annotations

import os
from pathlib import Path
import time

from protocol import ContractError, OperationalError, canonical, sha256


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    os.replace(temporary, path)


def evaluate(tasks, planner, environment, output_dir, *, max_requests):
    """All planned tasks stay in the denominator, including failures/not_run."""
    output_dir = Path(output_dir)
    results, stopped = [], None
    starting_requests = planner.total_requests
    trace_path = output_dir / "trace.jsonl"
    for task in tasks:
        row = {
            **task.public_dict(), "status": "not_run", "task_success": False,
            "environment_steps": 0, "planner_requests": 0, "invalid_actions": 0,
            "planner_errors": 0, "reference_answer_visible": False,
        }
        if stopped:
            row["reason_code"] = stopped
            results.append(row)
            continue
        started = time.monotonic()
        planner.reset()
        try:
            transition = environment.open(task)
            if transition.observation.step != 0 or transition.done:
                raise OperationalError("invalid_initial_observation")
            instruction = transition.instruction
            row["instruction_sha256"] = sha256(instruction.encode("utf-8"))
            row["status"] = "failure"
            while not transition.done:
                if planner.total_requests - starting_requests >= max_requests:
                    stopped = "run_request_budget_exhausted"
                    row["status"], row["reason_code"] = "operational_error", stopped
                    break
                plan = planner.act(transition.observation, instruction)
                receipt = dict(planner.last_receipt)
                with trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(canonical({"task_id": task.task_id, **receipt}).decode("utf-8") + "\n")
                for index, action in enumerate(plan.actions):
                    previous_step = transition.observation.step
                    transition = environment.step(action, first_in_plan=index == 0)
                    if (transition.observation.step != previous_step + 1
                            or transition.instruction != instruction
                            or transition.task_success and not transition.done):
                        raise OperationalError("transition_identity_mismatch")
                    row["environment_steps"] = transition.observation.step
                    row["invalid_actions"] += int(not transition.last_action_success)
                    with trace_path.open("a", encoding="utf-8") as stream:
                        stream.write(canonical({
                            "event": "environment_step", "task_id": task.task_id,
                            "action_id": action, "step": transition.observation.step,
                            "image_sha256": transition.observation.image_sha256,
                            "last_action_success": transition.last_action_success,
                            "done": transition.done, "task_success": transition.task_success,
                            "reference_answer_visible": False,
                        }).decode("utf-8") + "\n")
                    planner.update_info(
                        action=action, feedback=transition.feedback,
                        last_action_success=transition.last_action_success,
                        step=transition.observation.step,
                    )
                    if transition.done:
                        row["task_success"] = transition.task_success
                        row["status"] = "success" if transition.task_success else "failure"
                        break
                    if transition.observation.step == 20:
                        raise OperationalError("missing_environment_termination")
                    if not transition.last_action_success:
                        break  # Discard the rest of this plan and reobserve/replan.
        except ContractError as exc:
            row["status"], row["reason_code"] = "planner_error", str(exc)
            row["planner_errors"] = 1
        except OperationalError as exc:
            stopped = str(exc)
            row["status"], row["reason_code"] = "operational_error", stopped
        except Exception:
            stopped = "runner_environment_error"
            row["status"], row["reason_code"] = "operational_error", stopped
        row["planner_requests"] = planner.planner_steps
        row["elapsed_seconds"] = round(time.monotonic() - started, 6)
        if row["status"] in {"planner_error", "operational_error"} and planner.last_receipt:
            with trace_path.open("a", encoding="utf-8") as stream:
                stream.write(canonical({
                    "task_id": task.task_id, "event": "episode_error",
                    **planner.last_receipt, "reason_code": row["reason_code"],
                }).decode("utf-8") + "\n")
        results.append(row)
        write_json(output_dir / "episodes" / f"{task.task_id.replace(':', '-')}.json", row)
    # Materialize placeholders as well, so incomplete runs cannot hide denominators.
    for row in results:
        write_json(output_dir / "episodes" / f"{row['task_id'].replace(':', '-')}.json", row)
    successful = sum(row["task_success"] for row in results)
    summary = {
        "status": "complete" if stopped is None else "incomplete",
        "stop_reason": stopped, "planned_tasks": len(tasks),
        "successful_tasks": successful,
        "task_success_rate": successful / len(tasks) if tasks else None,
        "per_subset": {
            subset: {
                "tasks": sum(row["subset"] == subset for row in results),
                "successful_tasks": sum(row["task_success"] for row in results if row["subset"] == subset),
                "task_success_rate":
                    sum(row["task_success"] for row in results if row["subset"] == subset)
                    / sum(row["subset"] == subset for row in results),
            } for subset in sorted({task.subset for task in tasks})
        },
        "statuses": {status: sum(row["status"] == status for row in results)
                     for status in sorted({row["status"] for row in results})},
        "model_requests": planner.total_requests - starting_requests,
        "max_requests": max_requests, "automatic_retry": False,
        "reference_answer_visible": False,
        "planner_mode": planner.config.mode, "skill_sha256": planner.skill_sha256,
        "uses_official_default_planner": False,
        "uses_official_prompt": getattr(planner, "uses_official_prompt", False),
        "uses_official_messages": getattr(planner, "uses_official_messages", False),
        "protocol_deviations": getattr(planner, "protocol_deviations", []),
        "geometry_tools_enabled": False, "skill_evolution_enabled": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
