"""Independent run entry, with the existing full-plan/action/metric contracts."""
import time
from pathlib import Path

from target_detection_r05.runner import DEVIATIONS as ORIGINAL_DEVIATIONS, Journal, RequestGate, record_packet, save
from .planner import MAX_SHOTS, TargetPlanner
from .perception import Perception, PlannerError
from .sources import ContractError, LIMITS, OperationalError, REVISION, canonical, digest, implementation_hashes

DEVIATIONS = [row for row in ORIGINAL_DEVIATIONS if "GroundingDINO" not in row] + [
    "Qwen3.8-Flash extracts targets and plans. Qwen3-VL-8B-Instruct-FP8 detects each "
    "category using original unannotated RGB only. Its normalized 0-1000 boxes are mapped "
    "to original pixel edges before overlap grouping, display and crop review.",
    "Qwen3-VL supplies no calibrated detection confidence; no scores or score threshold are invented.",
    "Targets are extracted once. Detection and up to two crops are prepared together; later "
    "observations are prepared automatically. The model reviews crops and emits its native full plan "
    "in one response; optional extra tools still consume the fixed planning-request budget.",
    "The tool protocol appears only in system. User context holds a current candidate summary, "
    "one overlay and the latest crop set; full tool evidence stays in the journal.",
    "Few-shot demonstrations are capped at three in every environment, honoring smaller configured counts.",
    "Native plans require non-empty visual_state_description, reasoning_and_reflection and language_plan. "
    "ALFRED's system protocol emphasizes its one-object hand constraint and successful action feedback.",
]


def episode(task, config, sources, client, session_factory, directory):
    directory.mkdir()
    journal, session = Journal(directory), None
    start = time.monotonic()
    packet, planner, requests, steps, serial = None, None, 0, 0, 0
    perception = Perception(config, directory, journal)
    rewards, action_successes, invalid_count = [], [], 0
    status, error, pending_action = "operational_error", None, False
    try:
        journal.append({"type": "episode_start", "task": task.as_dict(),
                        "reference_answer_visible": False, "automatic_retry": False})
        session = session_factory(task, directory, journal)
        packet = session.request("open", task=task.as_dict())
        record_packet(journal, directory, packet, task, serial)
        if packet["public"]["step"] != 0 or packet["official"]["done"]:
            raise OperationalError("invalid_reset_receipt")
        planner = TargetPlanner(sources, task, config["model"]["name"],
                                config["n_shots"][task.environment], packet["public"]["actions"],
                                config["model"]["response_format"])
        while not packet["official"]["done"]:
            if client.stopped.is_set():
                status = "stopped_after_peer_failure"
                break
            public = packet["public"]
            if requests >= config["max_requests"][task.environment]:
                status = "request_budget"
                break
            if public["step"] >= LIMITS[task.environment]:
                raise OperationalError("official_termination_missing")
            if task.environment == "manipulation" and steps:
                packet = session.request("observe")
                serial += 1
                record_packet(journal, directory, packet, task, serial)
                public = packet["public"]
            perception.bind(public, serial)
            if perception.targets is not None and not perception.detected:
                perception.prepare()
            plan = None
            while requests < config["max_requests"][task.environment]:
                messages = planner.messages(public, perception)
                payload = client.payload(messages, planner.schema)
                requests += 1
                request_path = directory / f"request-{requests:03d}.json"
                save(request_path, payload)
                journal.append({"type": "model_request", "request_index": requests,
                                "environment_step": public["step"], "observation_serial": serial,
                                "request_file": request_path.name,
                                "payload_sha256": digest(canonical(payload)),
                                "completed_perception_tool_calls": perception.tool_calls,
                                "reference_answer_visible": False})
                response = client.complete(payload)
                save(directory / f"response-{requests:03d}.json", response)
                journal.append({"type": "model_response_received", "request_index": requests,
                                "response_file": f"response-{requests:03d}.json",
                                "response_sha256": response.get("response_sha256"),
                                "usage": response.get("usage", {})})
                plan = planner.decision(response["text"], perception)
                journal.append({"type": "model_response", "request_index": requests,
                                "response": response, "plan": vars(plan) if plan else None})
                if plan is not None:
                    break
            if plan is None:
                status = "request_budget"
                break
            if plan.status == "empty":
                status = "planner_error" if task.environment == "navigation" else "empty_plan"
                if task.environment == "manipulation":
                    rewards.append(0)
                    action_successes.append(0)
                    packet["official"].update(task_success=0, episode_elapsed_seconds=0)
                break
            if plan.status == "invalid":
                if task.environment not in ("alfred", "habitat"):
                    status = "planner_error"
                    break
                packet = session.request("invalid", reasoning=response["text"])
                serial += 1
                record_packet(journal, directory, packet, task, serial)
                rewards.append(-1)
                invalid_count += 1
                if packet["official"]["done"]:
                    status = "invalid_action_limit"
                continue
            remaining = LIMITS[task.environment] - public["step"]
            executable = plan.actions[:remaining]
            journal.append({"type": "plan_execution", "request_index": requests,
                            "executed_prefix": executable, "proposed_count": plan.proposed_count})
            for index, action in enumerate(executable):
                if client.stopped.is_set():
                    status = "stopped_after_peer_failure"
                    break
                previous_step = packet["public"]["step"]
                journal.append({"type": "action_sent", "action": action,
                                "request_index": requests, "index_in_plan": index,
                                "expected_step": previous_step + 1})
                pending_action = True
                packet = session.request("step", action=action, first_in_plan=index == 0,
                                         reasoning=response["text"])
                serial += 1
                record_packet(journal, directory, packet, task, serial)
                if packet["public"]["step"] != previous_step + 1:
                    raise OperationalError("official_step_identity_mismatch")
                pending_action = False
                steps += 1
                rewards.append(packet["official"]["reward"])
                succeeded = packet["public"]["last_action_success"]
                action_successes.append(int(succeeded))
                invalid_count += int(not succeeded)
                planner.feedback(action, packet["public"])
                journal.append({"type": "action_ack", "action": action,
                                "official": packet["official"], "last_action_success": succeeded})
                if packet["official"]["done"]:
                    status = ("invalid_action_limit" if packet["official"]["invalid_actions"] >= 10
                              and task.environment in ("alfred", "habitat") else "official_done")
                    break
                if task.environment != "manipulation" and not succeeded:
                    break
        if packet["official"]["done"] and status == "operational_error":
            status = "official_done"
    except PlannerError as exc:
        status, error = "planner_error", {"code": str(exc)}
        if planner is not None:
            planner.adapter.output_json_error += 1
        journal.append({"type": "planner_failure", **error, "automatic_retry": False})
    except (OperationalError, ContractError) as exc:
        status = "operational_error"
        error = {"code": exc.code if isinstance(exc, OperationalError) else "protocol_contract_error"}
        if getattr(exc, "http_status", None) is not None:
            error["http_status"] = exc.http_status
        journal.append({"type": "failure", **error, "automatic_retry": False,
                        "pending_action_may_have_executed": pending_action})
        client.stopped.set()
    except Exception as exc:
        status, error = "operational_error", {"code": "runner_error", "exception_type": type(exc).__name__}
        journal.append({"type": "failure", **error, "automatic_retry": False})
        client.stopped.set()
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                journal.append({"type": "cleanup_failure", "code": "worker_close_error"})
                status, error = "operational_error", {"code": "worker_close_error"}
                client.stopped.set()
        official = dict(packet["official"]) if packet else None
        metrics = {
            "instruction": packet["public"]["instruction"] if packet else None,
            "task_success": official.get("task_success") if official else None,
            "num_steps": packet["public"]["step"] if packet else 0,
            "planner_steps": requests,
            "planner_output_error": planner.adapter.output_json_error if planner else 0,
            "num_invalid_actions": invalid_count,
            "num_invalid_action_ratio": invalid_count / steps if steps else 0,
            "empty_plan": int(status == "empty_plan"),
            "reward": sum(rewards) / len(rewards) if rewards else None,
            "episode_elapsed_seconds": official.get("episode_elapsed_seconds") if official else None,
        }
        if task.environment == "manipulation":
            metrics["avg_reward"] = metrics.pop("reward")
            metrics["action_success"] = action_successes
        if official and "task_progress" in official:
            metrics["task_progress"] = official["task_progress"]
        result = {"task": task.as_dict(), "status": status, "error": error,
                  "official": official, "metrics": metrics,
                  "perception": {"targets": perception.targets, "tool_calls": perception.tool_calls,
                                 "pipeline_preparations": perception.pipeline_preparations,
                                 "detector_requests": perception.detector_calls,
                                 "crop_count": perception.crop_count},
                  "rewards": rewards, "host_elapsed_seconds": time.monotonic() - start}
        save(directory / "result.json", result)
        journal.append({"type": "episode_end", "result": result})
        journal.close()
    return result


def suite(tasks, config, sources, client, session_factory, directory):
    config = dict(config, n_shots={kind: min(n, MAX_SHOTS) for kind, n in config["n_shots"].items()})
    directory = Path(directory)
    directory.parent.mkdir(exist_ok=True)
    directory.mkdir(exist_ok=False)
    model_deviations = [
        "Explicit JSON-object response mode replaces the official schema.",
        "Manipulation appends the pinned official template_manip to convey the output format.",
    ]
    if config["model"]["name"] != "gpt-4o-mini":
        model_deviations.append("Explicit configured planning model replaces gpt-4o-mini.")
    if config["model"]["extra_body"]:
        model_deviations.append("Explicit extra planning-model parameters are recorded in config.json.")
    save(directory / "config.json", config)
    save(directory / "identity.json", {
        "revision": REVISION, "implementation_hashes": implementation_hashes(),
        "source_hashes": sources.hashes, "protocol_deviations": DEVIATIONS + model_deviations,
        "selected_tasks": [t.as_dict() for t in tasks], "reference_answer_visible": False,
        "automatic_retry": False, "mode": "four_environments_rgbd_qwen3vl_target_detection"})
    gate = RequestGate(client, 1)
    results = [{"task": t.as_dict(), "status": "not_started", "official": None} for t in tasks]
    for index, task in enumerate(tasks):
        if gate.stopped.is_set():
            break
        results[index] = episode(task, config, sources, gate, session_factory, directory / task.directory)
        save(directory / "results.json", results)
    by_environment = {}
    for kind in sorted({t.environment for t in tasks}):
        rows = [r for r in results if r["task"]["environment"] == kind]
        successes = sum(r["status"] not in ("not_started", "operational_error")
                        and r["official"] is not None and r["official"].get("task_success") == 1
                        for r in rows)
        by_environment[kind] = {
            "selected": len(rows), "started": sum(r["status"] != "not_started" for r in rows),
            "official_done": sum(bool(r["official"] and r["official"]["done"]) for r in rows),
            "successes": successes, "success_rate_selected": successes / len(rows),
            "operational_failures": sum(r["status"] == "operational_error" for r in rows)}
    summary = {
        "status": "stopped_on_operational_failure" if gate.stopped.is_set() else "finished",
        "scope": "explicit selected episodes; not a full benchmark score",
        "selected": len(tasks), "by_environment": by_environment, "automatic_retry": False,
        "model_tool_calls": sum(r.get("perception", {}).get("tool_calls", 0) for r in results),
        "detector_requests": sum(r.get("perception", {}).get("detector_requests", 0) for r in results),
        "execution": config["execution"], "http_requests": gate.requests,
        "peak_http_concurrency": gate.peak,
    }
    save(directory / "results.json", results)
    save(directory / "summary.json", summary)
    return summary
