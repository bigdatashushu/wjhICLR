"""Official reset -> plan -> full native plan -> step -> official metrics."""
import base64
import os
import time
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .planner import TargetPlanner
from .perception import Perception, PlannerError
from .protocol import validate_packet
from .sources import ContractError, LIMITS, OperationalError, REVISION, canonical, digest, implementation_hashes

DEVIATIONS = [
    "No upstream random fallback: malformed Navigation/Manipulation plans end the episode.",
    "Service/simulator/protocol failures stop the suite without retries or action re-sends.",
    "Discrete action IDs require actual integers; Manipulation requires a legal 7D vector.",
    "Explicit task selection and request budgets; fixed selected denominator, including unstarted tasks.",
    "Habitat iterator shuffle/grouping disabled to bind the requested dataset index.",
    "Runtime-only paths, private caches and owned Xvfb; official source files remain unchanged.",
    "Habitat's single-element elapsed-time tuple is converted to a number without changing its value.",
    "All four environments supply synchronized RGB and 500x500 simulated-depth visualization "
    "images to the model, with a fixed 0-10 meter inverse grayscale mapping.",
    "Raw float32 meter depth is retained for evidence but not included in model messages.",
    "Same Qwen extracts question-bound targets, calls pinned GPU-1 GroundingDINO per category, "
    "reviews lossless overlap groups and unannotated crops, then emits an official full plan.",
    "All perception rounds count toward the original model-request limit and never step/observe the simulator.",
    "Manipulation exposes only anonymous official coordinate projections for possible image associations.",
]


def save(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


class Journal:
    def __init__(self, directory):
        self.path = directory / "trajectory.jsonl"
        self.stream = self.path.open("x", encoding="utf-8")

    def append(self, event):
        self.stream.write(canonical(event).decode() + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def record_packet(journal, work, packet, label, serial):
    validate_packet(packet, label)
    for i, image in enumerate(packet["public"]["images"]):
        (work / f"observation-{serial:03d}-{i}.png").write_bytes(base64.b64decode(image["base64"]))
    (work / f"observation-{serial:03d}-raw.png").write_bytes(
        base64.b64decode(packet["public"]["raw_rgb"]["base64"]))
    depth = packet["public"]["depth"]
    (work / f"observation-{serial:03d}-depth.png").write_bytes(
        base64.b64decode(depth["image"]["base64"]))
    from .depth import decode_raw
    import numpy as np
    np.save(work / f"observation-{serial:03d}-depth-meters.npy", decode_raw(depth), allow_pickle=False)
    journal.append({"type": "observation", "serial": serial, **packet})


class RequestGate:
    """Bound HTTP concurrency and close admission on the first operational failure."""
    def __init__(self, client, maximum):
        self.client, self.stopped = client, threading.Event()
        self.slots = threading.BoundedSemaphore(maximum)
        self.lock, self.active, self.peak, self.requests = threading.Lock(), 0, 0, 0

    def payload(self, messages, schema):
        return self.client.payload(messages, schema)

    def complete(self, payload):
        with self.slots:
            if self.stopped.is_set():
                raise OperationalError("suite_request_admission_closed")
            with self.lock:
                self.active += 1
                self.requests += 1
                self.peak = max(self.peak, self.active)
            try:
                return self.client.complete(payload)
            except Exception:
                self.stopped.set()
                raise
            finally:
                with self.lock:
                    self.active -= 1


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
            if getattr(client, "stopped", None) is not None and client.stopped.is_set():
                status = "stopped_after_peer_failure"
                break
            public = packet["public"]
            if requests >= config["max_requests"][task.environment]:
                status = "request_budget"
                break
            if public["step"] >= LIMITS[task.environment]:
                raise OperationalError("official_termination_missing")
            # Annotation is needed only at model decisions, as in the evaluator.
            if task.environment == "manipulation" and steps:
                packet = session.request("observe")
                serial += 1
                record_packet(journal, directory, packet, task, serial)
                public = packet["public"]
            perception.bind(public, serial)
            plan = None
            # Tools operate on this exact packet; no observe/reset/step in this loop.
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
                    # Explicit empty-plan branch in the official evaluator.
                    rewards.append(0)
                    action_successes.append(0)
                    packet["official"].update(task_success=0, episode_elapsed_seconds=0)
                break
            if plan.status == "invalid":
                if task.environment not in ("alfred", "habitat"):
                    status = "planner_error"
                    break
                # Evaluator increments invalid count without an environment step
                # or planner feedback. The worker owns the same env counter.
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
                if getattr(client, "stopped", None) is not None and client.stopped.is_set():
                    status = "stopped_after_peer_failure"
                    break
                previous_step = packet["public"]["step"]
                # Persist intent BEFORE crossing the process boundary. If the
                # response is lost, its outcome remains unknown and is not retried.
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
        if getattr(client, "stopped", None) is not None:
            client.stopped.set()
    except Exception as exc:
        status, error = "operational_error", {"code": "runner_error", "exception_type": type(exc).__name__}
        journal.append({"type": "failure", **error, "automatic_retry": False})
        if getattr(client, "stopped", None) is not None:
            client.stopped.set()
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                journal.append({"type": "cleanup_failure", "code": "worker_close_error"})
                if status != "operational_error" or error is None:
                    status, error = "operational_error", {"code": "worker_close_error"}
                if getattr(client, "stopped", None) is not None:
                    client.stopped.set()
        official = dict(packet["official"]) if packet else None
        metrics = {
            "instruction": packet["public"]["instruction"] if packet else None,
            "task_success": official.get("task_success") if official else None,
            "num_steps": packet["public"]["step"] if packet else 0,
            "planner_steps": requests, "planner_output_error": planner.adapter.output_json_error if planner else 0,
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
                                 "detector_requests": perception.detector_calls,
                                 "crop_count": perception.crop_count},
                  "rewards": rewards, "host_elapsed_seconds": time.monotonic() - start}
        save(directory / "result.json", result)
        journal.append({"type": "episode_end", "result": result})
        journal.close()
    return result


def suite(tasks, config, sources, client, session_factory, directory):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    model_deviations = []
    if config["model"]["response_format"] != "official_schema":
        model_deviations.append("Explicit JSON-object response mode replaces the official schema.")
        model_deviations.append("Manipulation appends the pinned official template_manip to convey "
                                "the output format previously supplied through the official schema.")
    if config["model"]["name"] != "gpt-4o-mini":
        model_deviations.append("Explicit configured model replaces the default gpt-4o-mini.")
    if config["model"]["extra_body"]:
        model_deviations.append("Explicit extra model parameters are recorded in config.json.")
    save(directory / "config.json", config)
    save(directory / "identity.json", {
        "revision": REVISION, "implementation_hashes": implementation_hashes(),
        "source_hashes": sources.hashes, "protocol_deviations": DEVIATIONS + model_deviations,
        "selected_tasks": [t.as_dict() for t in tasks], "reference_answer_visible": False,
        "automatic_retry": False, "mode": "four_environments_rgbd_target_detection"})
    execution = config.get("execution", {
        "max_episodes": 1, "max_api_requests": 1, "max_gpu_episodes": 1,
        "per_environment": {kind: 1 for kind in LIMITS}})
    gate = RequestGate(client, execution["max_api_requests"])
    results = [{"task": t.as_dict(), "status": "not_started", "official": None} for t in tasks]
    pending, running, active = list(enumerate(tasks)), {}, {kind: 0 for kind in LIMITS}
    gpu_active = 0
    with ThreadPoolExecutor(max_workers=execution["max_episodes"]) as pool:
        while pending or running:
            if gate.stopped.is_set():
                pending.clear()
            for index, task in list(pending):
                uses_gpu = task.environment in ("navigation", "habitat")
                if (len(running) >= execution["max_episodes"]
                        or active[task.environment] >= execution["per_environment"][task.environment]
                        or uses_gpu and gpu_active >= execution["max_gpu_episodes"]):
                    continue
                if gate.stopped.is_set():
                    break
                future = pool.submit(episode, task, config, sources, gate, session_factory,
                                     directory / task.directory)
                running[future] = (index, task, uses_gpu)
                pending.remove((index, task))
                active[task.environment] += 1
                gpu_active += int(uses_gpu)
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                index, task, uses_gpu = running.pop(future)
                results[index] = future.result()
                active[task.environment] -= 1
                gpu_active -= int(uses_gpu)
            save(directory / "results.json", results)
    stopped = gate.stopped.is_set()
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
    summary = {"status": "stopped_on_operational_failure" if stopped else "finished",
               "scope": "explicit selected episodes; not a full benchmark score",
               "selected": len(tasks), "by_environment": by_environment,
               "automatic_retry": False,
               "model_tool_calls": sum(r.get("perception", {}).get("tool_calls", 0) for r in results),
               "detector_requests": sum(r.get("perception", {}).get("detector_requests", 0) for r in results),
               "execution": execution, "http_requests": gate.requests,
               "peak_http_concurrency": gate.peak}
    save(directory / "results.json", results)
    save(directory / "summary.json", summary)
    return summary
