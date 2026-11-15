from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

from xcodeapi import HTTPAPIError, XcodeClient, RequestLimitError

from .config import Config, RunnerError
from .environment import Runtime, controller_spec
from .policy import GPTPolicy
from .rollout import rollout
from .storage import Journal, digest, file_digest, read_json, source_identity, verify_files, write_json

COMPLETED = {"success", "task_failure", "invalid_action"}


@contextmanager
def run_lock(directory: Path):
    with (directory / ".owner.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunnerError("Another process owns this run directory.") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare(config: Config, directory: Path) -> dict:
    if directory.exists():
        raise RunnerError("Choose a new output directory; existing runs cannot be overwritten.")
    runtime = Runtime(config)
    tasks = runtime.tasks()
    # Every selected task must render and accept the official dummy actions.
    probes = [runtime.probe(task) for task in tasks]
    plan = []
    for task in tasks:
        for index in range(runtime.settings["episodes_per_task"]):
            plan.append({"episode_id": f"{task['suite']}-t{task['task_id']:02d}-i{index:03d}",
                         "suite": task["suite"], "task_id": task["task_id"],
                         "initial_state_index": index})
    manifest = {
        "schema_version": 1, "policy": "gpt_single_step_osc_pose",
        "config": config.to_dict(), "official_defaults": runtime.defaults,
        "evaluation_settings": runtime.settings, "upstream": runtime.identity,
        "dependencies": runtime.versions, "runner_source": source_identity(),
        "tasks": tasks, "episodes": plan, "preflight": probes, "model_requests": 0,
    }
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "manifest.sha256.json", {"sha256": digest(manifest)})
    summarize(directory)
    return manifest


def verified_manifest(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    if digest(manifest) != read_json(directory / "manifest.sha256.json")["sha256"]:
        raise RunnerError("Run manifest changed.")
    verify_files(manifest["runner_source"])
    verify_files(manifest["upstream"]["files"])
    for task in manifest["tasks"]:
        verify_files(task["files"])
    return manifest


def _ack_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(json.loads(line).get("event") == "action_ack"
               for line in path.read_text().splitlines())


def _verify_action_failure(episode: Path, row: dict):
    # Old frozen results did not contain rejection text; do not invent it.
    if "action_failure" not in row:
        return
    failure = row["action_failure"]
    if not isinstance(failure, dict) or row["status"] != "invalid_action":
        raise RunnerError("An action failure has an inconsistent result status.")
    step = failure.get("step")
    if type(step) is not int or step < 0 or step != row["control_steps"]:
        raise RunnerError("An action failure does not match its rejected control step.")
    name = f"failures/{step:04d}-action.json"
    if failure.get("file") != name:
        raise RunnerError("An action failure has an unexpected artifact path.")
    path = episode / name
    if (not path.resolve().is_relative_to(episode.resolve()) or not path.is_file()
            or file_digest(path) != failure.get("sha256")):
        raise RunnerError("An action failure artifact is missing or changed.")
    artifact = read_json(path)
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
        raise RunnerError("An action failure artifact has an invalid schema.")
    text = artifact.get("response_text")
    if (not isinstance(text, str)
            or hashlib.sha256(text.encode()).hexdigest() != failure.get("response_text_sha256")):
        raise RunnerError("An action failure does not match its response text hash.")
    validation = {key: failure.get(key) for key in ("branch", "message", "details")}
    if artifact.get("validation") != validation or any(
        artifact.get(key) != failure.get(key)
        for key in ("step", "request_index", "payload_sha256", "response_text_sha256")
    ):
        raise RunnerError("An action failure artifact does not match its result.")
    events = [json.loads(line) for line in (episode / "trace.jsonl").read_text().splitlines()]
    requests = [e for e in events if e.get("event") == "request_started" and e.get("step") == step]
    responses = [e for e in events if e.get("event") == "response_received" and e.get("step") == step]
    rejections = [e for e in events if e.get("event") == "action_rejected"]
    if (len(requests) != 1 or len(responses) != 1 or len(rejections) != 1
            or any(requests[0].get(key) != failure.get(key)
                   for key in ("request_index", "payload_sha256"))
            or responses[0].get("response_text_sha256") != failure.get("response_text_sha256")
            or responses[0].get("finish_reason") != artifact.get("finish_reason")
            or responses[0].get("metadata") != artifact.get("metadata")
            or rejections[0].get("step") != step
            or rejections[0].get("action_failure") != failure):
        raise RunnerError("An action failure does not match its request/response/rejection trace.")


def summarize(directory: Path) -> dict:
    manifest = verified_manifest(directory)
    receipt_path = directory / "final_collection.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("manifest_sha256") != digest(manifest):
            raise RunnerError("Final collection belongs to a different manifest.")
        for name, expected in receipt["files"].items():
            path = directory / name
            if not path.is_file() or file_digest(path) != expected:
                raise RunnerError("A collected result or trajectory changed after completion.")
    rows = []
    for item in manifest["episodes"]:
        path = directory / "episodes" / item["episode_id"] / "result.json"
        row = read_json(path) if path.exists() else {"status": "pending", "success": False}
        if path.exists():
            if row.get("manifest_sha256") != digest(manifest):
                raise RunnerError("An episode result belongs to a different manifest.")
            if row.get("control_steps") != _ack_count(path.parent / "trace.jsonl"):
                raise RunnerError("An episode result does not match its action ACK count.")
            if row.get("success") != (row["status"] == "success"):
                raise RunnerError("An episode result has inconsistent success/status fields.")
            _verify_action_failure(path.parent, row)
        rows.append({**item, **row})
    complete = all(r["status"] in COMPLETED for r in rows)
    per_suite = {}
    for suite in manifest["config"]["suites"]:
        selected = [r for r in rows if r["suite"] == suite]
        successes = sum(r["status"] == "success" for r in selected)
        completed = sum(r["status"] in COMPLETED for r in selected)
        per_suite[suite] = {
            "planned": len(selected), "completed": completed, "successes": successes,
            "protocol_failures": sum(r["status"] == "invalid_action" for r in selected),
            "operational_errors": sum(r["status"] == "operational_error" for r in selected),
            "remaining": len(selected) - completed,
            "success_rate": successes / len(selected) if completed == len(selected) else None,
            "success_rate_completed": successes / completed if completed else None,
        }
    summary = {
        "complete": complete, "planned": len(rows),
        "completed": sum(r["status"] in COMPLETED for r in rows),
        "successes": sum(r["status"] == "success" for r in rows),
        "success_rate": sum(r["status"] == "success" for r in rows) / len(rows)
        if complete else None,
        "suite_mean_success_rate": sum(s["success_rate"] for s in per_suite.values()) / len(per_suite)
        if complete else None,
        "per_suite": per_suite,
    }
    write_json(directory / "summary.json", summary)
    write_json(directory / "episode_results.json", rows)
    if complete:
        receipt_files = [directory / "manifest.json", directory / "episode_results.json",
                         directory / "summary.json"]
        receipt_files += list((directory / "episodes").glob("*/result.json"))
        receipt_files += list((directory / "episodes").glob("*/trace.jsonl"))
        receipt_files += list((directory / "episodes").glob("*/failures/*.json"))
        write_json(directory / "final_collection.json", {
            "complete": True, "manifest_sha256": digest(manifest),
            "expected_episodes": len(rows),
            "files": {str(p.relative_to(directory)): file_digest(p) for p in sorted(receipt_files)},
        })
    return summary


def run(directory: Path, *, credentials_file: Path | None = None,
        resume: bool = False) -> dict:
    with run_lock(directory):
        manifest = verified_manifest(directory)
        config = Config(**manifest["config"])
        runtime = Runtime(config)
        if runtime.identity != manifest["upstream"] or runtime.versions != manifest["dependencies"]:
            raise RunnerError("Simulator source or dependency versions changed after preflight.")
        if runtime.settings != manifest["evaluation_settings"]:
            raise RunnerError("Official evaluation defaults changed after preflight.")
        task_map = {(t["suite"], t["task_id"]): t for t in manifest["tasks"]}
        # A new invocation never retries an incomplete episode automatically.
        for item in manifest["episodes"]:
            episode = directory / "episodes" / item["episode_id"]
            if not episode.exists():
                continue
            result_path = episode / "result.json"
            if not result_path.exists() or read_json(result_path)["status"] not in COMPLETED:
                raise RunnerError(
                    f"Incomplete episode {item['episode_id']} requires explicit recovery "
                    "in a separate run. It will not be retried."
                )
            if not resume:
                raise RunnerError("This run has results. Use --resume to skip completed episodes.")
        # Budget is cumulative across invocations, including outcomes unknown.
        used = 0
        for path in (directory / "episodes").glob("*/trace.jsonl"):
            used += sum(json.loads(line).get("event") == "request_started"
                        for line in path.read_text().splitlines())
        if used >= config.max_requests:
            raise RunnerError("The frozen cumulative API request budget is exhausted.")
        kwargs = {"max_requests": config.max_requests - used}
        if credentials_file is not None:
            kwargs["credentials_file"] = credentials_file
        client = XcodeClient(config.api_profile(), **kwargs)
        for item in manifest["episodes"]:
            episode = directory / "episodes" / item["episode_id"]
            if (episode / "result.json").exists():
                continue
            if client.attempts >= config.max_requests - used:
                break
            task = task_map[(item["suite"], item["task_id"])]
            episode.mkdir(parents=True, exist_ok=False)
            journal = Journal(episode / "trace.jsonl")
            journal({"event": "episode_started", **item})
            env = None
            interrupted = False
            try:
                env = runtime.create_env(task)
                controller = controller_spec(env)
                expected = manifest["preflight"][manifest["tasks"].index(task)]["controller"]
                if controller != expected:
                    raise RunnerError("Controller changed after preflight.")
                policy = GPTPolicy(client, config, controller, journal, episode / "observations",
                                   failure_dir=episode / "failures")
                states = runtime.initial_states(item["suite"], item["task_id"])
                result = rollout(env, policy, states[item["initial_state_index"]],
                                 task["language"], runtime.settings["max_steps"],
                                 runtime.settings["settling_steps"], journal)
            except RequestLimitError:
                result = {"status": "budget_exhausted", "success": False,
                          "control_steps": _ack_count(episode / "trace.jsonl")}
                interrupted = True
            except Exception as exc:
                # Error type only: no credentials, native bodies, or exception payloads.
                result = {"status": "operational_error", "success": False,
                          "error_type": type(exc).__name__,
                          "control_steps": _ack_count(episode / "trace.jsonl")}
                if isinstance(exc, HTTPAPIError):
                    result["http_status"] = exc.status
                interrupted = True
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as exc:
                        result = {"status": "operational_error", "success": False,
                                  "error_type": "cleanup_" + type(exc).__name__,
                                  "control_steps": _ack_count(episode / "trace.jsonl")}
                        interrupted = True
            result["manifest_sha256"] = digest(manifest)
            write_json(episode / "result.json", result)
            journal({"event": "episode_finished", **result})
            summary = summarize(directory)
            print(json.dumps({"episode_id": item["episode_id"], **result,
                              "completed_total": summary["completed"]}, ensure_ascii=False), flush=True)
            if interrupted:
                break
        return summarize(directory)
