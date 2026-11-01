"""Adapter for the actual compact main solver, with passive episode tracing."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import time

from skill3d.schemas.tool_response import model_tool_response
from skill3d.small_image.blink_protocol import image_identities, original_messages, response_record
from skill3d.small_image.cv3d_protocol import VERSION, task_tool_names
from skill3d.small_image.runner import (
    SmallImageRunnerConfig, SmallImageSolver, _validate_prepared, build_solver_prompt,
    load_problem_frames,
)
from skill3d.small_image.tools import build_registry
from skill3d.synthesis.bailian_client import BailianClient
from skill3d.synthesis.request_audit import payload_bytes

from .common import digest, write_json
from .data import verified_prepared
from .inference import prepare_request as native_prepare, solve as native_solve

PROTOCOL = "cv3d-compact-main-skill-versus-direct/1"


def prepare_request(public, directory, *, arm, frozen, config):
    if arm == "direct":
        return native_prepare(public, arm=arm)
    if arm != "tools":
        raise ValueError("unknown experiment arm")
    problem = public.problem
    prepared = verified_prepared(public, directory)
    frames = load_problem_frames(problem, original_image_bytes=True)
    if _validate_prepared(prepared, problem, frames) is not None:
        raise ValueError("main solver prepared validation failed")
    registry, scene = build_registry(prepared, problem)
    allowed = set(registry.names_for_scope(
        scene.question_tool_scope, scene.available_artifacts,
        evidence_profile=scene.evidence_profile, question_type=scene.question_type,
        local_facts=scene.local_facts)) & task_tool_names(problem.task_type)
    text = build_solver_prompt(
        problem, frames, registry, allowed, skill_text=frozen.texts[problem.task_type],
        profile=config.prompt_profile, official_prompt=public.official_prompt,
        remaining_requests=config.max_requests)
    messages = original_messages(problem, text)
    identity = {
        **public.identity(), "arm": arm, "arm_label": "skill_program",
        "evaluation_role": "test_development", "prompt_profile": VERSION,
        "artifact_sha256": prepared.artifact_sha256,
        "initial_messages_sha256": digest(messages),
        "tools_sha256": digest(sorted(allowed)),
        "allowed_tools": sorted(allowed), "images": image_identities(problem),
        "skill_sha256": frozen.identities[problem.task_type],
        "skills_manifest_sha256": frozen.manifest_sha256,
        "runner_config": asdict(config),
    }
    return prepared, messages, sorted(allowed), None, identity


class ProgramClient(BailianClient):
    """One HTTP attempt, full response capture outside the sanitized budget."""

    def __init__(self, config, **kwargs):
        super().__init__(
            model=config.model, temperature=config.temperature,
            max_output_tokens=config.max_tokens, timeout_s=config.timeout_s, **kwargs)
        self.config = config
        self.directory = None
        self.request_count = 0
        self.tool_seconds = 0.0
        self.tool_traces = {}

    def chat(self, messages, **kwargs):
        step = self.request_count
        self.request_count += 1
        payload = self.build_payload(messages, **kwargs)
        write_json(self.directory / f"{step:04d}.intent.json", {
            "messages_sha256": digest(messages), "tools_sha256": digest([]),
            "wire_payload_sha256": hashlib.sha256(payload_bytes(payload)).hexdigest(),
            "model_configuration": self.config.public_dict(),
            "request_parameters": kwargs,
            "text_parts": [part["text"] for message in messages
                           for part in message["content"] if part["type"] == "text"],
            "images": [
                {"url_sha256": digest(part["image_url"]["url"])}
                for message in messages for part in message["content"]
                if part["type"] == "image_url"],
        }, exclusive=True)
        text = None
        try:
            text = super().chat(messages, **kwargs)
            actual = self.last_response_identity
            if actual.get("model") != self.model or actual.get("model_reported") is not True:
                raise ValueError("actual_model_mismatch")
            return text
        finally:
            record = response_record(self, text)
            write_json(self.directory / f"{step:04d}.response.json", {
                **record, "message": {
                    "role": "assistant", "content": text,
                    "reasoning_content": record["reasoning_content"]},
                "model": record["response_identity"].get("model"),
            }, exclusive=True)

    def traced_registry(self, prepared, problem):
        registry, scene = build_registry(prepared, problem)
        call = registry.call_tool

        def traced(*args, **kwargs):
            step = self.request_count - 1
            started = time.monotonic()
            trace = {"tool": args[0], "arguments": args[1] if len(args) > 1 else kwargs.get("args", {}),
                     "status": "error"}
            try:
                result = call(*args, **kwargs)
                # Preserve exactly the public values returned by the existing registry.
                envelope = model_tool_response(result)
                if isinstance(envelope["data"], dict):
                    public = registry.public_fields.get(result.tool, ())
                    envelope["data"] = {
                        key: value for key, value in envelope["data"].items()
                        if key in public}
                trace.update(status=result.status, response=envelope,
                             authorization=result.authorization)
                return result
            except Exception as exc:
                trace["error_code"] = type(exc).__name__
                raise
            finally:
                elapsed = time.monotonic() - started
                self.tool_seconds += elapsed
                trace["elapsed_seconds"] = round(elapsed, 6)
                traces = self.tool_traces.setdefault(step, [])
                traces.append(trace)
                write_json(self.directory / f"{step:04d}.tools.json", traces)

        registry.call_tool = traced
        return registry, scene


class EpisodeGovernor:
    def __init__(self, shared, public):
        self.shared = shared
        self.public = public

    @contextmanager
    def request(self, category, context):
        with self.shared.request(category, {
            **context, "variant_id": "tools", "slot_key": self.public.problem.sample_id,
        }) as receipt:
            yield receipt

    def summary(self):
        return self.shared.summary()


def solve(public, prepared_directory, *, arm, client, governor, output_directory,
          frozen, config: SmallImageRunnerConfig, stop_event=None, expected_identity=None):
    if arm == "direct":
        row = native_solve(
            public, prepared_directory, arm=arm, client=client, governor=governor,
            output_directory=output_directory, stop_event=stop_event,
            expected_identity=expected_identity)
        row.pop("receipt_sha256")
        row.update(protocol=PROTOCOL, evaluation_role="test_development", arm_label="qwen_direct")
        row["receipt_sha256"] = digest(row)
        write_json(Path(output_directory) / "result.json", row)
        return row
    prepared, _, _, _, identity = prepare_request(
        public, prepared_directory, arm=arm, frozen=frozen, config=config)
    if identity != expected_identity:
        raise ValueError("program input changed after preflight")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "started.json", {
        "protocol": PROTOCOL, "input_identity": identity,
    }, exclusive=True)
    client.directory = directory
    started = time.monotonic()
    result = SmallImageSolver(
        client, config=config, governor=EpisodeGovernor(governor, public),
        skill_text=frozen.texts[public.problem.task_type],
        official_prompt=public.official_prompt,
        registry_builder=client.traced_registry).solve(public.problem, prepared=prepared)
    status = {
        "request_exhausted": "budget_exhausted",
        "input_error": "host_error", "precompute_error": "host_error",
        "prepared_identity_mismatch": "host_error", "prepared_input_mismatch": "host_error",
        "precompute_contract_error": "host_error",
    }.get(result.status, result.status)
    if status in {"service_error", "host_error", "budget_exhausted"} and stop_event:
        stop_event.set()
    usage = {}
    for round_row in result.rounds:
        for key, value in round_row.get("usage", {}).items():
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"} and (
                    type(value) in (int, float) and math.isfinite(value)):
                usage[key] = usage.get(key, 0) + value
    row = {
        "protocol": PROTOCOL, "arm": arm, "arm_label": "skill_program",
        "evaluation_role": "test_development",
        "sample_id": public.problem.sample_id, "source_split": public.problem.source_split,
        "task_type": public.problem.task_type, "input_identity": identity,
        "status": status, "answer": result.answer, "request_count": client.request_count,
        "attempted_tool_calls": result.attempted_tool_calls,
        "succeeded_tool_calls": result.succeeded_tool_calls, "usage": usage,
        "tool_seconds": round(client.tool_seconds, 6),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "final_text": result.rounds[-1].get("model_response") if result.rounds else None,
        "error_code": result.status if result.status != "answered" else None,
        "program_result": result.to_dict(),
    }
    row["receipt_sha256"] = digest(row)
    write_json(directory / "result.json", row, exclusive=True)
    return row
