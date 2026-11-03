"""Frozen paired-run adapter and passive trace collection for the local solver."""
import base64
from dataclasses import asdict
import math
from pathlib import Path
import time

from skill3d.schemas.tool_response import model_tool_response
from skill3d.small_image.blink_protocol import image_identities
from skill3d.small_image.depth_image import VERSION as DEPTH_IMAGE_VERSION
from skill3d.small_image.runner import _validate_prepared, load_problem_frames

from CV3D.common import digest, write_json
from CV3D.data import verified_prepared
from CV3D.inference import prepare_request as native_prepare, solve as native_solve
from CV3D.program import ProgramClient as TraceClient, EpisodeGovernor

from .context import ReviewContext, VERSION as CONTEXT_VERSION
from .grounding import VERSION as GROUNDING_VERSION
from .protocol import build_messages, build_prompt, OVERLAY_VERSION
from .runner import CV3DSolver
from .tools import build_registry, allowed_tools

PROTOCOL = "cv3d-isolated-box-review-versus-direct/1"


def prepare_request(public, directory, *, arm, frozen, config):
    if arm == "direct":
        return native_prepare(public, arm=arm)
    if arm != "tools" or frozen.prompt_profile != config.prompt_profile:
        raise ValueError("arm or frozen Skill profile mismatch")
    problem = public.problem
    prepared = verified_prepared(public, directory)
    frames = load_problem_frames(problem, original_image_bytes=True)
    if _validate_prepared(prepared, problem, frames):
        raise ValueError("prepared validation failed")
    registry, scene = build_registry(prepared, problem)
    allowed = allowed_tools(registry, scene)
    text = build_prompt(
        problem, frames, registry, allowed, official_prompt=public.official_prompt,
        skill_text=frozen.texts[problem.task_type], context=ReviewContext(),
        remaining_requests=config.max_requests, send_depth_image=config.send_depth_image,
        catalog=scene.catalog.public())
    messages, png, display = build_messages(problem, text, prepared=prepared, config=config)
    identity = {
        **public.identity(), "arm": arm, "arm_label": "skill_program",
        "evaluation_role": "test_development", "prompt_profile": config.prompt_profile,
        "artifact_sha256": prepared.artifact_sha256, "initial_messages_sha256": digest(messages),
        "tools_sha256": digest(sorted(allowed)), "allowed_tools": sorted(allowed),
        "images": image_identities(problem), "skill_sha256": frozen.identities[problem.task_type],
        "skills_manifest_sha256": frozen.manifest_sha256, "runner_config": asdict(config),
        "context_protocol": CONTEXT_VERSION, "annotation_protocol": GROUNDING_VERSION,
        "annotation_catalog_sha256": scene.catalog.identity, "overlay_protocol": OVERLAY_VERSION,
    }
    if display is not None:
        identity.update(image_input_protocol=DEPTH_IMAGE_VERSION, depth_display=display)
        identity["images"].append({
            "view_index": 0, "role": "predicted_depth", "sha256": display["png_sha256"],
            "width": display["width"], "height": display["height"], "mime_type": "image/png"})
    return prepared, messages, sorted(allowed), png, identity


class ProgramClient(TraceClient):
    """Reuse transport tracing; choose a local registry without changing globals."""
    def chat(self, messages, **kwargs):
        parts = messages[0]["content"]
        if any(p.get("text", "").startswith("SAMPLING OVERLAY") for p in parts):
            url = parts[-1]["image_url"]["url"]
            with (self.directory / f"{self.request_count:04d}.sampling_overlay.png").open("xb") as stream:
                stream.write(base64.b64decode(url.split(",", 1)[1], validate=True))
        return super().chat(messages, **kwargs)

    def traced_registry(self, prepared, problem):
        registry, scene = build_registry(prepared, problem)
        original_call = registry.call_tool

        def traced(name, args, *rest, **kwargs):
            step = self.request_count - 1
            started = time.monotonic()
            trace = {"tool": name, "arguments": args, "status": "error"}
            try:
                result = original_call(name, args, *rest, **kwargs)
                trace.update(status=result.status, error_code=result.error_code,
                             authorization=result.authorization)
                if result.status == "ok" and not result.error and not result.error_code:
                    envelope = model_tool_response(result)
                    envelope["data"] = {k: v for k, v in envelope["data"].items()
                                        if k in registry.public_fields[result.tool]}
                    trace["response"] = envelope
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


def solve(public, prepared_directory, *, arm, client, governor, output_directory,
          frozen, config, stop_event=None, expected_identity=None):
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
    prepared, _, _, depth_png, identity = prepare_request(
        public, prepared_directory, arm=arm, frozen=frozen, config=config)
    if identity != expected_identity:
        raise ValueError("program input changed after preflight")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if depth_png is not None:
        with (directory / "depth_preview.png").open("xb") as stream:
            stream.write(depth_png)
    write_json(directory / "started.json", {"protocol": PROTOCOL, "input_identity": identity},
               exclusive=True)
    client.directory = directory
    started = time.monotonic()
    result = CV3DSolver(
        client, config=config, governor=EpisodeGovernor(governor, public),
        skill_text=frozen.texts[public.problem.task_type], official_prompt=public.official_prompt,
        registry_builder=client.traced_registry).solve(public.problem, prepared=prepared)
    # Kernel assigns occurrence IDs after the registry returns. Reconcile passive
    # traces with those authoritative IDs, including repeated same-argument calls.
    for step, round_row in enumerate(result.rounds):
        observed = iter(round_row.get("tool_result_receipts", ()))
        traces = client.tool_traces.get(step, [])
        for trace in traces:
            if "response" in trace:
                envelope = next(observed, None)
                if envelope is None or envelope["source_tool"] != trace["tool"]:
                    raise ValueError("tool trace does not match kernel observations")
                trace["response"] = envelope
                trace["authorization"]["result_id"] = envelope["result_id"]
        if traces:
            write_json(directory / f"{step:04d}.tools.json", traces)
    status = {
        "request_exhausted": "budget_exhausted", "input_error": "host_error",
        "precompute_error": "host_error", "prepared_identity_mismatch": "host_error",
        "prepared_input_mismatch": "host_error", "precompute_contract_error": "host_error",
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
        "evaluation_role": "test_development", "sample_id": public.problem.sample_id,
        "source_split": public.problem.source_split, "task_type": public.problem.task_type,
        "input_identity": identity, "status": status, "answer": result.answer,
        "request_count": client.request_count, "attempted_tool_calls": result.attempted_tool_calls,
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
