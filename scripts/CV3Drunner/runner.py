"""CV3D-only control loop. Shared sandbox classes are reused without mutation."""
from dataclasses import replace
import hashlib
from tempfile import TemporaryDirectory

from skill3d.sandbox.action_check import check_action
from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas.tool_response import model_tool_response
from skill3d.small_image.blink_protocol import explicit_answer, response_record
from skill3d.small_image.cv3d_context import correctable_guard, REPAIRABLE_RUNTIME
from skill3d.small_image.cv3d_protocol import validate_official_prompt, skill_for_stage
from skill3d.small_image.runner import (
    _result, _validate_prepared, load_problem_frames, parse_small_image_answer,
    public_model_identity, validate_request_config,
)
from skill3d.synthesis.program_assembler import assemble_program, SynthesisError
from skill3d.synthesis.request_governor import (
    GovernedClient, RequestBudgetExhausted, RequestGovernor, RequestGovernorConfig,
)

from .config import RunnerConfig
from .context import ReviewContext
from .grounding import digest
from .protocol import build_messages, build_prompt, sampling_overlay
from .tools import VERSION as TOOL_VERSION, TOOL_NAMES, MEASUREMENTS, allowed_tools, build_registry


class CV3DSolver:
    def __init__(self, client, *, config=None, governor=None, skill_text="",
                 official_prompt=None, registry_builder=build_registry):
        self.client = client
        self.config = config or RunnerConfig()
        self.governor = governor
        self.skill_text = skill_text
        self.official_prompt = official_prompt
        self.registry_builder = registry_builder

    def solve(self, problem, *, prepared=None):
        config = self.config
        identity = public_model_identity(self.client, seed=config.seed)
        counts, rounds, frames, kernel = {}, [], (), None

        def finish(status, answer=None):
            result = _result(
                problem, status=status, model_identity=identity, frames=frames,
                counts=counts, rounds=rounds, kernel=kernel, prepared=prepared, answer=answer,
                prompt_version=config.prompt_profile, evaluation_role=config.evaluation_role)
            measured = kernel and any(r.tool in MEASUREMENTS and r.status == "ok"
                                      and not r.error and not r.invalidated_by for r in kernel.tool_results)
            result = replace(result, tool_version=TOOL_VERSION, receipt_sha256="",
                             basis=("mixed" if measured else "visual_estimate") if answer else None)
            return replace(result, receipt_sha256=digest(result.to_dict()))

        try:
            validate_official_prompt(problem, self.official_prompt)
            frames = load_problem_frames(problem, original_image_bytes=True)
            validate_request_config(self.client, config)
            if (prepared is None or not isinstance(self.skill_text, str)
                    or not self.skill_text.strip() or len(self.skill_text) > 4500):
                return finish("input_error")
            status = _validate_prepared(prepared, problem, frames)
            if status:
                return finish(status)
            registry, scene = self.registry_builder(prepared, problem)
            if (set(registry.names()) != TOOL_NAMES or scene.question_tool_scope != "small_image"
                    or scene.frame != "per_view_camera"):
                return finish("precompute_contract_error")
            allowed = allowed_tools(registry, scene)
            kernel = RestrictedNamespaceKernel(
                registry, scene, frames=list(frames), episode_id=problem.sample_id,
                cell_timeout_s=config.cell_timeout_s)
        except (OSError, TypeError, ValueError):
            return finish("input_error")

        context = ReviewContext()
        overlay_png, overlay_display = None, None
        with TemporaryDirectory(prefix="cv3d-box-budget-") as budget_root:
            local = RequestGovernor(budget_root, RequestGovernorConfig(6, 1))
            client = GovernedClient(self.client, local, category="solver")
            if self.governor is not None:
                client = GovernedClient(client, self.governor, category="solver")

            def request(phase):
                prompt = build_prompt(
                    problem, frames, registry, allowed if phase == "solve" else set(),
                    official_prompt=self.official_prompt, skill_text=self.skill_text,
                    context=context, remaining_requests=6 - sum(counts.values()),
                    finalizing=phase == "finalization", send_depth_image=config.send_depth_image,
                    catalog=scene.catalog.public(), overlay_display=overlay_display)
                messages = build_messages(
                    problem, prompt, prepared=prepared, config=config, overlay_png=overlay_png)[0]
                row = {"phase": phase, "status": "started", "tool_calls": 0,
                       "delivered_measurement_ids": [
                           r["result_id"] for r in context.observations if r["source_tool"] in MEASUREMENTS],
                       "sampling_overlay": overlay_display}
                rounds.append(row)
                before = local.summary()["reserved_requests"]
                try:
                    kwargs = dict(max_tokens=config.max_tokens, seed=config.seed,
                                  temperature=0.0, enable_thinking=config.enable_thinking)
                    row["request_parameters"] = kwargs
                    row["messages_sha256"] = digest(messages)
                    row["frozen_skill_sha256"] = hashlib.sha256(self.skill_text.encode()).hexdigest()
                    delivered = skill_for_stage(self.skill_text, finalizing=phase == "finalization").strip()
                    row["delivered_skill_sha256"] = hashlib.sha256(delivered.encode()).hexdigest()
                    text = client.chat(messages, **kwargs)
                    if config.capture_responses:
                        row.update(response_record(client, text))
                    if config.expected_model:
                        actual = getattr(client, "last_response_identity", {})
                        if (actual.get("model") != config.expected_model
                                or actual.get("model_reported") is not True):
                            raise ValueError("actual_model_mismatch")
                except RequestBudgetExhausted:
                    row["status"] = "request_exhausted"
                    raise
                except Exception:
                    row["status"] = "service_error"
                    raise
                finally:
                    counts[phase] = counts.get(phase, 0) + (
                        local.summary()["reserved_requests"] - before)
                row["status"] = "received"
                return text, row

            def record(row, number, *, cell=None, error_code=None):
                nonlocal overlay_png, overlay_display
                observed = []
                if cell is not None:
                    for item in kernel.cell_results:
                        if item.status != "ok" or item.error or item.invalidated_by:
                            continue
                        envelope = model_tool_response(item)
                        envelope["data"] = {
                            k: v for k, v in envelope["data"].items()
                            if k in registry.public_fields[item.tool]}
                        observed.append(envelope)
                row["observations"] = observed
                context.add_observations(observed, number)
                fresh = [r["result_id"] for r in observed if r["source_tool"] in MEASUREMENTS]
                if fresh:
                    context.review = {
                        "required_next_request": True, "new_result_ids": fresh,
                        "provisional_answer": row.get("provisional_answer"),
                        "instruction": "Review sampled surfaces, color roles, values and visual conflicts.",
                    }
                else:
                    context.review = None
                context.record(number, row["status"], cell=cell, error_code=error_code,
                               note=cell.yielded_reason if cell else "",
                               result_ids=cell.yielded_result_ids if cell else ())
                overlay_png, overlay_display = sampling_overlay(
                    frames[0], context.observations, context.origins)
                row["context_state_after"] = context.state()
                return fresh

            for number in range(1, 6):
                try:
                    text, row = request("solve")
                except RequestBudgetExhausted:
                    return finish("request_exhausted")
                except Exception:
                    return finish("service_error")
                try:
                    program = assemble_program(text)
                except SynthesisError:
                    row["status"] = "program_parse_error"
                    record(row, number, error_code="program_parse_error")
                    continue
                row["program_source"] = program.program_source
                guarded = ast_guard(program.program_source, allowed_tools=allowed)
                if not guarded.ok or "ReadObservation" in program.program_source:
                    row["status"] = "program_rejected"
                    if "ReadObservation" not in program.program_source and correctable_guard(guarded.violations):
                        row["correctable"] = True
                        record(row, number, error_code="interface_misuse")
                        continue
                    return finish("program_rejected")
                if not check_action(program.program_source, tool_names=allowed).ok:
                    row["status"] = "program_missing_action"
                    record(row, number, error_code="program_missing_action")
                    continue
                # Only trustworthy or repairable preceding cells reach this reset.
                kernel.reset_user_namespace()
                cell = kernel.run_cell(program.program_source)
                row["tool_calls"] = len(kernel.cell_calls)
                # Preserve occurrence IDs even if the program later fails. These
                # are passive receipts; terminal errors cannot enter a next round.
                row["tool_result_receipts"] = []
                for item in kernel.cell_results:
                    if item.status == "ok" and not item.error and not item.invalidated_by:
                        envelope = model_tool_response(item)
                        envelope["data"] = {
                            k: v for k, v in envelope["data"].items()
                            if k in registry.public_fields[item.tool]}
                        row["tool_result_receipts"].append(envelope)
                if cell.error_code or cell.answer_untrusted or kernel.answer_untrusted:
                    row["status"] = "run_error"
                    row["execution_error"] = {
                        "code": cell.error_code, "type": cell.error_type,
                        "line": cell.error_line, "program_line": cell.error_source_line[:240],
                        "answer_untrusted": cell.answer_untrusted}
                    if (not cell.answer_untrusted and not kernel.answer_untrusted
                            and not kernel.contract_violations
                            and cell.error_code in {"violation_runtime", "violation_syntax"}
                            and cell.error_type in REPAIRABLE_RUNTIME):
                        row["correctable"] = True
                        record(row, number, cell=cell, error_code=(
                            "interface_misuse" if cell.error_type == "AttributeError" else "python_error"))
                    else:
                        return finish("run_error")
                elif cell.answer is not None:
                    try:
                        payload = kernel.answer_slot.payload
                        if payload is None or payload.unit != "option":
                            raise ValueError("option answer required")
                        answer = parse_small_image_answer(payload.value, problem.options)
                    except (ValueError, TypeError):
                        row["status"] = "invalid_answer"
                        record(row, number, cell=cell, error_code="invalid_answer")
                    else:
                        fresh = any(r.tool in MEASUREMENTS for r in kernel.cell_results)
                        if fresh:
                            row.update(status="measurement_review_required", provisional_answer=answer)
                            record(row, number, cell=cell)
                        else:
                            row["status"] = "answered"
                            record(row, number, cell=cell)
                            return finish("answered", answer)
                else:
                    valid = {r.result_id for r in kernel.tool_results
                             if r.status == "ok" and not r.error and not r.invalidated_by}
                    if set(cell.yielded_result_ids) - valid:
                        row["status"] = "invalid_observation_reference"
                        record(row, number, cell=cell, error_code="invalid_observation_reference")
                    else:
                        row["status"] = "yielded" if cell.terminated == "yield" else "no_answer"
                        record(row, number, cell=cell,
                               error_code="no_answer" if cell.terminated != "yield" else None)
                if context.feedback_chars() > config.max_feedback_chars:
                    return finish("feedback_limit")

            kernel.reset_user_namespace()
            kernel.set_tools_enabled(False)
            try:
                text, row = request("finalization")
            except RequestBudgetExhausted:
                return finish("request_exhausted")
            except Exception:
                return finish("service_error")
            try:
                answer, method = explicit_answer(text, problem.options)
                row["answer_parse_method"] = method
                source = f"ReturnAnswer(AnswerPayload(value={answer!r}, unit='option', basis='mixed'))"
                if (not ast_guard(source, allowed_tools=set()).ok
                        or not check_action(source, finalizing=True, tool_names=TOOL_NAMES).ok):
                    raise ValueError("host finalization invalid")
                cell = kernel.run_cell(source)
                if cell.error_code or cell.answer_untrusted or kernel.cell_calls:
                    row["status"] = "run_error"
                    return finish("run_error")
            except (ValueError, TypeError, SynthesisError):
                row["status"] = "invalid_finalization"
                return finish("invalid_finalization")
            row["status"] = "answered"
            return finish("answered", answer)
