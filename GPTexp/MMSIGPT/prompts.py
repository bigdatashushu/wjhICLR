"""MMSI-specific GPT prompts; keep future prompt changes in this module."""
from __future__ import annotations

import json
import inspect

from .public_input import INTERFACE_VERSION, PROMPT_VERSION, public_input


def build_solver_prompt(problem, frames, registry, allowed_tools, *,
                        finalizing=False, feedback=(), skill_text="",
                        profile=None, official_prompt=None, remaining_requests=0,
                        send_depth_image=False, context_state=None):
    from skill3d.sandbox.ast_guard import FORBIDDEN_CALLS, RESERVED_NAMES
    from skill3d.sandbox.safe_runtime import NUMERICAL_EXPORTS
    from skill3d.small_image.runner import build_solver_prompt as shared_prompt

    if send_depth_image or context_state is not None:
        raise ValueError("mmsi_requires_original_views_and_standard_feedback")
    options = {
        "finalizing": finalizing, "feedback": feedback, "skill_text": skill_text,
        "official_prompt": official_prompt, "remaining_requests": remaining_requests,
        "send_depth_image": send_depth_image, "context_state": context_state,
    }
    if profile is not None:
        options["profile"] = profile
    supported = inspect.signature(shared_prompt).parameters
    prompt = shared_prompt(problem, frames, registry, allowed_tools,
                           **{key: value for key, value in options.items() if key in supported})
    prefix, input_marker, tail = prompt.partition("\nInput: ")
    _old_input, observation_marker, observations = tail.partition("\nValid observations: ")
    if not input_marker or not observation_marker:
        raise ValueError("shared_prompt_input_contract_changed")
    prompt = (prefix + input_marker + json.dumps(public_input(problem, frames), ensure_ascii=True)
              + observation_marker + observations)
    # Use the shared framing and geometry contract, but one final-output contract.
    if finalizing:
        common, _marker, _tail = prompt.rpartition("Finalization: ")
        return common + (
            f"MMSI contract {PROMPT_VERSION} / {INTERFACE_VERSION}.\n"
            'Final answer: emit exactly {"answer":"A"} using the selected original '
            "option letter. Emit only that JSON object, with no reasoning field, "
            "code, tool calls, or observation reads. No further requests are available.")
    exports = {name: sorted(values) for name, values in sorted(NUMERICAL_EXPORTS.items())}
    return prompt + (
        f"\nMMSI contract {PROMPT_VERSION} / {INTERFACE_VERSION}.\n"
        "ctx is read-only. ctx.input is the Input mapping above; ctx.question is its "
        "question string; ctx.options is an ordered tuple of read-only mappings "
        "with 'label' and 'text' (it is not a tuple of strings). "
        "Use ctx.input['options'][i]['label'] or ctx.options[i]['label']; "
        "ctx.input['views'] supplies image sizes. ctx.frames holds immutable original "
        "RGB arrays; ctx.scene exposes only the documented question-state fields; "
        "ctx.tools exposes exactly the Allowed tools above. There are no paths, "
        "reference answers, prepared geometry, or host handles in ctx.input.\n"
        "ReturnAnswer, AnswerPayload and YieldObservations are already injected: "
        "call them directly and do not import them. There is no harness3d module "
        "available to generated code. Do not call solve yourself; the host calls it.\n"
        "AnswerPayload is keyword-only. unit must be 'option'; basis must be exactly "
        "'visual_estimate' for visual reasoning or 'mixed' when using tool facts. "
        "Put any explanation in code comments, never after the basis string. "
        "A failed/denied tool terminates this episode as run_error; catching the "
        "exception cannot authorize its output. Use original-image reasoning "
        "directly when the available tools do not answer the requested relation.\n"
        f"Forbidden calls: {', '.join(sorted(FORBIDDEN_CALLS))}. "
        f"Do not rebind reserved names: {', '.join(sorted(RESERVED_NAMES))}.\n"
        "Numerical imports are restricted to these module/member lists; they "
        "do not expose file IO, reflection or mutable module state: "
        + json.dumps(exports, ensure_ascii=True) + "\n"
    )


def direct_prompt(problem):
    """Preserve the original direct-arm prompt and option order."""
    return (
        "Answer this spatial multiple-choice question using the supplied original images in their exact order. "
        "Reason about the requested relationship and choose one option. Return JSON with "
        '"reasoning" (a concise explanation) and "answer" (one original option letter).\n'
        + json.dumps({
            "question": problem.question,
            "options": [
                {"label": chr(65 + i), "text": value}
                for i, value in enumerate(problem.options)
            ],
        }, ensure_ascii=False)
    )
