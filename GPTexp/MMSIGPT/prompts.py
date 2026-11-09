"""MMSI-specific GPT prompts; keep future prompt changes in this module."""
from __future__ import annotations

import json
import inspect

from .public_input import INTERFACE_VERSION, PROMPT_VERSION, public_input


def spatial_guidance(task_type):
    """Describe the spatial binding required by the public question category."""
    category = task_type.replace("–", "-").replace("—", "-")
    category = category.removeprefix("Positional Relationship (").removesuffix(")")
    guidance = {
        "Cam.-Cam.": (
            "Distinguish camera-position displacement from camera-view rotation. "
            "Use the camera named as the origin in the question, which may be photo 2. "
            "Bind shared landmark instances and their facing surfaces across views; "
            "a landmark's screen shift alone does not establish camera translation."
        ),
        "Cam.-Obj.": (
            "Bind the observer's location and facing from the specified photo or "
            "imagined action, such as sitting on a sofa. Locate the exact target "
            "object across the original views, then express its offset in that "
            "observer's axes."
        ),
        "Cam.-Reg.": (
            "Use the specified photo as the observer pose and locate the requested "
            "region or doorway through shared walls and fixtures. A doorway seen "
            "in a mirror belongs to the reflected scene; connect that scene to the "
            "specified observer pose before assigning a physical direction."
        ),
        "Obj.-Obj.": (
            "Keep the reference object and target object in their requested order. "
            "Bind object front from its stated orientation or visible facing "
            "features. For compass directions, ground the stated directional "
            "anchor in the floor plane and establish the perpendicular axis."
        ),
        "Obj.-Reg.": (
            "Bind the exact reference object, target region, and stated facing "
            "cue, such as a cabinet front or outward doorway direction. The "
            "photographer's screen axes need not match the object's or doorway's "
            "axes; express the region's offset in the requested frame."
        ),
        "Reg.-Reg.": (
            "Identify the reference and target regions using visible fixtures and "
            "shared boundaries. If the question places a person in a region, "
            "use the stated action as their facing cue; for going upstairs, bind "
            "the ascending direction from the stair structure. Compass questions "
            "use the stated wall or landmark orientation."
        ),
    }
    return guidance.get(category, (
        "Bind the origin, target, and orientation from the question wording."
    ))


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
        "For public trajectory review, assign a top-level evidence_audit dict before "
        "defining solve(ctx). It must have exactly these keys: "
        "'observations': 1-8 {'view_index': zero-based original image index, "
        "'fact': short visible observation} rows; 'reference_frame': short statement "
        "of the requested origin and facing direction; 'relation': short spatial "
        "conclusion; 'selected_option': original option letter; 'tool_rationale': "
        "short explanation of which available measurement helps, or why visual "
        "evidence suffices. This is a concise evidence summary, not private reasoning. "
        "Use at most 300 characters per fact and 400 per other text field. "
        "Keep visible facts distinct from inferred spatial relations. Do not claim "
        "a tool measurement unless you actually called that tool. Tools are optional; "
        "within-view relative depth or point distance cannot establish camera pose, "
        "compass orientation or cross-view metric geometry. ReturnAnswer still "
        "submits the answer; evidence_audit never authorizes tools or changes it.\n"
        "Spatial evidence binding: question wording determines the origin, target "
        "and axes, even when the task category is broader. Figure/photo N means "
        "original view_index N-1; the last image means the last supplied original "
        "view. Record visible landmarks by view_index and distinguish separate "
        "instances, front/back surfaces, and mirror reflections. In reference_frame, "
        "name the origin and the grounded facing or compass cue, including its "
        "view when applicable. An imagined person's facing can differ from the "
        "photographer's. In relation, state the target offset after expressing it "
        "in the requested axes; diagonal options require both directional signs. "
        "Image rows are not a floor-plane compass axis. Keep observations as "
        "visible facts and relation as a spatial estimate; name any unresolved "
        "landmark or axis binding instead of describing it as observed. These are "
        "compact public evidence statements, not a private reasoning transcript.\n"
        "Task-specific binding: " + spatial_guidance(problem.task_type) + "\n"
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
