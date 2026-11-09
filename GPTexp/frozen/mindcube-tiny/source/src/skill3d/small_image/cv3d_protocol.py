"""Official CV-Bench text and a compact, six-request program-solving prompt."""
from __future__ import annotations

import hashlib
import base64
import json
from pathlib import Path
import re

from .cv3d_protocol_v1 import VERSION as LEGACY_VERSION

VERSION = "cv3d-compact-program/2"
SUPPORTED_PROFILES = frozenset({LEGACY_VERSION, VERSION})


def build_compact_messages(problem, text, *, prepared, send_depth_image=False):
    """Preserve original RGB bytes and optionally attach its same-grid depth PNG."""
    from .blink_protocol import original_messages
    from .depth_image import render_depth_image

    messages = original_messages(problem, text)
    if not send_depth_image:
        return messages, None, None
    if prepared is None or 0 not in prepared.geometries:
        raise ValueError("depth image requires prepared same-view geometry")
    if prepared.problem["input_sha256"] != problem.input_sha256:
        raise ValueError("depth image input identity mismatch")
    geometry = prepared.geometries[0]
    height, width = prepared.frames[0].shape[:2]
    if (geometry["depth"].shape != (height, width)
            or geometry["valid_mask"].shape != (height, width)):
        raise ValueError("depth image must match original RGB grid")
    png, display = render_depth_image(geometry["depth"], geometry["valid_mask"])
    messages[0]["content"].extend([
        {"type": "text", "text": "Image 2: PREDICTED DEPTH MAP (same view_index=0)."},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}},
    ])
    return messages, png, display


def task_tool_names(task):
    if task not in {"Depth", "Distance"}:
        raise ValueError("unknown compact CV3D task")
    names = {"view_candidates", "view_depth_summary", "point_depth"}
    return names | ({"pair_point_distance"} if task == "Distance" else set())


class FrozenCV3DSkills:
    """Load both reviewed Skill versions with file hashes and revision provenance."""

    def __init__(self, manifest_path):
        path = Path(manifest_path).resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if (manifest.get("schema") != "cv3d-frozen-skills/1"
                or manifest.get("prompt_profile") not in SUPPORTED_PROFILES
                or manifest.get("evaluation_role") != "test_development"
                or set(manifest.get("tasks", {})) != {"Depth", "Distance"}):
            raise ValueError("invalid compact CV3D frozen Skill manifest")
        self.prompt_profile = manifest["prompt_profile"]
        self.texts = {}
        self.identities = {}
        for task, record in manifest["tasks"].items():
            skill_path = (path.parent / record["path"]).resolve()
            if not skill_path.is_relative_to(path.parent):
                raise ValueError("Skill path outside its frozen directory")
            data = skill_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            text = data.decode("utf-8")
            if digest != record["sha256"] or not text.strip() or len(text) > 4500:
                raise ValueError("frozen Skill content or identity changed")
            self.texts[task] = text
            self.identities[task] = digest
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()


class OfficialCV3DPrompts:
    """Retain only public official fields; reference answers are never retained."""

    def __init__(self, annotations):
        self.prompts = {}
        raw = Path(annotations).read_bytes()
        self.source_sha256 = hashlib.sha256(raw).hexdigest()
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task, question, choices, prompt = (
                row["task"], row["question"], row["choices"], row["prompt"])
            if (row.get("type") != "3D" or task not in {"Depth", "Distance"}
                    or not isinstance(question, str) or not question
                    or not isinstance(choices, list) or len(choices) != 2
                    or any(not isinstance(c, str) or not c for c in choices)
                    or not isinstance(prompt, str)):
                raise ValueError("invalid official CV3D public fields")
            expected = question + "\n" + "\n".join(
                f"({label}) {choice}" for label, choice in zip("AB", choices))
            if prompt != expected:
                raise ValueError("official CV3D prompt differs from question and choices")
            key = (task, question, tuple(choices))
            if key in self.prompts and self.prompts[key] != prompt:
                raise ValueError("ambiguous official CV3D prompt")
            self.prompts[key] = prompt
        if not self.prompts:
            raise ValueError("no official CV3D prompts")

    def for_problem(self, problem):
        if problem.benchmark != "cv3d":
            raise ValueError("official CV3D protocol cannot route another benchmark")
        choices = tuple(re.sub(r"^\([AB]\)\s+", "", c) for c in problem.options)
        key = (problem.task_type, problem.question, choices)
        if key not in self.prompts:
            raise ValueError("author-split question differs from official CV3D text")
        text = self.prompts[key]
        validate_official_prompt(problem, text)
        return text


def validate_official_prompt(problem, text):
    if (problem.benchmark != "cv3d" or problem.task_type not in {"Depth", "Distance"}
            or len(problem.image_paths) != 1 or len(problem.options) != 2):
        raise ValueError("compact CV3D needs one image and two options")
    choices = tuple(re.sub(r"^\([AB]\)\s+", "", c) for c in problem.options)
    expected = problem.question + "\n" + "\n".join(
        f"({label}) {choice}" for label, choice in zip("AB", choices))
    if not isinstance(text, str) or text != expected:
        raise ValueError("compact CV3D requires verified official prompt")


TOOL_DETAILS = {
    "point_depth": "data keys: pixel_xy, depth, n_valid, view_index. Returns median camera-depth from the valid 5x5 neighborhood.",
    "pair_point_distance": "data keys: a_xy, b_xy, distance, view_index. Euclidean distance between local median 3D camera points.",
    "view_candidates": "data keys: candidates, count. Candidates contain category, box_xyxy and review flags; hints may be incomplete and category confirmation does not certify mask geometry.",
    "view_depth_summary": "data keys: views (availability, n_valid, quantiles at 10/50/90 percent). Whole-view statistics, not target measurements.",
}


def compact_interfaces(interfaces):
    rows = []
    for line in interfaces.splitlines():
        signature = line.split(": ", 1)[0]
        name = signature.removeprefix("- ").split("(", 1)[0]
        if name not in TOOL_DETAILS:
            raise ValueError("unexpected compact tool interface")
        rows.append(f"{signature}: {TOOL_DETAILS[name]}")
    return "\n".join(rows)


def program_contract(task):
    field = "depth" if task == "Depth" else "distance"
    return (
        "Return only a complete Python program defining solve(ctx); the host calls it. "
        "Measurement functions live in ctx.tools. The global names ReturnAnswer, "
        "AnswerPayload and YieldObservations are already provided: do not import them "
        "or prefix them with ctx or ctx.tools.\n"
        "Tool returns are dictionaries: r['status'] == 'ok', r['data'][field], "
        f"r['result_id']; never r.status or r.data. Read the measurement as r['data']['{field}'].\n"
        "End with ReturnAnswer(AnswerPayload(value=letter, unit='option', "
        "basis='mixed')) when measurements contribute, or basis='visual_estimate' "
        "for a visual choice. letter must be the option you selected. "
        "To inspect measurements next round, call YieldObservations([r['result_id'], ...], "
        "reason='target-to-coordinate mapping and next check, at most 400 characters'). "
        "Both global controls end the cell immediately. Only reason is carried as your "
        "unverified working note; it cannot replace measured evidence.\n"
        "Complete API example (illustrative availability check, not a required action):\n"
        "```python\n"
        "def solve(ctx):\n"
        "    r = ctx.tools.view_depth_summary()\n"
        "    YieldObservations([r['result_id']], reason='Availability checked; locate the targets next.')\n"
        "```\n"
        "Use basic Python and documented interfaces. No ReadObservation, file/network access, "
        "host inspection, interface mutation or raw arrays. "
        "Correctable format/API/Python errors receive feedback within the same five program slots; "
        "security violations, tool failures, timeouts and service failures end the attempt."
    )


def skill_for_stage(skill_text, *, finalizing):
    if finalizing and "\n## Final assessment\n" in skill_text:
        return "## Final assessment\n" + skill_text.split("\n## Final assessment\n", 1)[1]
    return skill_text


def build_compact_prompt(problem, frames, *, official_prompt, skill_text,
                         feedback, interfaces, finalizing, remaining_requests,
                         send_depth_image=False, context_state=None):
    from .cv3d_context import CompactContext, VERSION as CONTEXT_VERSION
    validate_official_prompt(problem, official_prompt)
    h, w = frames[0].shape[:2]
    text = official_prompt + "\n\n" + (
        f"Image 1: ORIGINAL RGB, {w} x {h} pixels, view_index=0. "
        f"Use original pixel coordinates: x right, y down; 0 <= x < {w}, 0 <= y < {h}. "
        "The colored boxes identify the question's instances. "
        "Choose coordinates on this grid; do not assume an unspecified preview scale.\n"
    )
    if send_depth_image:
        from .depth_image import depth_image_description
        text += depth_image_description(w, h) + "\n"
    text += "Measurement units are per_view_relative from MoGe2, not calibrated meters.\n"
    skill_text = skill_for_stage(skill_text, finalizing=finalizing)
    text += f"Task reasoning skill:\n{skill_text.strip()}\n\n"
    if not finalizing:
        text += "Program tools:\n" + compact_interfaces(interfaces) + "\n"
        text += (
            "Samples use radius 2 around the rounded pixel (5x5, clipped at image edges), "
            "with at least 3 finite positive valid pixels; no object-mask filtering. "
            "n_valid measures predicted support, not object membership.\n\n"
        )
    if not isinstance(feedback, dict):
        empty = CompactContext()
        empty.add_observations(feedback, 0)
        feedback = empty.feedback()
    context_state = context_state or CompactContext().state()
    if finalizing and context_state.get("last_error"):
        context_state = {
            **context_state, "last_error": {"code": context_state["last_error"]["code"]}}
    stage = "final answer" if finalizing else "program"
    text += f"Current stage: {stage}; round {7 - remaining_requests}/6. Context {CONTEXT_VERSION}.\n"
    text += f"Remaining model requests: {remaining_requests}, including finalization.\n"
    text += ("No previous round or tool observations yet.\n" if remaining_requests == 6 else
             "Previous Python variables have been cleared. Earlier code and thinking are not "
             "available; use the observations, working notes and error feedback below.\n")
    text += (
        "Progress: " + json.dumps(context_state, ensure_ascii=True, allow_nan=False) + "\n"
        "Feedback constants apply to their observations; source_tool and result_id identify "
        "each measured result. Model notes are fallible context, not verified measurements.\n"
    )
    text += "Valid observations: " + json.dumps(
        feedback, ensure_ascii=True, allow_nan=False) + "\n\n"
    if finalizing:
        return text + (
            'Finalization: return ONLY {"answer":"<A or B>","reasoning":"<brief explanation>"}. '
            "Select one supplied option using the images and existing observations. "
            "Tools are disabled. Do not output Python or request another round.")
    return text + "Output contract:\n" + program_contract(problem.task_type)
