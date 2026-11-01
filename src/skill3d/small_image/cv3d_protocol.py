"""Official CV-Bench text and a compact, six-request program-solving prompt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

VERSION = "cv3d-compact-program/1"


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
                or manifest.get("prompt_profile") != VERSION
                or manifest.get("evaluation_role") != "test_development"
                or set(manifest.get("tasks", {})) != {"Depth", "Distance"}):
            raise ValueError("invalid compact CV3D frozen Skill manifest")
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


def build_compact_prompt(problem, frames, *, official_prompt, skill_text,
                         feedback, interfaces, finalizing, remaining_requests):
    validate_official_prompt(problem, official_prompt)
    h, w = frames[0].shape[:2]
    text = official_prompt + "\n\n" + (
        f"Image 1 is the ORIGINAL RGB image, {w} x {h} pixels, view_index=0. "
        "Coordinates use original pixels: x right, y down. The colored annotations "
        "are object boxes. Depth and 3D point distances are same-view MoGe2 "
        "predictions in per_view_relative units, not calibrated meters. "
        "point_depth samples a radius-2 (5x5) neighborhood and needs at least "
        "3 valid finite positive pixels; it does not filter by object masks. "
        "pair_point_distance measures Euclidean distance between local median "
        "predicted 3D points; it is available for Distance only. "
        "Candidate detections are optional, non-exhaustive localization hints; "
        "category confirmation does not validate a mask or geometry.\n"
    )
    text += f"Task reasoning skill:\n{skill_text}\n"
    text += "Valid observations: " + json.dumps(
        list(feedback), ensure_ascii=True, allow_nan=False) + "\n"
    text += (
        f"Remaining model requests: {remaining_requests}, including finalization. "
        "Previous Python variables have been cleared; use the observations above "
        "without assuming previous code or reasoning is available.\n"
    )
    if finalizing:
        return text + (
            'Finalization: return ONLY a JSON object with "answer" (one option letter) '
            'and optional "reasoning" (a concise explanation). '
            "No code, tool calls, or observation reads. "
            "No further evidence requests are available.")
    return text + (
        "Program tools:\n" + interfaces + "\n"
        "Successful returns are dictionaries with status='ok', values in data, "
        "and result_id for yielding; there is no outer ok boolean. A failed or "
        "untrusted executed call ends the attempt; do not probe failing calls or "
        "assume try/except enables recovery. Do not use ReadObservation.\n"
        "Return a complete def solve(ctx) Python program. Call ctx.tools.<name>(...) "
        "using the listed arguments. Terminate with "
        "ReturnAnswer(AnswerPayload(value=selected_letter, unit='option', "
        "basis='mixed')) when measurements support the choice, or "
        "basis='visual_estimate' for an image-based choice. To let the model inspect "
        "successful measurements, terminate with YieldObservations([result_id, ...]). "
        "Both controls end the cell immediately. Use only supplied evidence and "
        "allowed tools; do not access files, networks, host internals or raw arrays. "
        "Return one supplied option; do not abstain."
    )
