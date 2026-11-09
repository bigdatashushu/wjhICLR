"""Frozen CV3D v1 prompt text, retained for explicit historical replay."""
import json

VERSION = "cv3d-compact-program/1"


def build_compact_prompt(problem, frames, *, official_prompt, skill_text,
                         feedback, interfaces, finalizing, remaining_requests):
    from .cv3d_protocol import validate_official_prompt

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
