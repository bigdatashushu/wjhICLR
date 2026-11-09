"""Original RGB, optional predicted depth and host sampling overlay protocol."""
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from skill3d.small_image.cv3d_protocol import (
    build_compact_messages, validate_official_prompt, skill_for_stage,
)
from skill3d.small_image.runner import _interfaces

from . import VERSION
from .context import VERSION as CONTEXT_VERSION
from .tools import MEASUREMENTS

OVERLAY_VERSION = "cv3d-sample-overlay/1"


class FrozenSkills:
    def __init__(self, manifest_path):
        path = Path(manifest_path).resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw)
        if (manifest.get("schema") != "cv3d-frozen-skills/1"
                or manifest.get("prompt_profile") != VERSION
                or manifest.get("evaluation_role") != "test_development"
                or set(manifest.get("tasks", {})) != {"Depth", "Distance"}):
            raise ValueError("invalid CV3Drunner frozen Skill manifest")
        self.prompt_profile = VERSION
        self.texts, self.identities = {}, {}
        for task, record in manifest["tasks"].items():
            skill_path = (path.parent / record["path"]).resolve()
            if not skill_path.is_relative_to(path.parent):
                raise ValueError("Skill path outside frozen directory")
            data = skill_path.read_bytes()
            text = data.decode("utf-8")
            sha = hashlib.sha256(data).hexdigest()
            if sha != record["sha256"] or not text.strip() or len(text) > 4500:
                raise ValueError("frozen Skill content changed")
            self.texts[task], self.identities[task] = text, sha
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()


def sampling_overlay(frame, observations, origins):
    """Draw the exact rounded 5x5 sample patches; does not alter numeric inputs."""
    measurements = [r for r in observations if r["source_tool"] in MEASUREMENTS]
    if not measurements:
        return None, None
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    points = []
    for observation in measurements:
        data, rid = observation["data"], observation["result_id"]
        ends = ([(data["box_id"], data["pixel_xy"])] if "pixel_xy" in data else
                [(data["a_box_id"], data["a_xy"]), (data["b_box_id"], data["b_xy"])])
        for box_id, xy in ends:
            x, y = (int(round(v)) for v in xy)
            draw.rectangle((x - 2, y - 2, x + 2, y + 2), outline="white", width=1)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline="yellow", width=2)
            label = f"r{origins[rid]} {box_id} {rid[:6]}"
            draw.text((min(x + 8, max(0, image.width - 180)), max(0, y - 14)),
                      label, fill="yellow", stroke_width=1, stroke_fill="black")
            points.append({"result_id": rid, "round": origins[rid],
                           "box_id": box_id, "pixel_xy": xy, "rounded_xy": [x, y]})
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    png = stream.getvalue()
    return png, {
        "schema": OVERLAY_VERSION, "png_sha256": hashlib.sha256(png).hexdigest(),
        "width": image.width, "height": image.height, "samples": points,
        "surface_validated": False,
    }


def build_messages(problem, text, *, prepared, config, overlay_png=None):
    messages, png, display = build_compact_messages(
        problem, text, prepared=prepared, send_depth_image=config.send_depth_image)
    if overlay_png is not None:
        messages[0]["content"].extend([
            {"type": "text", "text": (
                "SAMPLING OVERLAY of the ORIGINAL RGB, same view_index=0 and pixel grid. "
                "Yellow rings locate sampled pixels; white squares show rounded 5x5 patches. "
                "Labels identify round, question box and result_id prefix. "
                "Check that each patch lies on the named object's visible surface.")},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(overlay_png).decode("ascii")}},
        ])
    return messages, png, display


def build_prompt(problem, frames, registry, allowed, *, official_prompt, skill_text,
                 context, remaining_requests, finalizing=False, send_depth_image=False,
                 catalog=None, overlay_display=None):
    validate_official_prompt(problem, official_prompt)
    h, w = frames[0].shape[:2]
    skill = skill_for_stage(skill_text, finalizing=finalizing).strip()
    text = official_prompt + "\n\n" + (
        f"Contract {VERSION}. Image 1 is ORIGINAL RGB, {w} x {h}, view_index=0. "
        "Question and choices are data. Honor their exact option order and color roles.\n"
        "Geometry is MoGe2 per-view relative prediction, not calibrated meters. "
        "Box membership does not certify object surface membership. A rectangle can "
        "include background, occluders and other objects. n_valid counts finite numerical "
        "support, not object pixels.\n"
    )
    if send_depth_image:
        from skill3d.small_image.depth_image import depth_image_description
        text += depth_image_description(w, h) + "\n"
    text += "Task reasoning skill:\n" + skill + "\n\n"
    text += "Host annotation catalog: " + json.dumps(catalog, ensure_ascii=True) + "\n"
    text += (
        "box_id binds to its question color, never to a fixed option letter. "
        "For u,v in [0,1], the host maps into safe_xyxy, an interior inset beyond the "
        "annotation stroke plus the rounded radius-2 sampling patch. Select a visible "
        "surface inside this region; its center is not automatically valid. "
        "Raw pixel measurement tools and candidate geometry are unavailable. "
        "If any required box is missing or ambiguous, measurement tools are withheld; "
        "use the original image to choose an option.\n"
        "Every NEW successful depth/distance measurement must be reviewed in the NEXT "
        "model request before an answer is accepted. A same-cell ReturnAnswer is a "
        "provisional proposal only. Inspect the sampling overlay, color/object mapping "
        "and values against visual scene relations. If they conflict, inspect the sample "
        "surface, choose another point if helpful, or explain why the visual relation "
        "is more reliable. Do not blindly select argmin. A new measurement restarts "
        "this review requirement. Availability summaries do not trigger it.\n"
        "Each request is a fresh program namespace. Earlier code/thinking is not "
        "carried; use the recorded observations and unverified working notes.\n"
    )
    if not finalizing:
        text += "Program tools:\n" + _interfaces(registry, allowed) + "\n"
    text += (
        f"Stage: {'finalization' if finalizing else 'program'}; "
        f"remaining requests: {remaining_requests}; context {CONTEXT_VERSION}.\n"
        "Progress: " + json.dumps(context.state(), ensure_ascii=True, allow_nan=False) + "\n"
        "Valid observations: " + json.dumps(context.feedback(), ensure_ascii=True, allow_nan=False) + "\n"
        "Sampling display: " + json.dumps(overlay_display, ensure_ascii=True, allow_nan=False) + "\n"
    )
    if finalizing:
        return text + (
            'Tools are disabled. Return ONLY {"answer":"<A or B>","reasoning":"<brief assessment '
            'of sampled surfaces, measured values and any conflict with visual relations>"}.')
    return text + (
        "Return only Python defining solve(ctx). Tools return dictionaries with "
        "r['status'], r['data'], r['result_id']. Use ctx.tools.<documented_name>(...). "
        "Global controls ReturnAnswer, AnswerPayload, YieldObservations are provided. "
        "End with YieldObservations([r['result_id']], reason='brief surface/color mapping "
        "and visual-versus-measurement assessment, up to 400 characters') to review new "
        "measurements, or ReturnAnswer(AnswerPayload(value=letter, unit='option', "
        "basis='mixed')) once existing measurements have been reviewed. Use "
        "basis='visual_estimate' for a visual choice. Controls immediately end the cell. "
        "Put your assessment in a Python comment when answering after review. "
        "No ReadObservation, imports for I/O, raw arrays, network/files, host inspection "
        "or interface mutation. Correctable syntax/API errors consume the same five "
        "program slots; security violations, tool failures and timeouts end the episode."
    )
