"""Fresh bounded messages, with original RGB and deterministic depth displays."""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json

from PIL import Image, ImageDraw

from skill3d.small_image.depth_image import VERSION as DEPTH_VERSION, render_depth_image
from .common import digest, skill_text
from .tools import MEASUREMENTS, empty_counts, phase_for, remaining

OVERLAY = "omni-depth-sampling-overlay/1"
INTERFACES = """Tool dictionaries: result_id, source_tool, status, data.
ctx.tools.view_candidates(view_index=0, category="") -> data.candidates: IDs/category/box_xyxy/review flags; incomplete detections.
ctx.tools.view_depth_summary() -> data.views: availability/quantiles.
ctx.tools.point_depth(view_index=0, x=100, y=100) -> data.depth, pixel_xy, n_valid.
ctx.tools.pair_point_distance(view_index=0, a_xy=[100,100], b_xy=[200,200]) -> data.distance.
5x5 local patches, relative units. Metric/world/cross-view/mask measurements disabled."""
OUTPUT = """Output one closed ```python block with only def solve(ctx):; host invokes it once. No prose/XML/top-level calls.
ctx.tools has only the four evidence tools. GLOBAL controls ReturnAnswer and YieldObservations end execution immediately; never ctx.tools.ReturnAnswer or ctx.tools.YieldObservations.
Sampling example (choose actual target pixels):
```python
def solve(ctx):
    r = ctx.tools.point_depth(view_index=0, x=100, y=100)
    z = r["data"]["depth"]
    YieldObservations([r["result_id"]], reason="Check target surface and depth " + str(z))
```
Submission after reviewing evidence:
```python
def solve(ctx):
    ReturnAnswer(1.25)
```
Review new measurements in the NEXT request; same-program answers are provisional. Variables reset; TOOL FACTS persist. Safe math/numpy allowed; host I/O/introspection forbidden."""
JSON_OUTPUT = 'Tools are closed. Output only {"answer": native_scalar}. Use a number without units for numeric answers, or the requested text; no explanation.'


def compact(observations):
    rows = []
    for r in observations:
        data = {k: v for k, v in r["data"].items()
                if k not in ("unit", "coordinate_convention", "definition")}
        rows.append({"result_id": r["result_id"], "source_tool": r["source_tool"], "data": data})
    return rows


def overlay(png, observations):
    image = Image.open(BytesIO(png)).convert("RGB")
    draw, markers = ImageDraw.Draw(image), []
    for row in observations:
        if row["source_tool"] not in MEASUREMENTS:
            continue
        data = row["data"]
        points = ([data["pixel_xy"]] if row["source_tool"] == "point_depth"
                  else [data["a_xy"], data["b_xy"]])
        for index, (x, y) in enumerate(points):
            ix, iy = round(x), round(y)
            label = row["result_id"][:6] + (f":{index}" if len(points) == 2 else "")
            draw.rectangle((ix-2, iy-2, ix+2, iy+2), outline=(0, 255, 0))
            draw.ellipse((ix-5, iy-5, ix+5, iy+5), outline=(255, 128, 0), width=1)
            draw.text((max(0, min(ix+7, image.width-90)), max(0, min(iy-10, image.height-12))),
                      label, fill=(255, 128, 0), stroke_width=1, stroke_fill=(0, 0, 0))
            markers.append({"result_id": row["result_id"], "point_index": index,
                            "pixel_xy": [x, y], "patch_xy": [ix, iy], "label": label})
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), {"version": OVERLAY, "markers": markers}


def messages(item, rgb, frame, cache, *, arm, config, round_index=0,
             observations=(), note="", errors=(), review_ids=(), counters=None):
    counters = empty_counts() if counters is None else dict(counters)
    if not 0 <= round_index < config.request_limit(arm):
        raise ValueError("round_outside_episode")
    phase = "final" if arm == "direct" else phase_for(config, round_index, counters)
    final = phase == "final"
    quotas = remaining(config, counters)
    h, w = frame.shape[:2]
    parts = [{"type": "text", "text": item["question"]}]
    display = None
    if arm == "direct":
        text = JSON_OUTPUT.replace("Tools are closed. ", "")
        parts.append({"type": "text", "text": text})
    elif arm == "skill":
        depth_png, display = render_depth_image(
            cache.geometries[0]["depth"], cache.geometries[0]["valid_mask"])
        mask = cache.geometries[0]["valid_mask"]
        display.update(version=DEPTH_VERSION, valid_mask_sha256=digest({
            "dtype": mask.dtype.str, "shape": list(mask.shape),
            "bytes_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest()}))
        if observations:
            depth_png, overlay_info = overlay(depth_png, observations)
            display = {**display, "overlay": overlay_info,
                       "parent_png_sha256": display["png_sha256"],
                       "png_sha256": hashlib.sha256(depth_png).hexdigest()}
        text = (
            f"Original grid {w}x{h}: x right, y down; view_index=0. "
            "Depth/points have relative scale, not meters; camera Z differs from point distance.\n"
            "STATIC SKILL\n" + skill_text())
        if not final:
            text += "\n" + INTERFACES
        requests_left = 1 if final else config.request_limit(arm)-round_index
        programs_left = 0 if final else config.max_program_rounds-round_index
        text += (f"\nPhase: {phase}; request {round_index+1}/{config.request_limit(arm)}; "
                 f"requests remaining INCLUDING THIS REQUEST: {requests_left}; "
                 f"program rounds remaining INCLUDING THIS REQUEST: {programs_left}; "
                 "1 JSON final reserved.")
        text += (f'\nEPISODE QUOTA (shared across rounds, never reset): tools '
                 f'{counters["calls"]}/{config.max_tool_calls} used, '
                 f'{quotas["tool_calls"]} remaining; measurements '
                 f'{counters["measurements"]}/{config.max_measurements} used, '
                 f'{quotas["measurements"]} remaining. '
                 "Each measurement costs 1 tool call; admitted failures count. "
                 "Excess calls are blocked; host ends the program and delivers successful "
                 "facts/IDs/markers next request.")
        if observations:
            text += "\nTOOL FACTS " + json.dumps(compact(observations), separators=(",", ":"))
        else:
            text += "\nNo tool observations yet."
        if review_ids:
            text += "\nReview required now: " + json.dumps(list(review_ids)) + (
                "; inspect object/surface binding, markers, values and RGB agreement before answering.")
        if note:
            text += "\nmodel_note_unverified: " + note[:200]
        if errors:
            text += "\nRecent error/unavailable states: " + json.dumps(list(errors), separators=(",", ":"))
        text += "\n" + (JSON_OUTPUT if final else OUTPUT)
        parts.append({"type": "text", "text": text})
    else:
        raise ValueError("unknown_arm")
    parts.extend([
        {"type": "text", "text": "Image 1: official original RGB."},
        {"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(rgb).decode()}}])
    if arm == "skill":
        parts.extend([
            {"type": "text", "text": (
                "Image 2: aligned predicted depth, same camera/grid. Dark near, light far; "
                "2%-98% grayscale, magenta invalid. Tools use raw arrays. "
                "Green 5x5 patches/orange markers, when present, identify sampled positions "
                "by the first six characters of result_id.")},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(depth_png).decode()}}])
    result = [{"role": "user", "content": parts}]
    text_chars = sum(len(p["text"]) for p in parts if p["type"] == "text")
    cap = 4000 if arm == "direct" else config.text_caps[-1 if final else round_index]
    if text_chars > cap:
        raise ValueError("context_text_limit")
    images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    receipt = {"messages_sha256": digest(result), "text_chars": text_chars, "text_cap": cap,
               "phase": phase, "round_index": round_index,
               "requests_remaining": 1 if final else config.request_limit(arm)-round_index,
               "tool_counts": counters, "remaining_quota": quotas,
               "tools_enabled": arm == "skill" and not final,
               "image_sha256": [hashlib.sha256(base64.b64decode(u.split(",", 1)[1])).hexdigest()
                                for u in images], "image_count": len(images),
               "reviewed_measurement_ids": list(review_ids), "depth_display": display,
               "skill_sha256": hashlib.sha256(skill_text().encode()).hexdigest()
               if arm == "skill" else None}
    if len(images) != (1 if arm == "direct" else 2):
        raise ValueError("image_count")
    return result, receipt
