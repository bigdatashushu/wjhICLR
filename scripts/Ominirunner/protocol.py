"""Fresh bounded messages, with original RGB and deterministic depth displays."""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json

from PIL import Image, ImageDraw

from skill3d.small_image.depth_image import VERSION as DEPTH_VERSION, render_depth_image
from .common import digest, skill_text
from .tools import MEASUREMENTS, empty_counts, phase_for
from .review import VERSION as REVIEW_VERSION, DECISIONS, pending, points
from .evidence import VERSION as EVIDENCE_VERSION
from .sampling import EDGE_NOTE

OVERLAY = "omni-rgb-depth-sampling-overlay/4"
PACKING = "omni-stage-evidence-context/5"
INTERFACES = """Tool dictionaries: result_id, source_tool, status, data.
ctx.tools.view_candidates(view_index=0, category="") -> data.candidates: IDs/category/box_xyxy/review flags; incomplete detections.
ctx.tools.view_depth_summary() -> data.views: availability/quantiles.
ctx.tools.point_depth(view_index=0, x=100, y=100) -> data.depth, pixel_xy, depth_samples, n_valid.
ctx.tools.pair_point_distance(view_index=0, a_xy=[100,100], b_xy=[200,200]) -> data.distance, a_xy, b_xy, a_depth_samples, b_depth_samples.
Metric/world/cross-view/mask measurements disabled."""
RAW_FORMAT = """Measurements use rounded/clamped original pixels and relative units. depth is raw camera Z at pixel_xy; distance is the raw 3D chord between a_xy/b_xy. Each depth_samples matrix has origin_xy and row-major values for the clipped 5x5 patch; null means invalid. No averaging. For different distance endpoints, remeasure when tools are open.
""" + EDGE_NOTE
OUTPUT = """Output one closed ```python block with only def solve(ctx):; host invokes it once. No prose/XML/top-level calls.
ctx.tools has only the four evidence tools. GLOBAL controls ReturnAnswer and YieldObservations end execution immediately; never ctx.tools.ReturnAnswer or ctx.tools.YieldObservations.
Sampling example (choose actual target pixels):
```python
def solve(ctx):
    r = ctx.tools.point_depth(view_index=0, x=100, y=100)
    samples = r["data"]["depth_samples"]
    YieldObservations([r["result_id"]], reason="Review target samples")
```
Submission example (valid after earlier measurements):
```python
def solve(ctx):
    ReturnAnswer(1.25, evidence={"kind":"visual_estimate","basis":"RGB estimate of the requested quantity"})
```
Calculation example (replace values and IDs with visible usable TOOL FACTS; do not reuse these example measurements):
```python
def solve(ctx):
    h = 3.141592653589793
    w = 1.2345678901234567
    answer = 2.0 * h / w
    evidence = {"kind":"calculation","expression":"2.0*h/w","bindings":{"h":{"result_id":"FULL_HEIGHT_ID","quantity":"height","geometry":{"coverage":"complete","axis":"object_vertical","anchors":["top","base"],"objects":["target","target"]}},"w":{"result_id":"FULL_WIDTH_ID","quantity":"width","geometry":{"coverage":"complete","axis":"object_horizontal","anchors":["left","right"],"objects":["scale_anchor","scale_anchor"]}}}}
    ReturnAnswer(answer, evidence=evidence)
```
Compute answer in Python from the original delivered values, using the same expression. Do not hand-compute or round a numeric answer literal. For point sample_xy, read that exact returned matrix cell; never average samples. Only declare full dimensions when RGB confirms complete same-object extrema.
Variables reset. Safe math/numpy allowed; host I/O/introspection forbidden."""
JSON_OUTPUT = 'Tools are closed. Output only {"answer": native_scalar}. Use a number without units for numeric answers, or the requested text; no explanation.'
REVIEW_OUTPUT = """SURFACE REVIEW: inspect RGB markers/crops and raw samples. Identifiable target values: usable (pairs require both center endpoints on intended surfaces); wrong target: resample; unclear: uncertain.
Updates contain only result_id + decision, no reason. Review visible measurement IDs; new points NEXT request. Record wrong points even if noticed in thinking. Saved decisions carry forward; resample/uncertain cannot support calculations."""
PROGRAM_REVIEW = 'Review updates: FIRST literal assignment inside solve: measurement_reviews = [{"result_id":"FULL_ID","decision":"usable"}]. Omit when unchanged.'
EVIDENCE_OUTPUT = """ANSWER EVIDENCE E: required after any measurement.
For a visual estimate E={"kind":"visual_estimate","basis":"short RGB reasoning, <=200 characters"}. For a calculation E={"kind":"calculation","expression":"2*z","bindings":{"z":{"result_id":"FULL_ID","quantity":"camera_z"}}}.
Bindings need visible TOOL FACTS with usable reviews. Binding names match [a-z][a-z0-9_]{0,15}. point quantity camera_z defaults to raw center depth; optional sample_xy=[x,y] selects an exact valid pixel from its returned matrix. point bindings forbid geometry. Pair quantity chord/height/width/length uses distance at the returned endpoints and forbids sample_xy. Z is not height/walking distance; chord is not full size and MUST omit geometry. Only height/width/length require geometry={"coverage":"complete","axis":"object_vertical","anchors":["top","base"],"objects":["chair","chair"]}. Height: object_vertical top/base; width: object_horizontal left/right; length: object_length front/back. Both endpoints must be the same object (each name <=80 chars), at full extrema; cropped/occluded/partial/uncertain extents cannot be complete.
Expression <=200 characters: numbers, binding names, parentheses, + - * /; use every binding. calculation requires a numeric answer; category/text comparisons use visual_estimate with a short basis. Host recomputes using original values and compares with rel_tol=1e-9, abs_tol=1e-9. Preserve full numeric precision. ERRORS identify the failing field and, for arithmetic mismatch, submitted/calculated values and tolerances. Relative scale needs a justified anchor in the formula; geometry and scale remain your visual declarations."""
FINAL_REVIEW = """Tools closed. Output only {"answer":value,"measurement_reviews":[{"result_id":"FULL_ID","decision":"usable"}],"evidence":E}. Use a native number without units or the requested text; no explanation. Record new review updates; use [] when unchanged. Example: {"answer":1.25,"measurement_reviews":[],"evidence":{"kind":"visual_estimate","basis":"RGB estimate"}}."""


def compact(observations, reviews=None, cache=None):
    reviews = reviews or {}
    rows = []
    for r in observations:
        tool = r["source_tool"]
        if tool == "view_candidates":
            candidates = r["data"]["candidates"]
            data = {"candidates": [{k: c[k] for k in (
                "candidate_id", "category", "box_xyxy", "category_confirmed",
                "mask_quality_assessed", "geometry_usable") if k in c}
                for c in candidates[:8]], "count": len(candidates),
                "omitted_candidates": max(0, len(candidates)-8), "exhaustive": False}
        else:
            data = {k: v for k, v in r["data"].items()
                    if k not in ("unit", "coordinate_convention", "definition", "view_index")}
        row = {"result_id": r["result_id"], "source_tool": r["source_tool"], "data": data}
        if tool in MEASUREMENTS:
            review = reviews.get(r["result_id"], {})
            row["review"] = review.get("decision", "unreviewed")
        rows.append(row)
    return rows


def packed(observations, reviews, cache, budget, review_ids=(), dependency_ids=(),
           max_measurements=8):
    """Whole typed records only; all omitted evidence remains in host receipts."""
    pending_rows = [r for r in observations if r["source_tool"] in MEASUREMENTS
                    and r["result_id"] not in reviews]
    usable = [r for r in reversed(observations)
              if reviews.get(r["result_id"], {}).get("decision") == "usable"]
    rejected = [r for r in reversed(observations)
                if r["result_id"] in reviews and r not in usable]
    latest, seen = [], set()
    for r in reversed(observations):
        if r["source_tool"] not in MEASUREMENTS and r["source_tool"] not in seen:
            latest.append(r)
            seen.add(r["source_tool"])
    all_ids = {r["result_id"]: r for r in observations if r["source_tool"] in MEASUREMENTS}
    dependencies = [all_ids[key] for key in dict.fromkeys(dependency_ids) if key in all_ids]
    by_id = {r["result_id"]: r for r in pending_rows}
    priority = [by_id[key] for key in dict.fromkeys(review_ids) if key in by_id]
    remaining = [r for r in pending_rows if r["result_id"] not in set(review_ids)]
    order = dependencies + priority + usable + remaining + rejected + latest
    chosen, rows, size, measurements, selected = [], [], 2, 0, set()
    labels = {r["result_id"]: f"M{n}" for n, r in enumerate(
        (r for r in observations if r["source_tool"] in MEASUREMENTS), 1)}
    for raw in order:
        if raw["result_id"] in selected:
            continue
        measurement = raw["source_tool"] in MEASUREMENTS
        if measurement and max_measurements is not None and measurements >= max_measurements:
            continue
        row = compact([raw], reviews, cache)[0]
        if measurement:
            row["marker"] = labels[raw["result_id"]]
        length = len(json.dumps(row, separators=(",", ":"))) + int(bool(rows))
        if budget is None or size + length <= budget:
            rows.append(row)
            chosen.append(raw)
            selected.add(raw["result_id"])
            size += length
            measurements += int(measurement)
    return chosen, rows, labels


def review_summary(reviews, labels, visible_ids=()):
    """Visible decisions are already on facts; summarize omitted blocked rows only."""
    visible = set(visible_ids)
    blocked = [r for r in reversed(list(reviews.values()))
               if r["decision"] != "usable" and r["result_id"] not in visible]
    return {"judgments": {labels[r["result_id"]]: r["decision"] for r in blocked[:32]},
            "omitted_blocked": max(0, len(blocked)-32)}


def overlay(png, observations, labels=None):
    image = Image.open(BytesIO(png)).convert("RGB")
    draw, markers = ImageDraw.Draw(image), []
    number = 0
    for row in observations:
        if row["source_tool"] not in MEASUREMENTS:
            continue
        number += 1
        samples = points(row)
        for index, (x, y) in enumerate(samples):
            ix, iy = round(x), round(y)
            label = (labels[row["result_id"]] if labels else f"M{number}") + (
                f":{'ab'[index]}" if len(samples) == 2 else "")
            draw.rectangle((ix-2, iy-2, ix+2, iy+2), outline=(0, 255, 0))
            draw.ellipse((ix-5, iy-5, ix+5, iy+5), outline=(255, 128, 0), width=1)
            draw.text((max(0, min(ix+7, image.width-90)), max(0, min(iy-10, image.height-12))),
                      label, fill=(255, 128, 0), stroke_width=1, stroke_fill=(0, 0, 0))
            markers.append({"result_id": row["result_id"], "point_index": index,
                            "pixel_xy": [x, y], "patch_xy": [ix, iy], "label": label})
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), {"version": OVERLAY, "markers": markers}


def crop_sheets(rgb, markers, review_ids, max_pages=2):
    image = Image.open(BytesIO(rgb)).convert("RGB")
    selected, seen = [], set()
    for marker in markers:
        coordinate = tuple(marker["patch_xy"])
        if marker["result_id"] in review_ids and coordinate not in seen:
            selected.append(marker)
            seen.add(coordinate)
    pages = []
    for start in range(0, min(len(selected), 8*max_pages), 8):
        group = selected[start:start+8]
        sheet = Image.new("RGB", (4*224, ((len(group)+3)//4)*224), (28, 28, 28))
        draw, entries = ImageDraw.Draw(sheet), []
        for n, marker in enumerate(group):
            ix, iy = marker["patch_xy"]
            x0 = max(0, min(ix-96, max(0, image.width-192)))
            y0 = max(0, min(iy-96, max(0, image.height-192)))
            box = [x0, y0, min(x0+192, image.width), min(y0+192, image.height)]
            tx, ty = (n % 4)*224+16, (n//4)*224+24
            sheet.paste(image.crop(box), (tx, ty))
            cx, cy = tx+ix-x0, ty+iy-y0
            draw.rectangle((cx-2, cy-2, cx+2, cy+2), outline=(0, 255, 0))
            draw.ellipse((cx-5, cy-5, cx+5, cy+5), outline=(255, 128, 0))
            draw.text((tx, ty-18), marker["label"]+f" ({ix},{iy})", fill=(255, 200, 100))
            entries.append({**marker, "source_box_xyxy": box, "sheet_pixel_xy": [cx, cy]})
        output = BytesIO()
        sheet.save(output, format="PNG")
        raw = output.getvalue()
        pages.append((raw, {"markers": entries, "png_sha256": hashlib.sha256(raw).hexdigest()}))
    return pages


def messages(item, rgb, frame, cache, *, arm, config, round_index=0,
             observations=(), note="", errors=(), review_ids=(), counters=None, reviews=None,
             dependency_ids=()):
    reviews = reviews or {}
    counters = empty_counts() if counters is None else dict(counters)
    if not 0 <= round_index < config.request_limit(arm):
        raise ValueError("round_outside_episode")
    phase = "final" if arm == "direct" else phase_for(config, round_index)
    final = phase == "final"
    cap = 10000 if arm == "direct" else config.text_caps[-1 if final else round_index]
    h, w = frame.shape[:2]
    parts = [{"type": "text", "text": item["question"]}]
    display, rgb_display, crops = None, None, []
    visible, fact_rows, selected_pending, saved_reviews = [], [], [], None
    if arm == "direct":
        text = JSON_OUTPUT.replace("Tools are closed. ", "")
        parts.append({"type": "text", "text": text})
    elif arm == "skill":
        text = (
            f"Grid {w}x{h}: x right, y down; view_index=0.\n"
            "STATIC SKILL\n" + skill_text())
        if not final:
            text += "\n" + INTERFACES
        programs_left = 0 if final else config.max_program_rounds-round_index
        text += (f"\nRequest {round_index+1}/{config.request_limit(arm)}; "
                 f"tools {'closed' if final else 'open; '+str(programs_left)+' program requests including this, then 1 final JSON'}. "
                 f'Used {counters["calls"]} calls/{counters["measurements"]} measurements; no count limits.')
        text += "\n" + RAW_FORMAT + "\n" + REVIEW_OUTPUT + "\n" + EVIDENCE_OUTPUT
        labels = {r["result_id"]: f"M{n}" for n, r in enumerate(
            (r for r in observations if r["source_tool"] in MEASUREMENTS), 1)}
        summary_reserve = len(json.dumps(review_summary(reviews, labels), separators=(",", ":"))) + 20
        dependency_reserve = (len(json.dumps([labels[k] for k in dependency_ids if k in labels],
                                             separators=(",", ":"))) + 40) if dependency_ids else 0
        if note:
            text += "\nmodel_note_unverified: " + note[:200]
        if errors:
            # Full diagnostics stay in execution receipts. Never cut a JSON record.
            visible_errors = []
            for error in list(errors)[:4]:
                candidate = visible_errors + [error]
                if len(json.dumps(candidate, separators=(",", ":"))) <= 1000:
                    visible_errors = candidate
            text += "\nERRORS " + json.dumps(visible_errors, separators=(",", ":"))
            if len(errors) > len(visible_errors):
                text += f"; omitted={len(errors)-len(visible_errors)}"
        text += "\n" + (FINAL_REVIEW if final else PROGRAM_REVIEW + "\n" + OUTPUT)
        # Reserve bounded omitted-state text and image captions before whole-record packing.
        fact_budget = (max(0, cap-len(item["question"])-len(text)-650-summary_reserve-dependency_reserve)
                       if cap is not None else None)
        visible, fact_rows, labels = packed(observations, reviews, cache, fact_budget,
            review_ids, dependency_ids, max_measurements=8 if final else None)
        saved_reviews = review_summary(reviews, labels, (r["result_id"] for r in visible))
        if saved_reviews["judgments"] or saved_reviews["omitted_blocked"]:
            text += "\nOMITTED REVIEWS " + json.dumps(saved_reviews, separators=(",", ":"))
        selected_pending = pending(visible, reviews)
        text += "\nTOOL FACTS " + json.dumps(fact_rows, separators=(",", ":"))
        missing_dependencies = [key for key in dependency_ids
                                if key not in {r["result_id"] for r in visible}]
        text += (f"\nArchive {len(observations)}; omitted {len(observations)-len(visible)}. "
                 "Review/calculate only from visible TOOL FACTS.")
        if missing_dependencies:
            text += "\nOmitted calculation dependencies: " + json.dumps(
                [labels[k] for k in missing_dependencies if k in labels], separators=(",", ":"))
        depth_png, display = render_depth_image(
            cache.geometries[0]["depth"], cache.geometries[0]["valid_mask"])
        mask = cache.geometries[0]["valid_mask"]
        display.update(version=DEPTH_VERSION, valid_mask_sha256=digest({
            "dtype": mask.dtype.str, "shape": list(mask.shape),
            "bytes_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest()}))
        if visible:
            depth_png, overlay_info = overlay(depth_png, visible, labels)
            display = {**display, "overlay": overlay_info,
                       "parent_png_sha256": display["png_sha256"],
                       "png_sha256": hashlib.sha256(depth_png).hexdigest()}
            if overlay_info["markers"]:
                rgb_overlay, rgb_display = overlay(rgb, visible, labels)
                rgb_display["png_sha256"] = hashlib.sha256(rgb_overlay).hexdigest()
                crops = crop_sheets(rgb, rgb_display["markers"], selected_pending)
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
                "2%-98% grayscale, magenta invalid. Green 5x5 patches/orange M markers match RGB.")},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(depth_png).decode()}}])
        if rgb_display:
            parts.extend([
                {"type": "text", "text": "Image 3: RGB sampling markers."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                  base64.b64encode(rgb_overlay).decode()}}])
        for n, (raw, info) in enumerate(crops):
            parts.extend([
                {"type": "text", "text": f"RGB crops {n+1}: original pixel scale; global x,y labels."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                  base64.b64encode(raw).decode()}}])
    result = [{"role": "user", "content": parts}]
    text_chars = sum(len(p["text"]) for p in parts if p["type"] == "text")
    if cap is not None and text_chars > cap:
        raise ValueError("context_text_limit")
    images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    receipt = {"messages_sha256": digest(result), "text_chars": text_chars, "text_cap": cap,
               "text_record_threshold": config.text_record_threshold,
               "text_over_threshold": text_chars > config.text_record_threshold,
               "text_over_threshold_chars": max(0, text_chars-config.text_record_threshold),
               "phase": phase, "round_index": round_index,
               "requests_remaining": 1 if final else config.request_limit(arm)-round_index,
               "tool_counts": counters,
               "tool_limits": {"tool_calls": None, "measurements": None},
               "tools_enabled": arm == "skill" and not final,
               "image_sha256": [hashlib.sha256(base64.b64decode(u.split(",", 1)[1])).hexdigest()
                                for u in images], "image_count": len(images),
               "reviewed_measurement_ids": selected_pending, "depth_display": display,
               "pending_measurement_ids": pending(observations, reviews),
               "visible_result_ids": [r["result_id"] for r in visible],
               "visible_measurement_ids": [r["result_id"] for r in visible if r["source_tool"] in MEASUREMENTS],
               "context_packing": {"version": PACKING, "archive_count": len(observations),
                                   "visible_count": len(visible),
                                   "omitted_count": len(observations)-len(visible),
                                   "archive_sha256": digest(list(observations)),
                                   "visible_facts_sha256": digest(fact_rows),
                                   "calculation_dependency_ids": list(dependency_ids),
                                   "omitted_dependency_ids": missing_dependencies if arm == "skill" else [],
                                   "max_visible_measurements": 8 if final or arm == "direct" else None,
                                   "max_crop_sheets": 2},
               "surface_review_version": REVIEW_VERSION if arm == "skill" else None,
               "answer_evidence_version": EVIDENCE_VERSION if arm == "skill" else None,
               "surface_reviews_sha256": digest(reviews) if arm == "skill" else None,
               "surface_review_state": saved_reviews,
               "surface_review_counts": {d: sum(r["decision"] == d for r in reviews.values())
                                         for d in DECISIONS} if arm == "skill" else None,
               "rgb_sampling_display": rgb_display,
               "rgb_crop_sheets": [info for _, info in crops],
               "skill_sha256": hashlib.sha256(skill_text().encode()).hexdigest()
               if arm == "skill" else None}
    if len(images) != (1 if arm == "direct" else 2+int(bool(rgb_display))+len(crops)):
        raise ValueError("image_count")
    return result, receipt
