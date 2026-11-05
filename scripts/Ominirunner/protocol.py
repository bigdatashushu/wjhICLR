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
from .evidence import VERSION as EVIDENCE_VERSION, expression_value
from .sampling import EDGE_NOTE
from .answer_contract import specification
from .coordinates import original_descriptor, VERSION as COORDINATE_VERSION
from .progress import unresolved, summary as progress_summary
from .recovery import VERSION as RECOVERY_VERSION
from .provenance import VERSION as ORIGIN_VERSION
from .native import PENDING_VERSION, context_feedback
from .submission import (VERSION as SUBMISSION_VERSION, MAX_CHARS as SUBMISSION_CAP,
                         NOTICE as SUBMISSION_NOTICE, delivery as submission_delivery)

OVERLAY = "omni-rgb-depth-sampling-overlay/5"
PACKING = "omni-stage-evidence-context/9"
INTERFACES = """Tool dictionaries: result_id, source_tool, status, data.
ctx.tools.view_candidates(view_index=0, category="") -> data.candidates: IDs/category/box_xyxy/review flags; incomplete detections.
ctx.tools.view_depth_summary() -> data.views: availability/quantiles.
ctx.tools.point_depth(view_index=0, x=100, y=100) -> data.depth, pixel_xy, depth_samples, n_valid.
ctx.tools.pair_point_distance(view_index=0, a_xy=[100,100], b_xy=[200,200]) -> data.distance, a_xy, b_xy, a_depth_samples, b_depth_samples.
Metric/world/cross-view/mask measurements disabled."""
INTERFACES += """
Sampling also accepts image_id="rgb-original", coordinate_space="normalized" (x,y in [0,1]), or a delivered crop ID with image_pixel/normalized coordinates. Host transforms coordinates to the original grid. Default original_pixel remains supported.
Add targets=[{"target":"chair","surface":"top"}] for point, two target/surface entries in a/b order for pair. To replace a visible wrong measurement add replaces="OLD_RESULT_ID"; the old review stays excluded.
For full dimensions, keep surface as the actual visible part and optionally add extent_role and extent_basis. Review with observed_surface, extent_role, extent_basis, scope_completeness="complete" and issues=[] only after RGB confirms the actual hit, axis and full extremum.
targets is input; intended_targets is output. Use 0<=x<width, 0<=y<height on the named grid. Correct located input errors next request.
view_candidates preserves cache IDs, review receipt identity and unknown NMS state. Overlap is an unresolved relation, not permission to merge objects."""
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
Compute answer in Python from the original delivered values, using the same expression. Do not hand-compute or round a numeric answer literal. For point sample_xy, read that exact returned matrix cell; never average samples. Only declare full dimensions when RGB confirms complete same-object extrema.
Variables reset. Safe math/numpy allowed; host I/O/introspection forbidden."""
JSON_OUTPUT = 'Tools are closed. Output only {"answer": native_scalar}. Use a number without units for numeric answers, or the requested text; no explanation.'
REVIEW_OUTPUT = """SURFACE REVIEW: inspect RGB marker/crop hits and raw values NEXT request. Unused records do not block independent RGB answers."""
PROGRAM_REVIEW = 'Review updates: measurement_reviews must be the FIRST literal assignment inside solve; use the review schema above. Omit when unchanged.'
EVIDENCE_OUTPUT = """ANSWER EVIDENCE E: required after any measurement.
RGB E={"kind":"visual_estimate","basis":"RGB reasoning <=200 chars"}. Calculation E={"kind":"calculation","expression":"2*z","bindings":{"z":{"result_id":"FULL_ID","quantity":"camera_z"}}}.
Bind visible usable TOOL FACTS; names match [a-z][a-z0-9_]{0,15}. Point camera_z uses raw center depth, or exact valid matrix pixel sample_xy=[x,y]; geometry forbidden. Pair chord/height/width/length/diameter uses returned distance; sample_xy forbidden. Z is not height/walking distance; chord is not full size and MUST omit geometry.
Dimensions need geometry={"coverage":"complete","axis":"object_vertical","anchors":["top","base"],"objects":["chair","chair"]}. Height: object_vertical top/base; width: object_horizontal left/right; length: object_length front/back; diameter: object_diameter rim_a/rim_b across the full diameter. Anchors match reviewed extent_role in a/b order, while observed_surface records the physical hit. Both extrema belong to the same object (names <=80 chars). Cropped/occluded/partial/uncertain extents cannot be complete.
Expression <=200 chars: numbers, bindings, parentheses, + - * /; use every binding. calculation needs a numeric answer; labels use tool_comparison. Host recomputes original values with rel_tol=1e-9, abs_tol=1e-9. Preserve precision. ERRORS give field and arithmetic submitted/calculated values. Geometry and scale remain your visual declarations."""
EVIDENCE_OUTPUT += """
Comparison E={"kind":"tool_comparison","bindings":{"a":{"result_id":"ID_A","quantity":"camera_z","reference":"camera"},"b":{"result_id":"ID_B","quantity":"camera_z","reference":"camera"}},"left":"a","right":"b","operator":"<","answers":{"true":"O1","false":"O2"}}. Operators <,<=,>,>=,==,!=; three_way uses less/equal/greater. Same units/reference; use all bindings.
Task units follow dimension algebra. Anchor relative dimensions with scale={"name":"s","value":PUBLIC_SPAN,"unit":TASK_UNIT,"anchor_binding":"w"}, e.g. s*h/w. Units m/ft/in/yd/mi. Convert to task unit using answer_unit_value (1 mi=5280 ft). w is the exact public target's complete stated dimension. Use that converted scale in Python. Relative Z cannot establish physical/walking/collision geometry.
Quantitative review example: measurement_reviews=[{"result_id":"FULL_ID","decision":"usable","observed_surfaces":[{"target":"chair","surface":"top","decision":"usable","visibility":"visible"}]}]. For pairs include both surfaces in a/b order and match intended targets; never put observed_surfaces inside E.
Counts require evidence.object_scope={include,exclude,entities}, even without tools; enumerate requested objects/subfeatures (holes also count as entities). Empty candidate_ids uses relationship=rgb_only; one ID uses single; multiple IDs need same_object/part_whole with RGB basis. Overlap/category review cannot merge; unconfirmed detections cannot prove absence.
Volume/volume_ratio require evidence.assumptions={"object_scope":"objects included/excluded","volume_model":"shape model"}: exactly two nonempty strings <=200 chars each. This object_scope string is different from the count entity dictionary.
visual_estimate basis must be nonempty <=200 characters; aim <=120. In Python check len(evidence["basis"]) before submitting; shorten explicitly, never silently truncate."""
REVIEW_OUTPUT += """
Record actual target/observed_surface in point/a/b order. decision=usable/resample/uncertain; visibility=visible/cropped/occluded/unclear. wrong/wrong target -> resample; face_left/left_edge -> left, face_right/right_edge -> right.
extent_role/extent_basis and scope_completeness describe the full extremum separately; geometry anchors follow those roles in a/b order. mantel_top is not automatically object top. Record unresolved hit/axis/extent doubts in issues. Wrong/hidden/partial/unclear hits or doubts prevent usable. New IDs/names do not resolve them. Usable remains your RGB judgment."""
FINAL_REVIEW = """Tools closed. Output only {"answer":value,"measurement_reviews":[],"evidence":E}; native scalar, no units/explanation. [] keeps existing full reviews. New usable reviews need all observed_surfaces in point/a/b order. Usable-only restatements preserve existing usable surfaces; prior resample/uncertain needs a full review. Preserve Python-computed answer/expression precision; never redo arithmetic mentally."""
ORIGIN_OUTPUT = """
ORIGIN: declare E.provenance={"source":"rgb_only","referenced_result_ids":[],"abandoned_result_ids":[],"abandonment_reason":""} for independent RGB. After a rejected calculation list every abandoned dependency and why its numeric values are not used. Explicit tool-derived answers require source=tool_informed, referenced_result_ids and checked bindings; use calculation/tool_comparison, or tool_estimate with basis and the same expression/binding checks. Changing kind cannot validate blocked/replaced values or wrong quantities. Origin is a declaration; matching old numbers alone proves no dependency."""


def evidence_output(task):
    """Keep the applicable schemas and omit unrelated task examples."""
    quantity = task["quantity"]
    return "\n".join(line for line in EVIDENCE_OUTPUT.splitlines()
                     if not (line.startswith("Comparison E=") and not task["allowed_answers"])
                     and not (line.startswith("Counts require") and
                              not (quantity == "count_ratio" or quantity == "count" and task["unit"] == "count"))
                     and not (line.startswith("Volume/volume_ratio") and quantity not in ("volume", "volume_ratio"))
                     and not (line.startswith("Task units follow") and not task["public_metric_values"]))


def task_output(task, final=False):
    """Complete task-specific submissions, with explicit illustrative values."""
    quantity, options = task["quantity"], task["allowed_answers"]
    answer = options[0]["option_id"] if options else 1
    evidence = {"kind": "visual_estimate", "basis": "RGB estimate of the requested quantity"}
    guidance = "Example values/entities/IDs are illustrative; replace with your actual RGB judgment."
    if quantity == "count" and task["unit"] == "count":
        evidence["basis"] = "RGB count of distinct requested entities"
        evidence["object_scope"] = {"include": "requested objects or subfeatures",
            "exclude": "reflections and unrequested parts", "entities": [{
                "entity_id": "E1", "category": "requested entity", "candidate_ids": [],
                "relationship": "rgb_only", "basis": "RGB shows one distinct requested entity",
                "included": True}]}
        guidance += (" Declare every counted entity, including holes when requested; "
                     "answer equals included entities. Zero count uses entities=[]. "
                     "New RGB evidence may revise scope/entities/count; do not lock an old list.")
    elif quantity == "count_ratio":
        targets = [o["target"] for o in task["operands"] if o["role"] in ("numerator", "denominator")]
        evidence["object_scope"] = {"include": "the requested numerator and denominator entities",
            "exclude": "unrequested objects and reflections", "entities": [{
                "entity_id": key, "category": target, "candidate_ids": [], "relationship": "rgb_only",
                "basis": "RGB shows one distinct requested entity", "included": True}
                for key, target in zip(("E1", "E2"), targets)]}
        evidence["count_groups"] = {"numerator": ["E1"], "denominator": ["E2"]}
        guidance += " Enumerate entities and divide numerator count by denominator count; do not compare the ratio with the total count."
    elif quantity in ("volume", "volume_ratio"):
        evidence["basis"] = "RGB comparison under the stated object scope and shape model"
        evidence["assumptions"] = {"object_scope": "one instance of each requested object; exclude unrelated furniture",
                                   "volume_model": "bounding cuboids for the requested objects"}
    elif options:
        guidance += " Choose an actual public option ID; host emits its exact label."
    else:
        guidance += " No public options are listed; return the requested number or text, not an invented O1/O2."
    text = "TASK SUBMISSION TEMPLATE: " + guidance
    if final:
        text += "\n" + json.dumps(
            {"answer": answer, "measurement_reviews": [], "evidence": evidence},
            separators=(",", ":"))
    else:
        text += "\n```python\ndef solve(ctx):\n    evidence = " + repr(evidence)
        text += '\n    assert 0 < len(evidence["basis"]) <= 200'
        text += f"\n    ReturnAnswer({answer!r}, evidence=evidence)\n```"
    dimensions = {"height": ("object_vertical", ["top", "base"]),
                  "width": ("object_horizontal", ["left", "right"]),
                  "length": ("object_length", ["front", "back"]),
                  "diameter": ("object_diameter", ["rim_a", "rim_b"])}
    requested = quantity.removesuffix("_ratio")
    spans = [p for p in task["public_metric_values"] if p["quantity"] in dimensions]
    metric = quantity in dimensions and task["unit"] in ("m", "ft", "in", "yd", "mi") and spans
    ratio = task.get("operation") == "ratio" and quantity != "count_ratio"
    dimensional_comparison = quantity == "dimension_comparison"
    relation = task.get("distance_relation")
    distance_comparison = relation and len(relation["candidates"]) == 2
    if not (ratio or metric or dimensional_comparison or distance_comparison):
        return text
    operands = [(i, o) for i, o in enumerate(task.get("operands", []))
                if (o["quantity"] in dimensions or distance_comparison and o["role"] not in ("reference", "origin"))
                and (o["role"] != "scale_anchor" or metric and i == spans[0].get("operand_index", i))]
    if not operands:
        return text
    bindings, values, role_names = {}, {}, {}
    for index, (operand_index, operand) in enumerate(operands):
        name = ("t", "a")[index] if index < 2 else f"d{index}"
        q, target = operand["quantity"], operand["target"] or "exact public target"
        q = "chord" if q == "object_distance" else q
        key = "TARGET_DIMENSION_ID" if index == 0 else "ANCHOR_DIMENSION_ID" if index == 1 else f"DIMENSION_ID_{index}"
        bindings[name] = {"result_id": key, "quantity": q, "operand_index": operand_index}
        if q in dimensions:
            axis, anchors = dimensions[q]
            bindings[name]["geometry"] = {
                "coverage": "complete", "axis": axis, "anchors": anchors, "objects": [target, target]}
        elif relation:
            bindings[name]["reference"] = relation.get("reference", relation.get("origin"))
        values[name] = 3.141592653589793 if index == 0 else 1.2345678901234567
        role_names.setdefault(operand["role"], []).append(name)
    calculation = {"kind": "calculation", "bindings": bindings}
    if ratio:
        numerator, denominator = ("+".join(role_names[role]) for role in ("numerator", "denominator"))
        calculation["expression"] = numerator + "/" + denominator if len(role_names["denominator"]) == 1 else f"({numerator})/({denominator})"
    elif dimensional_comparison or distance_comparison:
        operator = task["comparison"]["operator"]
        operator = "<" if operator == "argmin" else ">" if operator == "argmax" else operator
        calculation.update(kind="tool_comparison", left="t", right=f"{task['comparison'].get('factor',1)}*a",
                           operator=operator, answers={"true": "O1", "false": "O2"})
    else:
        calculation["expression"] = "s*t/a"
    if metric:
        span = spans[0]
        value = span.get("answer_unit_value", span["value"])
        calculation["scale"] = {"name": "s", "value": value, "unit": task["unit"], "anchor_binding": "a"}
        values["s"] = value
        if task.get("operation") == "difference":
            # The subtrahend and scale anchor are the same public object.
            bindings["a"].pop("operand_index", None)
            for name in list(bindings)[2:]:
                del bindings[name]
                del values[name]
            calculation["expression"] = "s*(t/a-1)"
    text += ("\nCALCULATION TEMPLATE after delivered usable measurements: replace IDs, original values, "
             "and formula with actual evidence. Each dimension matches its public operand; targets and "
             "scale anchors may use different axes. Complete RGB extrema/reviews are required. Values are illustrative.")
    if dimensional_comparison or distance_comparison:
        left = expression_value(calculation["left"], values, require_all=False)
        right = expression_value(calculation["right"], values, require_all=False)
        result = "O1" if (left > right if calculation["operator"] == ">" else left < right) else "O2"
        answer_code = f'"O1" if {calculation["left"]} {calculation["operator"]} {calculation["right"]} else "O2"'
    else:
        result = expression_value(calculation["expression"], values)
        answer_code = calculation["expression"]
    if final:
        text += "\n" + json.dumps({"answer": result, "measurement_reviews": [], "evidence": calculation},
                                 separators=(",", ":"))
    else:
        text += "\n```python\ndef solve(ctx):\n" + "".join(f"    {k} = {v!r}\n" for k, v in values.items())
        text += f"    answer = {answer_code}\n    evidence = {calculation!r}"
        text += "\n    ReturnAnswer(answer, evidence=evidence)\n```"
    return text


def compact(observations, reviews=None, cache=None):
    reviews = reviews or {}
    rows = []
    for r in observations:
        tool = r["source_tool"]
        if tool == "view_candidates":
            candidates = r["data"]["candidates"]
            data = {"candidates": [{k: c[k] for k in (
                "candidate_id", "category", "box_xyxy", "category_reviewed",
                "category_confirmed", "review_status", "lineage",
                "mask_quality_assessed", "geometry_usable") if k in c}
                for c in candidates[:8]], "count": len(candidates),
                "omitted_candidates": max(0, len(candidates)-8), "exhaustive": False}
            shown = {c["candidate_id"] for c in candidates[:8]}
            data["overlap_groups"] = [g for g in r["data"].get("overlap_groups", [])
                                     if set(g["candidate_ids"]) <= shown]
            data["instance_rule"] = "Overlap/category review alone cannot merge RGB entities."
        else:
            data = {k: v for k, v in r["data"].items()
                    if k not in ("coordinate_convention", "definition")}
        row = {"result_id": r["result_id"], "source_tool": r["source_tool"], "data": data}
        if tool in MEASUREMENTS:
            review = reviews.get(r["result_id"], {})
            row["review"] = review.get("decision", "unreviewed")
            if "observed_surfaces" in review:
                row["observed_surfaces"] = [
                    {k: s[k] for k in ("target", "surface", "observed_surface", "extent_role",
                                      "extent_basis", "scope_completeness", "issues",
                                      "decision", "visibility", "sample_xy") if k in s}
                    for s in review["observed_surfaces"]]
        rows.append(row)
    return rows


def packed(observations, reviews, cache, budget, review_ids=(), dependency_ids=(),
           max_measurements=8):
    """Whole typed records only; all omitted evidence remains in host receipts."""
    pending_keys = set(pending(observations, reviews))
    pending_rows = [r for r in observations if r["result_id"] in pending_keys]
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
    by_id = {}
    for marker in markers:
        by_id.setdefault(marker["result_id"], []).append(marker)
    for key in dict.fromkeys(review_ids):
        for marker in by_id.get(key, []):
            coordinate = tuple(marker["patch_xy"])
            if coordinate not in seen:
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
            entries.append({**marker, "source_box_xyxy": box, "sheet_pixel_xy": [cx, cy],
                            "image_id": "crop-"+marker["label"],
                            "width": box[2]-box[0], "height": box[3]-box[1],
                            "view_index": 0, "content_box_xyxy": [0, 0, box[2]-box[0], box[3]-box[1]],
                            "offset_xy": [tx, ty], "coordinate_version": COORDINATE_VERSION})
        output = BytesIO()
        sheet.save(output, format="PNG")
        raw = output.getvalue()
        image_hash = hashlib.sha256(raw).hexdigest()
        for entry in entries:
            entry["image_sha256"] = image_hash
        pages.append((raw, {"markers": entries, "png_sha256": image_hash}))
    return pages


def messages(item, rgb, frame, cache, *, arm, config, round_index=0,
             observations=(), note="", errors=(), review_ids=(), counters=None, reviews=None,
             dependency_ids=(), progress=(), last_submission=None, pending_program=None,
             measurement_reviewer=None):
    if measurement_reviewer is not None:
        if arm != "skill":
            raise ValueError("jev_requires_skill_arm")
        from .jev_protocol import messages as compact_messages
        return compact_messages(item, rgb, frame, cache, config=config,
                                policy=measurement_reviewer, round_index=round_index,
                                observations=observations, errors=errors, counters=counters,
                                reviews=reviews, dependency_ids=dependency_ids,
                                last_submission=last_submission, pending_program=pending_program)
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
    crop_priority = []
    delivered_submission = None
    delivered_pending = None
    if arm == "direct":
        text = JSON_OUTPUT.replace("Tools are closed. ", "")
        parts.append({"type": "text", "text": text})
    elif arm == "skill":
        text = (
            f"Grid {w}x{h}: x right, y down; view_index=0.\n"
            "STATIC SKILL\n" + skill_text())
        public_task = specification(item)
        # Host verification metadata is not a fact about furniture in the image.
        public_task.pop("geometry_verification", None)
        text += "\nPUBLIC ANSWER/TASK CONTRACT " + json.dumps(public_task, separators=(",", ":"))
        text += ("\nSelect the public option ID in answer; host emits its exact label."
                 if public_task["allowed_answers"] else
                 "\nNo public options: return the requested numeric value or text.")
        if not final:
            text += "\n" + INTERFACES
        programs_left = 0 if final else config.max_program_rounds-round_index
        text += (f"\nRequest {round_index+1}/{config.request_limit(arm)}; "
                 f"tools {'closed' if final else 'open; '+str(programs_left)+' program requests including this, then 1 final JSON'}. "
                 f'Used {counters["calls"]} calls/{counters["measurements"]} measurements; no count limits.')
        text += "\n" + RAW_FORMAT + "\n" + REVIEW_OUTPUT + "\n" + evidence_output(public_task) + ORIGIN_OUTPUT
        labels = {r["result_id"]: f"M{n}" for n, r in enumerate(
            (r for r in observations if r["source_tool"] in MEASUREMENTS), 1)}
        summary_reserve = len(json.dumps(review_summary(reviews, labels), separators=(",", ":"))) + 20
        dependency_reserve = (len(json.dumps([labels[k] for k in dependency_ids if k in labels],
                                             separators=(",", ":"))) + 40) if dependency_ids else 0
        if errors:
            # Full diagnostics stay in execution receipts. Never cut a JSON record.
            visible_errors = []
            for error in list(errors)[:4]:
                error = context_feedback(error, phase)
                candidate = visible_errors + [error]
                if len(json.dumps(candidate, separators=(",", ":"))) <= 1000:
                    visible_errors = candidate
            text += "\nERRORS " + json.dumps(visible_errors, separators=(",", ":"))
            if len(errors) > len(visible_errors):
                text += f"; omitted={len(errors)-len(visible_errors)}"
        text += "\n" + (FINAL_REVIEW if final else PROGRAM_REVIEW + "\n" + OUTPUT)
        text += "\n" + task_output(public_task, final)
        # Reserve bounded omitted-state text and image captions before whole-record packing.
        action_reserve = len(json.dumps(unresolved(observations, reviews)[:8], separators=(",", ":"))) + 200
        latest_progress = progress_summary(progress[-1]) if progress else None
        progress_reserve = len(json.dumps(latest_progress, separators=(",", ":"))) + 150
        fact_budget = (max(0, cap-len(item["question"])-len(text)-1000-summary_reserve-dependency_reserve
                           -action_reserve-progress_reserve-250*bool(note))
                       if cap is not None else None)
        error_ids = list(dict.fromkeys(error.get("measurement_result_id", error.get("result_id"))
                                      for error in errors if isinstance(error, dict)))
        pinned_ids = list(dict.fromkeys([*dependency_ids, *[k for k in error_ids if isinstance(k, str)]]))
        visible, fact_rows, labels = packed(observations, reviews, cache, fact_budget,
            review_ids, pinned_ids, max_measurements=8 if final else None)
        saved_reviews = review_summary(reviews, labels, (r["result_id"] for r in visible))
        if saved_reviews["judgments"] or saved_reviews["omitted_blocked"]:
            text += "\nOMITTED REVIEWS " + json.dumps(saved_reviews, separators=(",", ":"))
        selected_pending = pending(visible, reviews)
        text += "\nTOOL FACTS " + json.dumps(fact_rows, separators=(",", ":"))
        actionable = unresolved(visible, reviews)[:8]
        if actionable:
            text += "\nUNRESOLVED TARGETS " + json.dumps(actionable, separators=(",", ":"))
            text += ("\nReview visible samples; finish from usable evidence or RGB and abandon blocked values."
                     if final else
                     "\nReview unreviewed samples; replace blocked points or explicitly abandon their basis.")
        if progress:
            text += "\nOBSERVED PROGRESS " + json.dumps(latest_progress, separators=(",", ":"))
            if not progress[-1]["made_progress"]:
                text += ("\nNo new evidence/status change. Finish from usable evidence or RGB."
                         if final else
                         "\nNo new evidence/status change. Execute a specific unresolved check or finish from RGB.")
        if note:
            text += "\nMODEL HYPOTHESIS (model_note_unverified; not a tool fact): " + note[:200]
        missing_dependencies = [key for key in dependency_ids
                                if key not in {r["result_id"] for r in visible}]
        text += (f"\nArchive {len(observations)}; omitted {len(observations)-len(visible)}. "
                 "Review/calculate only from visible TOOL FACTS.")
        if missing_dependencies:
            text += "\nOmitted calculation dependencies: " + json.dumps(
                [labels[k] for k in missing_dependencies if k in labels], separators=(",", ":"))
        depth_png, display = render_depth_image(
            cache.geometries[0]["depth"], cache.geometries[0]["valid_mask"])
        grid_image = Image.open(BytesIO(depth_png)).convert("RGB")
        grid = ImageDraw.Draw(grid_image)
        for x in range(0, w, max(1, w//5)):
            grid.line((x, 0, x, h-1), fill=(80, 120, 80))
            grid.text((x+2, 2), str(x), fill=(255, 230, 80), stroke_width=1, stroke_fill=(0, 0, 0))
        for y in range(0, h, max(1, h//5)):
            grid.line((0, y, w-1, y), fill=(80, 120, 80))
            grid.text((2, y+2), str(y), fill=(255, 230, 80), stroke_width=1, stroke_fill=(0, 0, 0))
        grid_bytes = BytesIO()
        grid_image.save(grid_bytes, format="PNG")
        depth_png = grid_bytes.getvalue()
        display.update(grid_version=COORDINATE_VERSION,
                       parent_png_sha256=display["png_sha256"],
                       png_sha256=hashlib.sha256(depth_png).hexdigest())
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
                crop_priority = list(dict.fromkeys([*pinned_ids, *selected_pending]))
                crops = crop_sheets(rgb, rgb_display["markers"], crop_priority)
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
                "Image 2: aligned predicted depth with original-pixel coordinate grid. Dark near, light far; "
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
                {"type": "text", "text": f"RGB crops {n+1}: original pixel scale; tile IDs/local sizes: " +
                 json.dumps([{k: m[k] for k in ("image_id", "width", "height")}
                             for m in info["markers"]], separators=(",", ":"))},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                  base64.b64encode(raw).decode()}}])
    if arm == "skill" and pending_program is not None:
        # Fresh requests do not otherwise contain the previous model response.
        # This is quoted, unexecuted model state, never a tool fact or permission
        # to run extracted wrappers.
        header = ("\nUNEXECUTED MODEL PROGRAM: tools are closed; this plan never ran."
                  if final else
                  "\nUNEXECUTED MODEL PROGRAM: this plan never ran. Repair its envelope/syntax and "
                  "resend the intended code in one strict python fence; do not mark its planned tools completed.")
        delivered_pending = dict(pending_program)
        if final:
            delivered_pending.pop("code")
            delivered_pending["code_omitted"] = "tools_closed"
        used_chars = sum(len(p["text"]) for p in parts if p["type"] == "text")
        if cap is not None and used_chars + len(header) + len(json.dumps(delivered_pending)) > cap:
            delivered_pending.pop("code", None)
            delivered_pending["code_omitted"] = "context_budget"
        pending_text = header + "\n" + json.dumps(delivered_pending, sort_keys=True, separators=(",", ":"))
        if cap is None or used_chars + len(pending_text) <= cap:
            parts[1]["text"] += pending_text
        else:
            delivered_pending = None
    if arm == "skill" and last_submission is not None:
        # Pack facts and actual crop captions first, then use only remaining
        # text space for this model state. It never displaces a tool record.
        header = "\n" + SUBMISSION_NOTICE + "\nLAST REJECTED SUBMISSION "
        used_chars = sum(len(p["text"]) for p in parts if p["type"] == "text")
        budget = min(SUBMISSION_CAP, max(0, cap-used_chars-len(header))) if cap is not None else SUBMISSION_CAP
        delivered_submission = submission_delivery(last_submission, budget)
        if delivered_submission is not None:
            parts[1]["text"] += header + json.dumps(
                delivered_submission, sort_keys=True, separators=(",", ":"))
        else:
            omission = "\nLAST REJECTED SUBMISSION omitted: context_budget; unverified state retained in host trace."
            if cap is None or used_chars + len(omission) <= cap:
                parts[1]["text"] += omission
    result = [{"role": "user", "content": parts}]
    text_chars = sum(len(p["text"]) for p in parts if p["type"] == "text")
    if cap is not None and text_chars > cap:
        raise ValueError("context_text_limit")
    images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    image_coordinates = [original_descriptor(rgb, w, h)] if arm == "skill" else []
    image_coordinates.extend(m for _, info in crops for m in info["markers"])
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
                                   "crop_priority_ids": crop_priority,
                                   "omitted_dependency_ids": missing_dependencies if arm == "skill" else [],
                                   "max_visible_measurements": 8 if final or arm == "direct" else None,
                                   "max_crop_sheets": 2},
               "surface_review_version": REVIEW_VERSION if arm == "skill" else None,
               "input_recovery_version": RECOVERY_VERSION if arm == "skill" else None,
               "answer_evidence_version": EVIDENCE_VERSION if arm == "skill" else None,
               "answer_origin_version": ORIGIN_VERSION if arm == "skill" else None,
               "surface_reviews_sha256": digest(reviews) if arm == "skill" else None,
               "surface_review_state": saved_reviews,
               "surface_review_counts": {d: sum(r["decision"] == d for r in reviews.values())
                                         for d in DECISIONS} if arm == "skill" else None,
               "rgb_sampling_display": rgb_display,
               "rgb_crop_sheets": [info for _, info in crops],
               "image_coordinates": image_coordinates,
               "public_answer_contract": specification(item) if arm == "skill" else None,
               "skill_sha256": hashlib.sha256(skill_text().encode()).hexdigest()
               if arm == "skill" else None}
    if arm == "skill":
        receipt["last_submission"] = {
            "version": SUBMISSION_VERSION, "state_sha256": digest(last_submission),
            "delivered_sha256": digest(delivered_submission),
            "source_sha256": last_submission["source_sha256"] if last_submission else None,
            "omitted": last_submission is not None and delivered_submission != last_submission}
        receipt["pending_program"] = {
            "version": PENDING_VERSION, "state_sha256": digest(pending_program),
            "delivered_sha256": digest(delivered_pending),
            "source_sha256": pending_program["sha256"] if pending_program else None,
            "code_delivered": bool(delivered_pending and "code" in delivered_pending)}
    if len(images) != (1 if arm == "direct" else 2+int(bool(rgb_display))+len(crops)):
        raise ValueError("image_count")
    return result, receipt
