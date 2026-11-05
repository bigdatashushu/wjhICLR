"""Compact solver context: approved facts and terse reselect notices."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import math

from skill3d.small_image.depth_image import render_depth_image

from .answer_contract import specification, LENGTH_UNIT_METERS
from .common import digest, skill_text
from .coordinates import original_descriptor
from .evidence import public_span_matches
from .jev import Policy, VERSION
from .review import pending, points
from .tools import MEASUREMENTS, empty_counts, phase_for

CONTEXT_VERSION = "omni-host-binary-context/3"
REPAIR_VERSION = "omni-host-repair-context/1"
TOOL_GUIDE = """Original grid: x right, y down; view_index=0. Measurements have same-view relative units.
Tools:
ctx.tools.point_depth(view_index=0,x=X,y=Y,target="chair",quantity="camera_z")
ctx.tools.pair_point_distance(view_index=0,a_xy=[X,Y],b_xy=[X,Y],target="chair",quantity="height")
Pair also supports width,length,diameter; chord uses target=["chair","desk"].
Optional replaces="approved or reselect ID"; image_id="rgb-original",coordinate_space="normalized" uses [0,1].
ctx.tools.view_candidates(view_index=0,category="") and ctx.tools.view_depth_summary() give hints.
Return one strict ```python block containing only def solve(ctx):. Host invokes it once.
End with GLOBAL YieldObservations([r["result_id"]]) or ReturnAnswer(value,evidence=E).
Never call controls through ctx.tools. Variables reset; safe math/numpy; no host I/O.
RGB approval does not validate predicted depth accuracy or cross-view/metric geometry.
"""
EVIDENCE_GUIDE = """E calculation: {"kind":"calculation","expression":"v","bindings":{
"v":{"result_id":"APPROVED_ID","quantity":"APPROVED_QUANTITY"}}}.
Replace v with the task formula and use all named bindings: numbers,parentheses,+,-,*,/; <=200 chars.
Use exact approved values, targets, quantities and ordered operands. Binding operand_index selects
a public task role; reference records the common origin. No measurement_reviews,geometry,sample_xy.
Host checks arithmetic, operands, units and next-request delivery; never round intermediate values.
Independent RGB E: {"kind":"visual_estimate","basis":"brief RGB reasoning <=200 chars"}.
After abandoning a rejected calculation add E.provenance={"source":"rgb_only",
"referenced_result_ids":[],"abandoned_result_ids":[prior IDs],"abandonment_reason":"brief reason"}.
Tool-derived answers require checked calculation/tool_comparison bindings.
"""


def proposal(row):
    """Only sampling identity; no review internals or measured values."""
    data = row["data"]
    targets = [t["target"] for t in data["intended_targets"]]
    return {"result_id": row["result_id"], "target": targets if data["jev_quantity"] == "chord" else targets[0],
            "quantity": data["jev_quantity"], "view_index": data["view_index"], "points": points(row)}


def repair_errors(errors, final=False):
    """Retain actionable host diagnostics without old self-review instructions."""
    overrides = {
        "measurement_surface_required": "Use only the approved center; request new points and wait for approval.",
        "calculation_measurement_not_usable": "Reselect points; only approved TOOL FACTS support calculations.",
        "calculation_measurement_replaced": "Bind the replacement after its own approval; the old ID is excluded.",
        "final_answer_fields": 'Output only {"answer":value,"evidence":E}.',
        "answer_evidence_fields": "Use visual_estimate+basis, calculation+expression+bindings, or tool_comparison. Omit measurement_reviews and geometry.",
    }
    program_repairs = {
        "python_syntax_error": "Fix the reported Python syntax.",
        "solve_entry_required": "Use only unannotated, undecorated def solve(ctx), without defaults.",
        "control_namespace_error": "Use GLOBAL ReturnAnswer/YieldObservations.",
        "program_missing_action": "End solve with GLOBAL ReturnAnswer or YieldObservations.",
    }
    result = []
    for error in list(errors)[:4]:
        row = {k: error[k] for k in ("code", "field", "tool", "line", "syntax",
                                    "allowed_fields", "missing_fields", "unexpected_fields") if k in error}
        code = error.get("code")
        if error.get("executed") is False or error.get("currentphase") == "program":
            row["executed"] = False
            row["repair"] = ("Tools closed. The plan never ran; answer from RGB or approved delivered facts."
                             if final else program_repairs.get(code, "Remove prefix/tail/XML wrappers.") + (
                                 " Resend one closed python fence containing def solve(ctx); global controls, "
                                 "no measurement_reviews. The unexecuted tool proposals never ran."))
        elif code in ("omni_invalid_input", "invalid_tool_arguments"):
            if "expected" in error:
                row["expected"] = error["expected"]
            row["repair"] = ("Tools closed. Failed results are untrusted; use approved facts or an independent RGB fallback."
                             if final else
                             "Correct this call in a new solve cell; use target,quantity,coordinates. Failed results are untrusted.")
        elif final and code == "answer_calculation_mismatch":
            row["repair"] = "Recompute from approved values and the declared expression; do not round. Output answer/evidence JSON only."
        elif final and code in ("calculation_measurement_not_usable", "calculation_measurement_not_delivered",
                                "calculation_measurement_replaced", "measurement_surface_required",
                                "jev_review_quantity_mismatch", "calculation_measurement_unknown"):
            row["repair"] = "Tools closed. Use currently approved matching IDs, or an independent RGB fallback listing abandoned IDs."
        elif code in overrides:
            row["repair"] = overrides[code]
        elif isinstance(error.get("repair"), str):
            row["repair"] = error["repair"]
        result.append(row)
    return result


def calculation_state(snapshot, task, observations, approved_ids):
    """Quote bounded repair evidence, never its answer or blocked values."""
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("evidence"), dict):
        return None
    evidence = snapshot["evidence"]
    if evidence.get("kind") not in ("calculation", "tool_comparison", "tool_estimate", "visual_estimate"):
        return None
    state = {"status": "unverified_rgb_declaration" if evidence["kind"] == "visual_estimate"
             else "unverified_model_formula", "round": snapshot.get("round"),
             "source_sha256": snapshot["source_sha256"]}
    projected = {k: evidence[k] for k in ("kind", "basis", "left", "right", "operator", "answers") if k in evidence}
    blocked_values = [row["data"].get("depth" if row["source_tool"] == "point_depth" else "distance")
                      for row in observations
                      if row["source_tool"] in MEASUREMENTS and row["result_id"] not in approved_ids]
    expression = evidence.get("expression")
    if isinstance(expression, str) and len(expression) <= 200:
        try:
            tree = ast.parse(expression, mode="eval")
            leaks = any(isinstance(n, ast.Constant) and type(n.value) in (int, float)
                        and n.value in blocked_values for n in ast.walk(tree))
        except (SyntaxError, ValueError, RecursionError):
            leaks = True
        if not leaks:
            projected["expression"] = expression
        else:
            state["expression_omitted"] = "invalid_or_contains_blocked_value"
    if isinstance(evidence.get("bindings"), dict):
        projected["bindings"] = {
            name: {k: binding[k] for k in ("result_id", "quantity", "reference", "operand_index") if k in binding}
            for name, binding in evidence["bindings"].items() if isinstance(binding, dict)}
    scale = evidence.get("scale")
    if isinstance(scale, dict):
        projected["scale"] = {k: scale[k] for k in ("name", "unit", "anchor_binding") if k in scale}
        value, unit = scale.get("value"), scale.get("unit")
        if (type(value) in (int, float) and abs(value) < 1e100 and math.isfinite(value)
                and isinstance(unit, str) and unit in LENGTH_UNIT_METERS
                and any(public_span_matches(span, value, unit) for span in task["public_metric_values"])):
            projected["scale"]["value"] = value
        else:
            state["scale_value_omitted"] = "use_public_task_span_in_answer_unit"
    assumptions = evidence.get("assumptions")
    if isinstance(assumptions, dict):
        projected["assumptions"] = {k: assumptions[k] for k in ("object_scope", "volume_model")
                                    if isinstance(assumptions.get(k), str)}
    scope = evidence.get("object_scope")
    if isinstance(scope, dict):
        projected["object_scope"] = {k: scope[k] for k in ("include", "exclude") if k in scope}
        if isinstance(scope.get("entities"), list):
            projected["object_scope"]["entities"] = [
                {k: entity[k] for k in ("entity_id", "category", "candidate_ids", "relationship", "basis", "included")
                 if k in entity} for entity in scope["entities"] if isinstance(entity, dict)]
    if isinstance(evidence.get("count_groups"), dict):
        projected["count_groups"] = {k: evidence["count_groups"][k] for k in ("numerator", "denominator")
                                     if k in evidence["count_groups"]}
    origin = evidence.get("provenance")
    if isinstance(origin, dict):
        projected["provenance"] = {k: origin[k] for k in (
            "source", "referenced_result_ids", "abandoned_result_ids", "abandonment_reason") if k in origin}
    # Free prose in RGB scope/shape declarations is also model state. Do not
    # return a blocked value through a basis or an abandonment explanation.
    serialized = json.dumps(projected, separators=(",", ":"))
    if any(type(v) in (int, float) and str(v) in serialized for v in blocked_values):
        state["evidence_omitted"] = "contains_blocked_value"
        return state
    state["evidence"] = projected
    if len(json.dumps(state, separators=(",", ":"))) > 1800:
        return {"status": state["status"], "source_sha256": state["source_sha256"],
                "omitted": "repair_state_budget"}
    return state


def unexecuted_state(snapshot, final):
    """Extract literal tool proposals for context only; never runnable code."""
    if not isinstance(snapshot, dict):
        return None
    state = {"status": "unexecuted_model_tool_plan", "source_sha256": snapshot["sha256"],
             "rejection_code": snapshot["rejection_code"], "execution_authorized": False}
    if final:
        return {**state, "proposals_omitted": "tools_closed"}
    try:
        tree = ast.parse(snapshot["code"])
    except (SyntaxError, ValueError, RecursionError):
        return {**state, "proposals_omitted": "invalid_python"}
    proposals = []
    allowed = {"view_index", "x", "y", "a_xy", "b_xy", "target", "quantity", "replaces",
               "image_id", "coordinate_space", "category"}
    for node in ast.walk(tree):
        fn = node.func if isinstance(node, ast.Call) else None
        if not (isinstance(fn, ast.Attribute) and fn.attr in MEASUREMENTS | {"view_candidates", "view_depth_summary"}
                and isinstance(fn.value, ast.Attribute) and fn.value.attr == "tools"
                and isinstance(fn.value.value, ast.Name) and fn.value.value.id == "ctx"):
            continue
        args, unresolved = {}, ["positional_arguments"] if node.args else []
        for kw in node.keywords:
            if kw.arg not in allowed:
                unresolved.append(kw.arg or "expanded_arguments")
                continue
            try:
                value = ast.literal_eval(kw.value)
                json.dumps(value, allow_nan=False)
                args[kw.arg] = value
            except (ValueError, TypeError, RecursionError):
                unresolved.append(kw.arg)
        proposals.append({"tool": fn.attr, "arguments": args, "unresolved_fields": unresolved})
    state["literal_tool_proposals"] = proposals
    if len(json.dumps(state, separators=(",", ":"))) > 1800:
        return {k: v for k, v in state.items() if k != "literal_tool_proposals"} | {
            "proposals_omitted": "repair_state_budget"}
    return state


def messages(item, rgb, frame, cache, *, config, policy, round_index=0,
             observations=(), errors=(), counters=None, reviews=None, dependency_ids=(),
             last_submission=None, pending_program=None, **unused):
    policy = Policy.parse(policy)
    reviews = reviews or {}
    counters = empty_counts() if counters is None else dict(counters)
    if not 0 <= round_index < config.request_limit("skill"):
        raise ValueError("round_outside_episode")
    phase = phase_for(config, round_index)
    final = phase == "final"
    task = specification(item)
    public = {k: v for k, v in task.items() if k != "geometry_verification"}
    skill = skill_text(host_review=True)
    requests_left = config.request_limit("skill") - round_index
    text = (f"Grid {frame.shape[1]}x{frame.shape[0]}; request {round_index+1}/{config.request_limit('skill')}; "
            f"{0 if final else config.max_program_rounds-round_index} program requests left, then final JSON; "
            f"host reviews used {sum(r['source_tool'] in MEASUREMENTS for r in observations)}/{policy.max_requests}.\n")
    text += "\nPUBLIC TASK " + json.dumps(public, separators=(",", ":"))
    text += ("" if final else "\nMEASUREMENT SKILL\n" + skill + "\n" + TOOL_GUIDE)
    text += "\n" + EVIDENCE_GUIDE
    if task["allowed_answers"]:
        text += '\nComparison E: {"kind":"tool_comparison","bindings":{...},"left":"a","right":"b",' \
                '"operator":"<","answers":{"true":"O1","false":"O2"}}. Same units/reference; return public option ID.'
        if task["reference"]:
            text += '\nEach comparison binding requires reference="' + task["reference"] + \
                    '". Camera uses camera_z; object distance uses chord [candidate,reference] in A/B order.'
    if task["public_metric_values"]:
        text += '\nUse E.scale={name,value,unit,anchor_binding}; value is the public span converted ' \
                'to the task unit (public answer_unit_value); anchor_binding is that object\'s approved complete dimension. ' \
                'Expression e.g. s*h/w. Units m/ft/in/yd/mi; 1 mi=5280 ft.'
    if task["quantity"] in ("count", "count_ratio"):
        text += '\nE.object_scope={include,exclude,entities}; each entity has entity_id,category,' \
                'candidate_ids,relationship,basis,included. relationship rgb_only/single/same_object/part_whole; ' \
                'merge only with RGB basis, never box overlap. For count_ratio add count_groups ' \
                'with ordered numerator/denominator entity IDs.'
    if task["quantity"] in ("volume", "volume_ratio"):
        text += '\nE.assumptions={object_scope:"included objects",volume_model:"shape model"}; ' \
                'both nonempty strings <=200 chars.'
    replaced = {r["data"]["replaces_result_id"]: r["result_id"] for r in observations
                if r["source_tool"] in MEASUREMENTS and r["data"].get("replaces_result_id")}
    facts, visible = [], []
    notices = []
    for row in observations:
        key, data = row["result_id"], row["data"]
        if row["source_tool"] in MEASUREMENTS:
            if key in replaced:
                continue
            review = reviews.get(key, {})
            if review.get("decision") != "usable" or review.get("jev_decision") != "yes":
                notices.append({**proposal(row), "action": "reselect points"})
                continue
            facts.append({**proposal(row),
                          "value": data["depth" if row["source_tool"] == "point_depth" else "distance"],
                          "unit": data["unit"]})
            visible.append(row)
        else:
            # Match count validation's first-eight candidate contract.
            from .protocol import compact
            facts.extend(compact([row]))
            visible.append(row)
    text += "\nTOOL FACTS " + json.dumps(facts, separators=(",", ":"))
    if notices:
        text += "\nRESELECT " + json.dumps(notices, separators=(",", ":"))
        text += "\nReselect IDs authorize replacement only, never calculation. Rejection does not prove object absence."
    if replaced:
        text += "\nSUPERSEDED IDs (excluded) " + json.dumps(replaced, separators=(",", ":"))
    if dependency_ids:
        text += "\nPrior calculation IDs " + json.dumps(list(dependency_ids))
    if errors:
        text += "\nERRORS " + json.dumps(repair_errors(errors, final), separators=(",", ":"))
    if final:
        text += '\nTools closed. Output only {"answer":value,"evidence":E}; native scalar, no units. ' \
                'Recompute from approved values and the same task formula with full precision; do not copy a rejected answer.'
    parts = [{"type": "text", "text": item["question"]}, {"type": "text", "text": text},
             {"type": "text", "text": "Original RGB."},
             {"type": "image_url", "image_url": {
                 "url": "data:image/jpeg;base64," + base64.b64encode(rgb).decode()}}]
    depth, display = render_depth_image(cache.geometries[0]["depth"], cache.geometries[0]["valid_mask"])
    parts.extend([{"type": "text", "text": "Aligned predicted depth: dark near, light far; relative units."},
                  {"type": "image_url", "image_url": {
                      "url": "data:image/png;base64," + base64.b64encode(depth).decode()}}])
    context = [{"role": "user", "content": parts}]
    chars = sum(len(p["text"]) for p in parts if p["type"] == "text")
    cap = config.text_caps[-1 if final else round_index]
    approved_ids = [r["result_id"] for r in visible if r["source_tool"] in MEASUREMENTS]
    delivered_states = {}
    for header, state in (
            ("REPAIR EVIDENCE (unverified; recheck against task and approved facts)",
             calculation_state(last_submission, task, observations, approved_ids)),
            ("UNEXECUTED TOOL PLAN (proposals only; no calls completed)",
             unexecuted_state(pending_program, final))):
        value = state
        if state is not None:
            block = "\n" + header + " " + json.dumps(state, separators=(",", ":"))
            if cap is None or chars + len(block) <= cap:
                parts[1]["text"] += block
                chars += len(block)
            else:
                value = None
        delivered_states[header] = value
    if cap is not None and chars > cap:
        raise ValueError("context_text_limit")
    receipt = {"messages_sha256": digest(context), "phase": phase, "round_index": round_index,
               "context_version": CONTEXT_VERSION,
               "requests_remaining": requests_left,
               "skill_sha256": hashlib.sha256(skill.encode()).hexdigest(),
               "text_chars": chars, "text_cap": cap, "text_record_threshold": config.text_record_threshold,
               "text_over_threshold": chars > config.text_record_threshold,
               "text_over_threshold_chars": max(0, chars-config.text_record_threshold),
               "image_count": 2, "image_sha256": [hashlib.sha256(raw).hexdigest() for raw in (rgb, depth)],
               "image_coordinates": [original_descriptor(rgb, frame.shape[1], frame.shape[0])],
               "visible_result_ids": [r["result_id"] for r in visible],
               "visible_measurement_ids": [r["result_id"] for r in visible if r["source_tool"] in MEASUREMENTS],
               "reselect_measurement_ids": [r["result_id"] for r in notices],
               "superseded_measurement_ids": replaced,
               "pending_measurement_ids": pending(observations, reviews), "tool_counts": counters,
               "tools_enabled": not final, "depth_display": display,
               "measurement_reviewer": policy.public(), "surface_review_version": VERSION,
               "host_state_sha256": digest({"observations": list(observations), "reviews": reviews}),
               "visible_facts_sha256": digest(facts),
               "repair_context": {"version": REPAIR_VERSION,
                                  "state_sha256": digest({"last_submission": last_submission,
                                                          "pending_program": pending_program}),
                                  "delivered_sha256": digest(delivered_states)}}
    return context, receipt
