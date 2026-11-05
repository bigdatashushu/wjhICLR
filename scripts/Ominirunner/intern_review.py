"""Reuse the native Intern service for one measurement-level yes/no question."""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import math

from PIL import Image, ImageDraw

from skill3d.infra.m5_identity import content_sha256
from skill3d.segmentation.intern_decision_client import InternDecisionClient
from .common import digest

VERSION = "omni-intern-measurement-binary/2"
RENDERING_VERSION = "omni-intern-marked-overview/2"
DEFAULT_YES_THRESHOLD = .5
DEFAULT_ENDPOINT = "http://127.0.0.1:27248"
DEFAULT_IDENTITY = "c16e97ae7e43e8605e201de8388eb9a318bc5f260e05a59e869943a8762c0cb2"


def png(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def fit(image, edge):
    scale = min(1, edge / max(image.size))
    return image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))),
                        Image.Resampling.LANCZOS)


def mark(image, coordinates, source_box):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    left, top, right, bottom = source_box
    mapped = []
    for label, p in zip(("A", "B"), coordinates):
        x = (p[0]-left+.5) * image.width/(right-left)-.5
        y = (p[1]-top+.5) * image.height/(bottom-top)-.5
        mapped.append([x, y])
        draw.ellipse((x-6, y-6, x+6, y+6), outline=(255, 60, 40), width=2)
        draw.text((min(image.width-12, x+8), max(0, y-12)), label,
                  fill=(255, 255, 0), stroke_width=1, stroke_fill=(0, 0, 0))
    return marked, mapped


def question(proposed):
    """Ask only the RGB conditions required by this measurement quantity."""
    from .jev import ROLES
    quantity = proposed["quantity"]
    if quantity == "camera_z":
        instructions = (
            "Is the center of A on a directly visible physical surface of the stated target? "
            "A visible part of that target is a valid surface hit. Use the unmarked crop "
            "to inspect the center. Background, a gap, reflection, another object, an "
            "occluder or an unclear hit means no. Judge RGB surface membership only.")
        yes = "A center hits a visible physical surface of the stated target."
    elif quantity == "chord":
        instructions = (
            "Do A and B centers each hit a directly visible physical surface of their "
            "respective stated targets, in A/B order? Check both centers in the unmarked "
            "crop. Background, gaps, reflections, wrong objects, occluders or unclear "
            "hits mean no. Judge RGB surface membership only.")
        yes = "Both centers hit their respective visible target surfaces in A/B order."
    else:
        first, second = ROLES[quantity]
        instructions = (
            f"Do A and B mark the COMPLETE visible {quantity} of the stated object, "
            f"in A={first}, B={second} order? Both centers must hit the same physical "
            "object at its full extrema, and the segment must follow the requested "
            "object axis. Use the overview to check the whole extent and the unmarked "
            "crop to inspect the centers. Interior points, partial spans, background, "
            "wrong objects, wrong axis, hidden/cropped extrema or unclear hits mean no. "
            "Judge RGB hits, axis and complete extent only.")
        yes = f"Both centers are the same object's complete visible {quantity} extrema in A/B order."
    return {"type": "choice", "instructions": instructions,
            "criteria": {"yes": yes, "no": "A required RGB condition fails or is uncertain."}}


def payload(row, rgb, policy):
    from .jev import VERSION as REVIEW_VERSION, proposal
    proposed = proposal(row)
    with Image.open(BytesIO(rgb)) as image:
        original = image.convert("RGB")
    width, height = original.size
    coordinates = proposed["points"]
    if proposed["view_index"] != 0 or any(
            len(p) != 2 or any(type(v) is not int for v in p)
            or not 0 <= p[0] < width or not 0 <= p[1] < height for p in coordinates):
        raise ValueError("intern_pixel_invalid")
    xs, ys = zip(*coordinates)
    span_x, span_y = max(96, max(xs)-min(xs)+1), max(96, max(ys)-min(ys)+1)
    center_x, center_y = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    box = [max(0, math.floor(center_x-.7*span_x)),
           max(0, math.floor(center_y-.7*span_y)),
           min(width, math.ceil(center_x+.7*span_x+1)),
           min(height, math.ceil(center_y+.7*span_y+1))]
    crop = fit(original.crop(box), 448)
    marked, mapped = mark(crop, coordinates, box)
    overview, overview_points = mark(fit(original, 768), coordinates, [0, 0, width, height])
    images = [png(overview), png(crop), png(marked)]
    state = ("Three RGB images, in order: scene overview with hollow A/B markers; unmarked crop; "
             "the same crop with hollow A/B markers at the exact proposed pixel centers. "
             "Use the overview to locate the target and the unmarked crop to inspect the center. "
             "Marker rings are not object boundaries.\n"
             + json.dumps({"proposal": proposed, "original_size": [width, height],
                           "crop_source_box_xyxy": box, "marked_crop_points": mapped,
                           "overview_marked_points": overview_points},
                          separators=(",", ":")))
    request = {"state": state, "images": [
        "data:image/png;base64," + base64.b64encode(raw).decode() for raw in images],
               "questions": {"usable": question(proposed)}}
    identity = {"version": REVIEW_VERSION, "result_id": row["result_id"], "provider": "intern",
                "binary_prompt_version": VERSION,
                "review_render_version": RENDERING_VERSION,
                "expected_model": policy.expected_model,
                "policy_sha256": digest(policy.public()), "measurement_sha256": digest(row),
                "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
                "marked_image_sha256": hashlib.sha256(images[-1]).hexdigest(),
                "review_image_sha256": [hashlib.sha256(raw).hexdigest() for raw in images],
                "request_sha256": content_sha256(request),
                "service_identity_sha256": policy.service_identity_sha256,
                "reference_answer_visible": False, "automatic_retry": False}
    return request, identity


def validate_threshold(threshold):
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not .5 <= threshold <= 1):
        raise ValueError("intern_yes_threshold_invalid")


def validate_answer(answer, yes_threshold=DEFAULT_YES_THRESHOLD):
    validate_threshold(yes_threshold)
    probabilities = answer.get("probabilities", {})
    if (answer.get("type") != "choice" or set(probabilities) != {"yes", "no"}
            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                   for p in probabilities.values())
            or abs(sum(probabilities.values())-1) > 1e-5):
        raise ValueError("intern_binary_answer_invalid")
    choice = answer.get("choice")
    confidence = answer.get("confidence")
    if (choice not in probabilities or answer.get("decision") != choice
            or probabilities[choice] != max(probabilities.values())
            or type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not math.isclose(confidence, probabilities[choice], abs_tol=1e-6)):
        raise ValueError("intern_binary_answer_invalid")
    # The service choice remains in binary_answer; host admission uses the
    # calibrated probability and the threshold frozen in the policy.
    return "yes" if probabilities["yes"] > yes_threshold else "no"


def validate_vision(vision):
    if (vision.get("image_count") != 3 or type(vision.get("image_token_count")) is not int
            or vision["image_token_count"] <= 0 or not vision.get("pixel_values_shape")
            or len(vision.get("image_grid_thw", [])) != 3
            or not vision.get("decision_logit_positions")
            or set(vision.get("legal_symbol_token_ids", {})) != {"usable"}
            or len(vision["legal_symbol_token_ids"]["usable"]) != 2):
        raise ValueError("intern_visual_evidence_invalid")


def native_client(policy, opener):
    return InternDecisionClient(
        policy.endpoint, timeout_s=policy.timeout_s,
        expected_identity_sha256=policy.service_identity_sha256,
        opener=lambda request, timeout: opener.open(request, timeout=timeout))


def evaluate(client, row, rgb):
    """One inference; derive host admission from the frozen probability threshold."""
    from .jev import seal
    request, identity = payload(row, rgb, client.policy)
    receipt = {**identity, "status": "service_error", "decision": None,
               "request_attempted": False, "usage": {}}
    try:
        if client.count >= client.policy.max_requests:
            raise ValueError("jev_request_cap")
        native = native_client(client.policy, client.opener)
        service = native.identity()
        with client.budget.request(category="gate") as charge:
            client.count += 1
            receipt["request_attempted"] = True
            body = native._call("/decisions", {
                "request_sha256": identity["request_sha256"], "request": request})
            if (body.get("request_sha256") != identity["request_sha256"]
                    or body.get("identity_sha256") != client.policy.service_identity_sha256):
                raise ValueError("intern_response_identity_changed")
            result = body.get("result", {})
            if (result.get("model") != "Intern-Decision-4B" or result.get("backend") != "hf"
                    or result.get("calibration") != service["calibration"]
                    or set(result.get("answers", {})) != {"usable"}):
                raise ValueError("intern_response_identity_changed")
            answer = result["answers"]["usable"]
            decision = validate_answer(answer, client.policy.effective_yes_threshold)
            vision = body.get("vision_evidence", {})
            validate_vision(vision)
            usage = {"input_tokens": vision["input_tokens"]} if (
                type(vision.get("input_tokens")) is int and vision["input_tokens"] >= 0) else {}
            charge["usage"] = usage
            receipt.update(status="reviewed", decision=decision, http_status=200,
                           actual_model="Intern-Decision-4B", usage=usage,
                           binary_answer={k: answer[k] for k in (
                               "type", "choice", "decision", "probabilities", "confidence")},
                           vision_evidence={k: vision[k] for k in (
                               "image_count", "image_token_count", "pixel_values_shape",
                               "image_grid_thw", "decision_logit_positions", "legal_symbol_token_ids")},
                           calibration_sha256=digest(service["calibration"]))
    except Exception as exc:
        import urllib.error
        if isinstance(exc, urllib.error.HTTPError):
            receipt.update(http_status=exc.code, error_code="intern_http_error")
            exc.close()
        else:
            code = str(exc) if type(exc) is ValueError else ""
            receipt["error_code"] = code if code in {
                "jev_request_cap", "intern_response_identity_changed",
                "intern_binary_answer_invalid", "intern_visual_evidence_invalid"} else "intern_transport_error"
    return seal(receipt)
