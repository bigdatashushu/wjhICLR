"""One binary visual decision per measurement; receipts belong to the host."""
from __future__ import annotations

import ast
import base64
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

from PIL import Image, ImageDraw

from .common import digest, encoded
from .coordinates import intent
from .evidence import DIMENSIONS, fail
from .review import points

VERSION = "omni-jev-binary-review/1"
PROVIDERS = {
    "makers": ("https://ai-gateway.edgeone.link/v1/systemone", "@makers/jev",
               "MAKERS_MODELS_KEY", "jev.env"),
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "jev-1.14.0",
                 "TYPESAFE_API_KEY", "typesafe.env"),
}
ROLES = {"height": ("top", "base"), "width": ("left", "right"),
         "length": ("front", "back"), "diameter": ("rim_a", "rim_b")}
INSTRUCTIONS = (
    "Evaluate the marked RGB image. Model-supplied target names and quantity are "
    "proposals, not evidence. Answer yes only when ALL marked point centers hit "
    "the stated visible physical objects. For camera_z, the point must be on "
    "the object's visible surface. For chord, BOTH centers must be on their "
    "respective objects. For height/width/length/diameter, BOTH centers must "
    "be the COMPLETE extrema of the SAME object in A/B role order, and the "
    "segment must follow the stated object axis. An internal edge, part, "
    "background, wrong object, hidden/cropped extremum, incomplete extent, "
    "ambiguous hit or axis means no. Judge RGB hits and extent only; do not "
    "infer metric scale or validate predicted 3D accuracy."
)


@dataclass(frozen=True)
class Policy:
    provider: str = "makers"
    expected_model: str | None = None
    max_requests: int = 8
    timeout_s: int = 30
    endpoint: str = ""
    service_identity_sha256: str = ""
    yes_threshold: float | None = None

    def __post_init__(self):
        if self.expected_model is None:
            object.__setattr__(self, "expected_model",
                               "Intern-Decision-4B" if self.provider == "intern" else "jev-1.14.0")
        if self.provider == "intern":
            from urllib.parse import urlsplit
            from .intern_review import DEFAULT_ENDPOINT, DEFAULT_IDENTITY
            if not self.endpoint:
                object.__setattr__(self, "endpoint", DEFAULT_ENDPOINT)
            if not self.service_identity_sha256:
                object.__setattr__(self, "service_identity_sha256", DEFAULT_IDENTITY)
            parsed = urlsplit(self.endpoint)
            if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                    or parsed.username or parsed.password or parsed.query or parsed.fragment
                    or parsed.path not in ("", "/") or self.expected_model != "Intern-Decision-4B"
                    or not re.fullmatch("[0-9a-f]{64}", self.service_identity_sha256)):
                raise ValueError("intern_policy_invalid")
            from .intern_review import validate_threshold
            validate_threshold(self.effective_yes_threshold)
        elif (self.provider not in PROVIDERS or self.expected_model != "jev-1.14.0"
              or self.endpoint or self.service_identity_sha256 or self.yes_threshold is not None):
            raise ValueError("jev_policy_invalid")
        if (type(self.max_requests) is not int or not 1 <= self.max_requests <= 64
                or type(self.timeout_s) is not int or not 1 <= self.timeout_s <= 60):
            raise ValueError("jev_policy_invalid")

    def public(self):
        value = {"version": VERSION, **asdict(self)}
        # Preserve existing frozen policies and their receipt hashes.
        if self.yes_threshold is None:
            value.pop("yes_threshold")
        return value

    @property
    def effective_yes_threshold(self):
        from .intern_review import DEFAULT_YES_THRESHOLD
        return DEFAULT_YES_THRESHOLD if self.yes_threshold is None else self.yes_threshold

    @classmethod
    def parse(cls, value):
        fields = set(asdict(cls())) - {"yes_threshold"}
        if not isinstance(value, dict) or set(value) not in (
                {"version", *fields}, {"version", *fields, "yes_threshold"}):
            raise ValueError("jev_policy_invalid")
        policy = cls(**{k: v for k, v in value.items() if k != "version"})
        if value != policy.public():
            raise ValueError("jev_policy_changed")
        return policy


def sampling_intent(target, quantity, count):
    """The model supplies only object identity and the requested quantity."""
    if count == 1:
        if quantity != "camera_z" or not isinstance(target, str):
            raise ValueError("jev_point_requires_target_and_camera_z")
        names, roles = [target], ["visible_surface"]
    elif quantity in ROLES:
        if not isinstance(target, str):
            raise ValueError("jev_dimension_requires_one_target")
        names, roles = [target, target], ROLES[quantity]
    elif quantity == "chord":
        if not isinstance(target, list) or len(target) != 2:
            raise ValueError("jev_chord_requires_two_targets")
        names, roles = target, ["visible_surface", "visible_surface"]
    else:
        raise ValueError("jev_quantity_invalid")
    return intent([{"target": name, "surface": role} for name, role in zip(names, roles)], count)


def proposal(row):
    data = row["data"]
    quantity = data.get("jev_quantity")
    targets = data.get("intended_targets")
    count = 1 if row["source_tool"] == "point_depth" else 2
    target = targets[0]["target"] if targets and (count == 1 or quantity in ROLES) else (
        [t["target"] for t in targets] if targets else None)
    if sampling_intent(target, quantity, count) != targets:
        raise ValueError("jev_intent_changed")
    value = data.get("depth" if count == 1 else "distance")
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("jev_value_invalid")
    return {"source_tool": row["source_tool"], "quantity": quantity,
            "targets": targets, "points": points(row), "view_index": data["view_index"]}


def request_payload(row, rgb, policy):
    """Mark exact returned pixels on one image; no review histories or labels."""
    if policy.provider == "intern":
        from .intern_review import payload
        return payload(row, rgb, policy)
    proposed = proposal(row)
    with Image.open(BytesIO(rgb)) as original:
        image = original.convert("RGB")
    width, height = image.size
    xy = proposed["points"]
    if (proposed["view_index"] != 0 or any(
            len(p) != 2 or any(type(v) is not int for v in p)
            or not 0 <= p[0] < width or not 0 <= p[1] < height for p in xy)):
        raise ValueError("jev_pixel_invalid")
    # TypeSafe's image bounds. Pad small images without changing pixel positions;
    # resize large images and record the exact transform for the reviewer.
    scale = min(1.0, 1568 / width, 1568 / height)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    if resized != image.size:
        image = image.resize(resized, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max(280, resized[0]), max(280, resized[1])), (96, 96, 96))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    mapped = []
    for label, p in zip(("A", "B"), xy):
        x = (p[0] + .5) * resized[0] / width - .5
        y = (p[1] + .5) * resized[1] / height - .5
        mapped.append([x, y])
        # Hollow markers preserve the actual hit pixel.
        draw.ellipse((x-6, y-6, x+6, y+6), outline=(255, 60, 40), width=2)
        draw.text((min(canvas.width-12, x+8), max(0, y-12)), label,
                  fill=(255, 255, 0), stroke_width=1, stroke_fill=(0, 0, 0))
    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    png = buffer.getvalue()
    asset_id = "rgb_" + hashlib.sha256(png).hexdigest()[:24]
    state = {"image": asset_id, "proposal": proposed,
             "marker_order": ["A", "B"][:len(xy)], "marked_points": mapped,
             "original_size": [width, height], "image_content_size": list(resized),
             "padding_is_not_scene": True}
    payload = {"model": PROVIDERS[policy.provider][1], "state": state,
               "assets": {asset_id: {"type": "image", "format": "png",
                                    "data_base64": base64.b64encode(png).decode()}},
               "questions": {"usable": {"type": "choice", "instructions": INSTRUCTIONS,
                                        "criteria": {
                                            "yes": "All required RGB hit and full-extent conditions hold.",
                                            "no": "Any required condition fails or is uncertain."}}}}
    identity = {"version": VERSION, "result_id": row["result_id"],
                "provider": policy.provider, "expected_model": policy.expected_model,
                "policy_sha256": digest(policy.public()), "measurement_sha256": digest(row),
                "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
                "marked_image_sha256": hashlib.sha256(png).hexdigest(),
                "request_sha256": digest(payload), "reference_answer_visible": False,
                "automatic_retry": False}
    return payload, identity


def credentials(policy):
    """Read only the selected provider's key, never forward a key across hosts."""
    if policy.provider == "intern":
        return None
    _, _, name, filename = PROVIDERS[policy.provider]
    value = os.environ.get(name)
    if value:
        return value
    path = Path.home() / ".config" / "harness3d" / filename
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("jev_credential_unavailable")
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"(?:export\s+)?" + name + r"\s*=\s*(.+)", line.strip())
        if match:
            value = match[1].strip().strip("'\"")
            if value:
                return value
    raise ValueError("jev_credential_unavailable")


def seal(value):
    return {**value, "receipt_sha256": digest(value)}


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("jev_binary_answer_invalid")
        value[key] = item
    return value


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise urllib.error.HTTPError(request.full_url, code, "", headers, fp)


class Client:
    def __init__(self, policy, budget, *, loader=credentials, opener=None):
        self.policy, self.budget, self.loader = policy, budget, loader
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        self.count = 0

    def review(self, row, rgb):
        if self.policy.provider == "intern":
            from .intern_review import evaluate
            return evaluate(self, row, rgb)
        payload, identity = request_payload(row, rgb, self.policy)
        receipt = {**identity, "status": "service_error", "decision": None,
                   "request_attempted": False, "usage": {}}
        try:
            if self.count >= self.policy.max_requests:
                raise ValueError("jev_request_cap")
            secret = self.loader(self.policy)
            request = urllib.request.Request(
                PROVIDERS[self.policy.provider][0], data=encoded(payload),
                headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json",
                         "TypeSafe-Danger-Zone-Images": "true"}, method="POST")
            with self.budget.request(category="gate") as charge:
                self.count += 1
                receipt["request_attempted"] = True
                with self.opener.open(request, timeout=self.policy.timeout_s) as response:
                    raw = response.read()
                    receipt["http_status"] = response.status
                if receipt["http_status"] != 200:
                    raise ValueError("jev_http_status_invalid")
                if secret.encode() in raw:
                    raise ValueError("credential_in_response")
                value = json.loads(raw, object_pairs_hook=unique_object)
                usage = value.get("usage", {})
                receipt["usage"] = {k: v for k, v in usage.items()
                                    if k in ("input_tokens", "output_tokens")
                                    and type(v) is int and v >= 0}
                charge["usage"] = dict(receipt["usage"])
                if value.get("model") != self.policy.expected_model:
                    raise ValueError("jev_actual_model_changed")
                receipt["actual_model"] = self.policy.expected_model
                answers = value.get("answers", {})
                answer = answers.get("usable", {})
                if (set(answers) != {"usable"} or answer.get("type") != "choice"
                        or answer.get("choice") not in ("yes", "no")):
                    raise ValueError("jev_binary_answer_invalid")
                receipt.update(status="reviewed", decision=answer["choice"])
        except urllib.error.HTTPError as exc:
            # Never persist error bodies, exception messages or credentials.
            receipt.update(http_status=exc.code, error_code="jev_http_error")
            exc.close()
        except Exception as exc:
            code = str(exc) if type(exc) is ValueError else ""
            receipt["error_code"] = code if code in {
                "jev_request_cap", "jev_credential_unavailable", "credential_in_response",
                "jev_actual_model_changed", "jev_binary_answer_invalid",
                "jev_http_status_invalid"} else "jev_transport_error"
        return seal(receipt)


def host_review(row, receipt, rgb, policy):
    """Reconstruct the binary admission locally, including during offline audit."""
    _, expected = request_payload(row, rgb, policy)
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    if receipt.get("receipt_sha256") != digest(body) or any(
            receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("jev_receipt_changed")
    allowed = set(expected) | {"status", "decision", "request_attempted", "usage",
                               "http_status", "actual_model", "error_code", "receipt_sha256"}
    if policy.provider == "intern":
        allowed |= {"binary_answer", "vision_evidence", "calibration_sha256"}
    if (set(receipt) - allowed or type(receipt.get("request_attempted")) is not bool
            or not isinstance(receipt.get("usage"), dict)
            or any(k not in ("input_tokens", "output_tokens") or type(v) is not int or v < 0
                   for k, v in receipt["usage"].items())):
        raise ValueError("jev_receipt_invalid")
    usable = receipt.get("status") == "reviewed" and receipt.get("decision") == "yes"
    if receipt.get("status") == "reviewed":
        if (receipt.get("decision") not in ("yes", "no")
                or receipt.get("actual_model") != policy.expected_model
                or receipt.get("http_status") != 200 or not receipt["request_attempted"]
                or "error_code" in receipt):
            raise ValueError("jev_receipt_invalid")
        if policy.provider == "intern":
            from .intern_review import validate_answer, validate_vision
            if validate_answer(receipt.get("binary_answer", {}),
                               policy.effective_yes_threshold) != receipt["decision"]:
                raise ValueError("intern_binary_answer_invalid")
            validate_vision(receipt.get("vision_evidence", {}))
            if not re.fullmatch("[0-9a-f]{64}", receipt.get("calibration_sha256", "")):
                raise ValueError("intern_response_identity_changed")
    elif (receipt.get("status") != "service_error" or receipt.get("decision") is not None
          or not isinstance(receipt.get("error_code"), str)):
        raise ValueError("jev_receipt_invalid")
    surfaces = []
    for target in row["data"]["intended_targets"]:
        surface = {**target, "decision": "usable", "visibility": "visible"}
        if row["data"]["jev_quantity"] in ROLES:
            surface.update(extent_role=target["surface"], scope_completeness="complete",
                           extent_basis="Jev binary RGB hit, axis and complete-extrema judgment")
        surfaces.append(surface)
    review = {"result_id": row["result_id"], "decision": "usable" if usable else "resample",
              "reviewer": VERSION, "jev_decision": receipt["decision"],
              "jev_receipt_sha256": receipt["receipt_sha256"]}
    if usable:
        review["observed_surfaces"] = surfaces
    return review


def program_reviews(program):
    if any(isinstance(n, ast.Name) and n.id == "measurement_reviews"
           for n in ast.walk(ast.parse(program))):
        return {}, [{"code": "host_review_only", "field": "measurement_reviews",
                     "repair": "Submit target, quantity and points; the host reviews measurements."}]
    return {}, []


def checked_bindings(evidence, observations, reviews):
    """A yes cannot authorize a different quantity, pixel or declared geometry."""
    if not isinstance(evidence, dict) or not isinstance(evidence.get("bindings"), dict):
        return evidence
    rows = {r["result_id"]: r for r in observations}
    bindings = {}
    for name, original in evidence["bindings"].items():
        if not isinstance(original, dict) or original.get("result_id") not in rows:
            bindings[name] = original
            continue
        row, binding = rows[original["result_id"]], dict(original)
        field = f"evidence.bindings.{name}"
        review = reviews.get(row["result_id"], {})
        if (review.get("reviewer") != VERSION or review.get("jev_decision") != "yes"
                or review.get("decision") != "usable"):
            fail("calculation_measurement_not_usable", field + ".result_id")
        if binding.get("quantity") != row["data"].get("jev_quantity"):
            fail("jev_review_quantity_mismatch", field + ".quantity",
                 repair="Bind exactly the approved quantity, or request a new measurement.")
        if "geometry" in binding:
            fail("host_review_only", field + ".geometry",
                 repair="Omit geometry; the host binds the approved extrema.")
        if "sample_xy" in binding and binding["sample_xy"] != row["data"].get("pixel_xy"):
            fail("measurement_surface_required", field + ".sample_xy",
                 repair="Only the approved center pixel is usable; request another point.")
        quantity = binding.get("quantity")
        if quantity in DIMENSIONS:
            binding["geometry"] = {"coverage": "complete", "axis": DIMENSIONS[quantity][0],
                                   "anchors": list(ROLES[quantity]),
                                   "objects": [t["target"] for t in row["data"]["intended_targets"]]}
        bindings[name] = binding
    return {**evidence, "bindings": bindings}
