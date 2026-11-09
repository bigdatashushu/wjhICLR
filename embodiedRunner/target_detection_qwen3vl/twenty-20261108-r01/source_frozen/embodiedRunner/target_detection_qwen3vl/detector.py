"""Qwen normalized boxes become original-image pixel edges before grouping."""
import base64
import io
import json
import math
from pathlib import Path
import re
import shlex
import subprocess

from target_detection_r05.transport import child_environment
from .sources import OperationalError, canonical, digest


def pixel_box(box, size):
    """Map [x1,y1,x2,y2] on 0..1000 to 0..width/height, without early rounding."""
    if (not isinstance(box, list) or len(box) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
        raise OperationalError("invalid_normalized_box")
    width, height = size
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0
            or any(not 0 <= v <= 1000 for v in box)
            or box[0] >= box[2] or box[1] >= box[3]):
        raise OperationalError("invalid_normalized_box")
    return [box[0] * width / 1000, box[1] * height / 1000,
            box[2] * width / 1000, box[3] * height / 1000]


def detection_prompt(category):
    return (
        "Locate every clearly visible instance of this category in the supplied original RGB: "
        + json.dumps(category) + ". Return only a JSON object in this exact format: "
        '{"detections":[{"label":' + json.dumps(category)
        + ',"bbox_2d":[x1,y1,x2,y2]}]}. '
        "bbox_2d is [left,top,right,bottom] in NORMALIZED 0-1000 coordinates: "
        "x=0 is the left edge, x=1000 the right edge, y=0 the top edge, y=1000 the bottom edge. "
        "Use tight boxes around the visible object. Return each physical instance once. "
        "Only detect the requested category; never substitute a robot, background, or nearby object. "
        'If none is visible return {"detections":[]}. Do not output confidence scores, '
        "explanations, markdown, or pixel coordinates."
    )


class Detector:
    def __init__(self, config):
        self.config = config["detection"]
        self.script = Path(__file__).with_name("detector_worker.py").read_text()

    def request(self, mode, payload=None):
        service = self.config["service"]
        value = {"mode": mode, "service": service}
        if payload is not None:
            value["payload"] = payload
        args = ["/usr/bin/python3", "-B", "-"]
        if service["transport"] == "ssh":
            args = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                    "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=10",
                    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                    service["ssh_target"], shlex.join(args)]
        try:
            result = subprocess.run(
                args, input=("INPUT=" + repr(value) + "\n" + self.script).encode(),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=child_environment(), timeout=service["timeout_s"] + 30, check=False)
            report = json.loads(result.stdout)
            if result.returncode or report.get("status") != "ok":
                raise OperationalError(report.get("code", "qwen_detector_process_error"))
            if (report.get("model") != service["model"]
                    or report.get("service_identity_sha256") != digest(canonical(service))):
                raise OperationalError("qwen_detector_identity_mismatch")
            if payload is not None and (report.get("request_sha256") != digest(canonical(payload))
                                        or report.get("detector_requests") != 1):
                raise OperationalError("qwen_detector_request_identity_mismatch")
            return report
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
            raise OperationalError("qwen_detector_transport_or_protocol_error") from None

    def status(self):
        return self.request("status")

    def detect(self, image, category):
        if (not isinstance(category, str)
                or not re.fullmatch(r"[A-Za-z][A-Za-z -]{0,59}", category)):
            raise OperationalError("invalid_detection_category")
        from PIL import Image
        try:
            raw = base64.b64decode(image["base64"], validate=True)
            if digest(raw) != image["sha256"]:
                raise OperationalError("detection_image_hash_mismatch")
            with Image.open(io.BytesIO(raw)) as rgb:
                if rgb.format != "PNG":
                    raise OperationalError("detection_requires_original_png")
                rgb.load()
                width, height = rgb.size
        except (OSError, ValueError, KeyError):
            raise OperationalError("invalid_detection_image") from None
        payload = {
            "model": self.config["service"]["model"],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image["base64"]}},
                {"type": "text", "text": detection_prompt(category)}]}],
            "temperature": self.config["temperature"], "top_p": 1,
            "max_tokens": self.config["max_tokens"],
            "response_format": {"type": "json_object"},
        }
        report = self.request("detect", payload)
        value = report.pop("output")
        if not isinstance(value, dict) or set(value) != {"detections"}:
            raise OperationalError("invalid_qwen_detection_output")
        rows = value["detections"]
        if not isinstance(rows, list) or len(rows) > 128:
            raise OperationalError("invalid_qwen_detection_count")
        detections = []
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or set(row) != {"label", "bbox_2d"}
                    or row["label"] != category):
                raise OperationalError("invalid_qwen_detection_fields")
            detections.append({
                "id": f"D{index + 1:03d}", "label": row["label"],
                "bbox_1000": row["bbox_2d"], "bbox": pixel_box(row["bbox_2d"], (width, height)),
            })
        report.update(detections=detections, image_sha256=digest(raw), image_size=[width, height],
                      query=category, coordinate_system="original_image_pixel_edges",
                      normalized_coordinate_range=[0, 1000], confidence_available=False)
        return report
