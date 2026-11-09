"""Describe the actual masked region before the existing binary reviewer."""
from __future__ import annotations

import json
import base64
import hashlib
import math
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from skill3d.infra.m5_identity import content_sha256

VERSION = "vsi-qwen3vl8b-target-description/1"
PROMPT = """Describe the physical region actually covered by the turquoise mask.
The three images are: (1) the full scene with an orange bounding rectangle,
(2) an unmasked crop, (3) the same crop with a turquoise mask.
The requested object name below is a hint, not proof of what the mask covers.
Use the unmasked appearance and masked pixels to describe the object and the
visible part. For example, "refrigerator" may become "the lower half of a white
refrigerator". If it covers a different object, name that object; if mixed or
unclear, say so. Do not answer yes/no, assess metric geometry, solve a benchmark
question, or invent unseen parts. Use a short, concrete English noun phrase.
Return exactly one JSON object: {"description": "your description"}.
Requested object name: """
CONFIG_FIELDS = {"base_url", "model", "api_key_env", "timeout_s", "max_tokens"}


def description_config(value):
    if not isinstance(value, dict) or set(value) != CONFIG_FIELDS:
        raise ValueError("Qwen description requires the complete service configuration")
    result = dict(value)
    parsed = urlsplit(result["base_url"])
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"):
        raise ValueError("Qwen description base_url must end in /v1 without credentials")
    if result["model"] not in {
            "Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Instruct-FP8", "qwen3vl-8b-r0"}:
        raise ValueError("description model must be Qwen3-VL-8B")
    if not isinstance(result["api_key_env"], str) or not re.fullmatch(
            r"[A-Z][A-Z0-9_]*", result["api_key_env"]):
        raise ValueError("Qwen credential must be an environment variable name")
    timeout = result["timeout_s"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("invalid Qwen timeout")
    if type(result["max_tokens"]) is not int or not 64 <= result["max_tokens"] <= 1024:
        raise ValueError("invalid Qwen description token limit")
    return result


def parse_description(raw):
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate description field")
            result[key] = value
        return result
    result = json.loads(text, object_pairs_hook=unique)
    if not isinstance(result, dict) or set(result) != {"description"}:
        raise ValueError("description response fields differ")
    description = result["description"]
    if (not isinstance(description, str) or not 1 <= len(description.strip()) <= 240
            or any(ord(char) < 32 for char in description)
            or any(char in description for char in ('"', "`", "{", "}", "<", ">"))):
        raise ValueError("description must be a short plain noun phrase")
    return description.strip()


def description_request(images, target_category, config, seed):
    if (len(images) != 3 or not isinstance(target_category, str)
            or not re.fullmatch(r"[a-z]+(?:[ -][a-z]+)*", target_category)
            or len(target_category) > 80):
        raise ValueError("invalid description candidate")
    return {
        "model": config["model"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT + target_category},
            *[{"type": "image_url", "image_url": {"url": row["url"]}} for row in images],
        ]}],
        "temperature": 0.0, "top_p": 1.0, "seed": int(seed),
        "max_tokens": config["max_tokens"], "response_format": {"type": "json_object"},
    }


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise urllib.error.HTTPError(request.full_url, code, "", headers, fp)


class DescriptionFailure(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__("Qwen object description unavailable")


class Qwen8BDescriptionClient:
    def __init__(self, config, *, opener=None, authorize=None):
        self.config = description_config(config)
        self.opener = opener or urllib.request.build_opener(NoRedirect()).open
        self.authorize = authorize

    def audit_identity(self):
        return {"version": VERSION, "config": self.config, "prompt_sha256": content_sha256(PROMPT)}

    def describe(self, images, target_category, *, seed):
        payload = description_request(images, target_category, self.config, seed)
        receipt = {
            "version": VERSION, "status": "unavailable", "request_sent": False,
            "request": payload, "request_sha256": content_sha256(payload),
            "service": self.audit_identity(), "target_category": target_category,
            "image_sha256": [row["identity"]["sha256"] for row in images],
            "reference_answer_visible": False, "automatic_retry": False,
        }
        started = time.perf_counter()
        key = os.environ.get(self.config["api_key_env"], "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        try:
            # This optional admission hook is used by the preparation request cap.
            if self.authorize is not None:
                self.authorize(receipt["request_sha256"])
            from skill3d.infra.vsi_preparation_budget import reserve
            reserve("qwen3vl8b_description", payload)
            request = urllib.request.Request(
                self.config["base_url"].rstrip("/") + "/chat/completions",
                data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                headers=headers, method="POST")
            receipt["request_sent"] = True
            with self.opener(request, timeout=self.config["timeout_s"]) as response:
                raw_body = response.read(2 * 1024 * 1024 + 1)
            if len(raw_body) > 2 * 1024 * 1024:
                raise ValueError("description response exceeds size limit")
            # Check before preserving the raw provider response.
            if key and key.encode() in raw_body:
                raise ValueError("description response contains credentials")
            body = json.loads(raw_body)
            receipt["raw_response"] = body
            if body.get("model") != self.config["model"]:
                raise ValueError("description response model differs from requested 8B")
            choices = body.get("choices", [])
            if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
                raise ValueError("description response incomplete")
            text = choices[0]["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("description response is not text")
            description = parse_description(text)
            receipt.update(status="described", description=description,
                           description_sha256=content_sha256(description),
                           usage=body.get("usage", {}))
        except Exception as exc:
            receipt["error_type"] = type(exc).__name__
        receipt["elapsed_s"] = time.perf_counter() - started
        if receipt["status"] != "described":
            raise DescriptionFailure(receipt)
        return receipt


def validate_description_receipt(receipt, images, target_category):
    """Replay the text/image/response binding without making a model request."""
    try:
        config = description_config(receipt["service"]["config"])
        request = receipt["request"]
        expected = description_request(images, target_category, config, request["seed"])
        body = receipt["raw_response"]
        for row in images:
            prefix, data = row["url"].split(",", 1)
            if (prefix != "data:image/jpeg;base64"
                    or hashlib.sha256(base64.b64decode(data, validate=True)).hexdigest()
                    != row["identity"]["sha256"]):
                return False
        return bool(
            receipt["version"] == VERSION and receipt["status"] == "described"
            and receipt["request_sent"] is True
            and receipt["target_category"] == target_category
            and receipt["image_sha256"] == [r["identity"]["sha256"] for r in images]
            and receipt["service"] == Qwen8BDescriptionClient(config).audit_identity()
            and request == expected and receipt["request_sha256"] == content_sha256(expected)
            and body["model"] == config["model"]
            and len(body["choices"]) == 1 and body["choices"][0]["finish_reason"] == "stop"
            and parse_description(body["choices"][0]["message"]["content"]) == receipt["description"]
            and receipt["description_sha256"] == content_sha256(receipt["description"])
            and receipt["reference_answer_visible"] is False
            and receipt["automatic_retry"] is False)
    except (KeyError, IndexError, TypeError, ValueError):
        return False
