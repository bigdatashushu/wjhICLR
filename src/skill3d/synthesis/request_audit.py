"""Hash actual model inputs without persisting prompts, pixels or credentials."""

from __future__ import annotations

import base64
import hashlib
import json

REQUEST_AUDIT_VERSION = "model-request-audit/1.0"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload_bytes(payload: dict) -> bytes:
    """The exact UTF-8 JSON encoding used by BailianClient."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


def describe_request(payload: dict, *, model_identity: dict,
                     wire_encoding_known: bool = True) -> dict:
    messages = payload.get("messages") or []
    text_parts = []
    images = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for part_index, part in enumerate(parts or []):
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = str(part.get("text") or "")
                text_parts.append({
                    "message_index": message_index, "part_index": part_index,
                    "role": message.get("role"), "text_sha256": _sha(text.encode()),
                })
            elif part.get("type") == "image_url":
                value = part.get("image_url")
                url = value.get("url", "") if isinstance(value, dict) else str(value or "")
                image = {
                    "message_index": message_index, "part_index": part_index,
                    "url_sha256": _sha(url.encode()), "jpeg_sha256": None,
                }
                if url.startswith("data:image/") and ";base64," in url:
                    try:
                        image["jpeg_sha256"] = _sha(base64.b64decode(
                            url.split(",", 1)[1], validate=True))
                    except (ValueError, TypeError):
                        image["decode_status"] = "invalid_base64"
                images.append(image)
    return {
        "schema_version": REQUEST_AUDIT_VERSION,
        "model_identity": model_identity,
        "wire_payload_sha256": _sha(payload_bytes(payload)) if wire_encoding_known else None,
        "messages_sha256": _sha(payload_bytes({"messages": messages})),
        "text_parts": text_parts, "images_in_order": images,
        "parameters": {key: payload.get(key) for key in (
            "model", "seed", "temperature", "top_p", "max_tokens",
            "enable_thinking", "response_format",
        )},
        "parameter_status": "effective_payload" if wire_encoding_known else "client_arguments",
    }
