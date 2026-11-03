"""Canonical text/image messages, translated without changing image order."""

from __future__ import annotations

import base64
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ProtocolError

IMAGE_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp",
}


def text_part(text: str) -> dict:
    if not isinstance(text, str):
        raise ProtocolError("Text must be a string.")
    return {"type": "text", "text": text}


def image_url_part(url: str, *, detail: str | None = None) -> dict:
    """An HTTPS URL or base64 data URL; does not download remote images."""
    if not isinstance(url, str):
        raise ProtocolError("Image URL must be a string.")
    if url.startswith("data:"):
        _decode_data_url(url)
    else:
        try:
            parsed = urlsplit(url)
            parsed.port
        except ValueError:
            raise ProtocolError("Invalid image URL.") from None
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ProtocolError("Image must use HTTPS or an image base64 data URL.")
        if any(c.isspace() or ord(c) < 32 for c in url):
            raise ProtocolError("Invalid image URL.")
    image = {"url": url}
    if detail is not None:
        if detail not in ("auto", "low", "high"):
            raise ProtocolError("Image detail must be auto, low or high.")
        image["detail"] = detail
    return {"type": "image_url", "image_url": image}


def image_file_part(path: str | Path, *, detail: str | None = None) -> dict:
    """Read a local image as base64. The path is never sent to the provider."""
    path = Path(path)
    media_type = IMAGE_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ProtocolError("Supported images: JPEG, PNG, GIF and WebP.")
    try:
        data = path.read_bytes()
    except OSError:
        raise ProtocolError("Cannot read image file.") from None
    if not data:
        raise ProtocolError("Image file is empty.")
    return image_url_part(
        f"data:{media_type};base64," + base64.b64encode(data).decode("ascii"), detail=detail
    )


def _decode_data_url(url: str) -> tuple[str, str]:
    header, separator, data = url.partition(",")
    media_type = header.removeprefix("data:").removesuffix(";base64")
    if not separator or not header.endswith(";base64") or media_type not in IMAGE_TYPES.values():
        raise ProtocolError("Unsupported image data URL.")
    try:
        decoded = base64.b64decode(data, validate=True)
    except ValueError:
        raise ProtocolError("Invalid image base64 data.") from None
    if not decoded:
        raise ProtocolError("Image data is empty.")
    return media_type, data


def convert_messages(messages: list[dict], protocol: str) -> tuple[list[dict], str]:
    """Convenience path: system/user/assistant with text and user images.

    Native tool histories and other provider blocks use client.send(payload).
    """
    if not isinstance(messages, list) or not messages:
        raise ProtocolError("Provide a nonempty message list.")
    converted, systems = [], []
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content"}:
            raise ProtocolError("Use canonical role/content messages, or send a native payload.")
        role, content = message.get("role"), message.get("content")
        if role not in ("system", "user", "assistant"):
            raise ProtocolError("Convenience messages support system, user and assistant roles.")
        parts = [text_part(content)] if isinstance(content, str) else content
        if not isinstance(parts, list) or not parts:
            raise ProtocolError("Message content must be text or a nonempty content-part list.")
        output = []
        for part in parts:
            if not isinstance(part, dict):
                raise ProtocolError("Invalid content part.")
            if part.get("type") == "text" and set(part) == {"type", "text"}:
                text_part(part["text"])
                if role == "system":
                    systems.append(part["text"])
                elif protocol == "responses":
                    output.append({"type": "output_text" if role == "assistant" else "input_text", "text": part["text"]})
                else:
                    output.append(dict(part))
            elif part.get("type") == "image_url" and set(part) == {"type", "image_url"}:
                if role != "user":
                    raise ProtocolError("Images in convenience messages require user role.")
                image = part["image_url"]
                if not isinstance(image, dict) or set(image) - {"url", "detail"} or "url" not in image:
                    raise ProtocolError("Invalid image_url part.")
                validated = image_url_part(image["url"], detail=image.get("detail"))["image_url"]
                if protocol == "responses":
                    output.append({"type": "input_image", "image_url": validated["url"],
                                   **({"detail": validated["detail"]} if "detail" in validated else {})})
                elif protocol == "messages":
                    if "detail" in validated:
                        raise ProtocolError("Anthropic images do not accept OpenAI image detail.")
                    url = validated["url"]
                    if url.startswith("data:"):
                        media_type, data = _decode_data_url(url)
                        source = {"type": "base64", "media_type": media_type, "data": data}
                    else:
                        source = {"type": "url", "url": url}
                    output.append({"type": "image", "source": source})
                else:
                    output.append({"type": "image_url", "image_url": validated})
            else:
                raise ProtocolError("Unsupported content part; use send for native provider blocks.")
        if role != "system":
            converted.append({"role": role, "content": output})
    if not converted:
        raise ProtocolError("Include at least one user or assistant message.")
    return converted, "\n\n".join(systems)
