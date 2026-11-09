"""Target-blind mask observation followed by text-only category comparison."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import re
import time
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from skill3d.infra.m5_identity import content_sha256
from skill3d.synthesis.request_context import request_phase

VISUAL_REVIEW_VERSION = "m5-mask-observation-qwen38-v2-observation-identity"
RELATION_REVIEW_VERSION = "m5-mask-relation-qwen38-v4-examples"
VISUAL_ADMISSION_POLICY = "visual-quality-and-relation"
JPEG_QUALITY = 95
RELATION_MAX_TOKENS = 2048
VISUAL_PROMPT = """The three images show one candidate region:
1. the full scene with an orange rectangle,
2. an unmasked crop,
3. the same crop with a turquoise segmentation mask.

Identify the dominant physical object actually covered by the turquoise mask.
You are not given any requested target category. Do not guess the intended
label. If visual evidence is weak, use "unknown" and visibility="insufficient".
Use the most specific common category visually supported by the images,
including material or form factor when visible. Avoid an umbrella term when a
more specific subtype can be distinguished.

Return exactly one JSON object:
{
  "observed_object": "short common noun or unknown",
  "alternate_object": "short common noun or none",
  "visibility": "clear|partial|insufficient",
  "mask_purity": "clean|mixed|background",
  "mask_completeness": "whole|major_part|fragment|unknown"
}
No prose and no Markdown."""
RELATION_PROMPT = """Compare each already-frozen observed object name with its
target category. Do not reinterpret an image and do not replace the observed
name.

relation:
- match: the observed object is an instance or common synonym/subtype of target.
- mismatch: it is a different physical category.
- uncertain: the observed name is too generic or ambiguous to decide.

Material distinctions matter: a plastic bag is not a paper bag.
Use uncertain only for names such as unknown, object, container, furniture or
device that could genuinely include the target. A concrete category such as
box, cabinet, monitor or plastic bag is mismatch when the target is a different
concrete category.
Examples:
- radiator -> heater: match
- computer monitor -> computer tower: mismatch
- cardboard box -> paper bag: mismatch
- box -> paper bag: mismatch
- cabinet -> refrigerator: mismatch
Return exactly one JSON object:
{"results":[{"id":"candidate-id","relation":"match|mismatch|uncertain"}]}
Every input ID must appear exactly once. No prose or Markdown.

Input pairs:
"""


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    temporary.replace(path)


def _strip_fence(raw: str) -> str:
    text = str(raw).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    return re.sub(r"\s*```$", "", text)


def validate_visual_response(raw: str) -> dict[str, str]:
    row = json.loads(_strip_fence(raw))
    expected = {
        "observed_object", "alternate_object", "visibility",
        "mask_purity", "mask_completeness",
    }
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError("visual review must contain exactly the contracted fields")
    if not all(isinstance(row[name], str) and row[name].strip() for name in expected):
        raise ValueError("visual review fields must be nonempty strings")
    if row["visibility"] not in {"clear", "partial", "insufficient"}:
        raise ValueError("invalid visibility")
    if row["mask_purity"] not in {"clean", "mixed", "background"}:
        raise ValueError("invalid mask_purity")
    if row["mask_completeness"] not in {
            "whole", "major_part", "fragment", "unknown"}:
        raise ValueError("invalid mask_completeness")
    return {key: row[key].strip().lower() for key in expected}


def validate_relation_response(raw: str, expected_ids) -> dict[str, str]:
    data = json.loads(_strip_fence(raw))
    expected = {str(value) for value in expected_ids}
    if not isinstance(data, dict) or set(data) != {"results"}:
        raise ValueError("relation review must contain exactly results")
    rows = data["results"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("relation result count mismatch")
    result = {}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"id", "relation"}
                or str(row["id"]) not in expected
                or row["relation"] not in {"match", "mismatch", "uncertain"}
                or str(row["id"]) in result):
            raise ValueError("invalid relation result row")
        result[str(row["id"])] = row["relation"]
    if set(result) != expected:
        raise ValueError("relation IDs are missing or unexpected")
    return result


def visual_quality_passes(observation) -> bool:
    return bool(
        isinstance(observation, dict)
        and isinstance(observation.get("observed_object"), str)
        and observation["observed_object"]
        and observation["observed_object"] not in {"other", "unknown"}
        and observation.get("visibility") in {"clear", "partial"}
        and observation.get("mask_purity") == "clean"
        and observation.get("mask_completeness") in {"whole", "major_part"})


def _crop_bounds(box, width: int, height: int, margin: float = 0.20):
    x0, y0, x1, y1 = map(float, box)
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0, int(np.floor(x0 - dx))),
            max(0, int(np.floor(y0 - dy))),
            min(width, int(np.ceil(x1 + dx))),
            min(height, int(np.ceil(y1 + dy))))


def _jpeg(array: np.ndarray) -> tuple[str, dict]:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB")
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=JPEG_QUALITY)
    body = stream.getvalue()
    return ("data:image/jpeg;base64," + base64.b64encode(body).decode("ascii"),
            {"sha256": hashlib.sha256(body).hexdigest(), "size": len(body),
             "width": image.width, "height": image.height,
             "jpeg_quality": JPEG_QUALITY})


def candidate_images(frame, box, mask):
    rgb = np.asarray(frame, dtype=np.uint8)
    raw_mask = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or raw_mask.shape != rgb.shape[:2]:
        raise ValueError("candidate frame/mask shape mismatch")
    context = Image.fromarray(rgb).convert("RGB")
    ImageDraw.Draw(context).rectangle(tuple(map(float, box)),
                                      outline=(238, 143, 44), width=5)
    bounds = _crop_bounds(box, rgb.shape[1], rgb.shape[0])
    x0, y0, x1, y1 = bounds
    if x1 <= x0 or y1 <= y0:
        raise ValueError("candidate crop is empty")
    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = raw_mask[y0:y1, x0:x1]
    overlay = crop.copy()
    overlay[crop_mask] = (
        overlay[crop_mask].astype(np.float32) * 0.42
        + np.array([20, 220, 205], dtype=np.float32) * 0.58
    ).astype(np.uint8)
    result = []
    for name, image in (
            ("context", np.asarray(context)), ("crop", crop), ("mask_overlay", overlay)):
        url, identity = _jpeg(image)
        result.append({"kind": name, "url": url, "identity": identity})
    return result


def _usage(client) -> dict:
    return dict(getattr(client, "last_usage", {}) or {})


def _response_identity(client) -> dict:
    return dict(getattr(client, "last_response_identity", {}) or {})


def observe_candidate(candidate, frame, mask, client, *, seed, min_pixels,
                      max_pixels, out_dir=None):
    if hasattr(client, "observe_candidate"):
        return client.observe_candidate(candidate, frame, mask, seed=seed,
            min_pixels=min_pixels, max_pixels=max_pixels, out_dir=out_dir)
    candidate_id = str(candidate["candidate_id"])
    receipt = {
        "version": VISUAL_REVIEW_VERSION,
        "provider": "bailian",
        "policy": VISUAL_ADMISSION_POLICY,
        "candidate_id": candidate_id,
        "status": "unavailable",
    }
    started = time.perf_counter()
    try:
        images = candidate_images(frame, candidate["bbox"], mask)
        content = [{"type": "text", "text": VISUAL_PROMPT}]
        for image in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": image["url"]},
                "min_pixels": int(min_pixels),
                "max_pixels": int(max_pixels),
            })
        request_identity = {
            "version": VISUAL_REVIEW_VERSION,
            "candidate_id": candidate_id,
            "prompt": VISUAL_PROMPT,
            "images": [{"kind": row["kind"], **row["identity"]} for row in images],
            "model": getattr(client, "audit_identity", lambda **_: {
                "model": str(getattr(client, "model", ""))})(seed=seed),
            "seed": seed, "temperature": 0.0, "max_tokens": 256,
            "min_pixels": int(min_pixels), "max_pixels": int(max_pixels),
        }
        receipt["request_sha256"] = content_sha256(request_identity)
        receipt["request_identity"] = request_identity
        receipt["request_sent"] = True
        with request_phase("m5_mask_visual"):
            raw = client.chat(
                [{"role": "user", "content": content}],
                max_tokens=256, seed=seed, temperature=0.0,
                response_format={"type": "json_object"}, enable_thinking=False)
        receipt["raw_response"] = raw
        receipt["observation"] = validate_visual_response(raw)
        receipt["observation_sha256"] = content_sha256(receipt["observation"])
        receipt["model_identity_sha256"] = content_sha256(request_identity["model"])
        receipt["usage"] = _usage(client)
        receipt["response_identity"] = _response_identity(client)
        receipt["status"] = "observed"
    except Exception as exc:  # transport and schema errors are both fail-closed
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["elapsed_s"] = time.perf_counter() - started
    if out_dir is not None:
        _write(Path(out_dir) / "visual" / f"{candidate_id}.json", receipt)
    return receipt


def compare_categories(rows, client, *, seed, batch_size=32, out_dir=None):
    """Return candidate -> relation receipt; observations are already frozen."""
    result = {}
    actual_requests = 0
    batches = [rows[i:i + int(batch_size)] for i in range(0, len(rows), int(batch_size))]
    for batch_index, batch in enumerate(batches):
        pairs = [{
            "id": str(row["candidate_id"]),
            "observed_object": row["observation"]["observed_object"],
            "target_category": str(row["category"]),
        } for row in batch]
        request_identity = {
            "version": RELATION_REVIEW_VERSION,
            "prompt": RELATION_PROMPT,
            "pairs": pairs,
            "model": getattr(client, "audit_identity", lambda **_: {
                "model": str(getattr(client, "model", ""))})(seed=seed),
            "seed": seed, "temperature": 0.0,
            "max_tokens": RELATION_MAX_TOKENS,
        }
        receipt = {
            "version": RELATION_REVIEW_VERSION,
            "batch_index": batch_index,
            "candidate_ids": [row["id"] for row in pairs],
            "request_sha256": content_sha256(request_identity),
            "request_identity": request_identity,
            "status": "unavailable",
        }
        started = time.perf_counter()
        try:
            prompt = RELATION_PROMPT + json.dumps(pairs, ensure_ascii=False)
            actual_requests += 1
            with request_phase("m5_mask_relation"):
                raw = client.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=request_identity["max_tokens"], seed=seed, temperature=0.0,
                    response_format={"type": "json_object"}, enable_thinking=False)
            relations = validate_relation_response(
                raw, [row["candidate_id"] for row in batch])
            receipt.update(
                status="reviewed", raw_response=raw, relations=relations,
                usage=_usage(client), response_identity=_response_identity(client))
            for row in batch:
                cid = str(row["candidate_id"])
                result[cid] = {
                    "status": "reviewed", "relation": relations[cid],
                    "request_sha256": receipt["request_sha256"],
                    "batch_index": batch_index,
                }
        except Exception as exc:  # the whole malformed batch is unavailable
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            for row in batch:
                result[str(row["candidate_id"])] = {
                    "status": "unavailable", "relation": "uncertain",
                    "error": receipt["error"], "batch_index": batch_index,
                }
        receipt["elapsed_s"] = time.perf_counter() - started
        if out_dir is not None:
            _write(Path(out_dir) / "relation" / f"{batch_index:04d}.json", receipt)
    return result, {"actual_requests": actual_requests, "n_batches": len(batches)}
