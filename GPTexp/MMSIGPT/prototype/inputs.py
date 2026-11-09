"""Public-only manifest extraction, original-byte validation and label-free sampling."""
from __future__ import annotations

from collections import defaultdict
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re

from ...traces import digest

PUBLIC_FIELDS = frozenset({
    "benchmark", "sample_id", "source_split", "question", "options",
    "image_paths", "task_type", "input_sha256",
})


def public_problem(value):
    if not isinstance(value, dict) or set(value) != PUBLIC_FIELDS:
        raise ValueError("public_problem_fields_changed")
    if value["benchmark"] != "mmsi" or value["source_split"] != "test":
        raise ValueError("requires_mmsi_test_public_input")
    for field in ("sample_id", "question", "task_type"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError("invalid_public_text")
    options, paths = value["options"], value["image_paths"]
    if (not isinstance(options, list) or not 2 <= len(options) <= 26
            or any(not isinstance(o, str) or not o.strip() for o in options)):
        raise ValueError("invalid_ordered_options")
    # Duplicate option texts are retained; their letters still have distinct identities.
    if (not isinstance(paths, list) or not 1 <= len(paths) <= 6
            or any(not isinstance(p, str) or not Path(p).is_absolute() for p in paths)):
        raise ValueError("requires_original_absolute_image_paths")
    if not isinstance(value["input_sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", value["input_sha256"]):
        raise ValueError("invalid_input_sha256")
    return json.loads(json.dumps(value))


def original_input_sha256(problem, image_hashes):
    # Exact frozen SmallImageProblem contract, including absolute paths and bytes.
    identity = {
        "schema": "skill3d.small_image.input.v1",
        "problem": {key: problem[key] for key in sorted(PUBLIC_FIELDS - {"input_sha256"})},
        "image_sha256": image_hashes,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_images(problem, expected=None):
    from PIL import Image

    images = []
    for index, path in enumerate(problem["image_paths"]):
        raw = Path(path).read_bytes()
        with Image.open(BytesIO(raw)) as image:
            receipt = {
                "view_index": index, "sha256": hashlib.sha256(raw).hexdigest(),
                "width": image.width, "height": image.height,
                "mime_type": Image.MIME.get(image.format),
            }
            image.verify()
        if receipt["mime_type"] not in ("image/jpeg", "image/png", "image/webp"):
            raise ValueError("unsupported_original_image_format")
        images.append((receipt, raw))
    receipts = [row[0] for row in images]
    if original_input_sha256(problem, [r["sha256"] for r in receipts]) != problem["input_sha256"]:
        raise ValueError("original_input_bytes_or_identity_changed")
    if expected is not None and receipts != expected:
        raise ValueError("original_image_receipts_changed")
    return images


def extract_public(source):
    """Read public jobs or pairs, discarding Harness metadata without loading it."""
    candidates = []
    if isinstance(source, dict) and isinstance(source.get("jobs"), list):
        candidates = source["jobs"]
    elif isinstance(source, dict) and isinstance(source.get("pairs"), list):
        candidates = [arm for pair in source["pairs"] if pair.get("dataset") == "mmsi"
                      for arm in pair["arms"]]
    elif isinstance(source, dict) and source.get("schema") == "mmsi-direct-public/1":
        candidates = source["items"]
    else:
        raise ValueError("unsupported_public_manifest")
    rows = {}
    for spec in candidates:
        value = spec.get("problem")
        if value is None or value.get("benchmark") != "mmsi":
            continue
        problem = public_problem(value)
        sample = problem["sample_id"]
        images = spec.get("images")
        row = {"problem": problem, "images": images}
        if sample in rows:
            if rows[sample]["problem"] != problem:
                raise ValueError("duplicate_sample_public_input_disagrees")
            if images is not None and rows[sample]["images"] is not None and images != rows[sample]["images"]:
                raise ValueError("duplicate_sample_image_binding_disagrees")
            if images is not None:
                rows[sample]["images"] = images
        else:
            rows[sample] = row
    if not rows:
        raise ValueError("no_mmsi_public_questions")
    return list(rows.values())


def select(rows, *, sample_ids=None, excluded=(), per_type=1, all_samples=False, seed=137):
    excluded = set(excluded)
    available = {r["problem"]["sample_id"]: r for r in rows
                 if r["problem"]["sample_id"] not in excluded}
    if sample_ids is not None:
        if not sample_ids or len(set(sample_ids)) != len(sample_ids) or set(sample_ids) - available.keys():
            raise ValueError("invalid_or_excluded_sample_ids")
        return [available[sample] for sample in sample_ids]
    if all_samples:
        return list(available.values())
    if type(per_type) is not int or per_type < 1:
        raise ValueError("invalid_per_type_count")
    groups = defaultdict(list)
    for row in available.values():
        groups[row["problem"]["task_type"]].append(row)
    if len(groups) != 6 or any(len(group) < per_type for group in groups.values()):
        raise ValueError("requires_six_complete_task_strata")
    ordered = {
        task: sorted(group, key=lambda r: digest({
            "seed": seed, "sample_id": r["problem"]["sample_id"],
            "input_sha256": r["problem"]["input_sha256"]}))[:per_type]
        for task, group in sorted(groups.items())
    }
    return [group[index] for index in range(per_type) for group in ordered.values()]


def read_public_source(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    source = json.loads(raw)
    return extract_public(source), hashlib.sha256(raw).hexdigest()
