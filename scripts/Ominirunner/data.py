"""Two-layer image-cache and question-binding verification; no label reader."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from skill3d.small_image.storage import load_prepared
from .common import BINDINGS_SHA, IDS, MANIFEST_SHA, digest, read, selected_ids, sha
from .answer_contract import specification

PUBLIC = {"benchmark", "source_split", "sample_id", "question", "image_keys",
          "task_type", "options", "input_sha256"}
FIELDS = PUBLIC | {"image_path", "image_sha256", "prepared_path", "artifact_sha256",
                   "method_identity", "publication_receipt_sha256"}


def select(cache_root, sample_ids=IDS):
    root = Path(cache_root)
    if sha(root / "manifest.json") != MANIFEST_SHA or sha(root / "bindings.json") != BINDINGS_SHA:
        raise ValueError("cache_manifest_identity")
    manifest = {r["input_sha256"]: r for r in read(root / "manifest.json")["images"]}
    bindings = {r["sample_id"]: r for r in read(root / "bindings.json")["questions"]}
    catalog = {r["input_sha256"]: r for r in read(root / "prepared_catalog.json")["images"]}
    paths = read(root / "source_paths.json")
    if len(manifest) != 201 or len(bindings) != 501 or len(catalog) != 201:
        raise ValueError("incomplete_cache_catalog")
    result = []
    for sample_id in selected_ids({"sample_ids": sample_ids}):
        public = bindings[sample_id]
        if (set(public) != PUBLIC or len(public["image_keys"]) != 1
                or not isinstance(public["options"], list)):
            raise ValueError("unexpected_public_fields")
        specification(public)
        key = public["image_keys"][0]
        image, cache = manifest[key], catalog[key]
        if image["image_sha256"] != cache["image_sha256"]:
            raise ValueError("catalog_image_binding")
        result.append({
            **public, "image_path": paths[key], "image_sha256": image["image_sha256"],
            **{k: cache[k] for k in ("prepared_path", "artifact_sha256", "method_identity",
                                   "publication_receipt_sha256")}})
    return result


def load(item, *, prepared=True):
    if set(item) != FIELDS or item["benchmark"] != "Omni3D-Bench":
        raise ValueError("job_input_fields")
    if sha(item["image_path"]) != item["image_sha256"]:
        raise ValueError("original_image_changed")
    raw = Path(item["image_path"]).read_bytes()
    with Image.open(BytesIO(raw)) as image:
        if image.format != "JPEG":
            raise ValueError("original_jpeg_required")
        frame = np.array(image.convert("RGB"))
    if not prepared:
        return raw, frame, None
    cache = load_prepared(item["prepared_path"])
    # The cached union question is validated by storage, then withheld. Current
    # question identity is the separately frozen whitelist binding.
    if (cache.artifact_sha256 != item["artifact_sha256"]
            or cache.problem["input_sha256"] != item["image_keys"][0]
            or len(cache.frames) != 1 or not np.array_equal(cache.frames[0], frame)):
        raise ValueError("image_cache_binding_changed")
    if cache.provenance.get("method_identity") not in (None, item["method_identity"]):
        raise ValueError("cache_method_changed")
    geometry = cache.geometries[0]
    if (np.asarray(geometry["depth"]).shape != frame.shape[:2]
            or np.asarray(geometry["valid_mask"]).shape != frame.shape[:2]
            or np.asarray(geometry["points"]).shape != (*frame.shape[:2], 3)):
        raise ValueError("geometry_grid_changed")
    return raw, frame, cache


def problem(item):
    return SimpleNamespace(benchmark=item["benchmark"], question=item["question"],
                           input_sha256=item["input_sha256"], task_type=item["task_type"],
                           options=item["options"])


def pair_identity(item):
    return digest({k: item[k] for k in sorted(PUBLIC | {"image_sha256"})})
