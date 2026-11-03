"""Public tinybench bindings and verified single-image cache composition."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image

from skill3d.small_image.storage import load_prepared
from .common import digest, read, sha

PUBLIC = frozenset({"benchmark", "source_split", "sample_id", "question",
                    "image_keys", "task_type", "options", "input_sha256"})
IMAGE_PUBLIC = (PUBLIC - {"image_keys"}) | {"image_paths"}
PRODUCTION_PINS = {
    "manifest.json": "bf914020aad37d4293527141f34ffbf5db43dda36d082f88d4b8435be2d77886",
    "bindings.json": "43fa5d69e2f6aaf657fdce806a5e7ca1bb11483b1fafbf50b38d8f631d593632",
    "prepared_catalog.json": "045d5798ae281a93e54853576f415e41e23c1f0b0bcfde410cfefa58f1921ee4",
}
FORBIDDEN_CAPABILITIES = ("object_mask_geometry", "metric_geometry",
                          "world_geometry", "cross_view_identity")


def option_parts(question):
    """Keep the source text intact; derive labels and the stem for validation."""
    marks = list(re.finditer(r"(?<!\S)([A-D])\.[ \t]+", question))
    labels = [mark[1] for mark in marks]
    if not 2 <= len(labels) <= 4 or labels != list("ABCD"[:len(labels)]):
        raise ValueError("embedded_options_required")
    options = [question[mark.end():marks[i + 1].start() if i + 1 < len(marks)
                        else len(question)].strip() for i, mark in enumerate(marks)]
    if not all(options) or not question[:marks[0].start()].strip():
        raise ValueError("empty_question_or_option")
    return question[:marks[0].start()].strip(), labels, options


def public_record(row, *, image=False):
    expected = IMAGE_PUBLIC if image else PUBLIC
    path_field = "image_paths" if image else "image_keys"
    if (set(row) != expected or row["benchmark"].lower() != "mindcube"
            or not isinstance(row["sample_id"], str)
            or not isinstance(row["task_type"], str)
            or not isinstance(row["question"], str) or row["options"] != []
            or not isinstance(row[path_field], list)
            or not re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"])):
        raise ValueError("invalid_public_binding")
    return dict(row)


def indexed(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError("duplicate_catalog_identity")
        result[row[key]] = row
    return result


def select(cache_root, sample_ids=None, *, pins=None):
    """No label file is opened. Production defaults bind the audited snapshot."""
    root = Path(cache_root).resolve()
    pins = PRODUCTION_PINS if pins is None else pins
    if set(pins) != set(PRODUCTION_PINS) or any(sha(root / n) != v for n, v in pins.items()):
        raise ValueError("cache_catalog_identity_changed")
    manifest, bindings, catalog = [read(root / name) for name in
                                  ("manifest.json", "bindings.json", "prepared_catalog.json")]
    images = indexed(manifest["images"], "input_sha256")
    prepared = indexed(catalog["images"], "input_sha256")
    if catalog["question_bindings"] != bindings["questions"]:
        raise ValueError("catalog_question_bindings_changed")
    paths = read(root / "source_paths.json")
    tiny = [public_record(row) for row in bindings["questions"]
            if row["source_split"] == "tinybench"]
    by_id = indexed(tiny, "sample_id")
    if pins == PRODUCTION_PINS and (len(tiny) != 1050
            or len({key for row in tiny for key in row["image_keys"]}) != 428):
        raise ValueError("incomplete_tinybench")
    if sample_ids is None:
        chosen = tiny
    else:
        if (not isinstance(sample_ids, list) or not sample_ids
                or len(set(sample_ids)) != len(sample_ids) or set(sample_ids) - set(by_id)):
            raise ValueError("invalid_tinybench_selection")
        chosen = [by_id[key] for key in sample_ids]
    result = []
    for row in chosen:
        stem, labels, options = option_parts(row["question"])
        if not 2 <= len(row["image_keys"]) <= 4:
            raise ValueError("ordered_two_to_four_views_required")
        views = []
        for key in row["image_keys"]:
            image, cache = images[key], prepared[key]
            image_public = public_record({k: image[k] for k in IMAGE_PUBLIC}, image=True)
            if len(image_public["image_paths"]) != 1 or image_public["input_sha256"] != key:
                raise ValueError("single_image_manifest_required")
            if image["image_sha256"] != cache["image_sha256"]:
                raise ValueError("catalog_image_binding_changed")
            image_path = Path(paths[key])
            if not image_path.is_absolute():
                image_path = root / image_path
            cache_path = Path(cache["prepared_directory"])
            if not cache_path.is_absolute():
                cache_path = root / cache_path
            views.append({
                "image_key": key, "image_path": str(image_path),
                "image_sha256": cache["image_sha256"],
                "prepared_directory": str(cache_path),
                "artifact_sha256": cache["artifact_sha256"],
                "method_identity": cache["method_identity"], "image_public": image_public,
            })
        result.append({**row, "stem": stem, "labels": labels,
                       "option_texts": options, "views": views})
    return result


@dataclass
class Loaded:
    rgb: tuple[bytes, ...]
    mime: tuple[str, ...]
    frames: tuple[np.ndarray, ...]
    candidates: tuple[dict, ...]
    geometries: dict
    provenance: dict


def load(item, *, prepared=True):
    public_record({k: item[k] for k in PUBLIC})
    stem, labels, options = option_parts(item["question"])
    if (item["source_split"] != "tinybench" or item["stem"] != stem
            or item["labels"] != labels or item["option_texts"] != options
            or [v["image_key"] for v in item["views"]] != item["image_keys"]):
        raise ValueError("episode_binding_changed")
    frames, raws, mimes, candidates, geometries, receipts = [], [], [], [], {}, []
    for index, view in enumerate(item["views"]):
        raw = Path(view["image_path"]).read_bytes()
        if digest_bytes(raw) != view["image_sha256"]:
            raise ValueError("original_image_changed")
        with Image.open(BytesIO(raw)) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("unsupported_original_image_format")
            mime = "image/jpeg" if image.format == "JPEG" else "image/png"
            frame = np.asarray(image.convert("RGB")).copy()
        frame = readonly(frame)
        frames.append(frame)
        raws.append(raw)
        mimes.append(mime)
        receipt = {k: view[k] for k in
                   ("image_key", "image_sha256", "artifact_sha256", "method_identity")}
        receipts.append({"view_index": index, **receipt})
        if not prepared:
            continue
        cache = load_prepared(view["prepared_directory"], problem=view["image_public"])
        if (cache.artifact_sha256 != view["artifact_sha256"] or len(cache.frames) != 1
                or not np.array_equal(cache.frames[0], frame)
                or cache.provenance.get("method_identity") not in (None, view["method_identity"])
                or cache.provenance.get("labels_used") is not False
                or cache.errors or any(cache.capabilities.get(k) for k in FORBIDDEN_CAPABILITIES)):
            raise ValueError("prepared_view_identity_or_capabilities_changed")
        # Older publications keep method configuration inside their content
        # provenance rather than a top-level method_identity. The pinned catalog
        # binds the method to the exact loader-verified artifact in either form.
        receipts[-1]["method_binding_source"] = (
            "artifact_provenance_and_pinned_catalog"
            if cache.provenance.get("method_identity") is not None else "pinned_publication_catalog")
        if set(cache.geometries) != {0}:
            raise ValueError("single_view_relative_geometry_required")
        geometry = cache.geometries[0]
        h, w = frame.shape[:2]
        if (not geometry["relative_only"] or geometry["depth"].shape != (h, w)
                or geometry["valid_mask"].shape != (h, w)
                or geometry["points"] is None or geometry["points"].shape != (h, w, 3)):
            raise ValueError("geometry_grid_changed")
        geometries[index] = {k: readonly(v) if isinstance(v, np.ndarray) else v
                             for k, v in geometry.items()}
        # Prepared union questions, masks and review payloads stay host-only.
        for candidate in cache.candidates:
            if candidate["view_index"] != 0:
                raise ValueError("candidate_view_changed")
            candidates.append({
                "candidate_id": f"v{index}:c{len(candidates)}", "view_index": index,
                **{k: candidate[k] for k in
                   ("category", "box_xyxy", "category_confirmed", "review_status")},
                "mask_quality_assessed": False, "geometry_usable": False,
            })
    return Loaded(tuple(raws), tuple(mimes), tuple(frames), tuple(candidates), geometries,
                  {"schema": "mindcube-ordered-view-composition/1",
                   "input_sha256": item["input_sha256"], "views": receipts})


def digest_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def readonly(array):
    return np.frombuffer(np.ascontiguousarray(array).tobytes(),
                         dtype=array.dtype).reshape(array.shape)


def problem(item):
    return SimpleNamespace(benchmark=item["benchmark"], question=item["stem"],
                           option_texts=tuple(item["option_texts"]),
                           input_sha256=item["input_sha256"], task_type=item["task_type"])


def pair_identity(item):
    return digest({"public": {k: item[k] for k in PUBLIC},
                   "views": [{k: v[k] for k in
                              ("image_key", "image_sha256", "artifact_sha256", "method_identity")}
                             for v in item["views"]]})
