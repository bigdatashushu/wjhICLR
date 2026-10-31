"""Question-conditioned still-image preparation, independent of VSI/M4.

Injected callables have these contracts:
    detector(rgb, categories) -> iterable of pixel-box mappings
    segmenter(rgb, boxes_xyxy) -> one binary HxW mask per box
    reviewer(candidate, rgb, mask) -> native InternBinaryClient receipt
    depth_estimator(rgb) -> {depth, valid_mask, points?, intrinsics?}
    category_parser(question) -> bounded sequence of category names

Geometry is predicted, relative, and local to each view. Category review alone
never authorizes object-mask geometry, metric distances, or world identities.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from PIL import Image

METHOD_VERSION = "small-image-precompute/1"
_CATEGORY = re.compile(r"[a-z]+(?:[ -][a-z]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PUBLIC_FIELDS = ("benchmark", "sample_id", "source_split", "question",
                  "options", "image_paths", "task_type", "input_sha256")


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _content_record(value):
    """Typed JSON framing keeps array bytes and container boundaries unambiguous."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("prepared evidence cannot contain object arrays")
        return ["ndarray", list(value.shape), np.lib.format.dtype_to_descr(value.dtype),
                hashlib.sha256(value.tobytes(order="C")).hexdigest()]
    if isinstance(value, np.generic):
        return _content_record(value.item())
    if isinstance(value, Mapping):
        rows = [[_content_record(key), _content_record(item)]
                for key, item in value.items()]
        rows.sort(key=lambda row: json.dumps(
            row[0], ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        return ["mapping", rows]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_content_record(item) for item in value]]
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        # Failed native receipts can contain nonfinite scalars; frame them as text.
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", len(value), hashlib.sha256(value).hexdigest()]
    if isinstance(value, Path):
        return ["path", str(value)]
    raise TypeError(f"unsupported prepared content type: {type(value).__name__}")


def _readonly(array) -> np.ndarray:
    a = np.ascontiguousarray(array)
    root = a
    while isinstance(root, np.ndarray):
        root = root.base
    if isinstance(root, bytes):
        return a
    # Immutable backing bytes also prevent setflags(write=True).
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def _snapshot(value):
    if isinstance(value, np.ndarray):
        return _readonly(value)
    if isinstance(value, Mapping):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(v) for v in value)
    return copy.deepcopy(value)


def _binary_mask(value, shape) -> np.ndarray:
    a = np.asarray(value)
    if (a.shape != shape or a.dtype.kind not in "buif"
            or not np.isfinite(a).all() or not np.isin(a, (0, 1)).all()):
        raise ValueError("mask must be finite binary data on the RGB grid")
    return a.astype(bool)


class GeometryUnavailable(ValueError):
    """Missing or unauthorized geometry, never a numeric zero."""


@dataclass(frozen=True)
class PreparedSmallImage:
    problem: Mapping[str, Any]
    frames: tuple[np.ndarray, ...]
    candidates: tuple[dict, ...]
    depths: Mapping[int, np.ndarray]
    points: Mapping[int, np.ndarray]
    intrinsics: Mapping[int, np.ndarray]
    geometry_masks: Mapping[int, np.ndarray]
    geometries: dict[int, dict]
    categories: tuple[str, ...]
    provenance: dict[str, Any]
    capabilities: Mapping[str, bool]
    cache_key: str
    normalized_image_paths: tuple[str, ...] = ()
    errors: tuple[dict, ...] = ()
    artifact_sha256: str = ""

    @property
    def has_relative_depth(self) -> bool:
        return bool(self.depths)

    @property
    def has_relative_points(self) -> bool:
        return bool(self.points)

    @property
    def geometry_trusted(self) -> bool:
        return False

    def public_summary(self) -> dict:
        return {
            "cache_key": self.cache_key, "artifact_sha256": self.artifact_sha256,
            "n_views": len(self.frames),
            "categories": list(self.categories), "n_candidates": len(self.candidates),
            "geometry_views": sorted(self.geometries),
            "capabilities": dict(self.capabilities), "relative_only": True,
        }

    def require_candidate_geometry(self, candidate_id: str):
        """Category-only masks cannot support object distance/size summaries."""
        row = next((c for c in self.candidates
                    if c["candidate_id"] == candidate_id), None)
        if row is None:
            raise GeometryUnavailable("unknown candidate")
        if not row["mask_valid"]:
            raise GeometryUnavailable("invalid or missing candidate mask")
        if not row["category_reviewed"] or not row["category_confirmed"]:
            raise GeometryUnavailable("candidate category is not confirmed")
        # This method produces no independent mask-quality evidence.
        raise GeometryUnavailable("mask purity/completeness were not reviewed")


def prepared_content_sha256(prepared) -> str:
    """Hash actual evidence, not just declared identities; also usable by test fakes."""
    return _digest(_content_record({
        "schema": "small-image-prepared-content/1",
        "fields": {
            name: getattr(prepared, name, None)
            for name in PreparedSmallImage.__dataclass_fields__
            if name != "artifact_sha256"
        },
    }))


def validate_prepared_content(prepared) -> None:
    stored = getattr(prepared, "artifact_sha256", "")
    if (not isinstance(stored, str) or not _SHA256.fullmatch(stored)
            or stored != prepared_content_sha256(prepared)):
        raise ValueError("prepared content digest mismatch")


def _prepared(**values) -> PreparedSmallImage:
    values.pop("artifact_sha256", None)
    for key in ("problem", "provenance", "capabilities", "depths", "points",
                "intrinsics", "geometry_masks"):
        values[key] = MappingProxyType(_snapshot(values[key]))
    values["geometries"] = {
        index: {
            "depth": values["depths"][index],
            "points": values["points"].get(index),
            "intrinsics": values["intrinsics"].get(index),
            "valid_mask": values["geometry_masks"][index],
            "relative_only": geometry["relative_only"],
        }
        for index, geometry in values["geometries"].items()
    }
    values["provenance"] = _snapshot(values["provenance"])
    values["frames"] = tuple(_readonly(a) for a in values["frames"])
    values["candidates"] = tuple(_snapshot(c) for c in values["candidates"])
    values["errors"] = tuple(_snapshot(e) for e in values.get("errors", ()))
    result = PreparedSmallImage(**values)
    return replace(result, artifact_sha256=prepared_content_sha256(result))


class SmallImagePrecomputer:
    """Reusable preparation with explicit identities and an optional PNG cache.

    ``depths``/``points``/``intrinsics``/``geometry_masks`` are view-index
    mappings; missing views mean unavailable evidence. ``resize_to`` is (W,H)
    and is applied to RGB *before* every backend, never just to a mask/depth.
    Result caching is bounded and process-local; only complete runs are reused.
    """

    def __init__(
        self, *, detector: Callable | None = None,
        segmenter: Callable | None = None, reviewer: Callable | None = None,
        depth_estimator: Callable | None = None,
        category_parser: Callable | None = None,
        model_identities: Mapping | None = None,
        expected_review_identity_sha256: str = "",
        cache_dir: str | Path | None = None,
        method_identity: Any = METHOD_VERSION,
        resize_to: tuple[int, int] | None = None,
        yes_threshold: float = .77, max_categories: int = 16,
        max_candidates_per_view: int = 64, min_mask_pixels: int = 1,
        max_cache_entries: int = 4,
    ):
        if (type(yes_threshold) not in (int, float)
                or not math.isfinite(yes_threshold) or not .77 <= yes_threshold <= .99):
            raise ValueError("category threshold must be finite and >=0.77, <=0.99")
        for name, value, low, high in (
            ("max_categories", max_categories, 1, 32),
            ("max_candidates_per_view", max_candidates_per_view, 1, 256),
            ("min_mask_pixels", min_mask_pixels, 1, 10**9),
            ("max_cache_entries", max_cache_entries, 0, 64),
        ):
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if resize_to is not None and (
            len(resize_to) != 2 or any(type(n) is not int or n <= 0 for n in resize_to)
        ):
            raise ValueError("resize_to must be positive (width, height)")
        self.detector, self.segmenter = detector, segmenter
        self.reviewer, self.depth_estimator = reviewer, depth_estimator
        self.category_parser = category_parser
        identities = dict(model_identities or {})
        for name, backend in (
            ("detector", detector), ("segmenter", segmenter),
            ("reviewer", reviewer), ("depth_estimator", depth_estimator),
            ("category_parser", category_parser),
        ):
            if backend is None:
                identities[name] = {"enabled": False}
                continue
            identity = identities.get(name, getattr(backend, "identity", None))
            if not identity or callable(identity):
                raise ValueError(f"explicit model identity required for {name}")
            identities[name] = _plain(identity)
        self.model_identities = _plain(identities)
        _digest(self.model_identities)
        review_identity = self.model_identities["reviewer"]
        inferred_pin = (review_identity.get("expected_identity_sha256", "")
                        if isinstance(review_identity, Mapping) else review_identity)
        self.expected_review_identity_sha256 = (
            expected_review_identity_sha256 or inferred_pin
            or getattr(reviewer, "expected_identity_sha256", ""))
        if reviewer is not None and not _SHA256.fullmatch(
                self.expected_review_identity_sha256):
            raise ValueError("reviewer requires a pinned actual health identity SHA256")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.method_identity = _plain(method_identity)
        self.resize_to = tuple(resize_to) if resize_to is not None else None
        self.yes_threshold = float(yes_threshold)
        self.max_categories = max_categories
        self.max_candidates_per_view = max_candidates_per_view
        self.min_mask_pixels = min_mask_pixels
        self.max_cache_entries = max_cache_entries
        self._cache: OrderedDict[str, PreparedSmallImage] = OrderedDict()

    def _categories(self, question, supplied):
        if supplied is None:
            supplied = () if self.category_parser is None else self.category_parser(question)
        if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
            raise ValueError("categories must be a bounded sequence, not free-form text")
        if len(supplied) > self.max_categories:
            raise ValueError("category parser/supplied list exceeds category budget")
        result = []
        for name in supplied:
            if not isinstance(name, str):
                raise ValueError("category names must be strings")
            if len(name) > 80 or not name.isascii() or any(ord(c) < 32 for c in name):
                raise ValueError("category must be a bounded plain category name")
            name = " ".join(name.lower().split())
            if len(name) > 80 or not _CATEGORY.fullmatch(name):
                raise ValueError("category must be a plain lower-case category name")
            if name not in result:
                result.append(name)
        return tuple(result)

    def _decode(self, paths):
        frames, identities, normalized = [], [], []
        for index, path in enumerate(paths):
            raw = Path(path).read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                source_format, source_size = image.format, image.size
                image = image.convert("RGB")
                if self.resize_to is not None:
                    image = image.resize(self.resize_to, Image.Resampling.BICUBIC)
                frame = _readonly(np.asarray(image, dtype=np.uint8))
            h, w = frame.shape[:2]
            sx, sy = w / source_size[0], h / source_size[1]
            pixel_sha = hashlib.sha256(frame.tobytes()).hexdigest()
            identities.append({
                "view_index": index, "source_sha256": hashlib.sha256(raw).hexdigest(),
                "source_format": source_format, "source_size_wh": source_size,
                "rgb_sha256": pixel_sha, "rgb_size_wh": (w, h),
                "source_to_rgb_xy": (sx, sy),
                "source_to_rgb_affine": [[sx, 0, (sx - 1) / 2],
                                         [0, sy, (sy - 1) / 2], [0, 0, 1]],
                "coordinate_convention": "integer_pixel_centers; boxes_use_pixel_edges",
                "coordinate_grid": "rgb_pixels_xy", "exif_transpose": False,
            })
            frames.append(frame)
            if self.cache_dir is not None:
                destination = self.cache_dir / "images" / f"{w}x{h}-{pixel_sha}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    with Image.open(destination) as cached:
                        if (cached.format != "PNG" or cached.mode != "RGB"
                                or cached.size != (w, h)
                                or not np.array_equal(np.asarray(cached), frame)):
                            raise ValueError("normalized PNG cache content mismatch")
                else:
                    with tempfile.NamedTemporaryFile(
                        dir=destination.parent, suffix=".png", delete=False
                    ) as stream:
                        temporary = Path(stream.name)
                    try:
                        Image.fromarray(frame).save(temporary, format="PNG")
                        temporary.replace(destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                normalized.append(str(destination))
        return tuple(frames), identities, tuple(normalized)

    def process(self, problem, categories=None) -> PreparedSmallImage:
        # Do not serialize __dict__/public_dict: only named public inputs enter.
        public = {}
        for key in _PUBLIC_FIELDS:
            value = problem[key] if isinstance(problem, Mapping) else getattr(problem, key)
            public[key] = _plain(value() if callable(value) else value)
        if not isinstance(public["question"], str):
            raise ValueError("question must be text")
        paths = public["image_paths"]
        if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence) or not paths:
            raise ValueError("image_paths must be a nonempty ordered sequence")
        errors = []
        try:
            requested = self._categories(public["question"], categories)
        except Exception as exc:
            if categories is not None:
                raise
            requested = ()
            errors.append({"stage": "category_parser", "error": f"{type(exc).__name__}: {exc}"})
        frames, image_identities, normalized = self._decode(paths)
        config = {
            "version": METHOD_VERSION, "method": self.method_identity,
            "models": self.model_identities,
            "expected_review_identity_sha256": self.expected_review_identity_sha256,
            "yes_threshold": self.yes_threshold, "resize_to": self.resize_to,
            "max_categories": self.max_categories,
            "max_candidates_per_view": self.max_candidates_per_view,
            "min_mask_pixels": self.min_mask_pixels,
        }
        key = _digest({"problem": public, "images": image_identities,
                       "categories": requested, "config": config})
        if key in self._cache and not errors:
            cached = self._cache[key]
            validate_prepared_content(cached)
            self._cache.move_to_end(key)
            return _prepared(**{name: getattr(cached, name)
                               for name in PreparedSmallImage.__dataclass_fields__})
        candidates, depths, points, intrinsics, geometry_masks = [], {}, {}, {}, {}
        geometries = {}
        geometry_sources = {}
        for view_index, frame in enumerate(frames):
            if self.depth_estimator is not None:
                try:
                    geometry = _validate_geometry(self.depth_estimator(frame), frame.shape[:2])
                    depths[view_index] = geometry["depth"]
                    geometry_masks[view_index] = geometry["valid_mask"]
                    for name, destination in (("points", points), ("intrinsics", intrinsics)):
                        if geometry.get(name) is not None:
                            destination[view_index] = geometry[name]
                    geometry_sources[view_index] = geometry["points_source"]
                    geometries[view_index] = {
                        "depth": geometry["depth"], "points": geometry["points"],
                        "intrinsics": geometry["intrinsics"],
                        "valid_mask": geometry["valid_mask"], "relative_only": True,
                    }
                except Exception as exc:
                    errors.append({"stage": "depth", "view_index": view_index,
                                   "error": f"{type(exc).__name__}: {exc}"})
            rows = []
            if requested and self.detector is not None:
                try:
                    # Bound iteration as well as the resulting candidate list.
                    detections = iter(self.detector(frame, requested))
                    for _ in range(self.max_candidates_per_view):
                        detection = next(detections, None)
                        if detection is None:
                            break
                        try:
                            rows.append(_detection(detection, requested, frame.shape[:2]))
                        except (TypeError, ValueError, KeyError) as exc:
                            errors.append({"stage": "detection_row", "view_index": view_index,
                                           "error": str(exc)})
                    if next(detections, None) is not None:
                        errors.append({"stage": "detector", "view_index": view_index,
                                       "error": "candidate budget exceeded; candidates truncated"})
                except Exception as exc:
                    errors.append({"stage": "detector", "view_index": view_index,
                                   "error": f"{type(exc).__name__}: {exc}"})
            masks = [None] * len(rows)
            if rows and self.segmenter is not None:
                try:
                    masks = list(self.segmenter(frame, [r["box_xyxy"] for r in rows]))
                    if len(masks) != len(rows):
                        raise ValueError("segmentation mask count differs from box count")
                except Exception as exc:
                    masks = [None] * len(rows)
                    errors.append({"stage": "segmenter", "view_index": view_index,
                                   "error": f"{type(exc).__name__}: {exc}"})
            for slot, (row, raw_mask) in enumerate(zip(rows, masks)):
                mask, mask_error = None, ""
                try:
                    if raw_mask is None:
                        raise ValueError("segmentation unavailable")
                    mask = _binary_mask(raw_mask, frame.shape[:2])
                    if int(mask.sum()) < self.min_mask_pixels:
                        raise ValueError("candidate mask has insufficient foreground pixels")
                    mask = _readonly(mask)
                except ValueError as exc:
                    mask, mask_error = None, str(exc)
                cid = _digest({"preparation": key, "view_index": view_index, "slot": slot,
                               "detection": row,
                               "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()
                               if mask is not None else None})
                candidate = {
                    **row, "candidate_id": cid, "view_index": view_index,
                    "frame_slot": view_index, "bbox": list(row["box_xyxy"]),
                    "mask": mask, "mask_valid": mask is not None, "mask_error": mask_error,
                    "category_reviewed": False, "category_confirmed": False,
                    "review_status": "unavailable", "p_yes": None,
                    "mask_quality_reviewed": False, "mask_purity": None,
                    "mask_completeness": None, "mask_geometry_eligible": False,
                    "mask_quality_assessed": False, "geometry_usable": False,
                    "geometry_eligible": False, "world_identity_confirmed": False,
                    "geometry_scope": "none", "review_images": (), "review": {},
                }
                if mask_error:
                    errors.append({"stage": "mask", "view_index": view_index,
                                   "candidate_id": cid, "error": mask_error})
                if mask is not None:
                    from skill3d.segmentation.mask_semantic_review import candidate_images
                    images = candidate_images(frame, candidate["bbox"], mask)
                    candidate["review_images"] = images
                    if self.reviewer is not None:
                        try:
                            receipt = self.reviewer(_snapshot(candidate), frame, mask)
                            candidate["review"] = receipt
                            p_yes = _validate_review(
                                receipt, candidate, images,
                                self.expected_review_identity_sha256, self.yes_threshold)
                            candidate.update(
                                p_yes=p_yes, category_reviewed=True,
                                category_confirmed=p_yes >= self.yes_threshold,
                                review_status=("confirmed" if p_yes >= self.yes_threshold
                                               else "not_established"))
                        except Exception as exc:
                            candidate["review_error"] = f"{type(exc).__name__}: {exc}"
                            errors.append({"stage": "reviewer", "candidate_id": cid,
                                           "view_index": view_index,
                                           "error": candidate["review_error"]})
                candidates.append(candidate)
        capabilities = {
            "visual": bool(frames), "detection_2d": bool(candidates),
            "segmentation_2d": any(c["mask_valid"] for c in candidates),
            "category_review": any(c["category_reviewed"] for c in candidates),
            "category_confirmation": any(c["category_confirmed"] for c in candidates),
            "relative_depth": bool(depths), "relative_points": bool(points),
            "object_mask_geometry": False, "geometry_trusted": False,
            "metric_geometry": False, "world_geometry": False, "cross_view_identity": False,
        }
        result = _prepared(
            problem=public, frames=frames, candidates=candidates, depths=depths,
            points=points, intrinsics=intrinsics, geometry_masks=geometry_masks,
            geometries=geometries,
            categories=requested, cache_key=key, normalized_image_paths=normalized,
            capabilities=capabilities, errors=errors,
            provenance={"config": config, "images": image_identities,
                        "category_source": "explicit" if categories is not None else
                        "question_parser" if self.category_parser is not None else "not_supplied",
                        "geometry_scope": "per_view_relative_only",
                        "points_source_by_view": geometry_sources,
                        "metric_scale_verified": False, "world_frame_verified": False,
                        "mask_quality_assessed": False, "labels_used": False},
        )
        if not errors and self.max_cache_entries:
            self._cache[key] = _prepared(**{
                name: getattr(result, name) for name in PreparedSmallImage.__dataclass_fields__})
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)
        return result


def _detection(row, categories, shape):
    if not isinstance(row, Mapping):
        raise TypeError("detector must return mappings")
    category = str(row["category"]).lower().strip()
    if category not in categories:
        raise ValueError("detector category was not requested")
    box = np.asarray(row["box_xyxy"], dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("box_xyxy must be finite pixel coordinates")
    h, w = shape
    if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
        raise ValueError("box_xyxy is outside the RGB grid or reversed")
    confidence = float(row.get("confidence", 0))
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("invalid detection confidence")
    return {"category": category, "box_xyxy": box.tolist(), "confidence": confidence,
            "source_label": str(row.get("source_label", category))}


def _validate_geometry(output, shape):
    if not isinstance(output, Mapping):
        raise ValueError("depth backend must return a geometry mapping")
    if output.get("intrinsics_space", "pixel") != "pixel":
        raise ValueError("intrinsics must explicitly describe the RGB pixel grid")
    depth = np.asarray(output["depth"], dtype=np.float64)
    if depth.shape != shape:
        raise ValueError("depth shape differs from RGB; resize RGB before inference")
    valid = _binary_mask(output["valid_mask"], shape) & np.isfinite(depth) & (depth > 0)
    k = output.get("intrinsics")
    if k is not None:
        k = np.asarray(k, dtype=np.float64)
        if (k.shape != (3, 3) or not np.isfinite(k).all()
                or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1])
                or abs(np.linalg.det(k)) < 1e-12):
            raise ValueError("invalid intrinsics")
    points = output.get("points")
    source = "unavailable"
    if points is not None:
        points = np.asarray(points, dtype=np.float64)
        if points.shape != (*shape, 3):
            raise ValueError("point-map shape differs from RGB")
        valid &= np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
        if not np.allclose(points[..., 2][valid], depth[valid], rtol=1e-3, atol=1e-5):
            raise ValueError("point-map Z and depth disagree")
        if k is not None and valid.any():
            yy, xx = np.indices(shape)
            projected = points[valid] @ k.T
            uv = projected[:, :2] / projected[:, 2:]
            if not np.allclose(uv, np.stack((xx[valid], yy[valid]), axis=-1), atol=.6):
                raise ValueError("point map and intrinsics disagree on RGB grid")
        source = "model_point_map"
    elif k is not None:
        yy, xx = np.indices(shape)
        rays = np.stack((xx, yy, np.ones(shape)), axis=-1) @ np.linalg.inv(k).T
        with np.errstate(over="ignore", invalid="ignore"):
            points = rays * depth[..., None]
        valid &= np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
        source = "depth_backprojection"
    if not valid.any():
        raise ValueError("no finite positive supported geometry pixels")
    return {
        "depth": np.where(valid, depth, np.nan),
        "valid_mask": valid, "points_source": source,
        "points": np.where(valid[..., None], points, np.nan) if points is not None else None,
        "intrinsics": k,
    }


def _validate_review(receipt, candidate, images, expected_pin, threshold):
    """Bind native category evidence to this candidate and the original renders."""
    from skill3d.segmentation.intern_binary_review import (
        PROTOCOL_SHA256, RENDERING, _binary_answer, admission_identity, binary_request,
    )
    if not isinstance(receipt, Mapping) or receipt.get("status") != "observed":
        raise ValueError("category review unavailable")
    identity, binary, body = (receipt["request_identity"],
                              receipt["binary_review"], receipt["raw_response"])
    model = identity["model"]
    admission = admission_identity("category-only")
    p = _binary_answer(body["result"])
    raw_p = _binary_answer(body["uncalibrated_result"])
    prompt_version = binary["prompt_version"]
    request_sha = _digest(binary_request(
        [r["url"] for r in images], candidate["category"], prompt_version))
    if not (
        _digest(model) == expected_pin == receipt["model_identity_sha256"]
        == body["identity_sha256"]
        and receipt.get("candidate_id") == identity.get("candidate_id") == candidate["candidate_id"]
        and identity.get("frame_slot") == candidate["view_index"]
        and identity.get("bbox") == list(candidate["box_xyxy"])
        and identity.get("target_category") == binary.get("target_category") == candidate["category"]
        and receipt.get("request_sha256") == body.get("request_sha256") == request_sha
        and identity.get("images") == [r["identity"] for r in images]
        and receipt.get("admission") == identity.get("admission") == admission
        and binary.get("admission_policy") == "category-only"
        and identity.get("rendering") == RENDERING
        and identity.get("protocol_sha256") == binary.get("protocol_sha256") == PROTOCOL_SHA256
        and identity.get("prompt_version") == prompt_version
        and binary.get("threshold") == identity.get("yes_threshold") == threshold
        and binary.get("p_yes") == p["yes"] and binary.get("p_no") == p["no"]
        and binary.get("raw_p_yes") == raw_p["yes"] and binary.get("raw_p_no") == raw_p["no"]
        and binary.get("decision") == body["result"]["answers"]["target_match"]["decision"]
        and binary.get("semantic_pass") is (p["yes"] >= threshold)
        and receipt.get("binary_review_sha256") == _digest(binary)
        and receipt.get("observation") == {} and not receipt.get("frozen_quality")
        and receipt.get("observation_sha256") == _digest({})
        and not identity.get("quality_manifest_sha256")
        and body["result"].get("calibration") == model.get("calibration")
        and body["uncalibrated_result"].get("calibration", {}).get("temperature") == 1
    ):
        raise ValueError("category receipt/model/candidate/image identity mismatch")
    vision = body.get("vision_evidence", {})
    if (vision.get("image_count") != 3 or vision.get("image_token_count", 0) <= 0
            or not vision.get("pixel_values_shape") or len(vision.get("image_grid_thw", [])) != 3
            or len(vision.get("decision_logit_positions", [])) != 1):
        raise ValueError("native review did not encode all three candidate images")
    return float(p["yes"])
