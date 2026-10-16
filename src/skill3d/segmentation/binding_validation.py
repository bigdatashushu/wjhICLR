"""Small whole-object contract shared by M5, cache readers, tools and M11."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skill3d.infra.m5_identity import file_identity
from skill3d.schemas.reconstruction import BindingValidation

BINDING_VALIDATION_VERSION = "m5-binding-validation-v2-semantic-review"
NUMERIC_TOLERANCE = 1e-6


def validate_binding(obj, *, depth_maps=None, n_frames=None,
                     generation_sha256="", check_receipt=False):
    checks, files = {}, {}
    errors = []
    arrays = {}
    for field in ("mask_per_frame", "pointcloud_world", "pointconf_world"):
        ref = str(getattr(obj, field, "") or "")
        if field == "pointconf_world" and not ref:
            continue  # confidence is optional; never synthesize it
        try:
            path = Path(ref)
            arrays[field] = np.load(path, allow_pickle=False, mmap_mode="r")
            files[field] = file_identity(path)
        except (OSError, ValueError) as exc:
            errors.append(f"{field}: {type(exc).__name__}: {exc}")
    mask, pts = arrays.get("mask_per_frame"), arrays.get("pointcloud_world")
    conf = arrays.get("pointconf_world")
    mask_ok = (mask is not None and mask.ndim == 3
               and (n_frames is None or len(mask) == n_frames)
               and np.isin(mask, [0, 1]).all() and bool(mask.any()))
    checks["nonempty_mask"] = bool(mask_ok)
    if mask_ok:
        visible = np.flatnonzero(mask.any(axis=(1, 2))).tolist()
        checks["visible_frames"] = visible == list(obj.visible_frames)
        checks["mask_support"] = bool(np.isclose(
            obj.mask_support_ratio, len(visible) / len(mask),
            rtol=NUMERIC_TOLERANCE, atol=NUMERIC_TOLERANCE))
        if depth_maps is not None:
            depth = np.asarray(depth_maps)
            checks["valid_depth"] = bool(
                depth.shape == mask.shape
                and all(np.any((mask[i] != 0) & np.isfinite(depth[i])
                               & (depth[i] > 0)) for i in visible))
    points_ok = (pts is not None and pts.ndim == 2 and pts.shape[1] == 3
                 and len(pts) > 0 and np.isfinite(pts).all())
    checks["finite_points"] = bool(points_ok)
    checks["point_confidence"] = bool(conf is None or (
        points_ok and conf.ndim == 1 and len(conf) == len(pts)
        and np.isfinite(conf).all()))
    if points_ok:
        checks["centroid"] = bool(np.allclose(
            obj.centroid_world, pts.mean(axis=0), rtol=NUMERIC_TOLERANCE,
            atol=NUMERIC_TOLERANCE))
        checks["bbox"] = bool(len(obj.bbox) == 6 and np.allclose(
            obj.bbox, np.concatenate([pts.min(axis=0), pts.max(axis=0)]),
            rtol=NUMERIC_TOLERANCE, atol=NUMERIC_TOLERANCE))
    checks["no_binding_flags"] = not bool(obj.binding_flags)
    checks["instance_not_suspect"] = not obj.duplicate_suspect
    checks["candidate_identity"] = bool(obj.candidate_ids and obj.track_id
                                        and obj.question_roles)
    checks["semantic_review"] = bool(
        obj.grounding_status != "question_driven"
        or (obj.binding_provenance and all(
            row.get("semantic_relation") == "match"
            and row.get("semantic_visual_request_sha256")
            and row.get("semantic_relation_request_sha256")
            for row in obj.binding_provenance)))
    if check_receipt:
        receipt = obj.binding_validation
        checks["receipt"] = bool(
            receipt and receipt.version == BINDING_VALIDATION_VERSION
            and receipt.status == "validated" and receipt.files == files
            and receipt.generation_sha256
            and (not generation_sha256
                 or receipt.generation_sha256 == generation_sha256))
    errors.extend(name for name, passed in checks.items() if not passed)
    return BindingValidation(
        version=BINDING_VALIDATION_VERSION,
        status="validated" if not errors else "unavailable",
        checks=checks, errors=errors, generation_sha256=generation_sha256,
        files=files)


def binding_is_usable(obj, *, strict=False):
    if obj.binding_flags:
        return False
    if obj.grounding_status != "question_driven":
        return not strict
    try:
        return validate_binding(obj, check_receipt=True).status == "validated"
    except (OSError, ValueError, TypeError):
        return False
