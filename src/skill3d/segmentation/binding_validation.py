"""Small whole-object contract shared by M5, cache readers, tools and M11."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from skill3d.infra.m5_identity import file_identity, content_sha256
from skill3d.schemas.reconstruction import BindingValidation

BINDING_VALIDATION_VERSION = "m5-binding-validation-v3-sealed-semantics"
SEMANTIC_IDENTITY_VERSION = "m5-binding-semantic-identity-v1"
SEMANTIC_IDENTITY_KEY = "binding_semantics"
NUMERIC_TOLERANCE = 1e-6


def _semantic_identity(obj):
    # Stored beside array identities, so saved receipts bind outputs as well as inputs.
    return {
        "version": SEMANTIC_IDENTITY_VERSION,
        "sha256": content_sha256({
            "version": SEMANTIC_IDENTITY_VERSION,
            "obj_id": obj.obj_id, "track_id": obj.track_id,
            "category_name": obj.category_name, "source_label": obj.source_label,
            "grounding_status": obj.grounding_status,
            "candidate_ids": obj.candidate_ids, "question_roles": obj.question_roles,
            "binding_provenance": obj.binding_provenance,
        }),
    }


def _semantic_policy_passes(row):
    allowed = {
        "bailian": {"visual-quality-and-relation"},
        "intern-decision-local": {"visual-quality-and-relation"},
        "intern-decision-binary-local": {"category-only", "frozen-quality"},
    }
    policy = row.get("semantic_policy")
    provider = row.get("semantic_provider")
    admission = row.get("semantic_admission", {})
    return bool(
        isinstance(provider, str) and isinstance(policy, str)
        and isinstance(admission, dict)
        and policy in allowed.get(provider, set())
        and (not admission or admission.get("policy") == policy))


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
            isinstance(row, dict) and row.get("semantic_relation") == "match"
            and row.get("semantic_visual_request_sha256")
            and row.get("semantic_relation_request_sha256")
            for row in obj.binding_provenance)))
    try:
        files[SEMANTIC_IDENTITY_KEY] = _semantic_identity(obj)
        checks["semantic_policy"] = bool(
            obj.grounding_status != "question_driven"
            or (obj.binding_provenance and all(
                _semantic_policy_passes(row) for row in obj.binding_provenance)))
        checks["candidate_provenance"] = bool(
            obj.grounding_status != "question_driven"
            or ([row.get("candidate_id") for row in obj.binding_provenance]
                == obj.candidate_ids and all(
                    row.get("category") == obj.category_name
                    for row in obj.binding_provenance)))
        for row in obj.binding_provenance or []:
            if row.get("semantic_provider") == "bailian":
                from .mask_semantic_review import visual_quality_passes
                observation = row.get("semantic_observation")
                identity = row.get("semantic_request_identity") or {}
                checks["qwen_observation_integrity"] = bool(
                    visual_quality_passes(observation)
                    and row.get("semantic_observation_sha256") == content_sha256(observation)
                    and identity.get("candidate_id") == row.get("candidate_id")
                    and row.get("semantic_visual_request_sha256") == content_sha256(identity)
                    and row.get("semantic_model_identity_sha256") == content_sha256(identity.get("model")))
                if not checks["qwen_observation_integrity"]:
                    break
    except (ValueError, TypeError, KeyError, AttributeError):
        checks["semantic_identity"] = False
    for row in obj.binding_provenance or []:
        if row.get("semantic_provider") == "intern-decision-local":
            from .intern_decision_protocol import PROTOCOL_SHA256
            from .mask_semantic_review import visual_quality_passes
            observation = row.get("semantic_observation") or {}
            identity = row.get("semantic_request_identity") or {}
            try:
                checks["intern_observation_integrity"] = bool(
                    visual_quality_passes(observation)
                    and row.get("semantic_observation_sha256") == content_sha256(observation)
                    and identity.get("protocol_sha256") == PROTOCOL_SHA256
                    and row.get("semantic_model_identity_sha256") == content_sha256(identity.get("model")))
            except (ValueError, TypeError, KeyError, AttributeError):
                checks["intern_observation_integrity"] = False
            if not checks["intern_observation_integrity"]:
                break
    for row in obj.binding_provenance or []:
        if row.get("semantic_provider") == "intern-decision-binary-local":
            from .intern_binary_review import PROTOCOL_SHA256 as binary_protocol_sha, quality_passes, validate_threshold
            if row.get("semantic_policy") == "category-only":
                from .intern_binary_review import validate_category_receipt
                try:
                    checks["intern_binary_integrity"] = validate_category_receipt(row, obj.category_name)
                except (ValueError, TypeError, KeyError, AttributeError):
                    checks["intern_binary_integrity"] = False
                if not checks["intern_binary_integrity"]:
                    break
                continue
            if row.get("semantic_policy") != "frozen-quality":
                checks["intern_binary_integrity"] = False
                break
            observation = row.get("semantic_observation") or {}
            identity = row.get("semantic_request_identity") or {}
            binary = row.get("semantic_binary_review") or {}
            frozen = row.get("semantic_frozen_quality") or {}
            try:
                threshold = validate_threshold(binary.get("threshold"))
                probabilities_ok = (type(binary.get("p_yes")) in (int, float)
                    and type(binary.get("p_no")) in (int, float)
                    and np.isfinite([binary["p_yes"], binary["p_no"]]).all()
                    and 0 <= binary["p_yes"] <= 1 and 0 <= binary["p_no"] <= 1
                    and abs(binary["p_yes"] + binary["p_no"] - 1) <= 1e-5)
                checks["intern_binary_integrity"] = bool(
                    probabilities_ok and binary["p_yes"] >= threshold and binary.get("semantic_pass") is True
                    and binary.get("target_category") == obj.category_name
                    and binary.get("target_category") == identity.get("target_category")
                    and binary.get("prompt_version") == identity.get("prompt_version")
                    and binary.get("protocol_sha256") == binary_protocol_sha
                    and identity.get("protocol_sha256") == binary_protocol_sha
                    and row.get("semantic_binary_review_sha256") == content_sha256(binary)
                    and row.get("semantic_observation_sha256") == content_sha256(observation)
                    and row.get("semantic_model_identity_sha256") == content_sha256(identity.get("model"))
                    and "observed_object" not in observation and quality_passes(observation)
                    and frozen.get("manifest_sha256") == identity.get("quality_manifest_sha256")
                    and frozen.get("manifest_sha256") and frozen.get("entry_sha256") == content_sha256(frozen.get("entry"))
                    and frozen["entry"]["candidate_id"] == row.get("candidate_id")
                    and frozen["entry"]["image_sha256"] == [im["sha256"] for im in identity["images"]]
                    and all(frozen["entry"]["observation"].get(k) == observation[k]
                            for k in ("visibility", "mask_purity", "mask_completeness")))
            except (ValueError, TypeError, KeyError, AttributeError):
                checks["intern_binary_integrity"] = False
            if not checks["intern_binary_integrity"]:
                break
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
        return (not strict and not obj.binding_validation and not any(
            "semantic_visual_request_sha256" in row for row in obj.binding_provenance))
    try:
        return validate_binding(obj, check_receipt=True).status == "validated"
    except (OSError, ValueError, TypeError):
        return False
