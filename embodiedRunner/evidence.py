"""Current-RGB preparation using the shared small-image pipeline without files."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import io
import re

import numpy as np
from PIL import Image

from action_contract import freeze, plain, public_hash
from action_host import PreparedEvidence
from protocol import ContractError, OperationalError, sha256
from sandbox_wire import pack_prepared
from skill3d.small_image.precompute import SmallImagePrecomputer, validate_prepared_content
from task_card import validate_perception

VERSION = "embodied-current-evidence/1"
POLICY = freeze({
    "schema": VERSION, "category_parser": "public-vocabulary/1",
    "max_categories": 4, "max_candidates_per_view": 1, "resize_to": None,
    "yes_threshold": .77, "min_mask_pixels": 1, "cache_entries": 1,
    "geometry_scope": "per_view_relative_only", "metric_world": False,
    "mask_quality_assessed": False, "automatic_retry": False,
})
# Long names precede shorter names. Only explicit public text matches enter.
VOCABULARY = tuple(sorted((
    "garbage can", "coffee table", "dining table", "side table", "light switch",
    "floor lamp", "desk lamp", "toilet paper", "soap bottle", "tissue box",
    "remote control", "paper towel", "wine bottle", "watering can", "bathtub",
    "refrigerator", "microwave", "countertop", "dishwasher", "cabinet", "drawer",
    "armchair", "sofa", "chair", "table", "desk", "bed", "shelf", "lamp",
    "door", "window", "sink", "toilet", "shower", "stove", "faucet",
    "apple", "potato", "tomato", "lettuce", "bread", "egg", "mug", "cup",
    "bowl", "plate", "knife", "fork", "spoon", "pan", "pot", "bottle",
    "book", "laptop", "phone", "pillow", "towel", "box", "basket", "plant",
), key=lambda x: (-len(x), x)))
PARSER_SHA256 = public_hash({"version": POLICY["category_parser"], "vocabulary": VOCABULARY})


def public_categories(instruction, action_table):
    """Instruction matches first, then action names; no method/model/GT input."""
    result = []
    for raw in (instruction, *action_table):
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw).lower()
        text = re.sub(r"[^a-z ]", " ", text)
        matches = sorted((m.start(), -len(name), name)
                         for name in VOCABULARY
                         for m in re.finditer(r"\b" + re.escape(name) + r"s?\b", text))
        occupied = []
        for start, _, name in matches:
            end = start + len(name)
            if any(start < b and end > a for a, b in occupied):
                continue
            occupied.append((start, end))
            if name not in result:
                result.append(name)
            if len(result) == POLICY["max_categories"]:
                return tuple(result)
    return tuple(result)


@dataclass(frozen=True)
class PerceptionBackends:
    """Each callable must perform exactly ONE invocation, without auto retry."""
    detector: object = None
    segmenter: object = None
    reviewer: object = None
    depth_estimator: object = None

    def identities(self):
        result = {}
        for name in ("detector", "segmenter", "reviewer", "depth_estimator"):
            fn = getattr(self, name)
            identity = getattr(fn, "identity", None) if fn is not None else None
            if fn is not None and (not isinstance(identity, Mapping) or not identity):
                raise ContractError("perception_identity_required")
            # Do not persist endpoints/checkpoint paths/backend configuration.
            result[name] = {"identity_sha256": public_hash(identity)} if fn else {"enabled": False}
        if self.reviewer is not None:
            pin = getattr(self.reviewer, "expected_identity_sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", pin):
                raise ContractError("review_identity_required")
            result["reviewer"]["expected_identity_sha256"] = pin
            client = getattr(self.reviewer, "client", None)
            if client is not None and getattr(client, "_identity", None) is None:
                raise ContractError("review_health_preflight_required")
        return result


class CombinedDINO:
    """One compound prompt replaces the shared adapter's per-category loop."""
    def __init__(self, *, identity, endpoint, detect_fn=None):
        self.identity = {"backend_sha256": public_hash(identity),
                         "compound_prompt": True, "max_attempts": 1, "box_threshold": .25}
        self.endpoint, self.detect_fn = endpoint, detect_fn

    def __call__(self, rgb, categories):
        from skill3d.segmentation.open_vocab_detector import detect, assign_categories
        fn = self.detect_fn or detect
        detections = fn(
            rgb, " . ".join(categories) + " .", endpoint=self.endpoint,
            box_threshold=.25, timeout_s=60, max_attempts=1, total_timeout_s=60)
        if getattr(fn, "last_error", ""):
            raise OperationalError("perception_service_error")
        assigned, _ = assign_categories(detections, categories)
        return [{"category": category, "box_xyxy": list(item.bbox_xyxy),
                 "confidence": float(item.confidence), "source_label": str(item.label)}
                for category in categories for item in assigned[category]]


class _StopService(BaseException):
    """Bypass the shared preparer's catch-and-continue policy."""
    def __init__(self, cause):
        self.cause = cause


class _MemoryPrecomputer(SmallImagePrecomputer):
    def __init__(self, png, **kwargs):
        self.png = png
        super().__init__(**kwargs)

    def _decode(self, paths):
        if tuple(paths) != ("current-rgb",):
            raise ContractError("evidence_image_route")
        with Image.open(io.BytesIO(self.png)) as image:
            rgb = image.convert("RGB")
            w, h = rgb.size
            frame = np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(h, w, 3)
        return (frame,), [{
            "view_index": 0, "source_sha256": sha256(self.png),
            "rgb_sha256": sha256(frame.tobytes()), "source_format": "PNG",
            "source_size_wh": [w, h], "rgb_size_wh": [w, h],
            "source_to_rgb_xy": [1, 1], "source_to_rgb_affine": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "coordinate_convention": "integer_pixel_centers; boxes_use_pixel_edges",
            "coordinate_grid": "rgb_pixels_xy", "exif_transpose": False,
        }], ()


def evidence_scope(snapshot):
    obs = snapshot.observation
    # No final snapshot hash: that hash itself contains the manifest hash.
    return {"episode_id": obs.episode_id, "decision_id": snapshot.decision_id,
            "observation_revision": obs.observation_revision,
            "mutation_revision": obs.mutation_revision, "view_id": obs.view_id,
            "rgb_sha256": obs.rgb_sha256, "pixel_sha256": obs.pixel_sha256,
            "action_table_sha256": obs.action_table_sha256}


class CurrentEvidencePreparer:
    def __init__(self, backends=PerceptionBackends()):
        self.backends, self.identities = backends, backends.identities()
        self.config_sha256 = public_hash({
            "policy": POLICY, "parser_sha256": PARSER_SHA256, "models": self.identities,
            "dynamic_request": {"max_targets": 4, "one_candidate_per_category": True,
                                "optional_depth": True}})
        self.cache, self.manifests = {}, {}
        self.trajectory = None

    def __call__(self, snapshot, binding, budget, *, request=None):
        request = validate_perception(request) if request is not None else None
        scope = evidence_scope(snapshot)
        key = public_hash({"scope": scope, "config_sha256": self.config_sha256,
                           "request": request})
        if key in self.cache:
            result = self.cache[key]
            validate_prepared_content(result.prepared)
            return result
        obs = snapshot.observation
        categories = (tuple(row["category"] for row in request["targets"]) if request is not None
                      else public_categories(obs.instruction, obs.action_table))
        detector_completed = False
        max_candidates = max(1, len(categories)) if request is not None else 1
        def governed(name, stage):
            fn = getattr(self.backends, name)
            if fn is None:
                return None
            def invoke(*args):
                nonlocal detector_completed
                try:
                    with budget.request(stage, decision_id=snapshot.decision_id, view_id=obs.view_id) as permit:
                        if self.trajectory:
                            self.trajectory.emit("perception_request", decision_id=snapshot.decision_id,
                                                 backend=name, args=args)
                        value = fn(*args)
                        if self.trajectory:
                            self.trajectory.emit("perception_response", decision_id=snapshot.decision_id,
                                                 backend=name, output=value)
                        permit["usage"] = getattr(fn, "last_usage", None)
                        if name == "reviewer":
                            if not isinstance(value, Mapping) or value.get("status") != "observed":
                                raise OperationalError("perception_service_error")
                            raw = value.get("raw_response", {})
                            permit["usage"] = raw.get("result", {}).get("usage", raw.get("usage"))
                        if name == "detector":
                            # Bound iteration; one candidate means at most one native review.
                            rows = []
                            for index, row in enumerate(value):
                                if index >= 256:
                                    raise ContractError("detector_response_cap")
                                rows.append(row)
                            rows.sort(key=lambda r: (
                                categories.index(r["category"]), -float(r["confidence"]),
                                tuple(r["box_xyxy"])))
                            if request is None:
                                value = rows[:1]
                            else:
                                # Keep a candidate for EACH requested role/category,
                                # rather than allowing the first category to occupy all slots.
                                value = [next(row for row in rows if row["category"] == category)
                                         for category in categories
                                         if any(row["category"] == category for row in rows)]
                            detector_completed = bool(categories)
                        return value
                except Exception as exc:
                    # Never retain or expose provider exception text.
                    raise _StopService(exc) from None
            return invoke
        problem = {
            "benchmark": binding.environment, "sample_id": binding.official_task_id,
            "source_split": binding.subset, "question": obs.instruction,
            "options": [], "image_paths": ["current-rgb"],
            "task_type": "embodied_current_action", "input_sha256": binding.input_sha256,
        }
        try:
            pipeline = _MemoryPrecomputer(
                obs.png, detector=governed("detector", "dino"),
                segmenter=governed("segmenter", "sam"), reviewer=governed("reviewer", "category_review"),
                depth_estimator=(governed("depth_estimator", "moge")
                                 if request is None or request["depth"] else None),
                model_identities=self.identities, method_identity=self.config_sha256,
                expected_review_identity_sha256=self.identities["reviewer"].get("expected_identity_sha256", ""),
                max_categories=4, max_candidates_per_view=max_candidates, max_cache_entries=0)
            prepared = pipeline.process(problem, categories=categories)
        except _StopService as exc:
            from budget import BudgetExhausted
            if isinstance(exc.cause, BudgetExhausted):
                raise exc.cause from None
            raise OperationalError("perception_service_error") from None
        if prepared.errors:
            raise OperationalError("perception_result_invalid")
        validate_prepared_content(prepared)
        wire = pack_prepared(snapshot, prepared)
        capabilities = ["current_rgb"]
        if detector_completed:
            capabilities.append("candidate_detection")
        if prepared.candidates:
            capabilities.append("candidate_category")
        if prepared.has_relative_depth:
            capabilities.append("relative_depth")
        if prepared.has_relative_points:
            capabilities.append("relative_points")
        manifest = {
            "schema": VERSION, "scope": scope, "evidence_revision": snapshot.evidence_revision + 1,
            "config_sha256": self.config_sha256, "parser_sha256": PARSER_SHA256,
            "categories_sha256": public_hash(categories),
            "artifact_sha256": prepared.artifact_sha256, "wire_sha256": wire["sha256"],
            "capabilities": capabilities, "geometry_scope": "per_view_relative_only",
            "mask_quality_assessed": False, "metric_world": False, "labels_used": False,
            "reference_answer_visible": False,
        }
        if request is not None:
            manifest["request_sha256"] = public_hash(request)
            manifest["category_source"] = "qwen_current_subgoal"
        manifest_hash = public_hash(manifest)
        result = PreparedEvidence(prepared, manifest_hash, tuple(capabilities))
        # One current decision only. Identical RGB after mutation still re-prepares.
        self.cache = {key: result}
        self.manifests = {manifest_hash: freeze(manifest)}
        return result

    def validate(self, snapshot, result):
        validate_prepared_content(result.prepared)
        manifest = self.manifests.get(result.manifest_sha256)
        if (manifest is None or public_hash(manifest) != snapshot.evidence_manifest_sha256
                or manifest["scope"] != evidence_scope(snapshot)
                or manifest["evidence_revision"] != snapshot.evidence_revision
                or tuple(manifest["capabilities"]) != snapshot.capabilities
                or manifest["wire_sha256"] != pack_prepared(snapshot, result.prepared)["sha256"]):
            raise ContractError("current_evidence_binding_mismatch")
        return plain(manifest)


def summarize_requested_evidence(result, request):
    """Finite semantic summary; arrays remain behind current tool authorization."""
    rows = []
    for target in request["targets"]:
        candidates = [row for row in result.prepared.candidates if row["category"] == target["category"]]
        if not candidates:
            status = "no_detection"
        elif any(row["category_confirmed"] for row in candidates):
            status = "category_confirmed_mask_unassessed"
        else:
            status = "category_unconfirmed"
        rows.append({**plain(target), "status": status,
                     "candidates": [{"id": row["candidate_id"], "box_xyxy": list(row["box_xyxy"]),
                                     "category_confirmed": row["category_confirmed"]}
                                    for row in candidates]})
    return {"status": "prepared", "targets": rows, "depth_requested": request["depth"],
            "capabilities": list(result.capabilities), "unit": "per_view_relative",
            "mask_quality_assessed": False, "geometry_usable": False}
