"""Lazy adapters over existing detector/SAM2/Intern APIs and standalone MoGe2.

Constructing these adapters makes no model/service calls. Parent launchers own
health verification, endpoint selection, device allocation, and frozen assets.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _identified(identity, **settings):
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("an explicit model identity mapping is required")
    return {**dict(identity), "small_image_adapter": settings}


def _file_sha256(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


class GroundingDINOAdapter:
    """Existing harness detector wrapper, called separately for each target."""

    def __init__(self, *, identity, endpoint="", detect_fn=None,
                 box_threshold=.25, timeout_s=120, max_attempts=3,
                 total_timeout_s=180):
        self.endpoint, self.detect_fn = endpoint, detect_fn
        self.kwargs = dict(box_threshold=box_threshold, timeout_s=timeout_s,
                           max_attempts=max_attempts, total_timeout_s=total_timeout_s)
        self.identity = _identified(identity, api="open_vocab_detector.detect",
                                    endpoint=endpoint, **self.kwargs)
        if detect_fn is None and not endpoint:
            raise ValueError("detector endpoint or injected detect_fn is required")

    def __call__(self, rgb, categories):
        if self.detect_fn is None:
            from skill3d.segmentation.open_vocab_detector import detect
        else:
            detect = self.detect_fn
        rows = []
        for category in categories:
            detections = detect(rgb, category + ".", endpoint=self.endpoint, **self.kwargs)
            if getattr(detect, "last_error", ""):
                raise RuntimeError(f"GroundingDINO unavailable: {detect.last_error}")
            for detection in detections:
                rows.append({
                    "category": category, "box_xyxy": list(detection.bbox_xyxy),
                    "confidence": float(detection.confidence),
                    "source_label": str(detection.label),
                })
        return rows


class SAM2ImageAdapter:
    """No temporal propagation or world binding; reuse the frozen SAM2 model."""

    def __init__(self, *, identity, model=None, predictor=None,
                 checkpoint=None, config=None, build_fn=None, segment_fn=None):
        self.model, self.predictor = model, predictor
        self.checkpoint, self.config = checkpoint, config
        self.build_fn, self.segment_fn = build_fn, segment_fn
        self.identity = _identified(identity, api="sam2_tracker.segment_frame_boxes",
                                    checkpoint=checkpoint, config=config)

    def __call__(self, rgb, boxes):
        from skill3d.segmentation.sam2_tracker import build_image_predictor, segment_frame_boxes
        if self.predictor is None:
            builder = self.build_fn or build_image_predictor
            self.predictor = builder(
                model=self.model, checkpoint=self.checkpoint, config=self.config)
        return (self.segment_fn or segment_frame_boxes)(rgb, boxes, self.predictor)


class InternCategoryAdapter:
    """Exact original three-image, category-only native binary review."""

    def __init__(self, config=None, *, client=None, seed=137,
                 min_pixels=1, max_pixels=1048576):
        from skill3d.online.config import default_mask_review
        from skill3d.segmentation.intern_binary_review import PROTOCOL_SHA256
        from skill3d.segmentation.intern_decision_client import make_review_client
        config = {**default_mask_review(), **dict(config or {})}
        if (config.get("provider") != "intern-decision" or config.get("mode") != "binary"
                or config.get("admission_policy") != "category-only"
                or config.get("quality_manifest") or config.get("quality_manifest_sha256")
                or not config.get("expected_identity_sha256")
                or not .77 <= config.get("yes_threshold", 0) <= .99):
            raise ValueError("small-image Intern review requires pinned category-only >=.77")
        self.client = client if client is not None else make_review_client(config)
        self.expected_identity_sha256 = config["expected_identity_sha256"]
        if (getattr(self.client, "expected_identity_sha256", "") != self.expected_identity_sha256
                or getattr(self.client, "yes_threshold", None) != config["yes_threshold"]
                or getattr(self.client, "prompt_version", None) != config["prompt_version"]
                or getattr(self.client, "admission", {}).get("policy") != "category-only"):
            raise ValueError("injected Intern client differs from pinned review config")
        self.seed, self.min_pixels, self.max_pixels = seed, min_pixels, max_pixels
        self.identity = {
            "expected_identity_sha256": self.expected_identity_sha256,
            "protocol_sha256": PROTOCOL_SHA256, "config": config,
            "seed": seed, "min_pixels": min_pixels, "max_pixels": max_pixels,
        }

    def __call__(self, candidate, rgb, mask):
        return self.client.observe_candidate(
            candidate, rgb, mask, seed=self.seed, min_pixels=self.min_pixels,
            max_pixels=self.max_pixels, out_dir=None)


def _numpy(value, ndim):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != ndim:
        raise ValueError("unexpected MoGe2 output dimensions")
    return array


class MoGe2Adapter:
    """Standalone v2.infer, without VGGT K or any metric-fusion wrapper.

    MoGe's intrinsics are normalized image coordinates. This adapter converts
    them to the same pixel-center grid as RGB/depth. The output is deliberately
    exposed as relative-only even when the native model predicts metric scale.
    A local model.pt is mandatory unless an already loaded model is supplied.
    """

    def __init__(self, *, identity, checkpoint=None, model=None, device="cpu",
                 infer_kwargs=None):
        if model is None and checkpoint is None:
            raise ValueError("MoGe2 requires an explicit local checkpoint or loaded model")
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        if self.checkpoint is not None and self.checkpoint.is_dir():
            self.checkpoint = self.checkpoint / "model.pt"
        self.model, self.device = model, str(device)
        self.infer_kwargs = {"use_fp16": self.device.startswith("cuda"),
                             **dict(infer_kwargs or {})}
        if "fov_x" in self.infer_kwargs:
            raise ValueError("standalone MoGe2 must estimate its own camera, not use VGGT FOV")
        self.identity = _identified(
            identity, api="moge.model.v2.MoGeModel.infer",
            checkpoint=str(self.checkpoint) if self.checkpoint is not None else None,
            device=self.device, infer_kwargs=self.infer_kwargs,
            geometry_scope="per_view_relative_only")
        if self.checkpoint is not None:
            if not self.checkpoint.is_file():
                raise ValueError("MoGe2 checkpoint must exist locally; no downloads")
            actual = _file_sha256(self.checkpoint)
            if identity.get("checkpoint_sha256", actual) != actual:
                raise ValueError("MoGe2 checkpoint identity mismatch")
            self.identity["checkpoint_sha256"] = actual

    def __call__(self, rgb):
        import torch
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested MoGe2 CUDA device is unavailable; no CPU fallback")
        if self.model is None:
            if _file_sha256(self.checkpoint) != self.identity["checkpoint_sha256"]:
                raise ValueError("MoGe2 checkpoint changed before model loading")
            from moge.model.v2 import MoGeModel
            self.model = MoGeModel.from_pretrained(str(self.checkpoint)).to(self.device).eval()
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("MoGe2 input must be uint8 RGB")
        tensor = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)
        with torch.inference_mode():
            output = self.model.infer(tensor.to(self.device), **self.infer_kwargs)
        if not isinstance(output, Mapping) or "mask" not in output or "metric_scale" in output:
            raise ValueError("unexpected pinned MoGe2 v2.infer output")
        depth = _numpy(output["depth"], 2)
        mask = _numpy(output["mask"], 2)
        points = _numpy(output["points"], 3) if output.get("points") is not None else None
        k = _numpy(output["intrinsics"], 2).astype(np.float64, copy=True)
        if k.shape != (3, 3) or depth.shape != image.shape[:2] or mask.shape != depth.shape:
            raise ValueError("MoGe2 outputs do not describe the input RGB grid")
        h, w = depth.shape
        k[0] *= w
        k[1] *= h
        k[0, 2] -= .5
        k[1, 2] -= .5
        return {"depth": depth, "valid_mask": mask, "points": points,
                "intrinsics": k, "intrinsics_space": "pixel"}


def build_existing_backends(
    *, detector_endpoint, detector_identity, sam2_identity,
    sam2_model=None, sam2_predictor=None, sam2_checkpoint=None, sam2_config=None,
    review_config=None, depth_estimator=None, depth_identity=None,
):
    """Return constructor kwargs for SmallImagePrecomputer; no eager inference."""
    detector = GroundingDINOAdapter(identity=detector_identity, endpoint=detector_endpoint)
    segmenter = SAM2ImageAdapter(
        identity=sam2_identity, model=sam2_model, predictor=sam2_predictor,
        checkpoint=sam2_checkpoint, config=sam2_config)
    reviewer = InternCategoryAdapter(review_config)
    identities = {"detector": detector.identity, "segmenter": segmenter.identity,
                  "reviewer": reviewer.identity}
    if depth_estimator is not None:
        identities["depth_estimator"] = (
            depth_identity if depth_identity is not None else getattr(depth_estimator, "identity", None))
    return {"detector": detector, "segmenter": segmenter, "reviewer": reviewer,
            "depth_estimator": depth_estimator, "model_identities": identities,
            "yes_threshold": reviewer.client.yes_threshold}
