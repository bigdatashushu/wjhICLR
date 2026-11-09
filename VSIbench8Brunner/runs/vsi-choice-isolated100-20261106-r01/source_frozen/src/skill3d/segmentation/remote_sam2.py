"""Batch SAM2 transport preserving original RGB pixels and box coordinates."""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request

import numpy as np

from skill3d.infra.m5_identity import content_sha256, array_identity


class RemoteSAM2:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({})).open
        with self.opener(endpoint + "/health", timeout=30) as response:
            health = json.load(response)
        self._skill3d_m5_identity = health["identity"]
        self.identity_sha256 = content_sha256(health["identity"])
        if self.identity_sha256 != health["identity_sha256"]:
            raise ValueError("SAM2 service identity mismatch")
        self.receipts = []

    def set_image(self, image):
        raise RuntimeError("remote SAM2 requires batching")

    def predict(self, **kwargs):
        raise RuntimeError("remote SAM2 requires batching")

    def segment_frame_boxes(self, image, boxes):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, frame=image)
        payload = {"frame_npz": base64.b64encode(buffer.getvalue()).decode(),
                   "boxes": [[float(v) for v in box] for box in boxes]}
        token = os.environ["VSI_SAM2_TOKEN"]
        request = urllib.request.Request(
            self.endpoint + "/segment", data=json.dumps(payload).encode(), method="POST",
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        with self.opener(request, timeout=180) as response:
            raw = response.read()
        if token.encode() in raw:
            raise ValueError("SAM2 response contains credentials")
        result = json.loads(raw)
        if (result["request_sha256"] != content_sha256(payload)
                or result["identity_sha256"] != self.identity_sha256):
            raise ValueError("SAM2 request/service identity changed")
        with np.load(io.BytesIO(base64.b64decode(result["masks_npz"])),
                     allow_pickle=False) as arrays:
            masks = arrays["masks"].copy()
        if masks.shape != (len(boxes), *image.shape[:2]) or masks.dtype != np.bool_:
            raise ValueError("SAM2 mask count/shape/dtype mismatch")
        self.receipts.append({
            "schema": "vsi-sam2-transport/1", "request_sha256": result["request_sha256"],
            "identity_sha256": self.identity_sha256, "frame_identity": array_identity(image),
            "boxes": payload["boxes"], "masks": [array_identity(mask) for mask in masks],
            "automatic_retry": False,
        })
        return list(masks)
