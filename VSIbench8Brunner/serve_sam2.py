#!/usr/bin/env python3
"""Owned, authenticated SAM2 image service; no model requests or labels."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28102)
    args = parser.parse_args()
    from skill3d.segmentation import sam2_tracker as st
    from skill3d.infra.m5_identity import sam2_identity, content_sha256
    checkpoint, config = st.resolve_sam2_config()
    predictor = st.build_image_predictor()
    identity = sam2_identity(checkpoint, config, inference_mode="sam2_image_per_frame")
    if not identity:
        raise RuntimeError("SAM2 identity unavailable")
    token = os.environ["VSI_SAM2_TOKEN"]
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send(200, {"schema": "vsi-sam2-service/1", "identity": identity,
                            "identity_sha256": content_sha256(identity)})

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                self.send(401, {"error": "authentication"})
                return
            try:
                size = int(self.headers["Content-Length"])
                if not 0 < size < 64 * 1024 * 1024:
                    raise ValueError("body size")
                request = json.loads(self.rfile.read(size))
                with np.load(io.BytesIO(base64.b64decode(request["frame_npz"])),
                             allow_pickle=False) as arrays:
                    frame = arrays["frame"]
                with lock:
                    masks = st.segment_frame_boxes(frame, request["boxes"], predictor)
                buffer = io.BytesIO()
                np.savez_compressed(buffer, masks=np.stack(masks))
                self.send(200, {
                    "schema": "vsi-sam2-result/1",
                    "request_sha256": content_sha256(request),
                    "identity_sha256": content_sha256(identity),
                    "masks_npz": base64.b64encode(buffer.getvalue()).decode(),
                })
            except Exception as exc:
                self.send(500, {"error_type": type(exc).__name__})

    print(json.dumps({"status": "ready", "identity_sha256": content_sha256(identity)}),
          flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
