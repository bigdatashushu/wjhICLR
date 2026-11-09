#!/usr/bin/env python3
"""Compare frozen, label-free detector requests through owned gateways.

The input is a JSON manifest containing only image filenames, their PNG SHA256,
text prompts and thresholds. Images are encoded exactly like the detector client.
All compiled requests and endpoint identities are frozen before any POST. There
are no retries, labels, object bindings, solver calls or promotion decisions.
Use a serialized gateway for each endpoint; never benchmark a shared native
model with concurrent direct requests.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def read_requests(manifest):
    import cv2
    import numpy as np
    from PIL import Image

    manifest = Path(manifest)
    value = json.loads(manifest.read_bytes())
    if set(value) != {"schema", "requests"} or value["schema"] != "detector-request-benchmark/1":
        raise ValueError("invalid label-free request manifest")
    if not isinstance(value["requests"], list) or not 1 <= len(value["requests"]) <= 512:
        raise ValueError("invalid request count")
    requests = []
    identities = []
    for row in value["requests"]:
        if set(row) != {"image", "png_sha256", "text_prompt", "box_threshold"}:
            raise ValueError("unexpected request field")
        path = Path(row["image"])
        if not path.is_absolute():
            path = manifest.parent / path
        raw = path.read_bytes()
        if sha(raw) != row["png_sha256"]:
            raise ValueError("image identity changed")
        if not isinstance(row["text_prompt"], str) or not row["text_prompt"]:
            raise ValueError("invalid prompt")
        threshold = row["box_threshold"]
        if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
            raise ValueError("invalid threshold")
        with Image.open(io.BytesIO(raw)) as image:
            pixels = np.asarray(image.convert("RGB"))
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("image encoding failed")
        image_bytes = encoded.tobytes()
        # requests.post(json=...) uses these json.dumps defaults.
        payload = json.dumps({"image": base64.b64encode(image_bytes).decode(),
                              "text_prompt": row["text_prompt"],
                              "box_threshold": threshold}, allow_nan=False).encode()
        requests.append(payload)
        identities.append({"png_sha256": row["png_sha256"],
                           "jpeg_sha256": sha(image_bytes),
                           "request_sha256": sha(payload),
                           "request_bytes": len(payload),
                           "box_threshold": threshold,
                           "prompt_sha256": sha(row["text_prompt"].encode())})
    return requests, identities


def exchange(endpoint, path, token, payload=None, timeout=180):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(endpoint + path, data=payload, headers=headers)
    start = time.monotonic()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("response too large")
        pins = response.headers.get_all("X-Detector-Model-Sha256", [])
        if len(pins) > 1:
            raise ValueError("duplicate model pin")
        return raw, dict(response.headers), time.monotonic() - start


def identity(endpoint, pin, token):
    raw, _, _ = exchange(endpoint, "/identity", token)
    value = json.loads(raw)
    if (value.get("enabled") is not True or sha(canonical(value["model"])) != pin
            or value.get("model_sha256") != pin):
        raise ValueError("gateway model identity mismatch")
    return {"endpoint": endpoint, "model_sha256": pin, "identity_sha256": sha(raw)}


def compare(reference, candidate):
    """Compare actual detections, retaining raw response hashes separately."""
    if reference.get("success") is not True or candidate.get("success") is not True:
        raise ValueError("benchmark request did not produce validated success")
    left, right = reference["detections"], candidate["detections"]
    if len(left) != len(right):
        return {"equal": False, "reason": "detection_count"}
    max_box_delta = max_score_delta = 0.0
    for a, b in zip(left, right):
        if a.get("label") != b.get("label"):
            return {"equal": False, "reason": "labels_or_order"}
        if len(a["bbox"]) != 4 or len(b["bbox"]) != 4:
            raise ValueError("invalid detection box")
        max_box_delta = max(max_box_delta, *(abs(x - y) for x, y in zip(a["bbox"], b["bbox"])))
        max_score_delta = max(max_score_delta, abs(a["confidence"] - b["confidence"]))
    return {"equal": left == right, "max_box_delta": max_box_delta,
            "max_score_delta": max_score_delta, "detections": len(left)}


def run(args):
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    preflight_path = destination / "preflight.json"
    result_path = destination / "results.json"
    if preflight_path.exists() or result_path.exists():
        raise ValueError("benchmark output already exists")
    payloads, inputs = read_requests(args.manifest)
    token = os.environ.get("HARNESS3D_DETECTOR_TOKEN", "")
    identities = [identity(endpoint, args.model_sha256, token) for endpoint in args.endpoint]
    preflight = {"schema": "detector-request-benchmark-preflight/1", "at": time.time(),
                 "manifest_sha256": sha(Path(args.manifest).read_bytes()),
                 "inputs": inputs, "endpoints": identities, "concurrency": args.concurrency,
                 "label_access": False, "requests_sent": 0}
    preflight_path.write_bytes(canonical(preflight))
    results = []
    reference = None
    for endpoint in args.endpoint:
        current = []

        def infer(index):
            raw, headers, elapsed = exchange(endpoint, "/infer", token, payloads[index])
            if headers.get("X-Detector-Model-Sha256") != args.model_sha256:
                raise ValueError("inference response model pin mismatch")
            value = json.loads(raw)
            # Also validates a legitimate empty response.
            compare(value, value)
            return {"index": index, "request_sha256": inputs[index]["request_sha256"],
                    "response_sha256": sha(raw), "elapsed_s": elapsed,
                    "transport_headers": {k: v for k, v in headers.items()
                                          if k.lower().startswith("x-detector-")},
                    "value": value}

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            current = list(executor.map(infer, range(len(payloads))))
        elapsed = time.monotonic() - start
        if reference is None:
            reference = [item["value"] for item in current]
        for item, original in zip(current, reference):
            item["comparison"] = compare(original, item.pop("value"))
        results.append({"endpoint": endpoint, "wall_s": elapsed,
                        "successful_requests": len(current),
                        "requests_per_s": len(current) / elapsed,
                        "p50_s": statistics.median(r["elapsed_s"] for r in current),
                        "p95_s": sorted(r["elapsed_s"] for r in current)[
                            max(0, int(.95 * len(current) + .9999) - 1)],
                        "requests": current})
    result = {"schema": "detector-request-benchmark-results/1", "at": time.time(),
              "preflight_sha256": sha(preflight_path.read_bytes()), "results": results,
              "label_access": False, "retries": 0,
              "limitation": "Same-input output/throughput evidence; no accuracy measurement."}
    result_path.write_bytes(canonical(result))
    print(json.dumps({"status": "finished", "results_file": str(result_path),
                      "endpoints": len(results), "requests_per_endpoint": len(payloads)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--concurrency", type=int, default=8, choices=range(1, 17))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Credential-bearing URLs and redirects must never be accepted.
    from urllib.parse import urlsplit
    for endpoint in args.endpoint:
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or not parsed.port or parsed.username or parsed.password
                or parsed.path or parsed.query or parsed.fragment):
            parser.error("benchmark requires credential-free loopback gateway URLs")
    if len(args.model_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.model_sha256):
        parser.error("invalid model SHA256")
    run(args)


if __name__ == "__main__":
    main()
