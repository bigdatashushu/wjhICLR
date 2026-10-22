import base64
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

SPEC = importlib.util.spec_from_file_location(
    "detector_benchmark", Path(__file__).resolve().parents[2] /
    "scripts/benchmark_cluster_detector.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def manifest(tmp_path):
    pixels = np.arange(18 * 20 * 3, dtype=np.uint8).reshape(18, 20, 3)
    image = tmp_path / "image.png"
    Image.fromarray(pixels).save(image)
    row = {"image": image.name, "png_sha256": benchmark.sha(image.read_bytes()),
           "text_prompt": "chair.", "box_threshold": .25}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema": "detector-request-benchmark/1",
                                "requests": [row, row]}))
    return path, pixels


def test_compiled_jpeg_matches_original_client(tmp_path):
    path, pixels = manifest(tmp_path)
    payloads, identities = benchmark.read_requests(path)
    ok, expected = cv2.imencode(".jpg", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    assert base64.b64decode(json.loads(payloads[0])["image"]) == expected.tobytes()
    assert payloads[0] == payloads[1]
    assert identities[0]["request_sha256"] == benchmark.sha(payloads[0])


def test_tamper_is_rejected_before_any_identity_or_inference(tmp_path, monkeypatch):
    path, _ = manifest(tmp_path)
    (tmp_path / "image.png").write_bytes(b"changed")
    monkeypatch.setattr(benchmark, "identity", lambda *args: pytest.fail("network called"))
    with pytest.raises(ValueError, match="image identity"):
        benchmark.run(SimpleNamespace(manifest=path, output=tmp_path / "output",
                                      endpoint=["http://127.0.0.1:123"],
                                      model_sha256="a" * 64, concurrency=2))


def test_two_endpoint_results_retain_comparison_without_extra_requests(tmp_path, monkeypatch):
    path, _ = manifest(tmp_path)
    output = tmp_path / "output"
    pin = "a" * 64
    calls = []
    monkeypatch.setattr(benchmark, "identity", lambda endpoint, pin, token:
                        {"endpoint": endpoint, "model_sha256": pin})

    def exchange(endpoint, route, token, payload):
        assert json.loads((output / "preflight.json").read_text())["requests_sent"] == 0
        calls.append((endpoint, route, payload))
        confidence = .7 if endpoint.endswith(":123") else .70001
        return json.dumps({"success": True, "detections": [
            {"label": "chair", "bbox": [1, 2, 3, 4], "confidence": confidence}
        ]}).encode(), {"X-Detector-Model-Sha256": pin}, .01

    monkeypatch.setattr(benchmark, "exchange", exchange)
    benchmark.run(SimpleNamespace(manifest=path, output=output,
                                  endpoint=["http://127.0.0.1:123", "http://127.0.0.1:124"],
                                  model_sha256=pin, concurrency=2))
    result = json.loads((output / "results.json").read_bytes())
    assert len(calls) == 4
    assert all(row["comparison"]["equal"] for row in result["results"][0]["requests"])
    assert all(not row["comparison"]["equal"] for row in result["results"][1]["requests"])
    assert all("value" not in row for group in result["results"] for row in group["requests"])
    assert result["results"][1]["requests"][0]["comparison"]["max_score_delta"] > 0


def test_empty_success_and_errors_are_distinct():
    empty = {"success": True, "detections": []}
    assert benchmark.compare(empty, empty)["equal"]
    with pytest.raises(ValueError, match="validated success"):
        benchmark.compare(empty, {"success": False, "detections": []})
