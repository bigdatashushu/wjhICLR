"""Exact JPEG reuse with synthetic pixels and mocked, local-only transport."""

import base64
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import DEFAULT, Mock

import cv2
import numpy as np
import pytest
import requests

from skill3d.infra.m5_identity import content_sha256
from skill3d.segmentation import open_vocab_detector as ovd


_IMENCODE = cv2.imencode
_ENDPOINT = "http://detector.invalid"


def _frame():
    return np.random.default_rng(137).integers(0, 256, (13, 17, 3), dtype=np.uint8)


def _legacy_body(frame, prompt="fixture.", threshold=ovd.DEFAULT_BOX_THRESHOLD):
    img = np.asarray(frame)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    ok, buf = _IMENCODE(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                       [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    assert ok
    return {"image": base64.b64encode(buf.tobytes()).decode("ascii"),
            "text_prompt": prompt, "box_threshold": float(threshold)}


def _response(detections=None):
    response = Mock(status_code=200)
    response.json.return_value = {"success": True, "detections": detections or []}
    return response


@pytest.fixture
def client(monkeypatch):
    with ovd._jpeg_cache_lock:
        ovd._jpeg_cache.clear()
        ovd._jpeg_cache_bytes = 0
    post = Mock(return_value=_response())
    encode = Mock(wraps=_IMENCODE)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(cv2, "imencode", encode)
    yield SimpleNamespace(post=post, encode=encode)
    with ovd._jpeg_cache_lock:
        ovd._jpeg_cache.clear()
        ovd._jpeg_cache_bytes = 0


def _detect(frame, prompt="fixture.", **kwargs):
    with ovd.capture_detector_failures() as failures, ovd.capture_detector_attempts() as attempts:
        ovd.detect(frame, prompt, endpoint=_ENDPOINT, **kwargs)
    assert not failures and not ovd.detect.last_error
    assert len(attempts) == 1
    return attempts[0]


@pytest.mark.parametrize("variant", [
    "rgb", "float", "signed", "gray", "gray_float", "rgba", "fortran", "strided",
])
def test_jpeg_and_request_hash_match_legacy_on_miss_and_hit(client, variant):
    frame = _frame()
    if variant == "float":
        frame = frame.astype(np.float32) * 2 - 100.5
    elif variant == "signed":
        frame = frame.astype(np.int16) * 2 - 100
    elif variant == "gray":
        frame = frame[:, :, 0]
    elif variant == "gray_float":
        frame = frame[:, :, 0].astype(np.float64) * 2 - 100.5
    elif variant == "rgba":
        frame = np.concatenate([frame, frame[:, :, :1]], axis=-1)
    elif variant == "fortran":
        frame = np.asfortranarray(frame)
    elif variant == "strided":
        frame = frame[::-1, ::2, :]
    prompt = "fixture \u00e9."
    expected = _legacy_body(frame, prompt)
    first = _detect(frame, prompt)
    second = _detect(frame.copy(), prompt)
    assert client.encode.call_count == 1
    assert client.post.call_count == 2
    assert not first["jpeg_cache_hit"] and second["jpeg_cache_hit"]
    for call, attempt in zip(client.post.call_args_list, [first, second]):
        body = call.kwargs["json"]
        assert body == expected
        assert set(body) == {"image", "text_prompt", "box_threshold"}
        assert attempt["request_sha256"] == content_sha256(expected)
        wire_body = requests.Request("POST", _ENDPOINT, json=body).prepare().body
        assert attempt["request_bytes"] == len(wire_body)
        assert attempt["encode_ms"] >= 0
    assert not first["recovered"] and not second["recovered"]


def test_equal_normalized_pixels_reuse_encoding_across_dtype_and_layout(client):
    gray = _frame()[:, :, 0]
    rgb = np.stack([gray] * 3, axis=-1)
    frames = [gray, rgb, rgb.astype(np.float32), np.asfortranarray(rgb)]
    attempts = [_detect(frame) for frame in frames]
    assert [row["jpeg_cache_hit"] for row in attempts] == [False, True, True, True]
    assert len({row["request_sha256"] for row in attempts}) == 1
    assert client.encode.call_count == 1


def test_prompt_threshold_and_endpoint_change_request_but_not_encoding(client):
    frame = _frame()
    first = _detect(frame)
    second = _detect(frame, "other fixture.", box_threshold=.6)
    with ovd.capture_detector_attempts() as attempts:
        ovd.detect(frame, "fixture.", endpoint="http://other-detector.invalid")
    assert client.encode.call_count == 1 and client.post.call_count == 3
    assert second["jpeg_cache_hit"] and attempts[0]["jpeg_cache_hit"]
    assert first["request_sha256"] != second["request_sha256"]
    assert first["request_sha256"] == attempts[0]["request_sha256"]
    assert client.post.call_args_list[1].kwargs["json"] == _legacy_body(
        frame, "other fixture.", .6)
    assert client.post.call_args_list[2].args == ("http://other-detector.invalid/infer",)


def test_content_mutation_misses_and_restoring_content_hits(client):
    frame = _frame()
    original = frame.copy()
    first = _detect(frame)
    frame[:] = 0
    changed = _detect(frame)
    assert client.post.call_args.kwargs["json"] == _legacy_body(frame)
    frame[:] = original
    restored = _detect(frame)
    assert [row["jpeg_cache_hit"] for row in [first, changed, restored]] == [False, False, True]
    assert changed["request_sha256"] != first["request_sha256"]
    assert restored["request_sha256"] == first["request_sha256"]
    assert client.encode.call_count == 2


def test_shape_is_part_of_encoding_key_even_with_identical_pixel_bytes(client):
    frame = _frame()
    reshaped = frame.reshape(17, 13, 3)
    assert frame.tobytes() == reshaped.tobytes()
    first = _detect(frame)
    second = _detect(reshaped)
    assert client.encode.call_count == 2
    assert not second["jpeg_cache_hit"]
    assert second["request_sha256"] != first["request_sha256"]
    assert client.post.call_args.kwargs["json"] == _legacy_body(reshaped)


def test_caller_mutation_during_encoding_cannot_poison_cached_snapshot(client):
    frame = _frame()
    original = frame.copy()

    def mutate_caller(*args, **kwargs):
        frame[:] = 0
        return _IMENCODE(*args, **kwargs)

    client.encode.side_effect = mutate_caller
    first = _detect(frame)
    assert client.post.call_args.kwargs["json"] == _legacy_body(original)
    client.encode.side_effect = None
    changed = _detect(frame)
    restored = _detect(original)
    assert not first["jpeg_cache_hit"] and not changed["jpeg_cache_hit"]
    assert restored["jpeg_cache_hit"] and restored["request_sha256"] == first["request_sha256"]
    assert client.encode.call_count == 2
    assert all(isinstance(value, bytes) for value in ovd._jpeg_cache.values())


def test_encoder_settings_are_part_of_cache_key(client, monkeypatch):
    frame = _frame()
    first = _detect(frame)
    monkeypatch.setattr(cv2, "COLOR_RGB2BGR", cv2.COLOR_RGB2GRAY)
    second = _detect(frame)
    assert client.encode.call_count == 2 and not second["jpeg_cache_hit"]
    assert first["request_sha256"] != second["request_sha256"]


def test_entry_bound_evicts_least_recently_used_encoding(client, monkeypatch):
    monkeypatch.setattr(ovd, "JPEG_CACHE_MAX_ENTRIES", 2)
    a, b, c = [np.full((13, 17, 3), value, np.uint8) for value in [0, 50, 100]]
    attempts = [_detect(frame) for frame in [a, b, a, c, a, b]]
    assert [row["jpeg_cache_hit"] for row in attempts] == [False, False, True, False, True, False]
    assert client.encode.call_count == 4
    assert len(ovd._jpeg_cache) == 2
    assert ovd._jpeg_cache_bytes == sum(map(len, ovd._jpeg_cache.values()))


def test_byte_bound_evicts_encodings_without_changing_requests(client, monkeypatch):
    a, b = _frame(), np.zeros((13, 17, 3), np.uint8)
    bodies = [_legacy_body(frame) for frame in [a, b]]
    limit = sum(len(base64.b64decode(body["image"])) for body in bodies) - 1
    monkeypatch.setattr(ovd, "JPEG_CACHE_MAX_BYTES", limit)
    for frame in [a, b, a]:
        assert not _detect(frame)["jpeg_cache_hit"]
        assert ovd._jpeg_cache_bytes <= limit
        assert ovd._jpeg_cache_bytes == sum(map(len, ovd._jpeg_cache.values()))
    assert client.encode.call_count == 3 and len(ovd._jpeg_cache) == 1
    assert client.post.call_args.kwargs["json"] == bodies[0]


@pytest.mark.parametrize("bound", ["entries", "bytes"])
def test_disabled_or_oversized_cache_entry_still_sends_exact_jpeg(client, monkeypatch, bound):
    monkeypatch.setattr(ovd, "JPEG_CACHE_MAX_ENTRIES" if bound == "entries"
                        else "JPEG_CACHE_MAX_BYTES", 0 if bound == "entries" else 1)
    frame = _frame()
    assert not _detect(frame)["jpeg_cache_hit"]
    assert not _detect(frame)["jpeg_cache_hit"]
    assert client.encode.call_count == 2
    assert not ovd._jpeg_cache and ovd._jpeg_cache_bytes == 0
    assert client.post.call_args.kwargs["json"] == _legacy_body(frame)


@pytest.mark.parametrize("failure", ["false", "exception"])
def test_encoding_failures_never_enter_cache_or_send_requests(client, failure):
    frame = _frame()
    client.encode.side_effect = [
        (False, None) if failure == "false" else cv2.error("fixture encoding failure"),
        DEFAULT,
    ]
    with ovd.capture_detector_failures() as failures:
        if failure == "false":
            assert ovd.detect(frame, "fixture.", endpoint=_ENDPOINT) == []
            assert failures == ["JPEG encoding failed"]
        else:
            with pytest.raises(cv2.error, match="fixture encoding failure"):
                ovd.detect(frame, "fixture.", endpoint=_ENDPOINT)
    assert client.post.call_count == 0
    assert not ovd._jpeg_cache and ovd._jpeg_cache_bytes == 0
    assert not _detect(frame)["jpeg_cache_hit"]
    assert client.encode.call_count == 2


def test_transport_failure_does_not_cache_a_detection_result(client):
    client.post.side_effect = [requests.Timeout("fixture timeout"), _response([
        {"label": "fixture", "bbox": [1, 1, 10, 10], "confidence": .8},
    ])]
    frame = _frame()
    with ovd.capture_detector_failures() as failures, ovd.capture_detector_attempts() as attempts:
        assert ovd.detect(frame, "fixture.", endpoint=_ENDPOINT, max_attempts=1) == []
    assert len(failures) == 1 and attempts[0]["status"] == "error"
    with ovd.capture_detector_attempts() as recovered:
        result = ovd.detect(frame, "fixture.", endpoint=_ENDPOINT, max_attempts=1)
    assert len(result) == 1 and not ovd.detect.last_error
    assert recovered[0]["status"] == "ok" and recovered[0]["jpeg_cache_hit"]
    assert recovered[0]["request_sha256"] == attempts[0]["request_sha256"]
    assert recovered[0]["request_bytes"] == attempts[0]["request_bytes"]
    assert client.encode.call_count == 1 and client.post.call_count == 2


def test_retry_attempts_share_preparation_metrics_without_changing_retry_body(client, monkeypatch):
    monkeypatch.setattr(ovd.time, "sleep", lambda _: None)
    ticks = iter([1.0, 1.0125])
    monkeypatch.setattr(ovd.time, "perf_counter", lambda: next(ticks))
    client.post.side_effect = [requests.Timeout("fixture timeout"), _response()]
    frame = _frame()
    with ovd.capture_detector_failures() as failures, ovd.capture_detector_attempts() as outer:
        with ovd.capture_detector_attempts() as inner:
            assert ovd.detect(frame, "fixture.", endpoint=_ENDPOINT) == []
    assert not failures and outer == inner and len(inner) == 2
    for row in inner:
        assert row["encode_ms"] == pytest.approx(12.5)
        assert not row["jpeg_cache_hit"]
        assert row["request_sha256"] == content_sha256(_legacy_body(frame))
        wire_body = requests.Request("POST", _ENDPOINT, json=_legacy_body(frame)).prepare().body
        assert row["request_bytes"] == len(wire_body)
    assert inner[0]["will_retry"] and inner[1]["recovered"]
    assert client.encode.call_count == 1
    assert client.post.call_args_list[0].kwargs["json"] == client.post.call_args_list[1].kwargs["json"]


def test_concurrent_same_content_admission_keeps_byte_accounting_bounded(client):
    barrier = Barrier(2)

    def encode(*args, **kwargs):
        barrier.wait(timeout=5)
        return _IMENCODE(*args, **kwargs)

    client.encode.side_effect = encode
    frame = _frame()
    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = list(pool.map(_detect, [frame, frame.copy()]))
    assert not any(row["jpeg_cache_hit"] for row in attempts)
    assert len({row["request_sha256"] for row in attempts}) == 1
    assert len(ovd._jpeg_cache) == 1
    assert ovd._jpeg_cache_bytes == sum(map(len, ovd._jpeg_cache.values()))
    assert ovd._jpeg_cache_bytes <= ovd.JPEG_CACHE_MAX_BYTES
    client.encode.side_effect = None
    assert _detect(frame)["jpeg_cache_hit"]
    assert client.encode.call_count == 2
