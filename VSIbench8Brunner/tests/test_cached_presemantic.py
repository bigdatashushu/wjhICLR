from types import SimpleNamespace

import pytest

from skill3d.infra.m5_identity import content_sha256
from skill3d.segmentation.cached_presemantic import CachedPresemantic
from skill3d.segmentation.question_binding import _nms


def replay():
    detections = [
        SimpleNamespace(label="chair", bbox_xyxy=[0, 0, 100, 100], confidence=.9),
        SimpleNamespace(label="chair", bbox_xyxy=[1, 1, 101, 101], confidence=.8),
        SimpleNamespace(label="chair", bbox_xyxy=[120, 0, 220, 100], confidence=.7),
    ]
    rows, kept = _nms(detections, "chair", 0, "a" * 64)
    identity = {"question": {}, "frames": [], "geometry": {}, "frame_set_hash": "x",
                "reconstruction_identity": "y", "grid_transform": None,
                "sam2": {"checkpoint": "z", "config": {}, "settings": {}}}
    stats = {
        "generation": {"sha256": content_sha256(identity), "identity": identity},
        "detection_receipts": [{"category": "chair", "frame_slot": 0,
                                "request_sha256": "a" * 64, "candidates": rows}],
        "candidate_receipts": kept,
        "review_selection": {"selected_candidate_ids": [kept[1]["candidate_id"]]},
    }
    return CachedPresemantic(SimpleNamespace(stats=stats), {}), kept


def test_replays_nms_and_preserves_original_review_scope():
    cache, expected = replay()
    cache.validate({"identity": cache.stats["generation"]["identity"]})
    requirements = [{"name": "chair"}]
    actual, _ = cache.detections(["chair"], [0], requirements)
    assert actual == expected
    assert len(actual) == 2
    assert requirements[0]["scan_complete"]
    observations = [{"candidate": row} for row in actual]
    assert cache.select(observations) == [observations[1]]


def test_changed_boxes_or_incomplete_scan_are_rejected():
    cache, _ = replay()
    cache.stats["detection_receipts"][0]["candidates"][1]["bbox"] = [300, 0, 400, 100]
    with pytest.raises(ValueError, match="NMS ledger"):
        cache.detections(["chair"], [0], [{"name": "chair"}])
    cache, _ = replay()
    with pytest.raises(ValueError, match="incomplete"):
        cache.detections(["chair"], [0, 1], [{"name": "chair"}])
