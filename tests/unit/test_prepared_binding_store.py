"""Contracts for immutable question-binding storage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from skill3d.online.runner import PreparedObjectBinding
from skill3d.schemas import ObjectRecord
from skill3d.segmentation.prepared_binding_store import (
    PreparedBindingStore,
    PreparedBindingStoreError,
)


def _request(input_sha256: str) -> dict:
    body = {
        "schema_version": "test-request-v1",
        "input_sha256": input_sha256,
        "qa_id": "qa-1",
        "scene_name": "scene-1",
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **body,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _binding(root: Path) -> PreparedObjectBinding:
    mask = root / "mask.npy"
    points = root / "points.npy"
    confidence = root / "confidence.npy"
    np.save(mask, np.ones((2, 3, 4), dtype=np.uint8))
    np.save(points, np.ones((8, 3), dtype=np.float32))
    np.save(confidence, np.ones((8,), dtype=np.float32))
    obj = ObjectRecord(
        obj_id="obj-1",
        category_name="chair",
        grounding_status="question_driven",
        mask_per_frame=str(mask),
        pointcloud_world=str(points),
        pointconf_world=str(confidence),
    )
    return PreparedObjectBinding(
        objects=[obj],
        stats={
            "generation": {"sha256": "g" * 64},
            "diagnostic": np.arange(4, dtype=np.float32),
        },
        notes=["prepared once"],
        materialized=True,
    )


def test_store_replays_binding_without_copying_scene_geometry(tmp_path):
    geometry = tmp_path / "scene-depth.npy"
    np.save(geometry, np.zeros((2, 3, 4), dtype=np.float32))
    scene = tmp_path / "artifact.json"
    scene.write_text(
        json.dumps({"depth_maps": str(geometry)}),
        encoding="utf-8",
    )
    scene_identity = {"sha256": "i" * 64}
    request = _request(scene_identity["sha256"])
    store = PreparedBindingStore(tmp_path / "store")

    reference = store.publish(
        request_identity=request,
        scene_artifact=scene,
        scene_input_identity=scene_identity,
        binding=_binding(tmp_path),
    )
    loaded = store.load(
        reference["key"],
        expected_request_identity=request,
        scene_artifact=scene,
        scene_input_identity=scene_identity,
    )

    assert loaded.materialized
    assert loaded.notes == ["prepared once"]
    assert np.array_equal(
        loaded.stats["diagnostic"],
        np.arange(4, dtype=np.float32),
    )
    assert all(
        Path(getattr(loaded.objects[0], field)).is_relative_to(
            store.entry_path(reference["key"])
        )
        for field in (
            "mask_per_frame",
            "pointcloud_world",
            "pointconf_world",
        )
    )
    stored_names = {
        path.name
        for path in store.entry_path(reference["key"]).rglob("*")
        if path.is_file()
    }
    assert geometry.name not in stored_names
    assert scene.name not in stored_names


def test_store_rejects_changed_scene_or_binding_payload(tmp_path):
    scene = tmp_path / "artifact.json"
    scene.write_text("{}", encoding="utf-8")
    scene_identity = {"sha256": "i" * 64}
    request = _request(scene_identity["sha256"])
    store = PreparedBindingStore(tmp_path / "store")
    reference = store.publish(
        request_identity=request,
        scene_artifact=scene,
        scene_input_identity=scene_identity,
        binding=_binding(tmp_path),
    )

    scene.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(
        PreparedBindingStoreError,
        match="scene artifact identity changed",
    ):
        store.load(
            reference["key"],
            expected_request_identity=request,
            scene_artifact=scene,
            scene_input_identity=scene_identity,
        )

    scene.write_text("{}", encoding="utf-8")
    payload = next(
        path
        for path in store.entry_path(reference["key"]).rglob("*.npy")
    )
    payload.write_bytes(b"changed")
    with pytest.raises(
        PreparedBindingStoreError,
        match="payload identity changed",
    ):
        store.load(
            reference["key"],
            expected_request_identity=request,
            scene_artifact=scene,
            scene_input_identity=scene_identity,
        )
