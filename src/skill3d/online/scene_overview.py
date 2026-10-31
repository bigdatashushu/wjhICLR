"""Read precomputed scene-layout images independently of the frozen FrameSet."""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import threading

import numpy as np
from PIL import Image

OVERVIEW_SCHEMA = "harness3d-scene-overview/1"
MANIFEST_SCHEMA = "harness3d-vsi1000-scene-overviews/1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SCENE = re.compile(r"[A-Za-z0-9_-]+\Z")
_CACHE = OrderedDict()
_LOCK = threading.RLock()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode())


def _read(path):
    path = Path(path)
    _require(path.is_absolute() and ".." not in path.parts, "overview requires an absolute path")
    _require(not any(parent.is_symlink() for parent in (path, *path.parents)),
             "overview path contains a symbolic link")
    return path.read_bytes()


def read_overview_image(descriptor, *, scene_id=None):
    """Check actual PNG bytes every time; cache only decoding, never integrity."""
    _require(isinstance(descriptor, dict) and descriptor.get("schema") == OVERVIEW_SCHEMA,
             "unknown overview descriptor")
    declared_scene = descriptor.get("scene_id")
    _require(isinstance(declared_scene, str) and _SCENE.fullmatch(declared_scene),
             "invalid overview scene")
    _require(scene_id is None or scene_id == declared_scene, "overview belongs to another scene")
    for name in ("png_sha256", "rgb_sha256", "scene_artifact_sha256",
                 "frozen_scene_sha256", "render_spec_sha256"):
        _require(isinstance(descriptor.get(name), str) and _SHA.fullmatch(descriptor[name]),
                 "overview identity requires SHA256")
    _require(descriptor.get("resolution_wh") == [640, 480]
             and descriptor.get("render_only_cutaway") is True,
             "overview rendering contract changed")
    _require(descriptor.get("image_id") == (
        f"overview-{declared_scene}-{descriptor['rgb_sha256'][:12]}"), "overview image ID changed")
    _require(descriptor.get("world_frame_status") in {"available", "degraded", "unavailable"},
             "unknown overview orientation status")
    for name in ("selected_azimuth_degrees", "elevation_degrees"):
        _require(type(descriptor.get(name)) in (int, float)
                 and math.isfinite(descriptor[name]), "invalid overview camera angle")
    raw = _read(descriptor["image_path"])
    _require(len(raw) <= 4 * 1024 * 1024 and _sha(raw) == descriptor["png_sha256"],
             "overview PNG content changed")
    key = (descriptor["png_sha256"], descriptor["rgb_sha256"])
    with _LOCK:
        if key not in _CACHE:
            with Image.open(io.BytesIO(raw)) as image:
                _require(image.format == "PNG" and image.mode == "RGB" and image.size == (640, 480),
                         "overview image dimensions or encoding changed")
                pixels = image.tobytes()
            _require(_sha(pixels) == descriptor["rgb_sha256"], "overview RGB content changed")
            _CACHE[key] = np.frombuffer(pixels, dtype=np.uint8).reshape(480, 640, 3)
            while len(_CACHE) > 4:
                _CACHE.popitem(last=False)
        _CACHE.move_to_end(key)
        return _CACHE[key]


class SceneOverviewStore:
    """Small immutable manifest; disk paths are separate from content identity."""

    def __init__(self, manifest_path, expected_sha256=None):
        self.path = Path(manifest_path).absolute()
        raw = _read(self.path)
        self.manifest_sha256 = _sha(raw)
        _require(expected_sha256 is None or expected_sha256 == self.manifest_sha256,
                 "overview manifest hash changed")
        manifest = json.loads(raw)
        _require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("status") == "complete",
                 "overview manifest is not complete")
        spec = manifest.get("render_spec")
        _require(isinstance(spec, dict) and _digest(spec) == manifest.get("render_spec_sha256")
                 and spec.get("resolution_wh") == [640, 480],
                 "overview rendering identity changed")
        rows = manifest.get("images")
        _require(isinstance(rows, list) and len(rows) == manifest.get("n_scenes") and bool(rows),
                 "overview manifest scene denominator changed")
        self._rows = {}
        for row in rows:
            scene = row.get("scene_id")
            _require(isinstance(scene, str) and _SCENE.fullmatch(scene)
                     and scene not in self._rows, "duplicate or invalid overview scene")
            _require(row.get("file") == f"{scene}.png" and row.get("n_frozen_frames") == 32,
                     "overview filename or frame scope changed")
            for name in ("png_sha256", "rgb_sha256", "scene_artifact_sha256", "frozen_scene_sha256"):
                _require(isinstance(row.get(name), str) and _SHA.fullmatch(row[name]),
                         "overview source identity missing")
            self._rows[scene] = copy.deepcopy(row)
        self.render_spec_sha256 = manifest["render_spec_sha256"]
        self._identity = {
            "schema": MANIFEST_SCHEMA, "manifest_sha256": self.manifest_sha256,
            "render_spec_sha256": self.render_spec_sha256,
            "n_scenes": len(self._rows),
            "scene_images_sha256": _digest({scene: {
                name: row[name] for name in ("png_sha256", "rgb_sha256",
                                             "scene_artifact_sha256", "frozen_scene_sha256")}
                for scene, row in self._rows.items()}),
        }

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def descriptor(self, scene_id, *, entry=None):
        _require(_sha(_read(self.path)) == self.manifest_sha256, "overview manifest content changed")
        row = self._rows.get(scene_id)
        _require(row is not None, "scene overview is missing")
        if entry is not None:
            _require(entry.get("scene_id") == scene_id
                     and entry.get("scene_artifact") == row["scene_artifact"]
                     and entry.get("frozen_scene_sha256") == row["frozen_scene_sha256"],
                     "overview and published scene identity differ")
            artifact = Path(entry["scene_artifact"])
            if artifact.is_file():
                _require(_sha(_read(artifact)) == row["scene_artifact_sha256"],
                         "overview reconstruction artifact changed")
        descriptor = {
            "schema": OVERVIEW_SCHEMA, "scene_id": scene_id,
            "image_id": f"overview-{scene_id}-{row['rgb_sha256'][:12]}",
            "image_path": str(self.path.parent / row["file"]),
            "render_spec_sha256": self.render_spec_sha256,
            **{name: copy.deepcopy(row[name]) for name in (
                "png_sha256", "rgb_sha256", "scene_artifact_sha256", "frozen_scene_sha256",
                "world_frame_status", "resolution_wh", "selected_azimuth_degrees",
                "elevation_degrees", "render_only_cutaway")},
        }
        read_overview_image(descriptor, scene_id=scene_id)
        return descriptor

    def validate_entries(self, entries):
        _require(isinstance(entries, dict) and bool(entries), "published overview scope is empty")
        scenes = {entry["scene_id"] for entry in entries.values()}
        _require(scenes == set(self._rows), "overview scene set differs from published inputs")
        checked = {}
        for entry in entries.values():
            scene = entry["scene_id"]
            signature = (entry["scene_artifact"], entry["frozen_scene_sha256"])
            if scene in checked:
                _require(checked[scene] == signature, "published scene has conflicting identities")
            else:
                self.descriptor(scene, entry=entry)
                checked[scene] = signature
        return {"n_scenes": len(checked), "n_questions": len(entries), "model_requests": 0}
