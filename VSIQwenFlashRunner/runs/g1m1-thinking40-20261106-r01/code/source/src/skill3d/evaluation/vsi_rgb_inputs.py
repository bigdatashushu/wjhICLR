"""Replay only the frozen chronological RGB frames for the direct VSI arm."""

from __future__ import annotations

import copy
import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.adapters.frame_set import frame_set_hash, uniform_frame_ids, with_pixel_identity
from skill3d.cluster import scene_compute
from skill3d.evaluation.published_fitset_v11 import PublishedFitSetInputProvider, _require
from skill3d.schemas import VSIBenchEpisode


class FrozenRGBInputProvider(PublishedFitSetInputProvider):
    """Keep catalog anchors while never opening geometry, masks or bindings."""

    def __call__(self, variant):
        entry = self._entry(variant)
        root = scene_compute._root(Path(entry["input_root"]))
        path = scene_compute._local_file(root, "scene.json")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == entry["frozen_scene_sha256"],
                 "Frozen RGB scene identity changed")
        key = (str(root), entry["frozen_scene_sha256"])
        with self.cache_lock:
            if key not in self.cache:
                manifest = scene_compute._read_json(path)
                _require(path.read_bytes() == raw, "Frozen scene changed during reading")
                scene_compute._reject_labels(manifest)
                _require(manifest.get("schema") == scene_compute.FROZEN_SCENE_SCHEMA
                         and manifest.get("status") == "frozen"
                         and manifest.get("real_labels_available") is False
                         and manifest.get("scene_id") == entry["scene_id"]
                         and manifest.get("split") == entry["split"],
                         "Frozen RGB source scope changed")
                frames, files = manifest.get("frames"), manifest.get("files")
                _require(isinstance(frames, list) and len(frames) == 32
                         and isinstance(files, dict), "Frozen RGB needs 32 frames")
                pixels, sources = [], []
                for row in frames:
                    source = scene_compute._reference(row["source_path"], absolute=True)
                    frame = scene_compute._local_file(root, files[source]).read_bytes()
                    _require(len(frame) == row["size"]
                             and hashlib.sha256(frame).hexdigest() == row["png_sha256"],
                             "Frozen PNG changed")
                    with Image.open(BytesIO(frame)) as image:
                        _require(image.format == "PNG" and image.mode == "RGB",
                                 "Expected original RGB PNG")
                        array = np.array(image)
                    _require(list(array.shape) == row["shape"]
                             and str(array.dtype) == row["dtype"] == "uint8"
                             and array.ndim == 3 and array.shape[2] == 3
                             and hashlib.sha256(array.tobytes()).hexdigest() == row["rgb_sha256"],
                             "Frozen RGB pixels changed")
                    array.setflags(write=False)
                    pixels.append(array)
                    sources.append(source)
                _require(len(set(sources)) == 32
                         and len({files[source] for source in sources}) == 32,
                         "Duplicate frozen RGB frame")
                self.cache[key] = (manifest, pixels, sources)
                while len(self.cache) > self.max_scenes:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            manifest, pixels, sources = self.cache[key]
            questions = {row["episode"]["qa_id"]: row for row in manifest["questions"]}
            _require(set(questions) == set(manifest["qa_ids"])
                     and len(questions) == len(manifest["questions"]),
                     "Frozen public question identity changed")
            public = questions[variant["qa_id"]]
            _require("ground_truth" not in public["episode"] and public["source"] == "vsi_bench",
                     "Direct episode must be label free")
            episode = VSIBenchEpisode.model_validate({**copy.deepcopy(public["episode"]),
                                                       "ground_truth": ""})
            fs = episode.frame_set
            _require(episode.scene_name == variant["scene_id"]
                     and episode.question_type == variant["question_type"]
                     and episode.split == entry["split"]
                     and fs is not None and fs.n_frames == 32 and fs.decode_status == "ok"
                     and fs.frame_ids == uniform_frame_ids(fs.n_total_frames, 32)
                     and fs.frame_set_hash == frame_set_hash(fs.frame_ids)
                     and fs.source_frame_indices == fs.frame_ids
                     and (not fs.frame_refs or fs.frame_refs == sources)
                     and fs.pixel_sha256 == with_pixel_identity(fs, pixels).pixel_sha256,
                     "Direct frame/question identity changed")
            return EpisodeItem(episode, list(pixels), source=public["source"],
                               video_path=public["video_path"])
