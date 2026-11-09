"""Decode already verified immutable inputs; never rerun publication checks.

These capabilities are for the trusted framework, outside generated-code tools.
The publication catalog and exact physical transport descriptor remain the
identity authority. Transport verifies incoming bytes before granting a lease.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace


def load_verified_scene(root, *, question_cap):
    import numpy as np
    from PIL import Image
    from skill3d.adapters.episode_source import EpisodeItem
    from skill3d.cluster import scene_compute
    from skill3d.schemas import VSIBenchEpisode

    root = scene_compute._root(root)
    manifest = scene_compute._read_json(scene_compute._local_file(root, "scene.json"))
    # Label isolation remains a consumer boundary, independent of byte auditing.
    scene_compute._reject_labels(manifest)
    if (manifest.get("schema") != scene_compute.FROZEN_SCENE_SCHEMA
            or manifest.get("status") != "frozen"
            or manifest.get("real_labels_available") is not False):
        raise scene_compute.SceneComputeError("verified replay requires a public frozen scene")
    if len(manifest["qa_ids"]) > question_cap:
        raise scene_compute.SceneComputeError(f"scene question cap {question_cap} exceeded")
    pixels, png_bytes = [], []
    for frame in manifest["frames"]:
        path = scene_compute._local_file(root, manifest["files"][frame["source_path"]])
        raw = path.read_bytes()
        png_bytes.append(raw)
        with Image.open(BytesIO(raw)) as image:
            pixels.append(np.array(image))
    items = [
        EpisodeItem(
            VSIBenchEpisode.model_validate({**question["episode"], "ground_truth": ""}),
            pixels, source=question["source"], video_path=question["video_path"])
        for question in manifest["questions"]
    ]
    return SimpleNamespace(root=root, manifest=manifest, items=items,
                           png_bytes=tuple(png_bytes))


@dataclass(frozen=True)
class ProducerVerifiedBindingContext:
    """One decoded binding from the trusted provider; never inferred from stats."""

    binding: object
    request: dict

    def validate(self, binding, *, inputs, scene_identity, seed, parameters,
                 model_weights_sha256=""):
        if binding is not self.binding:
            raise ValueError("verified binding differs from the provider's decoded object")
        identity = binding.stats["generation"]["identity"]
        if identity["question"] != inputs or identity["reconstruction_identity"] != scene_identity:
            raise ValueError("verified binding question/scene differs from the active episode")
        if parameters != self.request["parameters"] or (
                seed is not None and seed != self.request["seed"]):
            raise ValueError("verified binding runtime parameters/seed changed")
        if model_weights_sha256 and (
                model_weights_sha256 != self.request["model"]["model_weights_sha256"]):
            raise ValueError("verified binding model weights changed")
        receipt = binding.stats.get("binding_derivation")
        if receipt is not None and receipt["destination"]["request_identity"] != self.request:
            raise ValueError("verified derived binding destination changed")
