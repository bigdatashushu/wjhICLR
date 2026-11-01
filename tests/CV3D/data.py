"""Official prompt lookup and verified, existing precompute replay. No model calls."""
from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from skill3d.small_image.datasets import SmallImageProblem, load_manifest, load_problems
from skill3d.small_image.storage import load_prepared

from .common import digest, read_json


@dataclass(frozen=True)
class PublicInput:
    problem: SmallImageProblem
    official_prompt: str

    def identity(self):
        return {
            "problem": self.problem.public_dict(),
            "official_prompt_sha256": hashlib.sha256(
                self.official_prompt.encode("utf-8")).hexdigest(),
        }


def official_prompts(official_annotations):
    official = {}
    with Path(official_annotations).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "3D":
                raise ValueError("official annotation contains a non-3D row")
            task, question, choices, prompt = (
                row["task"], row["question"], row["choices"], row["prompt"])
            if (task not in {"Depth", "Distance"} or not isinstance(question, str)
                    or not isinstance(choices, list) or len(choices) != 2
                    or any(not isinstance(c, str) or not c for c in choices)
                    or not isinstance(prompt, str) or not prompt):
                raise ValueError("invalid official public fields")
            # Verify choices against the source field, without rebuilding the request text.
            encoded = question + "\n" + "\n".join(
                f"({label}) {choice}" for label, choice in zip("AB", choices))
            if prompt != encoded:
                raise ValueError("official prompt does not match question and choices")
            key = (task, question, tuple(choices))
            if key in official and official[key] != prompt:
                raise ValueError("ambiguous official prompt")
            official[key] = prompt
            # The answer and other nonpublic fields are deliberately not retained.
    return official


def load_inputs(dataset_root, official_annotations, *, split="test", limit=None,
                sample_ids=()) -> list[PublicInput]:
    """Read only public official fields; duplicate image questions share prompt text."""
    official = official_prompts(official_annotations)
    problems = load_problems(dataset_root, "cv3d", split)
    if sample_ids:
        requested = set(sample_ids)
        if len(requested) != len(sample_ids):
            raise ValueError("duplicate selected sample ID")
        problems = [p for p in problems if p.sample_id in requested]
        if {p.sample_id for p in problems} != requested:
            raise ValueError("selected sample ID is outside the requested split")
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be positive")
        problems = problems[:limit]
    if not problems:
        raise ValueError("no selected CV3D inputs")
    inputs = []
    for p in problems:
        if (len(p.image_paths) != 1 or len(p.options) != 2
                or p.task_type not in {"Depth", "Distance"}):
            raise ValueError("CV3D requires one image and two choices")
        key = (p.task_type, p.question, p.options)
        if key not in official:
            raise ValueError("author split question does not match official annotations")
        inputs.append(PublicInput(p, official[key]))
    return inputs


def replay_inputs(public_manifest, dataset_root, official_annotations):
    official = official_prompts(official_annotations)
    output = []
    for p in load_manifest(public_manifest, dataset_root):
        if p.benchmark != "cv3d":
            continue
        key = (p.task_type, p.question, p.options)
        if key not in official or len(p.image_paths) != 1:
            raise ValueError("replay input differs from official CV3D prompt")
        output.append(PublicInput(p, official[key]))
    return output


def index_prepared(root) -> dict[str, Path]:
    """Index metadata only. Conflicting artifacts require explicit resolution."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError("prepared root must exist")
    index, identities = {}, {}
    for path in sorted(root.rglob("metadata.json")):
        metadata = read_json(path)
        if metadata.get("schema") != "small-image-prepared-storage/1":
            continue
        key, artifact = metadata["input_sha256"], metadata["artifact_sha256"]
        if key in identities and identities[key] != artifact:
            raise ValueError("multiple different prepared artifacts for one input")
        index[key], identities[key] = path.parent, artifact
    return index


def verified_prepared(public_input: PublicInput, directory):
    problem = public_input.problem
    if replace(problem).input_sha256 != problem.input_sha256:
        raise ValueError("original image bytes changed")
    prepared = load_prepared(directory, problem=problem)
    with Image.open(problem.image_paths[0]) as image:
        rgb = np.asarray(image.convert("RGB"))
    if (len(prepared.frames) != 1 or not np.array_equal(rgb, prepared.frames[0])
            or 0 not in prepared.geometries):
        raise ValueError("prepared RGB/depth does not match the original input")
    geometry = prepared.geometries[0]
    if (geometry["depth"].shape != rgb.shape[:2]
            or geometry["valid_mask"].shape != rgb.shape[:2]):
        raise ValueError("depth grid differs from original image")
    if problem.task_type == "Distance":
        points = geometry.get("points")
        if points is None or points.shape != (*rgb.shape[:2], 3):
            raise ValueError("Distance requires the original-grid predicted point map")
    return prepared


def render_depth(depth, valid_mask) -> tuple[bytes, dict]:
    """Linear 2%-98% grayscale display; raw numerical tool data is never modified."""
    depth = np.asarray(depth)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    if depth.ndim != 2 or valid.shape != depth.shape or not np.any(valid):
        raise ValueError("no valid predicted depth")
    lo, hi = map(float, np.quantile(depth[valid], [0.02, 0.98]))
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if hi > lo:
        gray[valid] = np.rint(
            np.clip((depth[valid] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    else:
        gray[valid] = 128
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    rgb[~valid] = [255, 0, 255]
    output = io.BytesIO()
    Image.fromarray(rgb).save(output, format="PNG")
    png = output.getvalue()
    return png, {
        "display": "linear_grayscale", "lower_quantile": 0.02, "upper_quantile": 0.98,
        "lower_depth": lo, "upper_depth": hi, "invalid_color": [255, 0, 255],
        "width": depth.shape[1], "height": depth.shape[0],
        "png_sha256": hashlib.sha256(png).hexdigest(),
        "raw_depth_sha256": digest({
            "dtype": depth.dtype.str, "shape": list(depth.shape),
            "bytes_sha256": hashlib.sha256(depth.tobytes(order="C")).hexdigest(),
        }),
    }


def _image_part(data, mime):
    return {"type": "image_url", "image_url": {
        "url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}}


def build_messages(public_input, depth_png, *, width, height, tool_instructions):
    """One user message: exact HF prompt, tool manual, original image, depth image."""
    original_path = Path(public_input.problem.image_paths[0])
    original = original_path.read_bytes()
    with Image.open(io.BytesIO(original)) as image:
        mime = Image.MIME.get(image.format)
        if image.size != (width, height) or mime not in {"image/png", "image/jpeg"}:
            raise ValueError("unsupported original image format or changed dimensions")
    text = public_input.official_prompt + "\n\n" + tool_instructions
    return [{"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "text", "text": (
            f"Image 1: ORIGINAL RGB IMAGE, {width} x {height} pixels, view_index=0.")},
        _image_part(original, mime),
        {"type": "text", "text": (
            f"Image 2: PREDICTED DEPTH MAP of Image 1, {width} x {height} pixels. "
            "It uses the same pixel grid. Darker grayscale means smaller camera Z "
            "(nearer); lighter means farther. Display values are clipped to the "
            "2nd-98th percentiles. Magenta pixels have invalid depth. "
            "This is relative MoGe2 prediction; numerical tools read the raw values.")},
        _image_part(depth_png, "image/png"),
    ]}]


def build_direct_messages(public_input):
    problem = public_input.problem
    if replace(problem).input_sha256 != problem.input_sha256:
        raise ValueError("original image bytes changed")
    data = Path(problem.image_paths[0]).read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        mime = Image.MIME.get(image.format)
    if mime not in {"image/png", "image/jpeg"}:
        raise ValueError("unsupported original image format")
    return [{"role": "user", "content": [
        {"type": "text", "text": public_input.official_prompt},
        _image_part(data, mime),
    ]}]
