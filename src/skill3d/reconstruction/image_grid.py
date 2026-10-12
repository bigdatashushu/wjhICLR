"""VGGT crop preprocessing shared by metric depth and original-frame masks.

This is the feed-forward VGGT grid, not the retired BA square-pad contract.
Original RGB frames remain unchanged for the solver and SAM2.
"""

from __future__ import annotations

import numpy as np

GRID_VERSION = "vggt-crop-518-patch14-v1"


def grid_layout(shapes) -> list[dict]:
    """Describe the vendored load_and_preprocess_images(mode='crop') operation."""
    rows = []
    for shape in shapes:
        h, w = map(int, shape[:2])
        if min(h, w) <= 0:
            raise ValueError("Empty image grid")
        resized_h = round(h * (518 / w) / 14) * 14
        if resized_h <= 0:
            raise ValueError("Image aspect ratio produces an empty VGGT grid")
        rows.append({
            "original_hw": [h, w], "resized_hw": [resized_h, 518],
            "crop_top": max(0, (resized_h - 518) // 2),
            "content_hw": [min(resized_h, 518), 518],
        })
    if not rows:
        raise ValueError("No images")
    height = max(row["content_hw"][0] for row in rows)
    for row in rows:
        row.update(version=GRID_VERSION, output_hw=[height, 518],
                   pad_top=(height - row["content_hw"][0]) // 2)
    return rows


def apply_grid(array: np.ndarray, layout: dict, *, mask: bool = False) -> np.ndarray:
    """Resize, crop and batch-pad in the same order as the VGGT loader."""
    from PIL import Image

    a = np.asarray(array)
    if list(a.shape[:2]) != layout["original_hw"] or layout["version"] != GRID_VERSION:
        raise ValueError("Input does not match the frozen VGGT grid")
    if not mask and (a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3):
        raise ValueError("Frozen RGB must be uint8 H,W,3")
    interpolation = Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC
    h, w = layout["resized_hw"]
    resized = np.asarray(Image.fromarray(a.astype(np.uint8) if mask else a).resize(
        (w, h), interpolation))
    top, content_h = layout["crop_top"], layout["content_hw"][0]
    cropped = resized[top:top + content_h]
    result = np.full((*layout["output_hw"], *a.shape[2:]), 0 if mask else 255,
                     dtype=np.uint8)
    pad = layout["pad_top"]
    result[pad:pad + content_h] = cropped
    return result.astype(bool) if mask else result


def prepare_metric_frames(frames) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    layouts = grid_layout([np.asarray(frame).shape for frame in frames])
    rgb = [apply_grid(frame, row) for frame, row in zip(frames, layouts)]
    valid = [apply_grid(np.ones(row["original_hw"], dtype=bool), row, mask=True)
             for row in layouts]
    return rgb, valid, layouts


def mask_on_vggt_grid(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Map an original SAM2 mask, including crop and mixed-height batch padding."""
    row = grid_layout([mask.shape])[0]
    target = list(map(int, shape))
    if target[1] != 518 or target[0] < row["content_hw"][0] or target[0] > 518:
        raise ValueError("Target is not a compatible VGGT crop grid")
    row.update(output_hw=target, pad_top=(target[0] - row["content_hw"][0]) // 2)
    return apply_grid(mask, row, mask=True)
