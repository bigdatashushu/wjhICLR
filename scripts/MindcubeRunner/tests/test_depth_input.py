from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from MindcubeRunner.data import digest_bytes
from MindcubeRunner.depth_input import DEPTH_LEGEND, PALETTE, depth_preview


def render(depth, mask=None, *, relative_only=True):
    depth = np.asarray(depth, dtype=float)
    mask = np.ones(depth.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    raw, caption, receipt = depth_preview(
        {"depth": depth, "valid_mask": mask, "relative_only": relative_only},
        view_index=2, width=depth.shape[1], height=depth.shape[0])
    with Image.open(BytesIO(raw)) as image:
        assert image.format == "PNG" and image.size == (depth.shape[1], depth.shape[0])
        pixels = np.asarray(image).copy()
    assert receipt["image_sha256"] == digest_bytes(raw)
    json.dumps(receipt, allow_nan=False)
    return raw, pixels, caption, receipt


def test_depth_grid_orientation_and_quantile_clipping():
    yy, xx = np.indices((7, 11))
    depth = 1 + 3 * xx + yy
    _, pixels, caption, receipt = render(depth)
    assert tuple(pixels[0, 0]) == PALETTE[0]
    assert tuple(pixels[-1, -1]) == PALETTE[-1]
    assert receipt["status"] == "available"
    assert [receipt["display_min"], receipt["display_max"]] == pytest.approx(
        np.quantile(depth, [0.02, 0.98]))
    # Equal Z at different coordinates must produce the same color without shifting the grid.
    assert depth[3, 0] == depth[0, 1]
    np.testing.assert_array_equal(pixels[3, 0], pixels[0, 1])
    assert not np.array_equal(pixels[0, -1], pixels[-1, 0])
    assert "Original Image 3" in caption and "view_index=2" in caption
    assert "PNG 11x7" in caption and "x=0..10" in caption and "y=0..6" in caption
    assert "no resize, crop, rotation or border" in DEPTH_LEGEND
    assert "2%-98%" in caption and receipt["units"] == "per_view_relative"


def test_depth_mask_and_nonfinite_or_nonpositive_pixels_are_black():
    depth = np.array([[1, np.nan, np.inf, -np.inf], [0, -3, 3, 9]], dtype=float)
    mask = np.ones(depth.shape, dtype=bool)
    mask[1, 3] = False
    _, pixels, caption, receipt = render(depth, mask)
    valid = np.zeros(depth.shape, dtype=bool)
    valid[0, 0] = valid[1, 2] = True
    assert (pixels[~valid] == 0).all() and (pixels[valid] != 0).all()
    assert tuple(pixels[0, 0]) == PALETTE[0] and tuple(pixels[1, 2]) == PALETTE[-1]
    assert receipt["valid_pixel_count"] == 2 and receipt["total_pixel_count"] == 8
    assert receipt["valid_depth_min"] == 1 and receipt["valid_depth_max"] == 3
    assert "Black means invalid" in DEPTH_LEGEND


def test_constant_depth_uses_midpoint_and_explains_no_contrast():
    depth = np.full((3, 5), 7.0)
    mask = np.ones(depth.shape, dtype=bool)
    mask[1, 2] = False
    _, pixels, caption, receipt = render(depth, mask)
    assert (pixels[mask] == PALETTE[2]).all() and (pixels[~mask] == 0).all()
    assert receipt["status"] == "no_contrast"
    assert receipt["display_min"] == receipt["display_max"] == 7
    assert "status=no_contrast" in caption
    assert "constant teal" in DEPTH_LEGEND and "no depth contrast" in DEPTH_LEGEND


@pytest.mark.parametrize("masked_out", [False, True])
def test_no_valid_depth_is_explicitly_unavailable(masked_out):
    depth = (np.full((2, 3), 7.0) if masked_out else
             np.array([[np.nan, np.inf, -np.inf], [0, -1, -2]]))
    mask = np.full(depth.shape, not masked_out, dtype=bool)
    _, pixels, caption, receipt = render(depth, mask)
    assert not pixels.any()
    assert receipt["status"] == "unavailable" and receipt["valid_pixel_count"] == 0
    assert receipt["display_min"] is receipt["display_max"] is None
    assert receipt["valid_depth_min"] is receipt["valid_depth_max"] is None
    assert "status=unavailable" in caption and "no valid finite positive depth" in caption
    assert "all-black map supplies no depth evidence" in DEPTH_LEGEND


def test_each_view_uses_its_own_relative_display_range():
    depth = np.arange(1, 101, dtype=float).reshape(10, 10)
    _, first, caption, a = render(depth)
    _, second, _, b = render(depth * 8 + 64)
    np.testing.assert_array_equal(first, second)
    assert b["display_min"] == pytest.approx(a["display_min"] * 8 + 64)
    assert b["display_max"] == pytest.approx(a["display_max"] * 8 + 64)
    assert "colors cannot be compared across views" in DEPTH_LEGEND
    assert "point_depth" not in caption


def test_rendering_is_deterministic_and_preserves_source_arrays():
    depth = np.arange(1, 49, dtype=float).reshape(6, 8)
    mask = depth % 5 != 0
    original_depth, original_mask = depth.copy(), mask.copy()
    depth.setflags(write=False)
    mask.setflags(write=False)
    first = render(depth, mask)
    second = render(depth, mask)
    assert first[0] == second[0] and first[2:] == second[2:]
    np.testing.assert_array_equal(depth, original_depth)
    np.testing.assert_array_equal(mask, original_mask)
    assert not depth.flags.writeable and not mask.flags.writeable


@pytest.mark.parametrize("field", ["depth", "valid_mask"])
def test_wrong_grid_cannot_be_rendered(field):
    geometry = {"depth": np.ones((3, 5)), "valid_mask": np.ones((3, 5), dtype=bool),
                "relative_only": True}
    geometry[field] = geometry[field].T
    with pytest.raises(ValueError, match="depth_preview_grid_changed"):
        depth_preview(geometry, view_index=0, width=5, height=3)


def test_metric_geometry_cannot_be_described_as_relative():
    with pytest.raises(ValueError, match="depth_preview_relative_geometry_required"):
        render(np.ones((3, 5)), relative_only=False)
