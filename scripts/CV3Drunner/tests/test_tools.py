import numpy as np
import pytest

from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.small_image.tools import build_registry as original_registry
from skill3d.tools.contract import ConfidenceGateError, DomainValueError
from CV3Drunner.tools import build_registry, allowed_tools, MEASUREMENTS
from fixtures import fixture, rgb_image


def test_box_conversion_stroke_margin_and_numerical_equivalence(tmp_path):
    public, _, prepared = fixture(tmp_path, task="Distance")
    registry, handle = build_registry(prepared, public.problem)
    old, old_handle = original_registry(prepared, public.problem)
    box, xy = handle.box_pixel(0, "v0:red", [.5, .5])
    r = registry.get("point_depth_in_box").fn(handle, 0, "v0:red", .5, .5)
    expected = old.get("point_depth").fn(old_handle, 0, *xy)
    assert r["depth"] == expected["depth"] and r["n_valid"] == expected["n_valid"]
    b, bxy = handle.box_pixel(0, "v0:blue", [.4, .6])
    distance = registry.get("pair_distance_in_boxes").fn(
        handle, 0, "v0:red", [.5, .5], "v0:blue", [.4, .6])
    raw = old.get("pair_point_distance").fn(old_handle, 0, xy, bxy)
    assert distance["distance"] == raw["distance"]
    for uv in ([0, 0], [1, 1], [.5, .5]):
        _, point = handle.box_pixel(0, "v0:red", uv)
        x, y = (round(v) for v in point)
        x0, y0, x1, y1 = box.xyxy
        assert x - 2 > x0 + box.stroke_px - 1 and x + 2 < x1 - box.stroke_px + 1
        assert y - 2 > y0 + box.stroke_px - 1 and y + 2 < y1 - box.stroke_px + 1
    assert not distance["surface_validated"]


@pytest.mark.parametrize("uv", [[-1, .5], [1.1, .5], [True, .5], [float("nan"), .5],
                                [float("inf"), .5], ["0.5", .5], [1], None])
def test_invalid_uv_never_reaches_geometry(tmp_path, uv):
    public, _, prepared = fixture(tmp_path)
    _, handle = build_registry(prepared, public.problem)
    with pytest.raises(DomainValueError):
        handle.box_pixel(0, "v0:red", uv)


def test_color_binding_view_and_ambiguous_gates(tmp_path):
    public, _, prepared = fixture(tmp_path)
    registry, handle = build_registry(prepared, public.problem)
    assert "pair_distance_in_boxes" not in allowed_tools(registry, handle)
    with pytest.raises(ConfidenceGateError):
        handle.box_pixel(0, "v0:green", [.5, .5])
    with pytest.raises(DomainValueError):
        handle.box_pixel(1, "v0:red", [.5, .5])
    from PIL import ImageDraw
    image = rgb_image()
    ImageDraw.Draw(image).rectangle((125, 75, 170, 110), outline="red", width=3)
    public, _, prepared = fixture(tmp_path / "ambiguous", image=image)
    registry, handle = build_registry(prepared, public.problem)
    assert not (allowed_tools(registry, handle) & MEASUREMENTS)
    kernel = RestrictedNamespaceKernel(registry, handle, frames=list(prepared.frames))
    cell = kernel.run_cell("def solve(ctx):\n    ctx.tools.point_depth_in_box(0, 'v0:red', .5, .5)")
    assert cell.answer_untrusted and cell.error_code == "tool_contract"
    assert kernel.authorization_receipts[-1]["allowed"] is False


def test_local_registry_has_no_raw_bypass_and_does_not_mutate_shared_tools(tmp_path):
    public, _, prepared = fixture(tmp_path)
    shared_before, _ = original_registry(prepared, public.problem)
    registry, handle = build_registry(prepared, public.problem)
    allowed = allowed_tools(registry, handle)
    assert not ast_guard(
        "def solve(ctx):\n    r=ctx.tools.point_depth(0, 10, 10)\n    ReturnAnswer('A')",
        allowed_tools=allowed).ok
    assert not ast_guard(
        "def solve(ctx):\n    r=ctx.tools.pair_point_distance(0, [10,10], [20,20])",
        allowed_tools=allowed).ok
    shared_after, _ = original_registry(prepared, public.problem)
    assert shared_before.names() == shared_after.names()
    assert not set(registry.names()) & set(shared_after.names()) - {"view_depth_summary"}
    before = prepared.frames[0].copy()
    handle.frames[0][0, 0] = [255, 0, 0]
    np.testing.assert_array_equal(before, prepared.frames[0])
