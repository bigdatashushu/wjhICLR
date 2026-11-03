import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

from CV3Drunner.grounding import AnnotationCatalog
from fixtures import rgb_image, RECTS


def catalog(image, task="Distance", question="red box blue box green box"):
    return AnnotationCatalog(np.asarray(image.convert("RGB")), question, task)


def test_complete_rectangle_edges_ignore_unrelated_natural_color_pixels():
    image = rgb_image()
    draw = ImageDraw.Draw(image)
    draw.ellipse((160, 85, 174, 105), fill=(240, 10, 10))
    draw.line((2, 2, 70, 2), fill="red", width=3)
    found = catalog(image)
    assert found.measurements_available
    assert found.boxes["v0:red"].xyxy == RECTS["red"]
    assert all(row["surface_validated"] is False for row in found.public()["boxes"])


def test_jpeg_strokes_are_extracted_without_rescaling():
    stream = io.BytesIO()
    rgb_image().save(stream, format="JPEG", quality=95)
    found = catalog(Image.open(io.BytesIO(stream.getvalue())))
    assert found.measurements_available
    for color, expected in RECTS.items():
        assert max(abs(a - b) for a, b in zip(found.boxes[f"v0:{color}"].xyxy, expected)) <= 2


def test_missing_and_ambiguous_required_boxes_withhold_measurements():
    image = rgb_image()
    ImageDraw.Draw(image).rectangle((15, 72, 115, 115), fill=(60, 65, 70))
    missing = catalog(image)
    assert missing.status["green"] == "missing" and not missing.measurements_available
    image = rgb_image()
    ImageDraw.Draw(image).rectangle((125, 75, 170, 110), outline=(255, 0, 0), width=3)
    ambiguous = catalog(image)
    assert ambiguous.status["red"] == "ambiguous" and not ambiguous.measurements_available
    assert "v0:red" not in ambiguous.boxes


def test_open_colored_shape_is_not_an_annotation_box():
    image = Image.new("RGB", (180, 120), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.line((20, 20, 60, 20, 60, 58, 20, 58), fill="red", width=3)
    assert catalog(image).status["red"] == "missing"


def test_question_color_order_is_not_bound_to_option_letters():
    a = catalog(rgb_image(), question="green box red box blue box")
    b = catalog(rgb_image(), question="blue box green box red box")
    assert [r["color"] for r in a.public()["question_bindings"]] == ["green", "red", "blue"]
    assert a.identity != b.identity
    assert all("option" not in r for r in a.public()["question_bindings"])
    assert not catalog(rgb_image(), "Depth", "Compare those two objects").measurements_available


def test_real_000775_boxes_exclude_the_failed_original_pixels():
    path = Path("/tmp/cv3d-disagreement-000775-rgb.png")
    if not path.exists():
        pytest.skip("optional public original RGB fixture is not installed")
    found = catalog(Image.open(path))
    assert found.measurements_available
    expected = {"red": (140, 398, 265, 435), "blue": (125, 255, 353, 379),
                "green": (32, 381, 350, 517)}
    for color, rect in expected.items():
        assert max(abs(a - b) for a, b in zip(found.boxes[f"v0:{color}"].xyxy, rect)) <= 2
    for color, xy in (("red", (200, 537)), ("blue", (200, 420)), ("green", (250, 600))):
        x0, y0, x1, y1 = found.boxes[f"v0:{color}"].safe_xyxy
        assert not (x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1)
