"""Keep historical prompt bytes, image count and terminal failures reproducible."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from skill3d.small_image.cv3d_protocol import LEGACY_VERSION, FrozenCV3DSkills
from skill3d.small_image.cv3d_protocol_v1 import build_compact_prompt
from skill3d.small_image.runner import SmallImageSolver
from test_cv3d_compact_program import config, prepared, prompt
from test_small_image_runner import ANSWER, YIELD, FakeClient, problem


@pytest.mark.parametrize("finalizing,expected", [
    (False, "ce783c4adf89792d9caba651b9d092fc908ddbf839071b62e098d8394a39f975"),
    (True, "c880e39c66c1ec8b4dc2211531160a141833a6ea2512a23c7e43d566fb2f5b9c"),
])
def test_v1_prompt_matches_original_bf1ccbd_bytes(finalizing, expected):
    p = SimpleNamespace(
        benchmark="cv3d", task_type="Depth", question="Which target is closer?",
        options=("red", "blue"), image_paths=("fixture.png",))
    text = build_compact_prompt(
        p, [np.zeros((8, 10, 3), dtype=np.uint8)],
        official_prompt=p.question + "\n(A) red\n(B) blue",
        skill_text="# Frozen skill\nCompare surfaces.\n",
        feedback=[{"result_id": "r1", "data": {"depth": 1.25}}],
        interfaces="- point_depth(view_index, x, y): frozen interface",
        remaining_requests=1, finalizing=finalizing)
    assert sha256(text.encode()).hexdigest() == expected


def test_frozen_v1_executes_six_original_rgb_requests_and_keeps_old_feedback(tmp_path):
    root = Path(__file__).resolve().parents[1]
    frozen = FrozenCV3DSkills(
        root / "skill_library/candidates/cv3d-compact-testdev-1/frozen_skills.json")
    p = problem(tmp_path)
    client = FakeClient([YIELD] * 5 + ['{"answer":"B"}'])
    result = SmallImageSolver(
        client, config=config(prompt_profile=LEGACY_VERSION, send_depth_image=False),
        skill_text=frozen.texts[p.task_type], official_prompt=prompt(p))(p, prepared=prepared(p))
    assert frozen.prompt_profile == result.prompt_version == LEGACY_VERSION
    assert result.status == "answered" and result.request_count == 6
    assert result.category_requests == 0 and result.succeeded_tool_calls == 5
    for messages, _ in client.calls:
        content = messages[0]["content"]
        assert len([part for part in content if part["type"] == "image_url"]) == 1
        text = content[0]["text"]
        assert frozen.texts[p.task_type] in text
        assert "cv3d-context/2" not in text and "PREDICTED DEPTH MAP" not in text
    assert '"source_tool": "view_depth_summary"' in client.calls[1][0][0]["content"][0]["text"]


def test_v1_runtime_error_remains_terminal_and_depth_input_is_rejected(tmp_path):
    p = problem(tmp_path)
    client = FakeClient([
        "def solve(ctx):\n    r = ctx.tools.point_depth(0, 5, 5)\n"
        "    x = r.status\n    YieldObservations([])", ANSWER])
    result = SmallImageSolver(
        client, config=config(prompt_profile=LEGACY_VERSION, send_depth_image=False),
        skill_text="Compare surfaces.", official_prompt=prompt(p))(p, prepared=prepared(p))
    assert result.status == "run_error" and result.request_count == 1
    with pytest.raises(ValueError, match="v2 only"):
        config(prompt_profile=LEGACY_VERSION, send_depth_image=True)
