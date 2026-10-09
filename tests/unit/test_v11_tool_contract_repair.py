"""Reproduce the audited contract failures with actual tools and Skill code."""

import json
from pathlib import Path
import re

import numpy as np
import pytest

from skill3d.online.runner import OnlineRunConfig, _yield_feedback
from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.tools import REGISTRY, call_tool
from skill3d.tools.docs_v11 import DESCRIPTIONS
from skill3d.synthesis.prompt_builder import PromptBuilder
from test_geometry_tools import _cluster, _handle, _obj, _scene

LIBRARY = Path(__file__).resolve().parents[2] / "skill_library"


@pytest.mark.parametrize(("tracks", "centers", "expected", "merges"), [
    (["same", "same"], [(0, 0, 0), (10, 0, 0)], 1, 0),
    (["same", "same", "other"], [(0, 0, 0), None, (10, 0, 0)], 2, 0),
    (["same", "same", "a", "b"], [None, None, (0, 0, 0), (0, 0, 0)], 2, 1),
    (["a", "b"], [(0, 0, 0), (0, 0, 0)], 1, 1),
    ([None, None], [(0, 0, 0), (10, 0, 0)], 2, 0),
])
def test_track_groups_are_never_split_by_geometry(tracks, centers, expected, merges):
    objs = [
        _obj(f"v11-{i}", "chair", (0, 0, 0), track=track,
             pts=None if center is None else _cluster(center))
        for i, (track, center) in enumerate(zip(tracks, centers))
    ]
    result = call_tool("count_objects", {"category_name": "chair"},
                       _handle(_scene(objs=objs), objs=objs))
    assert result.error is None
    data = json.loads(result.value)
    assert data["count"] == expected
    assert data["n_geometric_merges"] == merges
    assert data["count"] == data["n_distinct_tracks"] - data["n_geometric_merges"]
    assert data["n_records"] == len(tracks)


@pytest.mark.parametrize(("evidence", "up", "hand", "valid"), [
    ("available", [0, 2, 0], "right", True),
    ("degraded", [1, 0, 0], "left", True),
    ("unavailable", [0, 1, 0], "right", False),
    ("available", None, "right", False),
    ("available", [0, 0, 0], "right", False),
    ("available", [0, 1, 0], "invalid", False),
])
def test_centroid_direction_metadata_requires_evidence_and_real_values(evidence, up, hand, valid):
    obj = _obj("route-point", "chair", (1, 2, 3))
    scene = _scene(objs=[obj])
    scene.evidence_profile.world_frame = evidence
    scene.artifact = scene.artifact.model_copy(update={"world_up": up, "handedness": hand})
    handle = _handle(scene, objs=[obj], metric_scale=2.0)
    result = call_tool("object_centroid", {"obj_id": obj.obj_id}, handle)
    assert result.error is None
    data = json.loads(result.value)
    assert data["centroid_normalized"] == [1, 2, 3]
    assert data["centroid_metric"] is None  # scale exists but this question has no authorization
    assert (data["world_up_used"] is not None) == valid
    assert data["handedness_used"] == (hand if valid else None)
    if valid:
        assert np.linalg.norm(data["world_up_used"]) == pytest.approx(1.0)


@pytest.fixture
def turn_angle():
    text = (LIBRARY / "versions/S08/1.2.0/infer-route-turns/SKILL.md").read_text()
    code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    check = ast_guard(code + "\nReturnAnswer(1)\n", allowed_tools=set())
    assert check.ok, check.violations
    namespace = {}
    exec(code, namespace)
    return namespace["route_turn_angle"]


@pytest.mark.parametrize(("following", "expected"), [
    ((-1, 0, 1), 90), ((1, 0, 1), -90), ((0, 0, 2), 0), ((0, 0, 0), 180),
])
@pytest.mark.parametrize("mirrored", [False, True])
def test_skill_turn_example_is_rotation_and_handedness_invariant(turn_angle, following, expected, mirrored):
    from scipy.spatial.transform import Rotation
    transform = Rotation.from_rotvec([0.6, -0.9, 0.4]).as_matrix()
    if mirrored:
        transform = transform @ np.diag([-1.0, 1.0, 1.0])
    points = [np.array([0, 0, 0]), np.array([0, 0, 1]), np.array(following)]
    up = transform @ np.array([0, 1, 0])
    # Unequal landmark heights must not alter horizontal turns.
    coords = [transform @ p + up * height
              for p, height in zip(points, [4.0, -2.0, 3.0])]
    actual = turn_angle(*coords, up, "left" if mirrored else "right")
    if expected == 180:
        assert abs(actual) == pytest.approx(180)
    else:
        assert actual == pytest.approx(expected, abs=1e-8)


def test_skill_turn_example_rejects_missing_frame_or_vertical_segment(turn_angle):
    assert turn_angle([0, 0, 0], [0, 0, 1], [-1, 0, 1], None, "right") is None
    assert turn_angle([0, 0, 0], [0, 2, 0], [-1, 2, 0], [0, 1, 0], "right") is None
    assert turn_angle([0, 0, 0], [0, 0, 1], [-1, 0, 1], [0, 1, 0], None) is None


def test_control_examples_run_in_real_kernel():
    obj = _obj("control-object", "chair", (0, 0, 0), track="track")
    scene = _scene(objs=[obj], question_type="object_counting")
    handle = _handle(scene, objs=[obj])
    full = (
        "return ReturnAnswer(AnswerPayload(value=2, unit='count', "
        "basis='visual_estimate', used_result_ids=[], derivation=None))")
    kernel = RestrictedNamespaceKernel(REGISTRY, handle)
    assert ast_guard(full, allowed_tools=set()).ok
    cell = kernel.run_cell(full)
    assert cell.error is None
    assert kernel.answer_slot.payload.basis == "visual_estimate"
    assert kernel.answer_slot.answer == "2"

    request = 'return YieldObservations([result["result_id"]], "理由")'
    kernel = RestrictedNamespaceKernel(REGISTRY, handle)
    cell = kernel.run_cell('result = count_objects("chair")\n' + request)
    assert cell.error is None
    assert cell.yielded_result_ids == [kernel.tool_results[-1].result_id]
    assert '"count": 1' in _yield_feedback(cell, kernel, scene, OnlineRunConfig())

    request_all = 'return YieldObservations([], "理由")'
    kernel = RestrictedNamespaceKernel(REGISTRY, handle)
    cell = kernel.run_cell('records = list_objects("chair")\n'
                           'distance = euclidean_distance([0,0,0], [3,4,0])\n'
                           'show(records)\nshow(distance)\n' + request_all)
    assert cell.error is None
    assert isinstance(kernel.show_log[0], list)
    assert kernel.show_log[1] == 5
    kernel.tool_results[0].invalidated_by = "test-revocation"
    feedback = _yield_feedback(cell, kernel, scene, OnlineRunConfig())
    assert all(result.result_id in feedback for result in kernel.tool_results)
    assert "已被级联撤销" in feedback


def test_neutral_docs_cover_current_tools_without_old_recipes():
    assert set(DESCRIPTIONS) == set(REGISTRY.names())
    text = REGISTRY.docs()
    for stale in ("theta_deg", "方向题优先", "计数题必须", "用法照抄",
                  "外观顺序题", "最长边、以厘米计", "官方口径实现", "请改用"):
        assert stale not in text
    assert "candidate_categories" in text
    assert "centroid_normalized" in text and "world_up_used" in text
    assert "不保证已排序或去重" in text
    text = PromptBuilder().render(
        question="test", scene_summary="", scene_frame="world", tool_docs=text,
        scope="full_3d", question_type="object_rel_distance")
    assert "尺度 s 下自动约掉" not in text
    assert "AnswerPayload.value" in text


def test_repair_snapshot_preserves_original_sources_and_records_ancestry():
    from skill3d.skills.v11_library import validate_v11_snapshot, canonical_json_sha256
    original = json.loads((LIBRARY / "snapshots/snapshot_S0-v11-format-migration.json").read_text())
    revised = json.loads((LIBRARY / "snapshots/snapshot_S0-v11-contract-repair.json").read_text())
    old = {s.skill_id: s for s in validate_v11_snapshot(original, library_root=LIBRARY)}
    new = {s.skill_id: s for s in validate_v11_snapshot(revised, library_root=LIBRARY)}
    assert set(old) == set(new) and len(new) == 8
    assert original["manifest_hash"] == "992c7624e2af429df8efc63b22a2b5eaf82ba3a53772d0f5cde19bf839a54a8b"
    assert revised["parent_snapshot_id"] == original["snapshot_id"]
    assert revised["generation"] == 0
    for sid, spec in new.items():
        entry = revised["entries"][f"{sid}@{spec.version}"]
        if sid in ("S01", "S08"):
            assert spec.version == "1.2.0"
            assert entry["parent_version"] == f"{sid}@1.1.0"
            assert spec.content_sha256 != old[sid].content_sha256
        else:
            assert spec.skill_md == old[sid].skill_md
    manifest = json.loads((LIBRARY / revised["manifest_ref"]).read_text())
    claimed = manifest.pop("manifest_sha256")
    assert canonical_json_sha256(manifest) == claimed == revised["manifest_hash"]
