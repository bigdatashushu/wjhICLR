"""M6 几何 Tool 合成算例单测。"""

import json

import numpy as np
import pytest

from skill3d.schemas import ObjectInstance, SceneState
from skill3d.tools import REGISTRY, SceneHandle, call_tool


def _scene_state() -> SceneState:
    return SceneState(
        artifact_ref="artifact://demo",
        route="full_3d",
        frame="world",
        scale_known=True,
        objects=["chair_0"],
        summary="demo scene",
    )


def _handle() -> SceneHandle:
    obj = ObjectInstance(
        instance_id="chair_0",
        class_hint="chair",
        mask_per_frame="arr://mask",
        pointcloud_world="arr://pc",
        centroid_world=[1.0, 0.0, 1.0],
        bbox=[0.5, -0.1, 0.5, 1.5, 0.1, 1.5],  # y 为最小 extent → 竖直轴=y
        confidence=0.9,
    )
    # 相机位于原点，光轴朝 +z；K = diag(f,f,1), cx=cy=0
    c2w = np.eye(4)[None, ...]
    K = np.array([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]])
    return SceneHandle(_scene_state(), objects=[obj], c2w_list=c2w, intrinsics=K)


def test_euclidean_distance():
    r = call_tool("euclidean_distance", {"point_a": [0, 0, 0], "point_b": [3, 4, 0]}, _handle())
    assert r.error is None
    assert json.loads(r.value) == pytest.approx(5.0)
    assert r.source == "real"
    assert len(r.request_digest) == 64


def test_relative_direction_front_left():
    h = _handle()
    # up 轴 = y（bbox y extent=1.0 与 x/z 相同……改为显式用例：observer 原点，facing +z）
    front = call_tool(
        "relative_direction",
        {"observer": [0, 0, 0], "facing": [0, 0, 1], "target": [0, 0, 5]},
        h,
    )
    assert json.loads(front.value) == "front"
    left = call_tool(
        "relative_direction",
        {"observer": [0, 0, 0], "facing": [0, 0, 1], "target": [-5, 0, 0.1]},
        h,
    )
    assert json.loads(left.value) == "left"
    right = call_tool(
        "relative_direction",
        {"observer": [0, 0, 0], "facing": [0, 0, 1], "target": [5, 0, 0.1]},
        h,
    )
    assert json.loads(right.value) == "right"


def test_object_size_longest_dim():
    r = call_tool("object_size_longest_dim", {"object_id": "chair_0"}, _handle())
    assert json.loads(r.value) == pytest.approx(1.0)


def test_room_size_m2():
    # bbox 1x1x1 → up 轴任选，面积为 1
    r = call_tool("room_size_m2", {}, _handle())
    assert json.loads(r.value) == pytest.approx(1.0)


def test_reproject():
    h = _handle()
    # 世界点 (0,0,2)，c2w=I，f=100 → uv = (0,0)
    r = call_tool("reproject", {"p3d": [0.0, 0.0, 2.0], "frame_idx": 0}, h)
    assert json.loads(r.value) == [pytest.approx(0.0), pytest.approx(0.0)]
    # 世界点 (1,0,2) → u = 100*1/2 = 50
    r2 = call_tool("reproject", {"p3d": [1.0, 0.0, 2.0], "frame_idx": 0}, h)
    assert json.loads(r2.value)[0] == pytest.approx(50.0)
    # 相机后方的点 → Tool 确定性异常进 error
    r3 = call_tool("reproject", {"p3d": [0.0, 0.0, -1.0], "frame_idx": 0}, h)
    assert r3.error is not None


def test_exists_in_scene():
    h = _handle()
    assert json.loads(call_tool("exists_in_scene", {"name": "chair"}, h).value) is True
    assert json.loads(call_tool("exists_in_scene", {"name": "sofa"}, h).value) is False


def test_unknown_tool_raises():
    from skill3d.tools.registry import ToolNotFoundError

    with pytest.raises(ToolNotFoundError):
        call_tool("not_a_tool", {}, _handle())


def test_arg_validation():
    from skill3d.tools.registry import ToolArgValidationError

    with pytest.raises(ToolArgValidationError):
        call_tool("euclidean_distance", {"point_a": [0, 0, 0]}, _handle())
