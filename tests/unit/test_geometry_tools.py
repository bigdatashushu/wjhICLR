"""M6 几何 Tool 合成算例单测。"""

import json

import numpy as np
import pytest

from skill3d.schemas import ObjectInstance, SceneState
from skill3d.tools import REGISTRY, SceneHandle, call_tool


def _scene_state(metric_tasks=None, question_type="") -> SceneState:
    """v4 HC33：米制 Tool 需要 `allowed_metric_tasks` 授权 + 当前题型匹配。"""
    return SceneState(
        artifact_ref="artifact://demo",
        route="full_3d",
        frame="world",
        scale_known=True,
        objects=["chair_0"],
        summary="demo scene",
        scale_confidence="high",
        scale_ci_rel=0.03,
        allowed_metric_tasks=set(metric_tasks or set()),
        question_type=question_type,
    )


def _handle(metric_tasks=None, question_type="", metric_scale=1.0) -> SceneHandle:
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
    # metric_scale：世界单位 → 米 的换算系数（HC29）。测试几何本身以米为单位 →
    # 默认 1.0；换算行为由 `test_metric_tools_apply_metric_scale` 显式覆盖。
    return SceneHandle(_scene_state(metric_tasks, question_type), objects=[obj],
                       c2w_list=c2w, intrinsics=K, objects_materialized=True,
                       metric_scale=metric_scale)


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
    h = _handle(metric_tasks=["object_size_estimation"],
                question_type="object_size_estimation")
    r = call_tool("object_size_longest_dim", {"object_id": "chair_0"}, h)
    assert json.loads(r.value) == pytest.approx(1.0)


def test_room_size_m2():
    # bbox 1x1x1 → up 轴任选，面积为 1
    h = _handle(metric_tasks=["room_size_estimation"],
                question_type="room_size_estimation")
    r = call_tool("room_size_m2", {}, h)
    assert json.loads(r.value) == pytest.approx(1.0)


def test_object_distance_meters_v4():
    """v4 新增米制距离 Tool：仅在 object_abs_distance 授权时可调用。"""
    h = _handle(metric_tasks=["object_abs_distance"], question_type="object_abs_distance")
    r = call_tool("object_distance_meters",
                  {"object_a": "chair_0", "object_b": "chair_0"}, h)
    assert r.error is None and json.loads(r.value) == pytest.approx(0.0)


def test_metric_tool_fail_closed_without_authorization():
    """v4 HC33：未授权米制题型 → ConfidenceGateError（绝不返回相对单位当米制）。"""
    from skill3d.tools.contract import ArtifactUnavailableError, ConfidenceGateError

    h = _handle(metric_tasks=set(), question_type="object_counting")
    with pytest.raises(ConfidenceGateError):
        call_tool("object_size_longest_dim", {"object_id": "chair_0"}, h)
    # 题型被授权但 Tool 不支持该题型 → 同样拒
    h2 = _handle(metric_tasks=["room_size_estimation"], question_type="room_size_estimation")
    with pytest.raises(ConfidenceGateError):
        call_tool("object_distance_meters",
                  {"object_a": "chair_0", "object_b": "chair_0"}, h2)
    # 题型未分类（question_type=""）→ 米制 Tool 一律拒（fail-closed）
    h3 = _handle(metric_tasks=["object_abs_distance"], question_type="")
    with pytest.raises(ConfidenceGateError):
        call_tool("object_distance_meters",
                  {"object_a": "chair_0", "object_b": "chair_0"}, h3)
    # 授权通过但产物真缺（objects 未产出）→ ArtifactUnavailableError（硬约束 23）
    h4 = SceneHandle(_scene_state(["object_abs_distance"], "object_abs_distance"),
                     objects=[], objects_materialized=False)
    with pytest.raises(ArtifactUnavailableError):
        call_tool("object_distance_meters",
                  {"object_a": "chair_0", "object_b": "chair_0"}, h4)


def test_non_metric_3d_tools_unaffected_by_scale_low():
    """v4 HC33：尺度为 low **不得**连累非米制 3D Tool（objects/poses 照常可用）。"""
    h = _handle(metric_tasks=set(), question_type="object_counting")
    r = call_tool("exists_in_scene", {"name": "chair"}, h)
    assert r.error is None and json.loads(r.value) is True
    assert "objects" in h.available_artifacts and "poses" in h.available_artifacts
    assert "scale" not in h.available_artifacts


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


# ------------------------------------------------- HC29 世界单位 → 米 ----

def test_metric_tools_apply_metric_scale():
    """米制 Tool 必须乘 `metric_scale`（HC29：`scale_ci_abs_m = metric_scale × ci_rel`）。

    2026-09-21 定位的缺陷：三个米制 Tool 直接返回**世界单位**数值，未做换算。
    本机真实场景实测 `metric_scale=5.233`，即旧实现会系统性偏小 ~5.2 倍
    （面积类 ~27 倍）。未授权时这些 Tool 不可调用，故当时影响为零。
    """
    sc = 5.0
    h = _handle(metric_tasks=["object_size_estimation"],
                question_type="object_size_estimation", metric_scale=sc)
    assert json.loads(call_tool("object_size_longest_dim",
                                {"object_id": "chair_0"}, h).value) == pytest.approx(1.0 * sc)
    h2 = _handle(metric_tasks=["room_size_estimation"],
                 question_type="room_size_estimation", metric_scale=sc)
    # 面积按**平方**换算
    assert json.loads(call_tool("room_size_m2", {}, h2).value) == pytest.approx(1.0 * sc * sc)


def test_metric_tools_fail_closed_without_metric_scale():
    """无换算系数（未锚定/未标定）→ domain_value 契约错误，不得拿世界单位冒充米制。

    （执行期契约异常由 `call_tool` 归因到 `error_code`，进 D-3/M10 的 fail-closed 路径。）
    """
    h = _handle(metric_tasks=["object_size_estimation"],
                question_type="object_size_estimation", metric_scale=None)
    r = call_tool("object_size_longest_dim", {"object_id": "chair_0"}, h)
    assert r.error_code == "domain_value" and r.value in (None, "null")
    h2 = _handle(metric_tasks=["room_size_estimation"],
                 question_type="room_size_estimation", metric_scale=None)
    r2 = call_tool("room_size_m2", {}, h2)
    assert r2.error_code == "domain_value" and r2.value in (None, "null")


# ------------------------------------- MCA 所需的两个新原语（2026-09-21）----

def test_relative_direction_of_uses_object_reference_frame():
    """rel_dir 题型的正解原语：observer/facing_at/target 都传对象 id。

    回归背景：`relative_direction` 的 `facing` 是**方向向量**，而模型实测传了
    对象质心（一个点）→ 静默算出与题目无关的方位（outer_holdout 1518 因此答错）。
    `relative_direction_of` 用 id 表达"面向某物体"，消除该歧义。
    """
    from skill3d.schemas import ObjectInstance

    def obj(iid, hint, c):
        # bbox 给非退化值且 y 为最小 extent → `_up_axis` 判竖直轴 = y（地面平面 x-z）
        return ObjectInstance(instance_id=iid, class_hint=hint, mask_per_frame="",
                              pointcloud_world="", centroid_world=list(c),
                              bbox=[c[0] - 0.5, c[1] - 0.1, c[2] - 0.5,
                                    c[0] + 0.5, c[1] + 0.1, c[2] + 0.5],
                              confidence=1.0)

    # 场景：观察者在原点，目标在 +x 方向；面向 +z → 目标应在右侧
    scene = _scene_state()
    handle = SceneHandle(
        scene,
        objects=[obj("obj_0", "whiteboard", (0.0, 0.0, 0.0)),
                 obj("obj_1", "door", (0.0, 0.0, 2.0)),
                 obj("obj_2", "laptop", (2.0, 0.0, 0.0))],
        c2w_list=np.eye(4)[None, ...],
        intrinsics=np.array([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]]),
        objects_materialized=True)
    r = call_tool("relative_direction_of",
                  {"observer": "obj_0", "facing_at": "obj_1", "target": "obj_2"}, handle)
    assert r.error is None and json.loads(r.value) == "right"
    # 类别名同样可解析（大小写不敏感）
    r2 = call_tool("relative_direction_of",
                   {"observer": "whiteboard", "facing_at": "door", "target": "laptop"},
                   handle)
    assert json.loads(r2.value) == "right"


def test_object_visible_frames_from_binding():
    """逐帧可见性原语：供"物体首次出现顺序"题型按 min(可见帧) 排序。"""
    from skill3d.schemas import ObjectInstance

    o = ObjectInstance(instance_id="obj_0", class_hint="basket", mask_per_frame="",
                       pointcloud_world="", centroid_world=[0.0, 0.0, 1.0],
                       bbox=[0.0] * 6, confidence=1.0,
                       visible_frames=[4, 5, 9, 12])
    h = SceneHandle(_scene_state(), objects=[o], c2w_list=np.eye(4)[None, ...],
                    intrinsics=np.eye(3)[None, ...], objects_materialized=True)
    r = call_tool("object_visible_frames", {"object_id": "obj_0"}, h)
    assert r.error is None and json.loads(r.value) == [4, 5, 9, 12]


def test_list_objects_ids_are_opaque():
    """`list_objects()` 返回的是 obj_N；类别筛选只能靠 class_hint（文档已写明）。

    回归背景：实测模型写 `'door' in obj_id.lower()` 判断类别 → 永远为空 → abstain
    （outer_holdout 5017 route_planning）。这里锁住"id 不含类别名"这一事实与
    正确的筛选姿势。
    """
    from skill3d.schemas import ObjectInstance

    o = ObjectInstance(instance_id="obj_7", class_hint="door", mask_per_frame="",
                       pointcloud_world="", centroid_world=[0.0, 0.0, 1.0],
                       bbox=[0.0] * 6, confidence=1.0)
    h = SceneHandle(_scene_state(), objects=[o], c2w_list=np.eye(4)[None, ...],
                    intrinsics=np.eye(3)[None, ...], objects_materialized=True)
    assert json.loads(call_tool("list_objects", {}, h).value) == ["obj_7"]
    assert json.loads(call_tool("list_objects", {"class_hint": "door"}, h).value) == ["obj_7"]
    assert json.loads(call_tool("list_objects", {"class_hint": "chair"}, h).value) == []


# --------------------------- 竖直轴与左右手性（2026-09-21 真实缺陷回归）----

def _handle_with_c2w(c2w: np.ndarray, objs=None, metric_tasks=None, question_type=""):
    """用显式 c2w 构造句柄（竖直方向由相机位姿决定，而非包围盒启发式）。"""
    return SceneHandle(_scene_state(metric_tasks, question_type), objects=objs or [],
                       c2w_list=c2w, intrinsics=np.eye(3)[None, ...],
                       objects_materialized=True)


def test_up_direction_comes_from_camera_pose_not_bbox():
    """竖直轴取相机位姿（OpenCV：+y_cam 朝图像下方）→ 世界"上" = −mean(cam_y)。

    实测缺陷：`acd95847c5` 用"包围盒最小 extent 轴"选出 z（水平轴），
    而相机上方向主分量是 y → 四个 rel_direction 题全错。
    """
    # 相机水平朝 +z（R=I，OpenCV 约定）→ 图像上方 = −y
    h = _handle_with_c2w(np.eye(4)[None, ...])
    assert h.up_direction() == pytest.approx(np.array([0.0, -1.0, 0.0]))
    # 相机绕 x 轴 180°（倒置）→ 图像上方 = +y
    flip = np.eye(4)
    flip[1, 1] = flip[2, 2] = -1.0
    h2 = _handle_with_c2w(flip[None, ...])
    assert h2.up_direction() == pytest.approx(np.array([0.0, 1.0, 0.0]))
    # 无位姿 → None（调用方退回轴向启发式）
    h3 = _handle()
    h3._c2w = None  # noqa: SLF001 - 直接构造"缺位姿"状态
    assert h3.up_direction() is None


def test_relative_direction_handedness_follows_world_up():
    """right = forward × up（右手定则）：换世界"上"方向时左右必须翻转。"""
    h = _handle_with_c2w(np.eye(4)[None, ...])          # up = −y
    args = {"observer": [0.0, 0.0, 0.0], "facing": [0.0, 0.0, 1.0]}
    assert json.loads(call_tool("relative_direction",
                                {**args, "target": [2.0, 0.0, 0.0]}, h).value) == "right"
    assert json.loads(call_tool("relative_direction",
                                {**args, "target": [-2.0, 0.0, 0.0]}, h).value) == "left"
    assert json.loads(call_tool("relative_direction",
                                {**args, "target": [0.0, 0.0, -2.0]}, h).value) == "behind"
    assert json.loads(call_tool("relative_direction",
                                {**args, "target": [0.0, 0.0, 2.0]}, h).value) == "front"
    # 倒置相机（up = +y）→ 左右互换（同一个几何，不同的世界约定）
    flip = np.eye(4)
    flip[1, 1] = flip[2, 2] = -1.0
    h2 = _handle_with_c2w(flip[None, ...])
    assert json.loads(call_tool("relative_direction",
                                {**args, "target": [2.0, 0.0, 0.0]}, h2).value) == "left"
    # 目标与观察者仅差竖直方向 → 水平面内退化，必须抛契约错误（不猜）
    r = call_tool("relative_direction", {**args, "target": [0.0, -3.0, 0.0]}, h)
    assert r.error_code == "domain_value"


def test_relative_direction_tilted_camera_projects_onto_horizontal_plane():
    """相机带俯仰角时，判定必须在**水平面**内投影（竖直分量不得混入 left/right）。"""
    tilt = np.eye(4)
    # 绕 x 轴转 20°：图像上方带 +z 分量，但世界竖直仍是 −y 主导
    a = np.deg2rad(20.0)
    tilt[1, 1], tilt[1, 2] = np.cos(a), -np.sin(a)
    tilt[2, 1], tilt[2, 2] = np.sin(a), np.cos(a)
    h = _handle_with_c2w(tilt[None, ...])
    up = h.up_direction()
    assert abs(up[1]) > 0.9                       # 竖直轴仍是 y
    r = call_tool("relative_direction",
                  {"observer": [0, 0, 0], "facing": [0, 0, 1], "target": [2, 0, 1]}, h)
    assert json.loads(r.value) == "right"          # 目标在右前方
