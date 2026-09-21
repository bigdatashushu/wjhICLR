"""v4 集成契约：逐题型授权在整条在线链上的行为（HC29–33）。

覆盖三件在真实链路上必须成立的事：

1. **scale=low 不得摧毁 3D 能力**（HC33）：非米制 3D Tool 照常可用，
   只有 `scale` 产物被收回；
2. **未授权题型调用米制 Tool → ConfidenceGateError**（而不是拿相对单位当米制）；
3. **prompt 头部如实写"允许的米制题型"**（§3 M6），且 prompt 里不出现必然失败的
   米制 Tool（否则模型照着写必撞 fail-closed）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from skill3d.schemas import ObjectInstance, SceneState
from skill3d.tools import REGISTRY, SceneHandle, call_tool
from skill3d.tools.contract import (
    ArtifactUnavailableError,
    ConfidenceGateError,
    route_artifacts_for_question,
)


def _obj() -> ObjectInstance:
    return ObjectInstance(
        instance_id="obj_0", class_hint="table", mask_per_frame="", pointcloud_world="",
        centroid_world=[0.0, 0.0, 1.0], bbox=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        confidence=0.9)


def _scene(route="full_3d", *, allowed=(), question_type="", conf="low",
           scale_known=True) -> SceneState:
    return SceneState(
        artifact_ref="a", route=route, frame="world", scale_known=scale_known,
        objects=["obj_0"], summary="s", scale_confidence=conf, scale_ci_rel=0.02,
        allowed_metric_tasks=set(allowed), question_type=question_type,
        available_artifacts=route_artifacts_for_question(route, set(allowed),
                                                         question_type))


def _handle(scene: SceneState, metric_scale: float = 1.0) -> SceneHandle:
    # metric_scale=1.0：本测试的合成几何即米制单位（HC29 的换算系数为 1）。
    # 换算本身由 tests/unit/test_geometry_tools.py::test_metric_tools_apply_metric_scale 覆盖。
    return SceneHandle(scene, objects=[_obj()],
                       c2w_list=np.tile(np.eye(4), (2, 1, 1)),
                       intrinsics=np.tile(np.eye(3), (2, 1, 1)),
                       objects_materialized=True, metric_scale=metric_scale)


# ------------------------------------------------- 逐题可用产物集合 ----

def test_route_artifacts_for_question_includes_scale_only_when_authorized():
    base = route_artifacts_for_question("full_3d", set(), "object_counting")
    assert "scale" not in base
    # HC33：非米制 3D 产物不因尺度 low 被降级
    assert {"depth", "poses", "point_cloud", "objects"} <= base

    authorized = route_artifacts_for_question(
        "full_3d", {"object_abs_distance"}, "object_abs_distance")
    assert "scale" in authorized
    # 授权的题型不匹配时也不加入（逐题而非全局）
    other = route_artifacts_for_question("full_3d", {"object_abs_distance"},
                                         "object_rel_distance")
    assert "scale" not in other


def test_fallback_route_never_has_scale():
    for allowed in (set(), {"object_abs_distance"}):
        got = route_artifacts_for_question("fallback_2d_only", allowed,
                                           "object_abs_distance")
        assert got == {"frames", "intrinsics"}


# --------------------------------------------------- 执行期 fail-closed ----

def test_unmeasured_scale_retracts_metric_tools_only():
    """HC33：`low` 只收回米制工具；非米制 3D Tool 照常可用。"""
    scene = _scene(conf="low", question_type="object_counting")
    handle = _handle(scene)
    assert "scale" not in handle.available_artifacts
    assert {"objects", "poses", "intrinsics"} <= handle.available_artifacts

    # 非米制 Tool 正常
    r = call_tool("exists_in_scene", {"name": "table"}, handle)
    assert r.error is None and json.loads(r.value) is True
    r2 = call_tool("euclidean_distance",
                   {"point_a": [0, 0, 0], "point_b": [3, 4, 0]}, handle)
    assert r2.error is None and json.loads(r2.value) == pytest.approx(5.0)

    # 米制 Tool 被拒（授权检查先于产物检查 → ConfidenceGateError）
    for tool, args in (("object_size_longest_dim", {"object_id": "obj_0"}),
                       ("room_size_m2", {}),
                       ("object_distance_meters",
                        {"object_a": "obj_0", "object_b": "obj_0"})):
        with pytest.raises(ConfidenceGateError):
            call_tool(tool, args, handle)


def test_authorized_task_executes_metric_tool():
    scene = _scene(allowed={"object_size_estimation"},
                   question_type="object_size_estimation", conf="medium")
    handle = _handle(scene)
    assert "scale" in handle.available_artifacts
    r = call_tool("object_size_longest_dim", {"object_id": "obj_0"}, handle)
    assert r.error is None and json.loads(r.value) == pytest.approx(1.0)


def test_wrong_task_authorization_still_blocked():
    """授权了 A 题型不等于 B 题型也能用米制 Tool（逐题型闸门）。"""
    scene = _scene(allowed={"room_size_estimation"},
                   question_type="room_size_estimation", conf="medium")
    handle = _handle(scene)
    with pytest.raises(ConfidenceGateError):
        call_tool("object_size_longest_dim", {"object_id": "obj_0"}, handle)


def test_tools_requiring_scale_are_fail_closed_when_scale_artifact_missing():
    """授权通过但 `scale` 产物真缺（scale_known=False）→ ArtifactUnavailableError。"""
    scene = _scene(allowed={"object_size_estimation"},
                   question_type="object_size_estimation", conf="medium",
                   scale_known=False)
    handle = _handle(scene)
    assert "scale" not in handle.available_artifacts
    with pytest.raises(ArtifactUnavailableError):
        call_tool("object_size_longest_dim", {"object_id": "obj_0"}, handle)


# ------------------------------------------------ prompt / docs 一致性 ----

def test_docs_filter_matches_execution_contract():
    """属性断言：`docs()` 列出的米制 Tool 必然可执行（不会撞 fail-closed）。"""
    for qt, allowed in (("object_counting", set()),
                        ("object_size_estimation", {"object_size_estimation"}),
                        ("room_size_estimation", {"room_size_estimation"}),
                        ("object_abs_distance", {"object_abs_distance"})):
        scene = _scene(allowed=allowed, question_type=qt, conf="medium")
        handle = _handle(scene)
        names = REGISTRY.names_for_route(scene.route, handle.available_artifacts,
                                        allowed_metric_tasks=scene.allowed_metric_tasks,
                                        question_type=scene.question_type)
        for name in names:
            # docs 列出的每个 Tool 都必须能通过"授权 + 产物"两道门
            from skill3d.tools.contract import (
                check_artifact_contract,
                check_metric_task_contract,
            )

            spec = REGISTRY.spec(name)
            check_metric_task_contract(name, spec.supported_metric_tasks,
                                       scene.allowed_metric_tasks, scene.question_type)
            check_artifact_contract(name, spec.requires_artifacts,
                                    handle.available_artifacts, route=scene.route)


def test_docs_header_states_allowed_metric_tasks():
    header = REGISTRY.docs_header("full_3d", ["frames", "objects"], 
                                 allowed_metric_tasks=set(),
                                 question_type="object_counting")
    assert "允许的米制题型=（无）" in header
    assert "相对单位冒充米制" in header
    header2 = REGISTRY.docs_header("full_3d", ["frames", "scale"],
                                   allowed_metric_tasks={"object_size_estimation"},
                                   question_type="object_size_estimation")
    assert "object_size_estimation" in header2


def test_metric_tools_are_absent_from_docs_when_not_authorized():
    docs = REGISTRY.docs(route="full_3d", available=["frames", "objects", "poses",
                                                     "intrinsics", "depth",
                                                     "point_cloud"],
                         allowed_metric_tasks=set(), question_type="object_counting")
    for metric_tool in ("object_size_longest_dim", "room_size_m2",
                        "object_distance_meters"):
        assert metric_tool not in docs
    assert "exists_in_scene" in docs          # 非米制 3D Tool 仍在
    assert "reproject" in docs


def test_registry_rejects_metric_tool_without_scale_artifact():
    """注册期校验：声明了 supported_metric_tasks 却不依赖 scale → 直接拒绝。"""
    from skill3d.schemas import ToolSpec
    from skill3d.tools.registry import ToolRegistry

    reg = ToolRegistry()

    with pytest.raises(ValueError):
        @reg.register(ToolSpec(
            name="bad_metric", description="d", args_schema_ref="",
            returns_schema_ref="float", cost_estimate_ms=1.0, source_default="real",
            requires_artifacts=["objects"],
            supported_metric_tasks=["object_size_estimation"]))
        def bad_metric(handle):  # pragma: no cover - 注册即失败
            return 1.0

    with pytest.raises(ValueError):
        @reg.register(ToolSpec(
            name="bad_task", description="d", args_schema_ref="",
            returns_schema_ref="float", cost_estimate_ms=1.0, source_default="real",
            requires_artifacts=["objects", "scale"],
            supported_metric_tasks=["relative_direction"]))
        def bad_task(handle):  # pragma: no cover - 注册即失败
            return 1.0


def test_scene_metric_task_authorized_helper():
    scene = _scene(allowed={"object_size_estimation"},
                   question_type="object_size_estimation")
    assert scene.metric_task_authorized("object_size_estimation")
    assert not scene.metric_task_authorized("object_abs_distance")
    # 未分类 → 一律不授权（fail-closed）
    assert not _scene(allowed={"object_size_estimation"}).metric_task_authorized("")
