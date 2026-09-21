"""v3 契约测试：M1 FrameSet / M2 被动观测 / M6 Tool 契约与 ROUTE_ARTIFACTS。

对应《系统架构3.md》硬约束 21（统一固定 FrameSet）与 23（Tool 执行期 fail-closed）。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.adapters import frame_set as fs
from skill3d.gates.input_gate import annotate_frames, input_gate
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import InputFrame, SceneState, ToolSpec
from skill3d.schemas.episode import FrameSet
from skill3d.tools import REGISTRY
from skill3d.tools.contract import (
    ARTIFACT_INTRINSICS,
    ARTIFACT_OBJECTS,
    ROUTE_ARTIFACTS,
    ArtifactUnavailableError,
    available_artifacts_for,
    tool_allowed,
)
from skill3d.tools.scene_handle import SceneHandle

# ------------------------------------------------------------------ M1 FrameSet ----


@pytest.mark.parametrize("total", [32, 33, 100, 1000, 5045, 100000])
def test_uniform_frame_ids_are_unique_and_monotonic(total):
    """32 帧均匀采样：唯一、严格递增、覆盖首尾（§4 M1 验收条件 b）。"""
    ids = fs.uniform_frame_ids(total, 32)
    assert len(ids) == 32 and len(set(ids)) == 32
    assert ids == sorted(ids) and ids[0] == 0 and ids[-1] == total - 1


def test_too_short_video_is_input_legality_hard_fail():
    """不足 32 帧 → FrameSetError（硬约束 21：禁止重复帧补齐凑数）。"""
    with pytest.raises(fs.FrameSetError):
        fs.uniform_frame_ids(31, 32)


def test_frame_set_hash_deterministic_content_addressed():
    a = fs.build_frame_set(1000, fps=30.0)
    b = fs.build_frame_set(1000, fps=30.0)
    c = fs.build_frame_set(1000, fps=60.0)      # fps 不影响帧号 → 哈希不变
    d = fs.build_frame_set(2000, fps=30.0)
    assert a.frame_set_hash == b.frame_set_hash == c.frame_set_hash
    assert a.frame_set_hash != d.frame_set_hash
    assert a.frame_ids == a.source_frame_indices


def test_assert_same_frame_set_rejects_double_frame_set():
    """硬约束 21/18：两条臂帧集不一致必须抛错（禁止双帧集）。"""
    a = fs.build_frame_set(1000)
    b = fs.build_frame_set(1000)
    fs.assert_same_frame_set(a, b)               # 不抛
    with pytest.raises(fs.FrameSetError):
        fs.assert_same_frame_set(a, fs.build_frame_set(1000, n_frames=16))


# ------------------------------------------------------------------ M2 被动观测 ----

def _frames(n: int = 32, blur: float = 200.0) -> list[InputFrame]:
    return [InputFrame(frame_idx=i, timestamp=float(i), blur_var=blur,
                       overexposed_ratio=0.0, underexposed_ratio=0.0, quality_ok=True)
            for i in range(n)]


def test_m2_never_changes_the_frame_set():
    """M2 只打 flag/权重：annotate_frames 前后帧数/帧序完全一致（硬约束 21）。"""
    frames = _frames()
    for i in (2, 5, 9, 20):
        frames[i] = frames[i].model_copy(update={"blur_var": 1.0})
    verdict = input_gate(frames)
    out = annotate_frames(frames, verdict)
    assert len(out) == len(frames) == 32
    assert [f.frame_idx for f in out] == [f.frame_idx for f in frames]
    assert verdict.degraded_frame_ids == [2, 5, 9, 20]
    assert all(f.quality_weight == 1.0 for i, f in enumerate(out)
               if i not in verdict.degraded_frame_ids)
    assert all(f.quality_weight < 1.0 for i, f in enumerate(out)
               if i in verdict.degraded_frame_ids)


def test_annotate_frames_rejects_frame_count_mismatch():
    """帧集与像素数量不一致 = 帧集被改动 → 必须报错（硬约束 21）。"""
    with pytest.raises(ValueError):
        annotate_frames(_frames(32), input_gate(_frames(32)),
                        [np.zeros((8, 8, 3), dtype=np.uint8)] * 31)


# ------------------------------------------------------- M6 Tool 契约 fail-closed ----

def _scene(route: str, *, with_objects: bool = True, with_poses: bool = True,
           quality_status: str = "computed", overall: float = 0.9) -> SceneState:
    return SceneState(
        artifact_ref="test", route=route, frame="world", scale_known=True,
        objects=["obj_0"] if with_objects else [], summary="s",
        scale_confidence="medium", available_artifacts=set(available_artifacts_for(route)),
    )


def _handle(scene: SceneState) -> SceneHandle:
    from skill3d.schemas import ObjectInstance

    objs = []
    if scene.objects:
        objs = [ObjectInstance(
            instance_id="obj_0", class_hint="table", mask_per_frame="", pointcloud_world="",
            centroid_world=[0.0, 0.0, 1.0], bbox=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            confidence=0.9)]
    c2w = np.tile(np.eye(4), (2, 1, 1)) if "poses" in scene.available_artifacts else None
    k = np.tile(np.eye(3), (2, 1, 1)) if "intrinsics" in scene.available_artifacts else None
    return SceneHandle(scene, objects=objs, c2w_list=c2w, intrinsics=k)


def test_tool_docs_route_filter_matches_contract_map():
    """属性断言：每个 Tool × 每个 route 的暴露判定与 ROUTE_ARTIFACTS 一致（D-3a）。"""
    for route in ROUTE_ARTIFACTS:
        exposed = set(REGISTRY.names_for_route(route))
        expected = {n for n in REGISTRY.names()
                    if tool_allowed(REGISTRY.requires_artifacts(n), route)}
        assert exposed == expected, route


def test_tool_docs_monotonic_over_routes():
    """`docs(full_3d) ⊇ docs(fallback_2d_only) ⊇ docs(unanswerable)`（§4 M6）。"""
    full = set(REGISTRY.names_for_route("full_3d"))
    fb = set(REGISTRY.names_for_route("fallback_2d_only"))
    un = set(REGISTRY.names_for_route("unanswerable"))
    assert full >= fb >= un


def test_every_tool_declares_requires_artifacts_explicitly():
    """硬约束 23：requires_artifacts 必须在注册处显式声明（不写就构造报错）。"""
    with pytest.raises(Exception):
        ToolSpec(name="t", description="d", args_schema_ref="", returns_schema_ref="float",
                 cost_estimate_ms=1.0, source_default="real")  # 缺 requires_artifacts
    for name in REGISTRY.names():
        assert isinstance(REGISTRY.requires_artifacts(name), list)


def test_registry_rejects_unknown_artifact_name():
    """产物名必须属于已知词汇表（否则 route 映射会静默失效）。"""
    from skill3d.tools.registry import ToolRegistry

    reg = ToolRegistry()

    def fn(handle, x: float) -> float:  # pragma: no cover - 不会被调用
        return x

    with pytest.raises(ValueError):
        reg.register(ToolSpec(name="bad", description="d", args_schema_ref="x:float",
                              returns_schema_ref="float", cost_estimate_ms=1.0,
                              source_default="real",
                              requires_artifacts=["not_a_real_artifact"]))(fn)


def test_fallback_route_tool_call_is_fail_closed_not_false():
    """硬约束 23：route=fallback_2d_only 下调 exists_in_scene 必须抛错而非返回 False。"""
    scene = _scene("fallback_2d_only")
    handle = _handle(scene)
    with pytest.raises(ArtifactUnavailableError) as ei:
        REGISTRY.call_tool("exists_in_scene", {"name": "table"}, handle, mode="real")
    assert ei.value.missing == ["objects"]
    assert ei.value.route == "fallback_2d_only"


def test_kernel_maps_contract_violation_to_tool_contract_and_untrusts_answer():
    """M10：契约违规 → error_code=tool_contract，且 ReturnAnswer 的答案不得采纳。"""
    scene = _scene("fallback_2d_only")
    handle = _handle(scene)
    kernel = RestrictedNamespaceKernel(REGISTRY, handle, frames=[], mode="real")
    cell = kernel.run_cell(
        'if exists_in_scene("table"):\n    n = 1\nelse:\n    n = 0\nReturnAnswer(n)\n')
    assert cell.error_code == "tool_contract"
    assert cell.answer is None
    assert cell.answer_untrusted is True
    assert cell.contract_violations[0]["missing_artifacts"] == ["objects"]
    # 调用本身仍进 ProgramExecutionTrace（审计可回放）
    assert [r.tool for r in kernel.tool_results] == ["exists_in_scene"]
    assert kernel.tool_results[0].error_code == "tool_contract"


def test_exists_in_scene_false_iff_objects_available_and_truly_absent():
    """`exists_in_scene(name)==False` 当且仅当 objects 可用且真无此实例。"""
    scene = _scene("full_3d")
    handle = _handle(scene)
    assert REGISTRY.call_tool("exists_in_scene", {"name": "table"}, handle,
                              mode="real").value == "true"
    assert REGISTRY.call_tool("exists_in_scene", {"name": "spaceship"}, handle,
                              mode="real").value == "false"


def test_domain_errors_are_attributed_not_swallowed():
    """域值错误（点不存在 / 单位/几何非法）→ error_code=domain_value，静默返回假值禁止。"""
    scene = _scene("full_3d")
    handle = _handle(scene)
    r = REGISTRY.call_tool("object_centroid", {"object_id": "ghost"}, handle, mode="real")
    assert r.error_code == "domain_value" and r.value == "null"
    kernel = RestrictedNamespaceKernel(REGISTRY, handle, frames=[], mode="real")
    cell = kernel.run_cell('c = object_centroid("ghost")\nReturnAnswer(c)\n')
    assert cell.error_code == "tool_contract" and cell.answer is None


def test_scene_handle_narrows_available_artifacts_to_loaded_arrays():
    """句柄只声明"确实装载了数组"的产物：避免声明可用但读取时静默失败。"""
    scene = _scene("full_3d")
    handle = SceneHandle(scene, objects=[], c2w_list=None, intrinsics=None)
    assert "poses" not in handle.available_artifacts
    assert "objects" not in handle.available_artifacts
    assert "frames" in handle.available_artifacts


# ------------------------------------------------------------------ M7 尺度谓词 ----

def test_skill_hard_filter_requires_medium_or_high_scale_confidence():
    """M7 硬过滤：metric_scale_required 只接受 medium/high（low 一律过滤）。"""
    from skill3d.routing.skill_retriever import hard_filter, metric_scale_usable
    from skill3d.schemas import SkillSpec

    skill = SkillSpec(
        skill_id="sk", semver="1.0.0", task_type="object_abs_distance",
        description="d", call_graph_template="t", requires_artifacts=[],
        minimum_quality=0.0, supported_coordinate_frames=["world"],
        metric_scale_required=True, validation_assertions=[])

    low = _scene("full_3d").model_copy(update={"scale_confidence": "low"})
    med = _scene("full_3d").model_copy(update={"scale_confidence": "medium"})
    none = _scene("full_3d").model_copy(update={"scale_confidence": None})
    assert not metric_scale_usable(low)
    assert not metric_scale_usable(none)          # 旧 artifact 的 None 视为不可用
    assert metric_scale_usable(med)
    assert not hard_filter(skill, low) and hard_filter(skill, med)


# --------------------------------------------- 硬约束 18：paired A/B 同源断言 ----

def test_paired_ab_asserts_same_artifact_and_frame_set_per_episode():
    """逐 episode 断言 A/B 同 artifact ref 与同 frame_set_hash（硬约束 18/21）。"""
    from skill3d.skills.paired_ab import (
        PairedArtifactMismatchError,
        PairedFrameSetMismatchError,
        assert_paired_outcomes_share_artifact,
    )

    class _O:
        def __init__(self, ref, h, qa="q1"):
            self.artifact_ref, self.frame_set_hash, self.qa_id = ref, h, qa

    a = [_O("art/x.json", "hash-a"), _O("art/y.json", "hash-b", "q2")]
    b = [_O("art/x.json", "hash-a"), _O("art/y.json", "hash-b", "q2")]
    assert_paired_outcomes_share_artifact(a, b)          # 同源 → 通过

    with pytest.raises(PairedArtifactMismatchError):
        assert_paired_outcomes_share_artifact(a, [_O("art/other.json", "hash-a"),
                                                  _O("art/y.json", "hash-b", "q2")])
    with pytest.raises(PairedFrameSetMismatchError):
        assert_paired_outcomes_share_artifact(a, [_O("art/x.json", "hash-DIFFERENT"),
                                                  _O("art/y.json", "hash-b", "q2")])
    with pytest.raises(PairedArtifactMismatchError):
        assert_paired_outcomes_share_artifact(a, b[:1])   # episode 数不一致
