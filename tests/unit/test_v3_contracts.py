"""v3 契约测试：M1 FrameSet / M2 被动观测 / M6 Tool 契约与 ROUTE_ARTIFACTS。

对应《系统架构3.md》硬约束 21（统一固定 FrameSet）与 23（Tool 执行期 fail-closed）。
v6 口径（D4/D6/D7）：逐题工具集由 `question_tool_scope`（`docs(scene_route)` 的子集）
表达，`route` 一词三义已废止；Tool 暴露由 `EvidenceProfile` × `requires_evidence` 决定；
`ReturnAnswer` 之后再调 Tool 在运行期抛 `AnswerAlreadyGiven`、在 AST 层被静态拒绝。
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from skill3d.adapters import frame_set as fs
from skill3d.gates.input_gate import annotate_frames, input_gate
from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import InputFrame, ObjectInstance, SceneState, SkillSpec, ToolSpec
from skill3d.schemas.episode import FrameSet
from skill3d.schemas.evidence import (
    GATE_SUBCONDITIONS,
    GATE_VERSION,
    EvidenceProfile,
    MetricEvidenceGateResult,
)
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES
from skill3d.tools import REGISTRY
from skill3d.tools.contract import (
    ARTIFACT_INTRINSICS,
    ARTIFACT_OBJECTS,
    QUESTION_TOOL_SCOPES,
    ROUTE_ARTIFACTS,
    SCOPE_FALLBACK_2D_ONLY,
    SCOPE_FULL_3D,
    SCOPE_METRIC_ENABLED,
    ToolContractError,
    available_artifacts_for,
    question_tool_scope_of,
    scope_allows,
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

def _profile(**over) -> EvidenceProfile:
    """v6 证据画像：默认全 available（`temporal`/`image_2d` 恒 available，§5.4）。"""
    base = dict(geometry_3d="available", world_frame="available",
                metric_scale="available", object_detection="available",
                track_consensus="available", object_grounding="available")
    base.update(over)
    return EvidenceProfile(**base)


def _scene(scene_route: str = "full_3d", *, question_type: str = "",
           objects: tuple[str, ...] = ("obj_0",), profile=None,
           gate_passed: bool | None = None) -> SceneState:
    """v6 SceneState：`scene_route`（M4 质量）× `question_tool_scope`（逐题派生）正交。

    v5 的 `route` / `frame` / `scale_known` / `scale_confidence` / `allowed_metric_tasks`
    都已废止（D4/§20）：scope 由 `question_tool_scope_of(...)` 派生，米制可用性由
    `EvidenceProfile.metric_scale` + `MetricEvidenceGateResult` 表达。
    """
    if gate_passed is None:
        gate_passed = scene_route == "full_3d"
    gate = (MetricEvidenceGateResult(gate_passed=gate_passed, gate_version=GATE_VERSION,
                                     sub_results={n: gate_passed
                                                  for n in GATE_SUBCONDITIONS})
            if gate_passed else None)
    return SceneState(
        artifact_ref="test", scene_route=scene_route,
        question_tool_scope=question_tool_scope_of(
            scene_route, metric_question=bool(question_type), gate_passed=gate_passed),
        available_artifacts=set(available_artifacts_for(
            scene_route, question_type=question_type, gate_passed=gate_passed,
            metric_question=question_type in METRIC_TASK_TYPES)),
        evidence_profile=profile or _profile(),
        metric_evidence_gate_result=gate,
        objects=list(objects), summary="s", question_type=question_type,
    )


def _handle(scene: SceneState) -> SceneHandle:
    objs = []
    if scene.objects:
        objs = [ObjectInstance(
            obj_id="obj_0", category_name="table", mask_per_frame="", pointcloud_world="",
            centroid_world=[0.0, 0.0, 1.0], bbox=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            det_conf=0.9)]
    c2w = np.tile(np.eye(4), (2, 1, 1)) if "poses" in scene.available_artifacts else None
    k = np.tile(np.eye(3), (2, 1, 1)) if "intrinsics" in scene.available_artifacts else None
    return SceneHandle(scene, objects=objs, c2w_list=c2w, intrinsics=k)


def test_tool_docs_scope_filter_matches_contract_map():
    """属性断言：每个 Tool × 每个 scope 的暴露判定与 `contract.scope_allows` 一致（D-3a）。

    v5 的 `names_for_route(route)` 已由 `names_for_scope(scope, …)` 取代（D4）：scope
    才是"逐题工具集"的维度，`scene_route` 只决定产物集合。
    """
    for scope in QUESTION_TOOL_SCOPES:
        exposed = set(REGISTRY.names_for_scope(scope))
        expected = {n for n in REGISTRY.names()
                    if scope_allows(REGISTRY.spec(n), scope)}
        assert exposed == expected, scope

    # 产物维度交叉核对（scene_route 侧）：2D-only 暴露的 Tool 所需产物必须 ⊂ 该 route 的产物集
    fallback = set(REGISTRY.names_for_scope(SCOPE_FALLBACK_2D_ONLY))
    assert fallback == {"euclidean_distance"}          # 纯算术：无产物、无证据依赖
    for name in fallback:
        assert set(REGISTRY.requires_artifacts(name)) <= set(
            ROUTE_ARTIFACTS["fallback_2d_only"])


def test_tool_docs_monotonic_over_scopes():
    """`docs(full_3d) ⊆ docs(metric_enabled)`；米制 Tool 只在 `metric_enabled` 下出现。

    §5.3/D4：逐题只**收窄**，唯一允许的"新增"是米制 Tool（gate 通过时才出现）。
    v5 的三段 route 单调（full_3d ⊇ fallback_2d_only ⊇ unanswerable）不再适用：
    `unanswerable` 是 scene_route 的取值而非 scope，逐题维度只剩上述两条包含关系。
    """
    full = set(REGISTRY.names_for_scope(SCOPE_FULL_3D))
    metric = set(REGISTRY.names_for_scope(SCOPE_METRIC_ENABLED))
    fallback = set(REGISTRY.names_for_scope(SCOPE_FALLBACK_2D_ONLY))
    metric_tools = {n for n in REGISTRY.names() if REGISTRY.is_metric_tool(n)}

    assert full <= metric                      # 逐题只新增米制 Tool，不新增别的
    assert fallback <= full                    # 2D-only 是 full_3d 的子集
    assert metric_tools                                   # 工具面里确实有米制 Tool
    assert metric_tools.isdisjoint(full)                  # full_3d 下米制 Tool 不可见
    assert metric - full == metric_tools                  # 差集恰好就是米制 Tool
    # scene_route=unanswerable 时没有 scope 能越过它（§5.3 不变量，Schema 层拦截）
    with pytest.raises(ValidationError):
        SceneState(artifact_ref="t", scene_route="unanswerable",
                   question_tool_scope=SCOPE_FULL_3D)


def test_every_tool_declares_requires_artifacts_and_evidence_explicitly():
    """硬约束 23 / §17.4：`requires_artifacts` 与 `requires_evidence` 都必须显式声明。"""
    with pytest.raises(Exception):
        ToolSpec(name="t", description="d", args_schema_ref="", returns_schema_ref="float",
                 cost_estimate_ms=1.0, source_default="real")  # 两个都缺
    with pytest.raises(Exception):
        ToolSpec(name="t", description="d", args_schema_ref="", returns_schema_ref="float",
                 cost_estimate_ms=1.0, source_default="real", requires_artifacts=[])
    with pytest.raises(Exception):
        ToolSpec(name="t", description="d", args_schema_ref="", returns_schema_ref="float",
                 cost_estimate_ms=1.0, source_default="real", requires_evidence=[])  # 缺产物声明
    for name in REGISTRY.names():
        assert isinstance(REGISTRY.requires_artifacts(name), list)
        assert isinstance(REGISTRY.requires_evidence(name), list)   # 显式（[] 合法）


def test_registry_rejects_unknown_artifact_name():
    """产物名必须属于已知词汇表（否则 route 映射会静默失效）。"""
    from skill3d.tools.registry import ToolRegistry

    reg = ToolRegistry()

    def fn(handle, x: float) -> float:  # pragma: no cover - 不会被调用
        return x

    with pytest.raises(ValueError, match="未知产物"):
        reg.register(ToolSpec(name="bad", description="d", args_schema_ref="x:float",
                              returns_schema_ref="float", cost_estimate_ms=1.0,
                              source_default="real", requires_evidence=[],
                              requires_artifacts=["not_a_real_artifact"]))(fn)


def test_fallback_scope_tool_call_is_fail_closed_not_false():
    """硬约束 23：`fallback_2d_only` 下调 exists_in_scene 必须抛错而非返回 False。

    归因必须写明"缺哪些产物"（§9.12）：scope 收窄与产物缺失同源，异常里带
    `missing=["objects"]` 与 `route="fallback_2d_only"`，供 M10 回灌与审计。
    """
    scene = _scene("fallback_2d_only")
    handle = _handle(scene)
    with pytest.raises(ToolContractError) as ei:
        REGISTRY.call_tool("exists_in_scene", {"name": "table"}, handle, mode="real")
    assert ei.value.error_code == "tool_contract"
    assert ei.value.missing == ["objects"]           # objects 不在 2D-only 的可用产物里
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
    r = REGISTRY.call_tool("object_centroid", {"obj_id": "ghost"}, handle, mode="real")
    assert r.error_code == "domain_value" and r.value == "null"
    kernel = RestrictedNamespaceKernel(REGISTRY, handle, frames=[], mode="real")
    cell = kernel.run_cell('c = object_centroid("ghost")\nReturnAnswer(c)\n')
    assert cell.error_code == "tool_contract" and cell.answer is None


def test_scene_handle_narrows_available_artifacts_to_loaded_arrays():
    """句柄只声明"确实装载了数组"的产物：避免声明可用但读取时静默失败。"""
    scene = _scene("full_3d")
    handle = SceneHandle(scene, objects=[], c2w_list=None, intrinsics=None)
    assert "poses" not in handle.available_artifacts
    assert "intrinsics" not in handle.available_artifacts   # 内参未注入 → 不得声明可用
    assert "objects" not in handle.available_artifacts
    assert {"frames", "depth", "point_cloud"} <= handle.available_artifacts


# ------------------------------------- §15.1：答后调 Tool 的双层保护 ----

_RETURN_ANSWER_THEN_TOOL = (
    'ids = list_objects(category_filter="spaceship")\n'   # 空清单 → 走"无力回答"分支
    'if not ids:\n'
    '    ReturnAnswer("abstain")\n'
    'n = count_objects(category_name="chair")\n'          # 答后调 Tool → 必须受控抛错
    'ReturnAnswer(n)\n'
)


def test_tool_call_after_return_answer_raises_at_runtime():
    """§15.1 运行层：`ReturnAnswer` 之后再调 Tool → 受控 `AnswerAlreadyGiven`。

    保留"记录/反作弊"语义（不中止执行）的同时，绝不让程序继续跑到 IndexError
    崩成假的"服务失败"（v5 实测内测方向题全栽在这里：模型写完
    `if not ids: ReturnAnswer("abstain")` 后继续 `object_centroid(ids[0])`）。
    """
    scene = _scene("full_3d")
    handle = _handle(scene)
    kernel = RestrictedNamespaceKernel(REGISTRY, handle, frames=[], mode="real")
    cell = kernel.run_cell(_RETURN_ANSWER_THEN_TOOL)
    # 答案已记录（记录语义保留），但后续 Tool 调用被拦下并按契约违规归因
    assert cell.answer == "abstain"
    assert cell.error_code == "tool_contract"
    assert cell.answer_untrusted is True
    assert cell.contract_violations[0]["error_code"] == "answer_already_given"
    assert "AnswerAlreadyGiven" in (cell.error or "")


def test_tool_call_after_return_answer_is_rejected_statically():
    """§15.1 静态层：同一个程序必须被 AST guard 直接拒绝（提交前拦截）。"""
    result = ast_guard(_RETURN_ANSWER_THEN_TOOL)
    assert not result.ok
    assert any("ReturnAnswer 之后再调用 Tool" in v for v in result.violations)
    # 只把 ReturnAnswer 放在最后 → 通过
    assert ast_guard('ids = list_objects()\nReturnAnswer(len(ids))\n').ok


# ------------------------------------- M7 检索：证据签名 + 米制门双重 fail-closed ----

def _evidence_scene(*, metric_scale: str = "available", gate_passed: bool = True,
                    gate_version: str = GATE_VERSION) -> SceneState:
    """v6 SceneState：EvidenceProfile（metric_scale 三值）+ 米制门（供 M7 检索测试用）。"""
    return SceneState(
        artifact_ref="test", scene_route="full_3d",
        evidence_profile=EvidenceProfile(
            geometry_3d="available", world_frame="available",
            metric_scale=metric_scale, object_detection="available",
            track_consensus="available", object_grounding="available"),
        metric_evidence_gate_result=MetricEvidenceGateResult(
            gate_passed=gate_passed, gate_version=gate_version,
            sub_results={name: gate_passed for name in GATE_SUBCONDITIONS}),
        objects=[], summary="s")


def _abs_distance_skill() -> SkillSpec:
    """米制 Skill：声明证据签名 + requires_metric_evidence + gate 版本（§13.6）。"""
    return SkillSpec(
        skill_id="sk", version="1.0.0",
        applicable_question_types=["object_abs_distance"],
        required_evidence_signature={"metric_scale": "available"},
        requires_metric_evidence=True, applicable_gate_version=GATE_VERSION,
        skill_family="metric", source="real",
        description="d", call_graph_template="t",
        supported_coordinate_frames=["world"], validation_assertions=[])


def test_skill_hard_filter_metric_requires_gate_and_evidence_signature():
    """M7 硬过滤（v6 §17.1/§13.6）：米制 Skill 要求证据签名达标 **且** 米制门通过且版本匹配。

    v5 的 `metric_scale_required` + `scale_confidence ∈ {medium,high}` 全局粗粒度判据已废止：
    现在由 EvidenceProfile 的 `metric_scale` 分项能力 + `MetricEvidenceGateResult` 双重表达。
    """
    from skill3d.routing.skill_retriever import hard_filter, metric_evidence_usable

    skill = _abs_distance_skill()
    assert metric_evidence_usable(_evidence_scene(metric_scale="available"))
    # 融合失败/有限值不过 → metric_scale=unavailable，米制能力不可用
    assert not metric_evidence_usable(
        _evidence_scene(metric_scale="unavailable", gate_passed=False))
    # 融合成功但自洽松 → degraded：签名要求 available 的米制 Skill 仍不得被检索
    assert not metric_evidence_usable(_evidence_scene(metric_scale="degraded"))

    ok = _evidence_scene()
    assert hard_filter(skill, ok, question_type="object_abs_distance")
    assert not hard_filter(  # gate 版本不匹配 → fail-closed（§13.6 双重校验）
        skill, _evidence_scene(gate_version="metric-evidence-gate-v5"),
        question_type="object_abs_distance")
    assert not hard_filter(  # 题型不匹配 → 不检索（§17.1）
        skill, ok, question_type="object_counting")


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
