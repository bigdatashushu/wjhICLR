"""v9 P3：工具授权收据（§6.4）、证据不可用原因码（§6.1）与门三态（§8.2）。

守三件事：

1. **收据与判定同源**：`allowed` 必须来自真正决定放行的函数
   （`tools.contract.authorize_tool_call`），而不是事后拼一个通过凭据；
2. **允许也要留痕**：此前授权只以异常形式存在，成功调用没有"谁授权了它"的凭据，
   于是"证据摘要与授权收据不能互相矛盾"根本无法校验；
3. **三态不许伪装**：非米制工具是 `not_applicable` + `gate_passed=None`，
   不能用默认 `False` 假装成"门失败"；`unavailable` 必须能说出原因码。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.schemas import (
    AnswerPayload,
    SceneState,
    ToolAuthorizationReceipt,
    ToolResult,
)
from skill3d.schemas.authorization import (
    DECISION_VERSION,
    UNAVAILABLE_REASON_CODES,
    not_applicable_gate,
)
from skill3d.schemas.evidence import (
    GATE_SUBCONDITIONS,
    GATE_VERSION,
    EvidenceProfile,
    MetricEvidenceGateResult,
)
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import ObjectInstance
from skill3d.tools import REGISTRY
from skill3d.tools.contract import authorize_tool_call, available_artifacts_for, question_tool_scope_of
from skill3d.tools.scene_handle import SceneHandle

RECEIPT_FIELDS = ("episode_id", "tool_id", "argument_digest", "evidence_version",
                  "dependency_refs", "allowed", "reason_codes", "metric_gate_result",
                  "decision_version")


# --------------------------------------------------- §6.4 收据字段与同源 ----

def _profile(**over) -> EvidenceProfile:
    base = dict(geometry_3d="available", world_frame="available",
                metric_scale="available", object_detection="available",
                track_consensus="available", object_grounding="available")
    base.update(over)
    return EvidenceProfile(**base)


def _scene(question_type: str = "object_counting") -> SceneState:
    gate = MetricEvidenceGateResult(
        gate_passed=True, gate_version=GATE_VERSION,
        sub_results={n: True for n in GATE_SUBCONDITIONS})
    return SceneState(
        artifact_ref="t", scene_route="full_3d",
        question_tool_scope=question_tool_scope_of(
            "full_3d", metric_question=question_type in METRIC_TASK_TYPES,
            gate_passed=True),
        available_artifacts=set(available_artifacts_for(
            "full_3d", question_type=question_type, gate_passed=True,
            metric_question=question_type in METRIC_TASK_TYPES)),
        evidence_profile=_profile(), metric_evidence_gate_result=gate,
        objects=["obj_0"], summary="s", question_type=question_type)


def _handle(scene: SceneState) -> SceneHandle:
    return SceneHandle(
        scene,
        objects=[ObjectInstance(
            obj_id="obj_0", category_name="chair", mask_per_frame="",
            pointcloud_world="", centroid_world=[0.0, 0.0, 1.0],
            bbox=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], det_conf=0.9)],
        c2w_list=np.tile(np.eye(4), (2, 1, 1)),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)))


def test_receipt_has_every_doc_required_field():
    assert set(RECEIPT_FIELDS) <= set(ToolAuthorizationReceipt.model_fields)
    assert ToolAuthorizationReceipt.model_fields["decision_version"].default == \
        DECISION_VERSION


def test_allowed_call_carries_a_receipt_with_allowed_true():
    """允许的调用同样要有收据（此前只有拒绝才留痕）。"""
    scene = _scene()
    handle = _handle(scene)
    result = REGISTRY.call_tool("count_objects", {"category_name": "chair"}, handle,
                               mode="real", episode_id="qa-1")
    assert result.status == "ok"
    receipt = result.authorization
    assert receipt, "成功调用必须带授权收据"
    assert set(RECEIPT_FIELDS) <= set(receipt)
    assert receipt["allowed"] is True
    assert receipt["reason_codes"] == ["allowed"]
    assert receipt["episode_id"] == "qa-1"
    assert receipt["tool_id"] == "count_objects"
    assert receipt["result_id"] == result.result_id, "收据必须与 ToolResult 关联"
    assert receipt["argument_digest"] == result.request_digest, \
        "收据的参数摘要必须与工具结果的请求摘要同源"
    assert receipt["dependency_refs"], "应记录本次调用依赖的产物/证据"


def test_denied_call_receipt_says_not_allowed_with_reason_codes():
    """finalization 封锁下的调用：收据 `allowed=False`，且与"确实没执行"一致。"""
    kernel = RestrictedNamespaceKernel(REGISTRY, _handle(_scene()), frames=[],
                                       mode="real", tools_enabled=False,
                                       episode_id="qa-2")
    kernel.run_cell('n = count_objects(category_name="chair")\nReturnAnswer(n)\n')
    assert len(kernel.authorization_receipts) == 1
    receipt = kernel.authorization_receipts[0]
    assert receipt["allowed"] is False
    assert "tools_disabled" in receipt["reason_codes"]
    assert receipt["episode_id"] == "qa-2"
    # 被拒绝的调用也要有可追踪参数摘要（§17.1：失败也有 result_id）
    assert receipt["argument_digest"]
    assert kernel.tool_results[0].status == "failed"


def test_denial_reason_codes_are_from_the_documented_vocabulary():
    """拒绝原因码必须落在既有契约错误码 + 作用域/对象级拒绝的集合内。"""
    from skill3d.schemas.authorization import DENIAL_REASON_CODES

    assert {"tool_contract", "confidence_gate", "domain_value",
            "answer_already_given"} <= DENIAL_REASON_CODES
    assert {"tools_disabled", "scope_denied", "object_unbound"} <= DENIAL_REASON_CODES


def test_authorize_is_the_same_function_for_receipt_and_decision():
    """§6.3/§6.4：收据与实际放行判定必须来自同一次决策。"""
    scene = _scene()
    spec = REGISTRY.get("count_objects").spec
    decision = authorize_tool_call(
        "count_objects", spec, scope=scene.question_tool_scope,
        profile=scene.evidence_profile,
        available_artifacts=scene.available_artifacts,
        gate=scene.metric_evidence_gate_result,
        supported_metric_tasks=spec.supported_metric_tasks,
        allowed_metric_tasks=_handle(scene).allowed_metric_tasks,
        question_type=scene.question_type, args={"category_name": "chair"})
    assert decision.allowed is True
    # 同一次调用经 registry 后收据的 allowed 必须与决策一致
    result = REGISTRY.call_tool("count_objects", {"category_name": "chair"},
                               _handle(scene), mode="real")
    assert result.authorization["allowed"] is decision.allowed


def test_evidence_gap_denial_names_the_missing_capability():
    """证据不足的拒绝要能说出缺哪个能力（收据与错误文案同源）。"""
    scene = _scene()
    scene = scene.model_copy(update={"evidence_profile": _profile(
        geometry_3d="unavailable", object_detection="unavailable",
        metric_scale="unavailable")})
    spec = REGISTRY.get("count_objects").spec
    decision = authorize_tool_call(
        "count_objects", spec, scope=scene.question_tool_scope,
        profile=scene.evidence_profile, available_artifacts=scene.available_artifacts,
        gate=scene.metric_evidence_gate_result,
        supported_metric_tasks=spec.supported_metric_tasks,
        allowed_metric_tasks=_handle(scene).allowed_metric_tasks,
        question_type=scene.question_type)
    assert decision.allowed is False
    assert "tool_contract" in decision.reason_codes
    assert any("unavailable" in m for m in decision.missing_refs), decision.missing_refs


def test_non_metric_tool_gate_is_not_applicable_not_failed():
    """§6.4/§8.2：非米制工具的米制门是 `not_applicable`，不是"失败"。"""
    scene = _scene("object_counting")
    spec = REGISTRY.get("count_objects").spec
    decision = authorize_tool_call(
        "count_objects", spec, scope=scene.question_tool_scope,
        profile=scene.evidence_profile, available_artifacts=scene.available_artifacts,
        gate=scene.metric_evidence_gate_result,
        supported_metric_tasks=spec.supported_metric_tasks,
        allowed_metric_tasks=_handle(scene).allowed_metric_tasks,
        question_type=scene.question_type)
    assert decision.metric_gate_applicable is False
    assert decision.metric_gate_result["status"] == "not_applicable"
    assert decision.metric_gate_result["gate_passed"] is None
    assert decision.allowed is True, "不适用不得导致拒绝"


def test_metric_tool_gate_carries_the_real_gate_result():
    scene = _scene("object_abs_distance")
    spec = REGISTRY.get("object_3d_extent").spec
    decision = authorize_tool_call(
        "object_3d_extent", spec, scope=scene.question_tool_scope,
        profile=scene.evidence_profile, available_artifacts=scene.available_artifacts,
        gate=scene.metric_evidence_gate_result,
        supported_metric_tasks=spec.supported_metric_tasks,
        allowed_metric_tasks=_handle(scene).allowed_metric_tasks,
        question_type=scene.question_type)
    assert decision.metric_gate_applicable is True
    assert decision.metric_gate_result["gate_passed"] is True
    assert decision.metric_gate_result["status"] == "pass"


# --------------------------------------------------- §8.2 门三态 ----

def test_gate_status_is_derived_and_consistent():
    assert MetricEvidenceGateResult(
        gate_passed=True, gate_version="v",
        sub_results={"a": True}).status == "pass"
    assert MetricEvidenceGateResult(gate_passed=False, gate_version="v").status == "fail"
    na = MetricEvidenceGateResult(gate_passed=None, gate_version="v")
    assert na.status == "not_applicable" and na.gate_passed is None


def test_gate_rejects_status_that_contradicts_gate_passed():
    with pytest.raises(Exception):
        MetricEvidenceGateResult(gate_passed=None, gate_version="v", status="fail")
    with pytest.raises(Exception):
        MetricEvidenceGateResult(gate_passed=True, gate_version="v", status="not_applicable")


def test_gate_still_rejects_forged_pass_with_failing_subcondition():
    with pytest.raises(Exception):
        MetricEvidenceGateResult(gate_passed=True, gate_version="v",
                                 sub_results={"a": False})


def test_not_applicable_helper_shape():
    payload = not_applicable_gate(gate_version="g")
    assert payload["status"] == "not_applicable"
    assert payload["gate_passed"] is None
    assert payload["gate_version"] == "g"


# --------------------------------------------------- §6.1 原因码 ----

def test_unavailable_reason_vocabulary_covers_four_classes_plus_quality_gate():
    """词表 = §6.4 四类 **+** `quality_gate_not_passed`（用户 2026-09-27 裁定扩展）。

    本断言**取代**原先的 `== {四类}`：§6.4 的四类里没有"**运行成功但质量门未过**"
    （M4 主门未过、world frame 置信低、尺度自洽低于阈值…），那类情况此前只能留空 ——
    审计读到的是"没记原因"，而不是"跑了但不达标"。请用户裁决后，用户裁定**扩展词表**，
    因此这里从"恰好四类"改为"四类必须都在 + 第五值必须可表达 + 不得有未登记取值"。
    这不是放松断言：四类仍逐一核对，第五值另有生产者接线测试
    （`test_v9_evidence_reasons.py`，含"与 not_run / producer_failed 不塌成同一个"）。
    """
    four = {"not_run", "producer_failed", "invalidated", "unsupported"}
    assert four <= UNAVAILABLE_REASON_CODES, "§6.4 的四类一个都不能少"
    assert "quality_gate_not_passed" in UNAVAILABLE_REASON_CODES, \
        "第五值（用户 2026-09-27 裁定）必须可表达"
    assert UNAVAILABLE_REASON_CODES == four | {"quality_gate_not_passed"}, \
        "词表不得出现未登记的额外取值"


def test_state_reasons_accepts_documented_codes_and_rejects_others():
    base = dict(geometry_3d="unavailable", world_frame="unavailable",
                metric_scale="unavailable", object_detection="unavailable",
                track_consensus="unavailable")
    ok = EvidenceProfile(**base, state_reasons={"metric_scale": "not_run",
                                                "object_detection": "producer_failed"})
    assert ok.state_reasons["metric_scale"] == "not_run"
    with pytest.raises(Exception):
        EvidenceProfile(**base, state_reasons={"metric_scale": "guess"})
    with pytest.raises(Exception):
        EvidenceProfile(**base, state_reasons={"ghost": "not_run"})


def test_state_reasons_defaults_to_empty_meaning_unregistered():
    """没登记原因就是空 —— 不猜一个原因码填上。"""
    profile = _profile()
    assert profile.state_reasons == {}


# --------------------------------------------------- 端到端落盘 ----

def test_episode_trace_persists_authorization_receipts(tmp_path):
    from skill3d.adapters.episode_source import load_synthetic_items
    from skill3d.online.runner import OnlineRunConfig, run_episode

    items = load_synthetic_items("inner_validation",
                                 question_types=["object_counting"],
                                 frame_size=(120, 160))
    cfg = OnlineRunConfig(mode="mock_light", trace_dir=str(tmp_path / "t"),
                          memory_dir="")
    out = run_episode(items[0].episode, items[0].pixels, cfg,
                      geometry=items[0].geometry)
    receipts = out.episode_trace.authorization_receipts
    assert receipts, "授权收据必须落盘（§17.1 Tool 层）"
    for receipt in receipts:
        assert set(RECEIPT_FIELDS) <= set(receipt)
        assert receipt["episode_id"] == items[0].episode.qa_id
