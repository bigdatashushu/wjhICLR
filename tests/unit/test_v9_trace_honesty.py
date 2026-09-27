"""v9 P0：证据诚实性与 Schema 身份（§12.2 / §17.1 / §17.2）。

本文件守三类**已发生过的失真**：

1. `synthesis_source` 兜底把不认识的取值洗成 `vllm_ok` —— mock_light 的收口答案
   因此在 TraceRecord 里冒充过真实模型输出（§19.3 要消灭的正是这种混写）；
2. 执行前被拒绝的调用被记成 `status="ok"`、`result_id=""` —— 可能作为"有效观测"
   回灌给模型（§14.1 / §17.1"失败也有 result_id"）；
3. 旧版 episode trace 被静默升格成当前 Schema —— 新增字段的默认值会让"当时没记"
   读成"当时不存在"（§17.2）。

轮次原因（`round_trigger` / `finalization_used`）与程序来源分开记录是 v9 §12.2 的
要求；这里同时守住"分开之后不再互相顶替"。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.online.recovery import collect_validated
from skill3d.online.runner import _synthesis_source_enum
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import ObjectInstance, SceneState, ToolResult
from skill3d.schemas.evidence import (
    GATE_SUBCONDITIONS,
    GATE_VERSION,
    EvidenceProfile,
    MetricEvidenceGateResult,
)
from skill3d.schemas.legacy import LegacyEpisodeTrace, LegacyEpisodeTraceError
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES
from skill3d.schemas.trace import (
    EPISODE_TRACE_SCHEMA_VERSION,
    KNOWN_SYNTHESIS_SOURCES,
    EpisodeTrace,
)
from skill3d.legacy.readers import (
    assert_runtime_eligible,
    read_episode_trace,
    read_legacy_episode_trace,
)
from skill3d.tools import REGISTRY
from skill3d.tools.contract import available_artifacts_for, question_tool_scope_of
from skill3d.tools.scene_handle import SceneHandle

CURRENT = EPISODE_TRACE_SCHEMA_VERSION


# ----------------------------------------------------------------- synthesis_source ----

def test_known_synthesis_sources_round_trip_unchanged():
    for value in sorted(KNOWN_SYNTHESIS_SOURCES):
        assert _synthesis_source_enum(value) == value


def test_v5_alias_maps_to_mock_stub_not_model_output():
    """v5 的 `deterministic_stub` 是非模型输出 → 归一到 `mock_stub`。"""
    assert _synthesis_source_enum("deterministic_stub") == "mock_stub"


def test_no_model_output_aliases_are_not_claimed_as_vllm_ok():
    """空/none/unavailable 只允许断言"没拿到模型输出"。"""
    for value in ("", "none", "unavailable", None):
        assert _synthesis_source_enum(value) == "vllm_service_error"


@pytest.mark.parametrize("value", ["forced_answer", "forced_answer_final", "finalization",
                                   "mystery_source", "vllm"])
def test_unknown_source_is_never_rewritten_to_vllm_ok(value):
    """不认识的取值 → 空串（来源未知）。**绝不**冒充 `vllm_ok`。

    `forced_answer` / `finalization` 是 v8 写进 `synthesis_source` 的轮次原因：
    回读旧 trace 时无法判定其真实来源，只能记"未知"。
    """
    got = _synthesis_source_enum(value)
    assert got != "vllm_ok"
    assert got == ""


# ------------------------------------------------------- 执行前拒绝的失败归因 ----

def _profile(**over) -> EvidenceProfile:
    base = dict(geometry_3d="available", world_frame="available",
                metric_scale="available", object_detection="available",
                track_consensus="available", object_grounding="available")
    base.update(over)
    return EvidenceProfile(**base)


def _scene(question_type: str = "") -> SceneState:
    gate = MetricEvidenceGateResult(gate_passed=True, gate_version=GATE_VERSION,
                                    sub_results={n: True for n in GATE_SUBCONDITIONS})
    return SceneState(
        artifact_ref="test", scene_route="full_3d",
        question_tool_scope=question_tool_scope_of(
            "full_3d", metric_question=bool(question_type), gate_passed=True),
        available_artifacts=set(available_artifacts_for(
            "full_3d", question_type=question_type, gate_passed=True,
            metric_question=question_type in METRIC_TASK_TYPES)),
        evidence_profile=_profile(),
        metric_evidence_gate_result=gate,
        objects=["obj_0"], summary="s", question_type=question_type,
    )


def _kernel(*, tools_enabled: bool = True) -> RestrictedNamespaceKernel:
    scene = _scene()
    handle = SceneHandle(
        scene,
        objects=[ObjectInstance(
            obj_id="obj_0", category_name="table", mask_per_frame="",
            pointcloud_world="", centroid_world=[0.0, 0.0, 1.0],
            bbox=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], det_conf=0.9)],
        c2w_list=np.tile(np.eye(4), (2, 1, 1)),
        intrinsics=np.tile(np.eye(3), (2, 1, 1)))
    return RestrictedNamespaceKernel(REGISTRY, handle, frames=[], mode="real",
                                     tools_enabled=tools_enabled)


def test_pre_execution_denial_is_failed_and_has_trackable_result_id():
    """finalization 封锁下的调用：`status="failed"` + 可追踪 result_id（§17.1）。"""
    kernel = _kernel(tools_enabled=False)
    cell = kernel.run_cell('c = object_centroid("obj_0")\nReturnAnswer(c)\n')

    assert cell.error_code == "tool_contract"
    assert len(kernel.tool_results) == 1
    denied = kernel.tool_results[0]
    assert denied.status == "failed", "执行前被拒绝的调用不得记成成功"
    assert denied.result_id, "失败调用同样必须有 result_id（§17.1）"
    assert denied.source_tool == "object_centroid"
    assert denied.request_digest, "失败调用也要有可比的请求摘要"
    assert denied.evidence_version, "应记录产出该判定时的证据版本"


def test_denied_call_is_not_surfaced_as_validated_observation():
    """被拒绝的调用不得进入 validated observations（否则会当有效观测回灌）。"""
    kernel = _kernel(tools_enabled=False)
    kernel.run_cell('c = object_centroid("obj_0")\nReturnAnswer(c)\n')
    assert collect_validated(kernel) == []


def test_answer_already_given_denial_is_also_failed():
    """`ReturnAnswer` 之后再调工具 → 同样按失败归因，且有 result_id。

    模拟"程序吞掉了终结信号后继续跑"（kernel 注释所述场景）：直接置位控制槽，
    而不是调用它 —— 调用会抛 `AnswerTerminate`（BaseException，正常终结）。
    """
    kernel = _kernel()
    kernel.answer_slot.answer = "0"
    kernel.answer_slot.given = True
    kernel.run_cell('c = object_centroid("obj_0")\n')
    assert kernel.tool_results, "答后的调用要留痕（审计可回放）"
    denied = kernel.tool_results[-1]
    assert denied.status == "failed"
    assert denied.result_id
    assert collect_validated(kernel) == []


def test_failed_result_id_is_deterministic_across_replays():
    """同参数重复拒绝 → 各自可区分，但同一次运行序列可复现（§4 M17）。"""
    def run() -> list[str]:
        kernel = _kernel(tools_enabled=False)
        kernel.run_cell('a = object_centroid("obj_0")\n'
                        'b = object_centroid("obj_0")\nReturnAnswer(a)\n')
        return [r.result_id for r in kernel.tool_results]

    first, second = run(), run()
    assert first == second, "同 seed 重放必须字节级一致"
    assert len(set(first)) == len(first), "重复拒绝必须彼此可区分"


# --------------------------------------------------------------- Schema 身份 ----

def _current_trace_payload() -> dict:
    return {"episode_id": "qa-1", "qa_id": "qa-1", "final_state": "answer",
            "program_trace_ref": "", "geometry_check_ref": "", "evaluation_ref": "",
            "failure": None, "active_snapshot_ref": "S0-seed-20260925-v1",
            "schema_version": CURRENT}


def test_current_schema_payload_loads_and_declares_v9():
    trace = read_episode_trace(_current_trace_payload(), source_path="mem://t")
    assert isinstance(trace, EpisodeTrace)
    assert trace.schema_version == CURRENT


def test_episode_trace_default_identity_is_current_not_v6():
    """新构造的 EpisodeTrace 必须声明当前合同，而不是沿用 6.0。"""
    trace = EpisodeTrace(**_current_trace_payload())
    assert trace.schema_version == CURRENT != "6.0"


def test_legacy_trace_is_not_silently_upgraded():
    """6.0 记录必须 hard fail 到只读审计入口，不得静默补默认值。"""
    legacy = _current_trace_payload() | {"schema_version": "6.0"}
    legacy.pop("round_trigger", None)
    legacy.pop("finalization_used", None)
    with pytest.raises(LegacyEpisodeTraceError) as exc:
        read_episode_trace(legacy, source_path="data/v6_smoke/traces/episode_trace.jsonl")
    assert "read_legacy_episode_trace" in str(exc.value)

    carrier = read_legacy_episode_trace(legacy, source_path="legacy.jsonl")
    assert isinstance(carrier, LegacyEpisodeTrace)
    assert carrier.detected_schema_version == "6.0"
    assert carrier.eligible_for_runtime is False
    # 明确写出"当时没记这些事实"，而不是把它们读成不存在
    assert any("round_trigger" in w and "finalization_used" in w
               for w in carrier.warnings)


def test_legacy_trace_is_rejected_by_runtime_guard():
    carrier = read_legacy_episode_trace(
        _current_trace_payload() | {"schema_version": "6.0"}, source_path="x.jsonl")
    with pytest.raises(LegacyEpisodeTraceError):
        assert_runtime_eligible(carrier)


def test_unknown_schema_version_fails_closed():
    with pytest.raises(LegacyEpisodeTraceError):
        read_episode_trace(_current_trace_payload() | {"schema_version": "7.3"},
                           source_path="x.jsonl")


def test_trace_record_keeps_round_facts_separate_from_source():
    """`round_trigger`/`finalization_used` 与 `synthesis_source` 互不顶替（§12.2）。"""
    from skill3d.schemas.trace import TraceRecord

    rec = TraceRecord(episode_id="qa-1", synthesis_source="mock_stub",
                      round_trigger="finalize", finalization_used=True)
    assert rec.synthesis_source == "mock_stub", "来源字段不得被轮次原因覆盖"
    assert rec.round_trigger == "finalize"
    assert rec.finalization_used is True

    default = TraceRecord(episode_id="qa-1")
    assert default.synthesis_source == "", "未知来源不得默认成 vllm_ok"
    assert default.round_trigger == "initial"
    assert default.finalization_used is False


def test_tool_result_default_status_is_not_used_for_denials():
    """护栏：`ToolResult` 的 `status` 默认值是 ok，因此拒绝路径必须显式写 failed。"""
    assert ToolResult().status == "ok"
