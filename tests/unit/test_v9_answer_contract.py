"""v9 P2：答案合同（§10.1/§4）与工具归因核验（§12）。

守四件事：

1. **规范示例必须能跑**：§10.2 的 `ReturnAnswer(AnswerPayload(...))` 曾被 M9 静态
   检查判为"未定义函数调用"而拒掉 —— 规范自己的示例通不过实现；
2. **单位不靠猜**：`unit` 按 §4 的"题型 → 单位"官方合同登记；历史短写
   `ReturnAnswer(value)` 走显式适配器，basis 保守记 `mixed` 并留问题记录；
3. **声明由框架核验**：`declared_used_result_ids` 与 `verified_used_result_ids`
   分开记；`observed` 与 `ignored` 也分离（调用 ≠ 使用）；
4. **不凭归因问题否决预测**：无法充分证实的 `tool_derived` 降级为 `mixed`，
   答案照常进分。
"""

from __future__ import annotations

import pytest

from skill3d.sandbox.ast_guard import ast_guard
from skill3d.schemas import (
    AnswerPayload,
    AttributionLedger,
    CANONICAL_UNIT_BY_QUESTION_TYPE,
    DERIVATION_OPS,
    parse_answer_payload,
    verify_attribution,
)
from skill3d.schemas.answer import ADAPTER_VERSION_LEGACY_SHORT_FORM
from skill3d.schemas.answer import EpisodeStatus  # noqa: F401  (类型契约存在性)

SPEC_EXAMPLE = (
    "def solve(ctx):\n"
    "    # value 是在线 agent 对当前图片的预测。\n"
    "    return ReturnAnswer(AnswerPayload(\n"
    "        value=4, unit=\"count\", basis=\"visual_estimate\",\n"
    "        used_result_ids=[], derivation=None,\n"
    "    ))\n"
)


# ------------------------------------------------------- §10.2 规范示例可运行 ----

def test_spec_zero_tool_example_passes_static_check():
    """§10.2 的零工具示例必须通过 M9 静态检查（此前被拒）。"""
    check = ast_guard(SPEC_EXAMPLE, allowed_tools=set())
    assert check.ok, check.violations
    assert check.allowed_tool_calls == []


def test_answer_payload_is_reserved_and_cannot_be_reassigned():
    check = ast_guard("AnswerPayload = 1\nReturnAnswer(1)\n", allowed_tools=set())
    assert not check.ok
    assert any("AnswerPayload" in v for v in check.violations)


def test_control_interface_not_blocked_by_empty_tool_face():
    """§6.3：工具面为空时仍必须接收图片与提交接口。"""
    check = ast_guard(SPEC_EXAMPLE, allowed_tools=set())
    assert check.ok


# ------------------------------------------------------------- §4 单位合同 ----

def test_unit_contract_covers_all_eight_canonical_types():
    """§4 的表是单位的唯一来源，八题型必须齐备。"""
    assert set(CANONICAL_UNIT_BY_QUESTION_TYPE) == {
        "object_counting", "object_abs_distance", "object_size_estimation",
        "room_size_estimation", "object_rel_distance", "object_rel_direction",
        "route_planning", "obj_appearance_order"}
    assert CANONICAL_UNIT_BY_QUESTION_TYPE["object_counting"] == "count"
    assert CANONICAL_UNIT_BY_QUESTION_TYPE["object_abs_distance"] == "m"
    assert CANONICAL_UNIT_BY_QUESTION_TYPE["object_size_estimation"] == "cm"
    assert CANONICAL_UNIT_BY_QUESTION_TYPE["room_size_estimation"] == "m2"
    for t in ("object_rel_distance", "object_rel_direction", "route_planning",
              "obj_appearance_order"):
        assert CANONICAL_UNIT_BY_QUESTION_TYPE[t] == "option"


def test_full_payload_passes_through_unadapted():
    payload = AnswerPayload(value="A", unit="option", basis="visual_estimate")
    got, problems = parse_answer_payload(payload, question_type="route_planning")
    assert got is payload and problems == []


def test_legacy_short_form_is_explicitly_adapted_and_not_claimed_as_tool_derived():
    """历史短写：单位按题型合同登记，basis 保守记 mixed，且留下适配记录。"""
    payload, problems = parse_answer_payload("20", question_type="room_size_estimation")
    assert payload.value == "20"
    assert payload.unit == "m2", "单位应来自 §4 题型合同"
    assert payload.basis == "mixed", "模型没声明工具贡献 → 不得记 tool_derived"
    assert payload.used_result_ids == [] and payload.derivation is None
    assert any(ADAPTER_VERSION_LEGACY_SHORT_FORM in p for p in problems)


def test_legacy_short_form_with_unknown_type_does_not_guess_unit():
    payload, problems = parse_answer_payload("7", question_type="mystery_task")
    assert payload.basis == "mixed"
    assert any("无 §4 单位合同" in p for p in problems)


# ------------------------------------------------------- §12 归因核验 ----

def _payload(**over) -> AnswerPayload:
    base = dict(value="3", unit="count", basis="tool_derived",
                used_result_ids=["r1"],
                derivation={"op": "count", "input_result_ids": ["r1"],
                            "parameters": {"field": "category_name"}})
    base.update(over)
    return AnswerPayload(**base)  # type: ignore[arg-type]


def test_derivation_op_must_be_registered():
    assert _payload().problems() == []
    bad = _payload(derivation={"op": "exec_python", "input_result_ids": ["r1"],
                               "parameters": {}})
    assert any("不在登记操作" in p for p in bad.problems())


def test_derivation_cannot_reference_undeclared_results():
    bad = _payload(derivation={"op": "field", "input_result_ids": ["r9"],
                               "parameters": {}})
    assert any("未在 used_result_ids 声明" in p for p in bad.problems())


def test_derivation_rejects_unregistered_keys():
    bad = _payload(derivation={"op": "field", "input_result_ids": ["r1"],
                               "parameters": {}, "code": "os.system('rm -rf /')"})
    assert any("未登记键" in p for p in bad.problems())


def test_registered_ops_cover_the_documented_repertoire():
    """§12：只允许登记操作 —— 字段提取/换算/计数/排序/argmin/选项映射。"""
    assert DERIVATION_OPS == {"field", "convert", "count", "sort", "argmin",
                              "option_map"}


def test_visual_estimate_cannot_declare_tool_results():
    bad = _payload(basis="visual_estimate")
    assert any("visual_estimate 却声明了工具结果" in p for p in bad.problems())


def test_tool_derived_requires_derivation_and_declaration():
    assert any("缺少可重放 derivation" in p
               for p in _payload(derivation=None).problems())
    assert any("未声明任何 used_result_ids" in p
               for p in _payload(used_result_ids=[], derivation=None).problems())


# ------------------------------------------- 台账：声明 / 核实 / 忽略 三分离 ----

def test_verified_attribution_separates_declared_verified_and_ignored():
    verdict = verify_attribution(
        _payload(used_result_ids=["r1"]),
        observed_result_ids=["r1", "r2"], succeeded_result_ids=["r1", "r2"],
        invalidated_result_ids=[], attempted_tool_calls=2, succeeded_tool_calls=2)
    assert verdict.downgraded is False
    assert verdict.verified_basis == "tool_derived"
    ledger: AttributionLedger = verdict.ledger
    assert ledger.attempted_tool_calls == 2 and ledger.succeeded_tool_calls == 2
    assert ledger.observed_result_ids == ["r1", "r2"]
    assert ledger.declared_used_result_ids == ["r1"]
    assert ledger.verified_used_result_ids == ["r1"]
    # 观测到但未声明 → ignored（§12：调用不等于使用）
    assert ledger.ignored_result_ids == ["r2"]


def test_unverifiable_tool_derived_downgrades_to_mixed_without_rejecting():
    """§12：无法充分验证 → 保守记 mixed 并保留问题，不否决格式合法的预测。"""
    verdict = verify_attribution(
        _payload(used_result_ids=["ghost"]),
        observed_result_ids=["r1"], succeeded_result_ids=["r1"],
        invalidated_result_ids=[])
    assert verdict.declared_basis == "tool_derived"
    assert verdict.verified_basis == "mixed" and verdict.downgraded is True
    assert verdict.ledger.verified_used_result_ids == []
    assert any("不在本 episode 工具台账中" in p for p in verdict.problems)
    assert "保守记 mixed" in verdict.note


def test_invalidated_evidence_is_dropped_and_downgrades():
    """§12/§14.1：使用已失效证据 → 不计入核实结果并降级（答案仍保留）。"""
    verdict = verify_attribution(
        _payload(used_result_ids=["r1"]),
        observed_result_ids=["r1"], succeeded_result_ids=["r1"],
        invalidated_result_ids=["r1"])
    assert verdict.ledger.verified_used_result_ids == []
    assert verdict.verified_basis == "mixed"
    assert any("已被级联撤销" in p for p in verdict.problems)


def test_failed_declared_result_is_not_verified():
    verdict = verify_attribution(
        _payload(used_result_ids=["r1"]),
        observed_result_ids=["r1"], succeeded_result_ids=[],
        invalidated_result_ids=[])
    assert verdict.ledger.verified_used_result_ids == []
    assert any("状态非 ok" in p for p in verdict.problems)


def test_mixed_is_accepted_as_declared():
    verdict = verify_attribution(
        _payload(basis="mixed", used_result_ids=["r1"], derivation=None),
        observed_result_ids=["r1"], succeeded_result_ids=["r1"],
        invalidated_result_ids=[])
    assert verdict.verified_basis == "mixed" and verdict.downgraded is False


def test_zero_tool_visual_estimate_is_legitimate():
    """§12：调用不等于使用 —— 零调用也可以是合法的视觉作答。"""
    verdict = verify_attribution(
        AnswerPayload(value="4", unit="count", basis="visual_estimate"),
        observed_result_ids=[], succeeded_result_ids=[], invalidated_result_ids=[],
        attempted_tool_calls=0, succeeded_tool_calls=0)
    assert verdict.verified_basis == "visual_estimate"
    assert verdict.downgraded is False and verdict.problems == []


# ------------------------------------------------- 端到端：合同进 trace ----

def test_mock_light_episode_records_answer_contract_and_attribution(tmp_path):
    """每个 episode 都要落盘答案合同、核验后的 basis 与六字段台账。"""
    from skill3d.adapters.episode_source import load_synthetic_items
    from skill3d.online.runner import OnlineRunConfig, run_episode

    items = load_synthetic_items("inner_validation",
                                 question_types=["room_size_estimation"],
                                 frame_size=(120, 160))
    cfg = OnlineRunConfig(mode="mock_light", trace_dir=str(tmp_path / "t"))
    out = run_episode(items[0].episode, items[0].pixels, cfg,
                      geometry=items[0].geometry)
    trace = out.episode_trace
    assert trace.episode_status == "answered"
    assert trace.answer_basis in ("visual_estimate", "tool_derived", "mixed")
    assert trace.answer["unit"] == "m2", trace.answer
    assert trace.answer["basis"] == trace.attribution["declared_basis"]
    # §12 六字段台账齐备
    ledger = trace.attribution["ledger"]
    for key in ("attempted_tool_calls", "succeeded_tool_calls",
                "observed_result_ids", "declared_used_result_ids",
                "verified_used_result_ids", "ignored_result_ids"):
        assert key in ledger, key
    # 声明的每一项都必须被核实或进问题记录（不允许静默丢弃）
    assert set(ledger["verified_used_result_ids"]) <= set(
        ledger["declared_used_result_ids"])


def test_double_submission_is_blocked_at_runtime():
    """§10.2：提交后不得提交第二次 —— 此前第二次会静默覆盖首个答案。"""
    from skill3d.sandbox.kernel import AnswerTerminate, _AnswerSlot
    from skill3d.tools.contract import AnswerAlreadyGiven

    slot = _AnswerSlot()
    slot._question_type = "object_counting"  # noqa: SLF001
    with pytest.raises(AnswerTerminate):
        slot("4")
    assert slot.answer == "4" and slot.payload is not None
    assert slot.payload.unit == "count"

    with pytest.raises(AnswerAlreadyGiven):
        slot("9")
    assert slot.answer == "4", "第二次提交不得覆盖首个答案"
    assert str(slot.payload.value) == "4"


def test_slot_records_full_payload_when_model_declares_one():
    """模型声明完整合同时，槽位保留原始对象供 §12 核验（不是只留字符串）。"""
    from skill3d.sandbox.kernel import AnswerTerminate, _AnswerSlot

    slot = _AnswerSlot()
    slot._question_type = "object_counting"  # noqa: SLF001
    payload = AnswerPayload(value=7, unit="count", basis="tool_derived",
                            used_result_ids=["r1"],
                            derivation={"op": "count", "input_result_ids": ["r1"],
                                        "parameters": {}})
    with pytest.raises(AnswerTerminate):
        slot(payload)
    assert slot.payload is payload
    assert slot.adapter_problems == [], "完整合同不需要兼容适配"
    assert slot.answer == "7"
