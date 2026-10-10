"""G-65/G-66 单测：过程指标聚合与多 seed 统计（§16.2/§16.3）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest

from skill3d.evaluation.multi_seed_aggregator import (
    format_main_table_row,
    macro_average_over_tasks,
)
from skill3d.evaluation.process_metrics import (
    aggregate_process_metrics,
    build_learning_curve,
    per_generation_delta,
    regression_rate,
    sample_efficiency,
    write_process_metrics_csv,
)


# ------------------------------------------------------------------ 伪产物 ----

@dataclass
class _Trace:
    results: list = field(default_factory=list)
    error_code: object = None
    steps: int = 1
    wallclock_s: float = 0.1


@dataclass
class _Result:
    tool: str


@dataclass
class _Verify:
    passed: bool


@dataclass
class _Outcome:
    final_state: str = "answer"
    is_mca: bool = True
    correct: bool = True
    mra_value: object = None
    answer_flags: list = field(default_factory=list)
    states: list = field(default_factory=list)
    scene_route: str = "full_3d"
    synthesis_source: str = "vllm"
    program_trace: object = None
    verify: object = None


def _outcome(final_state="answer", correct=True, is_mca=True, mra=None,
             tools=("euclidean_distance",), error=None, verify=True,
             flags=(), states=(), source="vllm", steps=2) -> _Outcome:
    return _Outcome(
        final_state=final_state, is_mca=is_mca, correct=correct, mra_value=mra,
        answer_flags=list(flags), states=list(states), synthesis_source=source,
        program_trace=_Trace(results=[_Result(t) for t in tools], error_code=error,
                             steps=steps, wallclock_s=0.2),
        verify=(_Verify(passed=verify) if verify is not None else None),
    )


# ------------------------------------------------------------------ G-65 ----

def test_process_metrics_core_rates():
    outs = [
        _outcome(),                                             # 成功
        _outcome(final_state="unanswerable", error="timeout", verify=False, steps=0),
        _outcome(final_state="answer_best_effort", flags=["no_tool_fallback"]),
        _outcome(final_state="unavailable", verify=None, steps=0),
        # 旧 trace 的 v5 门控 flag：v6 不再产生（G8 已退役，§20），但**读旧 trace**
        # 必须仍然识别（`process_metrics` 明确保留该 flag 只为回读历史落盘）
        _outcome(final_state="unanswerable", flags=["g8_size_reject"]),
    ]
    m = aggregate_process_metrics(outs)
    assert m.n_episodes == 5
    assert (m.n_answer, m.n_answer_best_effort, m.n_unanswerable, m.n_unavailable) == \
        (1, 1, 2, 1)
    assert m.program_success_rate == pytest.approx(3 / 5)     # 3 个 step>0 且无错误
    assert m.geometry_pass_rate == pytest.approx(3 / 4)       # 4 个有 verify
    assert m.fallback_rate == pytest.approx(1 / 5)            # best_effort 那次
    assert m.gated_rate == pytest.approx(1 / 5)               # g8_size_reject
    assert m.tool_calls_total == 5 and m.unique_tools == 1
    assert m.tool_reuse_rate == pytest.approx(1 / 5)
    assert m.wallclock_s_total == pytest.approx(1.0)


def test_process_metrics_empty_is_none_not_zero():
    m = aggregate_process_metrics([])
    assert m.n_episodes == 0
    assert m.program_success_rate is None and m.geometry_pass_rate is None
    assert m.fallback_rate is None and m.tool_reuse_rate is None
    assert any("不臆造 0" in n for n in m.notes)


def test_process_metrics_report_input_errors_separately():
    m = aggregate_process_metrics([
        _outcome(final_state="input_error", correct=False, is_mca=True,
                 tools=(), verify=None, steps=0),
    ])

    assert m.n_input_error == 1
    assert m.n_unavailable == 0
    assert m.n_unanswerable == 0
    assert m.coverage == 0.0


def test_process_metrics_flags_mock_light():
    outs = [_outcome(source="deterministic_stub") for _ in range(4)]
    m = aggregate_process_metrics(outs)
    assert m.stub_program_ratio == pytest.approx(1.0)
    assert any("mock_light" in n for n in m.notes)


def test_mean_regen_rounds_from_states():
    outs = [_outcome(states=["SYNTHESIZE_PROGRAM", "STATIC_CHECK"]),
            _outcome(states=["SYNTHESIZE_PROGRAM", "STATIC_CHECK",
                             "SYNTHESIZE_PROGRAM", "STATIC_CHECK"])]
    m = aggregate_process_metrics(outs)
    assert m.mean_regen_rounds == pytest.approx(0.5)          # (0 + 1)/2


def test_sample_efficiency_and_missing_target():
    curve = [(1, 0.2), (2, 0.35), (3, 0.55), (4, 0.6)]
    assert sample_efficiency(curve, target=0.5) == 3
    assert sample_efficiency(curve, target=0.9) is None
    m = aggregate_process_metrics([_outcome()], learning_curve=curve, target_metric=0.9)
    assert m.sample_efficiency is None
    assert any("未在曲线上达到" in n for n in m.notes)


def test_per_generation_delta_and_regression_rate():
    assert per_generation_delta([(1, 0.40), (2, 0.44), (3, 0.43)]) == \
        pytest.approx([0.04, -0.01])
    assert regression_rate([1.0, 0.0, 1.0], [1.0, 1.0, 0.0]) == pytest.approx(1 / 3)
    assert regression_rate([], []) is None
    m = aggregate_process_metrics([_outcome()], generations=[(1, 0.4), (2, 0.45)],
                                  before_scores=[1.0, 1.0], after_scores=[1.0, 0.0],
                                  bugs_found=3)
    assert m.per_generation_delta == pytest.approx([0.05])
    assert m.regression_rate == pytest.approx(0.5)
    assert m.bugs_found == 3


def test_learning_curve_and_csv(tmp_path):
    outs = [_outcome(correct=True), _outcome(correct=False),
            _outcome(is_mca=False, mra=0.8), _outcome(correct=True)]
    curve = build_learning_curve(outs, metric="combined", curve_csv=tmp_path / "c.csv")
    assert len(curve) == 4
    assert curve[0][1] == pytest.approx(1.0)
    assert curve[-1][1] == pytest.approx(2.8 / 4)
    assert (tmp_path / "c.csv").is_file()
    only_mca = build_learning_curve(outs, metric="mca")
    assert len(only_mca) == 3                                  # NA 题不计入 mca 曲线


def test_write_process_metrics_csv(tmp_path):
    m = aggregate_process_metrics([_outcome()], generations=[(1, 0.4), (2, 0.5)])
    out = write_process_metrics_csv(m, tmp_path / "sub" / "p.csv")
    assert out.is_file()
    lines = out.read_text(encoding="utf-8").splitlines()
    header, row = lines[0].split(","), lines[1].split(",")
    assert "program_success_rate" in header and "per_generation_delta" in header
    rec = dict(zip(header, row))
    assert json.loads(rec["per_generation_delta"]) == [0.1]   # 列表序列化为 JSON


# ------------------------------------------------------------------ G-66 ----


# --------------------------------------------------- v3 可靠性指标（D-3/§4 M6）----

class _Outcome:
    """最小 outcome 替身（覆盖 ProcessMetrics 读取的字段）。"""

    def __init__(self, **kw):
        self.final_state = "answer"
        self.is_mca = False
        self.correct = None
        self.mra_value = None
        self.answer_flags = []
        self.synthesis_source = "vllm"
        self.program_trace = None
        self.verify = None
        self.scene_route = ""
        self.states = []
        self.tool_contract_hits = 0
        self.replay_used = False
        self.trimmed_regen_used = False
        self.abstained = False
        for k, v in kw.items():
            setattr(self, k, v)


def test_reliability_metrics_separate_from_main_table():
    """v3：refusal/coverage/tool_contract 率与 coverage-conditioned 指标单独报告。"""
    ok = _Outcome(mra_value=0.8)
    best = _Outcome(final_state="answer_best_effort", mra_value=0.4,
                    answer_flags=["no_tool_fallback"])
    abstained = _Outcome(final_state="unanswerable", mra_value=0.0, abstained=True,
                         answer_flags=["tool_contract", "abstain"],
                         tool_contract_hits=2, replay_used=True)
    unavailable = _Outcome(final_state="unavailable", mra_value=None)

    m = aggregate_process_metrics([ok, best, abstained, unavailable])
    assert m.coverage == pytest.approx(0.5)                 # 2/4 作答
    assert m.refusal_rate == pytest.approx(0.25)            # 1/4 abstain/unanswerable
    assert m.abstain_rate == pytest.approx(0.25)
    assert m.tool_contract_rate == pytest.approx(0.25)
    assert m.tool_contract_recovered == 1
    # coverage-conditioned 只在作答 episode 上算（不混主表口径）
    assert m.coverage_conditioned_mra == pytest.approx(0.6)  # (0.8 + 0.4)/2
    assert any("tool_contract" in n for n in m.notes)


def test_tool_contract_call_count_from_program_trace():
    """tool_contract 调用数从 ProgramExecutionTrace.results 的 error_code 统计。"""
    from skill3d.schemas import ProgramExecutionTrace, ToolResult

    def _res(tool, code):
        return ToolResult(tool=tool, args={}, value="null", source="real",
                          request_digest="d", latency_ms=0.0, error="e", error_code=code)

    trace = ProgramExecutionTrace(
        program_id="p", calls=[], results=[_res("exists_in_scene", "tool_contract"),
                                          _res("room_size_m2", None),
                                          _res("reproject", "domain_value")],
        stdout_tail="", error_code="tool_contract", steps=3, wallclock_s=0.1)
    m = aggregate_process_metrics([_Outcome(program_trace=trace, tool_contract_hits=1)])
    assert m.tool_contract_calls == 1
    assert m.program_success_rate == pytest.approx(0.0)


# ----------------------------------------------- §8.3 官方 8 任务主表口径 ----

def _full_per_task() -> dict:
    return {
        "object_counting": {"n": 2, "n_mca": 0, "n_na": 2, "accuracy": None, "mra": 0.9},
        "object_abs_distance": {"n": 1, "n_mca": 0, "n_na": 1, "accuracy": None, "mra": 0.5},
        "object_size_estimation": {"n": 1, "n_mca": 0, "n_na": 1, "accuracy": None, "mra": 1.0},
        "room_size_estimation": {"n": 1, "n_mca": 0, "n_na": 1, "accuracy": None, "mra": 0.8},
        "object_rel_distance": {"n": 1, "n_mca": 1, "n_na": 0, "accuracy": 1.0, "mra": None},
        "object_rel_direction": {
            "n": 3, "n_mca": 3, "n_na": 0, "accuracy": 0.3333,
            "levels": {
                "object_rel_direction_easy": {"n_mca": 1, "n_na": 0, "accuracy": 1.0},
                "object_rel_direction_medium": {"n_mca": 1, "n_na": 0, "accuracy": 0.0},
                "object_rel_direction_hard": {"n_mca": 1, "n_na": 0, "accuracy": 0.0},
            }},
        "route_planning": {"n": 1, "n_mca": 1, "n_na": 0, "accuracy": 0.0, "mra": None},
        "obj_appearance_order": {"n": 1, "n_mca": 1, "n_na": 0, "accuracy": 1.0, "mra": None},
    }


def test_macro_average_matches_official_table10_layout():
    """§8.3：8 任务简单算术平均 ×100；rel_direction 三档先等权聚合后进平均。"""
    m = macro_average_over_tasks(_full_per_task())
    assert m["scale"] == 100 and m["n_tasks"] == 8 and m["missing_tasks"] == []
    # rel_direction：三档 (1.0, 0.0, 0.0) 等权 → 33.33（不是按 episode 平均的 33.33 以外值）
    assert m["per_task_x100"]["object_rel_direction"] == pytest.approx(33.33, abs=0.01)
    assert m["level_detail_x100"]["object_rel_direction"][
        "object_rel_direction_easy"] == pytest.approx(100.0)
    # 8 任务算术平均
    expected = np.mean([90.0, 50.0, 100.0, 80.0, 100.0, 33.3333, 0.0, 100.0])
    assert m["avg_x100"] == pytest.approx(expected, abs=0.01)
    row = format_main_table_row("Ours", _full_per_task())
    assert row.startswith("Ours | ") and row.count("|") == 9      # name + Avg + 8 任务


def test_macro_average_reports_missing_tasks_instead_of_zero():
    """未覆盖任务不计入平均（不臆造 0），并在 missing_tasks 里显式列出。"""
    m = macro_average_over_tasks({"object_counting": {"n": 1, "n_na": 1, "n_mca": 0,
                                                      "accuracy": None, "mra": 1.0}})
    assert m["n_tasks"] == 1 and m["avg_x100"] == pytest.approx(100.0)
    assert len(m["missing_tasks"]) == 7
