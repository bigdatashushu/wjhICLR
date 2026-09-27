"""v9 P5：与规范**直接冲突**的三处修正（§17.3 编码前核验清单）。

1. §11.3（规范 L429 原文）："无围栏但完整程序**不计**解析恢复" ——
   此前该路径返回回退标记，把正常产出记成 `m8_parse_recovered`；
2. §14.3：候选准入必须在**该题型完整 inner 面板**上判定，`outer_holdout` 是快照
   冻结后的独立验证证据 —— 此前准入消费的是 outer，用 holdout 选代会让它失效；
3. §14.3 与 G-31 简化阶段门的冲突：简化不能导致"没有准入证据"。
"""

from __future__ import annotations

from skill3d.evolution.panel import FULL_LEVELS, admit
from skill3d.schemas import CandidateRevision
from skill3d.schemas.evolution import PairedOutcome
from skill3d.synthesis.program_assembler import extract_program_source_ex

PROGRAM = "def solve(ctx):\n    return ReturnAnswer(4)\n"


# --------------------------------------------- §11.3 无围栏不计解析恢复 ----

def test_unfenced_complete_program_is_not_parse_recovery():
    """§429："无围栏但完整程序不计解析恢复"。"""
    source, recovered = extract_program_source_ex(PROGRAM)
    assert source.strip() == PROGRAM.strip()
    assert recovered is False, "无围栏的完整程序属正常产出，不得记 m8_parse_recovered"


def test_fenced_program_is_still_the_clean_path():
    _, recovered = extract_program_source_ex(f"```python\n{PROGRAM}```\n")
    assert recovered is False


def test_unclosed_fence_prefix_is_still_recovery():
    """截断后的可解析前缀**是**回退（§11.3 允许并标记）。"""
    _, recovered = extract_program_source_ex(f"```python\n{PROGRAM}")
    assert recovered is True


def test_truncated_body_recovery_is_still_flagged():
    truncated = "```python\ndef solve(ctx):\n    x = 1\n    y = 2\n"
    _, recovered = extract_program_source_ex(truncated)
    assert recovered is True


# --------------------------------------------- §14.3 准入消费 inner 面板 ----

def _candidate() -> CandidateRevision:
    return CandidateRevision(
        revision_id="rev-1", root_candidate_id="cand-1", parent_version="1.0.0",
        candidate_type="skill", spec_content="{}", status="testing",
        induction_trace_refs=["t1"], evidence_lineage_ref="",
        created_by="gpt6_induction", created_at="2026-09-27T00:00:00Z")


def _outcome(delta: float, *, metric: str = "accuracy") -> PairedOutcome:
    return PairedOutcome(
        pair_id="p1", snapshot_id="s1", arm_a_branch_id="a", arm_b_branch_id="b",
        n_episodes=30, metric=metric, mean_a=0.5, mean_b=0.5 + delta,
        delta=delta, ci95_lo=max(0.0, delta - 0.01), ci95_hi=delta + 0.1,
        wilcoxon_p=0.01, slice_table={}, resource_cost={},
        slice_no_regression=True, within_budget=True)


def test_admit_takes_the_admission_panel_not_an_outer_panel():
    """§14.3：`admit` 的第二个参数是**准入面板**结果，跨场景计数取该面板。

    参数名与文档用词对齐（`panel_items`）：签名里不再出现 `outer_items`，
    避免调用方误把 holdout 结果当准入门。
    """
    import inspect

    params = inspect.signature(admit).parameters
    assert "panel_items" in params
    assert "outer_items" not in params, "签名不得再暗示 outer 可作为准入门"


def test_admission_decision_uses_the_panel_outcome_it_was_given():
    decision = admit(_candidate(), _outcome(0.10), panel_items=[],
                     no_leakage=True, n_min=1, min_delta_mca=0.02,
                     min_delta_mra=0.02)
    assert decision.candidate_id == "cand-1"
    assert len(decision.paired_outcomes) == 1
    assert decision.paired_outcomes[0].pair_id == "p1"
    assert decision.paired_outcomes[0].delta == 0.10


def test_levels_constant_keeps_full_inner_for_admission():
    assert "L2_full_inner" in FULL_LEVELS


def test_simplified_phase_gate_still_runs_the_admission_panel():
    """G-31 简化不能产生"没有准入证据"的晋升（§14.3）。"""
    from skill3d.evolution.offline_driver import OfflineDriver, OfflineDriverConfig

    cfg = OfflineDriverConfig(simplified_phase_gate=True)
    driver = OfflineDriver(cfg)
    levels = driver._admission_safe_levels()  # noqa: SLF001
    assert "L2_full_inner" in levels, "简化阶段门仍必须给出准入面板结果"
    assert levels == FULL_LEVELS
