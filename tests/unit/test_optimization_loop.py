"""M19 优化循环测试：outer 失败后不再产新 patch；parent_version 链完整（硬约束 10/11）。"""

from skill3d.evolution.optimization_loop import (
    CandidateArchive,
    TestReport,
    run_optimization_loop,
)
from skill3d.schemas import BudgetLimit, CandidateRevision

_LIMIT = BudgetLimit(max_gpt6_tokens=500000, max_gpu_hours=100.0,
                     max_rollouts=50, max_revisions=10, patience=3)


def _rev(rid: str, parent: str | None) -> CandidateRevision:
    return CandidateRevision(
        revision_id=rid, root_candidate_id="cand-1", parent_version=parent,
        candidate_type="skill", spec_content=f"spec-{rid}", status="draft",
        induction_trace_refs=["t1"], evidence_lineage_ref="",
        created_by="gpt6_revision" if parent else "gpt6_induction",
        created_at="2026-09-17T00:00:00Z")


def test_outer_failure_rejects_without_new_patch():
    """L3 失败即 reject，绝不调用 revise_fn（硬约束 10）。"""
    revise_calls: list[str] = []

    def test_fn(rev, level):
        return TestReport(passed=(level != "L3_outer_holdout"), feedback=level)

    def revise_fn(current, feedback):
        revise_calls.append(current.revision_id)
        return _rev(f"rev-{len(revise_calls)}", current.revision_id)

    run = run_optimization_loop(_rev("rev-0", None), test_fn, revise_fn,
                                budget_limit=_LIMIT)
    assert run.status == "rejected"
    assert run.termination_reason == "outer_failed_no_revise"
    assert revise_calls == []  # outer 失败后无新 patch
    assert run.revision_history == ["rev-0"]


def test_inner_iterate_then_promote_parent_chain():
    """L1 失败→修订→新版本→最终 promote；parent_version 链完整（不可变）。"""
    revised: list[CandidateRevision] = []

    def test_fn(rev, level):
        if rev.revision_id == "rev-0" and level == "L1_minimal_slice":
            return TestReport(passed=False, feedback="l1 fail")
        return TestReport(passed=True)

    def revise_fn(current, feedback):
        new = _rev(f"rev-{len(revised) + 1}", current.revision_id)
        revised.append(new)
        return new

    run = run_optimization_loop(_rev("rev-0", None), test_fn, revise_fn,
                                budget_limit=_LIMIT)
    assert run.status == "promoted"
    assert run.revision_history == ["rev-0", "rev-1"]
    # parent_version 链完整：rev-1 的父是 rev-0，rev-0 无父
    assert revised[0].parent_version == "rev-0"
    assert run.current_revision_id == "rev-1"


def test_patience_exhaustion_rejects():
    def test_fn(rev, level):
        return TestReport(passed=False, feedback="always fail")

    counter = {"n": 0}

    def revise_fn(current, feedback):
        counter["n"] += 1
        return _rev(f"rev-{counter['n']}", current.revision_id)

    run = run_optimization_loop(_rev("rev-0", None), test_fn, revise_fn,
                                budget_limit=_LIMIT)
    assert run.status == "rejected"
    assert run.termination_reason == "patience_exhausted"
    assert run.patience_counter == _LIMIT.patience


def test_similar_to_rejected_auto_reject():
    archive = CandidateArchive()
    archive.add_rejected("spec-rev-0")  # 与 root 完全相同 → 相似度 1.0 > 0.85
    run = run_optimization_loop(
        _rev("rev-0", None), lambda r, l: TestReport(True),
        lambda c, f: c, archive=archive, has_new_evidence=False)
    assert run.status == "rejected"
    assert run.termination_reason == "similar_to_rejected_no_new_evidence"
