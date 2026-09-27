"""M19 Candidate Optimization Loop：SYNTHESIZE→STATIC_CHECK→TEST(L1→L2→L3)→DIAGNOSE→REVISE。

红线：
- Inner（L1/L2）可反复迭代；Outer（L3）只跑一次，失败即 reject，绝不允许基于 outer
  失败再修订（硬约束 10）；
- 候选不可变：每次修订产新版本（parent_version 链），不原地修改（硬约束 11）；
- 预算/停止：BudgetUsage vs BudgetLimit；patience/max_revisions 停止；
- 与已 reject 候选相似度 > 0.85 且无新证据 → 自动 reject（失败留档防重复）。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable

from skill3d.schemas import BudgetLimit, BudgetUsage, CandidateRevision, OptimizationRun

# TODO_CALIBRATE：默认预算/停止条件（起始参考值）
DEFAULT_BUDGET_LIMIT = BudgetLimit(
    max_gpt6_tokens=500000, max_gpu_hours=100.0, max_rollouts=50,
    max_revisions=10, patience=3)
DUP_REJECT_THRESHOLD = 0.85  # TODO_CALIBRATE：与已 reject 候选相似度阈值

LEVELS = ("L1_minimal_slice", "L2_full_inner", "L3_outer_holdout")


@dataclass
class TestReport:
    """run_test_fn 返回：某级测试结果。"""

    passed: bool
    feedback: str = ""  # 失败类型摘要（不含答案）


@dataclass
class CandidateArchive:
    """失败留档与 revision 索引，防重复并支持最终版本晋升（§7）。"""

    rejected_specs: list[str] = field(default_factory=list)
    revisions: dict[str, CandidateRevision] = field(default_factory=dict)

    def add_rejected(self, spec_content: str) -> None:
        self.rejected_specs.append(spec_content)

    def remember(self, revision: CandidateRevision) -> None:
        self.revisions[revision.revision_id] = revision


    def similar_rejected(self, spec_content: str,
                         threshold: float = DUP_REJECT_THRESHOLD) -> list[str]:
        return [s for s in self.rejected_specs
                if SequenceMatcher(None, s, spec_content).ratio() > threshold]


def _new_run(root: CandidateRevision, limit: BudgetLimit) -> OptimizationRun:
    return OptimizationRun(
        run_id=f"run-{uuid.uuid4().hex[:12]}",
        root_candidate_id=root.root_candidate_id,
        current_revision_id=root.revision_id,
        budget_used=BudgetUsage(tokens=0, gpu_hours=0.0, rollouts=0, revisions=0),
        budget_limit=limit,
        patience_counter=0,
        revision_history=[root.revision_id],
        parallel_branches=[],
        selection_strategy="sequential",
        status="running",
        termination_reason=None,
    )


def run_optimization_loop(
    root: CandidateRevision,
    run_test_fn: Callable[[CandidateRevision, str], TestReport],
    revise_fn: Callable[[CandidateRevision, str], CandidateRevision],
    static_check_fn: Callable[[CandidateRevision], bool] | None = None,
    archive: CandidateArchive | None = None,
    budget_limit: BudgetLimit = DEFAULT_BUDGET_LIMIT,
    has_new_evidence: bool = False,
) -> OptimizationRun:
    """运行单根候选的优化循环，返回 OptimizationRun 终态。

    run_test_fn(revision, level) → TestReport；revise_fn(current, feedback) → 新版本。
    """
    run = _new_run(root, budget_limit)
    archive = archive if archive is not None else CandidateArchive()

    # 防重复：与已 reject 候选相似且无新证据 → 自动 reject（§7）
    if not has_new_evidence and archive.similar_rejected(root.spec_content):
        run.status = "rejected"
        run.termination_reason = "similar_to_rejected_no_new_evidence"
        archive.add_rejected(root.spec_content)
        return run

    current = root
    archive.remember(root)
    outer_attempted = False  # 硬约束 10：outer 只跑一次

    while run.status == "running":
        # 预算检查
        if run.budget_used.revisions >= budget_limit.max_revisions:
            run.status = "budget_exhausted"
            run.termination_reason = "max_revisions"
            archive.add_rejected(current.spec_content)
            break

        # STATIC_CHECK：失败 → 回 SYNTHESIZE（修订，有限次，受 max_revisions 约束）
        if static_check_fn is not None and not static_check_fn(current):
            feedback = "static_check_failed"
            current = revise_fn(current, feedback)
            archive.remember(current)
            run.budget_used.revisions += 1
            run.revision_history.append(current.revision_id)
            run.current_revision_id = current.revision_id
            continue

        # TEST L1 → L2 → L3
        l1 = run_test_fn(current, "L1_minimal_slice")
        run.budget_used.rollouts += 1
        if not l1.passed:
            run.patience_counter += 1
            if run.patience_counter >= budget_limit.patience:
                run.status = "rejected"
                run.termination_reason = "patience_exhausted"
                archive.add_rejected(current.spec_content)
                break
            current = revise_fn(current, l1.feedback)
            archive.remember(current)
            run.budget_used.revisions += 1
            run.revision_history.append(current.revision_id)
            run.current_revision_id = current.revision_id
            continue

        l2 = run_test_fn(current, "L2_full_inner")
        run.budget_used.rollouts += 1
        if not l2.passed:
            run.patience_counter += 1
            if run.patience_counter >= budget_limit.patience:
                run.status = "rejected"
                run.termination_reason = "patience_exhausted"
                archive.add_rejected(current.spec_content)
                break
            current = revise_fn(current, l2.feedback)
            archive.remember(current)
            run.budget_used.revisions += 1
            run.revision_history.append(current.revision_id)
            run.current_revision_id = current.revision_id
            continue

        # L3 outer：只跑一次；失败即 reject，绝不基于 outer 失败再修订（硬约束 10）
        if not outer_attempted:
            outer_attempted = True
            l3 = run_test_fn(current, "L3_outer_holdout")
            run.budget_used.rollouts += 1
            if l3.passed:
                run.status = "promoted"
                run.termination_reason = "outer_passed"
            else:
                run.status = "rejected"
                run.termination_reason = "outer_failed_no_revise"
                archive.add_rejected(current.spec_content)
            break

    return run
