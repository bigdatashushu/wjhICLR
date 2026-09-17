"""M17/M19 准入门测试：各硬门分别不过 → promotes=False；全过 → True（硬约束 13）。"""

from skill3d.evolution.admission import evaluate_admission
from skill3d.schemas import PairedOutcome


def _outcome(ci_lo=0.05, delta=0.05, slice_ok=True, budget=True) -> PairedOutcome:
    return PairedOutcome(
        pair_id="p1", snapshot_id="s1", arm_a_branch_id="a", arm_b_branch_id="b",
        n_episodes=30, metric="accuracy", mean_a=0.5, mean_b=0.5 + delta,
        delta=delta, ci95_lo=ci_lo, ci95_hi=ci_lo + 0.1, wilcoxon_p=0.01,
        slice_table={}, resource_cost={}, slice_no_regression=slice_ok,
        within_budget=budget)


def test_all_gates_pass_promotes():
    d = evaluate_admission("c1", [_outcome()], no_leakage=True,
                           n_cross_scene=3, within_budget=True)
    assert d.promotes is True
    assert d.reason == "全部硬门通过"


def test_ci_not_significant_blocks():
    d = evaluate_admission("c1", [_outcome(ci_lo=-0.01)], no_leakage=True,
                           n_cross_scene=3, within_budget=True)
    assert d.ci_significant is False and d.promotes is False


def test_delta_below_min_blocks():
    d = evaluate_admission("c1", [_outcome(ci_lo=0.001, delta=0.001)],
                           no_leakage=True, n_cross_scene=3, within_budget=True)
    assert d.promotes is False  # delta < min_delta_required(0.02)


def test_slice_regression_blocks():
    d = evaluate_admission("c1", [_outcome(slice_ok=False)], no_leakage=True,
                           n_cross_scene=3, within_budget=True)
    assert d.slice_no_regression is False and d.promotes is False


def test_leakage_blocks():
    d = evaluate_admission("c1", [_outcome()], no_leakage=False,
                           n_cross_scene=3, within_budget=True)
    assert d.no_leakage is False and d.promotes is False


def test_insufficient_cross_scene_blocks():
    d = evaluate_admission("c1", [_outcome()], no_leakage=True,
                           n_cross_scene=1, within_budget=True)  # < N_min=3
    assert d.cross_scene_multisample is False and d.promotes is False


def test_over_budget_blocks():
    d = evaluate_admission("c1", [_outcome(budget=False)], no_leakage=True,
                           n_cross_scene=3, within_budget=False)
    assert d.within_budget is False and d.promotes is False


def test_sim2real_blocks():
    d = evaluate_admission("c1", [_outcome()], no_leakage=True,
                           n_cross_scene=3, within_budget=True, sim2real_robust=False)
    assert d.promotes is False


def test_gpt6_review_cannot_override():
    # 硬约束 13：GPT-6 审查引用绝不改变硬门结果
    d = evaluate_admission("c1", [_outcome(ci_lo=-0.5)], no_leakage=True,
                           n_cross_scene=3, within_budget=True,
                           gpt6_review="gov-xxx")
    assert d.promotes is False and d.gpt6_review == "gov-xxx"
