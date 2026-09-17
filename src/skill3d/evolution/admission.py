"""M17/M19 causal admission 硬门（全确定性，硬约束 13：GPT-6 建议不能覆盖）：

ci_significant（CI 下界>0 且 delta ≥ min_delta）
+ slice_no_regression + no_leakage + cross_scene_multisample（≥N_min）
+ within_budget + sim2real_robust（MVP 恒 True 占位）
→ AdmissionDecision.promotes。任何硬门不过即不晋升。
"""

from __future__ import annotations

import uuid

from skill3d.schemas import AdmissionDecision, CounterexampleCase, PairedOutcome

# TODO_CALIBRATE：起始参考值（与 configs/admission_thresholds.yaml 对齐）
MIN_DELTA_REQUIRED = 0.02
N_MIN_CROSS_SCENE = 3


def evaluate_admission(
    candidate_id: str,
    paired_outcomes: list[PairedOutcome],
    counterexamples: list[CounterexampleCase] | None = None,
    no_leakage: bool = True,
    n_cross_scene: int = 0,
    within_budget: bool = True,
    sim2real_robust: bool = True,  # MVP 恒 True 占位
    min_delta_required: float = MIN_DELTA_REQUIRED,
    n_min: int = N_MIN_CROSS_SCENE,
    gpt6_review: str | None = None,
) -> AdmissionDecision:
    """硬门评估（确定性）。gpt6_review 仅为事后引用，绝不影响 promotes。"""
    if not paired_outcomes:
        return AdmissionDecision(
            decision_id=f"adm-{uuid.uuid4().hex[:12]}",
            candidate_id=candidate_id, paired_outcomes=[],
            counterexamples=counterexamples or [],
            min_delta_required=min_delta_required,
            ci_significant=False, slice_no_regression=False, no_leakage=no_leakage,
            cross_scene_multisample=False, within_budget=within_budget,
            sim2real_robust=sim2real_robust, promotes=False,
            reason="无 paired_outcome", gpt6_review=gpt6_review)

    primary = paired_outcomes[0]
    ci_significant = primary.ci95_lo > 0 and primary.delta >= min_delta_required
    slice_ok = all(o.slice_no_regression for o in paired_outcomes)
    budget_ok = within_budget and all(o.within_budget for o in paired_outcomes)
    cross_scene_ok = n_cross_scene >= n_min

    gates = {
        "ci_significant": ci_significant,
        "slice_no_regression": slice_ok,
        "no_leakage": no_leakage,
        "cross_scene_multisample": cross_scene_ok,
        "within_budget": budget_ok,
        "sim2real_robust": sim2real_robust,
    }
    promotes = all(gates.values())
    failed = [k for k, v in gates.items() if not v]
    reason = "全部硬门通过" if promotes else f"硬门未过: {failed}"

    return AdmissionDecision(
        decision_id=f"adm-{uuid.uuid4().hex[:12]}",
        candidate_id=candidate_id,
        paired_outcomes=paired_outcomes,
        counterexamples=counterexamples or [],
        min_delta_required=min_delta_required,
        ci_significant=ci_significant,
        slice_no_regression=slice_ok,
        no_leakage=no_leakage,
        cross_scene_multisample=cross_scene_ok,
        within_budget=budget_ok,
        sim2real_robust=sim2real_robust,
        promotes=promotes,
        reason=reason,
        gpt6_review=gpt6_review,
    )
