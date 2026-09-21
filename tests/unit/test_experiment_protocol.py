"""Phase 6 §18 实验协议 + §19.4 split 访问审计 单测。

覆盖（对应 §23.1 Phase 6 的 DoD 与红线 9）：
- 三级隔离：run ledger 拦住第二次 `final`、拦住第二次 `outer`（除非显式 override +
  决策记录）、ledger 缺失/损坏时对 `final` fail-closed；
- 样本量三档：16/32/32+、3/3/5 seed、SE = 0.5/√N、"池 < 32 用全池不子采样到 4"；
- 噪声底：逐题一致率 + per-task 波动 + "多副本轮询不得用于对比"的 assert；
- paired A/B：两臂可比性断言（同重建产物 / 同 frame_set_hash …）；
- 统计：McNemar（合成表已知方向）、配对 bootstrap、效应量、Bonferroni（手算对照）、
  退化样本 → p=None + "退化样本 → 判为不显著"；
- 报告：95% CI（已知样本手算对照）、`evidence_state_breakdown` 分组与三个率；
- paper_eligible 四要件逐项拒绝 + 四个 `[待实验]` 项的 PoC 证据要求；
- 红线 9：outer 调出来的策略一律拒绝；§19.4 读 outer 必须留理由、跨 split 改动需决策记录。
"""

from __future__ import annotations

import json
import math

import pytest

from skill3d.evaluation.experiment_protocol import (
    DEGENERATE_NOTE,
    FORBIDDEN_SUBSAMPLE_PER_TASK,
    MIN_SEEDS_NOISE_FLOOR,
    MIN_SEEDS_PAPER,
    MISSING_SIGNATURE,
    PENDING_POC_ITEMS,
    SAMPLE_TIERS,
    SPLIT_OF_STAGE,
    AuditError,
    IsolationViolation,
    LedgerCorruptError,
    PairedABViolation,
    ProtocolError,
    RedLineViolation,
    RunLedger,
    SplitAccessLog,
    Stage,
    StrategyRecord,
    assert_isolation,
    assert_paired_ab,
    assert_round_robin_not_for_comparison,
    assert_strategy_paper_eligible,
    binary_se,
    binary_se_points,
    check_paired_ab,
    check_paper_eligible,
    check_round_robin_usage,
    check_sample_size,
    check_strategy_provenance,
    ci95_of_seed_values,
    coerce_stage,
    evidence_state_breakdown,
    format_sample_tiers,
    isolation_violations,
    noise_floor_report,
    normalize_answer_source,
    normalize_synthesis_source,
    paired_statistics,
    paired_statistics_by_task,
    per_task_ci95,
    required_seeds_for_pool,
    signature_key,
    strategy_record_from_file,
    trace_view,
    wilson_ci,
)
from skill3d.evaluation.multi_seed_aggregator import (
    MIN_SEEDS_FOR_MAIN_TABLE,
    per_question_agreement,
    per_question_seed_matrix,
)
from skill3d.evaluation.process_metrics import MOCK_SYNTHESIS_SOURCES
from skill3d.schemas import EpisodeTrace, TraceRecord

FULL_STATES = {c: "available" for c in (
    "geometry_3d", "world_frame", "metric_scale", "object_detection",
    "track_consensus", "temporal", "image_2d", "object_grounding")}
DEGRADED_STATES = dict(FULL_STATES, geometry_3d="degraded", metric_scale="unavailable")


def _trace(qa_id: str, *, states: dict | None = None, answer_source: str = "tool_program",
           scope: str = "metric_scope", abstained: bool = False,
           synthesis_source: str = "vllm_ok", final_state: str = "answer",
           failure_code: str | None = None) -> EpisodeTrace:
    return EpisodeTrace(
        episode_id=qa_id, qa_id=qa_id, final_state=final_state,
        program_trace_ref="", geometry_check_ref="", evaluation_ref=f"eval:{qa_id}",
        failure=None, active_snapshot_ref="genesis",
        evidence_states=dict(FULL_STATES if states is None else states),
        answer_source=answer_source, question_tool_scope=scope,
        abstained=abstained, synthesis_source=synthesis_source,
        failure_code=failure_code, frame_set_hash="fs-hash")


# ====================================================== §18.1 三级隔离 ----

def test_ledger_blocks_second_final_run(tmp_path):
    """final 完全隔离、只盲评一次：第二次一律拒绝（override 也不放行）。"""
    led = RunLedger(tmp_path / "ledger")
    led.initialize(actor="tester")
    assert led.check(Stage.FINAL).allowed
    perm = led.begin_run(Stage.FINAL, "run-f1", n_seeds=5, n_per_task=32)
    assert perm.allowed and perm.stage == "final"
    # 同一目录的另一个 RunLedger 实例（模拟另一进程）也必须看见
    other = RunLedger(tmp_path / "ledger")
    assert other.count(Stage.FINAL) == 1
    bad = other.check(Stage.FINAL)
    assert not bad.allowed and bad.code == "final_already_run"
    with pytest.raises(IsolationViolation):
        other.record(Stage.FINAL, "run-f2")
    # override 对 final 不适用
    assert not other.check(Stage.FINAL, override=True, note="想再跑一次").allowed
    with pytest.raises(IsolationViolation):
        other.record(Stage.FINAL, "run-f3", override=True, note="想再跑一次")
    # 被拒的尝试留痕（审计价值最高的记录）
    assert len(other.blocked_attempts(Stage.FINAL)) == 2
    assert other.count(Stage.FINAL) == 1


def test_ledger_fails_closed_when_missing_or_corrupt(tmp_path):
    """ledger 缺失/损坏 → 不得默认放行 final（fail-closed，§18.1/红线 5）。"""
    missing = RunLedger(tmp_path / "empty")
    perm = missing.check(Stage.FINAL)
    assert not perm.allowed and perm.code == "ledger_missing_fail_closed"
    with pytest.raises(IsolationViolation):
        missing.record(Stage.FINAL, "run-x")
    # inner/outer 首次运行允许（记录时创建 ledger）
    assert missing.check(Stage.INNER).allowed
    assert missing.check(Stage.OUTER).allowed

    corrupt_dir = tmp_path / "corrupt"
    corrupt_dir.mkdir()
    (corrupt_dir / RunLedger.FILENAME).write_text("{ not json", encoding="utf-8")
    corrupt = RunLedger(corrupt_dir)
    for stage in Stage:
        p = corrupt.check(stage)
        assert not p.allowed and p.code == "ledger_corrupt"
    # 结构非法（schema 不对）同样算损坏
    (corrupt_dir / RunLedger.FILENAME).write_text(
        json.dumps({"entries": []}), encoding="utf-8")
    with pytest.raises(LedgerCorruptError):
        corrupt.load()
    # 初始化不得覆盖既有账本
    missing.begin_run(Stage.INNER, "inner-first")
    assert missing.exists()
    with pytest.raises(IsolationViolation):
        missing.initialize()


def test_ledger_outer_once_then_override_with_decision_note(tmp_path):
    """outer 只跑一次；第二次必须显式 override **且**留决策记录（§18.1）。"""
    led = RunLedger(tmp_path)
    led.begin_run(Stage.OUTER, "outer-s1", n_seeds=3, n_per_task=32)
    blocked = led.check(Stage.OUTER)
    assert not blocked.allowed and blocked.code == "outer_already_run"
    assert blocked.requires_override
    with pytest.raises(IsolationViolation):
        led.record(Stage.OUTER, "outer-s2")
    # override 但没有决策记录 → 仍然拒绝
    no_note = led.check(Stage.OUTER, override=True, note="   ")
    assert not no_note.allowed and no_note.code == "outer_override_needs_note"
    # override + 决策记录 → 放行，并在账本里留痕
    perm = led.check(Stage.OUTER, override=True, note="仅因 GPU 掉卡重跑，不算二次验证")
    assert perm.allowed and perm.code == "outer_rerun_override"
    led.record(Stage.OUTER, "outer-s2", override=True,
               note="仅因 GPU 掉卡重跑，不算二次验证")
    assert led.count(Stage.OUTER) == 2
    summary = led.summary()
    assert summary["outer_rerun_override_used"] is True
    # inner 可反复迭代
    for i in range(4):
        led.begin_run(Stage.INNER, f"inner-{i}")
    assert led.count(Stage.INNER) == 4
    assert led.check(Stage.INNER).code == "inner_iteration"


def test_stage_rules_and_isolation_operations():
    """§18.1 的操作许可位：final 不碰离线模型/不写 Memory/Skill/不进沙箱/不重跑。"""
    assert coerce_stage("inner_validation") is Stage.INNER
    assert coerce_stage("outer_holdout") is Stage.OUTER
    assert coerce_stage("final_test") is Stage.FINAL
    assert SPLIT_OF_STAGE[Stage.FINAL] == "final_test"
    with pytest.raises(ProtocolError):
        coerce_stage("whatever")
    assert isolation_violations(Stage.INNER, ["strategy_change", "threshold_change",
                                              "memory_write", "sandbox_container"]) == []
    assert isolation_violations(Stage.OUTER, ["strategy_change"])       # outer 只验不调
    final_bad = isolation_violations(Stage.FINAL, [
        "offline_model", "memory_write", "skill_write", "sandbox_container", "rerun"])
    assert len(final_bad) == 5
    assert any("offline_model" in x for x in final_bad)
    # 未知操作名 fail-closed
    assert isolation_violations(Stage.INNER, ["vibes_based_tuning"])
    with pytest.raises(IsolationViolation):
        assert_isolation(Stage.FINAL, ["offline_model"])
    assert_isolation(Stage.INNER, ["strategy_change"])


# ==================================================== §18.2 样本量三档 ----

def test_sample_size_tiers_match_spec_table():
    """§18.2 三档表：16/32/32+ 题、3/3/5 seed、6.25/3.125/≤3.125 分、SE 12.5/8.8/≤8.8。"""
    inner, outer, final = (SAMPLE_TIERS[Stage.INNER], SAMPLE_TIERS[Stage.OUTER],
                           SAMPLE_TIERS[Stage.FINAL])
    assert (inner.n_per_task, inner.n_seeds) == (16, 3)
    assert (outer.n_per_task, outer.n_seeds) == (32, 3)
    assert (final.n_per_task, final.n_seeds) == (32, 5)
    assert inner.reported_weight == 6.25 and inner.weight_per_question_max == 6.25
    assert outer.reported_weight == 3.125 and outer.reported_se_points == 8.8
    assert final.reported_weight <= 3.125 and final.reported_se_points <= 8.8
    assert MIN_SEEDS_NOISE_FLOOR == MIN_SEEDS_FOR_MAIN_TABLE == 3
    assert MIN_SEEDS_PAPER == 5
    assert "SE = 0.5/√N" in format_sample_tiers()


def test_binary_se_is_half_over_sqrt_n():
    """§18.2：逐题二值结果 SE = 0.5/√N。"""
    assert binary_se(16) == pytest.approx(0.125)
    assert binary_se(32) == pytest.approx(0.5 / math.sqrt(32))
    assert binary_se_points(16) == pytest.approx(12.5)             # 一题 6.25 分 → SE 12.5
    assert binary_se_points(32) == pytest.approx(8.8388, abs=1e-3)  # 报告值 8.8
    assert binary_se_points(FORBIDDEN_SUBSAMPLE_PER_TASK) == pytest.approx(25.0)
    with pytest.raises(ValueError):
        binary_se(0)


def test_check_sample_size_reports_shortfalls():
    assert check_sample_size(Stage.INNER, 16, 3).ok
    assert check_sample_size(Stage.OUTER, 32, 3).ok
    assert check_sample_size(Stage.FINAL, 32, 5).ok
    # 题数不足
    r = check_sample_size(Stage.OUTER, 16, 3)
    assert not r.ok
    assert any("≥32" in s for s in r.shortfalls)
    # v5 的 4 题/题型被明确废止
    r4 = check_sample_size(Stage.OUTER, 4, 3, pool_size=32)
    assert not r4.ok
    assert any("4" in s and "废止" in s for s in r4.shortfalls)
    assert any("不得缩减题数" in s for s in r4.shortfalls)
    # seed 不足（final 需要 ≥5，§18.6）
    rf = check_sample_size(Stage.FINAL, 32, 3)
    assert not rf.ok and any("≥5 seed" in s for s in rf.shortfalls)
    # 池里有题却缩题 → 不许缩
    rp = check_sample_size(Stage.OUTER, 24, 3, pool_size=64)
    assert not rp.ok and any("不得缩减" in s for s in rp.shortfalls)


def test_pool_smaller_than_32_uses_whole_pool_and_more_seeds():
    """§18.2：池 < 32 题/题型 → 用池内全部题（不子采样到 4），靠 seed 补足运行次数。"""
    r = check_sample_size(Stage.OUTER, 20, 5, pool_size=20)
    assert r.ok and r.pool_limited and r.true_pool_size == 20
    assert r.effective_n_per_task == 20 and r.required_n_seeds == 5   # ceil(96/20)=5
    assert r.binary_se_points == pytest.approx(11.1803, abs=1e-3)     # 如实按真池大小算
    assert any("如实报告池大小" in n for n in r.notes)
    assert any("池" in n and "全部题" in n for n in r.notes)
    # seed 补不够 → 报缺
    r2 = check_sample_size(Stage.OUTER, 20, 3, pool_size=20)
    assert not r2.ok and any("5" in s for s in r2.shortfalls)
    # 子采样到 8 题（池里有 20）→ 缺
    r3 = check_sample_size(Stage.OUTER, 8, 5, pool_size=20)
    assert not r3.ok and any("池内全部题" in s for s in r3.shortfalls)
    # 池比档位要求大 → 不算 pool_limited
    r4 = check_sample_size(Stage.OUTER, 32, 3, pool_size=64)
    assert r4.ok and not r4.pool_limited
    assert required_seeds_for_pool(Stage.OUTER, 20) == 5
    assert required_seeds_for_pool(Stage.OUTER, 96) == 3              # 不低过档位要求
    assert required_seeds_for_pool(Stage.FINAL, 8) == 20              # ceil(160/8)


# ==================================================== §18.3 噪声底协议 ----

def test_noise_floor_agreement_and_fluctuation():
    per_q = {"q1": [True, True, True], "q2": [True, False, True],
             "q3": [True, None, None]}
    rep = noise_floor_report(per_q, per_task_scores={"t": [0.50, 0.55, 0.45]},
                             n_seeds=3)
    assert rep.per_question_agreement_rate == pytest.approx(1 / 2)   # q3 只有 1 seed → 不计
    assert rep.agreement["n_questions_single_seed"] == 1
    assert rep.agreement["split_questions"] == ["q2"]
    assert rep.ok_noise_floor and not rep.ok_paper                    # 3 seed < 5
    fl = rep.per_task_fluctuation["t"]
    assert fl["n_seeds"] == 3 and fl["range"] == pytest.approx(0.10)
    assert fl["std"] == pytest.approx(0.05)
    assert any("paper" in n for n in rep.notes)
    # 2 seed 连噪声底都不满足
    low = noise_floor_report({"q1": [True, True]}, n_seeds=2)
    assert not low.ok_noise_floor and not low.ok_paper
    empty = noise_floor_report({}, n_seeds=3)
    assert empty.per_question_agreement_rate is None                  # 分母 0 → None


def test_noise_floor_from_real_traces_per_seed():
    """§18.3：真实 EpisodeTrace 逐 seed 重复 → 逐题一致率（trace 矩阵 → 报告）。"""
    runs = {
        0: [_trace("q1"), _trace("q2")],
        1: [_trace("q1"), _trace("q2")],
        2: [_trace("q1"), _trace("q2")],
    }
    matrix = per_question_seed_matrix(runs, {"q1": True, "q2": False})
    assert matrix == {"q1": [True, True, True], "q2": [False, False, False]}
    assert per_question_agreement(matrix)["agreement_rate"] == pytest.approx(1.0)
    # 某个 seed 缺该题 → None（不臆造）
    runs[2] = [_trace("q1")]
    matrix2 = per_question_seed_matrix(runs, {"q1": True, "q2": False})
    assert matrix2["q2"] == [False, False, None]
    rep = noise_floor_report(matrix2, per_task_scores={"t": [1.0, 1.0, 0.5]},
                            n_seeds=3)
    assert rep.per_question_agreement_rate == pytest.approx(1.0)   # q2 剩 2 seed 且一致
    assert rep.per_task_fluctuation["t"]["range"] == pytest.approx(0.5)
    assert rep.ok_noise_floor and not rep.ok_paper


def test_round_robin_only_for_throughput():
    """§18.3：多副本轮询只用于吞吐，不得用于任何对比实验。"""
    assert check_round_robin_usage(False, "")[0] is True
    assert check_round_robin_usage(True, "throughput")[0] is True
    for bad in ("comparison", "ablation", "paper", "A/B"):
        ok, reason = check_round_robin_usage(True, bad)
        assert not ok and "不得用于任何对比实验" in reason
        with pytest.raises(ProtocolError):
            assert_round_robin_not_for_comparison(True, bad)
    # 没声明用途也 fail-closed
    assert check_round_robin_usage(True, "")[0] is False


# ==================================================== §18.4 paired A/B ----

def _arms(**over):
    base = {"frame_set_hash": "fs-1", "recon_artifact_hash": "recon-abc",
            "model": "qwen3-vl-8b", "template_version": "tpl-v6",
            "hardware": "4090-docker-digest", "evidence_signature": FULL_STATES}
    base.update(over)
    return dict(base)


def test_paired_ab_requires_same_frames_recon_model_template_hardware_evidence():
    """§16.3：两臂必须同 frames/同重建产物/同模型/同模板/同硬件/同 EvidenceProfile。"""
    a, b = _arms(), _arms()
    chk = check_paired_ab(a, b)
    assert chk.compatible and len(chk.shared) == 6
    assert_paired_ab(a, b)
    for field, bad_value in (("frame_set_hash", "fs-2"),
                             ("recon_artifact_hash", "recon-xyz"),
                             ("model", "other-model"),
                             ("template_version", "tpl-v5"),
                             ("hardware", "other-gpu"),
                             ("evidence_signature", DEGRADED_STATES)):
        mism = check_paired_ab(_arms(**{field: bad_value}), _arms())
        assert not mism.compatible, field
        assert any(field.split("_")[0] in m for m in mism.mismatches)
        with pytest.raises(PairedABViolation):
            assert_paired_ab(_arms(**{field: bad_value}), _arms())
    # 重建产物也可以给对象（按 model_dump/规范化哈希比较）
    art = {"artifact_id": "abc", "scale": 1.0}
    obj_a = _arms(**{"reconstruction_artifact": art})
    obj_a.pop("recon_artifact_hash")
    obj_b = _arms(**{"reconstruction_artifact": dict(art)})
    obj_b.pop("recon_artifact_hash")
    assert check_paired_ab(obj_a, obj_b).compatible
    obj_c = _arms(**{"reconstruction_artifact": {"artifact_id": "other"}})
    obj_c.pop("recon_artifact_hash")
    assert not check_paired_ab(obj_a, obj_c).compatible
    # 缺字段 → fail-closed（无法证明相同就不得当作相同的 A/B）
    incomplete = _arms()
    incomplete.pop("template_version")
    chk2 = check_paired_ab(incomplete, _arms())
    assert not chk2.compatible and "template_version" in chk2.missing


def test_mcnemar_on_synthetic_table_with_known_direction():
    """合成 paired 表：程序 10 胜 0 负 → 方向 program，精确 p = 2/2^10（可手算）。"""
    program = [True] * 10 + [True] * 6
    direct = [False] * 10 + [True] * 6
    res = paired_statistics(program, direct)
    assert res["n_pairs"] == 16 and res["n_program_win"] == 10
    assert res["n_direct_win"] == 0 and res["n_tie"] == 6
    assert res["mcnemar_method"] == "exact_binomial"
    assert res["mcnemar_p"] == pytest.approx(2 / 2 ** 10)
    assert res["p_bonferroni"] == pytest.approx(2 / 2 ** 10)
    assert res["direction"] == "program" and res["significant"] is True
    # Cliff's delta：10×16 个"程序更好"对 / 256（6 对 1v1 打平不算方向）
    assert res["cliffs_delta"] == pytest.approx(160 / 256)
    assert res["cliffs_delta"] > 0
    assert res["bootstrap_method"].startswith("scipy.bootstrap")
    # 反向：直答 10 胜 → 方向 direct，不得表述为程序增益
    rev = paired_statistics(direct, program)
    assert rev["direction"] == "direct" and rev["significant"] is False
    assert any("直答" in n for n in rev["notes"])


def test_mcnemar_matches_scipy_chi2_and_binomtest():
    """与 scipy 交叉核对（渐近卡方 + 精确二项两条路径都要对得上）。"""
    from scipy import stats as st

    # 渐近路径：b+c = 25 ≥ MCNEMAR_EXACT_MAX_DISCOUNT → 带连续性校正的卡方
    program = [True] * 20 + [False] * 5 + [True] * 5
    direct = [False] * 20 + [True] * 5 + [True] * 5
    res = paired_statistics(program, direct)
    assert res["mcnemar_method"] == "chi2_continuity_corrected"
    stat = res["mcnemar_statistic"]
    assert stat == pytest.approx((20 - 5 - 1) ** 2 / 25)          # (|b-c|-1)^2/(b+c)
    assert res["mcnemar_p"] == pytest.approx(float(st.chi2.sf(stat, 1)))
    # 精确路径与 scipy.stats.binomtest 一致
    exact = paired_statistics([True] * 7, [False] * 7)
    assert exact["mcnemar_p"] == pytest.approx(
        float(st.binomtest(7, 7, 0.5, alternative="two-sided").pvalue))


def test_degenerate_paired_sample_is_none_and_not_significant():
    """§18.4：退化样本 → p=None + "退化样本 → 判为不显著"，绝不呈现为显著。"""
    same = [True, False, True, False, True, False]
    res = paired_statistics(same, list(same))
    assert res["degenerate"] is True
    assert res["mcnemar_p"] is None and res["p_bonferroni"] is None
    assert res["significant"] is False
    assert DEGENERATE_NOTE in res["degenerate_note"]
    assert any("不显著" in n for n in res["notes"])
    assert res["direction"] == "tie"
    # 零个配对样本同样退化（不得报 p=1.0 冒充"已检验"）
    empty = paired_statistics([], [])
    assert empty["degenerate"] and empty["mcnemar_p"] is None
    assert empty["significant"] is False
    # bootstrap 主判据下退化样本同样不得显著
    boot = paired_statistics(same, list(same), primary="bootstrap")
    assert boot["significant"] is False


def test_bonferroni_correction_math_is_hand_checkable():
    """Bonferroni 数学：p_corrected = min(1, p × m)，与手算值逐位对齐。"""
    p = 2 / 2 ** 10                       # 0.001953125
    one = paired_statistics([True] * 10, [False] * 10)
    assert one["p_bonferroni"] == pytest.approx(p)
    five = paired_statistics([True] * 10, [False] * 10, n_comparisons=5)
    assert five["p_bonferroni"] == pytest.approx(0.009765625)          # p × 5
    assert five["alpha_corrected"] == pytest.approx(0.05 / 5)
    assert five["significant"] is True                                 # 0.0098 ≤ 0.05
    hundred = paired_statistics([True] * 10, [False] * 10, n_comparisons=100)
    assert hundred["p_bonferroni"] == pytest.approx(0.1953125)         # p × 100
    assert hundred["significant"] is False                             # 0.195 > 0.05
    capped = paired_statistics([True] * 4, [False] * 4, n_comparisons=1000)
    assert capped["p_bonferroni"] == pytest.approx(min(1.0, 0.125 * 1000))
    # 分切片时按切片数校正
    program = [True] * 8 + [True] * 8 + [True] * 8 + [True] * 8
    direct = [False] * 8 + [False] * 8 + [False] * 8 + [False] * 8
    by_task = paired_statistics_by_task(
        program, direct, ["t1"] * 8 + ["t2"] * 8 + ["t3"] * 8 + ["t4"] * 8)
    assert by_task["n_comparisons"] == 4
    assert by_task["by_task"]["t1"]["p_bonferroni"] == pytest.approx(
        min(1.0, by_task["by_task"]["t1"]["mcnemar_p"] * 4))


def test_paired_bootstrap_primary_and_effect_sizes():
    """BCa CI 参与判定（primary=bootstrap）+ 效应量存在且方向一致。"""
    program = [True] * 20 + [True] * 20
    direct = [False] * 20 + [True] * 20          # 差值有方差（1 与 0 混合）
    res = paired_statistics(program, direct, primary="bootstrap", n_resamples=999,
                            seed=3)
    assert res["bootstrap_ci_lo"] is not None and res["bootstrap_ci_lo"] > 0
    assert res["bootstrap_method"].startswith("scipy.bootstrap")
    assert res["significant"] is True
    assert res["cohens_d"] == pytest.approx(0.5 / 0.5063696835418333, rel=1e-6)
    assert 0 < res["cliffs_delta"] <= 1.0
    # 常数差（全胜）→ bootstrap 退化，但 McNemar 精确检验仍然有定义
    const = paired_statistics([True] * 6, [False] * 6)
    assert const["zero_variance"] is True and const["degenerate"] is False
    assert any("零方差" in n for n in const["notes"])
    assert const["mcnemar_p"] is not None


# ======================================================== §18.5 报告 ----

def test_wilson_ci_on_known_sample():
    """95% CI 手算对照：8/10 → Wilson (0.4902, 0.9433)。"""
    lo, hi = wilson_ci(8, 10)
    z, n, p = 1.96, 10, 0.8
    z2 = z * z
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = (z / (1 + z2 / n)) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    assert lo == pytest.approx(center - half) and hi == pytest.approx(center + half)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert lo < 0.8 < hi
    with pytest.raises(ValueError):
        wilson_ci(0, 0)
    # 多 seed 均值的 CI（[0.40, 0.45, 0.50]）：0.45 ± 1.96×0.05/√3
    lo2, hi2 = ci95_of_seed_values([0.40, 0.45, 0.50])
    assert lo2 == pytest.approx(0.45 - 1.96 * 0.05 / math.sqrt(3))
    assert hi2 == pytest.approx(0.45 + 1.96 * 0.05 / math.sqrt(3))
    assert ci95_of_seed_values([0.5]) == (None, None)              # n<2 → 不臆造


def test_per_task_ci95_reports_interval_and_sample_size():
    got = per_task_ci95({"object_abs_distance": [True] * 8 + [False] * 2,
                         "route_planning": [None, None]})
    a = got["object_abs_distance"]
    assert a["n"] == 10 and a["accuracy"] == pytest.approx(0.8)
    assert a["ci95"] == pytest.approx(list(wilson_ci(8, 10)))
    assert a["binary_se"] == pytest.approx(binary_se(10))
    assert got["route_planning"]["accuracy"] is None and "不臆造" in got["route_planning"]["note"]


def test_evidence_state_breakdown_groups_and_three_rates():
    """§18.5/§16.5：按证据状态分组报 选择率 / 覆盖率 / 胜负。"""
    records = [
        _trace("q1", answer_source="tool_program"),                      # 程序、覆盖
        _trace("q2", answer_source="direct_vlm_routed"),                 # 直答、覆盖
        _trace("q3", states=DEGRADED_STATES, answer_source="tool_program"),
        _trace("q4", states=DEGRADED_STATES, answer_source="abstain",
               abstained=True, scope="", final_state="unanswerable"),
    ]
    correct = {"q1": True, "q2": False, "q3": True, "q4": False}
    direct_correct = {"q1": False, "q2": False, "q3": False, "q4": False}
    rep = evidence_state_breakdown(
        records, correct_by_qa=correct, direct_correct_by_qa=direct_correct,
        task_by_qa={"q1": "t_abs", "q2": "t_abs", "q3": "t_abs", "q4": "t_route"})
    full = rep["by_evidence_state"][signature_key(FULL_STATES)]
    assert full["n"] == 2 and full["n_selected"] == 1
    assert full["selection_rate"] == pytest.approx(0.5)
    assert full["coverage"] == pytest.approx(1.0)
    assert (full["wins"], full["losses"]) == (1, 0)
    assert full["win_rate"] == pytest.approx(1.0)          # 1 胜 0 负 → 判别对里全胜
    deg = rep["by_evidence_state"][signature_key(DEGRADED_STATES)]
    assert deg["n"] == 2 and deg["n_selected"] == 1 and deg["n_abstain"] == 1
    assert deg["coverage"] == pytest.approx(0.5)           # 只有 1 条有覆盖证据
    # q3 是判别对（程序对/直答错）→ 该组 1 胜 0 负；q4 打平
    assert (deg["wins"], deg["losses"], deg["ties"]) == (1, 0, 1)
    assert deg["win_rate"] == pytest.approx(1.0)
    assert deg["answer_source_counts"] == {"tool_program": 1, "abstain": 1}
    # 只有打平对的组 → 胜负率 None（不臆造）
    tie_only = evidence_state_breakdown(
        [_trace("q8", states={"geometry_3d": "degraded"}, scope="s")],
        correct_by_qa={"q8": True}, direct_correct_by_qa={"q8": True})
    assert tie_only["by_evidence_state"][signature_key({"geometry_3d": "degraded"})][
        "win_rate"] is None
    assert rep["by_task"]["t_abs"]["n"] == 3
    assert rep["by_task"]["t_abs"]["selection_rate"] == pytest.approx(2 / 3)
    assert rep["n_records"] == 4
    # 缺证据快照的 trace 单独成桶，不得混进"全 available"
    missing = evidence_state_breakdown([_trace("q9", states={}, scope="")])
    assert MISSING_SIGNATURE in missing["by_evidence_state"]
    assert any(MISSING_SIGNATURE in n or "evidence_states" in n for n in missing["notes"])
    # 没给两臂正确性 → 胜负率全 None（不臆造）
    no_pairs = evidence_state_breakdown(records)
    assert all(g["win_rate"] is None for g in no_pairs["by_evidence_state"].values())
    assert any("胜负率" in n for n in no_pairs["notes"])
    # Skill 路径维度：显式映射优先
    ex = evidence_state_breakdown(records, skill_path_by_qa={"q1": "counting_family"})
    assert "counting_family" in ex["by_skill_path"]


def test_trace_view_normalizes_v5_fields_and_trace_record():
    """v5 取值回读（vllm/deterministic_stub/program）与 TraceRecord 的 evidence_profile。"""
    assert normalize_synthesis_source("deterministic_stub") == "mock_stub"
    assert normalize_synthesis_source("vllm") == "vllm_ok"
    assert normalize_synthesis_source("mock_stub") == "mock_stub"
    assert normalize_synthesis_source("weird_source") == "weird_source"  # 不洗成 vllm_ok
    assert normalize_answer_source("program") == "tool_program"
    assert normalize_answer_source("direct_vlm") == "direct_vlm_routed"
    assert "deterministic_stub" in MOCK_SYNTHESIS_SOURCES
    assert "mock_stub" in MOCK_SYNTHESIS_SOURCES
    legacy = {"episode_id": "e1", "synthesis_source": "deterministic_stub",
              "answer_source": "program",
              "evidence_profile": {"geometry_3d": "available",
                                   "metric_scale": "unavailable"},
              "question_tool_scope": "metric"}
    view = trace_view(legacy)
    assert view.qa_id == "e1" and view.synthesis_source == "mock_stub"
    assert view.selected_program and view.covered is True
    assert view.signature == signature_key({"geometry_3d": "available",
                                            "metric_scale": "unavailable"})
    rec = TraceRecord(episode_id="e2", evidence_profile=dict(FULL_STATES),
                      answer_source="tool_program", synthesis_source="vllm_ok",
                      partial_tool_recovery=True, recovery_count=2,
                      used_result_ids=["r1"])
    rv = trace_view(rec)
    assert rv.signature == signature_key(FULL_STATES) and rv.recovery_count == 2
    assert rv.partial_tool_recovery is True


# ================================================== §18.6 paper_eligible ----

def _poc_ok() -> dict:
    return {k: {"passed": True, "evidence_ref": f"receipt://{k}.json"}
            for k in PENDING_POC_ITEMS}


def _eligible_kwargs(**over) -> dict:
    kw = {
        "scene_ids_by_split": {"inner_validation": ["s1", "s2"],
                               "outer_holdout": ["s3"], "final_test": ["s4", "s5"]},
        "n_seeds": 5,
        "stat_gate": {"significant": True, "p_bonferroni": 0.01, "degenerate": False,
                      "direction": "program"},
        "modes": ["real"],
        "records": [],
        "poc_evidence": _poc_ok(),
    }
    kw.update(over)
    return kw


def test_paper_eligible_requires_all_four_requirements_and_poc_evidence():
    ok = check_paper_eligible(**_eligible_kwargs())
    assert ok.eligible and not ok.reasons
    assert set(ok.checks) == {"data_isolation", "seed_requirement",
                              "statistical_gate", "non_mock_evidence"}
    assert all(c.ok for c in ok.pending_poc.values())


@pytest.mark.parametrize("field,bad,needle", [
    ("scene_ids_by_split",
     {"inner_validation": ["s1", "s2"], "outer_holdout": ["s2", "s3"]},
     "不互斥"),
    ("n_seeds", 3, "seed"),
    ("stat_gate", None, "统计检验"),
    ("stat_gate", {"significant": False, "p_bonferroni": 0.4}, "统计门未通过"),
    ("stat_gate", {"significant": True, "p_bonferroni": None, "degenerate": True},
     "退化样本"),
    ("modes", ["mock_light"], "HC24"),
    ("modes", [], "来源不明"),
])
def test_paper_eligible_refuses_each_unmet_requirement(field, bad, needle):
    verdict = check_paper_eligible(**_eligible_kwargs(**{field: bad}))
    assert verdict.eligible is False
    assert any(needle in r for r in verdict.reasons), verdict.reasons


def test_paper_eligible_refuses_mock_episode_records():
    recs = [_trace("q1", synthesis_source="mock_stub")]
    v = check_paper_eligible(**_eligible_kwargs(records=recs))
    assert not v.eligible
    assert any("mock" in r for r in v.reasons)
    legacy = [{"episode_id": "q2", "synthesis_source": "deterministic_stub"}]
    v2 = check_paper_eligible(**_eligible_kwargs(records=legacy))
    assert not v2.eligible                       # v5 mock 名同样拦


def test_paper_eligible_refuses_pending_poc_items_without_evidence():
    """§18.6 末条：四个 `[待实验]` 项未提供 PoC 证据 → 不得进论文主表。"""
    for item in PENDING_POC_ITEMS:
        poc = _poc_ok()
        poc.pop(item)
        v = check_paper_eligible(**_eligible_kwargs(poc_evidence=poc))
        assert not v.eligible
        assert any(item in r for r in v.reasons)
    # 只给布尔、或没有 receipt 引用 → 同样拒绝（不可复核）
    for bad_ev in (True, {"passed": True}, {"passed": False, "evidence_ref": "x"}):
        poc = _poc_ok()
        poc["moge2_effect"] = bad_ev
        assert not check_paper_eligible(**_eligible_kwargs(poc_evidence=poc)).eligible
    # 完全不提供 poc_evidence → 四项全拒
    v = check_paper_eligible(**_eligible_kwargs(poc_evidence=None))
    assert not v.eligible
    assert sum(1 for r in v.reasons if "[待实验]" in r) == len(PENDING_POC_ITEMS)


# ================================== §22 红线 9 + §19.4 split 访问审计 ----

def test_paper_eligible_checks_ledger_isolation_when_provided(tmp_path):
    """§18.1 联动：outer 用过 override、或 final 不是恰好 1 次 → 不得 paper-eligible。"""
    led = RunLedger(tmp_path / "ledger")
    led.initialize()
    led.begin_run(Stage.OUTER, "outer-s1", n_seeds=3, n_per_task=32)
    v = check_paper_eligible(**_eligible_kwargs(ledger=led))
    assert not v.eligible and any("final" in r for r in v.reasons)   # 还没跑 final
    led.begin_run(Stage.FINAL, "final-1", n_seeds=5, n_per_task=32)
    ok = check_paper_eligible(**_eligible_kwargs(ledger=led))
    assert ok.eligible and ok.checks["isolation_ledger"].ok
    # outer 被 override 重跑 → 验证依据不干净
    led.record(Stage.OUTER, "outer-s2", override=True, note="GPU 掉卡重跑")
    bad = check_paper_eligible(**_eligible_kwargs(ledger=led))
    assert not bad.eligible
    assert any("override 重跑" in r for r in bad.reasons)
    # 不传 ledger 时不做这一项检查（checks 仍是四要件）
    assert set(check_paper_eligible(**_eligible_kwargs()).checks) == {
        "data_isolation", "seed_requirement", "statistical_gate", "non_mock_evidence"}


def test_split_access_audit_requires_reason_for_outer(tmp_path):
    """§19.4：读 outer 必须留理由（空理由 raise），访问落 JSONL 可回读。"""
    log = SplitAccessLog(tmp_path / "audit")
    with pytest.raises(AuditError):
        log.log_access("outer_holdout", "")
    with pytest.raises(AuditError):
        log.log_access(Stage.OUTER, "   ")
    assert log.reads_of("outer_holdout") == 0
    rec = log.log_access(Stage.OUTER, "核对 outer 覆盖的题型分布", actor="wangjh",
                         run_id="outer-s1", purpose="coverage_audit")
    assert rec["reason"] == "核对 outer 覆盖的题型分布"
    assert rec["split"] == "outer_holdout" and rec["stage"] == "outer"
    assert log.reads_of(Stage.OUTER) == 1
    rows = [json.loads(line) for line in
            (log.access_path).read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["reason"] and rows[-1]["at"]
    # inner 同样留痕（只是 outer 是硬要求）
    log.log_access("inner_validation", "例行内测")
    assert log.reads_of("inner") == 1
    # 新的 audit 目录是干净的（跨进程可查）
    assert SplitAccessLog(tmp_path / "audit").reads_of("outer") == 1


def test_cross_split_change_needs_explicit_decision_record(tmp_path):
    log = SplitAccessLog(tmp_path)
    ok, reason = log.check_cross_split_change("outer_holdout", "inner_validation")
    assert not ok and "缺少显式决策记录" in reason
    with pytest.raises(AuditError):
        log.assert_cross_split_change("outer_holdout", "inner_validation")
    with pytest.raises(AuditError):
        log.record_cross_split_decision("D0", "outer", "inner", "maybe", "note")
    with pytest.raises(AuditError):
        log.record_cross_split_decision("D0", "outer", "inner", "approved", "")
    # rejected 决策不构成放行
    log.record_cross_split_decision("D1", "outer_holdout", "inner_validation",
                                    "rejected", "理由不充分")
    assert not log.check_cross_split_change("outer_holdout", "inner_validation")[0]
    log.record_cross_split_decision("D2", "outer_holdout", "inner_validation",
                                    "approved", "发现 scene 泄漏，重新切分")
    ok2, reason2 = log.check_cross_split_change("outer", "inner_validation")
    assert ok2 and "D2" in reason2
    assert log.check_cross_split_change("inner_validation", "inner_validation")[0]


def test_red_line_9_rejects_outer_derived_strategy(tmp_path):
    """红线 9：策略/阈值只能在 inner 上定；outer 调出来的一律不得进论文。"""
    path = tmp_path / "distance_thresholds.yaml"
    path.write_text("quantile_q: 0.01\nvoxel_size: 0.02\n", encoding="utf-8")
    outer_rec = strategy_record_from_file(path, name="distance-q", derived_from="outer",
                                          note="在 outer 上看了趋势才定的 q")
    verdict = check_strategy_provenance(outer_rec)
    assert not verdict.paper_allowed
    assert any("红线 9" in r for r in verdict.reasons)
    with pytest.raises(RedLineViolation):
        assert_strategy_paper_eligible(outer_rec)
    # 未声明 derived_from → fail-closed
    unknown = StrategyRecord(name="x", derived_from="", content_hash="h")
    assert not check_strategy_provenance(unknown).paper_allowed
    # derived_from 写错名字 → 拒绝
    typo = StrategyRecord(name="x", derived_from="innner", content_hash="h")
    assert not check_strategy_provenance(typo).paper_allowed
    # final 上定策略同样拒绝
    final_rec = strategy_record_from_file(path, name="q-final", derived_from="final_test")
    assert not check_strategy_provenance(final_rec).paper_allowed
    # inner → 放行
    inner_rec = strategy_record_from_file(path, name="distance-q", derived_from="inner",
                                          note="inner 网格搜索冻结")
    good = check_strategy_provenance(inner_rec)
    assert good.paper_allowed and good.derived_from == "inner"
    assert_strategy_paper_eligible(inner_rec)
    # 哈希不符（文件登记后被改动）→ 拒绝
    tampered = StrategyRecord(name="distance-q", derived_from="inner",
                              source_path=str(path), content_hash="deadbeef")
    bad_hash = check_strategy_provenance(tampered)
    assert not bad_hash.paper_allowed and any("哈希不符" in r for r in bad_hash.reasons)
    # 策略生成晚于一次 outer 读取且无决策记录 → 拒绝（§19.4 跨 split）
    log = SplitAccessLog(tmp_path)
    log.log_access(Stage.OUTER, "看了 outer 的分题型分布", at="2026-01-01T00:00:00+00:00")
    late = strategy_record_from_file(path, name="distance-q", derived_from="inner",
                                     created_at="2026-02-01T00:00:00+00:00")
    late_v = check_strategy_provenance(late, access_log=log)
    assert not late_v.paper_allowed and any("outer 读取之后" in r for r in late_v.reasons)
    late.decision_note = "本次改动与 outer 观察无关，仅修 typo（已留决策记录）"
    late_v2 = check_strategy_provenance(late, access_log=log)
    assert late_v2.paper_allowed
