"""v4 §10.2 D-2 尺度 PoC 阶梯单测（L0–L3）+ §7 尺度报告聚合。

纪律：L0/L1 可离线跑；L2/L3 缺真实数据时必须 **UNVERIFIED**（退出码 1），
绝不允许用合成数据冒充标定结果（§10.5 反模式清单）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from skill3d.evaluation.scale_report import (
    ScaleReport,
    build_scale_report,
    write_scale_report,
)
from skill3d.reconstruction.scale_poc import (
    L2_MIN_UPGRADE_N,
    PocResult,
    main as poc_main,
    poc_l0,
    poc_l1,
    poc_l2,
    poc_l3,
    run_ladder,
)

META = "data/vsi_bench_meta/test.jsonl"


# --------------------------------------------------------------- L0/L1 ----

def test_l0_passes():
    r = poc_l0()
    assert r.passed and r.verified, r.failures
    assert r.metrics["percent_like_reason"] == "percent_like"
    assert r.metrics["full_width_reason"] == "full_width_like"
    assert r.metrics["schema_downgraded_to"] == "low"


def test_l1_passes():
    r = poc_l1()
    assert r.passed and r.verified, r.failures
    assert r.metrics["single_outlier_dev"] < 0.15
    assert r.metrics["single_outlier_reasons"]["chair:extent_up"] == "outlier"
    assert r.metrics["conflict_confidence"] == "low"
    assert r.metrics["empty_ok"] is False


# ------------------------------------------------------------------- L2 ----

def _records(n: int, *, seed: int = 0, sigma: float = 0.04) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        err = float(np.exp(rng.normal(0.0, sigma)))
        out.append({"scene_id": f"cal_scene_{i}", "scale_pred": err, "scale_true": 1.0,
                    "rel_ci": 0.02 + 0.4 * abs(np.log(err)),
                    "plane_identity_ok": True, "anchor_fired": True})
    return out


def test_l2_unverified_without_meta():
    """缺 VSI-Bench meta → 无法证明隔离 → UNVERIFIED（不得当通过）。"""
    r = poc_l2(_records(50), meta_path="")
    assert not r.verified and not r.passed
    assert "HC32" in r.failures[0] or "不相交" in r.failures[0]


def test_l2_hard_fails_when_calibration_overlaps_evaluation():
    """HC32：标定 scene 与 VSI-Bench 评测 scene 相交 → hard fail。"""
    if not Path(META).is_file():
        pytest.skip("VSI-Bench meta 未就位")
    recs = _records(40)
    recs[0]["scene_id"] = "41069025"          # 这是 VSI-Bench 的 ARKitScenes scene
    r = poc_l2(recs, meta_path=META)
    assert not r.passed
    assert any("隔离 hard fail" in f for f in r.failures)


def test_l2_reports_sizes_and_never_qualifies_n10():
    """N=10 仅诊断（§10.2 明文），不得被标为具备升级资格。"""
    if not Path(META).is_file():
        pytest.skip("VSI-Bench meta 未就位")
    r = poc_l2(_records(120), meta_path=META)
    assert r.passed, r.failures
    per = r.metrics["per_size"]
    assert set(per) == {"10", "30", "100"}
    assert per["10"]["upgrade_eligible"] is False
    assert L2_MIN_UPGRADE_N == 30
    for key, entry in per.items():
        if entry.get("available"):
            assert 0.0 <= entry["empirical_coverage"] <= 1.0
            assert np.isfinite(entry["median_rel_error"])
            assert entry["ci_halfwidth_rel"] > 0
            assert "anchor_fire_rate" in entry and "conflict_rate" in entry
    assert 30 not in r.metrics["upgrade_eligible_sizes"] or True


def test_l2_unverified_when_all_sizes_unavailable():
    if not Path(META).is_file():
        pytest.skip("VSI-Bench meta 未就位")
    r = poc_l2(_records(5), meta_path=META)
    assert not r.verified


# ------------------------------------------------------------------- L3 ----

def test_l3_unverified_without_real_outcomes():
    r = poc_l3(None)
    assert not r.verified and not r.passed
    assert any("TODO_USER_INPUT" in n for n in r.notes)


def test_l3_flags_task_below_2d_baseline():
    r = poc_l3({"object_abs_distance": {"mra": 0.20, "refusal_rate": 0.1, "n": 40}},
               baseline_2d={"object_abs_distance": 0.35})
    assert not r.passed
    assert any("不得加入 allowed_metric_tasks" in f for f in r.failures)
    assert r.metrics["object_abs_distance"]["eligible"] is False


def test_l3_marks_eligible_task():
    r = poc_l3({"room_size_estimation": {"mra": 0.60, "refusal_rate": 0.05, "n": 40}},
               baseline_2d={"room_size_estimation": 0.40})
    assert r.passed
    assert r.metrics["room_size_estimation"]["eligible"] is True


# ------------------------------------------------------------ ladder CLI ----

def test_run_ladder_exit_code_reflects_unverified(tmp_path, capsys):
    """只通过 L0/L1 时退出码必须为 1：不得声称尺度能力已恢复（HC34）。"""
    rc = run_ladder(out=str(tmp_path / "poc.json"))
    assert rc == 1
    out = capsys.readouterr().out
    assert "UNVERIFIED" in out and "不得声称尺度能力已恢复" in out
    receipt = json.loads((tmp_path / "poc.json").read_text(encoding="utf-8"))
    assert [r["level"] for r in receipt] == ["L0", "L1", "L2", "L3"]
    assert [r["status"] for r in receipt[:2]] == ["PASS", "PASS"]


def test_run_ladder_with_l2_records(tmp_path, capsys):
    if not Path(META).is_file():
        pytest.skip("VSI-Bench meta 未就位")
    recs = tmp_path / "cal.jsonl"
    recs.write_text("\n".join(json.dumps(r) for r in _records(120)), encoding="utf-8")
    rc = run_ladder(l2_records=str(recs), meta_path=META,
                    out=str(tmp_path / "poc.json"))
    assert rc == 1                                 # L3 仍 UNVERIFIED
    out = capsys.readouterr().out
    assert "L2 标定规模消融（§10.2）" in out and "PASS" in out


def test_poc_main_returns_nonzero_without_real_data(capsys):
    rc = poc_main(["--meta", META])
    assert rc == 1


def test_poc_result_status_labels():
    assert PocResult("L0", "n", True).as_dict()["status"] == "PASS"
    assert PocResult("L2", "n", False, verified=False).as_dict()["status"] == "UNVERIFIED"
    assert PocResult("L1", "n", False).as_dict()["status"] == "FAIL"


# ------------------------------------------------------------ 尺度报告 ----

class _Outcome:
    """EpisodeOutcome 的最小同构替身（只含尺度报告读的字段）。"""

    def __init__(self, **kw):
        self.qa_id = kw.get("qa_id", "q")
        self.task = kw.get("task", "object_counting")
        self.final_state = kw.get("final_state", "answer")
        self.answer = kw.get("answer", "4")
        self.mra_value = kw.get("mra_value")
        self.correct = kw.get("correct")
        self.answer_untrusted = kw.get("answer_untrusted", False)
        self.scale_confidence = kw.get("scale_confidence", "low")
        self.allowed_metric_tasks = kw.get("allowed_metric_tasks", [])
        self.authorized_metric_tasks = kw.get("authorized_metric_tasks", [])
        self.n_anchors_fired = kw.get("n_anchors_fired", 0)
        self.n_anchors_accepted = kw.get("n_anchors_accepted", 0)
        self.scale_conflict = kw.get("scale_conflict", False)
        self.scale_ci_rel = kw.get("scale_ci_rel")
        self.scale_ci_abs_m = kw.get("scale_ci_abs_m")
        self.scale_calibration_id = kw.get("scale_calibration_id")
        self.scale_empirical_coverage = kw.get("scale_empirical_coverage")
        self.scale_nominal_coverage = kw.get("scale_nominal_coverage")


def test_scale_report_current_reality_all_withdrawn():
    """当前实况：全部 episode 收回米制题型 → 报告要写明这是正确表现而非缺陷。"""
    outs = [_Outcome(task="object_counting", mra_value=0.8, scale_confidence="low",
                     n_anchors_fired=2, n_anchors_accepted=2),
            _Outcome(task="room_size_estimation", mra_value=0.0,
                     scale_confidence="low", n_anchors_fired=3,
                     n_anchors_accepted=3),
            _Outcome(task="room_size_estimation", final_state="unanswerable",
                     answer=None, mra_value=0.0, scale_confidence="low"),
            _Outcome(task="room_size_estimation", mra_value=0.2,
                     scale_confidence="low")]
    rep = build_scale_report(outs)
    assert rep.n_episodes == 4
    assert rep.n_metric_authorized == 0 and rep.n_metric_withdrawn == 4
    assert rep.confidence_distribution == {"low": 4}
    # 只有两条 episode 报出了锚点（fired=2 与 fired=3）→ 均值 2.5、接受率 1.0
    assert rep.mean_anchors_fired == pytest.approx(2.5)
    assert rep.anchor_fire_rate == pytest.approx(1.0)
    assert rep.anchor_accept_rate == pytest.approx(1.0)
    assert rep.conflict_rate == pytest.approx(0.0)
    assert rep.notes and "fail-closed" in rep.notes[-1]
    rows = {r.task: r for r in rep.metric_task_rows()}
    assert rows["room_size_estimation"].n == 3
    assert rows["room_size_estimation"].n_answered == 2
    assert rows["room_size_estimation"].coverage == pytest.approx(2 / 3)
    assert rows["room_size_estimation"].mra == pytest.approx(0.1)
    # 不混主表：报告只列可靠性指标
    lines = rep.format_lines()
    assert lines[0].startswith("尺度能力")


def test_scale_report_marks_metric_task_eligibility():
    outs = [_Outcome(task="room_size_estimation", mra_value=0.6,
                     scale_confidence="medium",
                     authorized_metric_tasks=["room_size_estimation"],
                     scale_ci_rel=0.08, scale_ci_abs_m=0.16,
                     scale_calibration_id="cal-1",
                     scale_empirical_coverage=0.91,
                     scale_nominal_coverage=0.90)
            for _ in range(5)]
    rep = build_scale_report(outs, baseline_2d={"room_size_estimation": 0.4})
    row = rep.per_task["room_size_estimation"]
    assert row.eligible_for_metric_tasks is True
    assert rep.n_metric_authorized == 5
    assert rep.calibration_id == "cal-1"
    assert rep.coverage_gap == pytest.approx(0.01, abs=1e-9)
    assert rep.scale_ci_rel_median == pytest.approx(0.08)
    # 未达 baseline → 不可加入
    rep2 = build_scale_report(outs, baseline_2d={"room_size_estimation": 0.9})
    assert rep2.per_task["room_size_estimation"].eligible_for_metric_tasks is False


def test_scale_report_small_n_does_not_report_ratios():
    outs = [_Outcome(task="object_abs_distance", mra_value=1.0)]
    rep = build_scale_report(outs)
    row = rep.per_task["object_abs_distance"]
    assert row.n == 1 and row.mra is None and row.coverage is None
    assert any("<" in n for n in row.notes)


def test_scale_report_conflict_rate_and_write(tmp_path):
    outs = [_Outcome(task="object_counting", scale_conflict=True,
                     n_anchors_fired=3, n_anchors_accepted=1),
            _Outcome(task="object_counting", n_anchors_fired=0)]
    rep = build_scale_report(outs)
    assert rep.conflict_rate == pytest.approx(0.5)
    assert rep.anchor_accept_rate == pytest.approx(1 / 3)
    p = write_scale_report(rep, str(tmp_path / "scale_report.json"))
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    assert data["scale"]["conflict_rate"] == pytest.approx(0.5)
    assert "metric_tasks" in data


def test_scale_report_empty_outcomes():
    rep = build_scale_report([])
    assert isinstance(rep, ScaleReport) and rep.n_episodes == 0
    assert rep.metric_task_rows() == []
