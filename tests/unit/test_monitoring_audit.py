"""G-37/G-39/G-40 单测：离线监控 → 回滚/再审，与审计回溯。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from skill3d.evolution.monitoring import (
    RunHealth,
    answer_distribution_kl,
    collect_health,
    monitor,
    rollback_if_needed,
    write_health_report,
)
from skill3d.infra.audit import (
    CHAIN_TOPICS,
    AuditTrail,
    build_audit_trail,
    list_candidates,
    write_audit_trail,
)
from skill3d.trace.store import TraceStore


# ------------------------------------------------------------------ 伪产物 ----

@dataclass
class _Trace:
    wallclock_s: float = 0.3


@dataclass
class _Verify:
    passed: bool


@dataclass
class _Outcome:
    final_state: str = "answer"
    answer_flags: list = field(default_factory=list)
    scene_route: str = "full_3d"
    program_trace: object = None
    verify: object = None


def _out(state="answer", *, verify=True, flags=(), route="full_3d"):
    return _Outcome(final_state=state, answer_flags=list(flags), scene_route=route,
                    program_trace=_Trace(),
                    verify=(_Verify(verify) if verify is not None else None))


# ------------------------------------------------------------------ G-37 指标 ----

def test_collect_health_rates_and_histogram():
    outs = [_out(), _out("answer_best_effort", flags=["no_tool_fallback"]),
            # 旧 trace 的 v5 门控 flag（v6 已不再产生，§20；读旧 trace 必须仍识别）
            _out("unavailable", verify=None), _out("unanswerable", flags=["g8_size_reject"])]
    h = collect_health(outs, label="gen-1")
    assert h.n_episodes == 4
    assert h.success_rate == pytest.approx(0.25)
    assert h.answer_rate == pytest.approx(0.5)
    assert h.geometry_pass_rate == pytest.approx(1.0)      # 3 个有 verify，全过
    assert h.fallback_rate == pytest.approx(0.25)
    assert h.unavailable_rate == pytest.approx(0.25)
    assert h.gate_reject_rate == pytest.approx(0.25)
    assert h.mean_wallclock_s == pytest.approx(0.3)
    assert h.answer_histogram["answer"] == 1
    assert h.to_dict()["label"] == "gen-1"


def test_collect_health_empty_is_none():
    h = collect_health([])
    assert h.n_episodes == 0 and h.success_rate is None and h.geometry_pass_rate is None


def test_answer_distribution_kl_and_empty_cases():
    assert answer_distribution_kl({"a": 1, "b": 1}, {"a": 1, "b": 1}) == \
        pytest.approx(0.0, abs=1e-9)
    assert answer_distribution_kl({"a": 3}, {"b": 3}) > 0.5      # 完全换分布 → 大漂移
    assert answer_distribution_kl({}, {"a": 1}) is None
    assert answer_distribution_kl({"a": 1}, {}) is None


def test_monitor_no_alerts_when_stable():
    prev = collect_health([_out() for _ in range(10)])
    cur = collect_health([_out() for _ in range(10)])
    rep = monitor(cur, prev)
    assert rep.alerts == [] and rep.action == "none"
    assert "未见异常" in rep.reason
    assert rep.kl_drift == pytest.approx(0.0, abs=1e-9)


def test_monitor_single_alert_triggers_review():
    """单项异常 → review（不做回滚）；此处用"分布不变、几何验证率下降"构造单告警。"""
    prev = collect_health([_out() for _ in range(10)])
    cur = collect_health([_out() for _ in range(8)] + [_out(verify=False)] * 2)
    rep = monitor(cur, prev)
    assert rep.alerts == ["几何验证率下降 +0.200 > 0.1"]
    assert rep.action == "review"
    assert rep.kl_drift == pytest.approx(0.0, abs=1e-9)


def test_monitor_multiple_alerts_trigger_rollback():
    prev = collect_health([_out() for _ in range(10)])
    cur = collect_health(
        [_out("unanswerable", verify=False, route="fallback_2d_only",
              flags=["no_tool_fallback"])] * 6 + [_out()] * 4)
    rep = monitor(cur, prev)
    assert len(rep.alerts) >= 2 and rep.action == "rollback"
    assert "成功率下降" in rep.reason


def test_monitor_disabled_by_governance_ablation():
    """G2 档（无监控）：不产出任何告警/动作（§8.4）。"""
    prev = collect_health([_out() for _ in range(10)])
    cur = collect_health([_out("unanswerable")] * 10)
    rep = monitor(cur, prev, monitoring_enabled=False)
    assert rep.action == "none" and rep.alerts == []
    assert "G2" in rep.reason


def test_monitor_without_previous_only_records():
    rep = monitor(collect_health([_out()]))
    assert rep.action == "none" and "无上一代" in rep.reason


def test_write_health_report(tmp_path):
    rep = monitor(collect_health([_out()]), collect_health([_out()]))
    out = write_health_report(rep, tmp_path / "sub" / "health.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["action"] == "none" and "current" in data and "alerts" in data


def test_rollback_only_on_rollback_action(tmp_path):
    store = tmp_path / "skills"
    from skill3d.skills.promote_atomic import promote, read_active_snapshot
    from skill3d.schemas import CandidateRevision

    store.mkdir(parents=True, exist_ok=True)
    cand = CandidateRevision(
        revision_id="rev-1", root_candidate_id="cand-1", parent_version=None,
        candidate_type="skill", spec_content=json.dumps({
            # v6 §5.8 SkillSpec（题型 + 证据签名；v5 的 requires_artifacts /
            # minimum_quality / metric_scale_required 已废止，§20）
            "skill_id": "sk-1", "version": "1.0.0",
            "applicable_question_types": ["object_counting"],
            "required_evidence_signature": {"object_detection": "available"},
            "skill_family": "counting", "source": "real",
            "description": "d", "call_graph_template": "x"}),
        status="promoted", induction_trace_refs=[], evidence_lineage_ref="",
        created_by="human", created_at="1970-01-01T00:00:00+00:00")
    promote(str(store), cand)
    assert read_active_snapshot(store)["entries"]

    # action=none → 不回滚
    assert rollback_if_needed(monitor(collect_health([_out()])), store) is None

    # action=rollback → 调用回滚原语，active 指针回退
    prev = collect_health([_out() for _ in range(10)])
    cur = collect_health([_out("unanswerable", verify=False,
                               route="fallback_2d_only",
                               flags=["no_tool_fallback"])] * 8 + [_out()] * 2)
    rep = monitor(cur, prev)
    assert rep.action == "rollback"
    store_obj = TraceStore(str(tmp_path / "traces"))
    snap = rollback_if_needed(rep, store, trace_store=store_obj)
    assert snap is not None
    assert (tmp_path / "traces" / "rollback.jsonl").is_file()


# ------------------------------------------------------------------ G-40 审计 ----

def _write_chain(trace_dir, revision_id="rev-1", candidate_id="cand-1", *,
                 with_decision=True) -> None:
    store = TraceStore(str(trace_dir))
    store.append("candidate_revision", {
        "revision_id": revision_id, "root_candidate_id": candidate_id,
        "induction_trace_refs": ["qa-1", "qa-2"], "spec_content": "{}"})
    store.append("leakage_check", {"candidate_id": candidate_id, "passed": True})
    store.append("counterexample_bundle", {
        "bundle_id": "b1", "source_revision_id": revision_id,
        "failed_episode_refs": ["qa-9"]})
    store.append("paired_outcome", {
        "pair_id": "p1", "arm_b_branch_id": "arm-b",
        "mean_a": 0.4, "mean_b": 0.5, "delta": 0.1,
        "root_candidate_id": candidate_id, "revision_id": revision_id})
    store.append("optimization_run", {
        "run_id": "r1", "root_candidate_id": candidate_id,
        "current_revision_id": revision_id, "revision_history": [revision_id],
        "status": "promoted"})
    store.append("admission_decision", {
        "decision_id": "d1", "candidate_id": candidate_id, "promotes": True})
    if with_decision:
        store.append("skill_governance_decision", {
            "decision_id": "g1", "candidate_id": revision_id,
            "review_summary": "ok"})
    store.append("promotion", {"revision_id": revision_id,
                               "snapshot_before": "genesis", "snapshot_after": "s1"})


def test_build_audit_trail_complete_chain(tmp_path):
    _write_chain(tmp_path)
    trail = build_audit_trail(tmp_path, revision_id="rev-1")
    assert trail.complete and trail.missing == []
    assert trail.candidate_id == "cand-1"
    assert trail.n_records >= len(CHAIN_TOPICS)
    for topic in CHAIN_TOPICS:
        assert trail.steps.get(topic), topic
    assert trail.summary().startswith("revision=rev-1")


def test_build_audit_trail_reports_missing_steps(tmp_path):
    _write_chain(tmp_path, with_decision=False)
    trail = build_audit_trail(tmp_path, candidate_id="cand-1")
    assert not trail.complete
    assert trail.missing == ["skill_governance_decision"]


def test_build_audit_trail_requires_an_id(tmp_path):
    with pytest.raises(ValueError, match="需要 revision_id"):
        build_audit_trail(tmp_path)


def test_build_audit_trail_unknown_id_is_empty_but_explicit(tmp_path):
    _write_chain(tmp_path)
    trail = build_audit_trail(tmp_path, revision_id="rev-nope")
    assert trail.n_records == 0 and not trail.complete


def test_list_candidates_and_write_trail(tmp_path):
    _write_chain(tmp_path)
    _write_chain(tmp_path, revision_id="rev-2", candidate_id="cand-2")
    cands = list_candidates(tmp_path)
    assert {c["candidate_id"] for c in cands} == {"cand-1", "cand-2"}
    assert all(c["n_chain_steps"] > 3 for c in cands)

    trail = build_audit_trail(tmp_path, revision_id="rev-2")
    out = write_audit_trail(trail, tmp_path / "audit_rev2.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["revision_id"] == "rev-2" and data["complete"] is True


def test_audit_tolerates_corrupted_lines(tmp_path):
    _write_chain(tmp_path)
    with open(tmp_path / "paired_outcome.jsonl", "a", encoding="utf-8") as f:
        f.write("{not json}\n")
    trail = build_audit_trail(tmp_path, revision_id="rev-1")
    assert trail.steps.get("paired_outcome")                 # 有效行仍被读到
    assert trail.complete


def test_audit_trail_dataclass_contract():
    t = AuditTrail(revision_id="r", candidate_id="c", steps={"promotion": [{}]},
                   missing=["paired_outcome"], n_records=1)
    assert not t.complete and t.to_dict()["missing"] == ["paired_outcome"]
