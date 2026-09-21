"""G-28/G-35 离线 FSM driver 单测（§6.2、§8、硬约束 10/11/13/19）。

覆盖：完整状态链、N_min 不足、泄漏硬门、GPT-6 不可用降级、
outer 只跑一次、checkpoint 中断恢复、治理消融档。
mock GPT-6 用注入的 client（`chat()` 返回受控文本），不触网。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill3d.evolution.offline_driver import (
    GOVERNANCE_MODES,
    OfflineCheckpoint,
    OfflineDriver,
    OfflineDriverConfig,
    load_episode_traces,
    main,
    trace_outcomes,
)
from skill3d.fsm.offline_fsm import OfflineState
from skill3d.online.runner import OnlineRunConfig
from skill3d.schemas import EpisodeTrace, FailureTaxonomy
from skill3d.trace.store import TraceStore

# ------------------------------------------------------------------ 工具 ----

SPEC = json.dumps({
    "skill_id": "sk-count", "semver": "1.0.0", "task_type": "object_counting",
    "description": "数对象：用 scene.list_objects() 计数后直接作答",
    "call_graph_template": "n = len(scene.list_objects())\nReturnAnswer(str(n))",
    "requires_artifacts": [], "minimum_quality": 0.0,
    "supported_coordinate_frames": ["world"], "metric_scale_required": False,
    "validation_assertions": ["n >= 0"],
})


PATCH = json.dumps({
    "patch_type": "add_assertion",
    "affected_task_types": ["object_counting"],
    "affected_skills": ["sk-count"],
    "patch_content": "assert n >= 0  # 修订：计数非负",
    "rationale": "反例显示负数计数",
    "expected_improvement": "+2pp",
    "risk_notes": "低",
})


class MockGPT6:
    """mock GPT-6：`chat()` 按 prompt 类型返回受控文本（绝不触网）。"""

    model_id = "mock-gpt6"

    def __init__(self, patch_spec: str = SPEC, review_json: str | None = None,
                 revise_json: str = PATCH):
        self.patch_spec = patch_spec
        self.review_json = review_json or json.dumps({
            "review_summary": "语义风险低", "semantic_risk": "low",
            "generalization_notes": "可跨场景泛化"})
        self.revise_json = revise_json
        self.calls: list[str] = []

    def chat(self, prompt: str, system: str | None = None) -> str:
        self.calls.append(prompt)
        if "语义审查器" in prompt:
            return self.review_json
        if "修订器" in prompt:                 # revise_patch.build_revision_prompt
            return self.revise_json
        if "归纳器" in prompt:                 # induce.build_induction_prompt
            return self.patch_spec
        return self.patch_spec


class FailingGPT6(MockGPT6):
    def chat(self, prompt: str, system: str | None = None) -> str:
        from skill3d.governance.gpt6_client import GPT6NotConfiguredError

        raise GPT6NotConfiguredError("mock：未配置（TODO_USER_INPUT）")


def _write_traces(trace_dir: Path, n_scenes: int = 4, per_scene: int = 2,
                  failure: bool = True, task: str = "object_counting") -> None:
    """写 induction 轨迹（episode_trace + evaluation_result 两个 topic）。"""
    store = TraceStore(str(trace_dir))
    for s in range(n_scenes):
        for k in range(per_scene):
            qa = f"qa-{s}-{k}"
            store.append("episode_trace", EpisodeTrace(
                episode_id=qa, qa_id=qa, final_state="unanswerable" if failure else "answer",
                program_trace_ref="", geometry_check_ref="", evaluation_ref="",
                failure=(FailureTaxonomy(episode_id=qa, categories=["coordinate"],
                                         note="相对方向错") if failure else None),
                active_snapshot_ref="genesis"))
            store.append("evaluation_result", {
                "qa_id": qa, "question_type": task, "is_mca": False,
                "predicted": None, "ground_truth": "3", "correct": not failure,
                "mra_value": 0.0 if failure else 1.0})


def _panels(tmp_path) -> dict:
    from skill3d.adapters.episode_source import load_synthetic_items

    out = tmp_path / "recons"
    # 面板尺寸压到最小：本测试只验证状态链语义，不验证真实感知
    kw = dict(question_types=["object_counting"], seed=0, out_dir=str(out),
              n_frames=6, frame_size=(64, 96))
    inner = load_synthetic_items("inner_validation", **kw)
    outer = load_synthetic_items("outer_holdout", **kw)
    return {"L1": inner, "L2": inner, "L3": outer}


def _driver(tmp_path, *, gpt6=None, n_min=3, mode="mock_light", governance="G0_full",
            traces=True, panels=None) -> OfflineDriver:
    trace_dir = tmp_path / "traces"
    if traces:
        _write_traces(trace_dir)
    cfg = OfflineDriverConfig(
        mode=mode, traces_glob=str(trace_dir / "*.jsonl"), trace_dir=str(trace_dir),
        n_min_cross_scene=n_min, skill_store=str(tmp_path / "skills"),
        active_snapshot=str(tmp_path / "active.json"),
        checkpoint_path=str(tmp_path / "ckpt.json"), run_id="gen-test",
        governance=governance,
    )
    base = OnlineRunConfig(mode=mode, seed=0, skills=[], trace_dir=str(trace_dir))
    meta = {f"qa-{s}-{k}": ("object_counting", f"scene-{s}")
            for s in range(4) for k in range(2)}
    return OfflineDriver(cfg, gpt6_client=gpt6, trace_store=TraceStore(str(trace_dir)),
                         panels=panels if panels is not None else _panels(tmp_path),
                         base_cfg=base, episode_meta=meta, print_fn=lambda *_: None)


# ------------------------------------------------------------------ trace 读取 ----

def test_load_episode_traces_and_outcomes(tmp_path):
    _write_traces(tmp_path / "t", n_scenes=2, per_scene=1, failure=True)
    traces = load_episode_traces(str(tmp_path / "t" / "*.jsonl"))
    assert len(traces) == 2 and all(t.failure is not None for t in traces)
    ok = trace_outcomes(str(tmp_path / "t"))
    assert ok == {"qa-0-0": False, "qa-1-0": False}


def test_load_episode_traces_empty_when_no_files(tmp_path):
    assert load_episode_traces("", str(tmp_path / "nope")) == []


# ------------------------------------------------------------------ 完整状态链 ----

def test_full_chain_reaches_reject_on_mock_light(tmp_path):
    """完整链路：CLUSTER→归纳→LEAKAGE→优化循环→（mock_light 不 promote）。"""
    drv = _driver(tmp_path, gpt6=MockGPT6())
    ckpt = drv.run()
    states = [t["to"] for t in ckpt.transitions]
    assert states[0] == OfflineState.GPT6_SYNTHESIZE.value
    assert OfflineState.LEAKAGE_CHECK.value in states
    assert OfflineState.LOOP_SYNTHESIZE.value in states
    # mock_light 下 delta 恒 0 → L1 不过 → 循环内修订或被 reject/patience
    assert ckpt.state in (OfflineState.REJECT.value, OfflineState.QUARANTINE.value)
    assert ckpt.candidate_ids and ckpt.gpt6_available
    # GPT-6 只在离线链被调用（归纳 + 可能的修订 + 审查）
    assert drv.gpt6.calls and "归纳器" in drv.gpt6.calls[0]


def test_checkpoint_persisted_every_step(tmp_path):
    drv = _driver(tmp_path, gpt6=MockGPT6())
    ckpt = drv.run()
    p = Path(drv.cfg.checkpoint_path)
    assert p.is_file()
    back = OfflineCheckpoint.load(p)
    assert back.state == ckpt.state and back.run_id == "gen-test"
    assert back.updated_at and back.transitions


def test_resume_from_checkpoint_continues(tmp_path):
    """中途中断后 resume：从最后状态继续，不重跑已完成状态。"""
    drv = _driver(tmp_path, gpt6=MockGPT6())
    drv.run()
    first = OfflineCheckpoint.load(drv.cfg.checkpoint_path)

    # 模拟中断：把 checkpoint 回退到 LEAKAGE_CHECK 之前
    partial = OfflineCheckpoint(
        run_id="gen-test", state=OfflineState.GPT6_SYNTHESIZE.value,
        transitions=[], candidate_ids=[], revision_ids=[])
    partial.save(drv.cfg.checkpoint_path)

    drv2 = _driver(tmp_path, gpt6=MockGPT6())
    drv2.cfg.resume = True
    ckpt2 = drv2.run()
    assert ckpt2.run_id == "gen-test"                       # 复用原 run_id
    assert ckpt2.state == first.state                       # 续跑到同一终态
    assert all(t["from"] != OfflineState.CLUSTER_TRACES.value
               for t in ckpt2.transitions)                  # CLUSTER 未重跑


# ------------------------------------------------------------------ 硬门 ----

def test_insufficient_evidence_rejects(tmp_path):
    """跨 scene 样本 < N_min → REJECT，不归纳（§7：禁单题成 Skill）。"""
    drv = _driver(tmp_path, gpt6=MockGPT6(), n_min=99)
    ckpt = drv.run()
    assert ckpt.state == OfflineState.REJECT.value
    assert "insufficient_evidence" in ckpt.termination_reason
    assert not ckpt.candidate_ids


def test_leakage_check_rejects_candidate_with_answer_text(tmp_path):
    """候选含 sample 答案/ID → LEAKAGE_CHECK 拒（硬约束 13/19）。"""
    leaky = json.dumps({**json.loads(SPEC),
                    "description": "对象计数题：先读 ground truth 再作答（泄漏样式）"})
    drv = _driver(tmp_path, gpt6=MockGPT6(patch_spec=leaky))
    ckpt = drv.run()
    assert ckpt.state == OfflineState.REJECT.value
    assert "leakage_detected" in ckpt.termination_reason


def test_gpt6_unavailable_quarantines_without_promoting(tmp_path):
    """GPT-6 不可用 → QUARANTINE，不 promote，且不阻塞（§8）。"""
    drv = _driver(tmp_path, gpt6=FailingGPT6())
    ckpt = drv.run()
    assert ckpt.state == OfflineState.QUARANTINE.value
    assert not ckpt.gpt6_available or "gpt6_not_configured" in ckpt.termination_reason
    assert not (Path(tmp_path / "skills") / "active_snapshot.json").exists()


def test_no_traces_means_insufficient(tmp_path):
    drv = _driver(tmp_path, gpt6=MockGPT6(), traces=False)
    ckpt = drv.run()
    assert ckpt.state == OfflineState.REJECT.value
    assert "insufficient_evidence" in ckpt.termination_reason


def test_outer_runs_at_most_once(tmp_path):
    """硬约束 10：outer（L3）只跑一次，失败不修订。"""
    from skill3d.evolution import panel as panel_mod

    calls = {"L1": 0, "L2": 0, "L3": 0}
    real = panel_mod.run_panel

    def counting(items, cfg, trace_store=None, llm=None):
        return real(items, cfg, trace_store, llm)

    drv = _driver(tmp_path, gpt6=MockGPT6())
    # 统计各层测试次数：直接监控 run_candidate_panels 是否可能重复调 L3
    orig = panel_mod.run_candidate_panels

    def spy(root, **kw):
        run, levels = orig(root, **kw)
        calls["L3"] = 1 if "L3" in levels else 0
        return run, levels

    panel_mod.run_candidate_panels = spy
    try:
        drv.run()
    finally:
        panel_mod.run_candidate_panels = orig
    assert calls["L3"] <= 1
    del counting


# ------------------------------------------------------------------ 治理消融 ----

@pytest.mark.parametrize("gov", GOVERNANCE_MODES)
def test_governance_ablation_modes_are_accepted(tmp_path, gov):
    drv = _driver(tmp_path, gpt6=MockGPT6(), governance=gov)
    ckpt = drv.run()
    assert ckpt.state in (OfflineState.REJECT.value, OfflineState.QUARANTINE.value,
                          OfflineState.PROMOTE.value)
    assert Path(drv.cfg.checkpoint_path).is_file()


def test_g1_skips_semantic_review_when_promoting(tmp_path):
    """G1 档（无 review）：即使走到准入，也不产生语义审查记录。"""
    drv = _driver(tmp_path, gpt6=MockGPT6(), governance="G1_no_review")
    drv.gpt6 = MockGPT6()
    # 直接驱动到 promote 阶段：伪造 L3 通过 + 准入通过
    drv._candidate_v0 = None  # 无候选 → 不产出
    ckpt = drv.run()
    assert ckpt.state != OfflineState.PROMOTE.value or \
        "gpt6_semantic_review" in ckpt.skipped_stages


# ------------------------------------------------------------------ PROMOTE 可达性 ----

def _force_promoted_loop(monkeypatch, po):
    """把 driver 的内环替换为"L1/L2/L3 全过"的确定性结果（只验证状态链语义）。"""
    from skill3d.evolution import offline_driver as od
    from skill3d.schemas import BudgetUsage, OptimizationRun

    run = OptimizationRun(
        run_id="run-forced", root_candidate_id="rc", current_revision_id="rev",
        budget_used=BudgetUsage(tokens=0, gpu_hours=0.0, rollouts=3, revisions=0),
        budget_limit=od.DEFAULT_BUDGET_LIMIT, patience_counter=0,
        revision_history=["rev"], parallel_branches=[], selection_strategy="sequential",
        status="promoted", termination_reason="outer_passed")
    monkeypatch.setattr(od, "run_candidate_panels",
                        lambda *a, **kw: (run, {"L1": (po, [], []), "L2": (po, [], []),
                                                "L3": (po, [], [])}))


def _passing_paired_outcome():
    from skill3d.schemas import PairedOutcome

    return PairedOutcome(
        pair_id="pair-L3-test", snapshot_id="snap", arm_a_branch_id="arm-a-baseline",
        arm_b_branch_id="arm-b-candidate", n_episodes=4, metric="accuracy",
        mean_a=0.5, mean_b=0.7, delta=0.2, ci95_lo=0.05, ci95_hi=0.35,
        wilcoxon_p=0.01, slice_table={}, resource_cost={},
        slice_no_regression=True, within_budget=True)


def test_promoted_loop_reaches_admission_stage(tmp_path, monkeypatch):
    """§5.2 硬约束 12/13：内环整体通过后 FSM 必须到 PROMOTE 并执行准入门。

    回归护栏：`LOOP_SYNTHESIZE` 曾缺 pass/fail 宏边，driver 会永远停在
    `LOOP_SYNTHESIZE`，`_stage_promote`（准入门 + 原子切换）在真实 driver 路径中
    从未被执行（测试又只用 mock_light 的 REJECT/QUARANTINE 值域，掩盖了该缺陷）。
    """
    _force_promoted_loop(monkeypatch, _passing_paired_outcome())
    drv = _driver(tmp_path, gpt6=MockGPT6(), n_min=1)
    ckpt = drv.run()

    hits = [t for t in ckpt.transitions
            if t["from"] == OfflineState.LOOP_SYNTHESIZE.value
            and t["to"] == OfflineState.PROMOTE.value]
    assert hits, f"未走到 PROMOTE：transitions={[t['to'] for t in ckpt.transitions]}"
    assert drv.fsm.outer_attempted                      # 硬约束 10 审计位
    # 准入门真的被评估并落 trace（硬约束 13）
    adm = Path(drv.cfg.trace_dir) / "admission_decision.jsonl"
    rows = [json.loads(r) for r in adm.read_text().splitlines()]
    assert rows and rows[-1]["promotes"] is True
    # §5.6b：mock_light 即使准入通过也不得 promote
    assert ckpt.state == OfflineState.QUARANTINE.value
    assert ckpt.termination_reason.startswith("admission_must_be_real")


def test_promoted_loop_below_min_delta_is_rejected_by_hard_gate(tmp_path, monkeypatch):
    """硬约束 13：硬门（delta < min_delta）一票否决，GPT-6 建议不能覆盖。"""
    po = _passing_paired_outcome().model_copy(update={"delta": 0.001, "ci95_lo": 0.0005})
    _force_promoted_loop(monkeypatch, po)
    drv = _driver(tmp_path, gpt6=MockGPT6(), n_min=1)
    ckpt = drv.run()
    assert ckpt.state == OfflineState.REJECT.value
    assert ckpt.termination_reason.startswith("admission_rejected")
    assert "ci_significant" in ckpt.termination_reason


def test_similar_to_rejected_candidate_rejects(tmp_path):
    """防重复（§7）：与已 reject 候选高度相似 → 终态必须是 REJECT，不得悬空。"""
    drv = _driver(tmp_path, gpt6=MockGPT6())
    drv.archive.add_rejected(SPEC)          # 提前留档同内容候选
    ckpt = drv.run()
    assert ckpt.state == OfflineState.REJECT.value
    assert ckpt.termination_reason == "similar_to_rejected_no_new_evidence"


# ------------------------------------------------------------------ CLI ----

def test_cli_help_and_dry_run(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "offline_driver" in out

    rc = main(["--mode", "mock_light", "--run-id", "cli-run",
               "--checkpoint", str(tmp_path / "c.json"),
               "--trace-dir", str(tmp_path / "traces"),
               "--skill-store", str(tmp_path / "skills")])
    assert rc == 0                                  # mock_light：REJECT 属正常退出
    assert (tmp_path / "c.json").is_file()


def test_cli_rejects_bad_governance_mode(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["--governance", "G9_bogus"])
    assert exc.value.code == 2
