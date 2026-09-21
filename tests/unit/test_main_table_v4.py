"""§7/§16.3 多 seed 主表入口单测（`evaluation/main_table.py`）。

护栏（对应 HC24/HC34 与 §7）：
- **≥3 seed** 才可进主表；不足 → `eligible=False` + 退出码 1；
- **非 real 模式**（mock_light/合成）不得进主表；来源不明（mode 缺失）同样按不可信处理；
- **split 必须一致**，混 split 不得聚合成一行；
- 数字口径：8 任务宏平均 ×100（**只乘一次**），缺任务不臆造 0。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill3d.evaluation.main_table import (
    MainTableError,
    RunRecord,
    build_group,
    build_main_table,
    format_main_table,
    load_run,
    main as main_table_main,
)
from skill3d.evaluation.multi_seed_aggregator import MIN_SEEDS_FOR_MAIN_TABLE
from skill3d.schemas import EvaluationRun

TASKS_MCA = ("object_rel_distance", "object_rel_direction", "route_planning",
             "obj_appearance_order")
TASKS_NA = ("object_counting", "object_abs_distance", "object_size_estimation",
            "room_size_estimation")


def _run(seed: int, *, acc: float, mra: float, mode: str = "real",
         split: str = "inner_validation", per_task: dict | None = None,
         commit: str = "abc123") -> RunRecord:
    pt = per_task if per_task is not None else {
        **{t: {"accuracy": acc} for t in TASKS_MCA},
        **{t: {"mra": mra} for t in TASKS_NA},
    }
    run = EvaluationRun(
        run_id=f"run-s{seed}", split=split, n_episodes=50,
        accuracy=acc, mra=mra, per_task=pt,
        active_snapshot_ref="genesis", code_commit=commit, timestamp="deterministic")
    return RunRecord(run=run, mode=mode, baseline="C1_tools_program", seed=seed,
                     split=split, source="synthetic")


# ------------------------------------------------------------ 资格门禁 ----

def test_three_real_seeds_are_main_table_eligible():
    recs = [_run(s, acc=0.5 + 0.02 * s, mra=0.6 + 0.01 * s) for s in range(3)]
    g = build_group("Ours", recs)
    assert g.eligible_for_main_table, g.reasons
    assert g.aggregates["accuracy"].n_seeds == 3
    assert g.aggregates["accuracy"].std > 0
    assert g.macro["avg_x100"] is not None


def test_two_seeds_are_not_eligible():
    """§7：主表必须 ≥3 seed。"""
    g = build_group("Ours", [_run(s, acc=0.5, mra=0.6) for s in range(2)])
    assert not g.eligible_for_main_table
    assert any("不满足论文主表要求" in r for r in g.reasons)
    assert MIN_SEEDS_FOR_MAIN_TABLE == 3


def test_mock_runs_never_enter_main_table():
    """HC24：mock/合成不得进主表。"""
    recs = [_run(s, acc=0.5, mra=0.9, mode="mock_light") for s in range(3)]
    g = build_group("mock", recs)
    assert not g.eligible_for_main_table
    assert any("不得进主表" in r for r in g.reasons)
    # 显式放行时只降级为警告（仍标 non-eligible）
    g2 = build_group("mock", recs, allow_mock=True)
    assert g2.warnings and not g2.eligible_for_main_table


def test_run_without_mode_is_treated_as_untrusted():
    """来源不明（老产物没有 mode）→ 按不可信处理（fail-closed）。"""
    g = build_group("x", [_run(s, acc=0.5, mra=0.5, mode="") for s in range(3)])
    assert not g.eligible_for_main_table
    assert any("未记录" in r for r in g.reasons)


def test_mixed_split_is_not_eligible():
    recs = [_run(0, acc=0.5, mra=0.5),
            _run(1, acc=0.5, mra=0.5, split="outer_holdout"),
            _run(2, acc=0.5, mra=0.5)]
    g = build_group("x", recs)
    assert not g.eligible_for_main_table
    assert any("split 不一致" in r for r in g.reasons)


def test_empty_group_has_reason():
    g = build_group("x", [])
    assert not g.eligible_for_main_table and g.reasons


# ------------------------------------------------------------- 数字口径 ----

def test_macro_average_multiplied_once():
    """8 任务宏平均 ×100 **只乘一次**（历史 bug：二次缩放导致 3750 这种值）。"""
    recs = [_run(s, acc=1.0, mra=0.0) for s in range(3)]
    g = build_group("x", recs)
    # 4 个 MCA=1.0、4 个 NA=0.0 → 宏平均 0.5 → ×100 = 50
    assert g.macro["avg_x100"] == pytest.approx(50.0)
    assert g.macro["per_task_x100"]["route_planning"] == pytest.approx(100.0)
    assert g.macro["per_task_x100"]["object_counting"] == pytest.approx(0.0)
    assert g.macro["n_tasks"] == 8 and g.macro["missing_tasks"] == []


def test_missing_tasks_are_reported_not_zeroed():
    partial = {"object_counting": {"mra": 0.5}}
    recs = [_run(s, acc=0.4, mra=0.5, per_task=partial) for s in range(3)]
    g = build_group("x", recs)
    assert g.macro["n_tasks"] == 1
    assert set(g.macro["missing_tasks"]) == set(TASKS_MCA) | set(TASKS_NA) - {"object_counting"}
    # 缺任务不参与平均 → 宏平均等于唯一任务的分数
    assert g.macro["avg_x100"] == pytest.approx(50.0)


# ---------------------------------------------------------------- 读取 ----

def test_load_run_from_trace_dir_and_json(tmp_path):
    """trace 目录（JSONL，按 run_id 关联 online_run）与单 JSON 两种输入都要支持。"""
    d = tmp_path / "s0"
    d.mkdir()
    (d / "evaluation_run.jsonl").write_text(
        _run(0, acc=0.5, mra=0.5).run.model_dump_json() + "\n", encoding="utf-8")
    (d / "online_run.jsonl").write_text(json.dumps(
        {"run_id": "run-s0", "mode": "real", "baseline": "C1_tools_program",
         "seed": 0, "split": "inner_validation", "source": "vsi_bench"}) + "\n",
        encoding="utf-8")
    rec = load_run(d)
    assert rec.mode == "real" and rec.seed == 0 and not rec.is_mock
    assert rec.run.accuracy == pytest.approx(0.5)

    single = tmp_path / "run.json"
    single.write_text(json.dumps({
        "evaluation_run": json.loads(_run(1, acc=0.6, mra=0.5).run.model_dump_json()),
        "mode": "real", "seed": 1, "split": "inner_validation"}), encoding="utf-8")
    rec2 = load_run(single)
    assert rec2.seed == 1 and rec2.mode == "real"


def test_load_run_missing_path_raises(tmp_path):
    with pytest.raises(MainTableError):
        load_run(tmp_path / "nope")
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(MainTableError):
        load_run(d)


# ---------------------------------------------------------------- CLI ----

def test_cli_multi_group_and_gate(tmp_path, capsys):
    real = tmp_path / "real"
    mock = tmp_path / "mock"
    for base, mode in ((real, "real"), (mock, "mock_light")):
        for s in range(3):
            d = base / f"s{s}"
            d.mkdir(parents=True)
            (d / "evaluation_run.jsonl").write_text(
                _run(s, acc=0.5, mra=0.5, mode=mode).run.model_dump_json() + "\n",
                encoding="utf-8")
            (d / "online_run.jsonl").write_text(json.dumps(
                {"run_id": f"run-s{s}", "mode": mode, "baseline": "C1",
                 "seed": s, "split": "inner_validation"}) + "\n", encoding="utf-8")
    out = tmp_path / "table.json"
    rc = main_table_main([
        "--group", f"Ours={real}/s0,{real}/s1,{real}/s2",
        "--group", f"Mock={mock}/s0,{mock}/s1,{mock}/s2",
        "--out", str(out)])
    assert rc == 1                              # mock 组未过门禁
    data = json.loads(out.read_text(encoding="utf-8"))
    by_label = {g["label"]: g for g in data["groups"]}
    assert by_label["Ours"]["eligible_for_main_table"] is True
    assert by_label["Mock"]["eligible_for_main_table"] is False
    assert data["any_eligible"] and not data["all_eligible"]
    assert len(by_label["Ours"]["seeds"]) == 3
    txt = format_main_table(data)
    assert "Ours" in txt and "eligible=False" in txt


def test_cli_requires_runs(capsys):
    assert main_table_main([]) == 2


def test_cli_bad_group_spec(capsys):
    assert main_table_main(["--group", "no_equals_sign"]) == 2


def test_cli_all_real_passes(tmp_path):
    d = tmp_path / "runs"
    for s in range(3):
        sub = d / f"s{s}"
        sub.mkdir(parents=True)
        (sub / "evaluation_run.jsonl").write_text(
            _run(s, acc=0.5, mra=0.5).run.model_dump_json() + "\n", encoding="utf-8")
        (sub / "online_run.jsonl").write_text(json.dumps(
            {"run_id": f"run-s{s}", "mode": "real", "baseline": "C1",
             "seed": s, "split": "inner_validation"}) + "\n", encoding="utf-8")
    rc = main_table_main(["--runs", str(d / "s0"), str(d / "s1"), str(d / "s2"),
                          "--label", "Ours", "--out", str(tmp_path / "t.json")])
    assert rc == 0
