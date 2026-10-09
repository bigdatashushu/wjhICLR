"""v9 P5b：先冻结清单再加载输入（§5.3，§17.3 的"少帧输入／完整评估分母"）。

规范原文（§5.3）：

> 评估清单从题目元数据与预登记的 scene／任务采样规则生成，**不以视频存在、成功解码
> 或工具成功为筛选条件**。随后逐项加载，**每个预登记 qa_id 必须有结果行**；不足 32 帧
> 不能通过 skip 静默消失。适配错误单独记录，保留总体分母。
> 开发抽样以 **scene** 为单位控制覆盖与缓存成本，同时保证任务和方向难度覆盖。
> **禁止**默认按文件行序取每类前 N 题后把该子集当成随机代表样本；抽样算法、
> scene／qa_id 清单、seed 和 hash 都落盘。

守三件事：抽样与行序无关且可复现、抽样不看视频存在性、被丢弃的预登记 qa_id 有结果行。
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from skill3d.adapters.episode_source import (
    _sample_by_scene,
    _sampling_receipt,
    load_vsi_bench_items,
)
from skill3d.routing.task_classifier import canonical_task


def _rows(n_scenes: int = 12, per_scene: int = 4,
          task: str = "object_counting") -> list[dict]:
    rows = []
    for s in range(n_scenes):
        for k in range(per_scene):
            rows.append({"qa_id": f"q{s:02d}{k}", "scene_name": f"scene{s:02d}",
                         "question_type": task, "dataset": "scannet"})
    return rows


# --------------------------------------------- 抽样：与行序无关、可复现 ----

def test_sampling_is_not_row_order_first_n():
    """§5.3 明文禁止"按文件行序取每类前 N 题后当成随机代表样本"。"""
    rows = _rows()
    picked = _sample_by_scene(rows, canon=canonical_task, per_task_cap=8, seed=0)
    assert len(picked) == 8
    first_by_row_order = [r["qa_id"] for r in rows[:8]]
    assert [r["qa_id"] for r in picked] != first_by_row_order


def test_sampling_is_reproducible_for_the_same_seed():
    rows = _rows()
    a = _sample_by_scene(rows, canon=canonical_task, per_task_cap=8, seed=0)
    b = _sample_by_scene(rows, canon=canonical_task, per_task_cap=8, seed=0)
    assert a == b


def test_sampling_changes_with_the_seed():
    rows = _rows()
    a = _sample_by_scene(rows, canon=canonical_task, per_task_cap=8, seed=0)
    b = _sample_by_scene(rows, canon=canonical_task, per_task_cap=8, seed=1)
    assert a != b


def test_sampling_is_scene_unit_and_covers_every_task():
    """§5.3：以 scene 为单位；同时保证任务覆盖（每个题型都有名额）。"""
    rows = _rows(task="object_counting") + [
        {"qa_id": f"r{i}", "scene_name": f"other{i}", "question_type": "room_size_estimation",
         "dataset": "scannet"} for i in range(6)]
    picked = _sample_by_scene(rows, canon=canonical_task, per_task_cap=3, seed=0)
    tasks = {canonical_task(r["question_type"]) for r in picked}
    assert tasks == {"object_counting", "room_size_estimation"}
    for t in tasks:
        assert sum(1 for r in picked if canonical_task(r["question_type"]) == t) == 3


def test_sampling_does_not_depend_on_input_row_order():
    """同 seed 下，打乱输入行序不改变抽样结果（只由 (seed, 题型, scene) 决定）。"""
    rows = _rows()
    shuffled = list(reversed(rows))
    a = _sample_by_scene(rows, canon=canonical_task, per_task_cap=8, seed=7)
    b = _sample_by_scene(shuffled, canon=canonical_task, per_task_cap=8, seed=7)
    assert {r["qa_id"] for r in a} == {r["qa_id"] for r in b}


# --------------------------------------------- 抽样收据落盘（§5.3）----

def test_sampling_receipt_carries_algorithm_seed_list_and_hash():
    picked = _sample_by_scene(_rows(), canon=canonical_task, per_task_cap=8, seed=3)
    receipt = _sampling_receipt(picked, seed=3, strategy="seeded_scene_stratified",
                                per_task_cap=8)
    assert receipt["strategy"] == "seeded_scene_stratified"
    assert receipt["seed"] == 3
    assert receipt["n_preregistered"] == len(picked)
    assert receipt["qa_ids"] == [r["qa_id"] for r in picked]
    assert receipt["scenes"] and receipt["qa_id_sha256"]
    # hash 必须由清单内容决定（可复算）
    import hashlib

    assert receipt["qa_id_sha256"] == hashlib.sha256(
        "\n".join(receipt["qa_ids"]).encode()).hexdigest()


def test_receipt_hash_is_stable_across_runs_with_same_seed():
    def once() -> str:
        picked = _sample_by_scene(_rows(), canon=canonical_task, per_task_cap=8, seed=5)
        return _sampling_receipt(picked, seed=5, strategy="seeded_scene_stratified",
                                 per_task_cap=8)["qa_id_sha256"]

    assert once() == once()


# --------------------------------------------- 排除行与分母（§5.3）----

def test_loader_exposes_exclusions_and_sampling_receipt_parameters():
    params = inspect.signature(load_vsi_bench_items).parameters
    assert "exclusions" in params and "sampling_receipt" in params
    assert "sampling_seed" in params, "抽样口径必须与 seed 绑定"


def test_missing_preregistered_qa_ids_are_recorded_with_reasons(tmp_path, monkeypatch):
    """§5.3：不足 32 帧／视频缺失不得静默消失 —— 每条都要有结果行与原因。"""
    from skill3d.adapters import vsibench_loader
    from skill3d.adapters.episode_source import load_vsi_bench_items as loader

    # 元数据是输入夹具；不下载数据，也不依赖本机恰好存在的官方 meta。
    monkeypatch.setattr(vsibench_loader, "load_meta", lambda **kwargs: [{
        "id": "missing-video-qa", "qa_id": "missing-video-qa",
        "dataset": "scannet", "scene_name": "__no_such_scene__",
        "question_type": "object_rel_distance", "question": "Which is closest?",
        "options": ["chair", "table"], "ground_truth": "A",
    }])
    exclusions: list = []
    receipt: dict = {}
    try:
        loader("inner_validation", {"inner_validation_scene_ids": ["__no_such_scene__"]},
               video_root=str(tmp_path / "no_videos"), exclusions=exclusions,
               sampling_receipt=receipt)
    except Exception as exc:  # noqa: BLE001 - 无可用 episode 时抛来源错误（预期）
        assert "无可用 episode" in str(exc)
    # 抽样收据即使在无视频时也已生成（清单先于加载冻结，§5.3）
    assert receipt.get("strategy") in ("seeded_scene_stratified", "declared_order")
    assert receipt["qa_ids"] == ["missing-video-qa"]
    assert len(exclusions) == 1
    exclusion = exclusions[0]
    assert exclusion["status"] == "input_error"
    assert exclusion["qa_id"] == "missing-video-qa"
    assert exclusion["scene_name"] == "__no_such_scene__"
    assert exclusion["reason"] == "video_missing"
    assert exclusion["question_type"] == "object_rel_distance"
    assert exclusion["ground_truth"] == "A"
    assert exclusion["source_attempts"]


def test_input_error_is_a_zero_score_result_and_preserves_the_denominator(
        tmp_path, monkeypatch):
    from skill3d.adapters import vsibench_loader
    from skill3d.online.runner import OnlineRunConfig, run_split
    from skill3d.trace.store import TraceStore

    monkeypatch.setattr(vsibench_loader, "load_meta", lambda **kwargs: [{
        "id": "missing-video-qa", "qa_id": "missing-video-qa",
        "dataset": "scannet", "scene_name": "__no_such_scene__",
        "question_type": "object_rel_distance", "question": "Which is closest?",
        "options": ["chair", "table"], "ground_truth": "A",
    }])
    exclusions: list = []
    items = load_vsi_bench_items(
        "inner_validation",
        {"inner_validation_scene_ids": ["__no_such_scene__"]},
        video_root=str(tmp_path / "no_videos"),
        exclusions=exclusions,
        include_input_errors=True,
    )

    assert len(items) == 1 and items[0].input_error is not None
    trace_dir = tmp_path / "traces"
    outcomes, run = run_split(
        items,
        OnlineRunConfig(mode="real", trace_dir=str(trace_dir), memory_dir=""),
        trace_store=TraceStore(trace_dir),
    )

    out = outcomes[0]
    assert out.final_state == out.episode_status == "input_error"
    assert out.correct is False and out.answer is None
    assert run.n_episodes == 1 and run.accuracy == 0.0
    evaluation = json.loads(
        (trace_dir / "evaluation_result.jsonl").read_text().splitlines()[0])
    assert evaluation["qa_id"] == "missing-video-qa"
    assert evaluation["episode_status"] == "input_error"
    assert evaluation["input_error_reason"] == "video_missing"


def test_manifest_records_sampling_and_exclusions_fields():
    """§5.3：抽样算法／seed／清单／hash 与排除行都进 RunManifest。"""
    src = Path("src/skill3d/online/eval.py").read_text(encoding="utf-8")
    for key in ("sampling_strategy", "sampling_seed", "sampling_qa_ids",
                "sampling_qa_id_sha256", "exclusions", "denominator_preserved"):
        assert key in src, f"RunManifest 未记录 {key}"


def test_denominator_manifest_is_derived_from_actual_result_ids():
    from skill3d.online.eval import _denominator_fields

    complete = _denominator_fields(
        [
            SimpleNamespace(qa_id="ok", episode_status="answered"),
            SimpleNamespace(qa_id="bad-input", episode_status="input_error"),
        ],
        {"qa_ids": ["ok", "bad-input"]},
        [{"qa_id": "bad-input", "status": "input_error"}],
    )
    missing = _denominator_fields(
        [SimpleNamespace(qa_id="ok", episode_status="answered")],
        {"qa_ids": ["ok", "bad-input"]},
        [{"qa_id": "bad-input", "status": "input_error"}],
    )

    assert complete["denominator_preserved"] is True
    assert complete["n_input_error"] == 1
    assert missing["denominator_preserved"] is False
    assert missing["missing_result_qa_ids"] == ["bad-input"]


def test_eval_cli_still_imports_cleanly():
    out = subprocess.run([sys.executable, "-c",
                          "import skill3d.online.eval as m; print('ok')"],
                         capture_output=True, text=True,
                         env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert "ok" in out.stdout, out.stderr[-500:]
