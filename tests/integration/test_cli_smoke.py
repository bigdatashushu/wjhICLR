"""CLI 冒烟测试（§13.5 三条入口）：

```bash
python -m skill3d.online.eval           --split inner_validation --source synthetic --mode mock_light
python -m skill3d.reconstruction.run    --source jsonl --plan-only
python -m skill3d.evolution.optimize_loop --spec-file <SkillSpec> --mode mock_light
```

用子进程跑真命令行（验证入口、参数、退出码、落盘），不依赖 GPU / 真实数据集。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from skill3d.online import synthetic as syn


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", *args], capture_output=True,
                          text=True, cwd=str(cwd) if cwd else None, timeout=900)


@pytest.mark.parametrize("module", ["skill3d.online.eval", "skill3d.reconstruction.run",
                                    "skill3d.evolution.optimize_loop"])
def test_help_available(module):
    """三条 §13.5 入口均存在且可自述。"""
    r = _run(module, "--help")
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()


def test_online_eval_writes_traces(tmp_path):
    """在线评测 CLI 端到端：退出码 0，EvaluationRun 与 trace 落盘。"""
    trace_dir = tmp_path / "traces"
    r = _run("skill3d.online.eval", "--split", "inner_validation", "--source", "synthetic",
             "--mode", "mock_light", "--limit", "2", "--deterministic-replay",
             "--trace-dir", str(trace_dir), "--seed", "0")
    assert r.returncode == 0, r.stderr
    assert "EvaluationRun" in r.stdout
    for topic in ("episode_trace", "evaluation_run", "online_run"):
        p = trace_dir / f"{topic}.jsonl"
        assert p.is_file() and p.read_text(encoding="utf-8").strip(), topic
    rec = json.loads((trace_dir / "online_run.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["mode"] == "mock_light"
    assert "不构成任何精度结论" in rec["note"]


def test_online_eval_blocks_final_test_without_flag(tmp_path):
    """硬约束 9：final_test / --split test 默认被拒（退出码 2）。"""
    r = _run("skill3d.online.eval", "--split", "test", "--source", "synthetic",
             "--mode", "mock_light", "--limit", "1", "--trace-dir", str(tmp_path / "t"))
    assert r.returncode == 2
    assert "硬约束 9" in r.stderr


def test_online_eval_rejects_real_with_synthetic(tmp_path):
    """mode=real 必须配真实数据源（合成数据只允许 mock_light，§9.2）。"""
    r = _run("skill3d.online.eval", "--split", "inner_validation", "--source", "synthetic",
             "--mode", "real", "--limit", "1", "--trace-dir", str(tmp_path / "t"))
    assert r.returncode == 2
    assert "mock_light" in r.stderr


def test_reconstruction_plan_only(tmp_path):
    """重建 CLI：--plan-only 按 scene 归并并列出作业表，不触发真实重建。"""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    se = syn.make_synthetic_episode("room_size", scene_name="cli-scene-1", qa_id="c-1",
                                    frame_size=(60, 80))
    frame_paths = []
    for i, px in enumerate(se.frames):
        fp = frames_dir / f"f{i:02d}.png"
        assert cv2.imwrite(str(fp), cv2.cvtColor(px, cv2.COLOR_RGB2BGR))
        frame_paths.append(str(fp))
    jl = tmp_path / "episodes.jsonl"
    jl.write_text("\n".join(json.dumps({
        "qa_id": f"c-{k}", "scene_name": "cli-scene-1", "dataset": "scannet",
        "question_type": qt, "question": "q", "options": None, "ground_truth": "1.0",
        "split": "induction", "frame_paths": frame_paths,
    }) for k, qt in enumerate(["room_size", "object_size"])) + "\n", encoding="utf-8")

    r = _run("skill3d.reconstruction.run", "--source", "jsonl", "--episodes-jsonl", str(jl),
             "--split", "induction", "--method", "vggt", "--plan-only",
             "--recon-dir", str(tmp_path / "recon"))
    assert r.returncode == 0, r.stderr
    assert "cli-scene-1" in r.stdout and "episodes=2" in r.stdout


def test_reconstruction_blocks_final_test(tmp_path):
    """硬约束 9：final test 不参与批量重建。"""
    r = _run("skill3d.reconstruction.run", "--split", "induction,final_test",
             "--method", "vggt", "--plan-only")
    assert r.returncode == 2
    assert "硬约束 9" in r.stderr


def test_optimize_loop_pauses_without_gpt6(tmp_path):
    """§8：GPT-6 未配置 → 候选暂停不 promote（退出码 3），且不写入 active 快照。"""
    spec = {
        "skill_id": "sk-room-size", "semver": "1.0.0", "task_type": "room_size",
        "description": "房间面积：包围盒地面两轴乘积",
        "call_graph_template": "ReturnAnswer(str(round(room_size_m2(), 2)))",
        "requires_artifacts": ["objects"], "minimum_quality": 0.3,
        "supported_coordinate_frames": ["world"], "metric_scale_required": True,
        "validation_assertions": ["area > 0"],
    }
    spec_file = tmp_path / "skill.json"
    spec_file.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    store = tmp_path / "store"
    r = _run("skill3d.evolution.optimize_loop", "--spec-file", str(spec_file),
             "--root-candidate-id", "cand-cli", "--mode", "mock_light",
             "--panel-source", "synthetic", "--l1-limit", "1", "--limit", "1",
             "--seed", "0", "--skill-store", str(store),
             "--trace-dir", str(tmp_path / "traces"))
    assert r.returncode == 3, (r.stdout, r.stderr)
    assert "GPT-6 不可用" in r.stderr
    assert "准入必须 real" in r.stdout
    # 未 promote：没有任何 active 指针写入（硬约束 12）
    assert not (store / "active_snapshot.json").exists()
    # 泄漏检查与面板产物留档
    assert (tmp_path / "traces" / "leakage_check.jsonl").is_file()
    assert (tmp_path / "traces" / "paired_outcome.jsonl").is_file()
