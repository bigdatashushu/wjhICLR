"""CLI 冒烟测试（§13.5 三条入口）：

```bash
python -m skill3d.online.eval           --split inner_validation --source synthetic --mode mock_light
python -m skill3d.reconstruction.run    --source jsonl --plan-only
scripts/run_evolution_campaign_v11.py --help
```

用子进程跑真命令行（验证入口、参数、退出码、落盘），不依赖 GPU / 真实数据集。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from skill3d.online import synthetic as syn
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION


def _run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    src = str(Path(__file__).resolve().parents[2] / "src")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-m", *args], capture_output=True,
                          text=True, cwd=str(cwd) if cwd else None, timeout=900, env=env)


@pytest.mark.parametrize("module", ["skill3d.online.eval", "skill3d.reconstruction.run"])
def test_help_available(module):
    """三条 §13.5 入口均存在且可自述。"""
    r = _run(module, "--help")
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()


def test_online_eval_writes_traces(tmp_path):
    """在线评测 CLI 端到端：退出码 0，EvaluationRun 与 trace 落盘（v6 语义）。"""
    trace_dir = tmp_path / "traces"
    r = _run("skill3d.online.eval", "--split", "inner_validation", "--source", "synthetic",
             "--mode", "mock_light", "--limit", "2", "--deterministic-replay",
             "--trace-dir", str(trace_dir), "--seed", "0",
             # 记忆与 RunManifest 也落在 tmp，避免污染仓库工作目录

             "--run-manifest", str(tmp_path / "run_manifest.json"),
             "--frame-size", "120x160")
    assert r.returncode == 0, r.stderr
    assert "EvaluationRun" in r.stdout
    for topic in ("episode_trace", "evaluation_run", "online_run", "trace_record"):
        p = trace_dir / f"{topic}.jsonl"
        assert p.is_file() and p.read_text(encoding="utf-8").strip(), topic
    rec = json.loads((trace_dir / "online_run.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["mode"] == "mock_light"
    assert rec["template_version"] == PROMPT_TEMPLATE_VERSION
    assert "不构成任何精度结论" in rec["note"]
    # §19.3：mock 路径的 program 来源显式标 mock_stub（v5 的 deterministic_stub 已废止）
    assert "mock_stub" in r.stdout
    trec = json.loads((trace_dir / "trace_record.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    assert trec["synthesis_source"] == "mock_stub"
    assert trec["template_version"] == PROMPT_TEMPLATE_VERSION
    # §6.2：route 由真算的 M4 主门给出（合成 bundle 必须真的过门，不得绕过）
    assert trec["scene_route"] == "full_3d"
    assert trec["evidence_profile"]["geometry_3d"] == "available"
    # RunManifest 落在指定路径（§19.2）
    assert (tmp_path / "run_manifest.json").is_file()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["template_version"] == PROMPT_TEMPLATE_VERSION


def test_online_eval_writes_input_error_row_and_preserves_denominator(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({
        "qa_id": "missing-input",
        "scene_name": "missing-scene",
        "dataset": "scannet",
        "question_type": "object_rel_distance",
        "question": "Which object is closest?",
        "options": ["chair", "table"],
        "ground_truth": "A",
        "split": "inner_validation",
        "frame_paths": [str(tmp_path / "missing.png")],
    }) + "\n", encoding="utf-8")
    trace_dir = tmp_path / "traces"
    manifest_path = tmp_path / "run_manifest.json"

    result = _run(
        "skill3d.online.eval",
        "--split", "inner_validation",
        "--source", "jsonl",
        "--episodes-jsonl", str(episodes),
        "--mode", "mock_light",
        "--trace-dir", str(trace_dir),

        "--run-manifest", str(manifest_path),
    )

    assert result.returncode == 0, result.stderr
    evaluation = json.loads(
        (trace_dir / "evaluation_result.jsonl").read_text().splitlines()[0])
    run = json.loads((trace_dir / "evaluation_run.jsonl").read_text().splitlines()[0])
    manifest = json.loads(manifest_path.read_text())
    assert evaluation["qa_id"] == "missing-input"
    assert evaluation["episode_status"] == "input_error"
    assert evaluation["correct"] is False
    assert run["n_episodes"] == 1 and run["accuracy"] == 0.0
    assert manifest["n_input_error"] == 1
    assert manifest["denominator_preserved"] is True
    assert manifest["missing_result_qa_ids"] == []


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
    se = syn.make_synthetic_episode("room_size_estimation", scene_name="cli-scene-1", qa_id="c-1",
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
    }) for k, qt in enumerate(["room_size_estimation", "object_size_estimation"])) + "\n", encoding="utf-8")

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
