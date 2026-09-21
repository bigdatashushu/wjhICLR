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
    """在线评测 CLI 端到端：退出码 0，EvaluationRun 与 trace 落盘（v6 语义）。"""
    trace_dir = tmp_path / "traces"
    r = _run("skill3d.online.eval", "--split", "inner_validation", "--source", "synthetic",
             "--mode", "mock_light", "--limit", "2", "--deterministic-replay",
             "--trace-dir", str(trace_dir), "--seed", "0",
             # 记忆与 RunManifest 也落在 tmp，避免污染仓库工作目录
             "--memory-dir", str(tmp_path / "mem"),
             "--run-manifest", str(tmp_path / "run_manifest.json"))
    assert r.returncode == 0, r.stderr
    assert "EvaluationRun" in r.stdout
    for topic in ("episode_trace", "evaluation_run", "online_run", "trace_record"):
        p = trace_dir / f"{topic}.jsonl"
        assert p.is_file() and p.read_text(encoding="utf-8").strip(), topic
    rec = json.loads((trace_dir / "online_run.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["mode"] == "mock_light"
    assert "不构成任何精度结论" in rec["note"]
    # §19.3：mock 路径的 program 来源显式标 mock_stub（v5 的 deterministic_stub 已废止）
    assert "mock_stub" in r.stdout
    trec = json.loads((trace_dir / "trace_record.jsonl").read_text(
        encoding="utf-8").splitlines()[0])
    assert trec["synthesis_source"] == "mock_stub"
    # §6.2：route 由真算的 M4 主门给出（合成 bundle 必须真的过门，不得绕过）
    assert trec["scene_route"] == "full_3d"
    assert trec["evidence_profile"]["geometry_3d"] == "available"
    # RunManifest 落在指定路径（§19.2）
    assert (tmp_path / "run_manifest.json").is_file()


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


def test_optimize_loop_pauses_without_offline_model(tmp_path):
    """§3.4：离线强模型（DeepSeek-V4.1-Flash）不可用 → 候选暂停不 promote（退出码 3），
    且不写入 active 快照。密钥未注入 → 记 `offline_auth_error`（不重试、不降级为 mock）。"""
    spec = {
        "skill_id": "sk-room-size", "version": "1.0.0",
        "applicable_question_types": ["room_size_estimation"],
        # v6 §5.8：米制 Skill 必须声明证据签名 + gate 版本（双重 fail-closed）
        "required_evidence_signature": {"metric_scale": "available"},
        "requires_metric_evidence": True,
        "applicable_gate_version": "metric-evidence-gate-v6",
        "skill_family": "metric", "source": "real",
        "description": "房间面积：平面拟合的地面两轴乘积",
        "call_graph_template": ("area = plane_fit_room_size()\n"
                                "ReturnAnswer(str(round(area['room_area_m2'], 2)))"),
        "validation_assertions": ["area['room_area_m2'] > 0"],
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
    assert "离线强模型不可用" in r.stderr
    assert "offline_auth_error" in r.stderr      # §3.4 结局码可审计（非"模型答得不好"）
    assert "准入必须 real" in r.stdout
    # 未 promote：没有任何 active 指针写入（硬约束 12）
    assert not (store / "active_snapshot.json").exists()
    # 泄漏检查与面板产物留档
    assert (tmp_path / "traces" / "leakage_check.jsonl").is_file()
    assert (tmp_path / "traces" / "paired_outcome.jsonl").is_file()
