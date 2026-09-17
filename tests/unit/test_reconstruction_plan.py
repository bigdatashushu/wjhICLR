"""M3 重建 CLI 的纯逻辑单测（无 GPU / 无数据即可跑）：

- `plan_scene_jobs`：按 scene 归并 episode、已有 artifact 跳过（断点恢复幂等，§4 M21）；
- artifact JSON 往返：G1–G11 的 NaN 占位必须可无损序列化/回读
  （默认 null 序列化会让 artifact 落盘后读不回来，这里做回归保护）；
- `run_jobs`：无权重/依赖时逐 scene 失败但不崩，状态与 note 落账。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.online import synthetic as syn
from skill3d.reconstruction.run import (
    SceneJob,
    artifact_path,
    plan_scene_jobs,
    run_jobs,
)
from skill3d.schemas import QualityMetrics, ReconstructionArtifact


def _item(scene: str, qa_id: str, question_type: str = "room_size") -> EpisodeItem:
    se = syn.make_synthetic_episode(question_type, scene_name=scene, qa_id=qa_id,
                                    frame_size=(60, 80))
    return EpisodeItem(episode=se.episode, pixels=se.frames, geometry=se.geometry,
                       source="synthetic")


def test_plan_groups_by_scene():
    """同一 scene 的多个 episode 只产生一个重建作业（§4 M3：重建以 scene 为单位）。"""
    items = [_item("scene-a", "a1"), _item("scene-a", "a2"), _item("scene-b", "b1")]
    jobs = plan_scene_jobs(items, "/tmp/recon", "vggt")
    assert [j.scene_name for j in jobs] == ["scene-a", "scene-b"]
    assert jobs[0].n_episodes == 2 and jobs[1].n_episodes == 1
    assert all(j.status == "pending" for j in jobs)


def test_plan_skips_existing_artifact(tmp_path):
    """已有 artifact → 默认跳过（断点恢复幂等）；--force 时重跑。"""
    items = [_item("scene-a", "a1")]
    out = artifact_path(tmp_path, "scene-a", "vggt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{}", encoding="utf-8")

    jobs = plan_scene_jobs(items, tmp_path, "vggt")
    assert jobs[0].status == "skipped" and jobs[0].artifact_ref == str(out)

    jobs_force = plan_scene_jobs(items, tmp_path, "vggt", force=True)
    assert jobs_force[0].status == "pending"


def test_artifact_json_roundtrip_keeps_nan(tmp_path):
    """NaN 占位必须能 JSON 往返（否则落盘的 artifact 读不回来）。"""
    nan = float("nan")
    art = ReconstructionArtifact(
        artifact_id="a1", artifact_version="v1", scene_name="s1", recon_method="vggt",
        c2w_list="", intrinsics="", depth_maps="", point_map="", point_conf="",
        track_list=None, metric_scale=None, scale_known=False,
        quality=QualityMetrics(
            g1_blur_ok=nan, g2_brightness=nan, g3_motion_blur=nan, g4_frame_count=32,
            g5_reproj_err_median=nan, g5_reproj_err_p95=nan, g6_depth_var_coeff=nan,
            g7_dynamic_ratio=nan, g8_bbox_coverage_min=nan, g9_tracker_consistency=nan,
            g10_baseline_quality=nan, g11_scale_ci=nan, overall_quality=nan),
        confidence={"per_point_confidence": "", "coverage_count_per_frame": ""},
    )
    p = tmp_path / "art.json"
    p.write_text(art.model_dump_json(), encoding="utf-8")
    back = ReconstructionArtifact.model_validate_json(p.read_text(encoding="utf-8"))
    assert np.isnan(back.quality.overall_quality)
    assert back.quality.g4_frame_count == 32


def test_run_jobs_reports_failure_without_crash(tmp_path, monkeypatch):
    """无 VGGT 权重/依赖时：逐 scene 记 failed + note，不抛异常（§4 M3 字段 9）。"""
    items = [_item("scene-a", "a1")]
    jobs = plan_scene_jobs(items, tmp_path, "vggt")
    jobs = run_jobs(jobs, {"scene-a": items}, tmp_path, "vggt", gpus=[0])
    assert jobs[0].status == "failed"
    assert jobs[0].gpu_rank == 0
    assert "重建失败" in jobs[0].note


def test_run_jobs_dp_assignment_across_gpus(tmp_path):
    """多 scene 在 DP 下按卡轮转分配（M20）。"""
    items = [_item(f"scene-{i}", f"q{i}") for i in range(3)]
    jobs = plan_scene_jobs(items, tmp_path, "vggt")
    jobs = run_jobs(jobs, {j.scene_name: [it] for j, it in zip(jobs, items)},
                    tmp_path, "vggt", gpus=[0, 1])
    assert [j.gpu_rank for j in jobs] == [0, 1, 0]


def test_run_jobs_passes_one_episode_frames_per_scene(tmp_path, monkeypatch):
    """回归：同 scene 多 episode 来自同一段视频 → 只喂一份 32 帧，不拼接多份。

    拼接会把 32×N 帧塞进重建（错误输入），这里用替身捕获真实入参。
    """
    import skill3d.reconstruction.run as run_mod
    from skill3d.schemas import ConfidenceMap, QualityMetrics, ReconstructionArtifact

    captured: dict[str, int] = {}
    nan = float("nan")

    def fake_reconstruct(frames, scene_name, output_dir, method="vggt"):
        captured[scene_name] = len(frames)
        return ReconstructionArtifact(
            artifact_id="x", artifact_version="x", scene_name=scene_name,
            recon_method="vggt", c2w_list="", intrinsics="", depth_maps="",
            point_map="", point_conf="", track_list=None, metric_scale=None,
            scale_known=False,
            quality=QualityMetrics(
                g1_blur_ok=nan, g2_brightness=nan, g3_motion_blur=nan, g4_frame_count=32,
                g5_reproj_err_median=nan, g5_reproj_err_p95=nan, g6_depth_var_coeff=nan,
                g7_dynamic_ratio=nan, g8_bbox_coverage_min=nan, g9_tracker_consistency=nan,
                g10_baseline_quality=nan, g11_scale_ci=nan, overall_quality=nan),
            confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""),
        )

    monkeypatch.setattr(run_mod, "reconstruct", fake_reconstruct)
    items = [_item("scene-a", "a1"), _item("scene-a", "a2"), _item("scene-a", "a3")]
    jobs = plan_scene_jobs(items, tmp_path, "vggt")
    jobs = run_jobs(jobs, {"scene-a": items}, tmp_path, "vggt", gpus=[0])
    assert captured == {"scene-a": 32}
    assert jobs[0].status == "done"
