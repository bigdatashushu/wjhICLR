"""M3 重建 CLI 的纯逻辑单测（无 GPU / 无数据即可跑）：

- `plan_scene_jobs`：按 scene 归并 episode、已有 artifact 跳过（断点恢复幂等，§4 M21）；
- artifact JSON 往返：主门/诊断项的 NaN 占位必须可无损序列化/回读
  （默认 null 序列化会让 artifact 落盘后读不回来，这里做回归保护）；
  v6 artifact 只接受 `recon_method="vggt"` 且拒绝 v5 尺度字段（§5.2/§20）；
- `run_jobs`：无权重/依赖时逐 scene 失败但不崩，状态与 note 落账。
"""

from __future__ import annotations

import json

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


def _item(scene: str, qa_id: str,
          question_type: str = "room_size_estimation") -> EpisodeItem:
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
        track_list=None,
        quality_status="computed",
        quality=QualityMetrics(
            warp_inlier_ratio=nan, warp_photometric_inlier_ratio=nan,
            cloud_overlap_ratio=nan, main_gate_passed=False,
            g1_blur_ok=nan, g2_brightness=nan, g3_motion_blur=nan, g4_frame_count=32,
            g6_depth_var_coeff=nan, g7_dynamic_ratio=nan, g9_tracker_consistency=nan,
            g10_baseline_quality=nan, overall_quality=0.0),
        confidence={"per_point_confidence": "", "coverage_count_per_frame": ""},
    )
    p = tmp_path / "art.json"
    p.write_text(art.model_dump_json(), encoding="utf-8")
    back = ReconstructionArtifact.model_validate_json(p.read_text(encoding="utf-8"))
    assert back.quality.overall_quality == 0.0                 # 主门未过 → 0.0（§6.2）
    assert np.isnan(back.quality.warp_inlier_ratio)          # NaN 主门指标无损往返
    assert back.quality.g4_frame_count == 32
    assert back.recon_method == "vggt" and back.schema_version == "6.0"
    # G5 固定 not_available + None（BA 已关闭，§10.4）
    assert back.reprojection_status == "not_available"
    assert back.g5_reproj_err_median is None and back.g5_reproj_err_p95 is None
    # v5 的尺度字段落盘后再读回必须 hard fail（Schema 与 legacy 隔离，§5.2/§20）
    legacy = json.loads(p.read_text(encoding="utf-8"))
    legacy["scale_known"] = True
    with pytest.raises(Exception):
        ReconstructionArtifact.model_validate(legacy)


def test_run_jobs_reports_failure_without_crash(tmp_path, monkeypatch):
    """重建失败时：逐 scene 记 failed + note，不抛异常（§4 M3 字段 9）。

    单测不依赖 GPU/权重：把真实重建替换为确定性失败（否则会因他人占满显存而 OOM，
    错误信息随环境变化 —— 那是集成测试该覆盖的事，不是本单测的断言点）。
    """
    from skill3d.reconstruction import run as recon_run
    from skill3d.reconstruction.vggt_runner import ReconstructionFailed

    def _boom(*_a, **_kw):
        raise ReconstructionFailed("单测桩：权重缺失")

    monkeypatch.setattr(recon_run, "reconstruct", _boom, raising=False)
    items = [_item("scene-a", "a1")]
    jobs = plan_scene_jobs(items, tmp_path, "vggt")
    jobs = run_jobs(jobs, {"scene-a": items}, tmp_path, "vggt", gpus=[0])
    assert jobs[0].status == "failed"
    assert jobs[0].gpu_rank == 0
    assert "重建失败" in jobs[0].note or "单测桩" in jobs[0].note


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
    v6 的"BA 关闭"不再是 `use_ba=False` 参数，而是 `recon_method` 受控枚举只有
    `vggt`（§5.2/§20）—— 因此这里断言方法名与帧数，并断言产物里没有 legacy 字段。
    """
    import skill3d.reconstruction.run as run_mod
    from skill3d.schemas import ConfidenceMap, QualityMetrics, ReconstructionArtifact

    captured: dict = {}
    nan = float("nan")

    def fake_reconstruct(frames, scene_name, output_dir, method="vggt", **kwargs):
        captured[scene_name] = len(frames)
        captured["method"] = method
        captured["kwargs"] = sorted(kwargs)
        return ReconstructionArtifact(
            artifact_id="x", artifact_version="x", scene_name=scene_name,
            recon_method="vggt", c2w_list="", intrinsics="", depth_maps="",
            point_map="", point_conf="", track_list=None,
            quality_status="computed",
            quality=QualityMetrics(
                warp_inlier_ratio=0.9, warp_photometric_inlier_ratio=0.9,
                cloud_overlap_ratio=0.9, main_gate_passed=True,
                g1_blur_ok=nan, g2_brightness=nan, g3_motion_blur=nan,
                g4_frame_count=32, g6_depth_var_coeff=nan, g7_dynamic_ratio=nan,
                g9_tracker_consistency=nan, g10_baseline_quality=nan,
                overall_quality=0.9),
            confidence=ConfidenceMap(per_point_confidence="", coverage_count_per_frame=""),
        )

    monkeypatch.setattr(run_mod, "reconstruct", fake_reconstruct)
    items = [_item("scene-a", "a1"), _item("scene-a", "a2"), _item("scene-a", "a3")]
    jobs = plan_scene_jobs(items, tmp_path, "vggt")
    jobs = run_jobs(jobs, {"scene-a": items}, tmp_path, "vggt", gpus=[0])
    assert captured["scene-a"] == 32          # 只喂一份 32 帧，不拼接
    assert captured["method"] == "vggt"       # 唯一正式主线（无 BA / 无降级链）
    assert "use_ba" not in captured["kwargs"]
    assert jobs[0].status == "done"
    # v9 §5.2：产物按**缓存身份**（源标识 + 帧集内容哈希）命名，不再是纯 scene 名 ——
    # 用与运行时同一个解析器定位，顺带守住"写方与读方对路径的判断一致"。
    from skill3d.reconstruction.run import artifact_path as _ap

    path = _ap(tmp_path, "scene-a", "vggt",
               frame_set=items[0].episode.frame_set)
    assert path.name != "scene-a.json", "缓存键必须含源标识与帧集哈希（§5.2）"
    assert path.name.startswith("scene-a__")
    # 名字里的身份必须与 frame_set 的 cache_identity 一致（可复算、非随意命名）
    assert path.name == f"scene-a__{items[0].episode.frame_set.cache_identity()[:16]}.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["recon_method"] == "vggt"
    assert not {"scale_known", "scale_ci_rel", "allowed_metric_tasks"} & set(on_disk)
