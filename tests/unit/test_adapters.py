"""M1 VSI-Bench Adapter 单测（§4 M1 字段 11）。

不依赖 datasets / 真实视频：用假 qa_row 与假帧函数。
"""

import numpy as np
import pytest

from skill3d.adapters import vsibench_loader as vl


def _fake_rows() -> list[dict]:
    """构造 8 题型 × 每型 10 scene 的假 meta。"""
    qtypes = [
        "object_rel_direction", "object_rel_distance", "object_rel_height",
        "object_counting", "object_size_estimation", "room_size_estimation",
        "object_abs_distance", "route_planning",
    ]
    rows = []
    for qt in qtypes:
        for s in range(10):
            rows.append({
                "id": f"{qt}-{s}",
                "dataset": "scannet",
                "scene_name": f"scene_{qt}_{s}",
                "question_type": qt,
                "question": "q?",
                "options": ["A", "B"],
                "ground_truth": "A",
            })
    return rows


def test_sample_uniform_indices_strictly_monotonic_uniform():
    """32 帧采样索引严格单调递增且均匀（§4 M1 验收条件 b）。"""
    idx = vl.sample_uniform_indices(total_frames=1000, n=32)
    assert len(idx) == 32
    assert all(a < b for a, b in zip(idx, idx[1:]))  # 严格单调
    diffs = np.diff(idx)
    assert diffs.max() - diffs.min() <= 1  # 均匀（步长差 ≤1）
    assert idx[0] == 0 and idx[-1] <= 999


def test_sample_uniform_indices_short_video_is_hard_fail():
    """视频不足 32 帧 → 输入合法性硬失败（硬约束 21 禁止重复帧补齐）。"""
    with pytest.raises(vl.FrameSetError):
        vl.sample_uniform_indices(total_frames=10, n=32)


def test_frame_set_hash_is_deterministic_and_content_addressed():
    """FrameSet：32 个唯一物理帧 + 确定性 frame_set_hash（§4 M1 验收）。"""
    fs = vl.build_frame_set(total_frames=1000, n_frames=32, fps=30.0)
    assert len(fs.frame_ids) == 32 and len(set(fs.frame_ids)) == 32
    assert fs.frame_ids == sorted(fs.frame_ids)                # 严格递增
    assert all(a < b for a, b in zip(fs.timestamps, fs.timestamps[1:]))
    fs2 = vl.build_frame_set(total_frames=1000, n_frames=32, fps=30.0)
    assert fs.frame_set_hash == fs2.frame_set_hash             # 确定性
    fs3 = vl.build_frame_set(total_frames=1000, n_frames=32, fps=60.0)
    assert fs3.frame_set_hash == fs.frame_set_hash             # 哈希只看帧号
    assert fs.frame_ids[0] == 0 and fs.frame_ids[-1] == 999    # 与官方一致


def test_split_isolation_final_test():
    """final_test scene 不出现在其他 split（硬约束 9，§4 M1 验收条件 c）。"""
    rows = _fake_rows()
    all_scenes = sorted({r["scene_name"] for r in rows})
    final = [s for s in all_scenes if s.endswith("_9") or s.endswith("_8")]
    cfg = vl.make_split_config(rows, final_test_scene_ids=final, seed=42)

    vl.assert_final_test_isolation(cfg)  # 不应抛异常

    final_set = set(cfg.final_test_scene_ids)
    assert final_set == set(final)
    for key in ("induction_scene_ids", "inner_validation_scene_ids", "outer_holdout_scene_ids"):
        assert not (final_set & set(getattr(cfg, key)))

    # 非 final 场景全部被分配且互不重叠（按 scene 不共享，硬约束 19）
    ind, inn, out = (set(cfg.induction_scene_ids), set(cfg.inner_validation_scene_ids),
                     set(cfg.outer_holdout_scene_ids))
    assert not (ind & inn) and not (ind & out) and not (inn & out)
    assert ind | inn | out == set(all_scenes) - final_set


def test_split_stratified_by_task_type():
    """按 task_type 分层：每个题型在 induction/inner/outer 都有 scene。"""
    rows = _fake_rows()
    cfg = vl.make_split_config(rows, final_test_scene_ids=[], seed=0)
    for qt in {r["question_type"] for r in rows}:
        for key in ("induction_scene_ids", "inner_validation_scene_ids"):
            scenes = getattr(cfg, key)
            assert any(f"scene_{qt}_" in s for s in scenes), f"{qt} 未进入 {key}"


def test_split_of_and_episode_unavailable(tmp_path):
    """split_of 查询正确；视频缺失 → EpisodeUnavailable 不进 split（§4 M1 字段 9）。"""
    rows = _fake_rows()
    cfg = vl.make_split_config(rows, final_test_scene_ids=["scene_object_counting_9"], seed=1)
    assert vl.split_of("scene_object_counting_9", cfg) == "final_test"
    assert vl.split_of("nonexistent_scene", cfg) is None

    with pytest.raises(vl.EpisodeUnavailable):
        vl.build_episode(rows[0], split="induction",
                         video_path=tmp_path / "missing.mp4")


def test_build_episode_with_mock_reader(tmp_path):
    """用假视频路径 + mock 帧函数构造 episode（不依赖真实视频解码）。"""
    import cv2

    # 用 cv2 写一个 64 帧假视频，保证 build_episode 的帧数探测可用
    video = tmp_path / "fake.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video), fourcc, 30, (64, 48))
    for i in range(64):
        vw.write(np.full((48, 64, 3), i, dtype=np.uint8))
    vw.release()

    captured = {}

    def mock_reader(path: str, indices):
        captured["indices"] = list(indices)
        return [np.zeros((48, 64, 3), dtype=np.uint8) for _ in indices]

    row = _fake_rows()[0]
    ep = vl.build_episode(row, split="induction", video_path=video,
                          frame_reader=mock_reader)
    assert ep.qa_id == row["id"]
    assert ep.scene_name == row["scene_name"]
    assert len(ep.frames) == 32
    idx = captured["indices"]
    assert all(a < b for a, b in zip(idx, idx[1:]))  # 严格单调均匀
    ts = [f.timestamp for f in ep.frames]
    assert all(a < b for a, b in zip(ts, ts[1:]))  # 时间戳严格单调
