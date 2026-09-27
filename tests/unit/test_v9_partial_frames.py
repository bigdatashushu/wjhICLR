"""v9 P5f：§5.2 缺失掩码回填 —— 部分帧解码失败时用可读帧继续。

规范原文（§5.2）："**部分**帧无法解码，或源视频不足 32 帧但仍有真实可读帧：保留帧身份
和缺失掩码，使用可读帧继续作答；**不重复填充或偷偷改采样**。依赖完整帧集的工具按条件
禁用。部分缺帧样本标记 `input_degraded`，独立报告；不伪称完整 32 帧输入，也**不从评分
分母静默删除**。**成对实验必须复用同一实际可读集合**。"

守四件事：解码失败不再让整题失败、缺失帧号被如实记录、像素里不插占位帧、
成对比较会检查实际可读集合。
"""

from __future__ import annotations

import numpy as np
import pytest

from skill3d.adapters import frame_set as fs
from skill3d.adapters.episode_source import _read_frames


# --------------------------------------------- 解码：容错 + 缺失掩码 ----

def _write_video(path, n_frames: int = 64) -> str:
    import cv2

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(n_frames):
        vw.write(np.full((48, 64, 3), i % 255, dtype=np.uint8))
    vw.release()
    return str(path)


def test_read_frames_returns_readable_pixels_and_missing_ids(tmp_path):
    """读失败只进缺失清单，不再让整个 episode 失败（§5.2）。"""
    video = _write_video(tmp_path / "ok.mp4")
    planned = [0, 5, 10, 15]
    frames, missing = _read_frames(video, planned)
    assert len(frames) == 4 and missing == []
    assert all(isinstance(f, np.ndarray) and f.size > 0 for f in frames)


def test_read_frames_keeps_identity_without_padding(tmp_path):
    """规划的帧序保留、缺失帧不插占位、也不重复邻近帧（§5.2"不重复填充"）。"""
    video = _write_video(tmp_path / "ok.mp4", 32)
    planned = [0, 2, 4, 6]
    # 越过视频末尾的帧号 → 读不出来，进缺失清单
    frames, missing = _read_frames(video, planned[:3] + [10_000])
    assert missing == [10_000]
    # 返回的像素数 = 可读帧数（没有占位帧凑数）
    assert len(frames) == 3
    assert len(frames) == len(planned) - len(missing)


def test_undecodable_video_reports_every_frame_as_missing(tmp_path):
    """一帧都读不出 → 调用方按 §5.2 `input_error` 记排除（此处只验证掩码完整）。"""
    video = _write_video(tmp_path / "ok.mp4", 8)
    frames, missing = _read_frames(video, [100_000, 200_000])
    assert frames == [] and missing == [100_000, 200_000]


# --------------------------------------------- FrameSet：掩码 + partial ----

def test_build_frame_set_accepts_a_readable_subset():
    planned = fs.uniform_frame_ids(64, 32)
    readable = planned[:30]
    fset = fs.build_frame_set(64, readable_frame_ids=readable)
    assert fset.readable_frame_ids == readable
    assert len(fset.frame_ids) == 32, "规划帧身份完整保留"
    assert fset.decode_status == "partial"


def test_build_frame_set_default_means_all_readable():
    fset = fs.build_frame_set(64)
    assert fset.readable_frame_ids == fset.frame_ids
    assert fset.decode_status == "ok"


def test_readable_subset_must_be_planned_frames():
    with pytest.raises(Exception):
        fs.build_frame_set(64, readable_frame_ids=[1])   # 1 不是 64→32 采样出的槽位


def test_pairing_requires_the_same_readable_set():
    """§5.2："成对实验必须复用同一实际可读集合" —— 规划帧相同还不够。"""
    planned = fs.uniform_frame_ids(64, 32)
    a = fs.build_frame_set(64, readable_frame_ids=planned)
    b = fs.build_frame_set(64, readable_frame_ids=planned[:30])
    assert a.frame_set_hash == b.frame_set_hash, "规划帧相同 → 内容哈希相同"
    with pytest.raises(fs.FrameSetError) as exc:
        fs.assert_same_frame_set(a, b)
    assert "实际可读帧集不一致" in str(exc.value)


def test_pairing_passes_for_identical_readable_sets():
    planned = fs.uniform_frame_ids(64, 32)
    a = fs.build_frame_set(64, readable_frame_ids=planned[:30])
    b = fs.build_frame_set(64, readable_frame_ids=planned[:30])
    fs.assert_same_frame_set(a, b)     # 不抛异常即通过


# --------------------------------------------- 下游：按可读帧数对齐 ----

def test_runner_expects_readable_frame_count_not_planned():
    """M8 的帧数断言必须按**实际可读**帧数，否则部分解码会在提示词层炸掉。"""
    import inspect

    from skill3d.online import runner

    src = inspect.getsource(runner)
    assert "len(episode.frame_set.readable_frame_ids)" in src
    assert "len(episode.frame_set.frame_ids)" not in src


def test_partial_frame_set_still_passes_the_prompt_alignment_check():
    """部分可读帧集 + 对应像素 → 提示词层对齐检查通过。"""
    from skill3d.online.runner import _prompt_messages

    planned = fs.uniform_frame_ids(64, 32)
    readable = planned[:3]
    fset = fs.build_frame_set(64, readable_frame_ids=readable)
    pixels = [np.full((48, 64, 3), 100, dtype=np.uint8) for _ in readable]
    messages = _prompt_messages("问题", pixels, _cfg(), expected_frames=len(readable))
    assert messages and any(m.get("role") == "user" for m in messages)


def _cfg():
    from skill3d.online.runner import OnlineRunConfig

    return OnlineRunConfig(mode="real", max_images=32)
