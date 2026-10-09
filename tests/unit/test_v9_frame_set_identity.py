"""v9 P5c：FrameSet 的 §5.2 字段、源标识与缓存身份。

规范原文（§5.2）：

> `FrameSet` 至少包含：`schema_version, episode_id, dataset_id, video_id, scene_name,
> frame_ids, source_frame_indices, timestamps, frame_refs, frame_set_hash, decode_status,
> readable_frame_ids, preprocessing_version`。数组字段长度匹配，帧索引唯一且有序；
> **源标识及内容校验值参与缓存身份，禁止仅凭相同的帧索引列表跨视频复用。**

守两件事：规范点名的字段齐备且自洽；**同帧索引、不同视频不得复用同一缓存身份**。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from skill3d.adapters import frame_set as fs
from skill3d.schemas.episode import FrameSet

SPEC_FIELDS = ("schema_version", "episode_id", "dataset_id", "video_id", "scene_name",
               "frame_ids", "source_frame_indices", "timestamps", "frame_refs",
               "frame_set_hash", "decode_status", "readable_frame_ids",
               "preprocessing_version")


def test_frame_set_has_every_doc_required_field():
    assert set(SPEC_FIELDS) <= set(FrameSet.model_fields)


def test_build_frame_set_fills_source_identity():
    fset = fs.build_frame_set(64, n_frames=32, fps=30.0, episode_id="qa-1",
                              dataset_id="scannet", video_id="vid-9",
                              scene_name="scene-3")
    assert fset.episode_id == "qa-1"
    assert fset.dataset_id == "scannet"
    assert fset.video_id == "vid-9"
    assert fset.scene_name == "scene-3"
    assert fset.preprocessing_version
    assert fset.decode_status == "ok"


def test_readable_frame_ids_defaults_to_all_planned_frames():
    """当前实现下解码失败即硬失败，因此正常路径"规划帧 = 可读帧"（不伪称）。"""
    fset = fs.build_frame_set(64, n_frames=32, fps=30.0)
    assert fset.readable_frame_ids == fset.frame_ids
    assert len(fset.readable_frame_ids) == 32


def test_readable_frame_ids_must_be_a_subset_of_planned_frames():
    with pytest.raises(ValidationError):
        FrameSet(frame_ids=[1, 2], source_frame_indices=[1, 2],
                 timestamps=[0.0, 1.0], frame_set_hash="x", readable_frame_ids=[9])


def test_partial_readability_is_representable():
    """§5.2 的"缺失掩码"载体：可读帧是规划帧的**真子集**时也能表达。"""
    fset = FrameSet(frame_ids=[0, 10, 20, 30], source_frame_indices=[0, 10, 20, 30],
                    timestamps=[0.0, 1.0, 2.0, 3.0], frame_set_hash="h",
                    readable_frame_ids=[0, 20], decode_status="partial")
    assert fset.readable_frame_ids == [0, 20]
    assert fset.decode_status == "partial"


def test_array_lengths_must_match_and_indices_be_unique_ordered():
    base = dict(source_frame_indices=[0, 1], timestamps=[0.0, 1.0], frame_set_hash="x")
    with pytest.raises(ValidationError):
        FrameSet(frame_ids=[0, 1], source_frame_indices=[0], timestamps=[0.0, 1.0],
                 frame_set_hash="x")
    with pytest.raises(ValidationError):
        FrameSet(frame_ids=[0, 1], source_frame_indices=[0, 1], timestamps=[0.0],
                 frame_set_hash="x")
    with pytest.raises(ValidationError):
        FrameSet(frame_ids=[1, 1], **{**base, "source_frame_indices": [1, 1]})
    with pytest.raises(ValidationError):
        FrameSet(frame_ids=[2, 1], **base)


def test_frame_refs_length_must_match_when_given():
    with pytest.raises(ValidationError):
        FrameSet(frame_ids=[0, 1], source_frame_indices=[0, 1], timestamps=[0.0, 1.0],
                 frame_set_hash="x", frame_refs=["only-one"])


def test_cache_identity_must_differ_for_same_indices_different_video():
    """§5.2 原文禁止"仅凭相同的帧索引列表跨视频复用"。"""
    a = fs.build_frame_set(64, n_frames=32, fps=30.0, dataset_id="scannet",
                           video_id="vid-A", scene_name="scene-1")
    b = fs.build_frame_set(64, n_frames=32, fps=30.0, dataset_id="scannet",
                           video_id="vid-B", scene_name="scene-1")
    # 帧索引完全相同 → frame_set_hash 相同（这是它的定义）
    assert a.frame_set_hash == b.frame_set_hash
    # 但缓存身份必须不同
    assert a.cache_identity() != b.cache_identity()


def test_cache_identity_is_stable_for_the_same_source():
    def build() -> str:
        return fs.build_frame_set(64, n_frames=32, fps=30.0, dataset_id="scannet",
                                  video_id="vid-A", scene_name="scene-1").cache_identity()

    assert build() == build()


def test_frame_set_hash_definition_is_unchanged():
    """`frame_set_hash` 仍是"帧索引内容哈希"——缓存身份是**另一个**键，不改写它。"""
    fset = fs.build_frame_set(64, n_frames=32, fps=30.0)
    assert fset.frame_set_hash == fs.frame_set_hash(fset.frame_ids)


# ------------------------------------- §5.2 不足 32 帧：继续作答，不伪称完整 ----

def test_short_video_uses_all_available_frames_without_padding():
    """§5.2："使用可读帧继续作答；不重复填充或偷偷改采样"。"""
    fset = fs.build_frame_set(20, n_frames=32, fps=30.0)
    assert fset.frame_ids == list(range(20))
    assert len(set(fset.frame_ids)) == 20, "不得重复填充凑到 32"
    assert fset.n_total_frames == 20
    assert all(a < b for a, b in zip(fset.timestamps, fset.timestamps[1:]))


def test_short_video_frame_count_is_reported_honestly():
    """不得伪称完整 32 帧输入（§5.2）。"""
    fset = fs.build_frame_set(20, n_frames=32, fps=30.0)
    assert fset.n_frames == 20, "帧数必须如实记录，不得沿用名义 32"
    assert fset.decode_status == "partial"


def test_full_video_is_unchanged_by_the_short_video_policy():
    fset = fs.build_frame_set(64, n_frames=32, fps=30.0)
    assert fset.n_frames == 32 and fset.decode_status == "ok"
    assert len(fset.frame_ids) == 32


def test_zero_frame_video_is_still_an_input_error():
    """只有"一帧都没有"才是输入错误（§5.2 input_error）。"""
    with pytest.raises(fs.FrameSetError):
        fs.build_frame_set(0, n_frames=32, fps=30.0)


def test_short_video_gate_proceeds_with_input_degraded_flag():
    """§5.2 端到端：不足 32 帧 → M2 继续作答并显式降级，不从分母删除。"""
    import numpy as np

    from skill3d.gates.input_gate import input_gate

    fset = fs.build_frame_set(20, n_frames=32, fps=30.0)
    frames = [np.full((16, 16, 3), 128, dtype=np.uint8) for _ in fset.frame_ids]
    verdict = input_gate(frames, frame_set=fset)
    assert verdict.action == "proceed"
    assert "input_degraded" in verdict.degradation_flags
    assert verdict.n_frames == 20


# ------------------------------- §5.2 缓存身份落到 artifact 路径 ----

def test_artifact_path_uses_cache_identity_not_scene_name(tmp_path):
    """§5.2"禁止仅凭相同的帧索引列表跨视频复用"必须落到**落盘路径**上。"""
    from skill3d.reconstruction.run import artifact_path

    a = fs.build_frame_set(64, dataset_id="scannet", video_id="vid-A", scene_name="s1")
    b = fs.build_frame_set(64, dataset_id="scannet", video_id="vid-B", scene_name="s1")
    pa = artifact_path(tmp_path, "s1", "vggt", frame_set=a)
    pb = artifact_path(tmp_path, "s1", "vggt", frame_set=b)
    assert pa != pb, "同名 scene、不同视频不得共用同一缓存文件"
    assert pa.name == f"s1__{a.cache_identity()[:16]}.json"


def test_artifact_path_without_frame_set_keeps_legacy_naming(tmp_path):
    """不传 frame_set 时保持历史命名 —— 供 §17.2 迁移期读取旧产物。"""
    from skill3d.reconstruction.run import artifact_path

    assert artifact_path(tmp_path, "s1", "vggt").name == "s1.json"


def test_resolve_prefers_identity_then_falls_back_to_legacy(tmp_path):
    """§17.2：切缓存键不得让已算好的旧产物凭空失效（那会改变对比基线）。"""
    from skill3d.reconstruction.run import artifact_path, resolve_artifact_path

    fset = fs.build_frame_set(64, dataset_id="scannet", video_id="vid-A",
                              scene_name="s1")
    (tmp_path / "vggt").mkdir()
    legacy = tmp_path / "vggt" / "s1.json"
    legacy.write_text("{}", encoding="utf-8")

    got, used_legacy = resolve_artifact_path(tmp_path, "s1", "vggt", frame_set=fset)
    assert got == legacy and used_legacy is True

    fresh = artifact_path(tmp_path, "s1", "vggt", frame_set=fset)
    fresh.write_text("{}", encoding="utf-8")
    got2, used2 = resolve_artifact_path(tmp_path, "s1", "vggt", frame_set=fset)
    assert got2 == fresh and used2 is False, "身份路径存在时优先用它"


def test_resolve_returns_identity_path_when_nothing_exists(tmp_path):
    from skill3d.reconstruction.run import artifact_path, resolve_artifact_path

    fset = fs.build_frame_set(64, video_id="vid-A")
    (tmp_path / "vggt").mkdir()
    got, used = resolve_artifact_path(tmp_path, "s1", "vggt", frame_set=fset)
    assert got == artifact_path(tmp_path, "s1", "vggt", frame_set=fset)
    assert used is False


def test_runner_and_reconstruction_agree_on_the_artifact_path(tmp_path):
    """写方（run_jobs）与读方（runner 复用）必须对同一路径达成一致。

    此前两边各写了一份路径约定（`artifact_path` 与 runner 的内联 `f"{scene}.json"`），
    是重复定义；切缓存键会让两边失配，因此这里守住"读方委托写方"。
    """
    from skill3d.online.runner import _resolve_existing_artifact
    from skill3d.reconstruction.run import resolve_artifact_path

    fset = fs.build_frame_set(64, video_id="vid-A", scene_name="s1")

    class _Cfg:
        recon_dir = str(tmp_path)
        recon_method = "vggt"

    class _Episode:
        scene_name = "s1"
        frame_set = fset

    expected, expected_legacy = resolve_artifact_path(
        tmp_path, "s1", "vggt", frame_set=fset)
    actual, actual_legacy = _resolve_existing_artifact(_Cfg(), _Episode())
    assert actual == str(expected)
    assert actual_legacy is expected_legacy
