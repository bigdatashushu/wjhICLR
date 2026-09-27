"""v9 D1：§5.2 可重试加载 = **换来源/换副本**（用户 2026-09-27 裁定）。

规范原文（§5.2）：

> 全部指定图像缺失或无法解码：`input_error`，记录原因；**可重试加载**，不生成伪答案。

**决策记录**：这条"可重试加载"没写清"重试"是什么。对**同一个文件**反复解码只会得到
同一结果，等于没重试，因此 2026-09-27 请用户裁决，用户裁定：**换来源/换副本**
（另一个根目录、镜像目录或同名不同容器的副本）。

守四件事：

1. 主来源正常时行为与改动前**逐字一致**（候选顺序不影响正常路径，历史缓存键不变）；
2. 主来源缺失/不可解码时按序换副本，**用了副本要留痕**（抽样收据 + `video_id` 标签）；
3. 副本可能是另一种编码 → `video_id` 必须带来源标签，禁止与主来源共用重建缓存；
4. 所有来源都失败才记 `input_error` 排除行，原因按"走得最远"的失败给出。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skill3d.adapters import vsibench_loader as vl
from skill3d.adapters.episode_source import (
    _load_from_sources,
    _record_source_attempts,
    _source_failure_reason,
    load_vsi_bench_items,
)


def _write_video(path: Path, n_frames: int = 64) -> str:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(n_frames):
        vw.write(np.full((48, 64, 3), i % 255, dtype=np.uint8))
    vw.release()
    return str(path)


# 本地 meta 行（测试用；`load_meta` 被 monkeypatch 替换，不读真实 5130 行清单）
_META_ROW = {"id": "q1", "qa_id": "q1", "scene_name": "scene01", "dataset": "scannet",
             "question_type": "object_counting", "question": "how many?",
             "ground_truth": "1", "options": []}


def _row(scene="scene01", dataset="scannet") -> dict:
    return {"qa_id": "q1", "scene_name": scene, "dataset": dataset,
            "question_type": "object_counting"}


# --------------------------------------------- 候选顺序（主来源优先，不隐式启用镜像）----

def test_candidates_put_the_primary_convention_first(tmp_path):
    """主约定永远是第一候选；剩余候选只在主来源失败时才会被用到。"""
    cands = vl.video_source_candidates(_row(), tmp_path)
    assert cands[0].path == tmp_path / "scannet" / "scene01.mp4"
    assert cands[0].tag == "" and cands[0].convention == "primary"
    # 同目录同名不同容器（副本）排在主约定之后
    assert {c.convention for c in cands[1:]} == {"alt_container"}
    assert not any(c.convention.startswith("mirror") for c in cands), \
        "未配置备用根时不得隐式启用任何镜像"


def test_candidates_include_mirror_conventions_when_configured(tmp_path):
    """配置了备用根才出现镜像候选：目录式 + 平铺式（如 VSI_videos 的 `<ds>_<scene>.mp4`）。"""
    fb = tmp_path / "VSI_videos"
    cands = vl.video_source_candidates(_row(dataset="scannetpp", scene="09c1"),
                                       tmp_path / "VSI-Bench", [fb])
    paths = [c.path for c in cands]
    assert fb / "scannetpp" / "09c1.mp4" in paths          # 目录式镜像
    assert fb / "scannetpp_09c1.mp4" in paths              # 平铺式镜像
    flat = next(c for c in cands if c.convention == "mirror_flat")
    assert flat.tag == "VSI_videos"
    # 顺序：主约定 → 同目录副本 → 备用根
    assert paths[0] == tmp_path / "VSI-Bench" / "scannetpp" / "09c1.mp4"
    assert paths.index(fb / "scannetpp" / "09c1.mp4") > 0


def test_video_id_keeps_history_but_marks_copies(tmp_path):
    """§5.2：源标识参与缓存身份 —— 副本带标签，主来源保持原样（历史缓存不失效）。"""
    row = _row()
    primary = vl.video_source_candidates(row, tmp_path)[0]
    assert vl.video_id_for(row, primary) == "scene01"
    fb = tmp_path / "mirror"
    copy = next(c for c in vl.video_source_candidates(row, tmp_path, [fb])
                if c.convention == "mirror_flat")
    assert vl.video_id_for(row, copy) == "scene01@mirror"
    # 显式给了 video_id 时以它为准（历史行为不变）
    assert vl.video_id_for({**row, "video_id": "vid-9"}, copy) == "vid-9@mirror"


# --------------------------------------------- 实际加载：换副本 ----

def test_primary_missing_falls_back_to_flat_mirror(tmp_path):
    """主来源缺失 → 用平铺镜像副本继续，并**留痕**（收据 + video_id 标签）。"""
    root = tmp_path / "VSI-Bench"
    fb = tmp_path / "VSI_videos"
    _write_video(fb / "scannet_scene01.mp4")
    loaded, attempts = _load_from_sources(_row(), root, [fb], 32)
    assert loaded is not None, attempts
    assert loaded["source"].convention == "mirror_flat"
    assert Path(loaded["source"].path) == fb / "scannet_scene01.mp4"
    # 逐个候选记结果：前面全部 missing，最后一个是真正用上的那份
    assert attempts[-1]["result"] == "used"
    assert all(a["result"] == "missing" for a in attempts[:-1])
    assert len(loaded["pixels"]) == 32
    assert vl.video_id_for(_row(), loaded["source"]) == "scene01@VSI_videos"


def test_same_directory_copy_is_used_when_primary_is_undecodable(tmp_path):
    """主来源解不出帧（0 帧头）→ 换同目录副本；原因码按"走得最远"的失败给出。"""
    root = tmp_path / "VSI-Bench"
    # 写一个 0 字节的"视频"：文件存在、但打不开/没有帧
    (root / "scannet").mkdir(parents=True)
    (root / "scannet" / "scene01.mp4").write_bytes(b"")
    _write_video(root / "scannet" / "scene01.mkv")
    loaded, attempts = _load_from_sources(_row(), root, [], 32)
    assert loaded is not None, attempts
    assert loaded["source"].convention == "alt_container"
    assert Path(loaded["source"].path).suffix == ".mkv"
    assert "undecodable" in [a["result"] for a in attempts]


def test_all_sources_failing_still_reports_an_exclusion_reason():
    """所有来源都失败 → 调用方记 `input_error` 排除行（§5.3 分母保留，不生成伪答案）。"""
    only_missing = [{"result": "missing"}, {"result": "missing"}]
    assert _source_failure_reason(only_missing) == "video_missing"
    assert _source_failure_reason([{"result": "missing"},
                                   {"result": "undecodable"}]) == "video_undecodable"
    assert _source_failure_reason([{"result": "no_frames"}]) == "insufficient_frames"


def test_loader_records_source_retry_in_the_sampling_receipt(tmp_path):
    """用了副本必须进抽样收据（`source_retried_samples` 的来源），主来源成功则不写。"""
    receipt: dict = {}
    row = _row()
    _record_source_attempts(receipt, row, [{"source": "/a.mp4", "result": "missing"},
                                           {"source": "/b.mp4", "result": "used"}],
                            used="/b.mp4")
    assert receipt["source_retries"][0]["retried"] is True
    assert receipt["source_retries"][0]["used_source"] == "/b.mp4"

    clean: dict = {}
    _record_source_attempts(clean, row, [{"source": "/a.mp4", "result": "used"}],
                            used="/a.mp4")
    assert not clean.get("source_retries"), "主来源一次成功不应写任何收据条目"


def test_load_vsi_bench_items_uses_the_fallback_root(tmp_path, monkeypatch):
    """端到端：主目录没有视频，镜像里有 → 条目照常产出，且 video_id 带来源标签。"""
    monkeypatch.setattr(vl, "load_meta", lambda **kwargs: [_META_ROW])
    root = tmp_path / "VSI-Bench"
    fb = tmp_path / "VSI_videos"
    _write_video(fb / "scannet_scene01.mp4")
    receipt: dict = {}
    exclusions: list = []
    items = load_vsi_bench_items(
        "inner_validation", {"inner_validation_scene_ids": ["scene01"]},
        video_root=root, video_fallback_roots=[fb],
        n_frames=32, exclusions=exclusions, sampling_receipt=receipt)
    assert exclusions == []
    assert len(items) == 1
    assert items[0].video_path == str(fb / "scannet_scene01.mp4")
    assert items[0].episode.frame_set.video_id == "scene01@VSI_videos"
    assert receipt["source_retries"][0]["retried"] is True


def test_load_vsi_bench_items_excludes_when_every_source_is_gone(tmp_path, monkeypatch):
    """全部来源缺失 → 排除行带原因，分母保留（§5.3），且不产出伪条目。"""
    monkeypatch.setattr(vl, "load_meta", lambda **kwargs: [_META_ROW])
    exclusions: list = []
    receipt: dict = {}
    with pytest.raises(Exception):
        load_vsi_bench_items("inner_validation",
                             {"inner_validation_scene_ids": ["scene01"]},
                             video_root=tmp_path / "empty",
                             video_fallback_roots=[],
                             n_frames=32,
                             exclusions=exclusions, sampling_receipt=receipt)
    assert [e["reason"] for e in exclusions] == ["video_missing"]
    assert receipt["source_retries"][0]["retried"] is False
