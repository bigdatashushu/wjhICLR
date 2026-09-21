"""G-09 四层 split 构建单测（§4 M1 验收条件、硬约束 9/19）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skill3d.adapters.split_builder import (
    SPLIT_KEYS,
    SplitBuildError,
    build_split_config,
    load_local_meta,
    load_split_config,
    scene_inventory,
    validate_split,
    write_contamination_log,
    write_split_yaml,
)
from skill3d.adapters.vsibench_loader import assert_final_test_isolation

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_META = REPO_ROOT / "data" / "vsi_bench_meta" / "test.jsonl"

# 稀缺题型（rel_direction_easy 全库仅 76 scene）用于验证分层有效性
RARE = "object_rel_direction_easy"
TYPES = [RARE, "object_counting", "object_rel_direction_hard", "room_size_estimation"]


def _synth_rows(n_scenes: int = 40, seed: int = 0) -> list[dict]:
    """构造分层明显的合成 meta：每个 scene 覆盖 1~2 个题型，含跨数据集 scene。"""
    rows: list[dict] = []
    qid = 0
    for i in range(n_scenes):
        scene = f"scene{i:03d}"
        ds = ("scannet", "scannetpp", "arkitscenes")[i % 3]
        types = [RARE] if i % 4 == 0 else TYPES[1:]
        for t in types:
            for k in range(1 + (i % 3)):  # 每题多行，验证 QA 侧统计
                rows.append({"id": qid, "dataset": ds, "scene_name": scene,
                             "question_type": t, "question": f"q{qid}",
                             "ground_truth": "1", "options": None})
                qid += 1
    assert seed == 0  # 合成数据本身与 seed 无关
    return rows


# ------------------------------------------------------------------ meta 读取 ----

def test_load_local_meta_jsonl_json_csv(tmp_path):
    rows = _synth_rows(4)
    jl = tmp_path / "m.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert len(load_local_meta(jl)) == len(rows)

    js = tmp_path / "m.json"
    js.write_text(json.dumps(rows), encoding="utf-8")
    assert len(load_local_meta(js)) == len(rows)

    cs = tmp_path / "m.csv"
    cs.write_text("id,dataset,scene_name,question_type,question,ground_truth\n"
                  "0,scannet,s0,object_counting,q,1\n", encoding="utf-8")
    assert len(load_local_meta(cs)) == 1


def test_load_local_meta_missing_raises(tmp_path):
    with pytest.raises(SplitBuildError):
        load_local_meta(tmp_path / "nope.jsonl")


def test_scene_inventory_collision_across_datasets_raises():
    rows = [{"id": 0, "dataset": "scannet", "scene_name": "dup",
             "question_type": "object_counting", "question": "q", "ground_truth": "1"},
            {"id": 1, "dataset": "arkitscenes", "scene_name": "dup",
             "question_type": "object_counting", "question": "q", "ground_truth": "1"}]
    with pytest.raises(SplitBuildError, match="冲突"):
        scene_inventory(rows)


# ------------------------------------------------------------------ 切分正确性 ----

def test_split_disjoint_complete_and_stratified():
    rows = _synth_rows(40)
    cfg, report = build_split_config(rows, seed=0, final_ratio=0.2)
    lists = {k: getattr(cfg, v) for k, v in SPLIT_KEYS.items()}
    # 互斥
    for i, a in enumerate(lists):
        for b in list(lists)[i + 1:]:
            assert not (set(lists[a]) & set(lists[b])), f"{a} ∩ {b}"
    # 完整
    assert sum(len(v) for v in lists.values()) == report.n_scenes == 40
    # final 隔离（硬约束 9）
    assert_final_test_isolation(cfg)
    # 分层：稀缺题型在四层都有代表
    assert all(report.per_task_scenes[RARE][k] > 0 for k in SPLIT_KEYS)
    # 比例近似 5:3:2（非 final pool）
    non_final = {k: report.per_split_scenes[k]
                 for k in ("induction", "inner_validation", "outer_holdout")}
    total = sum(non_final.values())
    assert abs(non_final["induction"] / total - 0.5) < 0.1
    assert abs(non_final["inner_validation"] / total - 0.3) < 0.1
    assert abs(non_final["outer_holdout"] / total - 0.2) < 0.1


def test_split_is_deterministic_by_seed():
    rows = _synth_rows(40)
    a, ra = build_split_config(rows, seed=7, final_ratio=0.2)
    b, rb = build_split_config(rows, seed=7, final_ratio=0.2)
    assert a.model_dump() == b.model_dump()
    assert a.split_version == b.split_version
    c, _ = build_split_config(rows, seed=8, final_ratio=0.2)
    assert c.model_dump() != a.model_dump()


def test_explicit_final_scenes_respected_and_validated():
    rows = _synth_rows(20)
    cfg, _ = build_split_config(rows, final_scene_ids=["scene000", "scene007"])
    assert cfg.final_test_scene_ids == ["scene000", "scene007"]
    with pytest.raises(SplitBuildError, match="不在 meta"):
        build_split_config(rows, final_scene_ids=["ghost"])


def test_zero_final_ratio_leaves_final_empty():
    rows = _synth_rows(20)
    cfg, report = build_split_config(rows, final_ratio=0.0)
    assert cfg.final_test_scene_ids == []
    assert report.per_split_scenes["induction"] > 0
    assert sum(report.per_split_scenes.values()) == 20


def test_bad_bucket_fractions_rejected():
    rows = _synth_rows(10)
    with pytest.raises(SplitBuildError, match="份额之和"):
        build_split_config(rows, ratios={"induction": 0.9, "inner_validation": 0.3,
                                         "outer_holdout": 0.2})


def test_validate_split_detects_overlap():
    rows = _synth_rows(12)
    cfg, report = build_split_config(rows, final_ratio=0.0)
    bad = cfg.model_copy(update={
        "inner_validation_scene_ids": cfg.induction_scene_ids[:2]})
    with pytest.raises(SplitBuildError, match="相交"):
        validate_split(bad, scene_inventory(rows), report)


def test_validate_split_detects_incomplete_coverage():
    rows = _synth_rows(12)
    cfg, report = build_split_config(rows, final_ratio=0.0)
    bad = cfg.model_copy(update={"induction_scene_ids": [], "final_test_scene_ids": []})
    with pytest.raises(SplitBuildError, match="覆盖不完整|总数"):
        validate_split(bad, scene_inventory(rows), report)


# ------------------------------------------------------------------ 落盘 / 回读 ----

def test_write_and_reload_roundtrip(tmp_path):
    rows = _synth_rows(24)
    out = tmp_path / "vsi_bench_split.yaml"
    log = tmp_path / "contamination_check.log"
    cfg, report = build_split_config(rows, seed=3, final_ratio=0.25,
                                     meta_source="mem", log_ref=str(log))
    write_split_yaml(cfg, report, out)
    write_contamination_log(cfg, report, log, command="pytest")

    back = load_split_config(out)
    assert back.induction_scene_ids == cfg.induction_scene_ids
    assert back.final_test_scene_ids == cfg.final_test_scene_ids
    assert back.split_version == cfg.split_version

    text = log.read_text(encoding="utf-8")
    assert "PASS" in text and "final_test 隔离" in text
    assert RARE in text  # 分层证据表
    assert "TOTAL" in text


# ------------------------------------------------------------------ 真实 meta ----

@pytest.mark.skipif(not REAL_META.is_file(), reason="真实 VSI-Bench meta 未下载")
def test_real_vsi_bench_meta_matches_documented_scale():
    """§1.2 已核验规模：5130 QA / 288 scene / 10 个 question_type 取值。"""
    rows = load_local_meta(REAL_META)
    cfg, report = build_split_config(rows, seed=0, final_ratio=0.2,
                                     meta_source=str(REAL_META))
    assert report.n_rows == 5130
    assert report.n_scenes == 288
    assert report.dataset_scene_counts == {"scannet": 88, "scannetpp": 50,
                                           "arkitscenes": 150}
    assert len(report.per_task_scenes) == 10
    # 四列表互斥 + 完整（§4 M1 验收条件 c）
    assert sum(report.per_split_scenes.values()) == 288
    assert_final_test_isolation(cfg)
    # 每个题型在四层均有代表（分层有效）
    for t, per in report.per_task_scenes.items():
        assert all(per[k] > 0 for k in SPLIT_KEYS), f"{t} 分层缺层: {per}"
