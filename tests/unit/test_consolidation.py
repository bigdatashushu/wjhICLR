"""M14 巩固/泄漏/近重复测试（硬约束 19）。"""

import pytest

from skill3d.memory.consolidation import (
    LeakageCheckError,
    consolidate,
    is_near_duplicate,
    leakage_check_entry,
)
from skill3d.memory.working_episodic_semantic import (
    ProvenanceRequiredError,
    ThreeLayerMemory,
)
from skill3d.schemas import MemoryEntry


def _entry(mid: str, content: str, layer: str = "episodic",
           provenance: list[str] | None = None) -> MemoryEntry:
    return MemoryEntry(
        memory_id=mid, layer=layer, content=content,  # type: ignore[arg-type]
        provenance=provenance if provenance is not None else [f"trace-{mid}"],
        evidence_strength=0.5, contradiction_group_id=None,
        created_at="2026-09-17T00:00:00Z")


def test_leakage_check_rejects_answer_and_sample_id():
    with pytest.raises(LeakageCheckError):
        leakage_check_entry(_entry("m1", "记住：该 scene answer: B"))
    with pytest.raises(LeakageCheckError):
        leakage_check_entry(_entry("m2", "qa_ab12cd34ef 的做法是…"))
    with pytest.raises(LeakageCheckError):
        leakage_check_entry(_entry("m3", "相对距离用相机系 z 轴"),
                            forbidden_sample_ids={"trace-m3"})
    # 干净条目通过
    leakage_check_entry(_entry("m4", "相对方向题先用相机系再转世界系"))


def test_near_duplicate_detection():
    base = "相对方向题先把目标点变换到相机坐标系再判断左右"
    dup = "相对方向题先把目标点变换到相机坐标系再判断左右。"
    other = "计数题需要遍历所有实例 mask 并去重"
    assert is_near_duplicate(dup, [base]) is True
    assert is_near_duplicate(other, [base]) is False


def test_consolidate_merges_near_duplicates():
    by_scene = {
        "scene1": [_entry("a", "相对方向题先变换到相机系再判断左右")],
        "scene2": [_entry("b", "相对方向题先变换到相机系再判断左右。")],  # 近重复
        "scene3": [_entry("c", "相对方向题先变换到相机系再判断左右")],
    }
    out = consolidate(by_scene, existing_semantic=[], n_min=3)
    assert len(out) == 1  # 三个近重复合并为一条 semantic
    assert out[0].layer == "semantic"
    assert set(out[0].provenance) == {"trace-a", "trace-b", "trace-c"}


def test_consolidate_requires_cross_scene_n_min():
    by_scene = {"scene1": [_entry("a", "x")], "scene2": [_entry("b", "y")]}
    assert consolidate(by_scene, existing_semantic=[], n_min=3) == []


def test_consolidate_leakage_rejected():
    by_scene = {f"scene{i}": [_entry(f"m{i}", "该题 answer: C")] for i in range(3)}
    with pytest.raises(LeakageCheckError):
        consolidate(by_scene, existing_semantic=[], n_min=3)


def test_three_layer_memory_provenance_and_discard():
    mem = ThreeLayerMemory()
    with pytest.raises(ProvenanceRequiredError):
        mem.add(_entry("w0", "无来源", layer="working", provenance=[]))
    mem.add(_entry("w1", "临时", layer="working"))
    mem.add(_entry("e1", "场景经验", layer="episodic"), scene_id="scene1")
    assert len(mem.list_layer("working")) == 1
    assert len(mem.list_episodic_by_scene("scene1")) == 1
    mem.end_episode()  # working 即弃
    assert mem.list_layer("working") == []
    assert mem.get("e1") is not None
