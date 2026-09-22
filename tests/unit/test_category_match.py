"""类别名匹配测试（`tools/category_match.py`）。

真实依据（2026-09-21，32 题 inner_validation 逐题归因，scene 7b6477cb95）：
M5 清单里是 `phone` / `trash_can`，题面写 `telephone` / `trash can`，而当时的
匹配是朴素子串比较 —— `'trash can' in 'trash_can'` 与 `'telephone' in 'phone'`
**都返回 False** → `list_objects(...)` 返回空 → 程序走
`if not objs: ReturnAnswer("abstain")` → 4 个 `object_rel_direction` 题全 abstain。
这不是几何或证据问题，纯粹是字符串没归一。

本文件锁定三类差异（分隔符 / 同义词 / 修饰词与单复数）都必须命中，同时锁定
"空查询不得匹配一切"这条 fail-closed 边界。
"""

from __future__ import annotations

import pytest

from skill3d.tools.category_match import canonical, matches, normalize, tokens


@pytest.mark.parametrize("query,category", [
    # 分隔符差异
    ("trash can", "trash_can"),
    ("trash can", "trash-can"),
    ("coffee table", "coffee_table"),
    ("cutting board", "cutting-board"),
    # 大小写
    ("Telephone", "phone"),
    ("CHAIR", "chair"),
    # 同义词
    ("telephone", "phone"),
    ("sofa", "couch"),
    ("tv", "television"),
    ("fridge", "refrigerator"),
    ("rug", "carpet"),
    ("bookshelf", "bookcase"),
    ("closet", "wardrobe"),
    ("mug", "cup"),
    ("ceiling light", "ceiling lamp"),
    ("air conditioner", "air_conditioner"),
    # 单复数
    ("books", "book"),
    ("boxes", "box"),
    ("chairs", "chair"),
    # 修饰词 / 词序
    ("blue chair", "chair"),
    ("chair", "blue chair"),
    ("black desk chair", "chair"),
    ("computer mouse", "mouse"),
    ("desk counter", "desk"),
    # 精确
    ("chair", "chair"),
])
def test_true_matches(query, category):
    assert matches(query, category), f"{query!r} 应命中 {category!r}"


@pytest.mark.parametrize("query,category", [
    ("", "chair"),          # 空查询不得匹配一切（fail-closed）
    ("chair", ""),          # 空类别名不得被命中
    ("", ""),
    ("chair", "table"),
    ("microwave", "monitor"),
    ("pan", "pot"),         # 同义词表里没有的语义相近词不算等价
])
def test_false_matches(query, category):
    assert not matches(query, category), f"{query!r} 不应命中 {category!r}"


def test_normalize_folds_separators_and_case():
    assert normalize("  Trash__Can  ") == "trash can"
    assert normalize("COFFEE-TABLE") == "coffee table"
    assert normalize("Desk/Counter") == "desk counter"
    assert normalize(None) == ""


def test_canonical_applies_synonyms_and_plural():
    assert canonical("telephone") == "phone"
    assert canonical("Sofas") == "couch"
    assert canonical("books") == "book"
    # 同义词先折叠再单复数：'telephones' → 'phone'
    assert canonical("telephones") == "phone"


def test_tokens_are_word_sets():
    assert tokens("blue chair") == {"blue", "chair"}
    assert tokens("trash_can") == {"trash", "can"}


def test_scene_handle_uses_normalized_matching():
    """端到端：SceneHandle.list_objects_by_name 与 resolve_object_id 都走归一化匹配。"""
    from skill3d.schemas import ObjectRecord, SceneState
    from skill3d.tools.contract import available_artifacts_for
    from skill3d.tools.scene_handle import SceneHandle

    objs = [
        ObjectRecord(obj_id="obj_0", category_name="phone", mask_per_frame="",
                     pointcloud_world="", centroid_world=[0, 0, 0], bbox=[0] * 6,
                     det_conf=0.9),
        ObjectRecord(obj_id="obj_1", category_name="trash_can", mask_per_frame="",
                     pointcloud_world="", centroid_world=[1, 0, 0], bbox=[0] * 6,
                     det_conf=0.8),
        ObjectRecord(obj_id="obj_2", category_name="blue chair", mask_per_frame="",
                     pointcloud_world="", centroid_world=[2, 0, 0], bbox=[0] * 6,
                     det_conf=0.7),
    ]
    scene = SceneState(artifact_ref="a", scene_route="full_3d",
                       question_tool_scope="full_3d",
                       available_artifacts=available_artifacts_for("full_3d"))
    h = SceneHandle(scene, objects=objs, objects_materialized=True)

    # 题面用词 → 清单类别名（这三条正是实测里失败的三个）
    assert h.list_objects_by_name("telephone") == ["obj_0"]
    assert h.list_objects_by_name("trash can") == ["obj_1"]
    assert h.list_objects_by_name("black desk chair") == ["obj_2"]
    assert h.list_objects_by_name("") == ["obj_0", "obj_1", "obj_2"]
    assert h.list_objects_by_name("microwave") == []

    # 解析（方向/距离题拿 obj_id 用）
    assert h.resolve_object_id("telephone") == "obj_0"
    assert h.resolve_object_id("trash can") == "obj_1"
    assert h.resolve_object_id("obj_2") == "obj_2"
    with pytest.raises(KeyError):
        h.resolve_object_id("microwave")
