import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("question_pool_cli", SCRIPTS / "prepare_question_pool_v11.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_scene_pool_preserves_split_and_selects_one_per_canonical_task_without_labels():
    split = {"induction_scene_ids": ["a", "b"], "inner_validation_scene_ids": ["c"],
             "outer_holdout_scene_ids": ["d"], "final_test_scene_ids": ["e"]}
    rows = [
        {"id": str(i), "scene_name": scene, "dataset": "scannet", "question_type": task,
         "ground_truth": "unused"}
        for i, (scene, task) in enumerate([
            ("a", "object_rel_direction_easy"), ("a", "object_rel_direction_hard"),
            ("a", "object_counting"), ("b", "object_counting"), ("c", "room_size_estimation"),
            ("d", "object_counting"), ("e", "object_counting")])]
    result = cli.select_scene_rows(rows, split, ["induction", "inner_validation"],
                                   preferred_scenes=["c"])
    assert [r["scene_id"] for r in result] == ["c", "a", "b"]
    assert [r["split"] for r in result] == ["inner_validation", "induction", "induction"]
    assert len(result[1]["rows"]) == 2
    assert {cli.canonical_task(r["question_type"]) for r in result[1]["rows"]} == {
        "object_rel_direction", "object_counting"}
    for row in rows:
        row["ground_truth"] = "changed"
    assert [[r["id"] for r in group["rows"]] for group in result] == [
        [r["id"] for r in group["rows"]]
        for group in cli.select_scene_rows(rows, split, ["induction", "inner_validation"],
                                           preferred_scenes=["c"])]


def test_overlapping_splits_fail_before_precomputation():
    split = {"induction_scene_ids": ["a"], "inner_validation_scene_ids": ["a"],
             "outer_holdout_scene_ids": [], "final_test_scene_ids": []}
    with pytest.raises(ValueError, match="overlap"):
        cli.select_scene_rows([], split, ["induction"])
