"""English framework instructions, with immutable Skill ancestry and real M5 requests."""

import ast
import json
import re
from pathlib import Path

import numpy as np

from skill3d.segmentation import sam2_tracker as m5
from skill3d.skills.v11_library import validate_v11_snapshot

LIBRARY = Path(__file__).resolve().parents[2] / "skill_library"
CJK = re.compile(r"[\u4e00-\u9fff]")


def test_english_snapshot_preserves_parent_identity_and_executable_examples():
    parent = json.loads(
        (LIBRARY / "snapshots/snapshot_S0-v11-contract-repair.json").read_text())
    current = json.loads(
        (LIBRARY / "snapshots/snapshot_S0-v11-english.json").read_text())
    old = {s.skill_id: s for s in validate_v11_snapshot(parent, library_root=LIBRARY)}
    new = {s.skill_id: s for s in validate_v11_snapshot(current, library_root=LIBRARY)}
    assert len(new) == 8 and new.keys() == old.keys()
    assert current["parent_snapshot_id"] == parent["snapshot_id"]
    assert all(entry["created_by"] == "english_translation"
               for entry in current["entries"].values())
    for sid, spec in new.items():
        assert spec.name == old[sid].name
        assert spec.question_type == old[sid].question_type
        assert not CJK.search(spec.skill_md), sid
        assert len(spec.skill_md) <= 8000
        assert spec.version == ("1.3.0" if sid in ("S01", "S08") else "1.2.0")
        assert current["entries"][f"{sid}@{spec.version}"]["parent_version"] == (
            f"{sid}@{old[sid].version}")
        code = lambda text: [
            ast.dump(ast.parse(block), include_attributes=False)
            for block in re.findall(r"```python\n(.*?)```", text, re.S)]
        assert code(spec.skill_md) == code(old[sid].skill_md), sid


class _M5Recorder:
    def __init__(self):
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return '["chair"]' if isinstance(messages[0]["content"], str) else "[]"


def test_m5_inventory_noun_extraction_and_targeted_boxes_send_english(monkeypatch):
    client = _M5Recorder()
    frames = [np.zeros((24, 32, 3), dtype=np.uint8)]
    m5.box_prompts_from_vlm(frames, m5._GENERIC_INVENTORY_QUESTION, client, seed=7)
    monkeypatch.setattr(m5, "ovd_boxes_for_nouns", lambda *args: ([], [], [], []))
    objects, stats, _ = m5._bind_question_supplement(
        frames, [0], "How many chairs are visible?", [], None, None, None,
        grid_transform=None, out_dir=None, scene_name="test", predictor=None,
        vlm_client=client, handle=None, seed=7, question_type="object_counting")
    assert objects == [] and stats["grounding"]["missing_names"]
    text = json.dumps([messages for messages, _ in client.calls], ensure_ascii=False)
    assert not CJK.search(text)
    assert m5._GENERIC_INVENTORY_QUESTION in text
    assert "common English singular object names" in text
    assert "Locate the following objects" in text
    assert "How many chairs are visible?" in text
    assert all(kwargs["seed"] == 7 for _, kwargs in client.calls)
