"""English framework instructions, with immutable Skill ancestry and real M5 requests."""

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from skill3d.segmentation import question_binding as m5
from skill3d.infra.m5_identity import generation_identity
from skill3d.skills.v11_library import validate_v11_snapshot

LIBRARY = Path(__file__).resolve().parents[2] / "skill_library"
CJK = re.compile(r"[\u4e00-\u9fff]")


def test_english_snapshot_preserves_parent_identity_and_executable_examples():
    parent = json.loads(
        (LIBRARY / "snapshots/snapshot_S0-v11-contract-repair.json").read_text())
    current = json.loads(
        (LIBRARY / "snapshots/snapshot_S0-v11-english.json").read_text())
    from types import SimpleNamespace
    from skill3d.schemas.skill import parse_skill_markdown
    def archived(snapshot):
        values = {}
        for entry in snapshot["entries"].values():
            text = (LIBRARY / entry["source_ref"]).read_text()
            assert hashlib.sha256(text.encode()).hexdigest() == entry["content_sha256"]
            values[entry["skill_id"]] = SimpleNamespace(
                **entry, skill_md=text, name=parse_skill_markdown(text)[0]["name"])
        return values
    old = archived(parent)
    new = archived(current)
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
        return '{"objects":[{"category":"chair","role":"target","source_text":"chairs"}]}'


def test_m5_question_object_parse_sends_english_text_only():
    client = _M5Recorder()
    inputs = m5.question_input("How many chairs are visible?", "object_counting", [])
    identity = generation_identity(
        scene_name="s", frame_set_hash="f", frames=[], depth_maps=None)
    objects, receipt = m5.parse_question_objects(inputs, client, seed=7, identity=identity)
    assert objects[0]["category"] == "chair" and receipt["status"] == "parsed"
    text = json.dumps([messages for messages, _ in client.calls], ensure_ascii=False)
    assert not CJK.search(text)
    assert "common English" in text
    assert "How many chairs are visible?" in text
    assert all(kwargs["seed"] == 7 for _, kwargs in client.calls)
    assert all(kwargs["temperature"] == 0 for _, kwargs in client.calls)
    assert all(isinstance(m["content"], str) for messages, _ in client.calls for m in messages)
