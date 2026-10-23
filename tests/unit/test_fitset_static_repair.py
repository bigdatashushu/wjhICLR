"""A bounded fit-set formatting repair receives the actual prior candidate."""

import json
from pathlib import Path

from skill3d.evolution.fitset_revision_v11 import FitSetSlotReviser
from skill3d.skills.registry import load_active_skills


def test_static_repair_returns_failed_candidate_and_locations_without_new_diagnosis(tmp_path):
    library = Path(__file__).resolve().parents[2] / "skill_library"
    parent = next(s for s in load_active_skills(library / "snapshots")[0]
                  if s.slot_key == "object_counting/g0m0")
    failed = parent.skill_md + "\n```python\nvalue = tools.unregistered_geometry_tool()\n```\n"

    class Client:
        def __init__(self):
            self.messages = []

        def chat(self, messages, **_kwargs):
            self.messages.append(messages)
            if len(self.messages) == 1:
                return json.dumps({
                    "action": "revise", "full_skill_md": failed,
                    "modification_reason": "Try an unsupported helper.",
                    "change_hypothesis": "A helper may simplify the procedure.",
                    "changed_sections": ["__body__"],
                    "expected_process_effect": "Reduce procedural mistakes.",
                    "evidence_signature_refs": [],
                })
            prompt = messages[-1]["content"]
            repair = json.loads(prompt.split(
                "Previous candidate repair context (full safe text, hash and line positions):\n")[1])
            assert repair["candidate_text"] == failed
            assert repair["fully_delivered"]
            assert any(e["code"] == "unknown_tool_reference" for e in repair["errors"])
            return json.dumps({"action": "no_change", "modification_reason": "No supported replacement."})

    client = Client()
    reviser = FitSetSlotReviser(
        output_root=tmp_path, panel={}, panel_name="test", variants=[],
        accumulation_root=tmp_path, label_metadata={}, library_root=library,
        client=client, max_attempts=2,
    )
    kwargs = {
        "slot": ("object_counting", "g0m0"), "parent": parent,
        "feedback": {"signatures": []}, "forbidden_ids": set(),
    }
    assert reviser._revise_slot(**kwargs) is None
    assert reviser._revise_slot(**kwargs) is None
    assert len(client.messages) == 2
    assert len(list(tmp_path.glob("slots/*/*/request_intent_*.json"))) == 2
