"""Critical anchored-edit, reservation and fail-closed revision paths."""
import json
from pathlib import Path

import pytest

from skill3d.evolution.revision_edits_v11 import (
    PROTOCOL, AnchoredSlotReviser, EditProtocolError, RevisionBudget,
    apply_edits, digest, section_menu, tail_decision,
)
from skill3d.evolution.fitset_revision_v11 import RevisionOperationalError
from skill3d.skills.registry import load_active_skills


def proposal(text, before, after, section_id=0):
    return {"protocol": PROTOCOL, "base_sha256": digest(text), "action": "revise",
            "modification_reason": "Revise the source rule.", "change_hypothesis": "A clearer rule helps.",
            "expected_process_effect": "Follow the specified rule.",
            "edits": [{"section_id": section_id, "before": before, "after": after}]}


def test_fenced_headings_duplicate_titles_and_whole_body_selector():
    text = "# Same\nalpha\n```python\n# comment\n```\n# Same\nbeta\n"
    menu = section_menu(text)
    assert [row["heading"] for row in menu] == ["__body__", "Same", "Same"]
    candidate, receipt = apply_edits(text, proposal(text, "beta", "gamma", 2))
    assert candidate.endswith("gamma\n") and receipt[0]["section_id"] == 2
    assert apply_edits(text, proposal(text, "alpha", "omega"))[0].startswith("# Same\nomega")


@pytest.mark.parametrize("code,mutate", [
    ("declared_change_not_present", lambda p: p["edits"][0].update(after="alpha")),
    ("edit_anchor_not_unique", lambda p: p["edits"][0].update(before="missing")),
    ("edit_base_identity_mismatch", lambda p: p.update(base_sha256="0" * 64)),
    ("unknown_section_id", lambda p: p["edits"][0].update(section_id=10)),
])
def test_claims_without_valid_source_changes_are_rejected(code, mutate):
    p = proposal("# Rule\nalpha\n", "alpha", "beta")
    mutate(p)
    with pytest.raises(EditProtocolError, match=code):
        apply_edits("# Rule\nalpha\n", p)


def test_overlapping_source_edits_are_rejected():
    text = "# Rule\nalpha beta\n"
    p = proposal(text, "alpha beta", "other")
    p["edits"].append({"section_id": 0, "before": "beta", "after": "gamma"})
    with pytest.raises(EditProtocolError, match="overlapping_edits"):
        apply_edits(text, p)


def test_complete_evidence_reserves_repairs_and_tail_requires_drain_and_budget():
    budget = RevisionBudget(max_requests=8, repair_reserve=2)
    assert budget.plan(6)["admitted"]
    assert not budget.plan(7)["admitted"]
    kwargs = dict(n_cases=4, attempted_ever=False, stream_complete=True,
                  enabled=True, remaining_requests=4, planned_batches=2)
    assert tail_decision(**kwargs)["scheduled"]
    assert not tail_decision(**{**kwargs, "stream_complete": False})["scheduled"]
    assert not tail_decision(**{**kwargs, "remaining_requests": 3})["scheduled"]
    assert not tail_decision(**{**kwargs, "attempted_ever": True})["scheduled"]


class Client:
    transport_retries = 0
    last_usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    last_response_identity = {"model": "test-model"}
    last_request_identity = {}
    last_generation = {}

    def __init__(self, respond):
        self.respond = respond
        self.calls = 0

    def build_payload(self, messages, **kwargs):
        return {"messages": messages, "model": "test-model", **kwargs}

    def audit_identity(self, **kwargs):
        return {"model": "test-model", **kwargs}

    def chat(self, messages, **kwargs):
        self.calls += 1
        return self.respond(messages)


def parent():
    library = Path(__file__).resolve().parents[2] / "skill_library"
    return next(s for s in load_active_skills(library / "snapshots")[0]
                if s.slot_key == "object_counting/g0m0")


def test_unique_edit_produces_full_static_candidate_and_content_free_request_receipts(tmp_path):
    p = parent()
    before = p.skill_md.splitlines()[-1]
    response = proposal(p.skill_md, before, before + " Preserve supported observations.")
    client = Client(lambda _: json.dumps(response))
    reviser = AnchoredSlotReviser(output_root=tmp_path, client=client,
               budget=RevisionBudget(), method_context_max_chars=8000, expected_model="test-model")
    candidate = reviser.run(parent=p, feedback={}, batches=["authorized learning facts"],
                            delivery_plan={}, forbidden_ids=set(), cases=[])
    assert candidate and candidate.skill_md != p.skill_md
    final = json.loads((tmp_path / "final.json").read_text())
    assert final["status"] == "static_valid" and final["all_evidence_delivered"]
    request = json.loads((tmp_path / "request_01.json").read_text())
    intent = json.loads((tmp_path / "request_intent_01.json").read_text())
    assert "messages" not in intent and "response_text" not in request
    assert client.calls == 1


def test_repeated_unapplied_edit_stops_before_spending_all_repairs(tmp_path):
    p = parent()
    client = Client(lambda _: json.dumps(proposal(p.skill_md, "nonexistent source", "new rule")))
    reviser = AnchoredSlotReviser(output_root=tmp_path, client=client,
               budget=RevisionBudget(), method_context_max_chars=8000, expected_model="test-model")
    assert reviser.run(parent=p, feedback={}, batches=["facts"], delivery_plan={},
                       forbidden_ids=set(), cases=[]) is None
    assert client.calls == 2
    assert json.loads((tmp_path / "final.json").read_text())["status"] == "static_rejected"


def test_response_identity_failure_is_operational_and_never_repaired(tmp_path):
    p = parent()
    client = Client(lambda _: "{}")
    client.last_response_identity = {"model": "unexpected"}
    reviser = AnchoredSlotReviser(output_root=tmp_path, client=client,
               budget=RevisionBudget(), method_context_max_chars=8000, expected_model="test-model")
    with pytest.raises(RevisionOperationalError, match="response model"):
        reviser.run(parent=p, feedback={}, batches=["facts"], delivery_plan={},
                    forbidden_ids=set(), cases=[])
    assert client.calls == 1 and not (tmp_path / "final.json").exists()
