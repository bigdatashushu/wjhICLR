"""Critical anchored-edit, reservation and fail-closed revision paths."""
import json
from pathlib import Path

import pytest

from skill3d.evolution.revision_edits_v11 import (
    PROTOCOL, AnchoredSlotReviser, EditProtocolError, RevisionBudget,
    apply_edits, digest, section_menu, source_anchors, tail_decision,
)
from skill3d.evolution.fitset_revision_v11 import RevisionOperationalError
from skill3d.skills.registry import load_active_skills
from skill3d.synthesis.request_audit import describe_request


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
    ("edit_anchor_not_found", lambda p: p["edits"][0].update(before="missing")),
    ("invalid_edit_protocol", lambda p: p.update(protocol={})),
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


def test_catalog_selector_targets_one_duplicate_sentence_and_preserves_other_bytes():
    text = "---\nname: sample\n---\n# Method\nKeep evidence. Keep evidence.\n"
    anchors = source_anchors(text)
    assert len(anchors) == 2 and anchors[0]["source"] == anchors[1]["source"]
    selected = anchors[1]
    p = proposal(text, "unused", "Inspect the original frames.")
    p["edits"] = [{"section_id": selected["section_id"], "anchor_id": selected["anchor_id"],
                  "after": "Inspect the original frames."}]
    candidate, audit = apply_edits(text, p)
    assert candidate == text[:selected["start"]] + p["edits"][0]["after"] + text[selected["end"]:]
    assert candidate.count("Keep evidence.") == 1
    assert audit[0]["source_start"] == selected["start"]


def test_catalog_excludes_yaml_headings_and_fenced_code():
    text = "---\nname: fixed\n---\n# Method\nUse images.\n```python\n# Rule\nunsafe()\n```\nKeep facts.\n"
    assert [a["source"] for a in source_anchors(text)] == ["Use images.", "Keep facts."]


@pytest.mark.parametrize("mutation,code", [
    (lambda p: p["edits"][0].update(anchor_id=True), "unknown_anchor_id"),
    (lambda p: p["edits"][0].update(section_id=0), "anchor_section_mismatch"),
    (lambda p: p["edits"][0].update(before="Keep evidence."), "invalid_edit_anchor_selector"),
    (lambda p: p.update(base_sha256=digest("# Method\nChanged source.\n")),
     "edit_base_identity_mismatch"),
])
def test_catalog_selector_rejects_ambiguous_schema_or_changed_base(mutation, code):
    text = "# Method\nKeep evidence.\n"
    p = proposal(text, "unused", "Inspect the original frames.")
    p["edits"] = [{"section_id": 1, "anchor_id": 0, "after": "Inspect the original frames."}]
    mutation(p)
    with pytest.raises(EditProtocolError, match=code):
        apply_edits(text, p)


def test_catalog_and_substring_edits_share_overlap_checks_and_legacy_support():
    text = "# Method\nKeep evidence. Inspect frames.\n"
    p = proposal(text, "Keep evidence.", "Keep supported evidence.")
    p["edits"].append({"section_id": 1, "anchor_id": 0, "after": "Retain evidence."})
    with pytest.raises(EditProtocolError, match="overlapping_edits"):
        apply_edits(text, p)
    p["edits"].pop()
    p["protocol"] = "v11-anchored-revision-edits/1"
    assert apply_edits(text, p)[0] == "# Method\nKeep supported evidence. Inspect frames.\n"


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
        self.last_request_identity = describe_request(
            self.build_payload(messages, **kwargs),
            model_identity=self.audit_identity(seed=kwargs["seed"]),
            wire_encoding_known=True)
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


def test_catalog_selection_produces_a_static_candidate_without_echoing_before(tmp_path):
    p = parent()
    anchor = source_anchors(p.skill_md)[-1]
    response = proposal(p.skill_md, "unused", anchor["source"] + " Preserve supported observations.")
    response["edits"] = [{"section_id": anchor["section_id"], "anchor_id": anchor["anchor_id"],
                         "after": anchor["source"] + " Preserve supported observations."}]
    prompts = []

    def respond(messages):
        prompts.append(messages[-1]["content"])
        return json.dumps(response)

    client = Client(respond)
    reviser = AnchoredSlotReviser(output_root=tmp_path, client=client,
               budget=RevisionBudget(), method_context_max_chars=8000, expected_model="test-model")
    candidate = reviser.run(parent=p, feedback={}, batches=["authorized learning facts"],
                            delivery_plan={}, forbidden_ids=set(), cases=[])
    assert candidate and "Source sentence catalog:" in prompts[0]
    assert candidate.skill_md == (p.skill_md[:anchor["start"]] + response["edits"][0]["after"]
                                 + p.skill_md[anchor["end"]:])
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


def test_distinct_invalid_anchors_are_not_reported_as_duplicate(tmp_path):
    p = parent()
    outputs = iter([
        proposal(p.skill_md, "first nonexistent source", "new rule"),
        proposal(p.skill_md, "second nonexistent source", "new rule"),
        {"protocol": PROTOCOL, "base_sha256": digest(p.skill_md), "action": "no_change",
         "modification_reason": "The current method remains adequate."},
    ])
    client = Client(lambda _: json.dumps(next(outputs)))
    reviser = AnchoredSlotReviser(output_root=tmp_path, client=client,
               budget=RevisionBudget(), method_context_max_chars=8000, expected_model="test-model")
    assert reviser.run(parent=p, feedback={}, batches=["facts"], delivery_plan={},
                       forbidden_ids=set(), cases=[]) is None
    final = json.loads((tmp_path / "final.json").read_text())
    assert client.calls == 3 and final["status"] == "no_change"
    assert not any(check.get("duplicate") for check in final["checks"])


def test_static_repair_keeps_host_error_code_and_numeric_location(tmp_path, monkeypatch):
    from skill3d.evolution import revision_edits_v11 as module
    p = parent()
    before = p.skill_md.splitlines()[-1]
    response = proposal(p.skill_md, before, before + " Preserve supported observations.")
    received = []

    def respond(messages):
        received.append(messages[-1]["content"])
        if len(received) == 1:
            return json.dumps(response)
        return json.dumps({"protocol": PROTOCOL, "base_sha256": digest(p.skill_md),
                           "action": "no_change", "modification_reason": "Keep the valid parent."})

    monkeypatch.setattr(module, "_static_validate", lambda **kw:
        (None, ["execution_contract_conflict: line=7; rule=forbidden_io"]))
    client = Client(respond)
    reviser = AnchoredSlotReviser(output_root=tmp_path, client=client,
               budget=RevisionBudget(), method_context_max_chars=8000, expected_model="test-model")
    reviser.run(parent=p, feedback={}, batches=["facts"], delivery_plan={},
                forbidden_ids=set(), cases=[])
    assert '"code": "execution_contract_conflict"' in received[1]
    assert '"line": 7' in received[1] and '"rule": "forbidden_io"' in received[1]


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


def test_actual_wire_identity_failure_stops_before_consuming_or_repairing_reply(tmp_path):
    p = parent()

    def changed_wire(_):
        client.last_request_identity["wire_payload_sha256"] = "0" * 64
        return "{}"

    client = Client(changed_wire)
    reviser = AnchoredSlotReviser(output_root=tmp_path, client=client,
               budget=RevisionBudget(), method_context_max_chars=8000, expected_model="test-model")
    with pytest.raises(RevisionOperationalError, match="actual revision request identity"):
        reviser.run(parent=p, feedback={}, batches=["facts"], delivery_plan={},
                    forbidden_ids=set(), cases=[])
    assert client.calls == 1
    assert not (tmp_path / "final.json").exists()
    assert not list(tmp_path.glob("validation_*.json"))


def deferred_budget(n):
    return RevisionBudget(max_requests=2 * n + 2, repair_reserve=n + 2, deferred_repair=True)


def no_change(text):
    return {"protocol": PROTOCOL, "base_sha256": digest(text), "action": "no_change",
            "modification_reason": "Keep the supported working method."}


def test_deferred_replay_delivers_later_evidence_and_uses_newest_working_method(tmp_path):
    p = parent()
    before = p.skill_md.splitlines()[-1]
    after = before + " Preserve supported observations."
    working = apply_edits(p.skill_md, proposal(p.skill_md, before, after))[0]
    outputs = iter([proposal(p.skill_md, "missing", "replacement"),
                    proposal(p.skill_md, before, after), no_change(working)])
    received = []

    def respond(messages):
        received.append(messages[-1]["content"])
        return json.dumps(next(outputs))

    client = Client(respond)
    editor = AnchoredSlotReviser(output_root=tmp_path, client=client, budget=deferred_budget(2),
                                method_context_max_chars=8000, expected_model="test-model")
    candidate = editor.run(parent=p, feedback={}, batches=["first facts", "later facts"],
                          delivery_plan={}, forbidden_ids=set(), cases=[])
    assert candidate and candidate.skill_md == working
    final = json.loads((tmp_path / "final.json").read_text())
    assert final["all_evidence_delivered"] and final["all_batches_resolved"]
    assert [(row["batch"], row["phase"]) for row in final["checks"]] == [
        (0, "delivery"), (1, "delivery"), (0, "deferred_replay")]
    assert "first facts" in received[2] and working in received[2]
    assert "later facts" in received[1] and client.calls == 3


def test_unresolved_batch_blocks_valid_earlier_candidate_after_all_delivery(tmp_path):
    p = parent()
    before = p.skill_md.splitlines()[-1]
    after = before + " Preserve supported observations."
    working = apply_edits(p.skill_md, proposal(p.skill_md, before, after))[0]
    outputs = iter([proposal(p.skill_md, before, after)] +
                   [proposal(working, "missing", "replacement")] * 4)
    client = Client(lambda _: json.dumps(next(outputs)))
    editor = AnchoredSlotReviser(output_root=tmp_path, client=client, budget=deferred_budget(2),
                                method_context_max_chars=8000, expected_model="test-model")
    assert editor.run(parent=p, feedback={}, batches=["first", "second"], delivery_plan={},
                      forbidden_ids=set(), cases=[]) is None
    final = json.loads((tmp_path / "final.json").read_text())
    assert final["all_evidence_delivered"] and not final["all_batches_resolved"]
    assert final["unresolved_batch_indices"] == [1]
    assert client.calls == 5 and not (tmp_path / "SKILL.md").exists()


def test_deferred_mode_requires_full_replay_and_final_repair_reservation():
    assert not RevisionBudget(max_requests=6, repair_reserve=2, deferred_repair=True).plan(2)["admitted"]
    assert deferred_budget(2).plan(2)["admitted"]
