"""Verify full feedback delivery and single-completion progress across mutation."""
from dataclasses import replace
import json

from action_contract import plain
from action_run import FakeClient, fake_run
from protocol import canonical
from task_card import MAX_HISTORY, SubgoalLedger, build_card, validate_card
from task_context import examples
from test_action_contract import fixture
from test_task_card import goal, transition, changed


def test_all_actions_and_latest_forty_feedbacks_survive_small_target():
    obs, snapshot, _, _ = fixture()
    ledger = SubgoalLedger("eb_habitat")
    for i in range(45):
        ledger.observe(changed(obs, i + 1, f"Complete official feedback {i}."),
                       action=1, succeeded=True, kind="step")
    actions = tuple(f"find a Object{i}" for i in range(218))
    snapshot = replace(snapshot, observation=replace(obs, action_table=actions),
                       task_state=ledger.public_state())
    card, wire = build_card(snapshot, fixed_chars=15000)
    assert len(card["history"]) == MAX_HISTORY == 40
    assert [r["feedback"] for r in card["history"]] == [
        f"Complete official feedback {i}." for i in range(5, 45)]
    assert card["catalog"]["text"] == "\n".join(f"{i}:{name}" for i, name in enumerate(actions))
    assert card["catalog"]["omitted"] == card["packing"]["omitted_history"] == 0
    assert card["packing"]["required_fields_over_budget"]
    assert validate_card(wire) == card


def test_public_examples_are_deterministic_bounded_and_attributed():
    for env in ("eb_navigation", "eb_alfred", "eb_habitat"):
        rows = examples(env, "base")
        assert rows == examples(env, "base") and rows
        assert len(canonical(rows).decode()) <= 1500
        assert all(len(row["source_sha256"]) == 64 and row["excerpt"] for row in rows)
        if env == "eb_habitat":
            assert all("Output:" in row["excerpt"] and "action_id" in row["excerpt"] for row in rows)


def test_revision_is_explicit_integer_on_every_request():
    from action_solver import ActionSolver
    _, snapshot, _, _ = fixture()
    solver = ActionSolver(FakeClient("B01"), arm="B01", environment="eb_navigation", task_cards=True)
    for revision in (0, 1, 17):
        current = replace(snapshot, observation=replace(snapshot.observation, observation_revision=revision))
        messages, _, _ = solver.messages(current, {})
        assert f"MUST be the integer {revision}." in messages[0]["content"]
        assert 'Never write the string "current"' in messages[0]["content"]
        assert "*** Strategy ***" in messages[0]["content"]
        assert "Use Rotation Sparingly" in messages[0]["content"]


def test_progress_uses_one_completion_and_survives_next_observation(tmp_path):
    class Client(FakeClient):
        def chat(self, messages, **kwargs):
            if self.arm == "Direct":
                return super().chat(messages, **kwargs)
            self.calls += 1
            self.last_usage = {"total_tokens": 16, "prompt_tokens": 8, "completion_tokens": 8}
            self.last_response_identity = {"model": "offline-fake"}
            card = json.loads(messages[1]["content"][0]["text"])
            if card["history"]:
                assert card["previous_progress"]["source"] == "model_assessment"
                assert card["history"][-1]["feedback"]
                assert card["previous_progress"]["remaining"] == "Reach target"
            return canonical({
                "subgoal": goal(obj="", destination=""),
                "transition": transition(card["identity"]["observation_revision"]),
                "progress": {"completed": "Reviewed public feedback", "remaining": "Reach target",
                             "next": "Move using current image"},
                "program_source": "def solve(ctx):\n"
                    "    assert ctx.task_state['progress']['source'] == 'model_assessment'\n"
                    "    SubmitActions(ActionPayload(ctx.decision_id,ctx.observation_revision,"
                    "ctx.snapshot_sha256,ctx.action_table_sha256,[0],[]))",
            }).decode()
    result = fake_run(environments=("eb_navigation",), client_factory=Client,
                      task_cards=True, trajectory_dir=tmp_path / "run-episodes")
    assert result["audit"]["status"] == "passed", result["audit"]
    for attempt in result["attempts"][1:]:
        assert len(attempt["model_requests"]) == attempt["environment_steps"] == 2
    from audit_action_trajectory import audit
    path = tmp_path / "run.json"
    path.write_bytes(canonical(result))
    assert audit(path)["status"] == "passed"


def test_format_hint_survives_invalid_observation_and_new_decision():
    class Client(FakeClient):
        def chat(self, messages, **kwargs):
            if self.arm == "Direct":
                return super().chat(messages, **kwargs)
            self.calls += 1
            self.last_usage = {"total_tokens": 16, "prompt_tokens": 8, "completion_tokens": 8}
            self.last_response_identity = {"model": "offline-fake"}
            card = json.loads(messages[1]["content"][0]["text"])
            if self.calls == 2:
                assert card["repair"]["format_reason"] == "advance_source_none"
                assert card["repair"]["format_field"] == "transition"
                assert "Advance requires source official_feedback or visual" in card["repair"]["corrective_action"]
            return canonical({
                "subgoal": goal("main", "visual", "", ""),
                "transition": transition(card["identity"]["observation_revision"],
                                         kind="advance" if self.calls == 1 else "keep"),
                "program_source": "def solve(ctx):\n"
                    "    SubmitActions(ActionPayload(ctx.decision_id,ctx.observation_revision,"
                    "ctx.snapshot_sha256,ctx.action_table_sha256,[0],[]))",
            }).decode()
    result = fake_run(environments=("eb_navigation",), client_factory=Client, task_cards=True)
    assert result["audit"]["status"] == "passed"
    for attempt in result["attempts"][1:]:
        assert attempt["environment_steps"] == 2
        assert len(attempt["model_requests"]) == 3
        assert [e["code"] for e in attempt["errors"]] == ["model_output_format_error"]
