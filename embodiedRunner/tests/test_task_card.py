"""Task-card length, semantic state and conditional decision integration."""
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from action_contract import EpisodeBinding, plain, public_hash
from action_host import ActionHost
from action_run import FakeClient, FakeDepth, fake_run
from action_solver import ActionSolver
from budget import RunBudget
from evidence import CurrentEvidencePreparer, PerceptionBackends
from protocol import ContractError, canonical, sha256
from sandbox_wire import pack_snapshot, unpack_snapshot
from task_card import (MAX_CHARS, SubgoalLedger, build_card, validate_card, validate_goal,
                       validate_perception, validate_transition, build_legacy_card)
from test_action_contract import fixture


def goal(name="carry", condition="released", obj="plate", destination="right counter"):
    return {"id": name, "goal": "Place the plate at the right counter", "object": obj,
            "destination": destination, "complete_when": condition}


def transition(revision, kind="keep", source="none"):
    return {"kind": kind, "source": source, "observation_revision": revision,
            "detail": "" if kind == "keep" else "Public feedback supports the change"}


def changed(obs, revision, feedback, *, environment="eb_habitat",
            actions=("pick up the plate", "navigate to the right counter", "place at the right counter")):
    return replace(obs, environment=environment, observation_revision=revision,
                   environment_step=revision, public_feedback=feedback, action_table=actions)


def test_long_history_and_catalog_are_complete_whole_records():
    obs, snapshot, _, _ = fixture()
    names = tuple(f"find the complete object {i} " + "description " * 30 for i in range(1024))
    records = [{"action": f"action-{i}-" + "完整描述" * 100, "success": False,
                "revision": i, "ack": str(i)} for i in range(12)]
    state = {"goal": goal(), "facts": {"holding": {"status": "holding", "object": "plate"}},
             "last_action": records[-1], "feedback": "Robot is currently holding plate.",
             "history": records, "loop": {"switch_strategy": True}}
    snapshot = replace(snapshot, observation=replace(obs, action_table=names), task_state=state)
    card, wire = build_legacy_card(snapshot)
    assert len(wire) <= 4000 and json.loads(wire) == card
    assert card["task"] == obs.instruction and card["subgoal"] == goal()
    assert card["latest"]["action"] == records[-1] and card["facts"] == state["facts"]
    assert card["catalog"]["omitted"] > 0 and card["packing"]["omitted_history"] > 0
    for item in card["history"]:
        assert item in records
    for entry in card["catalog"]["entries"]:
        assert entry["name"] == names[entry["id"]]
    _, second = build_legacy_card(snapshot, catalog_offset=500)
    assert validate_card(second)["catalog"]["entries"][0]["id"] == 500


@pytest.mark.parametrize("atom", ["中文", '"\n\\', "🙂🧭", "target object "])
def test_serialized_unicode_and_escaping_never_exceed_cap(atom):
    obs, snapshot, _, _ = fixture()
    instruction = atom * 80
    snapshot = replace(snapshot, observation=replace(obs, instruction=instruction))
    card, wire = build_card(snapshot, repair={"code": "model_output_format_error"})
    assert len(wire) <= 4000 and card["task"] == instruction
    assert wire.endswith("}") and validate_card(wire)["packing"]["complete"]


def test_randomized_packing_retains_mandatory_semantic_atoms():
    rng = random.Random(137)
    obs, snapshot, _, _ = fixture()
    for _ in range(80):
        names = tuple("find object " + str(i) + " 名称" * rng.randint(0, 150)
                      for i in range(rng.randint(1, 150)))
        state = {"goal": goal(), "feedback": "完整最新反馈。", "facts": {"holding": {"status": "holding"}},
                 "history": [{"action": "old" + str(i) + "xyz" * rng.randint(0, 150),
                              "revision": i, "success": False} for i in range(12)]}
        current = replace(snapshot, observation=replace(obs, action_table=names), task_state=state)
        card, wire = build_legacy_card(current)
        assert len(wire) <= 4000 and card["subgoal"] == goal()
        assert card["latest"]["feedback"] == state["feedback"]
        assert card["facts"] == state["facts"] and validate_card(wire) == card


def test_mandatory_overflow_sends_complete_card_without_truncating():
    obs, snapshot, _, _ = fixture()
    snapshot = replace(snapshot, observation=replace(obs, instruction="完整总任务" * 4000))
    client = FakeClient("B01")
    solver = ActionSolver(client, arm="B01", environment="eb_navigation", task_cards=True)
    proposal = solver.generate(snapshot, feedback={}, usage={})
    assert proposal.perception_request is not None
    assert client.calls == 1
    card = solver.last_card
    assert card["task"] == snapshot.observation.instruction
    assert card["packing"]["required_fields_over_budget"] is True
    assert solver.receipts[0]["task_card_over_budget"] is True
    assert solver.receipts[0]["task_card_chars"] > MAX_CHARS
    assert validate_card(canonical(card).decode()) == card


def test_long_required_feedback_remains_complete_and_deliverable():
    _, snapshot, _, _ = fixture()
    feedback = "This entire detailed feedback must remain available. " * 400
    snapshot = replace(snapshot, task_state={"goal": goal(), "feedback": feedback, "facts": {}})
    card, wire = build_card(snapshot)
    assert len(wire) > MAX_CHARS
    assert card["latest"]["feedback"] == feedback and card["subgoal"] == goal()
    assert validate_card(wire) == card and card["packing"]["required_fields_over_budget"]


def test_optional_overflow_cannot_claim_to_be_required_overflow():
    _, snapshot, _, _ = fixture()
    card, _ = build_legacy_card(snapshot)
    card["history"] = [{"action": "extra optional history " * 300}]
    with pytest.raises(ContractError, match="task_card_context_cap"):
        validate_card(canonical(card).decode())


@pytest.mark.parametrize("environment", ["eb_navigation", "eb_manipulation"])
def test_required_overflow_does_not_abort_host_or_independent_audit(environment):
    from action_adapters import FakeAdapter
    from action_bridge import LocalTransport
    from mutation_protocol import MutationSession
    class LongInstruction(FakeAdapter):
        def _packet(self, **kwargs):
            packet = super()._packet(**kwargs)
            packet["instruction"] = "Complete the whole public task. " * 600
            return packet
    result = fake_run(
        environments=(environment,), task_cards=True,
        transport_factory=lambda env: LocalTransport(MutationSession(LongInstruction(env))))
    assert result["status"] == "passed" and result["audit"]["status"] == "passed", result["audit"]
    for attempt in result["attempts"][1:]:
        assert attempt["status"] == "finished" and attempt["environment_steps"] == 2
        assert all(row["task_card_chars"] > MAX_CHARS and row["task_card_over_budget"] is True
                   for row in attempt["model_requests"])


def test_holding_persists_through_generic_navigation_feedback_and_release_clears():
    obs, _, _, _ = fixture()
    ledger = SubgoalLedger("eb_habitat")
    ledger.observe(changed(obs, 1, "you are holding plate."), action=0, succeeded=True, kind="step")
    ledger.observe(changed(obs, 2, "Last action executed successfully."),
                   action=1, succeeded=True, kind="step")
    assert ledger.facts["holding"]["object"] == "plate"
    ledger.accept_goal(goal(), transition(2), 2)
    assert ledger.check_action(0, changed(obs, 2, "")) == "already_holding_before_pickup"
    ledger.observe(changed(obs, 3, "you are holding nothing."), action=2, succeeded=True, kind="step")
    assert ledger.facts["holding"]["status"] == "empty"
    assert ledger.check_action(2, changed(obs, 3, "")) == "not_holding_before_release"


def test_alfred_failed_pickup_invalidates_control_target():
    obs, _, _, _ = fixture()
    names = ("find a bathtub", "pick up the HandTowel", "put down the HandTowel")
    ledger = SubgoalLedger("eb_alfred")
    ledger.observe(changed(obs, 1, "Last action executed successfully.",
                           environment="eb_alfred", actions=names), action=0, succeeded=True, kind="step")
    assert ledger.facts["control_target"]["target"] == "bathtub"
    ledger.observe(changed(obs, 2, "Robot is currently holding HandTowel",
                           environment="eb_alfred", actions=names), action=1, succeeded=False, kind="step")
    assert ledger.facts["control_target"] is None
    ledger.accept_goal(goal(obj="HandTowel", destination="bathtub"), transition(2), 2)
    assert ledger.check_action(2, changed(obs, 2, "", environment="eb_alfred",
                                         actions=names)) == "find_destination_before_put"


def test_motion_ack_cannot_complete_a_grasp():
    obs, _, _, _ = fixture()
    ledger = SubgoalLedger("eb_manipulation")
    ledger.accept_goal(goal(name="grasp", condition="holding", obj="star", destination=""),
                       transition(0), 0)
    ledger.observe(changed(obs, 1, "Last action is valid.", environment="eb_manipulation"),
                   action=[57, 74, 30, 120, 60, 90, 1], succeeded=True, kind="step")
    with pytest.raises(ContractError, match="subgoal_completion_not_supported"):
        ledger.accept_goal(goal(name="carry", obj="star"),
                           transition(1, "advance", "official_feedback"), 1)
    assert ledger.goal["id"] == "grasp" and ledger.facts["holding"]["status"] == "unknown"


def test_snapshot_task_state_roundtrip_and_frozen_fields():
    obs, snapshot, _, _ = fixture()
    binding = EpisodeBinding("run", obs.episode_id, obs.official_task_id,
                             obs.environment, obs.subset, sha256(b"input"))
    snapshot = replace(snapshot, task_state={"goal": goal(), "facts": {"holding": {"status": "empty"}}})
    _, restored = unpack_snapshot(pack_snapshot(snapshot, binding))
    assert restored.snapshot_sha256 == snapshot.snapshot_sha256
    assert plain(restored.task_state) == plain(snapshot.task_state)
    with pytest.raises(TypeError):
        restored.task_state["goal"]["goal"] = "tamper"


def test_dynamic_categories_cover_star_and_container_with_separate_candidates():
    obs, snapshot, _, _ = fixture()
    calls = []
    class Detector:
        identity = {"backend": "fake-detector"}
        def __call__(self, rgb, categories):
            calls.append(("dino", categories))
            return [{"category": category, "box_xyxy": [10, 10, 40, 40], "confidence": .9}
                    for category in categories]
    class Segmenter:
        identity = {"backend": "fake-sam"}
        def __call__(self, rgb, boxes):
            calls.append(("sam", len(boxes)))
            return [np.ones((500, 500), dtype=bool) for _ in boxes]
    binding = EpisodeBinding("run", obs.episode_id, obs.official_task_id, obs.environment, obs.subset,
                             sha256(b"input"))
    budget = RunBudget("run", [obs.environment]).episode(obs.episode_id, obs.environment)
    prep = CurrentEvidencePreparer(PerceptionBackends(Detector(), Segmenter(), depth_estimator=FakeDepth()))
    request = {"targets": [{"role": "object", "category": "star"},
                           {"role": "destination", "category": "silver container"}], "depth": False}
    result = prep(snapshot, binding, budget, request=request)
    assert calls == [("dino", ("star", "silver container")), ("sam", 2)]
    assert {row["category"] for row in result.prepared.candidates} == {"star", "silver container"}
    assert not budget.counts["moge"] and budget.counts["dino"] == budget.counts["sam"] == 1
    assert all(not row["geometry_usable"] for row in result.prepared.candidates)


def test_direct_action_needs_no_mandatory_planning_or_perception_call():
    class ReadyClient(FakeClient):
        def chat(self, messages, **kwargs):
            self.calls += 1
            self.last_usage = {"total_tokens": 16, "prompt_tokens": 8, "completion_tokens": 8}
            self.last_response_identity = {"model": "offline-fake"}
            if self.arm == "Direct":
                return super().chat(messages, **kwargs)
            card = json.loads(messages[1]["content"][0]["text"])
            assert card["evidence"]["status"] == "not_requested"
            program = ("def solve(ctx):\n"
                       "    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,"
                       " ctx.snapshot_sha256, ctx.action_table_sha256, [0], []))")
            return canonical({"subgoal": goal(obj="", destination=""),
                              "transition": transition(card["identity"]["observation_revision"]),
                              "program_source": program}).decode()
    result = fake_run(environments=("eb_navigation",), client_factory=ReadyClient, task_cards=True)
    assert result["status"] == "passed" and result["audit"]["status"] == "passed"
    for attempt in result["attempts"][1:]:
        assert len(attempt["model_requests"]) == attempt["environment_steps"] == 2
        assert not attempt["budget"]["reserved"].get("moge", 0)
        assert all(r["task_card_chars"] <= MAX_CHARS for r in attempt["model_requests"])


def test_perception_service_failure_stops_without_model_retry():
    class BrokenDepth(FakeDepth):
        def __call__(self, rgb):
            raise RuntimeError("provider-secret")
    result = fake_run(environments=("eb_navigation",), task_cards=True,
                      preparer_factory=lambda _: CurrentEvidencePreparer(
                          PerceptionBackends(depth_estimator=BrokenDepth())))
    for attempt in result["attempts"][1:]:
        assert attempt["status"] == "aborted"
        assert len(attempt["model_requests"]) == 1 and attempt["budget"]["reserved"]["moge"] == 1
    assert "provider-secret" not in repr(result)


def test_repair_card_preserves_entire_subgoal_and_error():
    obs, snapshot, _, _ = fixture()
    snapshot = replace(snapshot, task_state={"goal": goal(), "facts": {},
                                            "feedback": "You are holding plate", "history": []})
    card, wire = build_card(snapshot, repair={"code": "already_holding_before_pickup"})
    assert card["subgoal"] == goal()
    assert card["repair"]["code"] == "already_holding_before_pickup"
    assert validate_card(wire) == card


def test_large_yield_feedback_keeps_complete_pages_and_marks_omissions():
    _, snapshot, _, _ = fixture()
    pages = [{"result_id": str(i), "data": {"candidates": ["full candidate record " * 25]},
              "scope": {"snapshot_sha256": snapshot.snapshot_sha256}} for i in range(20)]
    ids = [str(i) * 100 for i in range(32)]
    card, wire = build_legacy_card(snapshot, repair={"code": "yield_observations",
                                             "observations": pages, "yielded_result_ids": ids})
    assert len(wire) <= 4000 and card["repair"]["code"] == "yield_observations"
    assert card["repair"]["omitted_observations"] > 0
    assert all(page in pages for page in card["repair"]["observations"])
    assert all(result_id in ids for result_id in card["repair"]["yielded_result_ids"])
    assert validate_card(wire) == card


@pytest.mark.parametrize("extra", [
    {"perception_request": {"targets": [], "depth": True}},
    {"unknown": True},
])
def test_union_output_rejects_mixed_actions_and_requests(extra):
    _, snapshot, _, _ = fixture()
    class Mixed(FakeClient):
        def chat(self, *args, **kwargs):
            value = {"subgoal": goal(), "transition": transition(0), "program_source": "pass", **extra}
            self.last_usage = None
            return canonical(value).decode()
    solver = ActionSolver(Mixed("B01"), arm="B01", environment="eb_navigation", task_cards=True)
    with pytest.raises(ContractError, match="model_output_format_error"):
        solver.generate(snapshot, feedback={}, usage={})


def test_conflicting_pickup_is_repaired_without_environment_step():
    class ConflictClient(FakeClient):
        def chat(self, messages, **kwargs):
            if self.arm == "Direct":
                return super().chat(messages, **kwargs)
            self.calls += 1
            self.last_usage = {"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16}
            self.last_response_identity = {"model": "offline-fake"}
            card = json.loads(messages[1]["content"][0]["text"])
            index = 2 if card["repair"].get("code") == "already_holding_before_pickup" else 1
            source = ("def solve(ctx):\n"
                      "    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,"
                      f" ctx.snapshot_sha256, ctx.action_table_sha256, [{index}], []))")
            if index == 2:
                assert card["facts"]["holding"]["status"] == "holding"
                assert card["subgoal"] is not None
            return canonical({"subgoal": goal(obj="cup", destination=""),
                              "transition": transition(card["identity"]["observation_revision"]),
                              "program_source": source}).decode()
    result = fake_run(environments=("eb_habitat",), client_factory=ConflictClient, task_cards=True)
    assert result["audit"]["status"] == "passed"
    for attempt in result["attempts"][1:]:
        assert attempt["environment_steps"] == 2 and len(attempt["model_requests"]) == 3
        assert any(e["code"] == "already_holding_before_pickup" for e in attempt["errors"])


def test_transition_does_not_invent_completion_or_mutate_verified_holding():
    obs, _, _, _ = fixture()
    ledger = SubgoalLedger("eb_habitat")
    ledger.accept_goal(goal(name="grab", condition="holding", obj="plate"), transition(0), 0)
    ledger.observe(changed(obs, 1, "You are holding nothing."), action=1, succeeded=True, kind="step")
    with pytest.raises(ContractError, match="subgoal_completion_not_supported"):
        ledger.accept_goal(goal(), transition(1, "advance", "visual"), 1)
    assert ledger.facts["holding"]["status"] == "empty"


def test_program_can_read_the_frozen_public_task_state():
    from sandbox_process import execute_isolated
    obs, snapshot, envelope, _ = fixture()
    binding = EpisodeBinding("run", obs.episode_id, obs.official_task_id, obs.environment, obs.subset,
                             sha256(b"input"))
    snapshot = replace(snapshot, task_state={"facts": {"holding": {"status": "holding"}}})
    envelope = replace(envelope, snapshot_sha256=snapshot.snapshot_sha256)
    source = """def solve(ctx):
    i = 1 if ctx.task_state['facts']['holding']['status'] == 'holding' else 0
    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,
        ctx.snapshot_sha256, ctx.action_table_sha256, [i], []))
"""
    result = execute_isolated(source, snapshot, binding, envelope, max_actions=20)
    assert result.error_code is None and result.submission.actions == (1,)
    assert result.receipt["isolation_sealed"]


def test_yield_keeps_accepted_goal_and_rebinds_results_for_reworded_keep():
    class EmptyDetector:
        identity = {"backend": "offline-empty-detector", "version": "1"}
        def __call__(self, rgb, categories):
            assert categories == ("chair",)
            return []
    class YieldClient(FakeClient):
        def chat(self, messages, **kwargs):
            if self.arm == "Direct":
                return super().chat(messages, **kwargs)
            self.calls += 1
            self.last_usage = {"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16}
            self.last_response_identity = {"model": "offline-fake"}
            card = json.loads(messages[1]["content"][0]["text"])
            selected = goal(name="find", condition="visual", obj="chair", destination="")
            value = {"subgoal": selected,
                     "transition": transition(card["identity"]["observation_revision"])}
            if card["evidence"]["status"] == "not_requested":
                value["perception_request"] = {
                    "targets": [{"role": "object", "category": "chair"}], "depth": False}
            elif card["repair"].get("code") != "yield_observations":
                selected["goal"] = "Locate the chair using the current evidence"
                value["program_source"] = (
                    "def solve(ctx):\n"
                    "    assert ctx.task_state['goal']['goal'] == 'Locate the chair using the current evidence'\n"
                    "    r = view_candidates()\n"
                    "    YieldObservations([r['result_id']], 'inspect')\n")
            else:
                assert card["subgoal"]["goal"] == "Locate the chair using the current evidence"
                assert card["repair"]["observations"][0]["data"]["candidates"] == []
                result_id = card["repair"]["yielded_result_ids"][0]
                selected["goal"] = "Search the current view for the chair"
                value["program_source"] = (
                    "def solve(ctx):\n"
                    "    assert ctx.task_state['goal']['goal'] == 'Search the current view for the chair'\n"
                    f"    page = ReadObservation('{result_id}', ['candidates'])\n"
                    "    assert page['scope']['snapshot_sha256'] == ctx.snapshot_sha256\n"
                    "    assert page['data']['candidates'] == []\n"
                    "    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,"
                    " ctx.snapshot_sha256, ctx.action_table_sha256, [0], []))\n")
            return canonical(value).decode()
    result = fake_run(
        environments=("eb_navigation",), client_factory=YieldClient, task_cards=True,
        preparer_factory=lambda env: CurrentEvidencePreparer(PerceptionBackends(detector=EmptyDetector())))
    assert result["status"] == "passed" and result["audit"]["status"] == "passed", result["audit"]
    for attempt in result["attempts"][1:]:
        assert len(attempt["model_requests"]) == 6 and attempt["environment_steps"] == 2
        assert len(attempt["executions"]) == 4 and all(e["error_code"] is None
                                                       for e in attempt["executions"])
        assert attempt["budget"]["reserved"]["tools"] == 2
        assert attempt["budget"]["reserved"]["reads"] == 4
        assert all(r["task_card_chars"] <= MAX_CHARS for r in attempt["model_requests"])
        for request, execution in zip(attempt["model_requests"][1::3], attempt["executions"][::2]):
            assert request["snapshot_sha256"] != execution["receipt"]["snapshot_sha256"]


@pytest.mark.parametrize("key,value", [
    ("id", "other"), ("object", "cup"), ("destination", "left counter"),
    ("complete_when", "holding"),
])
def test_keep_allows_goal_paraphrase_but_rejects_changed_identity(key, value):
    ledger = SubgoalLedger("eb_habitat")
    original = goal()
    ledger.accept_goal(original, transition(0), 0)
    paraphrase = {**original, "goal": "Set the plate down on the right counter"}
    ledger.accept_goal(paraphrase, transition(0), 0)
    assert ledger.goal == paraphrase
    with pytest.raises(ContractError, match="subgoal_keep_changed"):
        ledger.accept_goal({**paraphrase, key: value}, transition(0), 0)
    assert ledger.goal == paraphrase
    assert "revise" in ledger.repair_feedback("subgoal_keep_changed")["corrective_action"]


def test_matching_official_find_completes_locate_visual_without_proving_holding():
    obs, _, _, _ = fixture()
    ledger = SubgoalLedger("eb_alfred")
    find = goal("find_plate", "visual", "Plate", "")
    find["goal"] = "Locate a Plate to begin setting the table."
    ledger.accept_goal(find, transition(0), 0)
    ledger.observe(changed(obs, 1, "Last action executed successfully.", environment="eb_alfred",
                           actions=("find a Plate",)), action=0, succeeded=True, kind="step")
    ledger.accept_goal(goal("pickup_plate", "holding", "Plate", ""),
                       transition(1, "advance", "official_feedback"), 1)
    assert ledger.facts["holding"]["status"] == "unknown"
    with pytest.raises(ContractError, match="subgoal_completion_not_supported"):
        ledger.accept_goal(goal(), transition(1, "advance", "official_feedback"), 1)


@pytest.mark.parametrize("name,succeeded,description", [
    ("find a Cup", True, "Locate a Plate"), ("find a Plate", False, "Locate a Plate"),
    ("find a Plate", True, "Place a Plate on the counter"),
])
def test_find_ack_cannot_complete_other_objects_failures_or_placement(name, succeeded, description):
    obs, _, _, _ = fixture()
    ledger = SubgoalLedger("eb_alfred")
    selected = goal("find_plate", "visual", "Plate", "")
    selected["goal"] = description
    ledger.accept_goal(selected, transition(0), 0)
    ledger.observe(changed(obs, 1, "Public action feedback.", environment="eb_alfred",
                           actions=(name,)), action=0, succeeded=succeeded, kind="step")
    with pytest.raises(ContractError, match="subgoal_completion_not_supported"):
        ledger.accept_goal(goal(), transition(1, "advance", "official_feedback"), 1)


def test_navigation_ack_matches_current_waypoint_with_legacy_final_destination():
    obs, _, _, _ = fixture()
    ledger = SubgoalLedger("eb_habitat")
    selected = goal("nav_sink", "navigation_ack", "plate", "right counter")
    selected["goal"] = "Navigate to the sink in the kitchen to locate the plate."
    ledger.accept_goal(selected, transition(0), 0)
    ledger.observe(changed(obs, 1, "Last action executed successfully.",
                           actions=("navigate to the sink in the kitchen",)),
                   action=0, succeeded=True, kind="step")
    ledger.accept_goal(goal("pickup_plate", "holding", "plate", ""),
                       transition(1, "advance", "official_feedback"), 1)
    assert ledger.goal["id"] == "pickup_plate"


@pytest.mark.parametrize("name", ["navigate to the left counter", "MoveAhead", "RotateRight"])
def test_navigation_ack_requires_matching_named_target(name):
    obs, _, _, _ = fixture()
    ledger = SubgoalLedger("eb_habitat")
    selected = goal("nav_sink", "navigation_ack", "plate", "sink")
    selected["goal"] = "Navigate to the sink to locate the plate."
    ledger.accept_goal(selected, transition(0), 0)
    ledger.observe(changed(obs, 1, "Last action executed successfully.", actions=(name,)),
                   action=0, succeeded=True, kind="step")
    with pytest.raises(ContractError, match="subgoal_completion_not_supported"):
        ledger.accept_goal(goal(), transition(1, "advance", "official_feedback"), 1)
    assert ledger.repair_feedback("subgoal_completion_not_supported")["expected_navigation_target"] == "the sink"


@pytest.mark.parametrize("period", [2, 6, 12])
def test_repeated_public_navigation_sequence_blocks_only_predicted_repeat(period):
    obs, _, _, _ = fixture()
    names = ("RotateRight", "MoveAhead", "RotateLeft")
    ledger = SubgoalLedger("eb_navigation")
    start = changed(obs, 0, "", environment="eb_navigation", actions=names)
    ledger.observe(replace(start, pixel_sha256=sha256(b"view-0")), kind="open")
    for step in range(1, 2 * period + 1):
        current = replace(
            changed(obs, step, "Last action executed successfully.", environment="eb_navigation",
                    actions=names), pixel_sha256=sha256(f"view-{step % period}".encode()))
        ledger.observe(current, action=(step - 1) % period % 2, succeeded=True, kind="step")
        if step < period + 2:
            assert ledger.loop["cycle_period"] == 0
    assert ledger.loop["cycle_period"] == period and ledger.loop["switch_strategy"]
    assert ledger.loop["cycle_next_action"] == "RotateRight"
    assert ledger.check_action(0, current) == "navigation_loop_action_blocked"
    assert ledger.check_action(2, current) is None
    ledger.observe(replace(current, observation_revision=current.observation_revision + 1),
                   kind="invalid")
    assert ledger.check_action(0, current) == "navigation_loop_action_blocked"
    ledger.observe(replace(current, pixel_sha256=sha256(b"new-view")),
                   action=2, succeeded=True, kind="step")
    assert not ledger.loop["switch_strategy"] and len(ledger.navigation_trace) <= 25


def test_repeated_views_with_different_actions_do_not_form_a_navigation_cycle():
    obs, _, _, _ = fixture()
    ledger = SubgoalLedger("eb_navigation")
    names = ("RotateRight", "MoveAhead")
    for step in range(9):
        current = replace(changed(obs, step, "", environment="eb_navigation", actions=names),
                          pixel_sha256=sha256(f"view-{step % 6}".encode()))
        ledger.observe(current, action=0 if step < 6 else 1, succeeded=True,
                       kind="open" if step == 0 else "step")
    assert ledger.loop["cycle_period"] == 0 and not ledger.loop["switch_strategy"]


def test_stationary_navigation_repeats_are_blocked_but_successful_interaction_is_not():
    obs, _, _, _ = fixture()
    ledger = SubgoalLedger("eb_navigation")
    names = ("MoveAhead", "RotateLeft")
    for step in range(1, 4):
        current = changed(obs, step, "Last action executed successfully.",
                          environment="eb_navigation", actions=names)
        ledger.observe(current, action=0, succeeded=True, kind="step")
    assert ledger.loop["same_action_no_progress"] == 2
    assert ledger.check_action(0, current) == "navigation_loop_action_blocked"
    assert ledger.check_action(1, current) is None
    manipulation = SubgoalLedger("eb_manipulation")
    for step in range(1, 5):
        manipulation.observe(changed(obs, step, "Last action is valid.", environment="eb_manipulation"),
                             action=[50, 50, 30, 120, 60, 90, 1], succeeded=True, kind="step")
    assert not manipulation.loop["switch_strategy"]


def test_loop_repair_chooses_alternative_without_executing_blocked_action():
    from action_adapters import FakeAdapter
    from action_bridge import LocalTransport
    from mutation_protocol import MutationSession
    class LoopClient(FakeClient):
        def chat(self, messages, **kwargs):
            if self.arm == "Direct":
                return super().chat(messages, **kwargs)
            self.calls += 1
            self.last_usage = {"prompt_tokens": 8, "completion_tokens": 8, "total_tokens": 16}
            self.last_response_identity = {"model": "offline-fake"}
            card = json.loads(messages[1]["content"][0]["text"])
            index = 1 if card["repair"].get("code") == "navigation_loop_action_blocked" else 0
            return canonical({
                "subgoal": goal("search", "visual", "chair", ""),
                "transition": transition(card["identity"]["observation_revision"]),
                "program_source": "def solve(ctx):\n"
                    "    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,"
                    f" ctx.snapshot_sha256, ctx.action_table_sha256, [{index}], []))",
            }).decode()
    result = fake_run(
        environments=("eb_navigation",), client_factory=LoopClient, task_cards=True,
        transport_factory=lambda env: LocalTransport(MutationSession(
            FakeAdapter(env, done_at=4, stationary_rgb=True))))
    assert result["status"] == "passed" and result["audit"]["status"] == "passed", result["audit"]
    for attempt in result["attempts"][1:]:
        assert attempt["environment_steps"] == 4 and len(attempt["model_requests"]) == 5
        assert [e["code"] for e in attempt["errors"]] == ["navigation_loop_action_blocked"]
        assert len([m for m in attempt["mutations"] if m["op"] == "step"]) == 4
