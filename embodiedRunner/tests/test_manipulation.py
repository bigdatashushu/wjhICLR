"""Seven-dimensional actions, official public inputs and real process boundaries."""
import copy
from dataclasses import replace

import pytest

from action_adapters import FakeAdapter, validate_packet
from action_bridge import ActionBridge, LocalTransport
from action_contract import (ActionPayload, DecisionSnapshot, EpisodeBinding, ExecutionEnvelope,
                             decode_action_payload, public_observation, validate_action_payload)
from action_host import ActionHost, Proposal
from budget import RunBudget
from manipulation_planner import parse_plan
from manipulation_protocol import public_input, vector
from mutation_protocol import MutationSession, make_command
from protocol import ContractError, canonical, sha256
from sandbox_process import execute_isolated

ENV = "eb_manipulation"
ACTION = [50, 50, 30, 120, 60, 90, 1]


def fixture():
    packet = FakeAdapter(ENV).open({"subset": "base", "index": 0, "source_sha256": sha256(b"source")})
    png = validate_packet(packet)
    binding = EpisodeBinding("test", "test-episode", "base:000", ENV, "base", packet["input_sha256"])
    obs = public_observation(
        binding, png=png, instruction=packet["instruction"], environment_step=0, mutation_revision=1,
        observation_revision=0, action_table=[], public_feedback="ready",
        manipulation_input=packet["manipulation_input"], planner_png=png)
    snapshot = DecisionSnapshot(obs, "test-decision", "", sha256(b"budget"))
    envelope = ExecutionEnvelope(binding.episode_id, "test-session", 1, 0,
                                 snapshot.decision_id, snapshot.snapshot_sha256)
    return binding, snapshot, envelope


@pytest.mark.parametrize("bad", [
    0, [], [1] * 6, [1] * 8, [True, 50, 30, 120, 60, 90, 1],
    [50.0, 50, 30, 120, 60, 90, 1], [-1, 50, 30, 120, 60, 90, 1],
    [101, 50, 30, 120, 60, 90, 1], [50, 50, 30, 121, 60, 90, 1],
    [50, 50, 30, 120, 60, 90, True], [50, 50, 30, 120, 60, 90, .5],
])
def test_bad_vector_rejected_before_mutation(bad):
    session = MutationSession(FakeAdapter(ENV))
    def command(op, params, sequence):
        return make_command(
            version=session.version, session_id=session.session_id, episode_id="test-episode",
            op=op, parameters=params, mutation_revision=session.revision, environment_step=session.step,
            table_sha256=session.table_sha256, sequence=sequence)
    session.execute(command("open", {"subset": "base", "index": 0,
                                    "source_sha256": sha256(b"source")}, 1))
    with pytest.raises(ContractError):
        session.execute(command("step", {"action": bad, "first_in_plan": True}, 2))
    assert session.step == 0 and session.revision == 1 and not session.poisoned


def test_complete_plan_validated_and_roundtrips_without_flattening():
    _binding, snapshot, envelope = fixture()
    payload = ActionPayload(snapshot.decision_id, 0, snapshot.snapshot_sha256,
                            snapshot.observation.action_table_sha256, [ACTION, ACTION])
    decoded = decode_action_payload(canonical(payload.to_dict()))
    assert decoded.actions == (tuple(ACTION), tuple(ACTION))
    assert validate_action_payload(decoded, snapshot, envelope, session_id="test-session", max_actions=15) == ()
    with pytest.raises(ContractError):
        validate_action_payload(replace(payload, actions=[ACTION, [1] * 6]), snapshot, envelope,
                                session_id="test-session", max_actions=15)


def test_anonymous_coordinate_provenance_and_unknown_private_fields():
    _binding, snapshot, _envelope = fixture()
    state = snapshot.observation.public_dict()
    assert not {"reward", "task_success", "depth", "mask", "waypoints", "object_informations"} & state.keys()
    value = copy.deepcopy(state["manipulation_input"])
    value["objects"]["red_target"] = [1, 2, 3]
    with pytest.raises(ContractError):
        public_input(value, "base")
    value = copy.deepcopy(state["manipulation_input"])
    value["objects"]["object 1"][0] += 1
    with pytest.raises(ContractError):
        public_input(value, "base")
    with pytest.raises(TypeError):
        snapshot.observation.manipulation_input["objects"]["object 1"][0] = 1


def test_official_parser_variants_and_no_random_fallback():
    for plan in ([{"action": str(ACTION)}], str([ACTION]), [ACTION]):
        assert parse_plan(canonical({"executable_plan": plan})) == (tuple(ACTION),)
    for value in ({}, {"executable_plan": [{"action": ACTION}, {"action": [1] * 6}]},
                  {"executable_plan": "malformed"}):
        with pytest.raises(ContractError):
            parse_plan(canonical(value))
    assert parse_plan('{"executable_plan": []}') == ()
    assert vector([0, 100, 0, 0, 120, 0, 1.0]) == (0, 100, 0, 0, 120, 0, 1)


def test_direct_includes_verified_official_json_output_template():
    from action_solver import ActionSolver
    from action_run import FakeClient
    _binding, snapshot, _envelope = fixture()
    solver = ActionSolver(FakeClient("Direct"), arm="Direct", environment=ENV)
    messages, _plan, _chars = solver.messages(snapshot, [])
    text = "".join(p.get("text", "") for m in messages if isinstance(m["content"], list)
                   for p in m["content"])
    assert solver.adapter.output_template in text
    assert "json" in text.lower() and "executable_plan" in text
    assert "embodiedbench/planner/planner_utils.py" in solver.prompt_hashes


def test_real_sandbox_reads_public_world_coordinates_and_submits_vector():
    binding, snapshot, envelope = fixture()
    source = """def solve(ctx):
    xyz = ctx.official_objects["object 1"]
    a = [xyz[0], xyz[1], 30, 120, 60, 90, 1]
    SubmitActions(ActionPayload(ctx.decision_id, ctx.observation_revision,
        ctx.snapshot_sha256, ctx.action_table_sha256, [a]))
"""
    result = execute_isolated(source, snapshot, binding, envelope, max_actions=15)
    assert result.error_code is None and result.submission.actions == ((40, 50, 30, 120, 60, 90, 1),)
    assert result.receipt["isolation_sealed"] and result.tool_calls == result.reader_calls == 0


def test_same_uuid_ack_and_failed_motion_count_as_step():
    adapter = FakeAdapter(ENV, failed_steps=(1,), stationary_rgb=True)
    budget = RunBudget("test", [ENV]).episode("test-episode", ENV)
    bridge = ActionBridge(LocalTransport(MutationSession(adapter)), environment=ENV,
                          episode_id="test-episode", budget=budget)
    before = bridge.open(subset="base", index=0, source_sha256=sha256(b"source"))
    after = bridge.step(ACTION)
    assert not after["last_action_success"] and after["step"] == 1
    assert before["image_sha256"] == after["image_sha256"]
    assert budget.counts["steps"] == 1
    with pytest.raises(ContractError):
        bridge.invalid()
    session = bridge.transport.session
    ack = list(session.cache.values())[-1]
    saved = make_command(
        version=session.version, session_id=session.session_id, episode_id="test-episode",
        op="step", parameters={"action": ACTION, "first_in_plan": True}, mutation_revision=1,
        environment_step=0, table_sha256=ack["action_table_sha256"], sequence=8,
        command_id=ack["command_id"])
    assert session.execute(saved) == ack and adapter.step_count == 1
