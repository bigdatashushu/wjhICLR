"""Contracts and revision failures using credential-free fake observations."""
import io
import json
from dataclasses import FrozenInstanceError, replace

import pytest
from PIL import Image

from action_contract import (ActionPayload, DecisionSnapshot, EpisodeBinding, ExecutionEnvelope,
                             decode_action_payload, public_observation, validate_action_payload)
from protocol import ContractError, sha256
from state import EpisodeState


def observation(*, revision=0, mutation=1, step=0, feedback="Ready", actions=("forward", "right")):
    stream = io.BytesIO()
    Image.new("RGB", (500, 500), "white").save(stream, format="PNG")
    binding = EpisodeBinding("run", "episode", "base:0", "eb_navigation", "base", sha256(b"input"))
    return public_observation(binding, png=stream.getvalue(), instruction="Find the chair",
                              environment_step=step, mutation_revision=mutation,
                              observation_revision=revision, action_table=actions, public_feedback=feedback)


def fixture():
    obs = observation()
    snapshot = DecisionSnapshot(obs, "decision", "", sha256(b"budget"))
    envelope = ExecutionEnvelope(obs.episode_id, "session", 1, 0, snapshot.decision_id, snapshot.snapshot_sha256)
    payload = ActionPayload("decision", 0, snapshot.snapshot_sha256, obs.action_table_sha256, [0])
    return obs, snapshot, envelope, payload


@pytest.mark.parametrize("actions", [[True], [0.0], [-1], [2], ["0"], list(range(31))])
def test_strict_actions(actions):
    _, snapshot, envelope, payload = fixture()
    with pytest.raises(ContractError):
        validate_action_payload(replace(payload, actions=actions), snapshot, envelope,
                                session_id="session", max_actions=20)


def test_empty_stop_and_public_whitelist_is_immutable():
    obs, snapshot, envelope, payload = fixture()
    assert validate_action_payload(replace(payload, actions=[]), snapshot, envelope,
                                   session_id="session", max_actions=20) == ()
    assert not ({"done", "reward", "task_success", "task_progress", "info"} & set(obs.public_dict()))
    with pytest.raises(TypeError):
        obs.public_memory["subgoal"] = "tampered"
    with pytest.raises(FrozenInstanceError):
        snapshot.decision_id = "tampered"


def test_unknown_duplicate_and_nonfinite_json_rejected():
    _, _, _, payload = fixture()
    value = payload.to_dict()
    value["task_success"] = True
    with pytest.raises(ContractError):
        decode_action_payload(value)
    text = json.dumps(payload.to_dict())
    with pytest.raises(ContractError):
        decode_action_payload(text[:-1] + ', "actions": [1]}')
    with pytest.raises(ContractError):
        decode_action_payload(text.replace('"memory_update": null', '"memory_update": NaN'))


def test_identity_and_omitted_dependencies():
    _, snapshot, envelope, payload = fixture()
    for broken in (replace(payload, observation_revision=True),
                   replace(payload, decision_id="other"),
                   replace(payload, action_table_sha256=sha256(b"other"))):
        with pytest.raises(ContractError):
            validate_action_payload(broken, snapshot, envelope, session_id="session", max_actions=20)
    with pytest.raises(ContractError):
        validate_action_payload(payload, snapshot, replace(envelope, episode_id="other"),
                                session_id="session", max_actions=20)
    with pytest.raises(ContractError):
        validate_action_payload(payload, snapshot, envelope, session_id="session", max_actions=20,
                                actual_result_ids=["old"], authorize_result=lambda *_: False)
    assert validate_action_payload(payload, snapshot, envelope, session_id="session", max_actions=20,
                                   actual_result_ids=["current"], authorize_result=lambda *_: True) == ("current",)


def test_memory_source_and_scope():
    _, snapshot, envelope, payload = fixture()
    record = {"text": "Chair on right", "source": "visual", "episode_id": "episode",
              "observation_revision": 0, "decision_id": "decision", "confidence": "uncertain"}
    memory = {"subgoal": "Approach chair", "records": [record]}
    assert validate_action_payload(replace(payload, memory_update=memory), snapshot, envelope,
                                   session_id="session", max_actions=20) == ()
    for key, value in (("source", "ground_truth"), ("episode_id", "other"),
                       ("decision_id", "old"), ("observation_revision", 1), ("task_success", True)):
        bad = {"subgoal": "", "records": [{**record, key: value}]}
        with pytest.raises(ContractError):
            validate_action_payload(replace(payload, memory_update=bad), snapshot, envelope,
                                    session_id="session", max_actions=20)


def test_same_rgb_step_and_invalid_revoke_current():
    obs = observation()
    binding = EpisodeBinding("run", "episode", "base:0", "eb_navigation", "base", sha256(b"input"))
    state = EpisodeState(binding, "session", "", sha256(b"budget"))
    state.advance_observation(obs, mutation_kind="open")
    old = state.prepare_snapshot()
    state.transition("synthesizing", "prepared")
    state.transition("executing", "program")
    state.transition("validating", "submitted")
    state.transition("applying", "accepted")
    next_obs = observation(revision=1, mutation=2, step=1, feedback="Action failed")
    assert obs.rgb_sha256 == next_obs.rgb_sha256
    state.advance_observation(next_obs, mutation_kind="step")
    assert old.decision_id in state.revoked_decisions and state.snapshot is None
    current = state.prepare_snapshot()
    state.transition("synthesizing", "prepared")
    state.advance_observation(observation(revision=2, mutation=3, step=1, feedback="Invalid plan"),
                              mutation_kind="invalid")
    assert current.decision_id in state.revoked_decisions
    assert state.observation.environment_step == 1 and len(state.ledger) == 10


def test_revision_failure_is_atomic_and_evidence_is_new_snapshot():
    binding = EpisodeBinding("run", "episode", "base:0", "eb_navigation", "base", sha256(b"input"))
    state = EpisodeState(binding, "session", "", sha256(b"budget"))
    state.advance_observation(observation(), mutation_kind="open")
    old = state.prepare_snapshot()
    new = state.extend_evidence(manifest_sha256=sha256(b"manifest"), capabilities=("current_rgb", "relative_depth"))
    assert old.decision_id == new.decision_id and old.snapshot_sha256 != new.snapshot_sha256
    assert old.evidence_revision == 0
    with pytest.raises(ContractError):
        state.advance_observation(observation(revision=1, mutation=2, step=1), mutation_kind="step")
    state.transition("synthesizing", "prepared")
    with pytest.raises(ContractError):
        state.advance_observation(observation(revision=1, mutation=2, step=0), mutation_kind="step")
    assert state.observation.observation_revision == 0 and not state.revoked_decisions
