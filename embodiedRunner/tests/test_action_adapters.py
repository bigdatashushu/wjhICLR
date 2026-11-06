"""Exercise V2 normalization around unchanged V1-shaped official interfaces."""
from copy import deepcopy

import pytest

from action_adapters import FakeAdapter, HouseholdAdapter, NavigationAdapter, validate_packet
from mutation_protocol import MutationSession, make_command
from protocol import ContractError, sha256


class OfficialShape:
    def __init__(self, environment):
        self.fake = FakeAdapter(environment)
        self.subset = "base"
        self.calls = []

    def _raw(self, packet):
        raw = {key: packet[key] for key in (
            "image_base64", "image_sha256", "step", "instruction", "last_action_success", "env_feedback")}
        raw.update(done=packet["official"]["done"], task_success=packet["official"]["task_success"])
        if self.fake.environment != "eb_navigation":
            raw.update(actions=packet["actions"], input_sha256=packet["input_sha256"],
                       task_progress=packet["official"]["task_progress"],
                       invalid_actions=packet["official"]["invalid_actions"], reward=0.0)
        return raw

    def open(self, *args):
        self.calls.append(("open", args))
        if self.fake.environment == "eb_navigation":
            subset, index, source = args
        else:
            index, source = args
            subset = self.subset
        return self._raw(self.fake.open({"subset": subset, "index": index, "source_sha256": source}))

    def step(self, *args):
        self.calls.append(("step", args))
        return self._raw(self.fake.step(args[0], True))

    def invalid(self):
        self.calls.append(("invalid", ()))
        return self._raw(self.fake.invalid())

    def close(self):
        self.fake.close()


@pytest.mark.parametrize("environment", ("eb_navigation", "eb_alfred", "eb_habitat"))
def test_existing_interfaces_are_delegated_with_no_model_reasoning(environment):
    official = OfficialShape(environment)
    adapter = NavigationAdapter(official) if environment == "eb_navigation" else HouseholdAdapter(official, environment)
    session = MutationSession(adapter)
    def mutate(op, parameters):
        return session.execute(make_command(
            version=session.version, session_id=session.session_id, episode_id="episode",
            op=op, parameters=parameters, mutation_revision=session.revision, environment_step=session.step,
            table_sha256=session.table_sha256, sequence=1))["packet"]
    first = mutate("open", {"subset": "base", "index": 0, "source_sha256": sha256(b"input")})
    assert validate_packet(first)
    second = mutate("step", {"action": 0, "first_in_plan": True})
    assert second["step"] == 1
    assert official.calls[-1] == ("step", (0, True) if environment == "eb_navigation" else (0,))
    if environment != "eb_navigation":
        third = mutate("invalid", {})
        assert third["step"] == 1 and third["official"]["invalid_actions"] == 1
    adapter.close()
    assert official.fake.closed


@pytest.mark.parametrize("field,value", [
    ("task_success", 1), ("task_progress", float("nan")), ("invalid_actions", True)])
def test_scoring_types_are_checked_only_in_private_official_packet(field, value):
    packet = FakeAdapter("eb_habitat").open(
        {"subset": "base", "index": 0, "source_sha256": sha256(b"input")})
    packet = deepcopy(packet)
    packet["official"][field] = value
    with pytest.raises(ContractError):
        validate_packet(packet)


def test_reward_or_raw_metadata_cannot_enter_v2_normalized_packet():
    packet = FakeAdapter("eb_alfred").open(
        {"subset": "base", "index": 0, "source_sha256": sha256(b"input")})
    for field in ("reward", "info", "gt_pose", "depth"):
        with pytest.raises(ContractError):
            validate_packet({**packet, field: 1})
