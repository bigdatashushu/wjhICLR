"""V2 checks precede any state write; duplicate commands never reset/step twice."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile

import pytest

from action_adapters import FakeAdapter
from action_bridge import ActionBridge, LocalTransport, NDJSONTransport, v2_environment
from action_worker_v2 import serve
from budget import RunBudget
from mutation_protocol import MutationSession, command_digest, make_command
from protocol import ContractError, OperationalError, canonical, sha256

ENVIRONMENTS = ("eb_navigation", "eb_alfred", "eb_habitat")


@pytest.fixture
def worker_directory():
    parent = Path(__file__).resolve().parents[1] / "tmp"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ndjson-test-", dir=parent) as directory:
        yield Path(directory)


def command(session, op, parameters, *, sequence=1, episode_id="episode"):
    return make_command(
        version=session.version, session_id=session.session_id, episode_id=episode_id,
        op=op, parameters=parameters, mutation_revision=session.revision,
        environment_step=session.step, table_sha256=session.table_sha256, sequence=sequence)


def open_command(session):
    return command(session, "open", {"subset": "base", "index": 0, "source_sha256": sha256(b"input")})


def open_session(environment="eb_navigation"):
    adapter = FakeAdapter(environment)
    session = MutationSession(adapter)
    session.execute(open_command(session))
    return session, adapter


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_open_and_step_exact_duplicates_return_original_ack(environment):
    adapter = FakeAdapter(environment)
    session = MutationSession(adapter)
    opening = open_command(session)
    ack = session.execute(opening)
    assert session.execute({**opening, "transport_sequence": 2}) == ack
    assert adapter.open_count == 1
    stepping = command(session, "step", {"action": 0, "first_in_plan": True}, sequence=3)
    step_ack = session.execute(stepping)
    assert session.execute({**stepping, "transport_sequence": 4}) == step_ack
    assert adapter.step_count == 1 and session.revision == 2 and session.step == 1
    step_ack["packet"]["actions"][0] = "tampered caller"
    assert session.execute(stepping)["packet"]["actions"][0] != "tampered caller"


@pytest.mark.parametrize("field,value", [
    ("command_id", "not-uuid"), ("session_id", "other"), ("episode_id", "other"),
    ("expected_mutation_revision", 0), ("expected_environment_step", 1),
    ("expected_action_table_sha256", sha256(b"other")), ("transport_sequence", True),
    ("parameters", {"action": True, "first_in_plan": True}),
    ("parameters", {"action": 0.0, "first_in_plan": True}),
    ("parameters", {"action": -1, "first_in_plan": True}),
    ("parameters", {"action": 8, "first_in_plan": True}),
    ("parameters", {"action": 0, "first_in_plan": False}),
    ("parameters", {"action": 0, "first_in_plan": True, "reward": 1}),
])
def test_invalid_state_types_and_parameters_do_not_mutate(field, value):
    session, adapter = open_session()
    stepping = command(session, "step", {"action": 0, "first_in_plan": True})
    stepping[field] = value
    stepping["command_digest"] = command_digest(stepping)
    with pytest.raises(ContractError):
        session.execute(stepping)
    assert session.revision == 1 and adapter.step_count == 0 and not session.poisoned


def test_digest_collision_stale_new_uuid_and_new_open_rejected_before_mutation():
    session, adapter = open_session()
    stepping = command(session, "step", {"action": 0, "first_in_plan": True})
    stale = command(session, "step", {"action": 1, "first_in_plan": True})
    session.execute(stepping)
    altered = {**stepping, "parameters": {"action": 1, "first_in_plan": True}}
    altered["command_digest"] = command_digest(altered)
    with pytest.raises(ContractError, match="collision"):
        session.execute(altered)
    with pytest.raises(ContractError, match="stale"):
        session.execute(stale)
    with pytest.raises(ContractError, match="already_open"):
        session.execute(open_command(session))
    tampered = command(session, "step", {"action": 1, "first_in_plan": True})
    tampered["command_digest"] = sha256(b"other")
    with pytest.raises(ContractError, match="digest_mismatch"):
        session.execute(tampered)
    assert adapter.step_count == 1 and adapter.open_count == 1


@pytest.mark.parametrize("environment", ("eb_alfred", "eb_habitat"))
def test_invalid_is_deduplicated_mutation_without_step_and_reuses_rgb(environment):
    session, adapter = open_session(environment)
    before = deepcopy(session.last_packet)
    invalid = command(session, "invalid", {})
    ack = session.execute(invalid)
    assert session.execute(invalid) == ack
    assert session.revision == 2 and session.step == 0 and adapter.invalid_count == 1
    assert ack["packet"]["image_base64"] == before["image_base64"]


def test_exception_after_state_write_poisoned_and_no_reexecution():
    session, adapter = open_session()
    def partially_mutate(action, first):
        adapter.step_count += 1
        raise RuntimeError("synthetic-private")
    adapter.step = partially_mutate
    stepping = command(session, "step", {"action": 0, "first_in_plan": True})
    with pytest.raises(OperationalError, match="action_outcome_unknown"):
        session.execute(stepping)
    assert session.poisoned and adapter.step_count == 1
    with pytest.raises(OperationalError, match="action_outcome_unknown"):
        session.execute(stepping)
    assert adapter.step_count == 1


def bridge(environment, transport):
    budget = RunBudget("run", [environment]).episode("episode", environment)
    return ActionBridge(transport, environment=environment, episode_id="episode", budget=budget)


def test_lost_ack_host_stops_and_keeps_action_reservation():
    session, adapter = open_session()
    # New host must handshake only to a fresh session.
    adapter = FakeAdapter("eb_navigation")
    session = MutationSession(adapter)
    class DropACK(LocalTransport):
        def exchange(self, command, timeout_s):
            ack = super().exchange(command, timeout_s)
            if command["op"] == "step":
                raise OperationalError("wire_lost")
            return ack
    host = bridge("eb_navigation", DropACK(session))
    host.open(subset="base", index=0, source_sha256=sha256(b"input"))
    with pytest.raises(OperationalError, match="action_outcome_unknown"):
        host.step(0)
    assert host.poisoned and adapter.step_count == 1 and host.step_count == 0
    with pytest.raises(OperationalError):
        host.step(0)
    assert adapter.step_count == 1 and host.budget.counts["steps"] == 1
    assert host.receipts[-1]["status"] == "unknown"
    host.close()


def test_bad_ack_identity_cannot_advance_host():
    session = MutationSession(FakeAdapter("eb_navigation"))
    class BadACK(LocalTransport):
        def exchange(self, command, timeout_s):
            ack = super().exchange(command, timeout_s)
            ack["mutation_revision"] = True
            return ack
    host = bridge("eb_navigation", BadACK(session))
    with pytest.raises(OperationalError):
        host.open(subset="base", index=0, source_sha256=sha256(b"input"))
    assert host.revision == 0 and host.packet is None and host.poisoned
    host.close()


def test_duplicate_json_rejected_by_actual_worker_loop():
    session = MutationSession(FakeAdapter("eb_navigation"))
    output = io.StringIO()
    serve(session, io.StringIO('{"op":"handshake","op":"handshake","transport_sequence":0}\n'), output)
    assert '"ok":false' in output.getvalue()
    assert session.revision == 0


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_real_ndjson_v2_process_fake_reset_step_and_invalid(tmp_path, worker_directory, environment):
    root = Path(__file__).resolve().parents[1]
    # Fake worker does no simulator/model imports or launches.
    bootstrap = "import sys; sys.path.insert(0," + repr(str(root)) + "); from action_worker_v2 import main; main()"
    transport = NDJSONTransport(
        [sys.executable, "-I", "-B", "-c", bootstrap, "--environment", environment,
         "--work-dir", str(worker_directory), "--fake"],
        work_dir=worker_directory, environment=environment)
    host = bridge(environment, transport)
    try:
        first = host.open(subset="base", index=0, source_sha256=sha256(b"input"))
        second = host.step(0)
        assert first["step"] == 0 and second["step"] == 1 and host.revision == 2
        if environment != "eb_navigation":
            third = host.invalid()
            assert third["image_sha256"] == second["image_sha256"] and host.step_count == 1
        final = host.step(0)
        assert final["official"]["done"] and final["official"]["task_success"]
    finally:
        host.close()
    assert transport.process.poll() is not None
    assert not any("TOKEN" in key or "SECRET" in key or "PROXY" in key
                   for key in v2_environment(tmp_path))
