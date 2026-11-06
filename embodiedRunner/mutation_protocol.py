"""V2 worker-side pre-mutation checks and live-session command deduplication.

Compatible with simulator Python 3.9; does not import agent/numerical modules.
"""
from __future__ import annotations

import re
import uuid

from protocol import ContractError, OperationalError, canonical, sha256

VERSIONS = {"eb_navigation": "embodiedrunner-navigation/2",
            "eb_alfred": "embodiedrunner-household/2",
            "eb_habitat": "embodiedrunner-household/2",
            "eb_manipulation": "embodiedrunner-manipulation/2"}
MAX_BYTES = 12_000_000
FIELDS = frozenset({"transport_sequence", "protocol_version", "session_id", "command_id",
                    "command_digest", "op", "episode_id", "expected_mutation_revision",
                    "expected_environment_step", "expected_action_table_sha256", "parameters"})
ACK_FIELDS = frozenset({"command_id", "command_digest", "session_id", "episode_id", "op",
                        "before_mutation_revision", "mutation_revision", "before_environment_step",
                        "environment_step", "action_table_sha256", "packet_sha256", "packet"})


def uuid_string(value):
    try:
        if type(value) is not str or str(uuid.UUID(value)) != value:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise ContractError("invalid_uuid") from None
    return value


def nonnegative(value):
    if type(value) is not int or value < 0:
        raise ContractError("invalid_revision")
    return value


def string_id(value):
    if type(value) is not str or not 1 <= len(value) <= 200:
        raise ContractError("invalid_episode_identity")
    return value


def hash_string(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError("invalid_digest")
    return value


def action_table_hash(names):
    if (type(names) is not list or not 0 <= len(names) <= 1024
            or any(type(name) is not str or not 1 <= len(name) <= 2000 for name in names)):
        raise ContractError("invalid_action_table")
    return sha256(canonical([{"action_id": index, "action_name": name}
                             for index, name in enumerate(names)]))


def command_digest(command):
    return sha256(canonical({key: value for key, value in command.items()
                             if key not in {"transport_sequence", "command_digest"}}))


def make_command(*, version, session_id, episode_id, op, parameters,
                 mutation_revision, environment_step, table_sha256,
                 sequence, command_id=None):
    value = {"transport_sequence": sequence, "protocol_version": version, "session_id": session_id,
             "command_id": command_id or str(uuid.uuid4()), "op": op, "episode_id": episode_id,
             "expected_mutation_revision": mutation_revision,
             "expected_environment_step": environment_step,
             "expected_action_table_sha256": table_sha256, "parameters": parameters}
    value["command_digest"] = command_digest(value)
    return value


class MutationSession:
    """Only this object may call an adapter's open/step/invalid methods."""
    def __init__(self, adapter):
        self.adapter, self.environment = adapter, adapter.environment
        self.version = VERSIONS[self.environment]
        self.session_id = str(uuid.uuid4())
        self.episode_id = ""
        self.revision, self.step, self.table_sha256 = 0, 0, ""
        self.cache, self.last_packet = {}, None
        self.poisoned = False

    def handshake(self):
        return {"protocol_version": self.version, "session_id": self.session_id,
                "environment": self.environment, "mutation_revision": self.revision,
                "environment_step": self.step, "action_table_sha256": self.table_sha256,
                "episode_id": self.episode_id, "automatic_retry": False}

    def _validate_parameters(self, op, parameters):
        if type(parameters) is not dict:
            raise ContractError("invalid_command_parameters")
        if op == "open":
            if set(parameters) != {"subset", "index", "source_sha256"}:
                raise ContractError("invalid_command_parameters")
            string_id(parameters["subset"])
            nonnegative(parameters["index"])
            hash_string(parameters["source_sha256"])
            self.adapter.validate_open(parameters)
        elif op == "step":
            if set(parameters) != {"action", "first_in_plan"}:
                raise ContractError("invalid_command_parameters")
            if self.environment == "eb_manipulation":
                from manipulation_protocol import vector
                vector(parameters["action"])
            else:
                action = nonnegative(parameters["action"])
                if action >= len(self.last_packet["actions"]):
                    raise ContractError("invalid_action_id")
            if type(parameters["first_in_plan"]) is not bool:
                raise ContractError("invalid_action_id")
            if parameters["first_in_plan"] is not True:
                raise ContractError("single_action_requires_first")
        elif op == "invalid":
            if parameters or self.environment in {"eb_navigation", "eb_manipulation"}:
                raise ContractError("invalid_command_parameters")
        else:
            raise ContractError("invalid_mutation_op")

    def execute(self, command):
        if self.poisoned:
            raise OperationalError("action_outcome_unknown")
        if (type(command) is not dict or set(command) != FIELDS
                or len(canonical(command)) > MAX_BYTES
                or command["protocol_version"] != self.version
                or command["session_id"] != self.session_id):
            raise ContractError("mutation_identity_mismatch")
        seq = nonnegative(command["transport_sequence"])
        if seq < 1:
            raise ContractError("invalid_transport_sequence")
        cid = uuid_string(command["command_id"])
        hash_string(command["command_digest"])
        if command["command_digest"] != command_digest(command):
            raise ContractError("command_digest_mismatch")
        string_id(command["episode_id"])
        nonnegative(command["expected_mutation_revision"])
        nonnegative(command["expected_environment_step"])
        # A matching cached command may contain its original now-stale revision.
        if cid in self.cache:
            ack = self.cache[cid]
            if ack["command_digest"] != command["command_digest"]:
                raise ContractError("command_id_collision")
            # Return a copy so callers cannot mutate the cached ACK.
            from protocol import json_object
            return json_object(canonical(ack))
        op = command["op"]
        if (command["expected_mutation_revision"] != self.revision
                or command["expected_environment_step"] != self.step
                or command["expected_action_table_sha256"] != self.table_sha256):
            raise ContractError("stale_mutation_state")
        if op == "open":
            if self.episode_id or self.revision:
                raise ContractError("episode_already_open")
        elif not self.episode_id or command["episode_id"] != self.episode_id:
            raise ContractError("mutation_episode_mismatch")
        if self.last_packet and self.last_packet["official"]["done"]:
            raise ContractError("episode_finished")
        self._validate_parameters(op, command["parameters"])
        if len(self.cache) >= 128:
            raise ContractError("mutation_cache_cap")
        before_revision, before_step = self.revision, self.step
        # Beyond this boundary even a ContractError may follow partial state
        # mutation. Any failure poisons the session, no re-send/reset/replay.
        try:
            parameters = command["parameters"]
            if op == "open":
                packet = self.adapter.open(parameters)
            elif op == "step":
                packet = self.adapter.step(parameters["action"], parameters["first_in_plan"])
            else:
                packet = self.adapter.invalid()
            self.adapter.validate_packet(packet)
            expected_step = before_step + (op == "step")
            if (packet["step"] != expected_step
                    or packet["environment"] != self.environment):
                raise ContractError("mutated_packet_state_mismatch")
            if self.last_packet:
                if any(packet[key] != self.last_packet[key] for key in (
                        "instruction", "input_sha256", "official_task_id", "subset")):
                    raise ContractError("mutated_task_identity_mismatch")
                before_invalid = self.last_packet["official"]["invalid_actions"]
                after_invalid = packet["official"]["invalid_actions"]
                if (op == "invalid" and (after_invalid != before_invalid + 1
                                        or packet["image_sha256"] != self.last_packet["image_sha256"])
                        or op == "step" and after_invalid < before_invalid):
                    raise ContractError("mutated_invalid_state_mismatch")
            table = action_table_hash(packet["actions"])
            ack = {"command_id": cid, "command_digest": command["command_digest"],
                   "session_id": self.session_id, "episode_id": command["episode_id"], "op": op,
                   "before_mutation_revision": before_revision, "mutation_revision": before_revision + 1,
                   "before_environment_step": before_step, "environment_step": expected_step,
                   "action_table_sha256": table, "packet_sha256": sha256(canonical(packet)), "packet": packet}
            if len(canonical(ack)) > MAX_BYTES:
                raise ContractError("mutation_ack_cap")
            from protocol import json_object
            self.last_packet = json_object(canonical(packet))
            self.cache[cid] = json_object(canonical(ack))
            self.episode_id, self.revision, self.step = command["episode_id"], before_revision + 1, expected_step
            self.table_sha256 = table
            return ack
        except BaseException:
            self.poisoned = True
            raise OperationalError("action_outcome_unknown") from None
