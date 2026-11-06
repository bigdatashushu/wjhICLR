"""Opt-in V2 host bridge: one outstanding command, no automatic re-send."""
from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from action_adapters import validate_packet
from mutation_protocol import (ACK_FIELDS, MAX_BYTES, VERSIONS, action_table_hash,
                               make_command, uuid_string)
from protocol import ContractError, OperationalError, canonical, json_object, sha256


class LocalTransport:
    """Same protocol path for fake endpoints; no host access to mutable adapter."""
    def __init__(self, session):
        self.session = session

    def handshake(self):
        return self.session.handshake()

    def exchange(self, command, timeout_s):
        return self.session.execute(command)

    def close(self):
        self.session.adapter.close()


def v2_environment(work_dir):
    """Never inherit auth/proxy/service variables into a simulator."""
    result = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
              "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
              "TMPDIR": str(work_dir), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for name in ("HOME", "LANG", "LC_ALL", "DISPLAY", "LD_LIBRARY_PATH",
                 "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        if name in os.environ:
            result[name] = os.environ[name]
    return result


class NDJSONTransport:
    """Bounded nonblocking writes/reads use one deadline for the entire command."""
    def __init__(self, command, *, work_dir, environment):
        work = Path(work_dir).resolve()
        if not work.is_dir():
            raise ContractError("v2_work_directory_missing")
        self.environment, self.closed, self.buffer = environment, False, b""
        self.process = subprocess.Popen(
            command, cwd=work, env=v2_environment(work), close_fds=True, start_new_session=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)

    def _exchange(self, payload, timeout_s):
        if self.closed or self.process.poll() is not None:
            raise OperationalError("action_outcome_unknown")
        wire, sent = canonical(payload) + b"\n", 0
        if len(wire) > MAX_BYTES:
            raise ContractError("mutation_command_cap")
        deadline = time.monotonic() + timeout_s
        selector = selectors.DefaultSelector()
        selector.register(self.process.stdin, selectors.EVENT_WRITE, "write")
        selector.register(self.process.stdout, selectors.EVENT_READ, "read")
        try:
            while b"\n" not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OperationalError("action_outcome_unknown")
                for key, _mask in selector.select(min(remaining, 0.05)):
                    if key.data == "write":
                        sent += os.write(self.process.stdin.fileno(), wire[sent:sent + 65536])
                        if sent == len(wire):
                            selector.unregister(self.process.stdin)
                    else:
                        chunk = os.read(self.process.stdout.fileno(), 65536)
                        if not chunk:
                            raise OperationalError("action_outcome_unknown")
                        self.buffer += chunk
                        if len(self.buffer) > MAX_BYTES:
                            raise OperationalError("action_outcome_unknown")
            line, self.buffer = self.buffer.split(b"\n", 1)
            if self.buffer or sent != len(wire):
                raise OperationalError("action_outcome_unknown")
            return json_object(line)
        except (OSError, ValueError, UnicodeError):
            raise OperationalError("action_outcome_unknown") from None
        finally:
            selector.close()

    def handshake(self):
        response = self._exchange({"op": "handshake", "transport_sequence": 0}, 10)
        if set(response) != {"transport_sequence", "ok", "data"} or response["ok"] is not True:
            raise OperationalError("v2_handshake_failure")
        if type(response["transport_sequence"]) is not int or response["transport_sequence"] != 0:
            raise OperationalError("v2_handshake_failure")
        return response["data"]

    def exchange(self, command, timeout_s):
        response = self._exchange(command, timeout_s)
        if (type(response.get("transport_sequence")) is not int
                or response["transport_sequence"] != command["transport_sequence"]):
            raise OperationalError("action_outcome_unknown")
        if set(response) == {"transport_sequence", "ok", "error_code"} and response["ok"] is False:
            # Rejection and mutation failures remain sanitized. Never retry.
            raise OperationalError("action_outcome_unknown" if response["error_code"] == "action_outcome_unknown"
                                   else "mutation_worker_rejected")
        if set(response) != {"transport_sequence", "ok", "data"} or response["ok"] is not True:
            raise OperationalError("action_outcome_unknown")
        return response["data"]

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=3)
        self.process.stdin.close()
        self.process.stdout.close()


class ActionBridge:
    def __init__(self, transport, *, environment, episode_id, budget):
        self.transport, self.environment, self.episode_id, self.budget = (
            transport, environment, episode_id, budget)
        self.version = VERSIONS[environment]
        self.poisoned, self.opened, self.sequence = False, False, 0
        self.revision, self.step_count, self.table_sha256 = 0, 0, ""
        self.packet, self.receipts = None, []
        handshake = transport.handshake()
        if (type(handshake) is not dict or set(handshake) != {
                "protocol_version", "session_id", "environment", "mutation_revision",
                "environment_step", "action_table_sha256", "episode_id", "automatic_retry"}
                or handshake["protocol_version"] != self.version or handshake["environment"] != environment
                or type(handshake["mutation_revision"]) is not int or handshake["mutation_revision"] != 0
                or type(handshake["environment_step"]) is not int or handshake["environment_step"] != 0
                or handshake["episode_id"] != "" or handshake["action_table_sha256"] != ""
                or handshake["automatic_retry"] is not False):
            raise OperationalError("v2_handshake_identity")
        self.session_id = uuid_string(handshake["session_id"])

    def _apply(self, op, parameters):
        if self.poisoned:
            raise OperationalError("action_outcome_unknown")
        self.budget.check_time()
        if op != "open":
            self.budget.reserve_mutation("step" if op == "step" else "invalid")
        self.sequence += 1
        command = make_command(
            version=self.version, session_id=self.session_id, episode_id=self.episode_id,
            op=op, parameters=parameters, mutation_revision=self.revision,
            environment_step=self.step_count, table_sha256=self.table_sha256, sequence=self.sequence)
        timeout = self.budget.policy["reset_timeout_s" if op == "open" else "step_timeout_s"]
        timeout = min(timeout, self.budget.policy["wall_s"]
                      - (self.budget.run.clock() - self.budget.started))
        row = {key: command[key] for key in (
            "command_id", "command_digest", "op", "episode_id", "expected_mutation_revision",
            "expected_environment_step", "expected_action_table_sha256")}
        row.update(status="sent", automatic_retry=False, protocol_version=self.version,
                   session_sha256=sha256(self.session_id.encode()),
                   transport_sequence=self.sequence)
        self.receipts.append(row)
        try:
            ack = self.transport.exchange(command, timeout)
            if (type(ack) is not dict or set(ack) != ACK_FIELDS
                    or any(ack[key] != command[key] for key in (
                        "command_id", "command_digest", "session_id", "episode_id", "op"))
                    or any(type(ack[key]) is not int for key in (
                        "before_mutation_revision", "mutation_revision",
                        "before_environment_step", "environment_step"))
                    or ack["before_mutation_revision"] != self.revision
                    or ack["mutation_revision"] != self.revision + 1
                    or ack["before_environment_step"] != self.step_count
                    or ack["environment_step"] != self.step_count + (op == "step")
                    or ack["packet_sha256"] != sha256(canonical(ack["packet"]))):
                raise ContractError("mutation_ack_identity")
            packet = ack["packet"]
            validate_packet(packet)
            if (packet["environment"] != self.environment or packet["step"] != ack["environment_step"]
                    or ack["action_table_sha256"] != action_table_hash(packet["actions"])):
                raise ContractError("mutation_ack_packet")
            if op == "open":
                if (packet["subset"] != parameters["subset"]
                        or packet["official_task_id"] != f"{parameters['subset']}:{parameters['index']:03d}"
                        or self.environment == "eb_navigation"
                        and packet["input_sha256"] != parameters["source_sha256"]):
                    raise ContractError("opened_task_identity")
            elif self.packet:
                if any(packet[key] != self.packet[key] for key in (
                        "instruction", "input_sha256", "official_task_id", "subset")):
                    raise ContractError("changed_task_identity")
                if op == "invalid" and (
                        packet["image_sha256"] != self.packet["image_sha256"]
                        or packet["official"]["invalid_actions"] != self.packet["official"]["invalid_actions"] + 1):
                    raise ContractError("invalid_ack_state")
            self.packet, self.revision, self.step_count, self.table_sha256 = (
                json_object(canonical(packet)), ack["mutation_revision"],
                ack["environment_step"], ack["action_table_sha256"])
            row.update(status="ack", mutation_revision=self.revision,
                       environment_step=self.step_count, action_table_sha256=self.table_sha256,
                       packet_sha256=ack["packet_sha256"], image_sha256=packet["image_sha256"])
            return self.packet
        except BaseException:
            row["status"] = "unknown"
            self.poisoned = True
            raise OperationalError("action_outcome_unknown") from None

    def open(self, *, subset, index, source_sha256):
        if self.opened:
            raise ContractError("bridge_already_open")
        result = self._apply("open", {"subset": subset, "index": index, "source_sha256": source_sha256})
        self.opened = True
        return result

    def step(self, action):
        if not self.opened or self.packet["official"]["done"]:
            raise ContractError("bridge_invalid_action")
        if self.environment == "eb_manipulation":
            from manipulation_protocol import vector
            action = list(vector(action))
        elif type(action) is not int or not 0 <= action < len(self.packet["actions"]):
            raise ContractError("bridge_invalid_action")
        return self._apply("step", {"action": action, "first_in_plan": True})

    def invalid(self):
        if not self.opened or self.environment in {"eb_navigation", "eb_manipulation"} or self.packet["official"]["done"]:
            raise ContractError("bridge_invalid_mutation")
        return self._apply("invalid", {})

    def close(self):
        self.transport.close()
