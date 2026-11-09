"""Credential-free execution worker. This module has no simulator/model client."""
from __future__ import annotations

import ctypes
import importlib
import os
import platform
import resource
import sys
import uuid

from action_contract import (ActionPayload, ExecutionEnvelope, integer,
                             validate_action_payload)
from action_runtime.ast_guard import ast_guard
from action_runtime.control_profile import ActionContextView, EMBODIED_ACTIONS_PROFILE
from action_runtime.kernel import RestrictedNamespaceKernel
from action_tools import ResultStore, action_registry, public_authorization
from protocol import ContractError, canonical, json_object
from sandbox_wire import VERSION, unpack_snapshot, unpack_prepared
from skill3d.sandbox.safe_runtime import NUMERICAL_EXPORTS

PACKET_CAP = 12_000_000


def seal_macos():
    """Tighten the bootstrap seatbelt after all trusted imports/inputs load."""
    library = ctypes.CDLL("/usr/lib/libsandbox.dylib")
    init = library.sandbox_init
    init.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p)]
    init.restype = ctypes.c_int
    error = ctypes.c_char_p()
    # Seatbelt does not revoke reads through already-open regular-file FDs.
    # Keep only the three inherited pipes, never host/evidence descriptors.
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    os.closerange(3, min(int(hard), 1_048_576))
    profile = b"""(version 1)
    (allow default)
    (deny file-read* file-write* network* process-exec process-fork)
    (allow signal (target self))
    """
    if init(profile, 0, ctypes.byref(error)) != 0:
        raise ContractError("sandbox_seal_failed")


def validate_limits(limits):
    if type(limits) is not dict or set(limits) != {
            "wall_s", "cpu_s", "memory_bytes", "output_bytes", "tool_calls", "reads"}:
        raise ContractError("invalid_execution_limits")
    for key, maximum in (("wall_s", 30), ("cpu_s", 30), ("memory_bytes", 2**31),
                         ("output_bytes", 65536), ("tool_calls", 12), ("reads", 3)):
        integer(limits[key], name=key, maximum=maximum)
        if limits[key] < 1 and key not in {"tool_calls", "reads"}:
            raise ContractError("invalid_execution_limits")
    if limits["memory_bytes"] < 16 * 1024 * 1024:
        raise ContractError("invalid_execution_limits")
    return limits


class ToolLimit(BaseException):
    pass


def execute(packet):
    fields = {
            "schema", "context", "envelope", "evidence", "program_source",
            "limits", "previous_results", "max_actions"}
    if packet.get("capture_details") is True:
        fields.add("capture_details")
    if type(packet) is not dict or set(packet) != fields or packet["schema"] != VERSION:
        raise ContractError("unexpected_execution_packet")
    binding, snapshot = unpack_snapshot(packet["context"])
    envelope = ExecutionEnvelope(**packet["envelope"])
    limits = validate_limits(packet["limits"])
    integer(packet["max_actions"], name="max_actions", maximum=30)
    source = packet["program_source"]
    if not isinstance(source, str) or len(source.encode()) > 100_000:
        raise ContractError("program_source_cap")
    prepared = unpack_prepared(snapshot, packet["evidence"])
    registry, scene = action_registry(prepared, binding, segment_id=uuid.uuid4().hex)
    store = ResultStore(snapshot, packet["previous_results"], max_reads=limits["reads"])
    obs = snapshot.observation
    ctx = ActionContextView(
        obs.episode_id, snapshot.decision_id, obs.observation_revision, snapshot.snapshot_sha256,
        obs.action_table_sha256, obs.action_table, obs.instruction, obs.public_feedback,
        obs.public_memory, snapshot.capabilities, snapshot.evidence_revision, obs.view_id,
        official_objects=obs.manipulation_input.get("objects"),
        action_spec=obs.manipulation_input.get("action_spec"),
        task_variation=obs.manipulation_input.get("task_variation", ""),
        task_state=snapshot.task_state)
    calls = [0]

    def before_tool(_name):
        # Reserve before argument validation/authorization; rejected calls count.
        calls[0] += 1
        if calls[0] > limits["tool_calls"]:
            raise ToolLimit()

    kernel = RestrictedNamespaceKernel(
        registry, scene, frames=prepared.frames, profile=EMBODIED_ACTIONS_PROFILE,
        public_context=ctx, episode_id=obs.episode_id, cell_timeout_s=limits["cpu_s"],
        action_constructor=ActionPayload,
        action_validator=lambda payload: validate_action_payload(
            payload, snapshot, envelope, session_id=envelope.session_id,
            max_actions=packet["max_actions"], actual_result_ids=tuple(store.records),
            authorize_result=store.authorize),
        observation_reader=store.read, result_observer=store.observe,
        before_tool=before_tool, output_max_bytes=limits["output_bytes"])
    guard = ast_guard(source, set(registry.names()), profile=EMBODIED_ACTIONS_PROFILE)
    # Lazy library imports would require filesystem access after the seal.
    for module in NUMERICAL_EXPORTS:
        importlib.import_module(module)
    importlib.import_module("skill3d.tools.failure_guidance")
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_s"] + 1, limits["cpu_s"] + 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if platform.system() == "Darwin":
        seal_macos()
    elif platform.system() == "Linux":
        from sandbox_linux import seal_linux
        seal_linux()
    else:
        raise ContractError("unverified_execution_platform")
    base = {"schema": VERSION, "snapshot_sha256": snapshot.snapshot_sha256,
            "envelope": packet["envelope"], "program_sha256": __import__("hashlib").sha256(source.encode()).hexdigest(),
            "submission": None, "terminated": None, "error_code": None, "error_line": None,
            "results": list(store.records.values()), "reads": [], "yielded_result_ids": [],
            "tool_calls": 0, "reader_calls": 0, "tool_receipts": [], "isolation_sealed": True}
    if packet.get("capture_details"):
        base["execution_trace"] = {"tool_calls": [], "stdout": ""}
    if not guard.ok:
        return {**base, "error_code": "violation_syntax"}
    try:
        cell = kernel.run_cell(source)
    except ToolLimit:
        return {**base, "error_code": "tool_budget", "tool_calls": limits["tool_calls"],
                **({"execution_trace": {"tool_calls": [
                    c.model_dump(mode="json") for c in kernel.cell_calls], "stdout": ""}}
                   if packet.get("capture_details") else {}),
                "results": list(store.records.values()), "reads": store.reads,
                "tool_receipts": [public_authorization(r) for r in kernel.authorization_receipts],
                "reader_calls": len(store.reads)}
    submission = None
    error = cell.error_code
    if cell.submission is not None and not cell.answer_untrusted and not error:
        submission = cell.submission.to_dict()
    elif cell.answer_untrusted and error is None:
        error = "evidence_rejected"
    if cell.terminated == "yield":
        if any(not store.authorize(r, snapshot) for r in cell.yielded_result_ids):
            error = "evidence_rejected"
    return {**base, "submission": submission, "terminated": cell.terminated,
            **({"execution_trace": {"tool_calls": [
                c.model_dump(mode="json") for c in kernel.cell_calls], "stdout": cell.stdout}}
               if packet.get("capture_details") else {}),
            "error_code": error, "error_line": cell.error_line,
            "results": list(store.records.values()), "reads": store.reads,
            "yielded_result_ids": cell.yielded_result_ids[:32],
            "tool_receipts": [public_authorization(r) for r in kernel.authorization_receipts],
            "tool_calls": calls[0], "reader_calls": len(store.reads)}


def main():
    packet = {}
    try:
        line = sys.stdin.buffer.readline(PACKET_CAP + 1)
        if len(line) > PACKET_CAP:
            raise ContractError("execution_input_cap")
        packet = json_object(line)
        result = execute(packet)
        # A short allocation can finish between parent polls. Never accept a
        # submission if the kernel's retained child peak exceeds the cap.
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak * 1024 if platform.system() == "Linux" else peak
        if peak_bytes > validate_limits(packet["limits"])["memory_bytes"]:
            result = {"schema": VERSION, "fatal": "oom"}
    except BaseException:
        # Never serialize raw exceptions, stdout, paths, source or tracebacks.
        result = {"schema": VERSION, "fatal": "execution_worker_failure"}
    wire = canonical(result)
    if len(wire) > (1_000_000 if packet.get("capture_details") else 256_000):
        wire = canonical({"schema": VERSION, "fatal": "execution_result_cap"})
    sys.stdout.buffer.write(wire + b"\n")
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
