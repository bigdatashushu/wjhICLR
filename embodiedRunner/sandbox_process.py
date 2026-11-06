"""OS-isolated action segments with parent-enforced resource monitoring."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import selectors
import signal
import subprocess
import sys
import tempfile
import time

from action_contract import decode_action_payload, validate_action_payload
from action_tools import ResultStore, AUTH_FIELDS
from protocol import ContractError, OperationalError, canonical, json_object, sha256
from sandbox_wire import VERSION, envelope_dict, pack_prepared, pack_snapshot

ROOT = Path(__file__).resolve().parent
DEFAULT_LIMITS = {
    "wall_s": 10, "cpu_s": 10, "memory_bytes": 1024**3,
    "output_bytes": 65536, "tool_calls": 12, "reads": 3}
ERROR_CODES = frozenset({
    None, "timeout", "oom", "output_limit", "tool_budget", "violation_syntax",
    "violation_runtime", "tool_contract", "evidence_rejected", "execution_worker_failure",
    "execution_result_cap", "worker_exit", "protocol_failure", "fallthrough"})


def minimal_environment(temp_dir):
    return {"PATH": "/usr/bin:/bin", "TMPDIR": str(temp_dir),
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
            "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def sandbox_command(temp_dir, *, python=sys.executable):
    bootstrap = ("import sys; sys.path[:0]=" + repr([str(ROOT), str(ROOT.parent / "src")])
                 + "; from sandbox_worker import main; main()")
    interpreter = [str(python), "-I", "-B", "-c", bootstrap]
    if platform.system() == "Darwin":
        if not Path("/usr/lib/libsandbox.dylib").exists():
            # Some macOS dylibs live solely in the shared cache.
            try:
                ctypes.CDLL("/usr/lib/libsandbox.dylib")
            except OSError:
                raise OperationalError("sandbox_unavailable") from None
        # sandbox-exec + sandbox_init cannot be stacked on this host (EPERM).
        # The worker applies the same OS seatbelt once, after trusted imports;
        # source remains inert JSON until that succeeds. No generated program
        # runs during bootstrap, and a failed seal produces no submission.
        return interpreter
    if platform.system() == "Linux":
        from sandbox_linux import linux_command
        return linux_command(python, bootstrap)
    raise OperationalError("sandbox_platform_requires_preflight")


class _Rusage(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_ubyte * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "utime", "stime", "pkg_idle_wakeups", "interrupt_wakeups", "pageins",
            "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
            "proc_exit_abstime")]


def resident_memory(pid):
    if platform.system() == "Darwin":
        library = ctypes.CDLL("/usr/lib/libproc.dylib")
        usage = _Rusage()
        fn = library.proc_pid_rusage
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        if fn(pid, 0, ctypes.byref(usage)) != 0:
            return None
        return max(usage.resident_size, usage.phys_footprint)
    if platform.system() == "Linux":
        # Popen owns bwrap, whose Python child lives in a new PID namespace.
        # Account the owned descendant tree, not just the small bwrap process.
        pending, seen, total, observed = [pid], set(), 0, False
        while pending and len(seen) < 64:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            try:
                rows = Path(f"/proc/{current}/status").read_text().splitlines()
                values = [int(row.split()[1]) * 1024 for row in rows
                          if row.startswith(("VmRSS:", "VmHWM:"))]
                if values:
                    total += max(values)
                    observed = True
                pending.extend(int(child) for child in
                               Path(f"/proc/{current}/task/{current}/children").read_text().split())
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        return total if observed else None
    return None


def terminate_owned(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=3)


def run_packet(packet, *, python=sys.executable):
    """No environment/FD inheritance; no retry or persistent execution worker."""
    from sandbox_worker import validate_limits
    limits = validate_limits(packet["limits"])
    wire = canonical(packet) + b"\n"
    if len(wire) > 12_000_000:
        raise ContractError("execution_input_cap")
    started = time.monotonic()
    peak, observed_memory = 0, False
    with tempfile.TemporaryDirectory(prefix="embodied-segment-") as temp_dir:
        process = subprocess.Popen(
            sandbox_command(temp_dir, python=python), cwd=temp_dir,
            env=minimal_environment(temp_dir), close_fds=True, start_new_session=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        selector = selectors.DefaultSelector()
        # Nonblocking stdin prevents a broken/bootstrap-stuck child from holding
        # the host while a multi-MB geometry packet is written.
        os.set_blocking(process.stdin.fileno(), False)
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, "input")
        selector.register(process.stdout, selectors.EVENT_READ, "output")
        sent, output, error = 0, bytearray(), None
        try:
            while True:
                if time.monotonic() - started >= limits["wall_s"]:
                    error = "timeout"
                    break
                memory = resident_memory(process.pid)
                if memory is not None:
                    observed_memory = True
                    peak = max(peak, memory)
                    if memory > limits["memory_bytes"]:
                        error = "oom"
                        break
                for key, _mask in selector.select(0.02):
                    if key.data == "input":
                        try:
                            sent += os.write(process.stdin.fileno(), wire[sent:sent + 65536])
                        except BrokenPipeError:
                            selector.unregister(process.stdin)
                            process.stdin.close()
                        else:
                            if sent == len(wire):
                                selector.unregister(process.stdin)
                                process.stdin.close()
                    else:
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if chunk:
                            output.extend(chunk)
                            if len(output) > (1_000_000 if packet.get("capture_details") else 256_000):
                                error = "execution_result_cap"
                                break
                        else:
                            selector.unregister(process.stdout)
                if error:
                    break
                if process.poll() is not None and process.stdout not in [
                        key.fileobj for key in selector.get_map().values()]:
                    break
            if error:
                terminate_owned(process)
            else:
                process.wait(timeout=3)
        finally:
            selector.close()
            terminate_owned(process)
            if not process.stdin.closed:
                process.stdin.close()
            process.stdout.close()
    resources = {"wall_ms": int((time.monotonic() - started) * 1000),
                 "peak_observed_memory_bytes": peak, "memory_monitor_observed": observed_memory,
                 "memory_policy": "parent_poll_20ms_kill; transient overshoot possible",
                 "automatic_retry": False}
    if error:
        return {"schema": VERSION, "fatal": error, "resources": resources}
    try:
        result = json_object(bytes(output))
    except (ValueError, UnicodeError):
        return {"schema": VERSION, "fatal": "worker_exit", "resources": resources}
    return {**result, "resources": resources}


@dataclass(frozen=True)
class ExecutionResult:
    submission: object
    terminated: str | None
    error_code: str | None
    error_line: int | None
    results: tuple
    reads: tuple
    yielded_result_ids: tuple
    tool_calls: int
    reader_calls: int
    receipt: dict


def execute_isolated(source, snapshot, binding, envelope, *, prepared=None,
                     evidence_wire=None, previous_results=(), limits=None, max_actions=20,
                     python=sys.executable, capture_details=False):
    packet = {"schema": VERSION, "context": pack_snapshot(snapshot, binding),
              "envelope": envelope_dict(envelope),
              "evidence": evidence_wire if evidence_wire is not None else pack_prepared(snapshot, prepared),
              "program_source": source, "limits": dict(limits or DEFAULT_LIMITS),
              "previous_results": list(previous_results), "max_actions": max_actions}
    if capture_details:
        packet["capture_details"] = True
    output = run_packet(packet, python=python)
    if "fatal" in output:
        error = output["fatal"] if output["fatal"] in ERROR_CODES else "protocol_failure"
        return ExecutionResult(None, None, error, None, (), (), (), 0, 0,
                               {"resources": output["resources"], "automatic_retry": False})
    expected = {"schema", "snapshot_sha256", "envelope", "program_sha256", "submission",
                "terminated", "error_code", "error_line", "results", "reads",
                "yielded_result_ids", "tool_calls", "reader_calls", "tool_receipts",
                "isolation_sealed", "resources"}
    if capture_details:
        expected.add("execution_trace")
    if (set(output) != expected or output["schema"] != VERSION
            or output["snapshot_sha256"] != snapshot.snapshot_sha256
            or output["envelope"] != envelope_dict(envelope)
            or output["program_sha256"] != sha256(source.encode())
            or output["isolation_sealed"] is not True or output["error_code"] not in ERROR_CODES):
        raise ContractError("execution_output_identity")
    if capture_details:
        trace = output["execution_trace"]
        if (type(trace) is not dict or set(trace) != {"tool_calls", "stdout"}
                or type(trace["tool_calls"]) is not list
                or len(trace["tool_calls"]) != output["tool_calls"]
                or type(trace["stdout"]) is not str
                or len(trace["stdout"].encode()) > packet["limits"]["output_bytes"]):
            raise ContractError("execution_trace_contract")
    store = ResultStore(snapshot, output["results"])
    previous_store = ResultStore(snapshot, previous_results)
    if any(store.records.get(key) != row for key, row in previous_store.records.items()):
        raise ContractError("execution_previous_ledger_lost")
    if (not isinstance(output["tool_receipts"], list)
            or len(output["tool_receipts"]) != output["tool_calls"]
            or any(not isinstance(r, dict) or set(r) - AUTH_FIELDS
                   or type(r.get("allowed")) is not bool or r.get("episode_id") != binding.episode_id
                   for r in output["tool_receipts"])):
        raise ContractError("execution_authorization_count")
    for key, cap in (("tool_calls", packet["limits"]["tool_calls"]),
                     ("reader_calls", packet["limits"]["reads"])):
        if type(output[key]) is not int or not 0 <= output[key] <= cap:
            raise ContractError("execution_output_count")
    if (output["terminated"] not in {None, "actions", "yield"}
            or len(output["reads"]) != output["reader_calls"]
            or not isinstance(output["yielded_result_ids"], list)
            or any(not store.authorize(r, snapshot) for r in output["yielded_result_ids"])):
        raise ContractError("execution_output_control")
    for page in output["reads"]:
        if (page.get("schema_version") != "embodied-observation-reader/1"
                or page.get("scope") != store.scope or page.get("reason_code") != "allowed"
                or not store.authorize(page.get("result_id"), snapshot)
                or page.get("payload_sha256") != __import__("action_contract").public_hash(page["data"])
                or len(canonical(page)) > 5000):
            raise ContractError("execution_reader_identity")
    payload = None
    if output["submission"] is not None:
        if output["terminated"] != "actions" or output["error_code"] is not None:
            raise ContractError("execution_output_submission")
        payload = decode_action_payload(output["submission"])
        validate_action_payload(
            payload, snapshot, envelope, session_id=envelope.session_id, max_actions=max_actions,
            actual_result_ids=tuple(store.records), authorize_result=store.authorize)
    error = output["error_code"]
    if error is None and output["terminated"] is None:
        error = "fallthrough"
    return ExecutionResult(
        payload, output["terminated"], error, output["error_line"],
        tuple(output["results"]), tuple(output["reads"]), tuple(output["yielded_result_ids"]),
        output["tool_calls"], output["reader_calls"],
        {"program_sha256": output["program_sha256"], "snapshot_sha256": snapshot.snapshot_sha256,
         "tool_receipts": output["tool_receipts"],
         **({"execution_trace": output["execution_trace"]} if capture_details else {}),
         "isolation_sealed": True, "resources": output["resources"], "automatic_retry": False})
