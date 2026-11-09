"""One outstanding simulator command; no resending after a lost response."""
import json
import os
from pathlib import Path
import selectors
import shlex
import signal
import subprocess
import time

from . import VERSION
from .sources import OperationalError, ROOT, canonical, implementation_hashes


def child_environment():
    # A purpose-built allowlist prevents API credentials reaching Python or SSH.
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    for name in ("HOME", "LANG", "LC_ALL", "SSH_AUTH_SOCK"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def command(config, task, *, work=None, probe=False):
    runtime = config["runtime"]
    args = [runtime["python"][task.environment], "-B",
            str(Path(runtime["code_root"]) / "official_direct/worker.py"),
            "--source-root", runtime["source_root"], "--environment", task.environment,
            "--subset", task.subset, "--index", str(task.index),
            "--seed", str(runtime["environment_seed"])]
    gpu = runtime["gpu"][task.environment]
    if gpu is not None:
        args += ["--gpu", str(gpu)]
    for key in ("manipulation_catalog", "manipulation_dataset", "gpu_min_free_mb"):
        if key in runtime:
            args += ["--" + key.replace("_", "-"), str(runtime[key])]
    if probe:
        args += ["--probe"]
    else:
        args += ["--work-dir", str(work)]
    if runtime["transport"] == "local":
        return args
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
           "-o", "UpdateHostKeys=no", "-o", "ConnectTimeout=10",
           "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]
    socket = ROOT / "checks/ssh4090"
    if socket.exists():
        ssh += ["-o", "ControlMaster=no", "-o", "ControlPath=" + str(socket)]
    return ssh + [runtime["ssh_target"], shlex.join(args)]


def preflight(config, task, sources):
    """Package/asset/GPU checks only. Does not import or launch a simulator."""
    try:
        completed = subprocess.run(command(config, task, probe=True),
                                   cwd=ROOT / "official_direct", env=child_environment(),
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   timeout=config["runtime"]["timeout_s"], check=False)
        if completed.returncode != 0:
            raise OperationalError("worker_preflight_process_error")
        report = json.loads(completed.stdout)
        if (report.get("simulator_launches") != 0 or report.get("model_requests") != 0
                or report.get("source_hashes") != sources.hashes
                or report.get("implementation_hashes") != implementation_hashes()):
            raise OperationalError("worker_preflight_identity_mismatch")
        if not report["runtime_ready"]:
            raise OperationalError("worker_runtime_not_ready")
        if not report["gpu"].get("admissible", report["gpu"]["idle"]):
            raise OperationalError("configured_gpu_headroom_insufficient")
        return report
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        raise OperationalError("worker_preflight_error") from None


class Session:
    def __init__(self, config, task, directory, journal, sources, run_id):
        self.timeout = config["runtime"]["timeout_s"]
        self.sources, self.sequence, self.closed, self.poisoned = sources, 0, False, False
        report = preflight(config, task, sources)
        journal.append({"type": "runtime_preflight", "report": report})
        work = Path(config["runtime"]["work_root"]) / run_id / "workers" / task.directory
        self.process = subprocess.Popen(
            command(config, task, work=work), cwd=directory, env=child_environment(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.buffer = b""

    def request(self, op, **parameters):
        if self.closed or self.poisoned:
            raise OperationalError("worker_connection_closed")
        self.sequence += 1
        if op == "open":
            parameters["source_hashes"] = self.sources.hashes
            parameters["implementation_hashes"] = implementation_hashes()
        payload = canonical({"id": self.sequence, "op": op, **parameters}) + b"\n"
        sent, deadline = 0, time.monotonic() + self.timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE, "write")
                selector.register(self.process.stdout, selectors.EVENT_READ, "read")
                while b"\n" not in self.buffer:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise OperationalError("worker_timeout_outcome_unknown")
                    for key, _ in selector.select(min(remaining, 1)):
                        if key.data == "write":
                            sent += os.write(key.fileobj.fileno(), payload[sent:sent + 65536])
                            if sent == len(payload):
                                selector.unregister(self.process.stdin)
                        else:
                            chunk = os.read(key.fileobj.fileno(), 65536)
                            if not chunk:
                                raise OperationalError("worker_closed_outcome_unknown")
                            self.buffer += chunk
                            if len(self.buffer) > 32 * 1024**2:
                                raise OperationalError("worker_message_too_large")
            line, self.buffer = self.buffer.split(b"\n", 1)
            value = json.loads(line)
            if (self.buffer or sent != len(payload) or value.get("id") != self.sequence
                    or value.get("version") != VERSION):
                raise OperationalError("worker_response_identity_mismatch")
            if value.get("ok") is not True:
                raise OperationalError("official_environment_error")
            return value["data"]
        except (OSError, ValueError, KeyError, TypeError, OperationalError) as exc:
            self.poisoned = True
            if isinstance(exc, OperationalError):
                raise
            raise OperationalError("worker_protocol_error") from None

    def close(self):
        if self.closed:
            return
        cleanup_error = None
        if not self.poisoned:
            try:
                self.request("close")
            except OperationalError as exc:
                cleanup_error = exc
        self.closed = True
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if self.process.poll() is None:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=5)
        finally:
            self.process.stdout.close()
        if cleanup_error is not None:
            raise cleanup_error
