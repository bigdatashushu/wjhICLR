"""Bounded NDJSON transport to an isolated official simulator process."""
from __future__ import annotations

import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

from protocol import (VERSION, OperationalError, action_id, canonical,
                      decode_transition, json_object)


def simulator_environment(root):
    root = Path(root).resolve()
    environment = {
        key: value for key, value in os.environ.items()
        if not any(part in key.upper() for part in
                   ("API_KEY", "TOKEN", "PASSWORD", "SECRET", "CREDENTIAL", "ENDPOINT"))
        and not key.upper().endswith(("_KEY", "_AUTH"))
        and key not in {"remote_url", "server_url", "OPENAI_BASE_URL", "DASHSCOPE_BASE_URL",
                        "PYTHONPATH", "PYTHONSTARTUP"}
    }
    for name, folder in (
        ("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache/xdg"),
        ("HF_HOME", "cache/huggingface"), ("TORCH_HOME", "cache/torch"),
        ("MPLCONFIGDIR", "cache/matplotlib"),
        ("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
        ("XDG_STATE_HOME", "state"),
    ):
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


class SimulatorBridge:
    def __init__(self, command, *, work_dir, resolution=500, timeout_s=120):
        self.resolution, self.timeout_s = resolution, timeout_s
        self.sequence, self.buffer, self.closed = 0, b"", False
        self.process = subprocess.Popen(
            command, cwd=str(work_dir), env=simulator_environment(work_dir),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def _line(self):
        deadline = time.monotonic() + self.timeout_s
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise OperationalError("simulator_timeout")
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise OperationalError("simulator_closed")
            self.buffer += chunk
            if len(self.buffer) > 12_000_000:
                raise OperationalError("simulator_message_too_large")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("utf-8")

    def request(self, op, **parameters):
        if self.closed:
            raise OperationalError("simulator_closed")
        self.sequence += 1
        try:
            self.process.stdin.write(canonical({
                "id": self.sequence, "op": op, **parameters}) + b"\n")
            self.process.stdin.flush()
            response = json_object(self._line())
        except (OSError, ValueError, UnicodeError):
            raise OperationalError("simulator_protocol_error") from None
        if response.get("id") != self.sequence or response.get("version") != VERSION:
            raise OperationalError("simulator_response_identity_mismatch")
        if response.get("ok") is not True:
            # The child returns only locally generated codes, not exception text.
            code = response.get("error_code")
            if code not in {"missing_simulator_dependencies", "unsupported_platform",
                            "official_source_changed", "environment_error",
                            "invalid_command", "invalid_action_id", "initialization_failed"}:
                code = "simulator_error"
            raise OperationalError(code)
        return response["data"]

    def open(self, task):
        packet = self.request("open", subset=task.subset, index=task.index,
                              input_sha256=task.input_sha256)
        return self._transition(packet)

    def step(self, action, *, first_in_plan):
        packet = self.request("step", action=action_id(action),
                              first_in_plan=bool(first_in_plan))
        return self._transition(packet)

    def _transition(self, packet):
        try:
            return decode_transition(packet, resolution=self.resolution)
        except (ValueError, KeyError, TypeError):
            raise OperationalError("simulator_observation_error") from None

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.selector.close()
        try:
            self.process.stdin.close()
            self.process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=3)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if self.process.poll() is None:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=3)
        finally:
            self.process.stdout.close()
