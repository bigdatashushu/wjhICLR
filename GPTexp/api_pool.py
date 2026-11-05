"""Cross-process request accounting and endpoint limits, without key switching."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import re
import time

from .traces import digest, read, write


class PoolUnavailable(RuntimeError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def resolve_key(api):
    name = api["api_key_env"]
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        raise ValueError("invalid_credential_environment_name")
    key = os.environ.get(name, "")
    if not key and api.get("credentials_file"):
        path = Path(api["credentials_file"])
        if path.stat().st_mode & 0o077:
            raise ValueError("credential_file_requires_600")
        values = {}
        comment_keys = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#"):
                match = re.fullmatch(r"#\s*(sk-[A-Za-z0-9_-]+)\s*", line)
                if match:
                    comment_keys.append(match.group(1))
                continue
            if not line:
                continue
            field, separator, value = line.partition("=")
            if not separator or field.strip() in values:
                raise ValueError("invalid_credential_file")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[field.strip()] = value
        if "credentials_comment_index" in api:
            index = api["credentials_comment_index"]
            if type(index) is not int or not 0 <= index < len(comment_keys):
                raise ValueError("invalid_comment_credential_index")
            key = comment_keys[index]
        else:
            key = values.get(name) or values.get("XCODE_API_KEY", "")
    if not key or any(c.isspace() or ord(c) < 32 for c in key):
        raise ValueError("missing_or_invalid_api_credential")
    return key


class ApiPool:
    def __init__(self, root, config):
        self.root = Path(root)
        self.config = config
        if (config["max_requests"] < 1 or config["max_inflight"] < 1
                or config["queue_timeout_seconds"] <= 0):
            raise ValueError("invalid_pool_limits")
        policy = config.get("service_failure_policy", {})
        if policy and (policy.get("mode") != "cooldown_continue"
                       or policy.get("cooldown_seconds", 0) <= 0
                       or not isinstance(policy.get("transient_http_statuses"), list)):
            raise ValueError("invalid_service_failure_policy")
        ids = set()
        for api in config["apis"]:
            if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", api["id"])
                    or api["id"] in ids or api["max_inflight"] < 1
                    or api["requests_per_second"] < 0 or api["timeout_seconds"] <= 0
                    or api["protocol"] != "responses" or api["model"] != config["model"]):
                raise ValueError("invalid_api_configuration")
            ids.add(api["id"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        with self._locked():
            if self.state_path.exists():
                self._state()
            else:
                write(self.state_path, {"policy_identity": digest(config), "requests": [], "rates": {}})

    @contextmanager
    def _locked(self):
        with (self.root / "state.lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _state(self):
        state = read(self.state_path)
        if state["policy_identity"] != digest(self.config):
            raise ValueError("pool_configuration_changed")
        return state

    def paused(self, api):
        return (self.root / f"paused-{api['id']}.json").exists()

    def pause(self, api, reason):
        write(self.root / f"paused-{api['id']}.json", {
            "api": api["id"], "reason": reason, "time": time.time(),
            "automatic_retry": False})

    def cooling(self, api):
        with self._locked():
            row = self._state().get("cooldowns", {}).get(api["id"], {})
        return time.time() < row.get("until", 0)

    def transient(self, *, reason_code=None, http_status=None):
        policy = self.config.get("service_failure_policy", {})
        if policy.get("mode") != "cooldown_continue":
            return False
        return (reason_code == "http_error"
                and http_status in policy["transient_http_statuses"]
                or reason_code in policy.get("transient_reason_codes", []))

    def service_failure(self, api, receipt):
        if not self.transient(reason_code=receipt.get("reason_code"),
                              http_status=receipt.get("http_status")):
            self.pause(api, receipt["error_type"])
            return "permanent_pause"
        now = time.time()
        with self._locked():
            state = self._state()
            cooldown = {"api": api["id"], "request_id": receipt["request_id"],
                        "at": now, "until": now + self.config["service_failure_policy"]["cooldown_seconds"],
                        "reason_code": receipt["reason_code"], "http_status": receipt["http_status"],
                        "automatic_retry": False, "next_action": "continue_subsequent_questions"}
            state.setdefault("cooldowns", {})[api["id"]] = cooldown
            state.setdefault("service_failures", []).append(cooldown)
            write(self.state_path, state)
        return "cooldown_continue"

    def _slot(self, prefix, count):
        for index in range(count):
            stream = (self.root / f"{prefix}-{index:04d}.lock").open("a+b")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return stream
            except BlockingIOError:
                stream.close()
        return None

    @contextmanager
    def request(self, api, operation, ordinal, payload_sha256):
        deadline = time.monotonic() + self.config["queue_timeout_seconds"]
        api_slot = global_slot = None
        admitted = False
        while not admitted:
            if self.paused(api):
                raise PoolUnavailable("api_paused")
            if time.monotonic() >= deadline:
                raise PoolUnavailable("queue_timeout")
            api_slot = self._slot("api-" + api["id"], api["max_inflight"])
            if api_slot is not None:
                global_slot = self._slot("global", self.config["max_inflight"])
            if global_slot is not None:
                try:
                    with self._locked():
                        state = self._state()
                        count = len(state["requests"])
                        if count >= self.config["max_requests"]:
                            raise PoolUnavailable("global_budget_exhausted")
                        next_start = max(state["rates"].get(api["id"], 0),
                                         state.get("cooldowns", {}).get(api["id"], {}).get("until", 0))
                        now = time.time()
                        if now >= next_start:
                            rate = api["requests_per_second"]
                            state["rates"][api["id"]] = now + (1 / rate if rate else 0)
                            request_id = count
                            state["requests"].append({
                                "request_id": request_id, "api": api["id"], "operation": operation,
                                "ordinal": ordinal, "payload_sha256": payload_sha256,
                                "status": "started", "started": now})
                            write(self.state_path, state)
                            admitted = True
                except BaseException:
                    global_slot.close()
                    api_slot.close()
                    raise
            if not admitted:
                if global_slot is not None:
                    global_slot.close()
                if api_slot is not None:
                    api_slot.close()
                api_slot = global_slot = None
                time.sleep(0.1)
        receipt = {"request_id": request_id, "error_type": None, "http_status": None,
                   "reason_code": None}
        try:
            yield receipt
        except BaseException as exc:
            receipt["error_type"] = type(exc).__name__
            receipt["reason_code"] = getattr(exc, "code", None)
            receipt["http_status"] = getattr(exc, "http_status", receipt["http_status"])
            receipt["failure_policy"] = self.service_failure(api, receipt)
            raise
        finally:
            try:
                with self._locked():
                    state = self._state()
                    state["requests"][request_id].update({
                        "status": "error" if receipt["error_type"] else "completed",
                        "finished": time.time(), "error_type": receipt["error_type"],
                        "http_status": receipt["http_status"], "reason_code": receipt["reason_code"],
                        "failure_policy": receipt.get("failure_policy")})
                    write(self.state_path, state)
            finally:
                global_slot.close()
                api_slot.close()

    def summary(self):
        from collections import Counter
        with self._locked():
            rows = self._state()["requests"]
        counts = Counter((row["api"], row["status"]) for row in rows)
        return [{"api": api, "status": status, "requests": count}
                for (api, status), count in sorted(counts.items())]
