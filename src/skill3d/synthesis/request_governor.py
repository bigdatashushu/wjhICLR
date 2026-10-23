"""Run-local request limits shared by solver, reviser and paired gate processes."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import time


class RequestBudgetExhausted(BaseException):
    """Cancel the run when its frozen paid-request budget is exhausted.

    Like KeyboardInterrupt this bypasses model-error recovery: budget
    exhaustion must never become a method failure or an admission rejection.
    The owning scheduler catches BaseException and drains current work.
    """


@dataclass(frozen=True)
class RequestGovernorConfig:
    max_requests: int
    max_inflight: int
    requests_per_second: float = 0.0

    def __post_init__(self):
        if self.max_requests < 1 or self.max_inflight < 1:
            raise ValueError("request budget and inflight cap must be positive")
        if not math.isfinite(self.requests_per_second) or self.requests_per_second < 0:
            raise ValueError("request rate must be finite and non-negative")


def _atomic(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RequestGovernor:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.config = config
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        with self._locked():
            if self.state_path.exists():
                self._state()
            else:
                _atomic(self.state_path, {
                    "policy": asdict(config), "requests": [], "next_start": 0.0,
                    "peak_inflight": 0,
                })

    @contextmanager
    def _locked(self):
        with (self.root / "state.lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def _state(self):
        state = json.loads(self.state_path.read_text())
        if state["policy"] != asdict(self.config):
            raise ValueError("request policy changed during resume")
        return state

    def _slot(self):
        for index in range(self.config.max_inflight):
            stream = (self.root / f"slot-{index:03d}.lock").open("a+b")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            return index, stream
        return None, None

    @contextmanager
    def request(self, category, context):
        if category not in {"solver", "reviser", "gate"}:
            raise ValueError("unknown request category")
        # No prompt, label, URL, environment or exception text is persisted.
        safe_context = {key: str(context[key]) for key in (
            "variant_id", "slot_key", "campaign_id", "run_identity")
            if key in context}
        permit = None
        while permit is None:
            delay = 0.05
            with self._locked():
                state = self._state()
                if len(state["requests"]) >= self.config.max_requests:
                    raise RequestBudgetExhausted("global request budget exhausted")
                now = time.time()
                if now >= state["next_start"]:
                    slot, permit = self._slot()
                    if permit is not None:
                        request_id = len(state["requests"])
                        permit.seek(0)
                        permit.truncate()
                        permit.write(str(request_id).encode("ascii"))
                        permit.flush()
                        state["requests"].append({
                            "request_id": request_id, "category": category,
                            "context": safe_context, "slot": slot,
                            "status": "started", "started_at": now,
                        })
                        interval = (1 / self.config.requests_per_second
                                    if self.config.requests_per_second else 0)
                        state["next_start"] = now + interval
                        # Probe OS locks to count live permits. Locks are
                        # released by the OS even if a process dies mid-call.
                        live = 0
                        for index in range(self.config.max_inflight):
                            if index == slot:
                                live += 1
                                continue
                            with (self.root / f"slot-{index:03d}.lock").open("a+b") as probe:
                                try:
                                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                except BlockingIOError:
                                    live += 1
                        state["peak_inflight"] = max(state["peak_inflight"], live)
                        _atomic(self.state_path, state)
                else:
                    delay = min(0.5, max(0.01, state["next_start"] - now))
            if permit is None:
                time.sleep(delay)
        result = {"usage": {}, "exception_type": None}
        try:
            yield result
        except BaseException as exc:
            result["exception_type"] = type(exc).__name__
            raise
        finally:
            try:
                with self._locked():
                    state = self._state()
                    row = state["requests"][request_id]
                    row.update({
                        "status": "error" if result["exception_type"] else "completed",
                        "completed_at": time.time(),
                        "exception_type": result["exception_type"],
                        "usage": result["usage"],
                    })
                    _atomic(self.state_path, state)
            finally:
                permit.close()

    def summary(self):
        with self._locked():
            state = self._state()
            active_ids = set()
            for index in range(self.config.max_inflight):
                with (self.root / f"slot-{index:03d}.lock").open("a+b") as probe:
                    try:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        probe.seek(0)
                        raw = probe.read().decode("ascii")
                        if raw.isdigit():
                            active_ids.add(int(raw))
            active_ids &= {r["request_id"] for r in state["requests"]
                           if r["status"] == "started"}
        categories = {}
        usage = {}
        for row in state["requests"]:
            categories[row["category"]] = categories.get(row["category"], 0) + 1
            for key, value in row.get("usage", {}).items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
        return {
            "policy": state["policy"], "reserved_requests": len(state["requests"]),
            "completed_requests": sum(r["status"] == "completed" for r in state["requests"]),
            "error_requests": sum(r["status"] == "error" for r in state["requests"]),
            "active_requests": len(active_ids),
            "unknown_requests": sum(
                r["status"] == "started" and r["request_id"] not in active_ids
                for r in state["requests"]),
            "peak_inflight": state["peak_inflight"], "categories": categories, "usage": usage,
        }


class GovernedClient:
    def __init__(self, client, governor, *, category, context=None):
        self.client = client
        self.governor = governor
        self.category = category
        self.context = dict(context or {})

    def __getattr__(self, name):
        return getattr(self.client, name)

    def chat(self, *args, **kwargs):
        with self.governor.request(self.category, self.context) as receipt:
            text = self.client.chat(*args, **kwargs)
            usage = getattr(self.client, "last_usage", {}) or {}
            receipt["usage"] = {
                key: value for key, value in usage.items()
                if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                and isinstance(value, (int, float)) and math.isfinite(value)
            }
            return text
