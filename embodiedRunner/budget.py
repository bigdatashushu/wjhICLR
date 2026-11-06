"""One task-local permit source for every role, action and execution segment."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
import uuid

from action_contract import freeze, integer, plain, public_hash
from protocol import ContractError, OperationalError

VERSION = "embodied-budget/1"
STAGES = {
    "synthesis": ("qwen", "models"),
    "repair": ("qwen", "models"),
    "category_parse": ("qwen", "models"),
    "category_review": ("reviews", "models"),
    "dino": ("dino",), "sam": ("sam",), "moge": ("moge",),
}
QWEN_STAGES = frozenset({"synthesis", "repair", "category_parse"})


class BudgetExhausted(OperationalError):
    def __init__(self, dimension):
        self.dimension = dimension
        super().__init__("budget_exhausted:" + dimension)


def policy_for(environment):
    if environment not in {"eb_navigation", "eb_alfred", "eb_habitat", "eb_manipulation"}:
        raise ContractError("invalid_budget_environment")
    nav = environment == "eb_navigation"
    manip = environment == "eb_manipulation"
    steps = 15 if manip else 20 if nav else 30
    return freeze({
        "schema": VERSION, "steps": steps, "invalid": 0 if nav or manip else 10,
        "qwen": 30 if manip else 20 if nav else 40, "models": 45 if manip else 40 if nav else 70,
        "reviews": steps, "dino": steps, "sam": steps, "moge": steps,
        "tools": 12 * steps, "reads": 6 * steps, "segments": 3 * steps,
        "synthesis_per_decision": 3, "tools_per_decision": 12,
        "reads_per_decision": 6, "segments_per_decision": 3,
        "reads_per_segment": 3, "wall_s": 600 if nav else 900,
        "qwen_token_soft_cap": 150000 if nav else 250000,
        "completion_tokens": 4096, "model_timeout_s": 120,
        "perception_timeout_s": 60, "step_timeout_s": 120, "reset_timeout_s": 180,
        "segment_wall_s": 10, "segment_cpu_s": 10,
        "segment_memory_bytes": 1024**3, "segment_output_bytes": 65536,
        "max_inflight": 1, "automatic_retry": False,
    })


def numeric_usage(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ContractError("invalid_usage")
    result = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if name in value and value[name] is not None:
            result[name] = integer(value[name], name="usage_" + name)
    if all(name in result for name in ("prompt_tokens", "completion_tokens", "total_tokens")):
        if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
            raise ContractError("usage_total_mismatch")
    return result


class RunBudget:
    """Fixed denominator, serial callers. An arm owns a separate instance/root.

    A Qwen RequestGovernor may additionally persist paid permits. Other model
    and GPU service calls use this same run/episode counter before invocation.
    Resuming an existing run requires a separately audited recovery entry.
    """
    def __init__(self, run_id, environments, *, qwen_governor=None, clock=time.monotonic):
        if not environments:
            raise ContractError("empty_budget_denominator")
        self.run_id, self.clock = run_id, clock
        self.environments = tuple(environments)
        self.caps = Counter()
        for environment in environments:
            policy = policy_for(environment)
            self.caps.update({key: policy[key] for key in (
                "steps", "invalid", "qwen", "models", "reviews", "dino", "sam",
                "moge", "tools", "reads", "segments")})
        self.counts, self.episodes = Counter(), {}
        self.lock, self.inflight = threading.RLock(), False
        self.qwen_governor = qwen_governor
        if qwen_governor is not None and (
                qwen_governor.config.max_requests != self.caps["qwen"]
                or qwen_governor.config.max_inflight != 1):
            raise ContractError("governor_policy_mismatch")

    def episode(self, episode_id, environment):
        with self.lock:
            if (episode_id in self.episodes
                    or len(self.episodes) >= len(self.environments)
                    or sum(b.environment == environment for b in self.episodes.values())
                    >= self.environments.count(environment)):
                raise ContractError("budget_episode_identity")
            budget = EpisodeBudget(self, episode_id, environment)
            self.episodes[episode_id] = budget
            return budget

    def summary(self):
        return {"schema": VERSION, "run_id": self.run_id,
                "planned_episodes": len(self.environments),
                "caps": dict(self.caps), "reserved": dict(self.counts),
                "automatic_retry": False, "max_inflight": 1}


@dataclass
class SegmentPermit:
    budget: object
    decision_id: str
    tools: int
    reads: int
    limits: dict
    settled: bool = False

    def settle(self, result):
        """Release unused tranches only after a verified complete worker reply.

        Killed/unknown execution keeps the full tranche charged. Source/hash,
        receipt counts and scopes must have passed execute_isolated first.
        """
        if self.settled:
            raise ContractError("segment_already_settled")
        self.settled = True
        if result.receipt.get("isolation_sealed") is not True:
            return
        tools = integer(result.tool_calls, name="used_tools", maximum=self.tools)
        reads = integer(result.reader_calls, name="used_reads", maximum=self.reads)
        with self.budget.run.lock:
            self.budget._release("tools", self.tools - tools, self.decision_id)
            self.budget._release("reads", self.reads - reads, self.decision_id)


class EpisodeBudget:
    def __init__(self, run, episode_id, environment):
        self.run, self.episode_id, self.environment = run, episode_id, environment
        self.policy = policy_for(environment)
        self.sha256 = public_hash(self.policy)
        self.counts, self.decisions = Counter(), {}
        self.receipts, self.service_views = [], set()
        self.started = run.clock()
        self.qwen_tokens = 0

    def check_time(self):
        if self.run.clock() - self.started >= self.policy["wall_s"]:
            raise BudgetExhausted("wall")

    def _reserve(self, dimensions, *, decision_id="", synthesis=False):
        self.check_time()
        if self.qwen_tokens >= self.policy["qwen_token_soft_cap"] and "qwen" in dimensions:
            raise BudgetExhausted("qwen_tokens_soft")
        dec = self.decisions.setdefault(decision_id, Counter()) if decision_id else Counter()
        for dimension, amount in dimensions.items():
            integer(amount, name="reservation")
            if self.counts[dimension] + amount > self.policy[dimension]:
                raise BudgetExhausted(dimension)
            if self.run.counts[dimension] + amount > self.run.caps[dimension]:
                raise BudgetExhausted("run_" + dimension)
            if decision_id and dimension in {"tools", "reads", "segments"}:
                if dec[dimension] + amount > self.policy[dimension + "_per_decision"]:
                    raise BudgetExhausted("decision_" + dimension)
        if synthesis:
            if not decision_id:
                raise ContractError("synthesis_requires_decision")
            if dec["synthesis"] >= self.policy["synthesis_per_decision"]:
                raise BudgetExhausted("decision_synthesis")
        for dimension, amount in dimensions.items():
            self.counts[dimension] += amount
            self.run.counts[dimension] += amount
            if decision_id:
                dec[dimension] += amount
        if synthesis:
            dec["synthesis"] += 1

    def _release(self, dimension, amount, decision_id):
        self.counts[dimension] -= amount
        self.run.counts[dimension] -= amount
        self.decisions[decision_id][dimension] -= amount

    @contextmanager
    def request(self, stage, *, decision_id="", view_id=""):
        """Use around the actual individual service invocation, never a batch."""
        if stage not in STAGES:
            raise ContractError("unknown_request_stage")
        service_view = (stage, view_id)
        with self.run.lock:
            if self.run.inflight:
                raise ContractError("concurrent_or_nested_service")
            if stage in {"dino", "sam", "moge"}:
                if not view_id or service_view in self.service_views:
                    raise ContractError("duplicate_per_view_service")
            self._reserve({key: 1 for key in STAGES[stage]}, decision_id=decision_id,
                          synthesis=stage in {"synthesis", "repair"})
            self.run.inflight = True
            if stage in {"dino", "sam", "moge"}:
                self.service_views.add(service_view)
            row = {"permit_id": str(uuid.uuid4()), "episode_id": self.episode_id,
                   "decision_id": decision_id, "stage": stage, "status": "started",
                   "view_sha256": public_hash(view_id) if view_id else "",
                   "usage": {}, "usage_missing": stage in QWEN_STAGES or stage == "category_review",
                   "automatic_retry": False}
            self.receipts.append(row)
        usage_slot = {"usage": None}
        try:
            governor = self.run.qwen_governor if stage in QWEN_STAGES else None
            if governor is None:
                yield usage_slot
            else:
                from skill3d.synthesis.request_governor import RequestBudgetExhausted
                try:
                    with governor.request("solver", {
                            "run_identity": public_hash(self.run.run_id),
                            "slot_key": public_hash(self.episode_id)}) as governed:
                        yield usage_slot
                        governed["usage"] = numeric_usage(usage_slot["usage"])
                except RequestBudgetExhausted:
                    raise BudgetExhausted("run_governor") from None
            usage = numeric_usage(usage_slot["usage"])
            row.update(status="completed", usage=usage,
                       usage_missing=row["usage_missing"] and "total_tokens" not in usage)
            if stage in QWEN_STAGES:
                self.qwen_tokens += usage.get("total_tokens",
                                             usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
        except BaseException:
            row["status"] = "error"
            # A completed model response can fail during parsing/assembly.
            # Preserve paid numeric usage even though the surrounding call failed.
            try:
                usage = numeric_usage(usage_slot["usage"])
            except ContractError:
                usage = {}
            row.update(usage=usage,
                       usage_missing=row["usage_missing"] and "total_tokens" not in usage)
            if stage in QWEN_STAGES:
                self.qwen_tokens += usage.get(
                    "total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
            # Deliberately no exception/prompt/provider text in audit.
            raise
        finally:
            self.run.inflight = False

    def reserve_mutation(self, kind):
        if kind not in {"step", "invalid"}:
            raise ContractError("unknown_mutation_budget")
        with self.run.lock:
            self._reserve({"steps" if kind == "step" else "invalid": 1})

    def reserve_feedback_read(self, decision_id):
        with self.run.lock:
            self._reserve({"reads": 1}, decision_id=decision_id)

    def reserve_segment(self, decision_id):
        with self.run.lock:
            dec = self.decisions.get(decision_id, Counter())
            tools = min(self.policy["tools_per_decision"] - dec["tools"],
                        self.policy["tools"] - self.counts["tools"],
                        self.run.caps["tools"] - self.run.counts["tools"])
            reads = min(self.policy["reads_per_segment"], self.policy["reads_per_decision"] - dec["reads"],
                        self.policy["reads"] - self.counts["reads"],
                        self.run.caps["reads"] - self.run.counts["reads"])
            self._reserve({"segments": 1, "tools": tools, "reads": reads}, decision_id=decision_id)
        remaining = int(self.policy["wall_s"] - (self.run.clock() - self.started))
        if remaining < 1:
            raise BudgetExhausted("wall")
        limits = {"wall_s": min(remaining, self.policy["segment_wall_s"]),
                  "cpu_s": self.policy["segment_cpu_s"], "memory_bytes": self.policy["segment_memory_bytes"],
                  "output_bytes": self.policy["segment_output_bytes"], "tool_calls": tools, "reads": reads}
        return SegmentPermit(self, decision_id, tools, reads, limits)

    def summary(self):
        return {"schema": VERSION, "episode_id": self.episode_id,
                "policy": plain(self.policy), "policy_sha256": self.sha256,
                "reserved": dict(self.counts),
                "decisions": {key: dict(value) for key, value in self.decisions.items()},
                "requests": list(self.receipts), "qwen_tokens_observed": self.qwen_tokens,
                "token_policy": "observed soft threshold; final call may overshoot",
                "automatic_retry": False}
