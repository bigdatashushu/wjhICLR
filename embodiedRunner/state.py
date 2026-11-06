"""Host-owned lifecycle, independent observation revisions, append-only history."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import uuid

from action_contract import (DecisionSnapshot, EpisodeBinding, PublicObservation,
                             public_hash, validate_memory)
from protocol import ContractError

TRANSITIONS = {
    "opening": {"observed", "aborted"},
    "observed": {"preparing", "finished", "aborted"},
    "preparing": {"synthesizing", "finished", "aborted"},
    "synthesizing": {"executing", "validating", "observed", "finished", "aborted"},
    "executing": {"synthesizing", "validating", "finished", "aborted"},
    "validating": {"applying", "observed", "synthesizing", "finished", "aborted"},
    "applying": {"observed", "finished", "aborted"},
    "finished": set(), "aborted": set(),
}


@dataclass
class EpisodeState:
    binding: EpisodeBinding
    session_id: str
    method_sha256: str
    budget_sha256: str
    phase: str = "opening"
    observation: PublicObservation | None = None
    snapshot: DecisionSnapshot | None = None
    terminal_reason: str = ""
    task_success: bool = False
    task_progress: float = 0.0
    invalid_actions: int = 0
    ledger: list[dict] = field(default_factory=list)
    revoked_decisions: set[str] = field(default_factory=set)

    def transition(self, phase, reason):
        if phase not in TRANSITIONS.get(self.phase, ()):
            raise ContractError("illegal_host_transition")
        row = {"ordinal": len(self.ledger), "from": self.phase, "to": phase,
               "reason": reason, "episode_id": self.binding.episode_id,
               "observation_revision": self.observation.observation_revision if self.observation else -1,
               "previous_sha256": self.ledger[-1]["sha256"] if self.ledger else ""}
        row["sha256"] = public_hash(row)
        self.ledger.append(row)
        self.phase = phase

    def advance_observation(self, observation, *, mutation_kind):
        if "observed" not in TRANSITIONS.get(self.phase, ()):
            raise ContractError("observation_outside_transition")
        if not isinstance(observation, PublicObservation) or (
                observation.episode_id, observation.official_task_id, observation.environment,
                observation.subset) != (
                self.binding.episode_id, self.binding.official_task_id, self.binding.environment,
                self.binding.subset):
            raise ContractError("observation_episode_mismatch")
        previous = self.observation
        if previous is None:
            if mutation_kind != "open" or (
                    observation.environment_step, observation.mutation_revision,
                    observation.observation_revision) != (0, 1, 0):
                raise ContractError("invalid_initial_revision")
        else:
            if mutation_kind not in {"step", "invalid", "public_update"}:
                raise ContractError("invalid_mutation_kind")
            increment = mutation_kind != "public_update"
            expected_step = previous.environment_step + (mutation_kind == "step")
            if (observation.environment_step != expected_step
                    or observation.mutation_revision != previous.mutation_revision + increment
                    or observation.observation_revision != previous.observation_revision + 1
                    or observation.instruction != previous.instruction):
                raise ContractError("observation_revision_mismatch")
            if self.snapshot:
                self.revoked_decisions.add(self.snapshot.decision_id)
        self.observation = observation
        self.snapshot = None
        self.transition("observed", mutation_kind)

    def prepare_snapshot(self, *, evidence_revision=0, evidence_manifest_sha256="",
                         capabilities=("current_rgb",)):
        if self.phase != "observed" or self.observation is None:
            raise ContractError("snapshot_requires_current_observation")
        self.transition("preparing", "prepare_current")
        self.snapshot = DecisionSnapshot(
            self.observation, uuid.uuid4().hex, self.method_sha256, self.budget_sha256,
            evidence_revision, evidence_manifest_sha256, tuple(capabilities))
        return self.snapshot

    def extend_evidence(self, *, manifest_sha256, capabilities):
        if self.snapshot is None or self.phase not in {"preparing", "synthesizing", "executing"}:
            raise ContractError("evidence_extension_outside_decision")
        self.snapshot = replace(
            self.snapshot, evidence_revision=self.snapshot.evidence_revision + 1,
            evidence_manifest_sha256=manifest_sha256, capabilities=tuple(capabilities))
        return self.snapshot

    def adopt_memory(self, payload):
        if self.snapshot is None or payload.memory_update is None:
            return None
        obs = self.observation
        return validate_memory(payload.memory_update, episode_id=self.binding.episode_id,
                               observation_revision=obs.observation_revision,
                               decision_id=self.snapshot.decision_id)

    def finish_episode(self, reason, *, aborted=False, task_success=False, task_progress=0.0):
        if type(task_success) is not bool:
            raise ContractError("invalid_official_success")
        self.transition("aborted" if aborted else "finished", reason)
        if self.snapshot:
            self.revoked_decisions.add(self.snapshot.decision_id)
        self.terminal_reason, self.task_success = reason, task_success
        self.task_progress = task_progress
