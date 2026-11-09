"""Sealed host-selected profiles for the task-local execution adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ControlProfile:
    name: str
    terminal: str
    constructor: str
    reserved_names: frozenset[str]
    forbidden_names: frozenset[str]
    context_fields: frozenset[str]


QA_V11_PROFILE = ControlProfile(
    "qa-v11", "ReturnAnswer", "AnswerPayload",
    frozenset({"show", "ReturnAnswer", "YieldObservations", "AnswerPayload",
               "ReadObservation", "tools", "scene", "frames", "ctx"}),
    frozenset(), frozenset())
ACTION_CONTEXT_FIELDS = frozenset({
    "episode_id", "decision_id", "observation_revision", "snapshot_sha256",
    "action_table_sha256", "action_table", "instruction", "feedback", "memory",
    "capabilities", "evidence_revision", "view_id",
    "official_objects", "action_spec", "task_variation",
    "task_state",
})
EMBODIED_ACTIONS_PROFILE = ControlProfile(
    "embodied-actions/1", "SubmitActions", "ActionPayload",
    frozenset({"show", "SubmitActions", "YieldObservations", "ActionPayload",
               "ReadObservation", "tools", "scene", "frames", "ctx"}),
    frozenset({"ReturnAnswer", "AnswerPayload", "reset", "step"}),
    ACTION_CONTEXT_FIELDS | {"tools", "frames"})


def validate_profile(profile):
    if profile is not QA_V11_PROFILE and profile is not EMBODIED_ACTIONS_PROFILE:
        raise ValueError("control profile must be a sealed host constant")
    return profile


@dataclass(frozen=True)
class ActionContextView:
    episode_id: str
    decision_id: str
    observation_revision: int
    snapshot_sha256: str
    action_table_sha256: str
    action_table: tuple[str, ...]
    instruction: str
    feedback: str
    memory: Any
    capabilities: tuple[str, ...]
    evidence_revision: int
    view_id: str
    official_objects: Any = field(default=None, kw_only=True)
    action_spec: Any = field(default=None, kw_only=True)
    task_variation: str = field(default="", kw_only=True)
    task_state: Any = field(default=None, kw_only=True)


@dataclass(frozen=True)
class ActionSolverContext(ActionContextView):
    tools: object
    frames: tuple
