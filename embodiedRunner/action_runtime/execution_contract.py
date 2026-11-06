"""Embodied-only execution contract; the existing QA contract remains unchanged."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from skill3d.schemas.tool_response import TOOL_RESPONSE_VERSION


@dataclass(frozen=True)
class ActionExecutionContract:
    method_context_max_chars: int = 16000
    contract_version: str = "embodied-execution-contract/1"
    environment: str = "eb_navigation"

    def to_dict(self):
        if type(self.method_context_max_chars) is not int or self.method_context_max_chars <= 0:
            raise ValueError("method context cap must be a positive integer")
        value = {
            "contract_version": self.contract_version,
            "method_context_max_chars": self.method_context_max_chars,
            "control_profile": "embodied-actions/1",
            "output": {"program_source": "Python source string; only JSON field"},
            "submission": "SubmitActions(ActionPayload(decision_id, observation_revision, "
                          "snapshot_sha256, action_table_sha256, actions, evidence_refs, memory_update))",
            "context_fields": ["episode_id", "decision_id", "observation_revision", "snapshot_sha256",
                               "action_table_sha256", "action_table", "instruction", "feedback",
                               "memory", "capabilities", "evidence_revision", "view_id", "tools", "frames"],
            "rules": [
                "Define solve(ctx); the host calls it. Every segment has a fresh user namespace.",
                "SubmitActions submits typed data and immediately terminates execution.",
                "YieldObservations(result_ids, reason) requests same-decision feedback and terminates.",
                "Actions are official integer IDs, not booleans or floats. An empty list stops.",
                "The host executes only the first accepted action then obtains a fresh observation.",
                "All current geometry/results are revoked after any environment mutation, even if RGB is unchanged.",
                "Tools return an envelope; read result['data'], and reference result['result_id'].",
                "Every actually exposed tool/reader result is a submission dependency, including omitted references.",
                "ReturnAnswer, AnswerPayload, file/network/process IO and environment reset/step are unavailable.",
                "Use only current original RGB, official feedback and bounded memory with source attribution.",
                "Memory is subgoal text plus at most 16 sourced records, at most 4000 text characters.",
                "Per-view predicted depth/points are relative; category confirmation does not validate surfaces.",
                "ReadObservation pages have at most 4000 characters and 32 items; at most 3 reads per segment.",
                "The same frozen complete method is delivered on every ordinary and repair request.",
                "A method cannot grant capabilities, increase budget or determine official success.",
            ],
            "geometry_unit": "per_view_relative", "metric_world": False,
            "history_policy": "current_only", "finalization": False,
            "tool_response_schema_version": TOOL_RESPONSE_VERSION,
        }
        if self.environment == "eb_manipulation":
            value["context_fields"] += ["official_objects", "action_spec", "task_variation"]
            value["rules"][3] = (
                "Each action is [x,y,z,roll,pitch,yaw,gripper]; position integer indices 0..100, "
                "rotation integer indices 0..120 (3 degrees), gripper 0/1. No action IDs. Empty stops.")
            value["rules"][10] = (
                "Use current original RGB, official anonymous object coordinates, feedback and sourced memory. "
                "Planner image annotations follow the official protocol. Prediction tools use original RGB.")
            value["official_input_source"] = "official_depth_instance_masks_and_camera_parameters"
            value["rules"].append(
                "Official object coordinates are separate world voxel inputs; predicted relative depth "
                "does not provide a conversion to that frame. No raw simulator depth/masks/state is exposed.")
        return value

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def render(self):
        return f"Execution contract {self.contract_version}; sha256={self.sha256}\n" + json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True)
