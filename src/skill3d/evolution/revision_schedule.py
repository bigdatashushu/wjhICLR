"""Deterministic fit-set scheduling; gate outcomes never become revision input."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

from skill3d.schemas.skill import normalize_skill_markdown, parse_skill_markdown

SCHEDULE_VERSION = "fitset-revision-schedule/1.0"
DEDUP_VERSION = "same-parent-method/1.0"


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False,
        separators=(",", ":")).encode()).hexdigest()


def failure_patterns(feedback: Mapping) -> dict[str, dict]:
    """Ignore sample IDs, raw scores, counts and request-dependent round indices."""
    patterns = {}
    for signature in feedback.get("signatures", []):
        process = signature.get("process") or {}
        errors = process.get("error_chain") or []
        band = signature.get("score_band")
        if band == "full_score" and not errors:
            continue
        execution = process.get("execution") or {}
        observation = process.get("observation_quality") or {}
        pattern = {
            "score_band": band,
            "terminal_action": execution.get("terminal_action", "unknown"),
            "answer_submitted": execution.get("answer_submitted"),
            "answer_assignment_without_submit": execution.get("answer_assignment_without_submit"),
            "errors": sorted({
                json.dumps({key: event.get(key) for key in (
                    "stage", "event_type", "tool_name", "exception_type",
                    "return_type", "recovered", "terminal", "capability_change",
                )}, sort_keys=True)
                for event in errors
            }),
            "observation": {key: observation.get(key) for key in (
                "candidate_coverage", "summary_truncated", "required_fields_omitted",
                "referenced_image_currently_present",
            )},
        }
        patterns[_hash(pattern)] = pattern
    return dict(sorted(patterns.items()))


def normalized_method_sha256(skill_md: str) -> str:
    """Normalize LF/final newline and version metadata, never code indentation."""
    metadata, body = parse_skill_markdown(normalize_skill_markdown(skill_md))
    method_metadata = {
        key: value for key, value in metadata.items()
        if key not in {"version", "skill_version", "parent_version"}
    }
    return _hash({"metadata": method_metadata, "body": normalize_skill_markdown(body)})


def candidate_dedup_key(*, slot: str, parent_hash: str, skill_md: str) -> str:
    return _hash({
        "version": DEDUP_VERSION, "slot": slot, "parent_sha256": parent_hash,
        "method_sha256": normalized_method_sha256(skill_md),
    })


def scheduling_decision(*, feedback: Mapping, previous_patterns: set[str],
                        mode: str, revision_requests_used: int,
                        gate_candidates_used: int, max_revision_requests: int,
                        max_gate_candidates: int) -> dict:
    patterns = failure_patterns(feedback)
    novel = sorted(set(patterns) - previous_patterns)
    remaining = max(0, max_revision_requests - revision_requests_used)
    if remaining == 0:
        reason = "slot_revision_request_limit"
    elif gate_candidates_used >= max_gate_candidates:
        reason = "slot_candidate_limit"
    elif mode == "novel_failure_v1" and not novel:
        reason = "no_new_failure_pattern"
    else:
        reason = "scheduled"
    return {
        "schema_version": SCHEDULE_VERSION, "mode": mode,
        "scheduled": reason == "scheduled", "reason_code": reason,
        "pattern_refs": sorted(patterns), "novel_pattern_refs": novel,
        "prior_pattern_refs": sorted(previous_patterns),
        "revision_requests_used": revision_requests_used,
        "gate_candidates_used": gate_candidates_used,
        "remaining_revision_requests": remaining,
        "max_revision_requests": max_revision_requests,
        "max_gate_candidates": max_gate_candidates,
    }
