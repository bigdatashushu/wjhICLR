"""Content-based progress facts; request budget and retry policy stay independent."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

PROGRESS_VERSION = "observation-progress/1.0"
# Pilot policy, frozen in the execution contract; not a calibrated quality claim.
NO_PROGRESS_LIMIT = 2
_BOOKKEEPING = frozenset({
    "result_id", "image_id", "produced_round", "delivered_rounds", "observed_rounds",
    "unobserved_reason", "request_digest", "audit", "binding_provenance",
    "n_new_objects", "new_object_ids",
    "note", "detector_endpoint_hash", "error",
})


def _content(value):
    if isinstance(value, dict):
        return {key: _content(item) for key, item in value.items()
                if key not in _BOOKKEEPING}
    if isinstance(value, (list, tuple)):
        items = [_content(item) for item in value]
        # Object/detection enumeration order is not additional evidence.
        if items and all(isinstance(item, dict) for item in items):
            return sorted(items, key=_json)
        return items
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def transient_results(results) -> bool:
    """Only explicit service facts pause progress counting."""
    for result in results:
        payload = getattr(result, "payload", None) or {}
        if isinstance(payload, dict) and (
                payload.get("service_healthy") is False or payload.get("status") == "fault"):
            return True
        if ((getattr(result, "source_tool", "") or getattr(result, "tool", "")) == "detect_objects"
                and isinstance(payload, dict) and payload.get("error")):
            return True  # Partial per-frame service failure is still a service fact
        cause = getattr(result, "failure_cause", None)
        code = str(getattr(cause, "code", "") or "")
        if code in {"service_error", "service_unavailable", "detector_service_fault"}:
            return True
        if getattr(result, "error_code", None) in {"timeout", "service_error", "service_unavailable"}:
            return True
    return False


@dataclass
class ProgressState:
    seen_observations: set[str] = field(default_factory=set)
    seen_images: set[str] = field(default_factory=set)
    context_hash: str = ""
    consecutive_no_progress: int = 0

    def update(self, observations, ledger, *, context, eligible: bool,
               transient: bool = False, pages=()) -> dict:
        keys = {
            _hash([obs.source_tool, obs.evidence_version, _content(obs.payload), _content(obs.args)])
            for obs in observations
            if not (isinstance(obs.payload, dict) and (
                obs.payload.get("service_healthy") is False or obs.payload.get("status") == "fault"))
        }
        keys.update(_hash(["read_page", _content(page)]) for page in pages)
        images = {
            _hash([image.content_sha256, image.source_frame_id, image.box_xyxy])
            for image in (ledger.all_images() if ledger is not None else [])
            if image.observed()
        }
        current_context = _hash(context)
        changed = bool(self.context_hash and current_context != self.context_hash)
        new_observations = keys - self.seen_observations
        new_images = images - self.seen_images
        made_progress = bool(new_observations or new_images or changed)
        if changed:
            # Revocation/new candidates reset the comparison against the valid state.
            self.seen_observations = set(keys)
        else:
            self.seen_observations.update(keys)
        self.seen_images.update(images)
        self.context_hash = current_context
        if made_progress:
            self.consecutive_no_progress = 0
        elif eligible and not transient:
            self.consecutive_no_progress += 1
        return {
            "schema_version": PROGRESS_VERSION,
            "made_progress": made_progress,
            "new_observation_count": len(new_observations),
            "new_response_completed_image_count": len(new_images),
            "candidate_or_capability_change": changed,
            "transient_service_error": transient,
            "eligible": eligible,
            "consecutive_no_progress": self.consecutive_no_progress,
            "frozen_limit": NO_PROGRESS_LIMIT,
            "finalize_required": eligible and not transient
            and self.consecutive_no_progress >= NO_PROGRESS_LIMIT,
        }
