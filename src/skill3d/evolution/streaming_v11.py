"""Persistent per-slot, per-parent buffers for continuous v11 Skill evolution."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

from skill3d.schemas import V11ExperienceCase

STREAM_SCHEMA_VERSION = "skill-evolution-stream-v11/1.0"


class StreamingEvolutionError(RuntimeError):
    """The stream cannot continue without violating version or receipt identity."""


def _json_bytes(value: object) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_create_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise StreamingEvolutionError(
                    f"immutable stream artifact already exists with different content: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class StreamingEvolutionConfig:
    batch_size: int = 12
    tail_for_unattempted_slots: bool = True

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError("batch_size must be a positive integer")


@dataclass(frozen=True)
class StreamingAttemptRequest:
    stream_id: str
    question_type: str
    evidence_state: str
    parent_snapshot_id: str
    parent_skill_version: str
    parent_content_sha256: str
    attempt_index: int
    trigger: Literal["batch", "tail"]
    cases: tuple[V11ExperienceCase, ...]
    new_case_count: int

    @property
    def slot_key(self) -> str:
        return f"{self.question_type}/{self.evidence_state}"


@dataclass(frozen=True)
class StreamingAttemptResult:
    outcome: Literal["promoted", "rejected", "deferred"]
    campaign_id: str
    reason: str = ""
    active_snapshot_id: str = ""
    active_skill_version: str = ""
    active_content_sha256: str = ""
    post_publish_cases: tuple[V11ExperienceCase, ...] = ()

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id must not be empty")
        if self.outcome == "promoted":
            if not (
                self.active_snapshot_id
                and self.active_skill_version
                and len(self.active_content_sha256) == 64
            ):
                raise ValueError("promoted result requires the new active identity")
        elif self.post_publish_cases:
            raise ValueError("only a promoted result can return post-publish cases")


@dataclass
class _SlotBuffer:
    question_type: str
    evidence_state: str
    parent_snapshot_id: str
    parent_skill_version: str
    parent_content_sha256: str
    case_refs: list[str] = field(default_factory=list)
    new_since_attempt: int = 0
    attempt_count: int = 0
    attempted_ever: bool = False
    deferred_reason: str = ""

    @property
    def slot_key(self) -> str:
        return f"{self.question_type}/{self.evidence_state}"


AttemptCallback = Callable[[StreamingAttemptRequest], StreamingAttemptResult]


class StreamingEvolutionRunner:
    """Accumulate immutable cases and trigger serialized, version-bound campaigns."""

    def __init__(
        self,
        root: str | Path,
        *,
        stream_id: str,
        config: StreamingEvolutionConfig | None = None,
    ) -> None:
        if not stream_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
            for char in stream_id
        ):
            raise ValueError("stream_id is not a safe identifier")
        self.root = Path(root).resolve()
        self.stream_id = stream_id
        self.config = config or StreamingEvolutionConfig()
        self.checkpoint_path = self.root / "stream.json"
        self.case_root = self.root / "cases"
        self.failure_root = self.root / "operational_failures"
        self.attempt_root = self.root / "attempts"
        self.slots: dict[str, _SlotBuffer] = {}
        self.attempts: list[dict] = []
        self.operational_failures: list[str] = []
        self._seen_case_ids: set[str] = set()
        self._load_or_initialize()

    def _load_or_initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        expected = asdict(self.config)
        if not self.checkpoint_path.exists():
            self._save()
            return
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != STREAM_SCHEMA_VERSION:
            raise StreamingEvolutionError("unsupported streaming checkpoint schema")
        if payload.get("stream_id") != self.stream_id:
            raise StreamingEvolutionError("stream_id differs from existing checkpoint")
        if payload.get("config") != expected:
            raise StreamingEvolutionError("streaming config changed during resume")
        self.slots = {
            key: _SlotBuffer(**value)
            for key, value in (payload.get("slots") or {}).items()
        }
        self.attempts = list(payload.get("attempts") or [])
        self.operational_failures = list(payload.get("operational_failures") or [])
        self._seen_case_ids = set(payload.get("seen_case_ids") or [])
        active_case_ids: set[str] = set()
        for slot in self.slots.values():
            for ref in slot.case_refs:
                path = self.root / ref
                if not path.is_file():
                    raise StreamingEvolutionError(f"checkpoint case is missing: {ref}")
                case = self._decode_case(path.read_bytes())
                if case.case_id in active_case_ids:
                    raise StreamingEvolutionError(
                        f"duplicate active checkpoint case_id: {case.case_id}")
                active_case_ids.add(case.case_id)
                if case.case_id not in self._seen_case_ids:
                    raise StreamingEvolutionError(
                        f"active checkpoint case is missing from seen_case_ids: {case.case_id}"
                    )

    def _payload(self) -> dict:
        return {
            "schema_version": STREAM_SCHEMA_VERSION,
            "stream_id": self.stream_id,
            "config": asdict(self.config),
            "slots": {
                key: asdict(value) for key, value in sorted(self.slots.items())
            },
            "attempts": list(self.attempts),
            "operational_failures": list(self.operational_failures),
            "seen_case_ids": sorted(self._seen_case_ids),
        }

    def _save(self) -> None:
        _write_atomic(self.checkpoint_path, _json_bytes(self._payload()))

    def _store_case(self, case: V11ExperienceCase) -> str:
        data = _json_bytes(case)
        digest = _sha256(data)
        path = self.case_root / digest[:2] / f"{digest}.json"
        _write_create_once(path, data)
        return path.relative_to(self.root).as_posix()

    def _decode_case(self, data: bytes) -> V11ExperienceCase:
        return V11ExperienceCase.model_validate_json(data)

    def _load_cases(self, slot: _SlotBuffer) -> tuple[V11ExperienceCase, ...]:
        return tuple(
            self._decode_case((self.root / ref).read_bytes())
            for ref in slot.case_refs
        )

    @staticmethod
    def _case_parent(case: V11ExperienceCase) -> tuple[str, str]:
        return case.delivered_skill_version, case.delivered_content_sha256

    def record_operational_failure(self, *, qa_id: str, reason: str, detail: dict) -> str:
        payload = {
            "stream_id": self.stream_id,
            "qa_id": str(qa_id or ""),
            "reason": str(reason or "unknown"),
            "detail": dict(detail),
        }
        data = _json_bytes(payload)
        digest = _sha256(data)
        path = self.failure_root / f"{digest}.json"
        _write_create_once(path, data)
        ref = path.relative_to(self.root).as_posix()
        if ref not in self.operational_failures:
            self.operational_failures.append(ref)
            self._save()
        return ref

    def ingest(
        self,
        case: V11ExperienceCase,
        *,
        parent_snapshot_id: str,
        attempt: AttemptCallback | None = None,
    ) -> StreamingAttemptResult | None:
        if case.case_id in self._seen_case_ids:
            return None
        slot_key = f"{case.question_type}/{case.evidence_state}"
        parent_version, parent_hash = self._case_parent(case)
        slot = self.slots.get(slot_key)
        if slot is None:
            slot = _SlotBuffer(
                question_type=case.question_type,
                evidence_state=case.evidence_state,
                parent_snapshot_id=parent_snapshot_id,
                parent_skill_version=parent_version,
                parent_content_sha256=parent_hash,
            )
            self.slots[slot_key] = slot
        expected = (slot.parent_skill_version, slot.parent_content_sha256)
        actual = (parent_version, parent_hash)
        if actual != expected:
            raise StreamingEvolutionError(
                f"{slot_key} received a case from another active parent: "
                f"expected={expected}, actual={actual}"
            )
        # Another slot may publish and advance the global snapshot while this
        # lineage remains unchanged. Bind the next campaign to the latest active
        # snapshot without discarding correctly versioned experience.
        slot.parent_snapshot_id = parent_snapshot_id
        slot.case_refs.append(self._store_case(case))
        slot.new_since_attempt += 1
        self._seen_case_ids.add(case.case_id)
        self._save()
        if (
            attempt is not None
            and slot.new_since_attempt >= int(self.config.batch_size)
            and not slot.deferred_reason
        ):
            return self._attempt(slot, trigger="batch", callback=attempt)
        return None

    def retry_deferred(
        self,
        slot_key: str,
        *,
        attempt: AttemptCallback,
    ) -> StreamingAttemptResult | None:
        slot = self.slots.get(slot_key)
        if slot is None or not slot.deferred_reason:
            return None
        return self._attempt(slot, trigger="batch", callback=attempt)

    def attempt_ready(
        self,
        slot_key: str,
        *,
        attempt: AttemptCallback,
    ) -> StreamingAttemptResult | None:
        """Recover a full batch whose callback did not reach its stream receipt."""
        slot = self.slots.get(slot_key)
        if (
            slot is None
            or slot.deferred_reason
            or slot.new_since_attempt < int(self.config.batch_size)
        ):
            return None
        return self._attempt(slot, trigger="batch", callback=attempt)

    def flush_tail(
        self,
        slot_key: str,
        *,
        attempt: AttemptCallback,
    ) -> StreamingAttemptResult | None:
        slot = self.slots.get(slot_key)
        if slot is None or not slot.case_refs:
            return None
        if slot.attempted_ever or not self.config.tail_for_unattempted_slots:
            return None
        return self._attempt(slot, trigger="tail", callback=attempt)

    def _attempt(
        self,
        slot: _SlotBuffer,
        *,
        trigger: Literal["batch", "tail"],
        callback: AttemptCallback,
    ) -> StreamingAttemptResult:
        request = StreamingAttemptRequest(
            stream_id=self.stream_id,
            question_type=slot.question_type,
            evidence_state=slot.evidence_state,
            parent_snapshot_id=slot.parent_snapshot_id,
            parent_skill_version=slot.parent_skill_version,
            parent_content_sha256=slot.parent_content_sha256,
            attempt_index=slot.attempt_count + 1,
            trigger=trigger,
            cases=self._load_cases(slot),
            new_case_count=slot.new_since_attempt,
        )
        result = callback(request)
        receipt = {
            "request": {
                **asdict(request),
                "cases": [case.case_id for case in request.cases],
            },
            "result": {
                **asdict(result),
                "post_publish_cases": [
                    case.case_id for case in result.post_publish_cases
                ],
            },
        }
        data = _json_bytes(receipt)
        path = self.attempt_root / f"{len(self.attempts) + 1:04d}-{result.campaign_id}.json"
        _write_create_once(path, data)
        ref = path.relative_to(self.root).as_posix()
        self.attempts.append(
            {
                "path": ref,
                "sha256": _sha256(data),
                "slot_key": slot.slot_key,
                "parent_skill_version": slot.parent_skill_version,
                "outcome": result.outcome,
            }
        )
        if result.outcome == "deferred":
            slot.deferred_reason = result.reason or "attempt deferred"
            self._save()
            return result

        slot.attempt_count += 1
        slot.attempted_ever = True
        slot.new_since_attempt = 0
        slot.deferred_reason = ""
        if result.outcome == "rejected":
            self._save()
            return result

        replacement = _SlotBuffer(
            question_type=slot.question_type,
            evidence_state=slot.evidence_state,
            parent_snapshot_id=result.active_snapshot_id,
            parent_skill_version=result.active_skill_version,
            parent_content_sha256=result.active_content_sha256,
            attempted_ever=True,
        )
        self.slots[slot.slot_key] = replacement
        self._save()
        for case in result.post_publish_cases:
            self.ingest(
                case,
                parent_snapshot_id=result.active_snapshot_id,
                attempt=None,
            )
        return result


__all__ = [
    "STREAM_SCHEMA_VERSION",
    "StreamingAttemptRequest",
    "StreamingAttemptResult",
    "StreamingEvolutionConfig",
    "StreamingEvolutionError",
    "StreamingEvolutionRunner",
]
