"""Resumable per-episode scheduling for production v11 Skill evolution."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evaluation.skill_ablation_v11 import (
    initial_state_for_item,
    quality_contract,
)
from skill3d.evolution.adapters_v11 import (
    V11PairedEvaluator,
    V11PostPublishVerifier,
    model_config_identity_v11,
    solver_config_identity_v11,
)
from skill3d.evolution.campaign_v11 import (
    V11CampaignBlocked,
    V11CampaignConfig,
    V11CampaignRunner,
)
from skill3d.evolution.experience_v11 import (
    V11BufferedCollector,
    V11CaseEligibility,
    build_v11_experience_bundle_from_trace_store,
    collect_v11_streaming_case,
)
from skill3d.evolution.streaming_v11 import (
    AttemptCallback,
    StreamingAttemptRequest,
    StreamingAttemptResult,
    StreamingEvolutionConfig,
    StreamingEvolutionRunner,
)
from skill3d.online import runner
from skill3d.online.runner import OnlineRunConfig
from skill3d.routing.task_classifier import canonical_task
from skill3d.schemas import SkillSpecV11
from skill3d.skills.promote_atomic import read_snapshot
from skill3d.skills.registry import (
    active_snapshot_provenance,
    load_active_skills,
)
from skill3d.skills.v11_library import validate_v11_snapshot

SlotKey = tuple[str, str]
StateResolver = Callable[[EpisodeItem, str | Path], str]
ActiveProvider = Callable[[], tuple[list[SkillSpecV11], str, str]]
SnapshotProvider = Callable[[str], tuple[list[SkillSpecV11], str]]
EpisodeExecutor = Callable[
    [EpisodeItem, str | Path, Sequence[SkillSpecV11], str, str, Path],
    str,
]
CaseCollector = Callable[..., V11CaseEligibility]


class V11LiveEvolutionError(RuntimeError):
    """The live stream cannot continue without losing auditability."""


class V11LiveEvolutionDeferred(V11LiveEvolutionError):
    """A campaign needs an explicit retry before later episodes may run."""


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
                raise V11LiveEvolutionError(
                    f"immutable live artifact changed: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def slot_key(question_type: str, evidence_state: str) -> str:
    return f"{canonical_task(question_type)}/{evidence_state}"


@dataclass(frozen=True)
class V11ValidationPanel:
    """One frozen exact-slot E11 panel."""

    question_type: str
    evidence_state: str
    items: tuple[EpisodeItem, ...]
    artifact_paths: Mapping[str, str]

    @property
    def key(self) -> str:
        return slot_key(self.question_type, self.evidence_state)

    @property
    def qa_ids(self) -> tuple[str, ...]:
        return tuple(item.episode.qa_id for item in self.items)


def partition_items_by_slot(
    items: Sequence[EpisodeItem],
    artifact_paths: Mapping[str, str | Path],
    *,
    target_slots: Sequence[SlotKey] | None = None,
    excluded_scene_ids: Sequence[str] = (),
    state_resolver: StateResolver = initial_state_for_item,
) -> dict[str, list[EpisodeItem]]:
    """Classify readable items before model calls using the shared artifact."""
    targets = (
        {
            slot_key(question_type, evidence_state)
            for question_type, evidence_state in target_slots
        }
        if target_slots is not None
        else None
    )
    excluded = {str(value) for value in excluded_scene_ids}
    grouped: dict[str, list[EpisodeItem]] = {}
    seen_qa_ids: set[str] = set()
    for item in items:
        qa_id = str(item.episode.qa_id)
        if not qa_id or qa_id in seen_qa_ids:
            raise V11LiveEvolutionError(
                f"empty or duplicate episode identity: {qa_id!r}"
            )
        seen_qa_ids.add(qa_id)
        if item.input_error is not None or qa_id not in artifact_paths:
            continue
        scene_id = str(item.episode.scene_name or "")
        if not scene_id:
            raise V11LiveEvolutionError(f"{qa_id}: missing scene identity")
        if scene_id in excluded:
            continue
        state = state_resolver(item, artifact_paths[qa_id])
        key = slot_key(item.episode.question_type, state)
        if targets is None or key in targets:
            grouped.setdefault(key, []).append(item)
    return grouped


def prepare_validation_panels(
    items: Sequence[EpisodeItem],
    artifact_paths: Mapping[str, str | Path],
    *,
    required_slots: Sequence[SlotKey],
    min_pairs: int = 8,
    excluded_scene_ids: Sequence[str] = (),
    state_resolver: StateResolver = initial_state_for_item,
) -> dict[str, V11ValidationPanel]:
    """Freeze at least ``min_pairs`` independent inner-validation rows per slot."""
    if int(min_pairs) <= 0:
        raise ValueError("min_pairs must be positive")
    required = [
        (canonical_task(question_type), evidence_state)
        for question_type, evidence_state in required_slots
    ]
    if len(set(required)) != len(required):
        raise V11LiveEvolutionError("required validation slots contain duplicates")
    if any(item.episode.split != "inner_validation" for item in items):
        raise V11LiveEvolutionError(
            "validation panel candidates must all use inner_validation"
        )
    grouped = partition_items_by_slot(
        items,
        artifact_paths,
        target_slots=required,
        excluded_scene_ids=excluded_scene_ids,
        state_resolver=state_resolver,
    )
    panels: dict[str, V11ValidationPanel] = {}
    shortages: dict[str, int] = {}
    for question_type, evidence_state in required:
        key = slot_key(question_type, evidence_state)
        candidates = grouped.get(key, [])
        if len(candidates) < int(min_pairs):
            shortages[key] = len(candidates)
            continue
        selected = tuple(candidates[: int(min_pairs)])
        selected_artifacts = {
            item.episode.qa_id: str(
                Path(artifact_paths[item.episode.qa_id]).resolve()
            )
            for item in selected
        }
        panels[key] = V11ValidationPanel(
            question_type=question_type,
            evidence_state=evidence_state,
            items=selected,
            artifact_paths=selected_artifacts,
        )
    if shortages:
        raise V11LiveEvolutionError(
            "inner_validation exact-slot panels require at least "
            f"{int(min_pairs)} pairs: {shortages}"
        )
    return panels


def validation_panel_manifest(
    panels: Mapping[str, V11ValidationPanel],
) -> dict:
    return {
        "schema_version": "v11-validation-panels/1.0",
        "slots": {
            key: {
                "question_type": panel.question_type,
                "evidence_state": panel.evidence_state,
                "n_pairs": len(panel.items),
                "qa_ids": list(panel.qa_ids),
                "scene_ids": [
                    item.episode.scene_name for item in panel.items
                ],
                "artifact_paths": dict(panel.artifact_paths),
            }
            for key, panel in sorted(panels.items())
        },
    }


@dataclass(frozen=True)
class V11LiveSchedulerConfig:
    stream_id: str
    root: str | Path
    library_root: str | Path
    batch_size: int = 12
    flush_tail: bool = True
    retry_deferred: bool = False


class V11LiveEvolutionScheduler:
    """Run induction episodes one at a time and feed version-bound buffers."""

    def __init__(
        self,
        cfg: V11LiveSchedulerConfig,
        *,
        items: Sequence[EpisodeItem],
        artifact_paths: Mapping[str, str | Path],
        base_cfg: OnlineRunConfig,
        attempt: AttemptCallback,
        state_resolver: StateResolver = initial_state_for_item,
        active_provider: ActiveProvider | None = None,
        snapshot_provider: SnapshotProvider | None = None,
        episode_executor: EpisodeExecutor | None = None,
        case_collector: CaseCollector = collect_v11_streaming_case,
        client_factory: Callable[[str], object] | None = None,
    ) -> None:
        self.cfg = cfg
        self.root = Path(cfg.root).resolve()
        self.library_root = Path(cfg.library_root).resolve()
        self.items = list(items)
        self.artifact_paths = {
            str(key): str(Path(path).resolve())
            for key, path in artifact_paths.items()
        }
        self.base_cfg = copy.deepcopy(base_cfg)
        self.attempt = attempt
        self.state_resolver = state_resolver
        self.active_provider = active_provider or self._active
        self.snapshot_provider = snapshot_provider or self._snapshot
        self.episode_executor = episode_executor or self._execute_episode
        self.case_collector = case_collector
        self.client_factory = client_factory
        self.stream = StreamingEvolutionRunner(
            self.root / "buffer",
            stream_id=cfg.stream_id,
            config=StreamingEvolutionConfig(batch_size=int(cfg.batch_size)),
        )

    @property
    def store_dir(self) -> Path:
        return self.library_root / "snapshots"

    def _active(self) -> tuple[list[SkillSpecV11], str, str]:
        skills, warnings, snapshot_id = load_active_skills(self.store_dir)
        active_id, manifest_hash = active_snapshot_provenance(self.store_dir)
        if warnings or active_id != snapshot_id or not manifest_hash:
            raise V11LiveEvolutionError(
                f"active v11 library is not trustworthy: {warnings}"
            )
        return skills, snapshot_id, manifest_hash

    def _snapshot(self, snapshot_id: str) -> tuple[list[SkillSpecV11], str]:
        snapshot = read_snapshot(self.store_dir, snapshot_id)
        skills = validate_v11_snapshot(
            snapshot,
            library_root=self.library_root,
        )
        manifest_hash = str(snapshot.get("manifest_hash") or "")
        if len(manifest_hash) != 64:
            raise V11LiveEvolutionError(
                f"snapshot {snapshot_id} lacks a manifest hash"
            )
        return skills, manifest_hash

    @staticmethod
    def _parent(
        skills: Sequence[SkillSpecV11],
        question_type: str,
        evidence_state: str,
    ) -> SkillSpecV11:
        matches = [
            skill
            for skill in skills
            if skill.question_type == question_type
            and skill.evidence_state == evidence_state
        ]
        if len(matches) != 1:
            raise V11LiveEvolutionError(
                f"active slot is not unique: {question_type}/{evidence_state}"
            )
        return matches[0]

    def _execute_episode(
        self,
        item: EpisodeItem,
        artifact_path: str | Path,
        skills: Sequence[SkillSpecV11],
        snapshot_id: str,
        manifest_hash: str,
        trace_dir: Path,
    ) -> str:
        cfg = replace(
            copy.deepcopy(self.base_cfg),
            skills=[skill.model_copy(deep=True) for skill in skills],
            active_snapshot_ref=snapshot_id,
            active_snapshot_manifest_sha256=manifest_hash,
            trace_dir=str(trace_dir),
            reuse_artifact=str(Path(artifact_path).resolve()),
            evaluation_binding=None,
        )
        llm = (
            self.client_factory(item.episode.qa_id)
            if self.client_factory is not None
            else None
        )
        _, run = runner.run_split([item], cfg, llm=llm)
        return f"online_run:{run.run_id}"

    @staticmethod
    def _trace_complete(trace_dir: Path) -> bool:
        return all(
            (trace_dir / name).is_file()
            for name in (
                "online_run.jsonl",
                "episode_input.jsonl",
                "episode_trace.jsonl",
                "evaluation_result.jsonl",
            )
        )

    def _episode_root(self, index: int, qa_id: str) -> Path:
        digest = hashlib.sha256(qa_id.encode("utf-8")).hexdigest()[:12]
        return self.root / "episodes" / f"{index:06d}-{digest}"

    def _recover_ready_attempts(self) -> None:
        for key in sorted(self.stream.slots):
            slot = self.stream.slots[key]
            result = None
            if slot.deferred_reason:
                if self.cfg.retry_deferred:
                    result = self.stream.retry_deferred(
                        key,
                        attempt=self.attempt,
                    )
                else:
                    raise V11LiveEvolutionDeferred(
                        f"{key}: {slot.deferred_reason}"
                    )
            else:
                result = self.stream.attempt_ready(
                    key,
                    attempt=self.attempt,
                )
            if result is not None and result.outcome == "deferred":
                raise V11LiveEvolutionDeferred(
                    f"{key}: {result.reason or 'campaign deferred'}"
                )

    def _process_item(self, index: int, item: EpisodeItem) -> dict:
        qa_id = str(item.episode.qa_id)
        if item.input_error is not None:
            raise V11LiveEvolutionError(
                f"{qa_id}: induction item has an input error"
            )
        artifact_path = self.artifact_paths.get(qa_id)
        if not artifact_path:
            raise V11LiveEvolutionError(
                f"{qa_id}: missing induction artifact"
            )
        question_type = canonical_task(item.episode.question_type)
        evidence_state = self.state_resolver(item, artifact_path)
        key = slot_key(question_type, evidence_state)
        episode_root = self._episode_root(index, qa_id)
        plan_path = episode_root / "plan.json"
        handled_path = episode_root / "handled.json"
        trace_dir = episode_root / "trace"
        if handled_path.is_file():
            return json.loads(handled_path.read_bytes())

        active_skills, active_snapshot, active_manifest = self.active_provider()
        if plan_path.is_file():
            plan = json.loads(plan_path.read_bytes())
            expected = {
                "qa_id": qa_id,
                "question_type": question_type,
                "evidence_state": evidence_state,
                "slot_key": key,
                "artifact_path": str(Path(artifact_path).resolve()),
            }
            if any(plan.get(name) != value for name, value in expected.items()):
                raise V11LiveEvolutionError(
                    f"{qa_id}: resumed episode plan changed"
                )
            planned_snapshot = str(plan["parent_snapshot_id"])
            planned_skills, planned_manifest = self.snapshot_provider(
                planned_snapshot
            )
            parent = self._parent(
                planned_skills,
                question_type,
                evidence_state,
            )
            if (
                f"{parent.skill_id}@{parent.version}"
                != plan["parent_skill_version"]
                or parent.content_sha256 != plan["parent_content_sha256"]
                or planned_manifest != plan["parent_manifest_sha256"]
            ):
                raise V11LiveEvolutionError(
                    f"{qa_id}: planned parent snapshot no longer replays"
                )
            if not trace_dir.exists() and active_snapshot != planned_snapshot:
                raise V11LiveEvolutionError(
                    f"{qa_id}: active snapshot moved before the planned episode ran"
                )
            source_run_ref = str(plan.get("source_run_ref") or "")
        else:
            parent = self._parent(
                active_skills,
                question_type,
                evidence_state,
            )
            planned_snapshot = active_snapshot
            planned_skills = active_skills
            planned_manifest = active_manifest
            source_run_ref = ""
            plan = {
                "schema_version": "v11-live-episode-plan/1.0",
                "stream_id": self.cfg.stream_id,
                "index": index,
                "qa_id": qa_id,
                "question_type": question_type,
                "evidence_state": evidence_state,
                "slot_key": key,
                "artifact_path": str(Path(artifact_path).resolve()),
                "parent_snapshot_id": planned_snapshot,
                "parent_manifest_sha256": planned_manifest,
                "parent_skill_version": f"{parent.skill_id}@{parent.version}",
                "parent_content_sha256": parent.content_sha256,
            }
            _write_create_once(plan_path, _json_bytes(plan))

        if trace_dir.exists() and not self._trace_complete(trace_dir):
            raise V11LiveEvolutionError(
                f"{qa_id}: incomplete isolated trace cannot be reused"
            )
        if not trace_dir.exists():
            source_run_ref = self.episode_executor(
                item,
                artifact_path,
                planned_skills,
                planned_snapshot,
                planned_manifest,
                trace_dir,
            )
        eligibility = self.case_collector(
            trace_dir,
            stream_id=self.cfg.stream_id,
            parent_snapshot_id=planned_snapshot,
            parent=parent,
            source_run_ref=source_run_ref,
        )
        result: StreamingAttemptResult | None = None
        if not eligibility.eligible or eligibility.case is None:
            self.stream.record_operational_failure(
                qa_id=qa_id,
                reason=eligibility.reason_code or "ineligible_episode",
                detail={
                    "detail": eligibility.detail,
                    "trace_dir": str(trace_dir),
                    "slot_key": key,
                    "parent_skill_version": f"{parent.skill_id}@{parent.version}",
                },
            )
            status = "operational_failure"
        else:
            result = self.stream.ingest(
                eligibility.case,
                parent_snapshot_id=planned_snapshot,
                attempt=self.attempt,
            )
            if result is None:
                result = self.stream.attempt_ready(
                    key,
                    attempt=self.attempt,
                )
            status = "eligible"
        handled = {
            "schema_version": "v11-live-episode-result/1.0",
            "qa_id": qa_id,
            "slot_key": key,
            "status": status,
            "attempt_outcome": result.outcome if result is not None else "",
            "campaign_id": result.campaign_id if result is not None else "",
        }
        _write_create_once(handled_path, _json_bytes(handled))
        if result is not None and result.outcome == "deferred":
            raise V11LiveEvolutionDeferred(
                f"{key}: {result.reason or 'campaign deferred'}"
            )
        return handled

    def run(self) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        self._recover_ready_attempts()
        handled = [
            self._process_item(index, item)
            for index, item in enumerate(self.items, start=1)
        ]
        if self.cfg.flush_tail:
            for key in sorted(self.stream.slots):
                result = self.stream.flush_tail(
                    key,
                    attempt=self.attempt,
                )
                if result is not None and result.outcome == "deferred":
                    raise V11LiveEvolutionDeferred(
                        f"{key}: {result.reason or 'tail campaign deferred'}"
                    )
        checkpoint = json.loads(self.stream.checkpoint_path.read_bytes())
        summary = {
            "schema_version": "v11-live-stream-summary/1.0",
            "stream_id": self.cfg.stream_id,
            "n_episodes": len(self.items),
            "n_eligible": sum(
                row["status"] == "eligible" for row in handled
            ),
            "n_operational_failures": len(
                checkpoint.get("operational_failures") or []
            ),
            "n_attempts": len(checkpoint.get("attempts") or []),
            "attempt_outcomes": [
                row["outcome"] for row in checkpoint.get("attempts") or []
            ],
            "slots": {
                key: {
                    "parent_skill_version": value["parent_skill_version"],
                    "case_count": len(value["case_refs"]),
                    "new_since_attempt": value["new_since_attempt"],
                    "attempt_count": value["attempt_count"],
                    "deferred_reason": value["deferred_reason"],
                }
                for key, value in sorted(
                    (checkpoint.get("slots") or {}).items()
                )
            },
        }
        _write_atomic(self.root / "summary.json", _json_bytes(summary))
        return summary


@dataclass
class V11ProductionAttempt:
    """Translate a streaming trigger into the existing strict campaign runner."""

    library_root: str | Path
    campaign_root: str | Path
    base_cfg: OnlineRunConfig
    validation_panels: Mapping[str, V11ValidationPanel]
    post_items_by_slot: Mapping[str, Sequence[EpisodeItem]]
    post_artifact_paths: Mapping[str, str | Path]
    reviser: Callable[..., object]
    quality_confirmation: dict
    post_limit: int = 4
    min_evaluation_pairs: int = 8
    max_revision_attempts: int = 3
    paired_client_factory: Callable[[str, str], object] | None = None
    post_client_factory: Callable[[str], object] | None = None
    _used_post_ids: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.library_root = Path(self.library_root).resolve()
        self.campaign_root = Path(self.campaign_root).resolve()
        if int(self.post_limit) <= 0:
            raise ValueError("post_limit must be positive")
        self._load_post_usage()

    @property
    def _usage_path(self) -> Path:
        return Path(self.campaign_root) / "post_publish_usage.json"

    def _load_post_usage(self) -> None:
        if self._usage_path.is_file():
            payload = json.loads(self._usage_path.read_bytes())
            self._used_post_ids = set(payload.get("used_qa_ids") or [])

    def _save_post_usage(self) -> None:
        _write_atomic(
            self._usage_path,
            _json_bytes({
                "schema_version": "v11-post-publish-usage/1.0",
                "used_qa_ids": sorted(self._used_post_ids),
            }),
        )

    def _active(
        self,
        question_type: str,
        evidence_state: str,
    ) -> tuple[SkillSpecV11, str]:
        skills, warnings, snapshot_id = load_active_skills(
            Path(self.library_root) / "snapshots"
        )
        if warnings:
            raise V11LiveEvolutionError(
                f"active library warnings: {warnings}"
            )
        matches = [
            skill
            for skill in skills
            if skill.question_type == question_type
            and skill.evidence_state == evidence_state
        ]
        if len(matches) != 1:
            raise V11LiveEvolutionError(
                f"active campaign slot is not unique: "
                f"{question_type}/{evidence_state}"
            )
        return matches[0], snapshot_id

    def _campaign_id(self, request: StreamingAttemptRequest) -> str:
        return (
            f"{request.stream_id}-{request.question_type}-"
            f"{request.evidence_state}-"
            f"{request.parent_content_sha256[:12]}-"
            f"a{request.attempt_index:03d}"
        )

    def _post_panel(
        self,
        request: StreamingAttemptRequest,
        campaign_id: str,
    ) -> tuple[list[EpisodeItem], dict[str, str]]:
        campaign_dir = Path(self.campaign_root) / campaign_id
        manifest_path = campaign_dir / "live_inputs.json"
        pool = list(self.post_items_by_slot.get(request.slot_key) or [])
        by_id = {item.episode.qa_id: item for item in pool}
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_bytes())
            expected_parent = {
                "parent_snapshot_id": request.parent_snapshot_id,
                "parent_skill_version": request.parent_skill_version,
                "parent_content_sha256": request.parent_content_sha256,
            }
            if any(
                manifest.get(key) != value
                for key, value in expected_parent.items()
            ):
                raise V11LiveEvolutionError(
                    f"{campaign_id}: buffered parent identity changed"
                )
            qa_ids = list(manifest.get("post_qa_ids") or [])
        else:
            available = [
                item
                for item in pool
                if item.episode.qa_id not in self._used_post_ids
                and item.episode.qa_id
                not in {case.qa_id for case in request.cases}
            ]
            if len(available) < int(self.post_limit):
                raise V11LiveEvolutionError(
                    f"{request.slot_key}: requires {self.post_limit} fresh "
                    f"post-publish episodes, found {len(available)}"
                )
            qa_ids = [
                item.episode.qa_id
                for item in available[: int(self.post_limit)]
            ]
            _write_create_once(
                manifest_path,
                _json_bytes({
                    "schema_version": "v11-live-campaign-inputs/1.0",
                    "campaign_id": campaign_id,
                    "slot_key": request.slot_key,
                    "parent_snapshot_id": request.parent_snapshot_id,
                    "parent_skill_version": request.parent_skill_version,
                    "parent_content_sha256": request.parent_content_sha256,
                    "case_ids": [case.case_id for case in request.cases],
                    "post_qa_ids": qa_ids,
                }),
            )
        try:
            items = [by_id[qa_id] for qa_id in qa_ids]
            artifacts = {
                qa_id: str(
                    Path(self.post_artifact_paths[qa_id]).resolve()
                )
                for qa_id in qa_ids
            }
        except KeyError as exc:
            raise V11LiveEvolutionError(
                f"{campaign_id}: frozen post-publish input is unavailable: {exc}"
            ) from exc
        return items, artifacts

    def __call__(
        self,
        request: StreamingAttemptRequest,
    ) -> StreamingAttemptResult:
        parent, active_snapshot = self._active(
            request.question_type,
            request.evidence_state,
        )
        if (
            active_snapshot != request.parent_snapshot_id
            or f"{parent.skill_id}@{parent.version}"
            != request.parent_skill_version
            or parent.content_sha256 != request.parent_content_sha256
        ):
            raise V11LiveEvolutionError(
                f"{request.slot_key}: trigger is not bound to the active parent"
            )
        panel = self.validation_panels.get(request.slot_key)
        if panel is None:
            raise V11LiveEvolutionError(
                f"{request.slot_key}: no frozen validation panel"
            )
        if len(panel.items) < int(self.min_evaluation_pairs):
            raise V11LiveEvolutionError(
                f"{request.slot_key}: validation panel is below "
                f"{self.min_evaluation_pairs} pairs"
            )
        campaign_id = self._campaign_id(request)
        post_items, post_artifacts = self._post_panel(
            request,
            campaign_id,
        )
        model_identity = model_config_identity_v11(self.base_cfg)
        solver_identity = solver_config_identity_v11(
            self.base_cfg,
            seed=int(self.base_cfg.seed),
        )
        contract = quality_contract()
        campaign_cfg = V11CampaignConfig(
            campaign_id=campaign_id,
            question_type=request.question_type,
            evidence_state=request.evidence_state,
            library_root=str(self.library_root),
            run_root=str(self.campaign_root),
            max_revision_attempts=int(self.max_revision_attempts),
            min_evaluation_pairs=int(self.min_evaluation_pairs),
            method_context_max_chars=(
                self.base_cfg.retrieval_policy.method_context_max_chars
            ),
            seed=int(self.base_cfg.seed),
            model_id=self.base_cfg.online_model,
            model_config_sha256=model_identity["sha256"],
            quality_contract_sha256=contract["sha256"],
            solver_config_sha256=solver_identity["sha256"],
            resume=True,
        )
        collector = V11BufferedCollector(
            cases=request.cases,
            source_run_ref=(
                f"stream-buffer:{request.stream_id}:{request.slot_key}:"
                f"{request.parent_skill_version}:a{request.attempt_index}"
            ),
            source_refs=tuple(case.trace_ref for case in request.cases),
        )
        evaluator = V11PairedEvaluator(
            items=panel.items,
            artifact_paths=panel.artifact_paths,
            base_cfg=self.base_cfg,
            output_root=self.campaign_root,
            library_root=self.library_root,
            expected_qa_ids=panel.qa_ids,
            quality_confirmation=self.quality_confirmation,
            client_factory=self.paired_client_factory,
        )
        verifier = V11PostPublishVerifier(
            items=post_items,
            artifact_paths=post_artifacts,
            base_cfg=self.base_cfg,
            output_root=self.campaign_root,
            library_root=self.library_root,
            client_factory=self.post_client_factory,
        )
        try:
            checkpoint = V11CampaignRunner(
                campaign_cfg,
                collector=collector,
                reviser=self.reviser,
                evaluator=evaluator,
                post_publish_verifier=verifier,
            ).run()
        except V11CampaignBlocked as exc:
            return StreamingAttemptResult(
                outcome="deferred",
                campaign_id=campaign_id,
                reason=str(exc),
            )
        if checkpoint.status == "rejected":
            return StreamingAttemptResult(
                outcome="rejected",
                campaign_id=campaign_id,
                reason="strict paired admission rejected the candidate",
            )
        if checkpoint.status != "promoted":
            return StreamingAttemptResult(
                outcome="deferred",
                campaign_id=campaign_id,
                reason=f"campaign ended with status={checkpoint.status}",
            )
        active, active_snapshot = self._active(
            request.question_type,
            request.evidence_state,
        )
        bundle = verifier.last_bundle
        if bundle is None:
            bundle = build_v11_experience_bundle_from_trace_store(
                Path(self.campaign_root)
                / campaign_id
                / "post_publish_learning",
                campaign_id=campaign_id,
                parent_snapshot_id=active_snapshot,
                parent=active,
                require_real=True,
            )
        self._used_post_ids.update(
            item.episode.qa_id for item in post_items
        )
        self._save_post_usage()
        return StreamingAttemptResult(
            outcome="promoted",
            campaign_id=campaign_id,
            active_snapshot_id=active_snapshot,
            active_skill_version=f"{active.skill_id}@{active.version}",
            active_content_sha256=active.content_sha256,
            post_publish_cases=tuple(bundle.cases),
        )


__all__ = [
    "V11LiveEvolutionDeferred",
    "V11LiveEvolutionError",
    "V11LiveEvolutionScheduler",
    "V11LiveSchedulerConfig",
    "V11ProductionAttempt",
    "V11ValidationPanel",
    "partition_items_by_slot",
    "prepare_validation_panels",
    "slot_key",
    "validation_panel_manifest",
]
