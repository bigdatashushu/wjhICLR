"""Harness3D 同题实验的槽位流、独立运行库和严格 A/B 准入。"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Literal, Mapping

from skill3d.evaluation.fitset_v11 import (
    ALL_FIT_SLOTS,
    REACHABLE_FIT_SLOTS,
    FitSetError,
    _require,
    _variant_key,
    _write_create_once,
    content_sha256,
    fit_slot_key,
    score_answer,
)
from skill3d.evaluation.gate_process_metrics import (
    episode_process_metrics, aggregate_gate_process_metrics,
)
from skill3d.evolution.fitset_revision_v11 import (
    FIT_REVISION_PROMPT_VERSION,
    FitSetSlotReviser,
    RevisionInputBudgetExhausted,
    RevisionOperationalError,
    load_active_parent_slots,
)
from skill3d.evolution.deployment_continuation_v11 import revision_intents_used
from skill3d.evolution.leakage import leakage_scan_text
from skill3d.evolution.revision_schedule import (
    candidate_dedup_key, normalized_method_sha256, scheduling_decision,
)
from skill3d.synthesis.request_audit import describe_request
from skill3d.evolution.process_feedback import (
    FEEDBACK_SCHEMA_VERSION, FEEDBACK_WHITELIST_VERSION, FIELD_SEMANTICS,
    ProcessFacts, extract_process_facts, legacy_process_facts, safe_process_dict,
    score_band,
)
from skill3d.evolution.streaming_v11 import (
    StreamingAttemptResult,
    StreamingEvolutionConfig,
    StreamingEvolutionRunner,
)
from skill3d.schemas import (
    SkillCandidateV11, SkillSpecV11, Spec, EvidenceProfile,
    V11ExperienceCase, V11ExperienceBundle, DataAccessRecord,
)
from skill3d.schemas.evidence import skill_evidence_state
from skill3d.schemas.data_access import (
    learning_authorization_for, validate_learning_authorization,
)
from pydantic import Field, model_validator
from skill3d.skills.promote_atomic import read_snapshot
from skill3d.skills.v11_library import (
    LoadedSkillV11,
    build_v11_snapshot,
    publish_v11_candidate,
    source_ref_for,
    validate_v11_snapshot,
    write_v11_snapshot,
)

FIT_STREAM_SCHEMA = "harness3d-fitset-stream/5.0"


@dataclass(frozen=True)
class FitSetEvolutionConfig:
    batch_size: int = 12
    gate_size: int = 8
    gate_seeds: tuple[int, ...] = (137, 138)
    max_revision_attempts: int = 3
    max_tokens: int = 8192
    revision_input_max_bytes: int = 128000
    revision_input_max_chars: int = 72000
    scheduling_mode: Literal["batch_v1", "novel_failure_v1"] = "novel_failure_v1"
    max_revision_requests_per_slot: int = 18
    max_gate_candidates_per_slot: int = 6
    learning_input_mode: str = "aggregate_v1"
    gate_mode: str = "same_question"
    diagnosis_max_cases: int = 4
    diagnosis_max_input_chars: int = 64000
    diagnosis_max_images: int = 8
    diagnosis_max_output_tokens: int = 4096

    def __post_init__(self):
        if self.batch_size != 12:
            raise ValueError("fit-set 槽位批次必须为 12 条")
        if self.gate_size < 1 or self.max_revision_attempts < 1 or self.max_tokens < 1:
            raise ValueError("gate/revision 配置必须为正数")
        if len(self.gate_seeds) != 2 or len(set(self.gate_seeds)) != 2:
            raise ValueError("fit-set A/B 必须使用两个不同的冻结 seed")
        if self.scheduling_mode not in ("batch_v1", "novel_failure_v1"):
            raise ValueError("unknown revision scheduling mode")
        if min(self.max_revision_requests_per_slot, self.max_gate_candidates_per_slot) < 1:
            raise ValueError("per-slot revision/gate limits must be positive")
        if self.revision_input_max_bytes < 4096 or self.revision_input_max_chars < 4096:
            raise ValueError("revision input byte budget must be at least 4096")
        if self.learning_input_mode not in {
                "aggregate_v1", "full_cases_v1", "diagnosis_v1", "diagnosis_multimodal_v1"}:
            raise ValueError("unknown learning input mode")
        if self.gate_mode not in {"same_question", "scene_disjoint", "qa_disjoint"}:
            raise ValueError("unknown gate mode")
        if self.gate_mode == "qa_disjoint" and self.gate_seeds != (137, 138):
            raise ValueError("qa_disjoint requires frozen dual seeds 137/138")
        if min(self.diagnosis_max_cases, self.diagnosis_max_input_chars,
               self.diagnosis_max_images, self.diagnosis_max_output_tokens) < 1:
            raise ValueError("diagnosis budgets must be positive")


class FitSetExperienceCase(Spec):
    """保留真实来源 split；仅此探索入口允许同题反馈。"""

    case_id: str
    qa_id: str
    scene_id: str
    question_type: str
    evidence_state: Literal["g0m0", "g1m0", "g1m1"]
    source_split: Literal["induction", "inner_validation", "outer_holdout"]
    experiment_role: Literal["fit_set_learning"] = "fit_set_learning"
    learning_authorization: dict = Field(default_factory=dict)
    delivered_skill_version: str
    delivered_content_sha256: str
    delivered_skill_md: str
    score: float = Field(ge=0, le=1)
    outcome: Literal["success", "failure"]
    failure_categories: list[str]
    process_signature: dict = Field(default_factory=dict)
    process_facts: ProcessFacts | None = None
    trace_ref: str
    trace_sha256: str
    initial_evidence_states: dict
    evidence_history: list[dict]
    full_case: V11ExperienceCase | None = None

    @model_validator(mode="after")
    def _validate_identity(self):
        if self.learning_authorization:
            validate_learning_authorization(self.learning_authorization, [self.qa_id])
        if hashlib.sha256(self.delivered_skill_md.encode()).hexdigest() != (
                self.delivered_content_sha256):
            raise ValueError("fit-set case delivery hash 改变")
        EvidenceProfile.model_validate(self.initial_evidence_states)
        if skill_evidence_state(self.initial_evidence_states) != self.evidence_state:
            raise ValueError("fit-set case 初始证据错配")
        if not self.evidence_history or (
                self.evidence_history[0]["evidence_states"]
                != self.initial_evidence_states):
            raise ValueError("fit-set case 缺少首次证据历史")
        for row in self.evidence_history:
            EvidenceProfile.model_validate(row["evidence_states"])
        if (self.outcome == "success") != (self.score == 1):
            raise ValueError("fit-set case outcome 与 score 不符")
        if self.full_case is not None:
            for field in (
                "qa_id", "scene_id", "question_type", "evidence_state", "source_split",
                "delivered_skill_version", "delivered_content_sha256", "delivered_skill_md",
                "score", "outcome", "trace_ref", "initial_evidence_states", "evidence_history",
                "learning_authorization",
            ):
                if getattr(self, field) != getattr(self.full_case, field):
                    raise ValueError(f"fit-set buffer/canonical case {field} mismatch")
        return self


class FitSetStreamingBuffer(StreamingEvolutionRunner):
    def __init__(self, root, *, stream_id, config=None, framework_state_io=None):
        self.framework_state_io = framework_state_io
        if framework_state_io is not None:
            from skill3d.storage.framework_state_io import bind_framework_writers
            framework_state_io.mkdir(Path(root))
            bind_framework_writers(self, StreamingEvolutionRunner, {
                "_write_atomic": framework_state_io.write_bytes,
                "_write_create_once": framework_state_io.write_bytes_once,
            })
            self._bounded_attempt = self.__dict__.pop("_attempt")
        super().__init__(root, stream_id=stream_id, config=config)

    def _attempt(self, *args, **kwargs):
        if self.framework_state_io is None:
            return super()._attempt(*args, **kwargs)
        result = self._bounded_attempt(*args, **kwargs)
        self.framework_state_io.after_batch(best=result.outcome == "promoted")
        return result

    def _decode_case(self, data: bytes) -> FitSetExperienceCase:
        return FitSetExperienceCase.model_validate_json(data)


@dataclass
class FrozenSkillProvider:
    skills: list[SkillSpecV11]
    identity: dict

    def __call__(self, _variant):
        return self.skills, dict(self.identity)


class SeededGateClient:
    """模型 seed 独立于封存的 M5 seed；记录每次真实请求参数。"""

    def __init__(self, client, seed: int, audit_path: Path, framework_state_io=None):
        self.client = client
        self.seed = seed
        self.audit_path = audit_path
        self.requests = []
        self.audit_context = {}
        self.request_context = {}
        self.framework_state_io = framework_state_io

    def _write_create_once(self, path, value):
        if self.framework_state_io is None:
            return _write_create_once(path, value)
        return self.framework_state_io.write_json_once(path, value)

    def __getattr__(self, name):
        return getattr(self.client, name)

    def set_audit_context(self, context: Mapping):
        self.audit_context = copy.deepcopy(dict(context))

    def set_request_context(self, context: Mapping):
        self.request_context = copy.deepcopy(dict(context))

    def chat(self, *args, **kwargs):
        kwargs["seed"] = self.seed
        build_payload = getattr(self.client, "build_payload", None)
        identity_fn = getattr(self.client, "audit_identity", None)
        model_identity = identity_fn(seed=self.seed) if callable(identity_fn) else {}
        payload = (build_payload(*args, **kwargs) if callable(build_payload) else {
            "messages": args[0] if args else kwargs.get("messages", []),
            **{key: value for key, value in kwargs.items() if key != "messages"},
        })
        request_identity = describe_request(
            payload, model_identity=model_identity, wire_encoding_known=callable(build_payload))
        index = len(self.requests) + 1
        request = {
            "request_seed": self.seed, "request_index": index,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request_identity": request_identity,
            "frozen_context": self.audit_context,
            "request_context": self.request_context,
            "retry_of": None,
        }
        self.requests.append(copy.deepcopy(request))
        self._write_create_once(
            self.audit_path.with_name(
                f"{self.audit_path.stem}-{index:03d}.json"), request,
        )
        started = time.perf_counter()
        error_type = None
        try:
            text = self.client.chat(*args, **kwargs)
            return text
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            self._write_create_once(self.audit_path.with_name(
                f"{self.audit_path.stem}-response-{index:03d}.json"), {
                "request_seed": self.seed,
                "request_index": index, "request_identity_sha256": content_sha256(request_identity),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": time.perf_counter() - started,
                "status": "error" if error_type else "completed",
                "exception_type": error_type,
                "usage": dict(getattr(self.client, "last_usage", {}) or {}) if not error_type else {},
                "response_identity": dict(
                    getattr(self.client, "last_response_identity", {}) or {}) if not error_type else {},
                "actual_request_identity": dict(
                    getattr(self.client, "last_request_identity", {}) or {}),
            })


def _active(root: Path):
    return load_active_parent_slots(root)


def initialize_worker_library(
    root: Path,
    *,
    seed_skills: list[SkillSpecV11],
    owned_slots: list[str],
    run_identity: str,
    framework_state_io=None,
) -> None:
    """从完整 seed 写独立文件；共享库无写入、无符号链接。"""
    seed_by_slot = {skill.slot_key: skill for skill in seed_skills}
    _require(set(seed_by_slot) == {fit_slot_key(*slot) for slot in ALL_FIT_SLOTS},
             "worker seed 必须完整覆盖 24 条")
    identity = {
        "schema_version": FIT_STREAM_SCHEMA,
        "run_identity": run_identity,
        "owned_slots": sorted(owned_slots),
        "seed_skills": {
            key: value.model_dump(mode="json")
            for key, value in sorted(seed_by_slot.items())
        },
    }
    marker = root.parent / "worker_identity.json"
    if marker.exists():
        _require(json.loads(marker.read_text()) == identity,
                 "worker 所有权或 seed 在 resume 时发生变化")
        _active(root)
        return
    _require(not root.exists(), "worker library 存在但缺少 identity，拒绝猜测")
    sources = []
    for skill in seed_skills:
        relative = source_ref_for(skill)
        path = root / relative
        if framework_state_io is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(skill.skill_md, encoding="utf-8")
        else:
            framework_state_io.write_text_once(path, skill.skill_md)
        sources.append(LoadedSkillV11(relative, skill.content_sha256, skill))
    snapshot, manifest = build_v11_snapshot(
        sources, snapshot_id="FIT-seed-" + run_identity[:16],
        require_complete=True, created_by="fitset_isolated_seed",
    )
    if framework_state_io is None:
        write_v11_snapshot(root, snapshot, manifest, activate=True)
        _write_create_once(marker, identity)
    else:
        framework_state_io.write_snapshot(root, snapshot, manifest, activate=True)
        framework_state_io.write_json_once(marker, identity)


def collect_fitset_case(variant: Mapping, outcome: Mapping,
                        parent: SkillSpecV11, label: Mapping,
                        trace_ref: str, *, experiment_split: Mapping | None = None,
                        ) -> FitSetExperienceCase:
    """只从完成轨迹形成 learning case，不把框架故障归为方法经验。"""
    authorization = (
        learning_authorization_for(experiment_split, [variant["qa_id"]])
        if experiment_split is not None else {})
    if authorization:
        _require(all(variant.get(field, "learning") == "learning"
                     for field in ("experiment_role", "experiment_split")),
                 "Validation role cannot enter learning case construction")
        _require(variant.get("split") in {"induction", "inner_validation"}
                 and variant.get("source_split", variant["split"]) == variant["split"],
                 "Authorized learning must preserve the real source_split")
    key = f"{parent.skill_id}@{parent.version}"
    _require(outcome.get("delivered_skill_versions") == [key],
             "实际 Skill delivery 与槽位 parent 不一致")
    selection = outcome.get("initial_skill_selection") or {}
    _require(
        selection.get("skill_version") == key
        and selection.get("content_sha256") == parent.content_sha256,
        "初始 Skill identity/hash 不匹配",
    )
    rounds = outcome.get("rounds") or []
    _require(any(row.get("executed") for row in rounds), "轨迹未执行任何程序")
    _require(not any(
        (row.get("execution_diagnostics") or {}).get("error_message")
        == "cell timeout requires a main-thread SIGALRM runtime"
        for row in rounds
    ), "SIGALRM 基础设施错误不得进入 Skill buffer")
    _require(not any(row.get("service_errors") for row in rounds),
             "服务错误不得进入 Skill buffer")
    _require(any(str(row.get("response_text") or "").strip() for row in rounds),
             "轨迹缺少模型回复")
    _require(
        label["scene_id"] == variant["scene_id"]
        and label["question_type"] == variant["question_type"],
        "离线 label identity 改变",
    )
    scored = score_answer(
        answer=outcome.get("answer"), task=parent.question_type,
        options=label["options"], reference=label["reference_answer"],
        available=outcome.get("final_state") == "answer",
    )
    initial = outcome["initial_evidence_states"]
    history = list(outcome["evidence_history"])
    score = scored["score"]
    return FitSetExperienceCase(
        case_id=content_sha256({
            "variant": variant["variant_id"], "parent": key,
            "outcome": content_sha256(outcome),
        }),
        qa_id=variant["qa_id"], scene_id=variant["scene_id"],
        question_type=parent.question_type, evidence_state=parent.evidence_state,
        source_split=variant["split"],
        learning_authorization=authorization,
        outcome="success" if score == 1 else "failure",
        delivered_skill_version=key, delivered_content_sha256=parent.content_sha256,
        delivered_skill_md=parent.skill_md, initial_evidence_states=initial,
        evidence_history=history,
        score=score,
        failure_categories=[] if score == 1 else [
            str(outcome.get("failure_code") or "answer_incorrect")],
        trace_ref=trace_ref, trace_sha256=content_sha256(outcome),
        process_facts=extract_process_facts(outcome),
        process_signature={
            "final_state": outcome.get("final_state"),
            "answer_source": outcome.get("answer_source"),
            "round_error_codes": sorted({
                str(row["error_code"]) for row in rounds if row.get("error_code")}),
            "model_rounds": len(rounds),
            "tool_calls": sum(len(row.get("results") or []) for row in rounds),
            "resolved_tool_object_count": len({
                str(obj) for row in rounds for observation in row.get("results") or []
                for obj in observation.get("resolved_object_ids", [])
            }),
        },
    )


def sanitized_slot_feedback(cases) -> dict:
    """发送给 reviser 的聚合信息不含题目、ID、选项或答案。"""
    _require(bool(cases), "空反馈")
    slots = {case.evidence_state + "/" + case.question_type for case in cases}
    parents = {case.delivered_skill_version for case in cases}
    _require(len(slots) == len(parents) == 1, "反馈混入不同槽位或 parent")
    signatures = []
    for case in cases:
        facts = case.process_facts or legacy_process_facts(case.process_signature)
        signatures.append({
            "score_band": score_band(case.score, facts.execution.answer_submitted),
            "process": safe_process_dict(facts.model_dump(mode="json")),
        })
    counts = Counter(json.dumps(row, sort_keys=True) for row in signatures)
    body = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "whitelist_version": FEEDBACK_WHITELIST_VERSION,
        "field_semantics": dict(FIELD_SEMANTICS),
        "question_type": cases[0].question_type,
        "evidence_state": cases[0].evidence_state,
        "n_cases": len(cases),
        "mean_score": sum(case.score for case in cases) / len(cases),
        "score_bands": dict(sorted(Counter(row["score_band"] for row in signatures).items())),
        "signatures": [
            {**json.loads(signature), "n": n,
             "signature_ref": content_sha256(json.loads(signature))}
            for signature, n in sorted(counts.items())
        ],
    }
    return {**body, "identity_sha256": content_sha256(body)}


class FitSetWorkerEvolution:
    """唯一 worker 消费所属槽位，批次完成后同步尝试并切换 parent。"""

    def __init__(self, runner, worker_index: int, owned_variants):
        self.runner = runner
        self.config = runner.evolution_config
        self.root = runner.root / "workers" / f"{worker_index:02d}"
        self.library = self.root / "library"
        self.owned_slots = sorted({
            fit_slot_key(row["question_type"], row["evidence_state"])
            for row in owned_variants
        })
        initialize_worker_library(
            self.library, seed_skills=runner.seed_skills,
            owned_slots=self.owned_slots, run_identity=runner.run_identity,
            framework_state_io=getattr(runner, "framework_state_io", None),
        )
        self.stream = FitSetStreamingBuffer(
            self.root / "buffer",
            stream_id=f"fit-worker-{worker_index:02d}-{runner.run_identity[:12]}",
            config=StreamingEvolutionConfig(
                batch_size=self.config.batch_size, tail_for_unattempted_slots=False),
            framework_state_io=getattr(runner, "framework_state_io", None),
        )
        self.rows = {row["variant_id"]: row for row in runner.variants}
        self.rows.update(getattr(runner, "gate_variant_rows", {}))

    def _write_create_once(self, path, value):
        if getattr(self.runner, "framework_state_io", None) is None:
            return _write_create_once(path, value)
        return self.runner._write_create_once(path, value)

    def provider(self, variant):
        parents, snapshot = _active(self.library)
        slot = (variant["question_type"], variant["evidence_state"])
        _require(fit_slot_key(*slot) in self.owned_slots, "worker 试图访问非所属槽位")
        parent = parents[slot]
        return [parent], {
            "mode": "fitset_slot_stream",
            "snapshot_ref": snapshot["snapshot_id"],
            "manifest_sha256": snapshot["manifest_sha256"],
            "slot_key": parent.slot_key,
        }

    def ingest_result(self, variant, row):
        authorization = self.runner.learning_authorization(variant)
        outcome_path = (self.runner.root / "variants"
                        / _variant_key(variant["variant_id"]) / "outcome.json")
        trace_ref = self.runner.logical_path(outcome_path)
        outcome = json.loads(outcome_path.read_text())
        identity = row["skill_identity"]
        snapshot = read_snapshot(self.library / "snapshots", identity["snapshot_ref"])
        specs = validate_v11_snapshot(snapshot, library_root=self.library)
        parent = next(spec for spec in specs
                      if spec.slot_key == fit_slot_key(
                          variant["question_type"], variant["evidence_state"]))
        accessed_at = datetime.now(timezone.utc).isoformat()
        label = self.runner.label_metadata[variant["qa_id"]]
        case = collect_fitset_case(
            variant, outcome, parent, label, trace_ref,
            experiment_split=self.runner.experiment_split)
        if self.config.learning_input_mode != "aggregate_v1":
            from skill3d.evolution.case_records_v11 import case_from_fitset_outcome

            source = self.runner.offline_case_input(variant)
            canonical_path = self.root / "canonical_cases" / f"{case.case_id}.json"
            if canonical_path.exists():
                full_case = V11ExperienceCase.model_validate_json(canonical_path.read_bytes())
                _require(full_case.delivered_content_sha256 == parent.content_sha256
                         and full_case.qa_id == variant["qa_id"]
                         and full_case.trace_ref == trace_ref
                         and full_case.learning_authorization == authorization,
                         "canonical learning case identity changed")
            else:
                full_case = case_from_fitset_outcome(
                    variant=variant, outcome=outcome, parent=parent, label=label,
                    trace_ref=trace_ref, question_text=source["question_text"],
                    options=source["options"], frame_set_hash=source["frame_set_hash"],
                    image_refs=source["image_refs"],
                    experiment_split=self.runner.experiment_split,
                )
                self._write_create_once(canonical_path, full_case.model_dump(mode="json"))
            case = FitSetExperienceCase.model_validate({
                **case.model_dump(mode="json"),
                "full_case": full_case.model_dump(mode="json"),
                "score": full_case.score, "outcome": full_case.outcome,
                "failure_categories": full_case.failure_categories,
            })
        access_path = self.root / "label_access" / f"{case.case_id}.json"
        access = {
                "experiment_role": "fit_set_learning",
                "source_split": variant["split"], "label_access": True,
                "accessed_at": accessed_at,
                "qa_id": case.qa_id, "trace_sha256": content_sha256(outcome),
                "reference_answer_sha256": hashlib.sha256(
                    label["reference_answer"].encode()).hexdigest(),
        }
        if authorization:
            access["learning_authorization"] = authorization
        if access_path.exists():
            prior = json.loads(access_path.read_text())
            access["accessed_at"] = prior["accessed_at"]
        self._write_create_once(access_path, access)
        result = self.stream.ingest(
            case, parent_snapshot_id=identity["snapshot_ref"], attempt=self.attempt)
        if result is None:
            result = self.stream.attempt_ready(parent.slot_key, attempt=self.attempt)
        if result is not None and result.outcome == "deferred":
            raise FitSetError(f"{parent.slot_key}: {result.reason}")

    def recover(self):
        for key in sorted(self.stream.slots):
            slot = self.stream.slots[key]
            if slot.deferred_reason:
                raise FitSetError(f"{key}: {slot.deferred_reason}")
            result = self.stream.attempt_ready(key, attempt=self.attempt)
            if result is not None and result.outcome == "deferred":
                raise FitSetError(f"{key}: {result.reason}")

    def _parent_for(self, request):
        snapshot = read_snapshot(self.library / "snapshots", request.parent_snapshot_id)
        specs = validate_v11_snapshot(snapshot, library_root=self.library)
        parent = next(spec for spec in specs if spec.slot_key == request.slot_key)
        _require(
            f"{parent.skill_id}@{parent.version}" == request.parent_skill_version
            and parent.content_sha256 == request.parent_content_sha256,
            "attempt parent 身份不一致",
        )
        return parent

    def _gate(self, request, parent, candidate, attempt_root):
        panel_ids = self.runner.gate_panels[request.slot_key]
        if self.config.gate_mode == "qa_disjoint":
            learning_qa = {row["qa_id"] for row in self.runner.variants}
            _require(not learning_qa & {self.rows[value]["qa_id"] for value in panel_ids},
                     "gate QA overlaps learning QA")
            _require(all(
                self.runner.experiment_split["qa_roles"].get(self.rows[value]["qa_id"])
                == "validation" for value in panel_ids),
                "gate contains non-validation QA")
            expected = [row["variant_id"] for row in self.runner.gate_variant_rows.values()
                        if row["slot_key"] == request.slot_key]
            _require(panel_ids == expected, "qa_disjoint must use all validation variants")
        panel_size = (
            len({self.rows[value]["qa_id"] for value in panel_ids})
            if self.config.gate_mode == "qa_disjoint" else len(panel_ids))
        if panel_size < self.config.gate_size:
            body = {
                "passed": False, "conditions": {"gate_size": False},
                "reason": "槽位题量不足冻结 gate_size", "arms": [],
                "panel_ids": panel_ids, "candidate_hash": candidate.content_sha256,
                "parent_hash": parent.content_sha256,
                "run_identity": self.runner.run_identity,
            }
            return {**body, "identity_sha256": content_sha256(body)}
        # Phase 1 — set up each (seed, arm) runner deterministically and collect
        # every panel episode as an independent work item. Setup writes the
        # run-local library and frozen identity; it performs no model request.
        arm_runners: dict[tuple[int, str], object] = {}
        episodes: list[tuple[object, Mapping]] = []
        for seed in self.config.gate_seeds:
            for name, spec in (("parent", parent), ("candidate", candidate)):
                output = attempt_root / "gate" / str(seed) / name
                evaluation_skills = [
                    spec if value.slot_key == spec.slot_key else value
                    for value in self.runner.seed_skills
                ]
                initialize_worker_library(
                    output / "library", seed_skills=evaluation_skills,
                    owned_slots=[spec.slot_key],
                    run_identity=content_sha256({
                        "run": self.runner.run_identity, "seed": seed,
                        "skill": spec.model_dump(mode="json"),
                    }),
                    framework_state_io=self.runner.framework_state_io,
                )
                _, evaluation_snapshot = _active(output / "library")
                gate_runner = copy.copy(self.runner)
                gate_runner.root = output
                gate_runner.deployment_continuation = None
                gate_runner._continuation = None
                gate_runner._request_governor_root = self.runner.request_governor_root
                gate_runner._request_category = "gate"
                gate_runner.evolution_config = None
                gate_runner._qa_disjoint_gate = self.config.gate_mode == "qa_disjoint"
                gate_runner.capture_offline_case_input = False
                gate_runner.label_metadata = {}
                gate_runner.experiment_split = None
                gate_runner.gate_deferred_reasons = {}
                if gate_runner.published_inputs is not None or gate_runner._qa_disjoint_gate:
                    gate_runner.input_contract_identity = copy.deepcopy(
                        self.runner.input_contract_identity)
                    gate_runner.input_contract_identity.pop("offline_metadata", None)
                gate_runner.seed_skills = []
                gate_runner.gate_panels = {}
                gate_runner.revision_client_factory = None
                gate_runner.gate_client_factory = None
                gate_runner.skills_for_variant = FrozenSkillProvider([spec], {
                    "snapshot_ref": evaluation_snapshot["snapshot_id"],
                    "manifest_sha256": evaluation_snapshot["manifest_sha256"],
                    "skill_version": f"{spec.skill_id}@{spec.version}",
                })
                gate_runner.workers = 1
                gate_runner.client_factory = self.runner.gate_client_factory(seed, output)
                binding_seeds = {
                    variant_id: self.runner.binding_refs[variant_id]["request_identity"]["seed"]
                    for variant_id in panel_ids
                } if getattr(self.runner, "published_inputs", None) is not None else {}
                frozen = {
                    "schema_version": FIT_STREAM_SCHEMA,
                    "seed": seed,
                    "binding_seed": (
                        next(iter(set(binding_seeds.values())))
                        if binding_seeds and len(set(binding_seeds.values())) == 1
                        else None if binding_seeds else self.runner.base_cfg.seed),
                    "snapshot": evaluation_snapshot,
                    "variant_ids": panel_ids, "spec": spec.model_dump(mode="json"),
                    "run_identity": self.runner.run_identity,
                    "panel_rows_sha256": content_sha256(
                        [self.rows[value] for value in panel_ids]),
                }
                if binding_seeds:
                    frozen["binding_seed_by_variant"] = binding_seeds
                self._write_create_once(output / "gate_identity.json", frozen)
                arm_runners[(seed, name)] = gate_runner
        schedule = []
        for seed_index, seed in enumerate(self.config.gate_seeds):
            for panel_index, variant_id in enumerate(panel_ids):
                order = ("parent", "candidate") if (seed_index + panel_index) % 2 == 0 else (
                    "candidate", "parent")
                for name in order:
                    episodes.append((arm_runners[(seed, name)], self.rows[variant_id]))
                    schedule.append({"seed": seed, "arm": name, "variant_id": variant_id})
        self._write_create_once(attempt_root / "gate_schedule.json", {
            "schema_version": "gate-balanced-order/1.0", "submission_order": schedule,
            "concurrency_limit": self.runner.episode_concurrency,
            "completion_order_guaranteed": False,
        })
        # Phase 2 — run every episode under the global permit. Each result lands
        # in its own arm directory; completed results from a prior attempt are
        # reused, so a resumed gate relaunches only the missing episodes.
        self.runner._run_gate_episodes(episodes)
        # Phase 3 — score the persisted results. This logic is identical to the
        # previous serial gate; only the order episodes ran in has changed.
        conditions = {}
        arms = []
        for seed in self.config.gate_seeds:
            pair = {}
            for name, spec in (("parent", parent), ("candidate", candidate)):
                gate_runner = arm_runners[(seed, name)]
                output = gate_runner.root
                rows = []
                for variant_id in panel_ids:
                    variant = self.rows[variant_id]
                    path = gate_runner._result_path(variant_id)
                    _require(path.exists(), "gate 执行后缺少 result")
                    row = json.loads(path.read_text())
                    _require(row["question_type"] == request.question_type
                             and row["evidence_state"] == request.evidence_state,
                             "gate result 槽位错配")
                    label = self.runner.label_metadata[variant["qa_id"]]
                    score = score_answer(
                        answer=row.get("answer"), task=spec.question_type,
                        options=label["options"], reference=label["reference_answer"],
                        available=row.get("status") == "completed"
                        and row.get("final_state") == "answer",
                    )
                    outcome = json.loads((output / "variants"
                                          / _variant_key(variant_id)
                                          / "outcome.json").read_text())
                    selection = outcome.get("initial_skill_selection") or {}
                    delivered = (
                        outcome.get("delivered_skill_versions")
                        == [f"{spec.skill_id}@{spec.version}"]
                        and selection.get("content_sha256") == spec.content_sha256
                        and selection.get("skill_version")
                        == f"{spec.skill_id}@{spec.version}")
                    seed_paths = list(
                        (output / "request_seeds").glob(
                            _variant_key(variant_id) + "-*.json"))
                    returned_models = sorted({
                        str((json.loads(path.read_text()).get(
                            "response_identity") or {}).get("model") or "")
                        for path in seed_paths
                        if "-response-" in path.name
                    })
                    responses = [json.loads(path.read_text()) for path in seed_paths
                                 if "-response-" in path.name]
                    fingerprints = sorted({
                        str((record.get("response_identity") or {}).get("backend_fingerprint"))
                        for record in responses
                        if (record.get("response_identity") or {}).get("backend_fingerprint") is not None
                    })
                    input_plan = json.loads((output / "variants"
                                             / _variant_key(variant_id)
                                             / "plan.json").read_text())
                    shared_plan = {
                        key: value for key, value in input_plan.items()
                        if key != "skill_identity"
                    }
                    rows.append({
                        "score": score["score"], "legal": score["legal_answer"],
                        "error": row.get("final_state") != "answer",
                        "program_error": any(r.get("error_code")
                                             for r in outcome.get("rounds") or []),
                        "untrusted": bool(outcome.get("answer_untrusted")),
                        "delivered": delivered,
                        "service_error": bool(row.get("service_errors")) or any(
                            record.get("synthesis_source") == "online_model_service_error"
                            for record in outcome.get("rounds") or []),
                        "seed_observed": bool(seed_paths) and all(
                            json.loads(path.read_text()).get("request_seed") == seed
                            for path in seed_paths),
                        "shared_input_sha256": content_sha256(shared_plan),
                        "returned_models": returned_models,
                        "backend_fingerprints": fingerprints,
                        "process_metrics": episode_process_metrics(outcome, responses),
                    })
                pair[name] = {
                    "mean_score": sum(r["score"] for r in rows) / len(rows),
                    "errors": sum(r["error"] for r in rows),
                    "program_errors": sum(r["program_error"] for r in rows),
                    "legal_rate": sum(r["legal"] for r in rows) / len(rows),
                    "trusted": not any(r["untrusted"] for r in rows),
                    "delivered": all(r["delivered"] for r in rows),
                    "service_ok": not any(r["service_error"] for r in rows),
                    "seed_observed": all(r["seed_observed"] for r in rows),
                    "shared_input_hashes": [
                        row["shared_input_sha256"] for row in rows],
                    "returned_models": sorted({
                        model for row in rows for model in row["returned_models"]}),
                    "backend_fingerprints": sorted({
                        value for row in rows for value in row["backend_fingerprints"]}),
                    "process_metrics": aggregate_gate_process_metrics(
                        [row["process_metrics"] for row in rows]),
                }
            pa, ca = pair["parent"], pair["candidate"]
            conditions[str(seed)] = (
                ca["mean_score"] > pa["mean_score"]
                and ca["errors"] <= pa["errors"]
                and ca["program_errors"] <= pa["program_errors"]
                and ca["legal_rate"] >= pa["legal_rate"]
                and pa["shared_input_hashes"] == ca["shared_input_hashes"]
                and pa["returned_models"] == ca["returned_models"]
                and pa["backend_fingerprints"] == ca["backend_fingerprints"]
                and all(arm[key] for arm in (pa, ca) for key in (
                    "trusted", "delivered", "service_ok", "seed_observed"))
            )
            arms.append({"seed": seed, **pair})
        body = {"passed": all(conditions.values()), "conditions": conditions,
                "arms": arms, "panel_ids": panel_ids,
                "schedule_sha256": content_sha256(schedule),
                "admission_program_error_rule": "legacy_any_round_error_code",
                "missing_backend_fingerprint_means": "unknown_backend_identity",
                "candidate_hash": candidate.content_sha256,
                "parent_hash": parent.content_sha256,
                "run_identity": self.runner.run_identity}
        return {**body, "identity_sha256": content_sha256(body)}

    def attempt(self, request):
        self.runner._require_storage_reserve(request.slot_key)
        _require(request.slot_key in self.owned_slots, "attempt 非所属槽位")
        token = content_sha256({
            "slot": request.slot_key, "parent": request.parent_skill_version,
            "cases": [case.case_id for case in request.cases],
            "attempt": request.attempt_index,
        })[:20]
        root = self.root / "campaigns" / token
        result_path = root / "result.json"
        if result_path.exists():
            return StreamingAttemptResult(**json.loads(result_path.read_text()))
        continuation = getattr(self.runner, "_continuation", None)
        revision_root = (root if continuation is None else
                         root / "deployments" / continuation.deployment_id)
        pending_intents = False
        if continuation is not None:
            namespaces = [root, *sorted((root / "deployments").glob("*"))]
            for namespace in namespaces:
                intents = list(namespace.glob("revision/slots/*/*/request_intent_*.json"))
                pending_intents |= bool(
                    intents or list(namespace.glob("diagnosis/*/request_intent.json")))
                if namespace == revision_root:
                    continue
                for intent in intents:
                    exchange = intent.with_name(intent.name.replace("request_intent_", "attempt_"))
                    if not exchange.is_file():
                        raise RevisionOperationalError(
                            "continuation cannot retry an ambiguous original revision request")
                    _require(intent.resolve().is_relative_to(self.root.resolve())
                             and exchange.resolve().is_relative_to(self.root.resolve()),
                             "continuation revision receipt escaped the worker locator")
                    prior = json.loads(intent.read_bytes())
                    _require(json.loads(exchange.read_bytes()).get("request") == prior["request"],
                             "continuation original intent/exchange identity changed")
                    if prior["request"]["prompt_version"] == FIT_REVISION_PROMPT_VERSION:
                        raise RevisionOperationalError(
                            "continuation cannot replay an unresolved request under the same protocol")
        parent = self._parent_for(request)
        if self.config.gate_mode == "qa_disjoint":
            for case in request.cases:
                validate_learning_authorization(
                    case.learning_authorization, [case.qa_id], self.runner.experiment_split)
            reason = self.runner.gate_deferred_reasons.get(request.slot_key)
            if reason:
                # Admission is deferred, not the learning stream's batch window.
                result = StreamingAttemptResult(
                    outcome="no_change", campaign_id=token, reason=reason)
                self._write_create_once(root / "gate_deferred.json", {
                    "admission_status": "deferred", "learning_continues": True,
                    "reason": reason, "gate_size": self.config.gate_size,
                    "panel_ids": self.runner.gate_panels[request.slot_key],
                    "run_identity": self.runner.run_identity,
                    "parent_hash": parent.content_sha256,
                    "case_ids": [case.case_id for case in request.cases],
                })
                self._write_create_once(result_path, asdict(result))
                return result
        feedback = sanitized_slot_feedback(request.cases)
        self._write_create_once(root / "feedback.json", feedback)
        self._write_create_once(root / "request.json", {
            "slot": request.slot_key, "parent": parent.model_dump(mode="json"),
            "parent_content_sha256": parent.content_sha256,
            "case_ids": [case.case_id for case in request.cases],
            "new_case_count": request.new_case_count,
            "run_identity": self.runner.run_identity,
        })
        schedule_path = revision_root / "schedule.json"
        if schedule_path.exists():
            schedule = json.loads(schedule_path.read_text())
            _require(schedule.get("identity_sha256") == content_sha256({
                key: value for key, value in schedule.items() if key != "identity_sha256"
            }), "revision schedule receipt changed")
        else:
            prior_patterns = set()
            revision_requests_used = revision_intents_used(self.root, request.slot_key)
            for path in sorted((self.root / "campaigns").glob("*/request.json")):
                if path.parent == root:
                    continue
                prior = json.loads(path.read_text())
                if prior["slot"] != request.slot_key:
                    continue
                prior_schedule = path.parent / "schedule.json"
                if (prior["parent_content_sha256"] == parent.content_sha256
                        and prior_schedule.exists()):
                    receipt = json.loads(prior_schedule.read_text())
                    _require(receipt.get("identity_sha256") == content_sha256({
                        key: value for key, value in receipt.items() if key != "identity_sha256"
                    }), "prior revision schedule receipt changed")
                    if receipt["scheduled"]:
                        prior_patterns.update(receipt["pattern_refs"])
            entries = [
                json.loads(path.read_text())
                for path in (self.root / "candidate_index").glob("*.json")
            ]
            schedule = scheduling_decision(
                feedback=feedback, previous_patterns=prior_patterns,
                mode=(self.config.scheduling_mode
                      if self.config.learning_input_mode == "aggregate_v1" else "batch_v1"),
                revision_requests_used=revision_requests_used,
                gate_candidates_used=sum(entry["slot"] == request.slot_key for entry in entries),
                max_revision_requests=self.config.max_revision_requests_per_slot,
                max_gate_candidates=self.config.max_gate_candidates_per_slot,
            )
            schedule["requested_mode"] = self.config.scheduling_mode
            schedule["learning_input_mode"] = self.config.learning_input_mode
            diagnosis_cost = int(self.config.learning_input_mode.startswith("diagnosis"))
            if schedule["scheduled"] and schedule["remaining_revision_requests"] <= diagnosis_cost:
                schedule["scheduled"] = False
                schedule["reason_code"] = "revision_request_budget_exhausted"
            schedule["revision_attempt_budget"] = min(
                self.config.max_revision_attempts,
                max(0, schedule["remaining_revision_requests"] - diagnosis_cost))
            schedule["identity_sha256"] = content_sha256(schedule)
            self._write_create_once(schedule_path, schedule)
        if not schedule["scheduled"]:
            if continuation is not None and pending_intents:
                raise RevisionOperationalError(
                    "continuation cannot turn an unresolved request into a method result")
            result = StreamingAttemptResult(
                outcome="no_change", campaign_id=token, reason=schedule["reason_code"])
            self._write_create_once(result_path, asdict(result))
            return result
        forbidden = {str(value) for row in self.runner.variants
                     for value in (row["qa_id"], row["scene_id"], row["variant_id"])}
        reviser = FitSetSlotReviser(
            output_root=revision_root / "revision", panel=self.runner.panel,
            panel_name=self.runner.panel_name, variants=[],
            accumulation_root=self.runner.root,
            label_metadata={}, library_root=self.library,
            client=self.runner._govern_client(
                self.runner.revision_client_factory(), "reviser",
                {"slot_key": request.slot_key, "campaign_id": token,
                 "run_identity": self.runner.run_identity},
            ),
            max_attempts=schedule["revision_attempt_budget"],
            max_tokens=self.config.max_tokens,
            method_context_max_chars=self.runner.base_cfg.retrieval_policy.method_context_max_chars,
            experiment_split=self.runner.experiment_split,
            input_max_bytes=self.config.revision_input_max_bytes,
            input_max_chars=self.config.revision_input_max_chars,
            evidence_request_budget=max(0, schedule["remaining_revision_requests"]
                                        - schedule["revision_attempt_budget"]
                                        - int(self.config.learning_input_mode.startswith("diagnosis"))),
        )
        if self.runner.framework_state_io is not None:
            self.runner.framework_state_io.adapt_reviser(reviser)
        _require(not leakage_scan_text(
            json.dumps(feedback), forbidden_sample_ids=forbidden),
                 "offline prompt 泄漏扫描失败")
        experience = None
        diagnosis = None
        if self.config.learning_input_mode != "aggregate_v1":
            full_cases = [case.full_case for case in request.cases]
            _require(all(case is not None for case in full_cases),
                     "complete learning mode requires canonical cases")
            bundle_path = root / "experience_bundle.json"
            if bundle_path.exists():
                experience = V11ExperienceBundle.model_validate_json(bundle_path.read_bytes())
                _require([case.case_id for case in experience.cases]
                         == [case.case_id for case in full_cases],
                         "canonical experience changed during resume")
            else:
                accesses = []
                for source in sorted({case.source_split for case in full_cases}):
                    source_cases = [case for case in full_cases if case.source_split == source]
                    accesses.append(DataAccessRecord(
                        record_id=f"fit-label-access-{token}-{source}",
                        at=datetime.now(timezone.utc).isoformat(), split=source,
                        purpose="skill_induction", component_role="inducer",
                        run_id=token, n_items=len(source_cases), label_access=True,
                        input_manifest_sha256=content_sha256(
                            [case.model_dump(mode="json") for case in source_cases]),
                        source_refs=[case.trace_ref for case in source_cases],
                        notes=["Authorized learning question labels only; no GT 3D access"],
                        learning_authorization=(
                            learning_authorization_for(self.runner.experiment_split, [
                                case.qa_id for case in source_cases])
                            if self.runner.experiment_split is not None else {}),
                    ))
                sources = [access.split for access in accesses]
                experience = V11ExperienceBundle(
                    campaign_id=token, parent_snapshot_id=request.parent_snapshot_id,
                    parent_skill_version=request.parent_skill_version,
                    parent_content_sha256=parent.content_sha256,
                    question_type=parent.question_type, evidence_state=parent.evidence_state,
                    source_split=sources[0] if len(sources) == 1 else sources,
                    split="learning", label_access=True,
                    learning_authorization=(
                        learning_authorization_for(self.runner.experiment_split, [
                            case.qa_id for case in full_cases])
                        if self.runner.experiment_split is not None else {}),
                    source_run_ref=str(root), label_access_record=accesses[0],
                    label_access_records=accesses if self.runner.experiment_split is not None else [],
                    cases=full_cases,
                )
                self._write_create_once(bundle_path, experience.model_dump(mode="json"))
            if self.config.learning_input_mode.startswith("diagnosis"):
                from skill3d.evolution.diagnosis_v11 import (
                    FrozenCaseImageProvider, OfflineDiagnosisConfig, V11CaseDiagnoser,
                )

                multimodal = self.config.learning_input_mode == "diagnosis_multimodal_v1"
                diagnoser = V11CaseDiagnoser(
                    self.runner._govern_client(
                        self.runner.revision_client_factory(), "diagnosis",
                        {"slot_key": request.slot_key, "campaign_id": token,
                         "run_identity": self.runner.run_identity}),
                    revision_root / "diagnosis",
                    OfflineDiagnosisConfig(
                        mode="multimodal" if multimodal else "text",
                        max_cases=self.config.diagnosis_max_cases,
                        max_input_chars=self.config.diagnosis_max_input_chars,
                        max_images=self.config.diagnosis_max_images,
                        max_output_tokens=self.config.diagnosis_max_output_tokens),
                )
                if self.runner.framework_state_io is not None:
                    self.runner.framework_state_io.adapt_diagnoser(
                        diagnoser, {"campaign_id": token})
                diagnosis = diagnoser.diagnose(
                    parent, experience, context={"campaign_id": token},
                    image_provider=FrozenCaseImageProvider(
                        [self.runner.root]) if multimodal else None,
                )
        decision_path = root / "decision.json"
        try:
            candidate = reviser._revise_slot(
                slot=(request.question_type, request.evidence_state),
                parent=parent, feedback=feedback, forbidden_ids=forbidden,
                experience=experience, diagnosis=diagnosis)
        except RevisionInputBudgetExhausted as exc:
            self._write_create_once(root / "input_budget_refusal.json", {
                "schema": "revision-input-budget-refusal/1",
                "source_bundle_sha256": exc.source_sha256,
                "needed_batches": exc.needed, "available_counted_requests": exc.available,
                "model_requests_initiated": 0, "method_assessed": False,
                "parent_preserved": True, "canonical_cases_retained": True,
            })
            result = StreamingAttemptResult(
                outcome="no_change", campaign_id=token,
                reason="revision_input_request_budget_exhausted")
        except FitSetError as exc:
            result = StreamingAttemptResult(
                outcome="rejected", campaign_id=token, reason=str(exc))
        else:
            if candidate is None:
                result = StreamingAttemptResult(
                    outcome="no_change", campaign_id=token,
                    reason="Current process evidence does not support a method revision")
                self._write_create_once(result_path, asdict(result))
                return result
            dedup_key = candidate_dedup_key(
                slot=request.slot_key, parent_hash=parent.content_sha256,
                skill_md=candidate.skill_md)
            index_path = self.root / "candidate_index" / f"{dedup_key}.json"
            entry = {
                "slot": request.slot_key, "parent_sha256": parent.content_sha256,
                "method_sha256": normalized_method_sha256(candidate.skill_md),
                "candidate_sha256": candidate.content_sha256,
                "campaign_id": token, "dedup_key": dedup_key,
            }
            if index_path.exists():
                previous = json.loads(index_path.read_text())
                _require(all(previous[key] == entry[key] for key in (
                    "slot", "parent_sha256", "method_sha256", "dedup_key")),
                    "candidate index method identity changed")
                if previous["campaign_id"] != token:
                    prior_decision_path = (
                        self.root / "campaigns" / previous["campaign_id"] / "decision.json")
                    _require(prior_decision_path.exists(),
                             "重复候选的原 A/B 未完成，不能伪造评估或重新请求")
                    prior_decision = json.loads(prior_decision_path.read_text())
                    prior_body = {key: value for key, value in prior_decision.items()
                                  if key != "identity_sha256"}
                    _require(prior_decision["identity_sha256"] == content_sha256(prior_body)
                             and prior_decision["candidate_hash"] == previous["candidate_sha256"]
                             and previous["parent_sha256"] == parent.content_sha256,
                             "重复候选原始证据身份改变")
                    self._write_create_once(root / "duplicate.json", {
                        **entry, "original_campaign_id": previous["campaign_id"],
                        "original_decision_sha256": content_sha256(prior_decision),
                        "gate_reexecuted": False,
                    })
                    result = StreamingAttemptResult(
                        outcome="duplicate", campaign_id=token,
                        reason="Identical method under the same parent; original gate retained")
                    self._write_create_once(result_path, asdict(result))
                    return result
                _require(previous == entry, "candidate index identity changed")
            else:
                self._write_create_once(index_path, entry)
            if decision_path.exists():
                decision = json.loads(decision_path.read_text())
                body = {key: value for key, value in decision.items()
                        if key != "identity_sha256"}
                _require(decision.get("identity_sha256") == content_sha256(body),
                         "A/B decision hash 改变")
                _require(decision.get("candidate_hash") == candidate.content_sha256
                         and decision.get("parent_hash") == parent.content_sha256
                         and decision.get("run_identity") == self.runner.run_identity,
                         "A/B decision candidate/parent 身份改变")
            else:
                decision = self._gate(request, parent, candidate, root)
                self._write_create_once(decision_path, decision)
            if not decision["passed"]:
                result = StreamingAttemptResult(
                    outcome="rejected", campaign_id=token, reason="严格 A/B 未提升")
            else:
                active, active_snapshot = _active(self.library)
                slot = (request.question_type, request.evidence_state)
                current = active[slot]
                if current.content_sha256 != candidate.content_sha256:
                    _require(current == parent, "publish 前 parent 移动")
                    proposal = SkillCandidateV11(
                        candidate_id=token,
                        parent_snapshot_id=active_snapshot["snapshot_id"],
                        parent_skill_version=request.parent_skill_version,
                        full_skill_spec=candidate,
                        modification_reason="fit-set dual-seed strict A/B passed",
                        source_run_ref=str(root),
                    )
                    if self.runner.framework_state_io is None:
                        publish_v11_candidate(self.library, proposal)
                    else:
                        self.runner.framework_state_io.publish_candidate(self.library, proposal)
                active, after = _active(self.library)
                _require(active[slot] == candidate, "发布后正常 loader/hash 验证失败")
                for other in ALL_FIT_SLOTS:
                    if fit_slot_key(*other) not in self.owned_slots:
                        seed = next(s for s in self.runner.seed_skills
                                    if s.slot_key == fit_slot_key(*other))
                        _require(active[other] == seed, "worker 修改了非所属槽位")
                result = StreamingAttemptResult(
                    outcome="promoted", campaign_id=token,
                    active_snapshot_id=after["snapshot_id"],
                    active_skill_version=f"{candidate.skill_id}@{candidate.version}",
                    active_content_sha256=candidate.content_sha256,
                )
                self._write_create_once(root / "post_load.json", {
                    "verified": True, "snapshot": after, "slot": request.slot_key,
                    "version": result.active_skill_version,
                    "content_sha256": candidate.content_sha256,
                })
        self._write_create_once(result_path, asdict(result))
        return result


def freeze_fitset_stream(system_root: str | Path, output_dir: str | Path,
                        *, framework_state_io=None, system_runner=None) -> dict:
    """合并唯一 owner 的最后发布版本；不产生额外终版修订。"""
    from skill3d.evaluation.fitset_v11 import (
        build_fit_skill_bundle,
    )

    root = Path(system_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    body = {key: value for key, value in manifest.items() if key != "identity_sha256"}
    _require(manifest.get("identity_sha256") == content_sha256(body)
             and summary.get("manifest_identity_sha256") == manifest["identity_sha256"],
             "freeze manifest 身份/hash 不符")
    _require(manifest.get("phase") == "accumulation"
             and summary.get("status") == "complete"
             and summary.get("evolution_finished") is True,
             "积累/演化未完成不能 freeze")
    evolution = manifest.get("evolution") or {}
    _require(bool(evolution), "不能冻结旧的一次性 fit-set 路径")
    seeds = [SkillSpecV11.model_validate(row) for row in evolution["seed_skills"]]
    seed_by_slot = {(s.question_type, s.evidence_state): s for s in seeds}
    _require(len(seed_by_slot) == 24, "freeze seed 缺槽位")
    selected = set(manifest["execution"]["slot_ownership"])
    changed = {}
    receipts = {}
    for index, shard in enumerate(manifest["execution"]["slot_shards"]):
        if not shard["slot_keys"]:
            continue
        worker_root = root / "workers" / f"{index:02d}"
        marker = json.loads((worker_root / "worker_identity.json").read_text())
        _require(marker["owned_slots"] == shard["slot_keys"]
                 and marker["run_identity"] == manifest["identity_sha256"],
                 "freeze worker 所有权不符")
        active, snapshot = _active(worker_root / "library")
        stream = json.loads((worker_root / "buffer" / "stream.json").read_text())
        for attempt in stream["attempts"]:
            receipt_path = worker_root / "buffer" / attempt["path"]
            _require(
                receipt_path.resolve().is_relative_to(
                    (worker_root / "buffer").resolve())
                and hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                == attempt["sha256"],
                "freeze buffer attempt 收据 hash 改变",
            )
        _require(not any(value.get("deferred_reason")
                         for value in stream["slots"].values()),
                 "槽位存在未解决 deferred")
        _require(not any(value["new_since_attempt"] >= 12
                         for value in stream["slots"].values()),
                 "存在未完成的 12 条触发")
        for slot, spec in active.items():
            if fit_slot_key(*slot) not in shard["slot_keys"]:
                _require(spec == seed_by_slot[slot], "非所属槽位发生变更")
                continue
            if spec != seed_by_slot[slot]:
                promotions = [attempt for attempt in stream["attempts"]
                              if attempt["slot_key"] == fit_slot_key(*slot)
                              and attempt["outcome"] == "promoted"]
                _require(bool(promotions), "active 新版本没有晋升记录")
                attempt = json.loads(
                    (worker_root / "buffer" / promotions[-1]["path"]).read_text())
                campaign = worker_root / "campaigns" / attempt["result"]["campaign_id"]
                decision = json.loads((campaign / "decision.json").read_text())
                decision_body = {key: value for key, value in decision.items()
                                 if key != "identity_sha256"}
                _require(decision.get("identity_sha256") == content_sha256(decision_body)
                         and decision["passed"] and all(decision["conditions"].values())
                         and decision["candidate_hash"] == spec.content_sha256,
                         "freeze 新版本缺少有效严格 A/B 决定")
                post = json.loads((campaign / "post_load.json").read_text())
                _require(post["verified"] and post["content_sha256"] == spec.content_sha256
                         and post["version"] == f"{spec.skill_id}@{spec.version}",
                         "freeze 新版本缺少发布后加载证明")
                changed[slot] = spec
        receipts[str(index)] = {
            "snapshot": snapshot,
            "stream_sha256": content_sha256(stream),
            "owned_slots": shard["slot_keys"],
        }
    _require(sum(len(shard["slot_keys"]) for shard in
                 manifest["execution"]["slot_shards"]) == len(selected),
             "freeze 槽位 owner 重复")
    result_paths = list((root / "results").glob("*.json"))
    ids = {json.loads(path.read_text())["variant_id"] for path in result_paths}
    result_count = len(result_paths)
    if system_runner is not None:
        _require(system_runner.root == root
                 and system_runner.run_identity == manifest["identity_sha256"],
                 "freeze recovery runner identity changed")
        completed = system_runner.completed()
        _require(ids <= set(completed), "freeze ordinary results changed")
        ids = set(completed)
        result_count = len(completed)
    _require(ids == set(manifest["variant_ids"])
             and len(ids) == result_count
             and summary["completed_variants"] == len(ids),
             "freeze 分母/结果文件不一致")
    return build_fit_skill_bundle(
        output_dir=output_dir,
        bundle_id="fit-final-" + manifest["identity_sha256"][:16],
        candidates=changed, active_parents=seed_by_slot,
        source_run_identity={
            "policy": "last_successfully_published_per_slot",
            "accumulation_manifest_sha256": manifest["identity_sha256"],
            "workers": receipts, "selected_slots": sorted(selected),
            "no_additional_revision": True,
        },
        **({"framework_state_io": framework_state_io}
           if framework_state_io is not None else {}),
    )
