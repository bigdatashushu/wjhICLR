"""Fit-set system execution and direct-Qwen comparison for v11.

This exploratory path intentionally reuses the same questions for method
induction and final comparison. It does not modify the production active Skill
pointer. Eight canonical task methods are adapted to the frozen initial
evidence state only at runtime so the existing task/state retrieval contract
can execute them without creating additional method bodies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import numpy as np

from skill3d.adapters.episode_source import EpisodeItem
from skill3d.evaluation.accuracy import extract_option_letter
from skill3d.evaluation.direct_qwen_v11 import (
    DIRECT_RESPONSE_SCHEMA,
    DIRECT_RUN_SCHEMA,
    file_identity,
)
from skill3d.evaluation.mra import mra_single, parse_numeric_answer
from skill3d.infra.experiment_startup import current_code_identity
from skill3d.routing.task_classifier import TASK_TYPES, canonical_task
from skill3d.schemas import SkillSpecV11
from skill3d.schemas.answer import CANONICAL_UNIT_BY_QUESTION_TYPE
from skill3d.schemas.evidence import SKILL_EVIDENCE_STATES
from skill3d.schemas.skill import parse_skill_markdown
from skill3d.skills.delivery import DEFAULT_METHOD_CONTEXT_MAX_CHARS
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION

if TYPE_CHECKING:
    from skill3d.online.runner import OnlineRunConfig

PANEL_SCHEMA = "v11-same-question-system-vs-direct-qwen-proposal/1.0"
FIT_SKILL_BUNDLE_SCHEMA = "v11-fit-set-task-method-bundle/1.0"
SYSTEM_RUN_SCHEMA = "v11-fit-set-system-run/1.0"
SYSTEM_RESULT_SCHEMA = "v11-fit-set-system-result/1.0"
COMPARISON_SCHEMA = "v11-fit-set-system-vs-direct-qwen/1.0"
PANEL_NAMES = ("ready_579", "recommended_core_897", "maximum_1166")
CORE_G1M1_TASKS = frozenset({
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
})
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.@-]+$")


class FitSetError(RuntimeError):
    """A frozen fit-set input or result cannot be trusted."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def content_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_value(value):
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(child)
                for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_create_once(path: Path, value: object) -> None:
    data = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FitSetError(f"immutable artifact changed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FitSetError(message)


def _variant_key(variant_id: str) -> str:
    return hashlib.sha256(variant_id.encode("utf-8")).hexdigest()[:20]


def _source_code_identity() -> dict:
    root = Path(__file__).resolve().parents[3]
    receipt = current_code_identity(root)
    if receipt:
        return {"source": "startup_receipt", **receipt}
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise FitSetError(
            f"cannot establish fit-set source identity: {exc}") from exc
    dirty = bool(status.strip())
    if os.environ.get("SKILL3D_RUN_MODE") == "formal" and dirty:
        raise FitSetError("formal fit-set execution requires a clean source worktree")
    return {
        "source": "git_worktree",
        "source_root": str(root),
        "base_commit": commit,
        "git_dirty": dirty,
        "status_sha256": hashlib.sha256(
            status.encode("utf-8")).hexdigest(),
    }


def _lexical_absolute(path: str | Path) -> str:
    """Normalize a declared path without touching a possibly remote filesystem."""
    return os.path.abspath(os.path.expanduser(str(path)))


def _normalized_variant(raw: Mapping) -> dict:
    variant_id = str(raw.get("variant_id") or "")
    qa_id = str(raw.get("qa_id") or "")
    state = str(raw.get("evidence_state") or "")
    task = canonical_task(str(raw.get("question_type") or ""))
    _require(bool(_SAFE_ID.fullmatch(variant_id)),
             f"invalid system variant_id: {variant_id!r}")
    _require(bool(_SAFE_ID.fullmatch(qa_id)), f"invalid qa_id: {qa_id!r}")
    _require(variant_id == f"{qa_id}@{state}",
             f"{variant_id}: variant identity does not match qa/state")
    _require(state in SKILL_EVIDENCE_STATES,
             f"{variant_id}: invalid evidence state {state!r}")
    artifact = str(raw.get("artifact_ref") or "")
    _require(bool(artifact), f"{variant_id}: artifact_ref is missing")
    return {
        **dict(raw),
        "variant_id": variant_id,
        "qa_id": qa_id,
        "scene_id": str(raw.get("scene_id") or ""),
        "question_type": task,
        "evidence_state": state,
        "split": str(raw.get("split") or ""),
        "artifact_ref": artifact,
        "binding_key": str(raw.get("binding_key") or ""),
        "binding_manifest_sha256": str(
            raw.get("binding_manifest_sha256") or ""),
    }


def load_fitset_panel(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(value.get("schema_version") == PANEL_SCHEMA,
             "unsupported fit-set panel schema")
    raw_variants = value.get("system_variants")
    direct_rows = value.get("direct_qwen_rows")
    _require(isinstance(raw_variants, list) and raw_variants,
             "system_variants must be a non-empty list")
    _require(isinstance(direct_rows, list) and direct_rows,
             "direct_qwen_rows must be a non-empty list")
    variants = [_normalized_variant(row) for row in raw_variants]
    by_variant: dict[str, dict] = {}
    for row in variants:
        variant_id = row["variant_id"]
        _require(variant_id not in by_variant,
                 f"duplicate system variant: {variant_id}")
        by_variant[variant_id] = row
    direct_by_qa: dict[str, dict] = {}
    mapped: set[str] = set()
    for raw in direct_rows:
        qa_id = str(raw.get("qa_id") or "")
        _require(qa_id and qa_id not in direct_by_qa,
                 f"duplicate direct qa_id: {qa_id!r}")
        variant_ids = [str(item) for item in (
            raw.get("system_variant_ids") or [])]
        _require(bool(variant_ids), f"{qa_id}: direct row has no variants")
        _require(not (set(variant_ids) & mapped),
                 f"{qa_id}: a system variant maps to multiple direct rows")
        for variant_id in variant_ids:
            _require(variant_id in by_variant,
                     f"{qa_id}: unknown mapped variant {variant_id}")
            _require(by_variant[variant_id]["qa_id"] == qa_id,
                     f"{variant_id}: direct qa_id mismatch")
            _require(
                by_variant[variant_id]["scene_id"]
                == str(raw.get("scene_id") or "")
                and by_variant[variant_id]["question_type"]
                == canonical_task(str(raw.get("question_type") or ""))
                and by_variant[variant_id]["split"]
                == str(raw.get("split") or ""),
                f"{variant_id}: direct row identity mismatch",
            )
        mapped.update(variant_ids)
        direct_by_qa[qa_id] = {**dict(raw), "qa_id": qa_id,
                               "system_variant_ids": variant_ids}
    _require(mapped == set(by_variant),
             "direct rows do not exactly cover system variants")
    return {
        **value,
        "system_variants": variants,
        "direct_qwen_rows": list(direct_by_qa.values()),
        "panel_file_sha256": file_identity(path)["sha256"],
    }


def select_panel_variants(panel: Mapping, panel_name: str) -> list[dict]:
    _require(panel_name in PANEL_NAMES,
             f"unknown panel {panel_name!r}; expected one of {PANEL_NAMES}")
    variants = list(panel["system_variants"])
    if panel_name == "ready_579":
        selected = [
            row for row in variants
            if row.get("system_input_status") == "ready"
        ]
    elif panel_name == "recommended_core_897":
        selected = [
            row for row in variants
            if row.get("system_input_status") == "ready"
            or (
                row["evidence_state"] == "g1m1"
                and row["question_type"] in CORE_G1M1_TASKS
            )
        ]
    else:
        selected = variants
    expected = int((panel.get("panels") or {}).get(
        panel_name, {}).get("system_input_variants") or 0)
    _require(expected > 0 and len(selected) == expected,
             f"{panel_name}: selected {len(selected)} variants, expected {expected}")
    return selected


def direct_rows_for_variants(
    panel: Mapping,
    variants: Sequence[Mapping],
) -> list[dict]:
    wanted = {str(row["variant_id"]) for row in variants}
    selected = []
    covered: set[str] = set()
    for row in panel["direct_qwen_rows"]:
        mapped = wanted & set(row["system_variant_ids"])
        if mapped:
            selected.append({
                **row,
                "system_variant_ids": [
                    value for value in row["system_variant_ids"]
                    if value in wanted
                ],
            })
            covered.update(mapped)
    _require(covered == wanted, "selected direct rows do not cover the panel")
    return selected


def load_variant_bindings(
    path: str | Path,
    variants: Sequence[Mapping],
) -> dict[str, dict]:
    """Load a binding manifest keyed by variant_id, falling back to qa_id.

    qa_id fallback is accepted only when its frozen artifact and binding
    identities exactly match the selected variant. This supports the original
    579-row manifest while allowing g1m0/g1m1 variants of one QA to coexist.
    """
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = manifest.get("bindings")
    _require(isinstance(rows, dict), "prepared binding manifest lacks bindings")
    resolved: dict[str, dict] = {}
    for variant in variants:
        variant_id = str(variant["variant_id"])
        qa_id = str(variant["qa_id"])
        reference = rows.get(variant_id)
        if reference is None:
            reference = rows.get(qa_id)
        _require(isinstance(reference, dict),
                 f"{variant_id}: prepared binding is missing")
        _require(reference.get("key") == variant["binding_key"],
                 f"{variant_id}: binding key changed")
        _require(
            reference.get("manifest_sha256")
            == variant["binding_manifest_sha256"],
            f"{variant_id}: binding manifest hash changed",
        )
        _require(
            _lexical_absolute(reference.get("scene_artifact_ref") or "")
            == _lexical_absolute(variant["artifact_ref"]),
            f"{variant_id}: binding artifact path changed",
        )
        resolved[variant_id] = copy.deepcopy(reference)
    return resolved


@dataclass(frozen=True)
class FitTaskMethod:
    question_type: str
    skill_id: str
    version: str
    source_ref: str
    content_sha256: str
    skill_md: str

    def adapt(self, evidence_state: str) -> SkillSpecV11:
        _require(evidence_state in SKILL_EVIDENCE_STATES,
                 f"invalid runtime evidence state: {evidence_state}")
        return SkillSpecV11(
            skill_id=f"FIT-{self.skill_id}-{evidence_state}",
            version=self.version,
            question_type=self.question_type,
            evidence_state=evidence_state,
            skill_md=self.skill_md,
        )


@dataclass(frozen=True)
class FitSkillBundle:
    bundle_id: str
    manifest_sha256: str
    methods: Mapping[str, FitTaskMethod]
    manifest: Mapping


def load_fit_skill_bundle(path: str | Path) -> FitSkillBundle:
    manifest_path = Path(path).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(value.get("schema_version") == FIT_SKILL_BUNDLE_SCHEMA,
             "unsupported fit Skill bundle schema")
    claimed = str(value.get("manifest_sha256") or "")
    body = {key: child for key, child in value.items()
            if key != "manifest_sha256"}
    _require(claimed == content_sha256(body),
             "fit Skill bundle manifest hash changed")
    tasks = value.get("tasks")
    _require(isinstance(tasks, dict) and set(tasks) == set(TASK_TYPES),
             "fit Skill bundle must contain exactly eight canonical tasks")
    methods: dict[str, FitTaskMethod] = {}
    for question_type, row in tasks.items():
        _require(isinstance(row, dict),
                 f"{question_type}: task method entry must be a mapping")
        source_ref = str(row.get("source_ref") or "")
        source_path = (manifest_path.parent / source_ref).resolve()
        _require(source_path.is_relative_to(manifest_path.parent),
                 f"{question_type}: source_ref escapes the bundle")
        text = source_path.read_text(encoding="utf-8")
        _require(hashlib.sha256(text.encode("utf-8")).hexdigest()
                 == row.get("content_sha256"),
                 f"{question_type}: canonical SKILL.md changed")
        _require(len(text) <= DEFAULT_METHOD_CONTEXT_MAX_CHARS,
                 f"{question_type}: canonical SKILL.md exceeds service limit")
        front, _ = parse_skill_markdown(text)
        _require(front["name"] == row.get("name"),
                 f"{question_type}: stable method name changed")
        method = FitTaskMethod(
            question_type=question_type,
            skill_id=str(row.get("skill_id") or ""),
            version=str(row.get("version") or ""),
            source_ref=source_ref,
            content_sha256=str(row["content_sha256"]),
            skill_md=text,
        )
        for state in SKILL_EVIDENCE_STATES:
            method.adapt(state)
        methods[question_type] = method
    return FitSkillBundle(
        bundle_id=str(value.get("bundle_id") or ""),
        manifest_sha256=claimed,
        methods=methods,
        manifest=value,
    )


def build_fit_skill_bundle(
    *,
    output_dir: str | Path,
    bundle_id: str,
    methods: Mapping[str, SkillSpecV11],
    source_run_identity: Mapping,
) -> dict:
    """Freeze eight validated method bodies without changing active_snapshot."""
    root = Path(output_dir).resolve()
    _require(not root.exists(), f"fit Skill bundle already exists: {root}")
    _require(set(methods) == set(TASK_TYPES),
             "exactly one candidate method is required for each canonical task")
    root.mkdir(parents=True)
    task_rows = {}
    for task in TASK_TYPES:
        spec = methods[task]
        _require(spec.question_type == task,
                 f"{task}: candidate question_type changed")
        _require(len(spec.skill_md) <= DEFAULT_METHOD_CONTEXT_MAX_CHARS,
                 f"{task}: candidate exceeds service limit")
        relative = Path("skills") / task / "SKILL.md"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(spec.skill_md, encoding="utf-8")
        task_rows[task] = {
            "skill_id": spec.skill_id,
            "version": spec.version,
            "name": spec.name,
            "source_ref": relative.as_posix(),
            "content_sha256": spec.content_sha256,
        }
    body = {
        "schema_version": FIT_SKILL_BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        "method_count": len(task_rows),
        "runtime_adapter_contract": (
            "one canonical task SKILL.md; evidence_state is attached as an "
            "experiment-local read-only identity for existing retrieval"
        ),
        "source_run_identity": dict(source_run_identity),
        "tasks": task_rows,
    }
    manifest = {**body, "manifest_sha256": content_sha256(body)}
    _write_create_once(root / "manifest.json", manifest)
    load_fit_skill_bundle(root / "manifest.json")
    return manifest


def _system_result_row(
    outcome,
    *,
    variant: Mapping,
    options: Sequence[str],
    elapsed_s: float,
    skill_identity: Mapping,
) -> dict:
    service_errors = [
        error
        for row in outcome.rounds
        for error in (row.get("service_errors") or [])
    ]
    incomplete = outcome.final_state == "unavailable" or bool(service_errors)
    legal = outcome.answer is not None and outcome.predicted is not None
    if outcome.is_mca:
        legal = legal and outcome.predicted in list(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]
        )
    return {
        "schema_version": SYSTEM_RESULT_SCHEMA,
        "variant_id": variant["variant_id"],
        "qa_id": variant["qa_id"],
        "scene_id": variant["scene_id"],
        "split": variant["split"],
        "question_type": variant["question_type"],
        "evidence_state": variant["evidence_state"],
        "status": "incomplete" if incomplete else "completed",
        "final_state": outcome.final_state,
        "answer": outcome.answer,
        "predicted": outcome.predicted,
        "score": None if incomplete else outcome.score,
        "scoring_status": (
            "incomplete" if incomplete else outcome.scoring_status
        ),
        "legal_answer": bool(legal),
        "answer_parse_status": outcome.answer_parse_status,
        "submission_status": outcome.submission_status,
        "answer_basis": outcome.answer_basis,
        "answer_source": outcome.answer_source,
        "episode_status": outcome.episode_status,
        "failure_code": outcome.failure_code,
        "service_errors": service_errors,
        "static_check_errors": list(outcome.static_check_errors),
        "agent_rounds": int(outcome.agent_rounds),
        "yield_count": int(outcome.yield_count),
        "model_request_count": int(outcome.agent_rounds) + int(
            bool(outcome.eval_visual_fallback.get("request_count"))
        ),
        "tool_call_count": sum(
            len(row.get("results") or []) for row in outcome.rounds
        ),
        "elapsed_s": float(elapsed_s),
        "main_gate_passed": outcome.main_gate_passed,
        "quality_status": outcome.quality_status,
        "scene_route": outcome.scene_route,
        "question_tool_scope": outcome.question_tool_scope,
        "initial_evidence_states": dict(outcome.initial_evidence_states),
        "initial_skill_selection": dict(outcome.initial_skill_selection),
        "delivered_skill_versions": list(
            outcome.delivered_skill_versions
        ),
        "skill_identity": dict(skill_identity),
    }


class SystemVariantRunner:
    """Run one frozen system input per isolated, resumable variant directory."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        panel: Mapping,
        panel_name: str,
        variants: Sequence[Mapping],
        item_provider: Callable[[Mapping], EpisodeItem],
        input_receipt_provider: Callable[
            [Mapping, EpisodeItem], Mapping],
        input_contract_identity: Mapping,
        binding_refs: Mapping[str, Mapping],
        base_cfg: OnlineRunConfig,
        phase: str,
        skills_for_variant: Callable[[Mapping], tuple[Sequence[SkillSpecV11], dict]],
        client_factory: Callable[[str], object] | None = None,
    ) -> None:
        _require(phase in {"accumulation", "final_system"},
                 "system phase must be accumulation or final_system")
        self.root = Path(output_root).resolve()
        self.panel = panel
        self.panel_name = panel_name
        self.variants = [dict(row) for row in variants]
        self.item_provider = item_provider
        self.input_receipt_provider = input_receipt_provider
        self.input_contract_identity = copy.deepcopy(
            dict(input_contract_identity))
        self.binding_refs = {
            str(key): copy.deepcopy(dict(value))
            for key, value in binding_refs.items()
        }
        self.base_cfg = copy.deepcopy(base_cfg)
        self.phase = phase
        self.skills_for_variant = skills_for_variant
        self.client_factory = client_factory

    def _manifest(self) -> dict:
        identities = []
        for row in self.variants:
            skills, identity = self.skills_for_variant(row)
            identities.append({
                "variant_id": row["variant_id"],
                "skills": [
                    {
                        "skill_version": f"{skill.skill_id}@{skill.version}",
                        "question_type": skill.question_type,
                        "evidence_state": skill.evidence_state,
                        "content_sha256": skill.content_sha256,
                    }
                    for skill in skills
                ],
                "identity": identity,
            })
        body = {
            "schema_version": SYSTEM_RUN_SCHEMA,
            "phase": self.phase,
            "panel_name": self.panel_name,
            "panel_proposal_sha256": self.panel["proposal_sha256"],
            "panel_file_sha256": self.panel["panel_file_sha256"],
            "variant_ids": [row["variant_id"] for row in self.variants],
            "variant_rows_sha256": content_sha256(self.variants),
            "skill_identities_sha256": content_sha256(identities),
            "input_contract_identity": self.input_contract_identity,
            "seed": int(self.base_cfg.seed),
            "online_model": self.base_cfg.online_model,
            "online_model_revision": self.base_cfg.online_model_revision,
            "prompt_version": PROMPT_TEMPLATE_VERSION,
            "tool_docs_version": TOOL_DOCS_VERSION,
            "code_identity": _source_code_identity(),
            "reference_answer_retained_for_input_identity_and_m12": True,
            "reference_answer_visible_to_online_model": False,
        }
        return {**body, "identity_sha256": content_sha256(body)}

    def initialize(self, *, resume: bool) -> dict:
        manifest = self._manifest()
        path = self.root / "manifest.json"
        if self.root.exists():
            _require(resume, "system output exists; use resume for the same run")
            _require(path.is_file(), "existing system output lacks manifest")
            _require(
                json.loads(path.read_text(encoding="utf-8")) == manifest,
                "system run manifest changed during resume",
            )
        else:
            self.root.mkdir(parents=True)
            _write_create_once(path, manifest)
        return manifest

    def _result_path(self, variant_id: str) -> Path:
        return self.root / "results" / f"{_variant_key(variant_id)}.json"

    def completed(self) -> dict[str, dict]:
        rows = {}
        for path in sorted((self.root / "results").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            variant_id = str(row.get("variant_id") or "")
            _require(variant_id and variant_id not in rows,
                     f"duplicate or empty system result: {path}")
            _require(path.stem == _variant_key(variant_id),
                     f"system result path identity changed: {path}")
            rows[variant_id] = row
        expected = {row["variant_id"] for row in self.variants}
        _require(set(rows) <= expected,
                 "system results contain variants outside the frozen panel")
        return rows

    def _prepare_one(self, variant: Mapping) -> dict:
        from skill3d.evaluation.skill_ablation_v11 import (
            input_identity,
            initial_state_for_item,
        )
        from skill3d.legacy.readers import load_artifact_v5
        from skill3d.online import runner
        from skill3d.segmentation.prepared_binding_store import (
            load_prepared_binding_reference,
        )
        from skill3d.segmentation.question_binding import (
            binding_parameters,
            question_input,
            validate_prepared_binding,
        )

        variant_id = str(variant["variant_id"])
        item = self.item_provider(variant)
        _require(isinstance(item, EpisodeItem),
                 f"{variant_id}: episode provider returned an invalid item")
        input_receipt = dict(self.input_receipt_provider(variant, item))
        _require(item.input_error is None,
                 f"{variant_id}: input-error variants require explicit rows")
        _require(item.episode.scene_name == variant["scene_id"],
                 f"{variant_id}: episode scene changed")
        _require(canonical_task(item.episode.question_type)
                 == variant["question_type"],
                 f"{variant_id}: episode question type changed")
        _require(item.episode.split == variant["split"],
                 f"{variant_id}: episode split changed")
        frame_set = item.episode.frame_set
        _require(
            frame_set is not None
            and len(frame_set.frame_ids) == 32
            and len(frame_set.readable_frame_ids) == 32
            and len(item.pixels) == 32,
            f"{variant_id}: system input requires exactly 32 readable frames",
        )
        artifact_path = Path(variant["artifact_ref"]).resolve()
        _require(artifact_path.is_file(),
                 f"{variant_id}: frozen artifact is missing")
        state = initial_state_for_item(item, artifact_path)
        _require(state == variant["evidence_state"],
                 f"{variant_id}: runtime evidence state changed to {state}")
        identity = input_identity(item, artifact_path)
        reference = self.binding_refs.get(variant_id)
        _require(reference is not None,
                 f"{variant_id}: prepared binding reference is missing")
        _require(reference.get("scene_input_identity_sha256")
                 == identity["sha256"],
                 f"{variant_id}: prepared binding input identity changed")
        prepared = load_prepared_binding_reference(
            reference,
            scene_input_identity=identity,
        )
        frozen_parameters = dict(
            (reference.get("request_identity") or {}).get("parameters") or {})
        frozen_review = copy.deepcopy(
            frozen_parameters.get("experimental_semantic_review"))
        runtime_base_cfg = replace(
            copy.deepcopy(self.base_cfg),
            semantic_review_config=frozen_review,
        )
        runtime_parameters = binding_parameters(
            semantic_min_pixels=runtime_base_cfg.semantic_min_pixels,
            semantic_max_pixels=runtime_base_cfg.semantic_max_pixels,
            semantic_relation_batch_size=(
                runtime_base_cfg.semantic_relation_batch_size),
            semantic_review_config=runtime_base_cfg.semantic_review_config,
        )
        _require(runtime_parameters == frozen_parameters,
                 f"{variant_id}: sealed M5 parameters no longer replay")
        artifact = load_artifact_v5(artifact_path)
        validate_prepared_binding(
            prepared,
            frames=item.pixels,
            depth=runner._load_npy(artifact.depth_maps),
            c2w=runner._load_npy(artifact.c2w_list),
            intrinsics=runner._load_npy(artifact.intrinsics),
            point_conf=runner._load_npy(artifact.depth_conf),
            depth_conf=runner._load_npy(artifact.depth_conf),
            point_map=runner._load_npy(artifact.point_map),
            inputs=question_input(
                item.episode.question,
                canonical_task(item.episode.question_type),
                item.episode.options,
            ),
            scene_identity=str(
                artifact.reconstruction_input_identity
                or artifact.frame_set_hash
                or artifact.artifact_id
            ),
            seed=int(runtime_base_cfg.seed),
            model_weights_sha256="",
            parameters=runtime_parameters,
        )
        skills, skill_identity = self.skills_for_variant(variant)
        _require(bool(skills), f"{variant_id}: no runtime Skill was supplied")
        matching = [
            skill for skill in skills
            if skill.question_type == variant["question_type"]
            and skill.evidence_state == state
        ]
        _require(len(matching) == 1,
                 f"{variant_id}: runtime Skill match is not unique")
        variant_root = self.root / "variants" / _variant_key(variant_id)
        plan = {
            "schema_version": "v11-fit-set-system-plan/1.0",
            "variant": dict(variant),
            "variant_sha256": content_sha256(variant),
            "input_identity_sha256": identity["sha256"],
            "direct_input_receipt": input_receipt,
            "binding_key": reference["key"],
            "binding_manifest_sha256": reference["manifest_sha256"],
            "m5_parameters_sha256": content_sha256(frozen_parameters),
            "skill_identity": dict(skill_identity),
        }
        _write_create_once(variant_root / "plan.json", plan)
        return {
            "variant_id": variant_id,
            "item": item,
            "artifact_path": artifact_path,
            "state": state,
            "identity": identity,
            "reference": reference,
            "prepared": prepared,
            "runtime_base_cfg": runtime_base_cfg,
            "skills": skills,
            "skill_identity": skill_identity,
            "variant_root": variant_root,
            "plan": plan,
        }

    def _run_one(self, variant: Mapping) -> dict:
        from skill3d.online import runner

        prepared_run = self._prepare_one(variant)
        variant_id = prepared_run["variant_id"]
        item = prepared_run["item"]
        artifact_path = prepared_run["artifact_path"]
        state = prepared_run["state"]
        prepared = prepared_run["prepared"]
        runtime_base_cfg = prepared_run["runtime_base_cfg"]
        skills = prepared_run["skills"]
        skill_identity = prepared_run["skill_identity"]
        variant_root = prepared_run["variant_root"]
        trace_dir = variant_root / "trace"
        _require(not trace_dir.exists(),
                 f"{variant_id}: incomplete trace exists without a result")
        cfg = replace(
            copy.deepcopy(runtime_base_cfg),
            skills=[skill.model_copy(deep=True) for skill in skills],
            active_snapshot_ref=str(skill_identity["snapshot_ref"]),
            active_snapshot_manifest_sha256=str(
                skill_identity["manifest_sha256"]),
            trace_dir=str(trace_dir),
            work_dir=str(variant_root / "work"),
            recon_dir=str(variant_root / "recon"),
            reuse_artifact=str(artifact_path),
            evaluation_binding=None,
        )
        llm = (
            self.client_factory(variant_id)
            if self.client_factory is not None
            else None
        )
        started = time.perf_counter()
        outcomes, evaluation_run = runner.run_split(
            [item],
            cfg,
            llm=llm,
            prepared_bindings={item.episode.qa_id: prepared},
        )
        elapsed = time.perf_counter() - started
        _require(len(outcomes) == 1,
                 f"{variant_id}: system runner returned multiple outcomes")
        outcome = outcomes[0]
        _require(outcome.initial_skill_selection.get("evidence_state") == state,
                 f"{variant_id}: selected Skill state changed")
        _write_create_once(
            variant_root / "outcome.json",
            _json_value(asdict(outcome)),
        )
        _write_create_once(
            variant_root / "evaluation_run.json",
            evaluation_run.model_dump(mode="json"),
        )
        row = _system_result_row(
            outcome,
            variant=variant,
            options=list(item.episode.options or []),
            elapsed_s=elapsed,
            skill_identity=skill_identity,
        )
        _write_create_once(self._result_path(variant_id), row)
        return row

    def preflight(
        self,
        *,
        resume: bool,
        max_variants: int = 0,
        variant_ids: Sequence[str] = (),
    ) -> dict:
        manifest = self.initialize(resume=resume)
        wanted = set(variant_ids)
        known = {row["variant_id"] for row in self.variants}
        _require(not (wanted - known),
                 f"requested variants are outside the panel: {sorted(wanted - known)}")
        completed = {}
        for path in sorted((self.root / "preflight").glob("*.json")):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            variant_id = str(receipt.get("variant_id") or "")
            _require(
                variant_id in known
                and path.stem == _variant_key(variant_id)
                and variant_id not in completed,
                f"invalid system preflight receipt: {path}",
            )
            completed[variant_id] = receipt
        pending = [
            row for row in self.variants
            if row["variant_id"] not in completed
            and (not wanted or row["variant_id"] in wanted)
        ]
        if max_variants:
            _require(max_variants > 0, "max_variants must be non-negative")
            pending = pending[:max_variants]
        for variant in pending:
            values = self._prepare_one(variant)
            binding = values["prepared"]
            receipt = {
                "schema_version": "v11-fit-set-system-preflight/1.0",
                "variant_id": values["variant_id"],
                "input_identity_sha256": values["identity"]["sha256"],
                "binding_key": values["reference"]["key"],
                "binding_manifest_sha256": values["reference"][
                    "manifest_sha256"],
                "m5_parameters_sha256": values["plan"][
                    "m5_parameters_sha256"],
                "binding_object_count": len(binding.objects),
                "binding_materialized": bool(binding.materialized),
                "binding_status": str(
                    binding.stats.get("status") or "available"),
                "runtime_evidence_state": values["state"],
                "skill_identity": dict(values["skill_identity"]),
                "model_requests": 0,
            }
            _write_create_once(
                self.root / "preflight"
                / f"{_variant_key(values['variant_id'])}.json",
                receipt,
            )
            completed[values["variant_id"]] = receipt
        summary = {
            "schema_version": "v11-fit-set-system-preflight/1.0",
            "status": (
                "complete"
                if len(completed) == len(self.variants)
                else "partial"
            ),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "selected_variants": len(self.variants),
            "completed_preflights": len(completed),
            "pending_preflights": len(self.variants) - len(completed),
            "model_requests": 0,
            "updated_at": time.time(),
        }
        _write_json_atomic(self.root / "preflight_summary.json", summary)
        return summary

    def run(
        self,
        *,
        resume: bool,
        max_variants: int = 0,
        variant_ids: Sequence[str] = (),
    ) -> dict:
        manifest = self.initialize(resume=resume)
        completed = self.completed()
        wanted = set(variant_ids)
        known = {row["variant_id"] for row in self.variants}
        _require(not (wanted - known),
                 f"requested variants are outside the panel: {sorted(wanted - known)}")
        pending = [
            row for row in self.variants
            if row["variant_id"] not in completed
            and (not wanted or row["variant_id"] in wanted)
        ]
        if max_variants:
            _require(max_variants > 0, "max_variants must be non-negative")
            pending = pending[:max_variants]
        for row in pending:
            result = self._run_one(row)
            completed[result["variant_id"]] = result
            self._write_summary(manifest, completed)
        return self._write_summary(manifest, completed)

    def _write_summary(self, manifest: Mapping, completed: Mapping[str, dict]) -> dict:
        counts: dict[str, int] = {}
        for row in completed.values():
            status = str(row.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        summary = {
            "schema_version": SYSTEM_RUN_SCHEMA,
            "phase": self.phase,
            "status": (
                "complete"
                if len(completed) == len(self.variants)
                else "partial"
            ),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "selected_variants": len(self.variants),
            "completed_variants": len(completed),
            "pending_variants": len(self.variants) - len(completed),
            "status_counts": counts,
            "updated_at": time.time(),
        }
        _write_json_atomic(self.root / "summary.json", summary)
        return summary


def load_label_metadata(path: str | Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw_id = raw.get("qa_id")
        if raw_id is None or raw_id == "":
            raw_id = raw.get("id")
        qa_id = "" if raw_id is None else str(raw_id)
        _require(qa_id and qa_id not in rows,
                 f"duplicate or empty label qa_id: {qa_id!r}")
        rows[qa_id] = {
            "qa_id": qa_id,
            "scene_id": str(raw.get("scene_name") or ""),
            "question_type": canonical_task(
                str(raw.get("question_type") or "")),
            "options": [str(value) for value in (raw.get("options") or [])],
            "reference_answer": str(raw.get("ground_truth") or ""),
        }
    return rows


def score_answer(
    *,
    answer: object,
    task: str,
    options: Sequence[str],
    reference: str,
    available: bool,
) -> dict:
    unit = CANONICAL_UNIT_BY_QUESTION_TYPE[task]
    metric = "Accuracy" if unit == "option" else "MRA"
    if unit == "option":
        predicted = extract_option_letter(
            None if answer is None else str(answer))
        allowed = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)])
        ground_truth = extract_option_letter(reference) or reference.upper()
        _require(ground_truth in allowed,
                 f"{task}: invalid MCA reference {reference!r}")
        legal = predicted in allowed
        score = float(bool(available and legal and predicted == ground_truth))
    else:
        numeric = parse_numeric_answer(
            None if answer is None else str(answer))
        ground_truth = parse_numeric_answer(reference)
        _require(ground_truth is not None and ground_truth > 0,
                 f"{task}: invalid numeric reference {reference!r}")
        predicted = numeric
        legal = numeric is not None
        score = (
            mra_single(float(numeric), float(ground_truth))
            if available and legal
            else 0.0
        )
    return {
        "metric": metric,
        "predicted": predicted,
        "legal_answer": bool(legal),
        "available": bool(available),
        "score": float(score),
    }


def _load_system_results(root: Path, expected: set[str]) -> dict[str, dict]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    _require(summary.get("schema_version") == SYSTEM_RUN_SCHEMA,
             "unsupported system run schema")
    _require(summary.get("status") == "complete",
             "final system run is not complete")
    rows = {}
    for path in sorted((root / "results").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        variant_id = str(row.get("variant_id") or "")
        _require(row.get("schema_version") == SYSTEM_RESULT_SCHEMA,
                 f"{path}: unsupported system result schema")
        _require(variant_id and variant_id not in rows,
                 f"duplicate system result: {variant_id!r}")
        rows[variant_id] = row
    _require(set(rows) == expected,
             "system results do not exactly cover selected variants")
    return rows


def _load_direct_results(
    root: Path,
    expected_qa_ids: set[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == DIRECT_RUN_SCHEMA,
             "unsupported direct run manifest")
    _require(summary.get("schema_version") == DIRECT_RUN_SCHEMA,
             "unsupported direct run summary")
    _require(summary.get("status") == "complete",
             "direct-Qwen run is not complete")
    selected_qa_ids = set(manifest.get("selected_qa_ids") or [])
    _require(expected_qa_ids <= selected_qa_ids,
             "direct-Qwen manifest does not cover selected QA IDs")
    responses = {}
    preflights = {}
    for qa_id in sorted(expected_qa_ids):
        response_path = root / "responses" / f"{qa_id}.json"
        preflight_path = root / "preflight" / f"{qa_id}.json"
        _require(response_path.is_file() and preflight_path.is_file(),
                 f"{qa_id}: direct response or preflight is missing")
        response = json.loads(response_path.read_text(encoding="utf-8"))
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        _require(response.get("schema_version") == DIRECT_RESPONSE_SCHEMA,
                 f"{qa_id}: direct response schema changed")
        _require(response.get("request_sha256")
                 == preflight.get("request_sha256"),
                 f"{qa_id}: direct request hash changed")
        responses[qa_id] = response
        preflights[qa_id] = preflight
    return responses, preflights


def _mean(rows: Sequence[Mapping], field: str) -> float | None:
    values = [float(row[field]) for row in rows
              if row.get(field) is not None]
    return sum(values) / len(values) if values else None


def _cluster_interval(
    rows: Sequence[Mapping],
    *,
    delta_field: str,
    seed: int = 137,
    samples: int = 2000,
) -> list[float] | None:
    by_scene: dict[str, list[float]] = {}
    for row in rows:
        by_scene.setdefault(str(row["scene_id"]), []).append(
            float(row[delta_field]))
    scenes = sorted(by_scene)
    if len(scenes) < 2:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(samples):
        sampled = [scenes[rng.randrange(len(scenes))]
                   for _ in range(len(scenes))]
        flat = [value for scene in sampled for value in by_scene[scene]]
        values.append(sum(flat) / len(flat))
    values.sort()
    lower = values[max(0, math.floor(0.025 * (samples - 1)))]
    upper = values[min(samples - 1, math.ceil(0.975 * (samples - 1)))]
    return [lower, upper]


def _summarize_group(rows: Sequence[Mapping]) -> dict:
    return {
        "n_variants": len(rows),
        "n_unique_qa": len({row["qa_id"] for row in rows}),
        "n_scenes": len({row["scene_id"] for row in rows}),
        "system_score": _mean(rows, "system_score"),
        "direct_score": _mean(rows, "direct_score"),
        "system_minus_direct": _mean(rows, "delta"),
        "scene_cluster_bootstrap_95pct": _cluster_interval(
            rows, delta_field="delta"),
        "system_legal_answer_rate": _mean(rows, "system_legal_answer"),
        "direct_legal_answer_rate": _mean(rows, "direct_legal_answer"),
        "system_available_rate": _mean(rows, "system_available"),
        "direct_available_rate": _mean(rows, "direct_available"),
        "system_mean_model_requests": _mean(
            rows, "system_model_request_count"),
        "system_mean_tool_calls": _mean(rows, "system_tool_call_count"),
        "system_mean_elapsed_s": _mean(rows, "system_elapsed_s"),
        "direct_mean_elapsed_s": _mean(rows, "direct_elapsed_s"),
    }


def build_fitset_comparison(
    *,
    panel: Mapping,
    panel_name: str,
    system_root: str | Path,
    direct_root: str | Path,
    label_metadata: Mapping[str, Mapping],
    output_dir: str | Path,
) -> dict:
    variants = select_panel_variants(panel, panel_name)
    expected_variants = {row["variant_id"] for row in variants}
    direct_rows = direct_rows_for_variants(panel, variants)
    expected_qa_ids = {row["qa_id"] for row in direct_rows}
    system = _load_system_results(
        Path(system_root).resolve(), expected_variants)
    direct, preflights = _load_direct_results(
        Path(direct_root).resolve(), expected_qa_ids)
    pairs = []
    for variant in variants:
        variant_id = variant["variant_id"]
        qa_id = variant["qa_id"]
        label = label_metadata.get(qa_id)
        _require(label is not None, f"{variant_id}: label metadata is missing")
        _require(label["scene_id"] == variant["scene_id"]
                 and label["question_type"] == variant["question_type"],
                 f"{variant_id}: label metadata identity changed")
        system_row = system[variant_id]
        direct_row = direct[qa_id]
        preflight = preflights[qa_id]
        _require(variant_id in preflight.get("system_variant_ids", []),
                 f"{variant_id}: direct preflight mapping changed")
        system_available = (
            system_row.get("status") == "completed"
            and system_row.get("final_state") == "answer"
            and not system_row.get("service_errors")
        )
        direct_available = (
            direct_row.get("status") == "accepted"
            and bool(direct_row.get("accepted"))
        )
        direct_answer = (
            (direct_row.get("answer") or {}).get("value")
            if isinstance(direct_row.get("answer"), dict)
            else direct_row.get("answer")
        )
        system_score = score_answer(
            answer=system_row.get("answer"),
            task=variant["question_type"],
            options=label["options"],
            reference=label["reference_answer"],
            available=system_available,
        )
        direct_score = score_answer(
            answer=direct_answer,
            task=variant["question_type"],
            options=label["options"],
            reference=label["reference_answer"],
            available=direct_available,
        )
        _require(system_score["metric"] == direct_score["metric"],
                 f"{variant_id}: arm metrics differ")
        pair = {
            "variant_id": variant_id,
            "qa_id": qa_id,
            "scene_id": variant["scene_id"],
            "question_type": variant["question_type"],
            "evidence_state": variant["evidence_state"],
            "split": variant["split"],
            "metric": system_score["metric"],
            "system_score": system_score["score"],
            "direct_score": direct_score["score"],
            "delta": system_score["score"] - direct_score["score"],
            "system_predicted": system_score["predicted"],
            "direct_predicted": direct_score["predicted"],
            "system_legal_answer": float(system_score["legal_answer"]),
            "direct_legal_answer": float(direct_score["legal_answer"]),
            "system_available": float(system_score["available"]),
            "direct_available": float(direct_score["available"]),
            "system_status": system_row.get("status"),
            "direct_status": direct_row.get("status"),
            "system_model_request_count": system_row.get(
                "model_request_count"),
            "system_tool_call_count": system_row.get("tool_call_count"),
            "system_elapsed_s": system_row.get("elapsed_s"),
            "direct_elapsed_s": direct_row.get("elapsed_s"),
            "direct_request_sha256": direct_row.get("request_sha256"),
        }
        pairs.append(pair)
    groups = {"all": _summarize_group(pairs)}
    for field in ("question_type", "evidence_state", "split"):
        values = sorted({str(row[field]) for row in pairs})
        for value in values:
            groups[f"{field}:{value}"] = _summarize_group([
                row for row in pairs if str(row[field]) == value
            ])
    body = {
        "schema_version": COMPARISON_SCHEMA,
        "status": "complete",
        "panel_name": panel_name,
        "panel_proposal_sha256": panel["proposal_sha256"],
        "n_variants": len(pairs),
        "n_unique_qa": len(expected_qa_ids),
        "interpretation": (
            "fit-set comparison of the final complete Skill-3D system versus "
            "qwen3.8-flash direct 32-frame visual answering; not a "
            "generalization, final-test, or isolated Skill-effect estimate"
        ),
        "denominator_contract": (
            "Every selected system variant remains in the denominator. "
            "Unavailable, service-error, missing, and illegal answers score zero."
        ),
        "groups": groups,
        "inputs": {
            "system_root": str(Path(system_root).resolve()),
            "direct_root": str(Path(direct_root).resolve()),
        },
    }
    report = {**body, "identity_sha256": content_sha256(body)}
    root = Path(output_dir).resolve()
    _require(not root.exists(), f"comparison output already exists: {root}")
    root.mkdir(parents=True)
    _write_create_once(root / "report.json", report)
    with (root / "paired_results.jsonl").open("wb") as stream:
        for row in pairs:
            stream.write(canonical_json_bytes(row) + b"\n")
    return report


__all__ = [
    "COMPARISON_SCHEMA",
    "FIT_SKILL_BUNDLE_SCHEMA",
    "FitSetError",
    "FitSkillBundle",
    "FitTaskMethod",
    "SystemVariantRunner",
    "build_fit_skill_bundle",
    "build_fitset_comparison",
    "direct_rows_for_variants",
    "load_fit_skill_bundle",
    "load_fitset_panel",
    "load_label_metadata",
    "load_variant_bindings",
    "score_answer",
    "select_panel_variants",
]
