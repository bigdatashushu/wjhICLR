"""Offline eight-method revision for the exploratory v11 fit-set run."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence
import uuid

from skill3d.evaluation.fitset_v11 import (
    SYSTEM_RESULT_SCHEMA,
    SYSTEM_RUN_SCHEMA,
    FitSetError,
    build_fit_skill_bundle,
    canonical_json_bytes,
    content_sha256,
    score_answer,
)
from skill3d.evolution.campaign_v11 import _unknown_tool_calls
from skill3d.evolution.leakage import leakage_scan_text
from skill3d.routing.task_classifier import TASK_TYPES
from skill3d.schemas import EvidenceProfile, SkillSpecV11
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES
from skill3d.skills.delivery import (
    DEFAULT_METHOD_CONTEXT_MAX_CHARS,
    check_service_limit,
)
from skill3d.skills.registry import bump_semver
from skill3d.tools.contract import KNOWN_ARTIFACTS

FIT_REVISION_SCHEMA = "v11-fit-set-eight-method-revision/1.0"
FIT_FEEDBACK_SCHEMA = "v11-fit-set-method-feedback/1.0"
FIT_REVISION_PROMPT_VERSION = "fit-set-eight-method-revision-v1"

DEFAULT_PARENT_METHODS = {
    "object_counting": (
        "S01", "1.3.0",
        "versions/S01/1.3.0/count-scene-objects/SKILL.md",
    ),
    "object_abs_distance": (
        "S02", "1.2.0",
        "versions/S02/1.2.0/estimate-object-distance/SKILL.md",
    ),
    "object_rel_distance": (
        "S03", "1.2.0",
        "versions/S03/1.2.0/rank-object-distances/SKILL.md",
    ),
    "object_size_estimation": (
        "S04", "1.2.0",
        "versions/S04/1.2.0/estimate-object-size/SKILL.md",
    ),
    "room_size_estimation": (
        "S05", "1.2.0",
        "versions/S05/1.2.0/estimate-room-area/SKILL.md",
    ),
    "object_rel_direction": (
        "S06", "1.2.0",
        "versions/S06/1.2.0/judge-relative-direction/SKILL.md",
    ),
    "obj_appearance_order": (
        "S07", "1.2.0",
        "versions/S07/1.2.0/order-first-appearances/SKILL.md",
    ),
    "route_planning": (
        "S08", "1.3.0",
        "versions/S08/1.3.0/infer-route-turns/SKILL.md",
    ),
}


def _write_create_once(path: Path, value: object) -> None:
    data = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FitSetError(f"immutable revision artifact changed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_create_once(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise FitSetError(f"immutable candidate changed: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _parse_response(text: str) -> dict:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FitSetError(
            f"offline reviser returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise FitSetError("offline reviser response must be one JSON object")
    if not isinstance(value.get("full_skill_md"), str):
        raise FitSetError("offline reviser response lacks full_skill_md")
    if not isinstance(value.get("modification_reason"), str):
        raise FitSetError("offline reviser response lacks modification_reason")
    return value


def load_unified_parent_methods(
    library_root: str | Path,
) -> dict[str, SkillSpecV11]:
    root = Path(library_root).resolve()
    methods = {}
    for task, (skill_id, version, relative) in DEFAULT_PARENT_METHODS.items():
        text = (root / relative).read_text(encoding="utf-8")
        methods[task] = SkillSpecV11(
            skill_id=skill_id,
            version=version,
            question_type=task,
            evidence_state="g0m0",
            skill_md=text,
        )
    return methods


def _tool_docs(question_type: str) -> dict[str, str]:
    from skill3d.tools import REGISTRY

    rows = {}
    for state in ("g0m0", "g1m0", "g1m1"):
        geometry = state != "g0m0"
        scale = state == "g1m1"
        metric = scale and question_type in METRIC_TASK_TYPES
        rows[state] = REGISTRY.docs(
            scope=(
                "fallback_2d_only"
                if not geometry
                else "metric_enabled" if metric else "full_3d"
            ),
            available=KNOWN_ARTIFACTS,
            evidence_profile=EvidenceProfile(
                geometry_3d="available" if geometry else "unavailable",
                metric_scale="available" if scale else "unavailable",
            ),
            gate_passed=metric,
            question_type=question_type,
            local_facts={"world_frame": True},
        )
    return rows


def _bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    return "4+"


def _load_accumulation_results(
    root: Path,
    expected_variant_ids: set[str],
) -> tuple[dict, dict[str, dict]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SYSTEM_RUN_SCHEMA
        or manifest.get("phase") != "accumulation"
        or summary.get("schema_version") != SYSTEM_RUN_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise FitSetError("accumulation run is not complete")
    rows = {}
    for path in sorted((root / "results").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        variant_id = str(row.get("variant_id") or "")
        if (
            row.get("schema_version") != SYSTEM_RESULT_SCHEMA
            or not variant_id
            or variant_id in rows
        ):
            raise FitSetError(f"invalid accumulation result: {path}")
        rows[variant_id] = row
    if set(rows) != expected_variant_ids:
        raise FitSetError(
            "accumulation results do not exactly cover the frozen panel")
    return manifest, rows


def build_method_feedback(
    *,
    variants: Sequence[Mapping],
    accumulation_root: str | Path,
    label_metadata: Mapping[str, Mapping],
) -> dict:
    """Aggregate label-aware outcomes without retaining sample content or IDs."""
    expected = {str(row["variant_id"]) for row in variants}
    manifest, results = _load_accumulation_results(
        Path(accumulation_root).resolve(), expected)
    by_task: dict[str, list[dict]] = {task: [] for task in TASK_TYPES}
    for variant in variants:
        variant_id = str(variant["variant_id"])
        qa_id = str(variant["qa_id"])
        row = results[variant_id]
        label = label_metadata.get(qa_id)
        if label is None:
            raise FitSetError(f"{variant_id}: offline label is missing")
        if (
            label["scene_id"] != variant["scene_id"]
            or label["question_type"] != variant["question_type"]
            or row["question_type"] != variant["question_type"]
            or row["evidence_state"] != variant["evidence_state"]
        ):
            raise FitSetError(f"{variant_id}: offline identity changed")
        available = (
            row.get("status") == "completed"
            and row.get("final_state") == "answer"
            and not row.get("service_errors")
        )
        scored = score_answer(
            answer=row.get("answer"),
            task=variant["question_type"],
            options=label["options"],
            reference=label["reference_answer"],
            available=available,
        )
        signature = {
            "evidence_state": variant["evidence_state"],
            "source_split_role": variant["split"],
            "metric": scored["metric"],
            "score": round(float(scored["score"]), 3),
            "legal_answer": bool(scored["legal_answer"]),
            "available": bool(scored["available"]),
            "final_state": str(row.get("final_state") or ""),
            "failure_code": str(row.get("failure_code") or "none"),
            "answer_basis": str(row.get("answer_basis") or "unknown"),
            "answer_source": str(row.get("answer_source") or "unknown"),
            "main_gate_passed": row.get("main_gate_passed"),
            "scene_route": str(row.get("scene_route") or "unknown"),
            "question_tool_scope": str(
                row.get("question_tool_scope") or "unknown"),
            "model_round_bucket": _bucket(
                int(row.get("model_request_count") or 0)),
            "tool_call_bucket": _bucket(
                int(row.get("tool_call_count") or 0)),
        }
        by_task[variant["question_type"]].append(signature)
    task_summaries = {}
    for task in TASK_TYPES:
        rows = by_task[task]
        if not rows:
            raise FitSetError(f"{task}: fit-set feedback has no cases")
        states = {}
        for state in ("g0m0", "g1m0", "g1m1"):
            selected = [row for row in rows
                        if row["evidence_state"] == state]
            states[state] = {
                "n": len(selected),
                "mean_score": (
                    sum(row["score"] for row in selected) / len(selected)
                    if selected else None
                ),
                "legal_answer_rate": (
                    sum(row["legal_answer"] for row in selected) / len(selected)
                    if selected else None
                ),
                "available_rate": (
                    sum(row["available"] for row in selected) / len(selected)
                    if selected else None
                ),
            }
        counts = Counter(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in rows
        )
        signatures = [
            {**json.loads(key), "n": count}
            for key, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        task_summaries[task] = {
            "n_cases": len(rows),
            "mean_score": sum(row["score"] for row in rows) / len(rows),
            "legal_answer_rate": (
                sum(row["legal_answer"] for row in rows) / len(rows)),
            "available_rate": (
                sum(row["available"] for row in rows) / len(rows)),
            "by_initial_evidence_state": states,
            "method_outcome_signatures": signatures,
        }
    body = {
        "schema_version": FIT_FEEDBACK_SCHEMA,
        "privacy_contract": {
            "raw_questions_included": False,
            "options_included": False,
            "reference_answers_included": False,
            "predicted_answers_included": False,
            "qa_ids_included": False,
            "scene_ids_included": False,
            "gt_3d_included": False,
            "labels_used_only_to_compute_scores": True,
        },
        "source_accumulation_manifest_sha256": content_sha256(manifest),
        "tasks": task_summaries,
    }
    return {**body, "identity_sha256": content_sha256(body)}


def _static_validate(
    *,
    parent: SkillSpecV11,
    candidate_text: str,
    forbidden_sample_ids: set[str],
    method_context_max_chars: int,
) -> tuple[SkillSpecV11 | None, list[str]]:
    problems = []
    version = bump_semver(parent.version, "MINOR")
    candidate = None
    try:
        candidate = SkillSpecV11(
            skill_id=parent.skill_id,
            version=version,
            question_type=parent.question_type,
            evidence_state="g0m0",
            skill_md=candidate_text,
        )
    except Exception as exc:
        problems.append(
            f"invalid complete SKILL.md: {type(exc).__name__}: {exc}")
    if candidate is not None:
        if candidate.skill_md == parent.skill_md:
            problems.append("candidate is identical to its parent")
        if candidate.name != parent.name:
            problems.append("candidate changed the stable frontmatter name")
        problems.extend(check_service_limit(
            candidate, max_chars=method_context_max_chars))
        unknown = _unknown_tool_calls(candidate.skill_md)
        if unknown:
            problems.append(f"candidate references unknown tools: {unknown}")
        leaks = leakage_scan_text(
            candidate.skill_md,
            forbidden_sample_ids=forbidden_sample_ids,
        )
        if leaks:
            problems.append(f"candidate matched leakage patterns: {leaks}")
    return candidate, problems


class FitSetEightMethodReviser:
    """Create one final cross-state method per task from sanitized aggregates."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        panel: Mapping,
        panel_name: str,
        variants: Sequence[Mapping],
        accumulation_root: str | Path,
        label_metadata: Mapping[str, Mapping],
        library_root: str | Path,
        client,
        max_attempts: int = 3,
        max_tokens: int = 8192,
        method_context_max_chars: int = DEFAULT_METHOD_CONTEXT_MAX_CHARS,
    ) -> None:
        self.root = Path(output_root).resolve()
        self.panel = panel
        self.panel_name = panel_name
        self.variants = [dict(row) for row in variants]
        self.accumulation_root = Path(accumulation_root).resolve()
        self.label_metadata = label_metadata
        self.library_root = Path(library_root).resolve()
        self.client = client
        self.max_attempts = int(max_attempts)
        self.max_tokens = int(max_tokens)
        self.method_context_max_chars = int(method_context_max_chars)
        if self.max_attempts <= 0 or self.max_tokens <= 0:
            raise FitSetError("revision attempts and token budget must be positive")

    def _manifest(self, parents: Mapping[str, SkillSpecV11]) -> dict:
        body = {
            "schema_version": FIT_REVISION_SCHEMA,
            "panel_name": self.panel_name,
            "panel_proposal_sha256": self.panel["proposal_sha256"],
            "variant_rows_sha256": content_sha256(self.variants),
            "accumulation_root": str(self.accumulation_root),
            "accumulation_manifest": hashlib.sha256(
                (self.accumulation_root / "manifest.json").read_bytes()
            ).hexdigest(),
            "label_metadata_sha256": content_sha256(self.label_metadata),
            "prompt_version": FIT_REVISION_PROMPT_VERSION,
            "max_attempts": self.max_attempts,
            "max_tokens": self.max_tokens,
            "method_context_max_chars": self.method_context_max_chars,
            "parents": {
                task: {
                    "skill_version": f"{parent.skill_id}@{parent.version}",
                    "content_sha256": parent.content_sha256,
                    "name": parent.name,
                }
                for task, parent in parents.items()
            },
            "reference_answer_visible_to_online_accumulation": False,
            "raw_sample_content_visible_to_offline_reviser": False,
        }
        return {**body, "identity_sha256": content_sha256(body)}

    def _prompt(
        self,
        *,
        task: str,
        parent: SkillSpecV11,
        feedback: Mapping,
        static_feedback: Sequence[str],
    ) -> str:
        return (
            "Revise one reusable method for a visual-spatial coding agent.\n"
            f"Protocol: {FIT_REVISION_PROMPT_VERSION}.\n"
            "Return exactly one JSON object with string fields full_skill_md "
            "and modification_reason. full_skill_md must contain the complete "
            "canonical SKILL.md with YAML frontmatter and full Markdown body. "
            "Use English for all natural-language text and code comments.\n"
            f"The method handles question type {task!r}. It must contain "
            "explicit branches for initial public evidence states g0m0 "
            "(no geometry/no scale), g1m0 (geometry/no scale), and g1m1 "
            "(geometry/scale). There is one canonical method body for the task; "
            "do not create separate methods or identities for these states.\n"
            "Keep the frontmatter name unchanged. Keep the method generic. "
            "Never include a question, options, answer, QA/sample/scene ID, "
            "or inferred GT 3D fact. The aggregate feedback contains no such "
            "content. A wrong score alone does not prove a binding or referent "
            "failure. Improve decision rules, checks, tool use, and fallbacks "
            "only when supported by the aggregate process facts. Do not invent "
            "tools or alter framework, model, prompt, scoring, or permissions.\n\n"
            f"Parent complete SKILL.md:\n{parent.skill_md}\n"
            "Authorized Tool interfaces by initial state:\n"
            f"{json.dumps(_tool_docs(task), ensure_ascii=False, sort_keys=True)}\n"
            "Sanitized aggregate method feedback:\n"
            f"{json.dumps(feedback, ensure_ascii=False, sort_keys=True)}\n"
            "Static feedback from prior formatting attempts:\n"
            f"{json.dumps(list(static_feedback), ensure_ascii=False)}\n"
        )

    def _request(
        self,
        *,
        task: str,
        attempt: int,
        prompt: str,
    ) -> dict:
        task_root = self.root / "tasks" / task
        exchange_path = task_root / f"attempt_{attempt:02d}.json"
        request = {
            "prompt_version": FIT_REVISION_PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an offline Skill method reviser. "
                        "Output strict JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "seed": 137 + attempt,
            "temperature": 0.0,
            "enable_thinking": False,
        }
        if exchange_path.is_file():
            exchange = json.loads(exchange_path.read_text(encoding="utf-8"))
            if exchange.get("request") != request:
                raise FitSetError(f"{task}: revision request changed on resume")
            return exchange
        text = self.client.chat(
            request["messages"],
            max_tokens=self.max_tokens,
            seed=request["seed"],
            temperature=0.0,
            enable_thinking=False,
        )
        exchange = {
            "schema_version": FIT_REVISION_SCHEMA,
            "task": task,
            "attempt": attempt,
            "request": request,
            "response_text": text,
            "response_sha256": hashlib.sha256(
                text.encode("utf-8")).hexdigest(),
            "usage": dict(getattr(self.client, "last_usage", None) or {}),
            "generation": dict(
                getattr(self.client, "last_generation", None) or {}),
            "response_identity": dict(
                getattr(self.client, "last_response_identity", None) or {}),
        }
        _write_create_once(exchange_path, exchange)
        return exchange

    def _revise_task(
        self,
        *,
        task: str,
        parent: SkillSpecV11,
        feedback: Mapping,
        forbidden_ids: set[str],
    ) -> SkillSpecV11:
        final_path = self.root / "tasks" / task / "final.json"
        skill_path = self.root / "tasks" / task / "SKILL.md"
        if final_path.is_file() and skill_path.is_file():
            receipt = json.loads(final_path.read_text(encoding="utf-8"))
            text = skill_path.read_text(encoding="utf-8")
            candidate, problems = _static_validate(
                parent=parent,
                candidate_text=text,
                forbidden_sample_ids=forbidden_ids,
                method_context_max_chars=self.method_context_max_chars,
            )
            if (
                problems
                or candidate is None
                or candidate.content_sha256 != receipt.get("content_sha256")
            ):
                raise FitSetError(f"{task}: frozen candidate no longer validates")
            return candidate
        static_feedback: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            prompt = self._prompt(
                task=task,
                parent=parent,
                feedback=feedback,
                static_feedback=static_feedback,
            )
            exchange = self._request(
                task=task,
                attempt=attempt,
                prompt=prompt,
            )
            try:
                value = _parse_response(exchange["response_text"])
                candidate, problems = _static_validate(
                    parent=parent,
                    candidate_text=value["full_skill_md"],
                    forbidden_sample_ids=forbidden_ids,
                    method_context_max_chars=self.method_context_max_chars,
                )
            except FitSetError as exc:
                candidate, problems = None, [str(exc)]
            validation = {
                "schema_version": FIT_REVISION_SCHEMA,
                "task": task,
                "attempt": attempt,
                "problems": problems,
                "passed": candidate is not None and not problems,
            }
            _write_create_once(
                self.root / "tasks" / task
                / f"validation_{attempt:02d}.json",
                validation,
            )
            if candidate is not None and not problems:
                _write_text_create_once(skill_path, candidate.skill_md)
                _write_create_once(final_path, {
                    "schema_version": FIT_REVISION_SCHEMA,
                    "task": task,
                    "attempt": attempt,
                    "parent_skill_version": (
                        f"{parent.skill_id}@{parent.version}"),
                    "candidate_skill_version": (
                        f"{candidate.skill_id}@{candidate.version}"),
                    "content_sha256": candidate.content_sha256,
                    "modification_reason": value["modification_reason"],
                    "exchange_sha256": content_sha256(exchange),
                })
                return candidate
            static_feedback = problems
        raise FitSetError(
            f"{task}: no valid candidate after {self.max_attempts} attempts")

    def run(self, *, resume: bool) -> dict:
        parents = load_unified_parent_methods(self.library_root)
        manifest = self._manifest(parents)
        manifest_path = self.root / "manifest.json"
        if self.root.exists():
            if not resume:
                raise FitSetError(
                    "revision output exists; use resume for the same run")
            if (
                not manifest_path.is_file()
                or json.loads(manifest_path.read_text(encoding="utf-8"))
                != manifest
            ):
                raise FitSetError("revision manifest changed during resume")
        else:
            self.root.mkdir(parents=True)
            _write_create_once(manifest_path, manifest)
        feedback = build_method_feedback(
            variants=self.variants,
            accumulation_root=self.accumulation_root,
            label_metadata=self.label_metadata,
        )
        _write_create_once(self.root / "method_feedback.json", feedback)
        forbidden_ids = {
            str(value)
            for row in self.variants
            for value in (
                row.get("variant_id"),
                row.get("qa_id"),
                row.get("scene_id"),
            )
            if value
        }
        candidates = {
            task: self._revise_task(
                task=task,
                parent=parents[task],
                feedback=feedback["tasks"][task],
                forbidden_ids=forbidden_ids,
            )
            for task in TASK_TYPES
        }
        bundle_root = self.root / "frozen_bundle"
        bundle_id = (
            f"fit-{self.panel_name}-"
            f"{manifest['identity_sha256'][:16]}"
        )
        if bundle_root.exists():
            from skill3d.evaluation.fitset_v11 import load_fit_skill_bundle

            bundle_manifest = load_fit_skill_bundle(
                bundle_root / "manifest.json").manifest
        else:
            bundle_manifest = build_fit_skill_bundle(
                output_dir=bundle_root,
                bundle_id=bundle_id,
                methods=candidates,
                source_run_identity={
                    "revision_manifest_sha256": manifest["identity_sha256"],
                    "feedback_sha256": feedback["identity_sha256"],
                },
            )
        summary = {
            "schema_version": FIT_REVISION_SCHEMA,
            "status": "complete",
            "panel_name": self.panel_name,
            "candidate_methods": len(candidates),
            "bundle_id": bundle_manifest["bundle_id"],
            "bundle_manifest_sha256": bundle_manifest["manifest_sha256"],
            "bundle_manifest": str(bundle_root / "manifest.json"),
        }
        _write_create_once(self.root / "summary.json", summary)
        return summary


__all__ = [
    "DEFAULT_PARENT_METHODS",
    "FIT_FEEDBACK_SCHEMA",
    "FIT_REVISION_SCHEMA",
    "FitSetEightMethodReviser",
    "build_method_feedback",
    "load_unified_parent_methods",
]
