"""Offline exact-slot revision for the exploratory v11 fit-set run."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence
import uuid
import re

from skill3d.evaluation.fitset_v11 import (
    ALL_FIT_SLOTS,
    REACHABLE_FIT_SLOTS,
    SYSTEM_RESULT_SCHEMA,
    SYSTEM_RUN_SCHEMA,
    FitSetError,
    build_fit_skill_bundle,
    canonical_json_bytes,
    content_sha256,
    fit_slot_key,
    score_answer,
)
from skill3d.evolution.campaign_v11 import _unknown_tool_calls
from skill3d.evolution.leakage import leakage_scan_text
from skill3d.schemas import EvidenceProfile, SkillSpecV11
from skill3d.schemas.reconstruction import METRIC_TASK_TYPES
from skill3d.skills.delivery import (
    DEFAULT_METHOD_CONTEXT_MAX_CHARS,
    check_service_limit,
)
from skill3d.skills.registry import bump_semver
from skill3d.skills.registry import (
    active_snapshot_provenance,
    load_active_skills,
)
from skill3d.tools.contract import KNOWN_ARTIFACTS

FIT_REVISION_SCHEMA = "v11-fit-set-slot-revision/2.0"
FIT_FEEDBACK_SCHEMA = "v11-fit-set-slot-feedback/2.0"
FIT_REVISION_PROMPT_VERSION = "fit-set-slot-revision-v2"


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


def load_active_parent_slots(
    library_root: str | Path,
) -> tuple[dict[tuple[str, str], SkillSpecV11], dict]:
    root = Path(library_root).resolve()
    store = root / "snapshots"
    skills, warnings, snapshot_id = load_active_skills(store)
    active_id, manifest_hash = active_snapshot_provenance(store)
    if warnings or active_id != snapshot_id or not manifest_hash:
        raise FitSetError(
            f"active Skill snapshot is not trustworthy: {warnings}")
    parents = {
        (skill.question_type, skill.evidence_state): skill
        for skill in skills
    }
    if set(parents) != set(ALL_FIT_SLOTS):
        raise FitSetError("active Skill snapshot does not cover 24 physical slots")
    return parents, {
        "snapshot_id": snapshot_id,
        "manifest_sha256": manifest_hash,
    }


def _tool_docs(question_type: str, evidence_state: str) -> str:
    from skill3d.tools import REGISTRY

    geometry = evidence_state != "g0m0"
    scale = evidence_state == "g1m1"
    metric = scale and question_type in METRIC_TASK_TYPES
    return REGISTRY.docs(
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
    selected_slots = {
        (str(row["question_type"]), str(row["evidence_state"]))
        for row in variants
    }
    if not selected_slots or not selected_slots <= set(REACHABLE_FIT_SLOTS):
        raise FitSetError(
            "fit-set feedback must cover a non-empty set of reachable slots")
    by_slot: dict[tuple[str, str], list[dict]] = {
        slot: [] for slot in selected_slots
    }
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
        by_slot[
            (variant["question_type"], variant["evidence_state"])
        ].append(signature)
    slot_summaries = {}
    for question_type, evidence_state in sorted(selected_slots):
        rows = by_slot[(question_type, evidence_state)]
        if not rows:
            raise FitSetError(
                f"{fit_slot_key(question_type, evidence_state)}: "
                "fit-set feedback has no cases")
        counts = Counter(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in rows
        )
        signatures = [
            {**json.loads(key), "n": count}
            for key, count in sorted(
                counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        slot_summaries[fit_slot_key(question_type, evidence_state)] = {
            "question_type": question_type,
            "evidence_state": evidence_state,
            "n_cases": len(rows),
            "mean_score": sum(row["score"] for row in rows) / len(rows),
            "legal_answer_rate": (
                sum(row["legal_answer"] for row in rows) / len(rows)),
            "available_rate": (
                sum(row["available"] for row in rows) / len(rows)),
            "slot_outcome_signatures": signatures,
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
        "grouping_key": ["question_type", "evidence_state"],
        "slot_count": len(slot_summaries),
        "slots": slot_summaries,
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
            evidence_state=parent.evidence_state,
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
        from skill3d.tools import REGISTRY

        call_pattern = re.compile(r"\b(?:tools\.)?([A-Za-z_]\w*)\s*\(")
        current = set(call_pattern.findall(candidate.skill_md))
        inherited = set(call_pattern.findall(parent.skill_md))
        docs = _tool_docs(parent.question_type, parent.evidence_state)
        new_tools = (current - inherited) & set(REGISTRY.names())
        for tool in sorted(new_tools):
            if not re.search(rf"\b{re.escape(tool)}\s*\(", docs):
                problems.append(f"candidate adds unavailable exact-slot tool: {tool}")
        leaks = leakage_scan_text(
            candidate.skill_md,
            forbidden_sample_ids=forbidden_sample_ids,
        )
        if leaks:
            problems.append(f"candidate matched leakage patterns: {leaks}")
    return candidate, problems


class FitSetSlotReviser:
    """Create one independent final method for every selected reachable slot."""

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

    def _selected_slots(self) -> tuple[tuple[str, str], ...]:
        slots = tuple(sorted({
            (str(row["question_type"]), str(row["evidence_state"]))
            for row in self.variants
        }))
        if not slots or not set(slots) <= set(REACHABLE_FIT_SLOTS):
            raise FitSetError(
                "revision variants must cover reachable exact slots")
        return slots

    def _manifest(
        self,
        parents: Mapping[tuple[str, str], SkillSpecV11],
        parent_snapshot: Mapping,
    ) -> dict:
        selected_slots = self._selected_slots()
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
            "grouping_key": ["question_type", "evidence_state"],
            "selected_slot_count": len(selected_slots),
            "selected_slots": [fit_slot_key(*slot) for slot in selected_slots],
            "active_parent_snapshot": dict(parent_snapshot),
            "parents": {
                fit_slot_key(*slot): {
                    "skill_version": f"{parent.skill_id}@{parent.version}",
                    "content_sha256": parent.content_sha256,
                    "name": parent.name,
                }
                for slot, parent in sorted(parents.items())
            },
            "reference_answer_visible_to_online_accumulation": False,
            "raw_sample_content_visible_to_offline_reviser": False,
        }
        return {**body, "identity_sha256": content_sha256(body)}

    def _prompt(
        self,
        *,
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
            f"The method handles only exact slot {parent.slot_key!r}. "
            "Do not reuse text, identity, feedback, or runtime adaptation from "
            "another evidence state. Do not add branches for other states.\n"
            "Keep the frontmatter name unchanged. Keep the method generic. "
            "Never include a question, options, answer, QA/sample/scene ID, "
            "or inferred GT 3D fact. The aggregate feedback contains no such "
            "content. A wrong score alone does not prove a binding or referent "
            "failure. Improve decision rules, checks, tool use, and fallbacks "
            "only when supported by the aggregate process facts. Do not invent "
            "tools or alter framework, model, prompt, scoring, or permissions.\n\n"
            f"Parent complete SKILL.md:\n{parent.skill_md}\n"
            "Authorized Tool interfaces for this exact slot:\n"
            f"{_tool_docs(parent.question_type, parent.evidence_state)}\n"
            "Sanitized aggregate feedback for this exact slot:\n"
            f"{json.dumps(feedback, ensure_ascii=False, sort_keys=True)}\n"
            "Static feedback from prior formatting attempts:\n"
            f"{json.dumps(list(static_feedback), ensure_ascii=False)}\n"
        )

    def _request(
        self,
        *,
        slot: tuple[str, str],
        attempt: int,
        prompt: str,
    ) -> dict:
        slot_name = fit_slot_key(*slot)
        slot_root = self.root / "slots" / slot[0] / slot[1]
        exchange_path = slot_root / f"attempt_{attempt:02d}.json"
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
                raise FitSetError(
                    f"{slot_name}: revision request changed on resume")
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
            "slot_key": slot_name,
            "question_type": slot[0],
            "evidence_state": slot[1],
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

    def _revise_slot(
        self,
        *,
        slot: tuple[str, str],
        parent: SkillSpecV11,
        feedback: Mapping,
        forbidden_ids: set[str],
    ) -> SkillSpecV11:
        slot_name = fit_slot_key(*slot)
        slot_root = self.root / "slots" / slot[0] / slot[1]
        final_path = slot_root / "final.json"
        skill_path = slot_root / "SKILL.md"
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
                raise FitSetError(
                    f"{slot_name}: frozen candidate no longer validates")
            return candidate
        static_feedback: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            prompt = self._prompt(
                parent=parent,
                feedback=feedback,
                static_feedback=static_feedback,
            )
            exchange = self._request(
                slot=slot,
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
                "slot_key": slot_name,
                "question_type": slot[0],
                "evidence_state": slot[1],
                "attempt": attempt,
                "problems": problems,
                "passed": candidate is not None and not problems,
            }
            _write_create_once(
                slot_root / f"validation_{attempt:02d}.json",
                validation,
            )
            if candidate is not None and not problems:
                _write_text_create_once(skill_path, candidate.skill_md)
                _write_create_once(final_path, {
                    "schema_version": FIT_REVISION_SCHEMA,
                    "slot_key": slot_name,
                    "question_type": slot[0],
                    "evidence_state": slot[1],
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
            static_feedback = [
                # 下一次 prompt 不回传候选正文或可能包含标签的校验详情。
                "Static validation failed; preserve the parent's exact-slot "
                "identity, known authorized tools, complete source and no sample content."
            ]
        raise FitSetError(
            f"{slot_name}: no valid candidate after "
            f"{self.max_attempts} attempts")

    def run(self, *, resume: bool) -> dict:
        raise FitSetError(
            "一次性 reviser 不允许直接冻结未经 A/B 的候选；"
            "请用 run-system accumulation 的 12 条槽位流，再用 freeze")


__all__ = [
    "FIT_FEEDBACK_SCHEMA",
    "FIT_REVISION_SCHEMA",
    "FitSetSlotReviser",
    "build_method_feedback",
    "load_active_parent_slots",
]
