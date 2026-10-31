"""Lossless round records and offline-only fit-set admission to v11 cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from typing import Any

from skill3d.evaluation.mra import parse_numeric_answer
from skill3d.routing.task_classifier import canonical_task
from skill3d.schemas import DataAccessRecord, SkillSpecV11, V11ExperienceCase, utcnow_iso
from skill3d.schemas.data_access import learning_authorization_for, validate_learning_source
from skill3d.schemas.evidence import EvidenceProfile, skill_evidence_state


class V11ExperienceBuildError(ValueError):
    """Recorded evidence cannot be bound to trustworthy parent learning."""


def _mapping(value: Any) -> dict:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise V11ExperienceBuildError("Expected a recorded mapping or trace object")


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V11ExperienceBuildError(message)


def _round_rows(trace: Mapping) -> list[dict]:
    rows = trace.get("rounds") or []
    _require(isinstance(rows, list) and all(isinstance(r, Mapping) for r in rows),
             "rounds must be recorded mappings")
    return [dict(row) for row in rows]


def collect_case_tool_observations(
    trace: Any, program_trace: Any = None,
) -> list[dict]:
    """Keep all raw results, including differing revisions of the same ID."""
    recorded = _mapping(trace)
    candidates = [
        _mapping(result)
        for row in _round_rows(recorded)
        for result in (row.get("results") or [])
    ]
    if program_trace is not None:
        candidates.extend(_mapping(r) for r in _mapping(program_trace).get("results", []))
    observations: list[dict] = []
    seen: set[str] = set()
    for row in candidates:
        identity = _sha256(row)
        if identity not in seen:
            seen.add(identity)
            observations.append(deepcopy(row))
    return observations


def extract_recorded_selected_object_ids(trace: Any) -> list[str]:
    """Only explicit selections, not resolved inputs or geometry dependencies."""
    recorded = _mapping(trace)
    return sorted({
        str(identity) for row in [recorded, *_round_rows(recorded)]
        for identity in row.get("selected_object_ids") or []
    })


# Only image provenance crosses this boundary, never pixels or GT annotations.
_IMAGE_REF_FIELDS = frozenset({
    "image_id", "kind", "produced_by", "result_id", "produced_round",
    "source_frame_id", "source_frame_index", "box_xyxy", "source_hw", "sent_hw",
    "scale", "transform_version", "content_sha256", "delivered_rounds",
    "observed_rounds", "sent_in_requests", "response_completed_in_requests",
    "present_in_current_request", "unobserved_reason", "n_pixels_sent",
    "token_estimate", "token_estimate_source", "frame_set_hash", "frame_id",
    "path", "uri", "url", "source_ref", "trace_ref", "round_index",
    "request_id", "request_sha256", "jpeg_sha256", "file_sha256",
})


def _image_ref(value: Any, *, source_ref: str) -> dict:
    row = deepcopy(_mapping(value))
    from skill3d.storage.offline_frames import OVERVIEW_DESCRIPTOR_FIELDS
    is_overview = row.get("kind") == "scene_overview"
    overview_fields = OVERVIEW_DESCRIPTOR_FIELDS if is_overview else set()
    _require(not set(row) - (_IMAGE_REF_FIELDS | overview_fields),
             "image_refs must contain provenance only, not pixels or GT annotations")
    if is_overview:
        from skill3d.schemas.image_ledger import SceneOverviewImage
        _require(row.get("content_sha256") in (
            row.get("rgb_sha256"), f"sha256:{row.get('rgb_sha256')}"),
            "Scene overview content hash differs from its RGB identity")
        _require("file_sha256" not in row or row["file_sha256"] == row.get("png_sha256"),
                 "Scene overview file hash differs from its PNG identity")
        provenance = {
            key: value for key, value in row.items()
            if key in SceneOverviewImage.model_fields or key == "schema"
        }
        # File refs use bare digests and omit delivery-only dimensions. Validate
        # the same typed provenance without inventing observation facts in row.
        provenance["content_sha256"] = f"sha256:{row.get('rgb_sha256')}"
        provenance.setdefault("source_hw", [480, 640])
        provenance.setdefault("sent_hw", [480, 640])
        try:
            SceneOverviewImage.model_validate(provenance)
        except ValueError as exc:
            raise V11ExperienceBuildError("Invalid scene overview provenance") from exc
        _require(all(row.get(key) is None for key in (
            "source_frame_id", "source_frame_index", "frame_id", "frame_set_hash", "box_xyxy")),
            "A scene overview cannot claim video frame provenance")
    _require(all(
        not isinstance(value, Mapping)
        and (not isinstance(value, (list, tuple))
             or all(not isinstance(item, (Mapping, list, tuple)) for item in value))
        for value in row.values()
    ), "image_refs cannot contain embedded annotations")
    for key in ("path", "uri", "url", "source_ref", "trace_ref", "image_path"):
        if key in row:
            _require(isinstance(row[key], str) and not row[key].lower().startswith("data:"),
                     "image_refs cannot contain embedded image bytes")
    # Verify JSON metadata types without reading or generating any image content.
    try:
        json.dumps(row, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise V11ExperienceBuildError("image_refs must be JSON provenance metadata") from exc
    return {
        **row,
        "experience_provenance": {
            "source_ref": source_ref,
            "content_identity": (
                "recorded" if row.get("content_sha256") or row.get("file_sha256") else "unknown"
            ),
            "observation_status": (
                "recorded" if "observed_rounds" in row else "unknown"
            ),
        },
    }


def extract_case_image_refs(
    trace: Any, *, trace_ref: str = "", image_refs: Sequence[Mapping] = (),
) -> list[dict]:
    """Return recorded image metadata; an ID or path never implies observation."""
    recorded = _mapping(trace)
    ledger = recorded.get("image_ledger") or {}
    refs = [
        _image_ref(row, source_ref=f"{trace_ref}#/image_ledger/images/{position}")
        for position, row in enumerate(ledger.get("images") or [])
    ]
    for position, row in enumerate(_round_rows(recorded)):
        for field in ("image_refs", "selected_image_refs", "image_ids"):
            for offset, ref in enumerate(row.get(field) or []):
                value = {"image_id": ref} if isinstance(ref, str) else ref
                refs.append(_image_ref(
                    value, source_ref=f"{trace_ref}#/rounds/{position}/{field}/{offset}",
                ))
    refs.extend(
        _image_ref(row, source_ref=f"adapter:image_refs/{position}")
        for position, row in enumerate(image_refs)
    )
    from skill3d.storage.offline_frames import OVERVIEW_DESCRIPTOR_FIELDS
    overviews = {}
    for ref in refs:
        if ref.get("kind") == "scene_overview":
            identity = {key: ref[key] for key in OVERVIEW_DESCRIPTOR_FIELDS - {"image_path"}
                        if key in ref}
            previous = overviews.setdefault(ref["image_id"], identity)
            _require(previous == identity, "Recorded scene overview identities disagree")
    return refs


def extract_case_rounds(trace: Any, *, trace_ref: str = "") -> list[dict]:
    """Copy actual rounds and add links, never fill facts from the final round."""
    recorded = _mapping(trace)
    rows = _round_rows(recorded)
    result_refs: dict[str, list[str]] = {}
    for position, row in enumerate(rows):
        for offset, result in enumerate(row.get("results") or []):
            identity = result.get("result_id")
            if identity:
                result_refs.setdefault(str(identity), []).append(
                    f"{trace_ref}#/rounds/{position}/results/{offset}")
    ledger_rounds = (recorded.get("image_ledger") or {}).get("rounds") or []
    images = extract_case_image_refs(recorded, trace_ref=trace_ref)
    rounds: list[dict] = []
    for position, original in enumerate(rows):
        row = deepcopy(original)
        _require("experience_provenance" not in row,
                 "raw round collides with reserved experience_provenance")
        links = []
        referenced = [
            (field, identity)
            for field in ("observed_result_ids", "used_result_ids", "invalidated_result_ids")
            for identity in row.get(field) or []
        ]
        referenced.extend(
            ("observation_reads", read["result_id"])
            for read in row.get("observation_reads") or []
            if isinstance(read, Mapping) and read.get("result_id")
        )
        referenced.extend(
            ("result_dependencies", identity)
            for result in row.get("results") or []
            for identity in result.get("result_dependencies") or []
        )
        for field, identity in referenced:
            refs = result_refs.get(str(identity), [])
            links.append({
                "relation": field, "result_id": identity,
                "source_refs": list(refs),
                "status": "recorded" if refs else "unresolved",
            })
        index = row.get("index")
        image_rounds = [
            {"source_ref": f"{trace_ref}#/image_ledger/rounds/{offset}",
             "record": deepcopy(image_round)}
            for offset, image_round in enumerate(ledger_rounds)
            if index is not None and image_round.get("round_index") == index
        ]
        image_ids = {
            identity for image_round in image_rounds
            for identity in image_round["record"].get("image_ids") or []
        }
        linked_images = [
            deepcopy(ref) for ref in images
            if (ref["experience_provenance"]["source_ref"].startswith(
                f"{trace_ref}#/rounds/{position}/")
                or ref.get("image_id") in image_ids
                or (index is not None and (
                    ref.get("produced_round") == index
                    or index in (ref.get("delivered_rounds") or [])
                    or index in (ref.get("observed_rounds") or []))))
        ]
        row["experience_provenance"] = {
            "source_ref": f"{trace_ref}#/rounds/{position}",
            "missing_fields": [
                field for field in (
                    "index", "trigger", "synthesis_source", "program_id", "program_sha256",
                    "program_source", "response_text", "executed", "results", "error_code",
                    "observation_reads", "used_result_ids", "invalidated_result_ids",
                    "selected_object_ids",
                ) if field not in row or row[field] is None
            ],
            "result_links": links,
            "image_rounds": image_rounds,
            "image_refs": linked_images,
            "image_attribution": "recorded" if image_rounds or linked_images else "unknown",
        }
        rounds.append(row)
    return rounds


_INFRASTRUCTURE_CODES = frozenset({
    "input_error", "service_unavailable", "service_error", "online_model_service_error",
    "timeout", "oom", "disk_full",
})
_REAL_SYNTHESIS_SOURCES = frozenset({
    "online_model_ok", "online_model_parse_error", "m8_parse_recovered",
    "partial_tool_recovery",
})


def _validate_tool_results(results: Any) -> None:
    for result in results or []:
        result = _mapping(result)
        payload = result.get("payload") or {}
        _require(result.get("source") == "real",
                 "Mock or unknown tool results are not genuine program evidence")
        _require(payload.get("status") != "fault"
                 and payload.get("service_healthy") is not False,
                 "Tool service faults are not method experience")


def _validate_fitset_trace(recorded: dict, parent: SkillSpecV11) -> list[dict]:
    rows = _round_rows(recorded)
    _require(bool(rows), "A genuine completed program trace requires actual rounds")
    _require(recorded.get("final_state") in {
        "answer", "answer_best_effort", "run_error", "unanswerable", "abstain",
    }, "Unavailable/input-error outcomes are not method experience")
    _require(recorded.get("episode_status") in {"answered", "run_error"},
             "A completed episode_status must be recorded")
    _require((recorded["episode_status"] == "answered") == (
        recorded["final_state"] in {"answer", "answer_best_effort"}
    ), "Final state and episode_status mismatch")
    _require(recorded.get("status", "completed") in {"completed", "complete"},
             "Incomplete outcomes are not method experience")
    _require(not recorded.get("input_error_reason") and not recorded.get("input_error"),
             "Input infrastructure errors are not method experience")
    _require(not recorded.get("service_errors") and not recorded.get("service_error")
             and not (recorded.get("eval_visual_fallback") or {}).get("service_error"),
             "Service errors are not method experience")
    _require(recorded.get("failure_code") not in _INFRASTRUCTURE_CODES,
             "Infrastructure failure codes are not method experience")
    _require(recorded.get("label_access", False) is False,
             "Online label access is forbidden")
    key = f"{parent.skill_id}@{parent.version}"
    selection = recorded.get("initial_skill_selection") or {}
    _require(
        selection.get("question_type") == parent.question_type
        and selection.get("evidence_state") == parent.evidence_state
        and selection.get("skill_version") == key
        and selection.get("content_sha256") == parent.content_sha256,
        "Initial parent slot/identity/hash mismatch",
    )
    _require(set(recorded.get("delivered_skill_versions") or []) == {key},
             "Actual parent delivery mismatch")
    deliveries = []
    for record in recorded.get("retrieval_records") or []:
        record = _mapping(record)
        delivered = set(record.get("delivered_skill_versions") or [])
        _require(not delivered - {key}, "Actual parent changed during the episode")
        if key not in delivered:
            continue
        candidates = [
            _mapping(row) for row in record.get("candidates") or []
            if _mapping(row).get("skill_version") == key
        ]
        _require(len(candidates) == 1 and candidates[0].get("delivered") is True,
                 "Parent delivery must match an actually delivered candidate")
        deliveries.append((record.get("delivered_content_sha256") or {}).get(key))
    _require(bool(deliveries) and all(d == parent.content_sha256 for d in deliveries),
             "Actual parent delivery hash missing or mismatched")
    executed = []
    for row in rows:
        _require(row.get("label_access", False) is False, "Online label access is forbidden")
        _require(not row.get("service_errors") and not row.get("service_error")
                 and row.get("error_code") not in _INFRASTRUCTURE_CODES,
                 "Round service/infrastructure errors are not method experience")
        diagnostics = row.get("execution_diagnostics") or {}
        _require("SIGALRM" not in str(diagnostics.get("error_message") or ""),
                 "SIGALRM infrastructure errors are not method experience")
        _require(row.get("synthesis_source") in _REAL_SYNTHESIS_SOURCES,
                 "A genuine model response source must be recorded")
        if "delivered_skill_versions" in row:
            _require(set(row["delivered_skill_versions"]) == {key},
                     "Round parent delivery mismatch")
        _validate_tool_results(row.get("results"))
        if row.get("executed") is True:
            source = row.get("program_source")
            _require(isinstance(source, str) and bool(source.strip())
                     and isinstance(row.get("response_text"), str)
                     and bool(row["response_text"].strip())
                     and isinstance(row.get("results"), list),
                     "Executed rounds require original program/response/results")
            if row.get("program_sha256"):
                _require(row["program_sha256"] == hashlib.sha256(
                    source.encode("utf-8")).hexdigest(), "Round program hash mismatch")
            executed.append(row)
    _require(bool(executed), "Trace never executed a genuine program")
    program_trace = recorded.get("program_trace")
    if program_trace:
        program_trace = _mapping(program_trace)
        _require(program_trace.get("error_code") not in _INFRASTRUCTURE_CODES,
                 "Program infrastructure errors are not method experience")
        _require(program_trace.get("program_id") == executed[-1].get("program_id"),
                 "Program trace is not linked to the latest executed round")
        _require("SIGALRM" not in str(program_trace.get("error_message") or ""),
                 "SIGALRM infrastructure errors are not method experience")
        _validate_tool_results(program_trace.get("results"))
    return rows


def case_from_fitset_outcome(
    *, variant: Mapping, outcome: Any, parent: SkillSpecV11, label: Mapping,
    trace_ref: str, question_text: str, options: Sequence[str],
    frame_set_hash: str = "", image_refs: Sequence[Mapping] = (),
    experiment_split: Mapping | None = None,
) -> V11ExperienceCase:
    """Admit one induction outcome offline; no I/O or online label access.

    The first round's experience_provenance carries the question-level offline
    DataAccessRecord. Raw round facts and their online label flags stay unchanged.
    """
    source_split = variant.get("split")
    authorization = {}
    if experiment_split is not None:
        try:
            authorization = learning_authorization_for(experiment_split, [variant.get("qa_id")])
        except ValueError as exc:
            raise V11ExperienceBuildError(str(exc)) from exc
        _require(all(variant.get(field, "learning") == "learning"
                     for field in ("experiment_role", "experiment_split")),
                 "Validation role cannot enter learning case construction")
        try:
            validate_learning_source(authorization, variant.get("qa_id"), source_split)
        except ValueError as exc:
            raise V11ExperienceBuildError(str(exc)) from exc
        _require(variant.get("source_split", source_split) == source_split,
                 "Authorized learning must preserve the real source_split")
    else:
        _require(source_split == "induction"
                 and variant.get("source_split", "induction") == "induction",
                 "Fit-set learning accepts induction only")
    _require(bool(str(trace_ref).strip()) and bool(str(question_text).strip()),
             "trace_ref and original question_text are required")
    recorded = _mapping(outcome)
    qa_id, scene_id = variant.get("qa_id"), variant.get("scene_id")
    _require(bool(qa_id) and bool(scene_id) and bool(variant.get("variant_id")),
             "Variant QA/scene/variant identity is missing")
    _require(recorded.get("qa_id") == qa_id, "Outcome QA identity mismatch")
    scenes = [recorded[k] for k in ("scene_id", "scene_name") if recorded.get(k)]
    _require(bool(scenes) and all(s == scene_id for s in scenes),
             "Outcome scene identity mismatch")
    _require(canonical_task(str(variant.get("question_type") or "")) == parent.question_type
             and variant.get("evidence_state") == parent.evidence_state,
             "Variant task/initial evidence does not match the parent slot")
    _require(canonical_task(str(recorded.get("question_type") or "")) == parent.question_type,
             "Outcome task identity mismatch")
    if recorded.get("task"):
        _require(canonical_task(recorded["task"]) == parent.question_type,
                 "Outcome canonical task mismatch")
    for field in ("split", "source_split"):
        if field in recorded:
            _require(recorded[field] == source_split, "Outcome source split mismatch")
    rows = _validate_fitset_trace(recorded, parent)
    initial = deepcopy(recorded.get("initial_evidence_states") or {})
    history = deepcopy(recorded.get("evidence_history") or [])
    try:
        EvidenceProfile.model_validate(initial)
        _require(skill_evidence_state(initial) == parent.evidence_state,
                 "Actual initial evidence does not match the parent slot")
        _require(bool(history) and history[0]["evidence_states"] == initial,
                 "Initial evidence history is missing or mismatched")
        for row in history:
            EvidenceProfile.model_validate(row["evidence_states"])
        final = deepcopy(recorded.get("evidence_states", history[-1]["evidence_states"]))
        _require(final == history[-1]["evidence_states"], "Final evidence history mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        raise V11ExperienceBuildError(f"Invalid recorded evidence history: {exc}") from exc
    nested = recorded.get("episode_trace")
    if nested:
        nested = _mapping(nested)
        for field in ("qa_id", "initial_evidence_states", "initial_skill_selection",
                      "delivered_skill_versions", "rounds", "evidence_history",
                      "final_state", "episode_status"):
            _require(nested.get(field) == recorded.get(field),
                     f"Nested episode_trace {field} mismatch")
        _require(nested.get("episode_id") == qa_id
                 and nested.get("label_access", False) is False
                 and not nested.get("input_error_reason")
                 and nested.get("failure_code") not in _INFRASTRUCTURE_CODES,
                 "Nested episode identity/online label access mismatch")
    hashes = [str(value) for value in (
        frame_set_hash, variant.get("frame_set_hash"), recorded.get("frame_set_hash"),
        (nested or {}).get("frame_set_hash"),
    ) if value]
    _require(len(set(hashes)) <= 1, "Frame-set hash mismatch")
    refs = extract_case_image_refs(recorded, trace_ref=trace_ref, image_refs=image_refs)
    _require(all(not ref.get("frame_set_hash") or not hashes
                 or ref["frame_set_hash"] == hashes[0] for ref in refs),
             "Image reference frame-set hash mismatch")
    _require(all(ref.get("kind") != "scene_overview" or ref["scene_id"] == scene_id
                 for ref in refs), "Image overview scene identity mismatch")
    rounds = extract_case_rounds(recorded, trace_ref=trace_ref)

    # Read only question-level labels after trace provenance passes admission.
    _require(label.get("scene_id") == scene_id
             and canonical_task(str(label.get("question_type") or "")) == parent.question_type
             and label.get("qa_id", qa_id) == qa_id,
             "Offline label QA/scene/task identity mismatch")
    _require(list(label.get("options") or []) == list(options),
             "Offline label options mismatch")
    for metadata in (variant, label, recorded):
        if metadata.get("question_text") is not None:
            _require(metadata["question_text"] == question_text, "Original question text mismatch")
        if metadata.get("options") is not None:
            _require(list(metadata["options"]) == list(options), "Original options mismatch")
    reference = label.get("reference_answer")
    _require(isinstance(reference, str) and bool(reference.strip()),
             "Question-level reference answer is missing")
    from skill3d.evaluation.fitset_v11 import FitSetError, score_answer

    try:
        scored = score_answer(
            answer=recorded.get("answer"), task=parent.question_type,
            options=options, reference=reference,
            available=recorded["final_state"] in {"answer", "answer_best_effort"},
        )
    except FitSetError as exc:
        raise V11ExperienceBuildError(str(exc)) from exc
    score = scored["score"]
    categories = sorted({
        str(code) for code in [
            recorded.get("failure_code"), *(row.get("error_code") for row in rows),
        ] if code
    })
    if score < 1 and not categories:
        categories = ["answer_incorrect"]
    direction = ""
    if score < 1:
        if recorded["episode_status"] != "answered":
            direction = recorded["episode_status"]
        elif scored["metric"] == "Accuracy":
            direction = "wrong_option"
        else:
            predicted = parse_numeric_answer(recorded.get("answer"))
            expected = parse_numeric_answer(reference)
            direction = (
                "unparseable_numeric_answer" if predicted is None else
                "underestimate" if predicted < expected else
                "overestimate" if predicted > expected else "score_below_one"
            )
    key = f"{parent.skill_id}@{parent.version}"
    identity = _sha256({
        "variant_id": variant["variant_id"], "parent": key,
        "parent_sha256": parent.content_sha256, "trace_ref": trace_ref,
        "rounds": rows,
    })
    access = DataAccessRecord(
        record_id=f"label-access-{identity[:20]}", at=utcnow_iso(),
        split=source_split, purpose="skill_induction", component_role="inducer",
        run_id=str(variant["variant_id"]), n_items=1, label_access=True,
        input_manifest_sha256=_sha256({
            "qa_id": qa_id, "scene_id": scene_id, "question_type": parent.question_type,
            "options": list(options), "reference_answer": reference,
        }),
        source_refs=[trace_ref, f"offline-question-label:{qa_id}"],
        notes=["Only this authorized learning question's label was read offline; no GT 3D access"],
        learning_authorization=authorization,
    )
    rounds[0]["experience_provenance"]["offline_label_access_record"] = access.model_dump(mode="json")
    observations = collect_case_tool_observations(recorded, recorded.get("program_trace"))
    responses = [row["response_text"] for row in rows if isinstance(row.get("response_text"), str)]
    latest_executed = next(row for row in reversed(rows) if row.get("executed") is True)
    return V11ExperienceCase(
        case_id=f"case-{identity[:20]}", qa_id=qa_id, scene_id=scene_id,
        question_type=parent.question_type, evidence_state=parent.evidence_state,
        question_text=question_text, options=list(options),
        source_split=source_split, split="learning", label_access=True,
        learning_authorization=authorization,
        outcome="success" if score == 1 else "failure",
        episode_status=recorded["episode_status"],
        delivered_skill_version=key, delivered_content_sha256=parent.content_sha256,
        delivered_skill_md=parent.skill_md,
        response_text=responses[-1], response_texts=responses,
        program_source=latest_executed["program_source"],
        rounds=rounds, tool_observations=observations,
        initial_evidence_states=initial, evidence_states=final, evidence_history=history,
        candidate_objects=deepcopy(history[-1].get("candidates") or []),
        selected_object_ids=extract_recorded_selected_object_ids(recorded),
        image_observations=deepcopy(recorded.get("image_ledger") or {}),
        image_refs=refs, frame_set_hash=hashes[0] if hashes else "",
        model_answer=recorded.get("answer"), reference_answer=reference, score=score,
        failure_categories=categories, error_direction=direction, trace_ref=trace_ref,
    )


__all__ = [
    "V11ExperienceBuildError", "case_from_fitset_outcome",
    "collect_case_tool_observations", "extract_case_image_refs", "extract_case_rounds",
    "extract_recorded_selected_object_ids",
]
