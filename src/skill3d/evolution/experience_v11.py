"""Build v11 revision evidence from one completed real induction trace store."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skill3d.evaluation.mra import parse_numeric_answer
from skill3d.online.submission import EXECUTION_PROTOCOL_VERSION
from skill3d.routing.task_classifier import canonical_task
from skill3d.schemas import (
    DataAccessRecord,
    EpisodeInputTrace,
    EpisodeTrace,
    EvaluationResultTrace,
    ProgramExecutionTrace,
    SkillSpecV11,
    V11ExperienceBundle,
    V11ExperienceCase,
    utcnow_iso,
)
from skill3d.synthesis.prompt_builder import PROMPT_TEMPLATE_VERSION
from skill3d.tools.docs_v11 import TOOL_DOCS_VERSION


class V11ExperienceBuildError(ValueError):
    """Trace rows cannot be bound into trustworthy parent-learning evidence."""


def _read_jsonl(path: Path, *, required: bool = True) -> list[dict]:
    if not path.is_file():
        if required:
            raise V11ExperienceBuildError(f"缺少 trace topic: {path.name}")
        return []
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise V11ExperienceBuildError(
                f"{path.name}:{line_number} 不是合法 JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise V11ExperienceBuildError(
                f"{path.name}:{line_number} 必须是 JSON mapping"
            )
        rows.append(value)
    if required and not rows:
        raise V11ExperienceBuildError(f"trace topic 为空: {path.name}")
    return rows


def _index(rows: list[dict], *, key: str, topic: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        identity = str(row.get(key) or "")
        if not identity:
            raise V11ExperienceBuildError(f"{topic} 行缺少 {key}")
        if identity in indexed:
            raise V11ExperienceBuildError(f"{topic} 存在重复 {key}: {identity}")
        indexed[identity] = row
    return indexed


def _same_snapshot(actual: str, expected: str) -> bool:
    left = Path(str(actual or "")).stem.removeprefix("snapshot_")
    right = Path(str(expected or "")).stem.removeprefix("snapshot_")
    return bool(left and right and left == right)


def _delivered_parent_hash(trace: EpisodeTrace, parent_key: str) -> str:
    hashes: set[str] = set()
    for record in trace.retrieval_records:
        if not isinstance(record, dict):
            continue
        delivered = set(record.get("delivered_skill_versions") or [])
        content = record.get("delivered_content_sha256") or {}
        if parent_key in delivered and content.get(parent_key):
            candidates = [
                row
                for row in (record.get("candidates") or [])
                if row.get("skill_version") == parent_key
            ]
            if len(candidates) != 1 or not candidates[0].get("delivered"):
                raise V11ExperienceBuildError(
                    f"{trace.qa_id}: 交付 mapping 与候选行不一致"
                )
            hashes.add(str(content[parent_key]))
    if parent_key not in set(trace.delivered_skill_versions):
        raise V11ExperienceBuildError(
            f"{trace.qa_id}: EpisodeTrace 未记录实际交付 {parent_key}"
        )
    if len(hashes) != 1:
        raise V11ExperienceBuildError(
            f"{trace.qa_id}: 父 Skill 交付 hash 缺失或冲突: {sorted(hashes)}"
        )
    return next(iter(hashes))


def _answer_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _score_of(result: EvaluationResultTrace) -> float:
    if result.is_mca:
        if result.correct is None:
            raise V11ExperienceBuildError(
                f"{result.qa_id}: MCA 结果缺少 correct"
            )
        return float(bool(result.correct))
    if result.mra_value is None or not math.isfinite(float(result.mra_value)):
        raise V11ExperienceBuildError(
            f"{result.qa_id}: NA 结果缺少有限 mra_value"
        )
    score = float(result.mra_value)
    if not 0.0 <= score <= 1.0:
        raise V11ExperienceBuildError(
            f"{result.qa_id}: mra_value 越界: {score}"
        )
    return score


def _error_direction(result: EvaluationResultTrace, score: float) -> str:
    if score == 1.0:
        return ""
    if result.episode_status and result.episode_status != "answered":
        return result.episode_status
    if result.is_mca:
        return "wrong_option"
    predicted = parse_numeric_answer(result.answer_text)
    reference = parse_numeric_answer(result.ground_truth)
    if predicted is None or reference is None:
        return "unparseable_numeric_answer"
    if predicted < reference:
        return "underestimate"
    if predicted > reference:
        return "overestimate"
    return "score_below_one"


def _failure_categories(
    trace: EpisodeTrace,
    program_trace: ProgramExecutionTrace | None,
    *,
    score: float,
) -> list[str]:
    categories = list(trace.failure.categories if trace.failure is not None else [])
    if trace.failure_code:
        categories.append(f"failure_code:{trace.failure_code}")
    if program_trace is not None and program_trace.error_code:
        categories.append(f"program:{program_trace.error_code}")
    if score < 1.0 and not categories:
        categories.append("answer_incorrect")
    return sorted(set(categories))


def _case_id(
    campaign_id: str,
    source_run_ref: str,
    qa_id: str,
    parent_key: str,
) -> str:
    payload = "|".join((campaign_id, source_run_ref, qa_id, parent_key))
    return "case-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _all_tool_observations(
    trace: EpisodeTrace,
    program_trace: ProgramExecutionTrace | None,
) -> list[dict]:
    """Collect every round's ToolResult once, including pre-final observations."""
    candidates: list[dict] = []
    for round_row in trace.rounds:
        for result in round_row.get("results") or []:
            if isinstance(result, dict):
                candidates.append(dict(result))
    if program_trace is not None:
        candidates.extend(
            result.model_dump(mode="json") for result in program_trace.results
        )
    observations: list[dict] = []
    seen: set[str] = set()
    for row in candidates:
        identity = str(row.get("result_id") or "") or _json_sha256(row)
        if identity in seen:
            continue
        seen.add(identity)
        observations.append(row)
    return observations


def build_v11_experience_bundle_from_trace_store(
    trace_dir: str | Path,
    *,
    campaign_id: str,
    parent_snapshot_id: str,
    parent: SkillSpecV11,
    source_run_ref: str = "",
    require_real: bool = True,
) -> V11ExperienceBundle:
    """Join v11 induction topics and return complete evidence for the reviser."""
    access_started_at = utcnow_iso()
    root = Path(trace_dir)
    online_rows = _read_jsonl(root / "online_run.jsonl")
    if len(online_rows) != 1:
        raise V11ExperienceBuildError("online_run 必须恰有一行")
    online = online_rows[0]
    if str(online.get("split") or "") != "induction":
        raise V11ExperienceBuildError("v11 Skill 修订只接受 split=induction")
    if require_real and str(online.get("mode") or "") != "real":
        raise V11ExperienceBuildError("正式 Skill 修订经验必须来自 mode=real")
    if online.get("label_access") is not False:
        raise V11ExperienceBuildError(
            "在线求解 run 必须显式记录 label_access=false")
    expected_protocol = {
        "template_version": PROMPT_TEMPLATE_VERSION,
        "execution_protocol_version": EXECUTION_PROTOCOL_VERSION,
        "tool_docs_version": TOOL_DOCS_VERSION,
    }
    mismatched = {
        key: (online.get(key), value)
        for key, value in expected_protocol.items()
        if online.get(key) != value
    }
    if mismatched:
        raise V11ExperienceBuildError(f"在线协议身份不匹配: {mismatched}")
    run_ref = str(source_run_ref or f"online_run:{online.get('run_id') or ''}")
    if not run_ref or run_ref.endswith(":"):
        raise V11ExperienceBuildError("缺少 source_run_ref/run_id")

    inputs = _index(
        _read_jsonl(root / "episode_input.jsonl"),
        key="qa_id",
        topic="episode_input",
    )
    traces = _index(
        _read_jsonl(root / "episode_trace.jsonl"),
        key="qa_id",
        topic="episode_trace",
    )
    evaluations = _index(
        _read_jsonl(root / "evaluation_result.jsonl"),
        key="qa_id",
        topic="evaluation_result",
    )
    programs = _index(
        _read_jsonl(root / "episode_program.jsonl", required=False),
        key="qa_id",
        topic="episode_program",
    )
    program_traces = _index(
        _read_jsonl(root / "program_trace.jsonl", required=False),
        key="program_id",
        topic="program_trace",
    )
    if set(inputs) != set(traces) or set(inputs) != set(evaluations):
        raise V11ExperienceBuildError(
            "episode_input、episode_trace、evaluation_result 的 qa_id 集合不一致"
        )
    bad_splits = sorted(
        qa_id
        for qa_id, row in inputs.items()
        if row.get("source_split") != "induction"
    )
    if bad_splits:
        raise V11ExperienceBuildError(
            f"trace 混入非 induction episode: {bad_splits}"
        )
    bad_label_rows = sorted(
        qa_id
        for qa_id in inputs
        if inputs[qa_id].get("label_access") is not False
        or evaluations[qa_id].get("label_access") is not False
        or traces[qa_id].get("label_access") is not False
    )
    if bad_label_rows:
        raise V11ExperienceBuildError(
            f"在线 trace 的 label_access 必须显式为 false: {bad_label_rows}")
    if online.get("n_episodes") is not None and int(online["n_episodes"]) != len(inputs):
        raise V11ExperienceBuildError(
            "online_run.n_episodes 与 episode_input 行数不一致"
        )

    parent_key = f"{parent.skill_id}@{parent.version}"
    selected_ids = sorted(
        qa_id
        for qa_id, row in inputs.items()
        if canonical_task(str(row.get("question_type") or ""))
        == parent.question_type
    )
    if not selected_ids:
        raise V11ExperienceBuildError(
            f"trace 中没有题型 {parent.question_type!r} 的 induction case"
        )

    cases: list[V11ExperienceCase] = []
    for qa_id in selected_ids:
        try:
            episode_input = EpisodeInputTrace.model_validate(inputs[qa_id])
            trace = EpisodeTrace.model_validate(traces[qa_id])
            result = EvaluationResultTrace.model_validate(evaluations[qa_id])
        except KeyError as exc:
            raise V11ExperienceBuildError(
                f"{qa_id}: episode_input/episode_trace/evaluation_result 连接不完整"
            ) from exc
        except Exception as exc:
            raise V11ExperienceBuildError(f"{qa_id}: trace Schema 非法: {exc}") from exc

        if episode_input.source_split != "induction":
            raise V11ExperienceBuildError(
                f"{qa_id}: source_split={episode_input.source_split!r}，不是 induction"
            )
        if canonical_task(episode_input.question_type) != parent.question_type:
            raise V11ExperienceBuildError(f"{qa_id}: episode 题型与父 Skill 不一致")
        if canonical_task(result.question_type) != parent.question_type:
            raise V11ExperienceBuildError(f"{qa_id}: 评分题型与父 Skill 不一致")
        if trace.qa_id != qa_id or trace.episode_id != qa_id:
            raise V11ExperienceBuildError(f"{qa_id}: EpisodeTrace 身份不一致")
        if not _same_snapshot(trace.active_snapshot_ref, parent_snapshot_id):
            raise V11ExperienceBuildError(
                f"{qa_id}: 运行快照不是父快照 {parent_snapshot_id}"
            )
        if episode_input.reference_answer != result.ground_truth:
            raise V11ExperienceBuildError(f"{qa_id}: 标准答案在输入与评分记录中不一致")
        if (
            episode_input.frame_set_hash
            and trace.frame_set_hash
            and episode_input.frame_set_hash != trace.frame_set_hash
        ):
            raise V11ExperienceBuildError(f"{qa_id}: frame_set_hash 不一致")

        delivery_sha = _delivered_parent_hash(trace, parent_key)
        if delivery_sha != parent.content_sha256:
            raise V11ExperienceBuildError(
                f"{qa_id}: 实际交付正文 hash 不是父 Skill hash"
            )

        program_row = programs.get(qa_id)
        program_source = ""
        response_text = ""
        response_texts: list[str] = []
        program_trace: ProgramExecutionTrace | None = None
        if program_row is not None:
            if parent_key not in set(program_row.get("delivered_skill_versions") or []):
                raise V11ExperienceBuildError(
                    f"{qa_id}: episode_program 未记录父 Skill 交付"
                )
            program_source = str(program_row.get("program_source") or "")
            raw_responses = program_row.get("response_texts")
            if not isinstance(raw_responses, list) or not raw_responses:
                raise V11ExperienceBuildError(
                    f"{qa_id}: episode_program 缺少原始 response_texts"
                )
            response_texts = [str(value) for value in raw_responses]
            response_text = response_texts[-1]
            program_id = str(program_row.get("program_id") or "")
            if program_id:
                raw_program_trace = program_traces.get(program_id)
                if raw_program_trace is None:
                    raise V11ExperienceBuildError(
                        f"{qa_id}: 缺少 program_trace:{program_id}"
                    )
                try:
                    program_trace = ProgramExecutionTrace.model_validate(
                        raw_program_trace
                    )
                except Exception as exc:
                    raise V11ExperienceBuildError(
                        f"{qa_id}: ProgramExecutionTrace 非法: {exc}"
                    ) from exc
                if trace.program_trace_ref not in (
                    "",
                    f"program_trace:{program_id}",
                ):
                    raise V11ExperienceBuildError(
                        f"{qa_id}: program_trace_ref 与 episode_program 不一致"
                    )
        elif trace.rounds:
            program_source = str(trace.rounds[-1].get("program_source") or "")
            response_texts = [
                str(row["response_text"])
                for row in trace.rounds
                if row.get("response_text") is not None
            ]
            if not response_texts:
                raise V11ExperienceBuildError(
                    f"{qa_id}: round trace 缺少原始 response_text"
                )
            response_text = response_texts[-1]
        else:
            raise V11ExperienceBuildError(f"{qa_id}: 缺少模型原始回复")

        score = _score_of(result)
        outcome = "success" if score == 1.0 else "failure"
        categories = _failure_categories(trace, program_trace, score=score)
        direction = _error_direction(result, score)
        cases.append(V11ExperienceCase(
            case_id=_case_id(campaign_id, run_ref, qa_id, parent_key),
            qa_id=qa_id,
            scene_id=episode_input.scene_id,
            question_type=parent.question_type,
            question_text=episode_input.question_text,
            options=list(episode_input.options),
            source_split="induction",
            split="learning",
            label_access=True,
            outcome=outcome,
            episode_status=result.episode_status or trace.episode_status,
            delivered_skill_version=parent_key,
            delivered_content_sha256=delivery_sha,
            delivered_skill_md=parent.skill_md,
            response_text=response_text,
            response_texts=response_texts,
            program_source=program_source,
            tool_observations=_all_tool_observations(trace, program_trace),
            model_answer=_answer_text(result.answer_text),
            reference_answer=episode_input.reference_answer,
            score=score,
            failure_categories=categories,
            error_direction=direction,
            trace_ref=f"{run_ref}:qa:{qa_id}",
        ))

    access_manifest = [
        {
            "qa_id": qa_id,
            "episode_input_sha256": _json_sha256(inputs[qa_id]),
            "evaluation_result_sha256": _json_sha256(evaluations[qa_id]),
        }
        for qa_id in selected_ids
    ]
    access_record = DataAccessRecord(
        record_id=(
            "label-access-"
            + _json_sha256({
                "campaign_id": campaign_id,
                "run_ref": run_ref,
                "parent": parent_key,
                "qa_ids": selected_ids,
            })[:20]
        ),
        at=access_started_at,
        split="induction",
        purpose="skill_induction",
        component_role="inducer",
        run_id=str(online.get("run_id") or ""),
        input_manifest_sha256=_json_sha256(access_manifest),
        n_items=len(cases),
        label_access=True,
        source_refs=[
            str(root / "episode_input.jsonl"),
            str(root / "evaluation_result.jsonl"),
            str(root / "episode_trace.jsonl"),
            str(root / "episode_program.jsonl"),
            str(root / "program_trace.jsonl"),
        ],
        notes=["只读取题目级答案与评分；未授权读取 GT 三维标注"],
    )
    return V11ExperienceBundle(
        campaign_id=campaign_id,
        parent_snapshot_id=parent_snapshot_id,
        parent_skill_version=parent_key,
        parent_content_sha256=parent.content_sha256,
        question_type=parent.question_type,
        source_split="induction",
        split="learning",
        label_access=True,
        source_run_ref=run_ref,
        label_access_record=access_record,
        cases=cases,
    )


@dataclass(frozen=True)
class V11TraceCollector:
    """Injectable campaign callback backed by a completed trace directory."""

    trace_dir: str | Path
    source_run_ref: str = ""
    require_real: bool = True

    def __call__(self, **kwargs) -> V11ExperienceBundle:
        parent = kwargs["parent"]
        if kwargs.get("question_type") != parent.question_type:
            raise V11ExperienceBuildError("collector question_type 与父 Skill 不一致")
        return build_v11_experience_bundle_from_trace_store(
            self.trace_dir,
            campaign_id=str(kwargs["campaign_id"]),
            parent_snapshot_id=str(kwargs["parent_snapshot_id"]),
            parent=parent,
            source_run_ref=self.source_run_ref,
            require_real=self.require_real,
        )


__all__ = [
    "V11ExperienceBuildError",
    "V11TraceCollector",
    "build_v11_experience_bundle_from_trace_store",
]
