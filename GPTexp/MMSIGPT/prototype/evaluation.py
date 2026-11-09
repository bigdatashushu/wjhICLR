"""Freeze public predictions first; read pinned host references only when scoring."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from ...traces import digest, file_digest, read, write
from .direct import operator_lock, trace_paths, validate_manifest
from .protocol import LETTERS, PARSER_VERSION, parse_answer, prompt


def collect_predictions(directory, manifest):
    records, files = [], {}
    started = read(Path(directory) / "execution.started.json")
    if (started["run_identity"] != manifest["run_identity"]
            or started["max_requests"] != manifest["max_requests"]
            or started["reference_answer_visible"] is not False or started["automatic_retry"] is not False):
        raise ValueError("execution_start_identity_changed")
    known_traces = {path.name for item in manifest["items"] for path in trace_paths(directory, item)}
    traces = Path(directory) / "traces"
    if traces.exists() and any(path.name not in known_traces for path in traces.iterdir()):
        raise ValueError("unexpected_trace_outside_request_budget")
    for name in ("manifest.json", "preparation.json", "execution.started.json", "execution.json"):
        path = Path(directory) / name
        if path.exists():
            files[name] = file_digest(path)
    for item in manifest["items"]:
        dispatch_path, result_path = trace_paths(directory, item)
        prediction = {
            "index": item["index"], "sample_id": item["problem"]["sample_id"],
            "task_type": item["problem"]["task_type"], "source_split": item["problem"]["source_split"],
            "input_sha256": item["problem"]["input_sha256"], "answer": None, "status": "not_dispatched",
        }
        if result_path.exists() and not dispatch_path.exists():
            raise ValueError("result_without_dispatch")
        if dispatch_path.exists():
            dispatch = read(dispatch_path)
            if (dispatch["run_identity"] != manifest["run_identity"]
                    or dispatch["sample_id"] != prediction["sample_id"] or dispatch["index"] != item["index"]
                    or dispatch["input_sha256"] != prediction["input_sha256"]
                    or dispatch["payload_sha256"] != item["payload_sha256"]
                    or dispatch["prompt_sha256"] != item["prompt_sha256"]
                    or dispatch["images"] != item["images"] or dispatch["requested"] != manifest["inference"]
                    or dispatch["reference_answer_visible"] is not False
                    or dispatch["automatic_retry"] is not False):
                raise ValueError("dispatch_identity_changed")
            files[str(dispatch_path.relative_to(directory))] = file_digest(dispatch_path)
            prediction["status"] = "outcome_unknown"
        if result_path.exists():
            result = read(result_path)
            if (result["run_identity"] != manifest["run_identity"] or result["index"] != item["index"]
                    or result["sample_id"] != prediction["sample_id"]
                    or result["input_sha256"] != prediction["input_sha256"]
                    or result["dispatch_sha256"] != file_digest(dispatch_path)
                    or result["reference_answer_visible"] is not False
                    or result["automatic_retry"] is not False):
                raise ValueError("result_identity_changed")
            if result["status"] in ("answered", "invalid_answer"):
                parsed = parse_answer(result["response"]["text"], len(item["problem"]["options"]))
                expected_status = "answered" if parsed["answer"] is not None else "invalid_answer"
                if (result["answer"] != parsed["answer"] or result["parse_status"] != parsed["parse_status"]
                        or result["status"] != expected_status
                        or result["response"]["tool_call_count"] != 0
                        or result["response"]["response_metadata_policy"]
                        != manifest["inference"]["response_metadata_policy"]):
                    raise ValueError("stored_prediction_parse_changed")
            elif result["status"] != "service_error" or result["answer"] is not None:
                raise ValueError("invalid_result_status")
            prediction.update(answer=result["answer"], status=result["status"])
            files[str(result_path.relative_to(directory))] = file_digest(result_path)
        records.append(prediction)
    dispatched = [record for record in records if record["status"] != "not_dispatched"]
    if [record["index"] for record in dispatched] != list(range(len(dispatched))):
        raise ValueError("dispatch_sequence_changed")
    if len(dispatched) > manifest["max_requests"]:
        raise ValueError("request_budget_exceeded")
    execution_path = Path(directory) / "execution.json"
    if execution_path.exists():
        execution = read(execution_path)
        if (execution["run_identity"] != manifest["run_identity"]
                or execution["dispatched"] != len(dispatched)
                or execution["reference_answer_visible"] is not False
                or execution["automatic_retry"] is not False):
            raise ValueError("execution_summary_identity_changed")
        status = execution["status"]
        if status == "complete":
            if len(dispatched) != manifest["questions"] or any(
                    record["status"] not in ("answered", "invalid_answer") for record in records):
                raise ValueError("complete_execution_missing_results")
        elif status == "stopped_service_error":
            if not dispatched or dispatched[-1]["status"] != "service_error" or any(
                    record["status"] not in ("answered", "invalid_answer") for record in dispatched[:-1]):
                raise ValueError("service_stop_summary_changed")
        elif status != "interrupted":
            raise ValueError("unknown_execution_status")
    return records, files


def freeze(directory):
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate_manifest(directory)
        if (directory / "frozen.json").exists():
            return validate_frozen(directory, manifest)
        if not (directory / "execution.started.json").exists():
            raise ValueError("cannot_freeze_unstarted_run")
        predictions, files = collect_predictions(directory, manifest)
        frozen = {
            "schema": "mmsi-direct-frozen/1", "run_identity": manifest["run_identity"],
            "parser_version": PARSER_VERSION, "predictions": predictions, "files": files,
            "questions": manifest["questions"],
            "status_counts": dict(Counter(p["status"] for p in predictions)),
            "reference_answer_visible": False, "automatic_retry": False, "frozen_at": time.time(),
        }
        frozen["freeze_identity"] = digest(frozen)
        write(directory / "frozen.json", frozen, exclusive=True)
        return frozen


def validate_frozen(directory, manifest):
    frozen = read(Path(directory) / "frozen.json")
    if (frozen["schema"] != "mmsi-direct-frozen/1"
            or frozen["run_identity"] != manifest["run_identity"]
            or frozen["parser_version"] != PARSER_VERSION
            or digest({k: v for k, v in frozen.items() if k != "freeze_identity"}) != frozen["freeze_identity"]):
        raise ValueError("prediction_freeze_identity_changed")
    predictions, files = collect_predictions(directory, manifest)
    if frozen["predictions"] != predictions or frozen["files"] != files:
        raise ValueError("records_changed_after_prediction_freeze")
    return frozen


def reference_answers(raw, predictions):
    data = json.loads(raw)
    rows = data.get("references") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("requires_host_reference_row_list")
    answers = {}
    for row in rows:
        if row.get("benchmark", row.get("dataset")) != "mmsi":
            continue
        sample = row["sample_id"]
        if sample in answers:
            raise ValueError("duplicate_host_reference")
        answers[sample] = row["reference_answer"]
    selected = {}
    for prediction in predictions:
        sample = prediction["sample_id"]
        if sample not in answers:
            raise ValueError("missing_host_reference")
        selected[sample] = answers[sample]
    return selected


def score(directory, references, expected_sha256):
    directory = Path(directory)
    with operator_lock(directory):
        # All integrity checks precede opening the reference file.
        manifest = validate_manifest(directory)
        frozen = validate_frozen(directory, manifest)
        raw = Path(references).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("host_reference_bytes_changed")
        answers = reference_answers(raw, frozen["predictions"])
        for item in manifest["items"]:
            if answers[item["problem"]["sample_id"]] not in LETTERS[:len(item["problem"]["options"])]:
                raise ValueError("invalid_reference_option_letter")
        scored = [
            {**prediction, "reference_answer": answers[prediction["sample_id"]],
             "correct": prediction["status"] == "answered"
                        and prediction["answer"] == answers[prediction["sample_id"]]}
            for prediction in frozen["predictions"]
        ]

        def summarize(rows):
            total = len(rows)
            legal = sum(row["status"] == "answered" for row in rows)
            correct = sum(row["correct"] for row in rows)
            return {
                "questions": total, "legal_answers": legal, "correct": correct,
                "accuracy": correct / total if total else None,
                "legal_answer_rate": legal / total if total else None,
                "accuracy_among_legal": correct / legal if legal else None,
                "status_counts": dict(Counter(row["status"] for row in rows)),
            }

        report = {
            "schema": "mmsi-direct-score/1", "run_identity": manifest["run_identity"],
            "freeze_identity": frozen["freeze_identity"], "reference_file_sha256": expected_sha256,
            "metric": "extracted_letter_exact_match", "judge_fallback": False,
            "denominator": "all_selected_questions_including_invalid_errors_and_unrun",
            "overall": summarize(scored),
            "by_task": {
                task: summarize([row for row in scored if row["task_type"] == task])
                for task in sorted(manifest["task_counts"])
            },
            "predictions": scored, "scored_at": time.time(),
            "reference_access": "host_only_after_prediction_freeze",
        }
        report["score_identity"] = digest(report)
        path = directory / "score.json"
        if path.exists():
            prior = read(path)
            if any(prior[key] != report[key] for key in report if key not in ("scored_at", "score_identity")):
                raise ValueError("existing_score_identity_changed")
            if digest({key: value for key, value in prior.items() if key != "score_identity"}) != prior["score_identity"]:
                raise ValueError("existing_score_identity_changed")
            return prior
        write(path, report, exclusive=True)
        return report


def export(directory):
    """Compact JSON review with original-image paths; no image copies or web assets."""
    directory = Path(directory)
    with operator_lock(directory):
        manifest = validate_manifest(directory)
        frozen = validate_frozen(directory, manifest)
        scoring = read(directory / "score.json") if (directory / "score.json").exists() else None
        if scoring and (
                scoring["freeze_identity"] != frozen["freeze_identity"]
                or scoring["run_identity"] != manifest["run_identity"]
                or digest({key: value for key, value in scoring.items() if key != "score_identity"})
                != scoring["score_identity"]):
            raise ValueError("score_freeze_identity_changed")
        scores = {p["sample_id"]: p for p in scoring["predictions"]} if scoring else {}
        rows = []
        for item, prediction in zip(manifest["items"], frozen["predictions"]):
            _, result_path = trace_paths(directory, item)
            result = read(result_path) if result_path.exists() else None
            rows.append({
                **prediction, "question": item["problem"]["question"],
                "options": dict(zip(LETTERS, item["problem"]["options"])),
                "original_image_paths": item["problem"]["image_paths"], "images": item["images"],
                "prompt": prompt(item["problem"]), "requested": manifest["inference"],
                "response": result.get("response") if result else None,
                "error_code": result.get("error_code") if result else None,
                **({"reference_answer": scores[prediction["sample_id"]]["reference_answer"],
                    "correct": scores[prediction["sample_id"]]["correct"]} if scoring else {}),
            })
        review = {
            "schema": "mmsi-direct-review/1", "run_identity": manifest["run_identity"],
            "freeze_identity": frozen["freeze_identity"], "scored": scoring is not None,
            "rows": rows, "overall": scoring["overall"] if scoring else None,
        }
        write(directory / "review.json", review)
        return {"path": str(directory / "review.json"), "questions": len(rows),
                "scored": review["scored"]}
