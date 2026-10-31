#!/usr/bin/env python3
"""Run Qwen on original BLINK multi-view images and score locally.

Reuse ../src/skill3d/synthesis/bailian_client.py, including its authentication,
OpenAI-compatible request body, timeout behavior and response identity.
No automatic model-call retries. GT is read only in the separate scoring stage.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error

from download_data import ROOT, SUBSET, sha256_file, write_json, write_jsonl

sys.path.insert(0, str(ROOT.parent / "src"))
from skill3d.synthesis.bailian_client import BailianClient, DEFAULT_BASE_URL

SCHEMA = "blink-qwen-evaluation/1"
MODEL = "qwen3.8-flash"
PARAMETERS = {
    "max_tokens": 8192, "temperature": 0.0, "seed": 137,
    "enable_thinking": True,
}
PROMPT_SUFFIX = (
    '\n\nUse the two supplied images in their original order. '
    'Give a concise explanation based on their visual differences, then select '
    'one option. Return only a valid JSON object with the fields '
    '"reasoning" (a brief explanation) and "answer" (one option letter, A or B).'
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parameters_for(args) -> dict:
    return {**PARAMETERS, "enable_thinking": args.thinking}


class ThinkingBailianClient(BailianClient):
    """Capture provider reasoning while retaining the existing client protocol.

    The inherited client reads the normal final content and metadata. This small
    transport wrapper also captures message.reasoning_content, which that
    client otherwise discards. Only provider-returned reasoning is recorded.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_reasoning_content = None
        self._transport_open = self._opener
        self._opener = self._capture_response

    def _capture_response(self, request, *, timeout):
        with self._transport_open(request, timeout=timeout) as response:
            raw = response.read()
        body = json.loads(raw.decode("utf-8"))
        choices = body.get("choices") or []
        if choices:
            reasoning = (choices[0].get("message") or {}).get("reasoning_content")
            self.last_reasoning_content = reasoning if isinstance(reasoning, str) else None
        return BytesIO(raw)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def select_inputs(root: Path, splits: list[str]) -> list[dict]:
    rows = [row for row in read_jsonl(root / "data" / "inputs.jsonl")
            if row["split"] in splits]
    rows.sort(key=lambda row: (splits.index(row["split"]), row["row_index"]))
    if not rows or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("empty_or_duplicate_input_index")
    for row in rows:
        if (row["subset"] != SUBSET or row.get("reference_answer_visible") is not False
                or any(key in row for key in
                       ("answer", "gt_answer", "gt_raw", "gt_explanation", "explanation"))):
            raise ValueError("public_input_contains_reference_fields")
        if len(row["images"]) != 2 or [o["label"] for o in row["options"]] != ["A", "B"]:
            raise ValueError("unexpected_multi_view_question_format")
        if not isinstance(row["benchmark_prompt"], str) or not row["benchmark_prompt"]:
            raise ValueError("missing_official_benchmark_prompt")
    return rows


def messages_for(row: dict, root: Path) -> list[dict]:
    content = [{"type": "text", "text": row["benchmark_prompt"] + PROMPT_SUFFIX}]
    for i, image in enumerate(row["images"], 1):
        path = (root / image["path"]).resolve()
        if not path.is_relative_to((root / "data" / "images").resolve()):
            raise ValueError("image_path_outside_dataset")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != image["sha256"]:
            raise ValueError("original_image_bytes_changed")
        content.append({"type": "text", "text": f"Image {i}:"})
        content.append({"type": "image_url", "image_url": {
            "url": f"data:{image['mime_type']};base64,"
                   + base64.b64encode(raw).decode("ascii"),
        }})
    return [{"role": "user", "content": content}]


def new_client(args) -> ThinkingBailianClient:
    return ThinkingBailianClient(
        model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
        timeout_s=args.timeout, temperature=PARAMETERS["temperature"],
        max_output_tokens=PARAMETERS["max_tokens"],
    )


def resolve_api_key(args) -> str:
    existing = os.environ.get(args.api_key_env, "").strip()
    if existing:
        return existing
    # Read the existing private key file; never copy it into BLINK or print it.
    key_file = args.key_file
    if key_file is None:
        default_file = ROOT.parent / "qwenkey.md"
        key_file = default_file if default_file.exists() else None
    if key_file is None:
        raise ValueError("missing_api_credential")
    text = key_file.read_text(encoding="utf-8").strip()
    # Existing credentials can contain periods; do not truncate at punctuation.
    if text.startswith("sk-") and re.fullmatch(r"[A-Za-z0-9._~+/-]+=*", text):
        return text
    candidates = re.findall(r"\bsk-[A-Za-z0-9._~+/-]+=*", text)
    if len(set(candidates)) != 1:
        raise ValueError("key_file_requires_exactly_one_api_key")
    return candidates[0]


def prepare_manifest(args) -> tuple[dict, list[dict]]:
    root, output = args.root, args.output
    parameters = parameters_for(args)
    rows = select_inputs(root, args.splits)
    data = json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(root / "data" / "inputs.jsonl") != data["input_index_sha256"]:
        raise ValueError("prepared_input_index_changed")
    client = new_client(args)
    inputs = []
    for row in rows:
        payload = client.build_payload(messages_for(row, root), **parameters)
        inputs.append({"sample_id": row["sample_id"], "payload_sha256": digest(payload)})
    source_files = [
        Path(__file__), ROOT / "download_data.py",
        ROOT.parent / "src" / "skill3d" / "synthesis" / "bailian_client.py",
        ROOT.parent / "src" / "skill3d" / "synthesis" / "request_audit.py",
        ROOT.parent / "src" / "skill3d" / "synthesis" / "client_errors.py",
    ]
    body = {
        "schema_version": SCHEMA, "subset": SUBSET, "splits": args.splits,
        "questions": len(rows), "model": args.model,
        "endpoint_sha256": client.endpoint_hash, "parameters": parameters,
        "prompt_suffix": PROMPT_SUFFIX, "prompt_source": "dataset.prompt",
        "image_encoding": "unchanged original bytes as base64 image_url",
        "images_per_question": 2, "model_calls_per_question": 1,
        "request_cap": len(rows), "concurrency": args.concurrency,
        "timeout_seconds": args.timeout, "automatic_retry": False,
        "reference_answer_visible": False, "gt_access": "local scoring only",
        "dataset_revision": data["resolved_revision"],
        "input_index_sha256": data["input_index_sha256"],
        "reference_index_sha256": data["reference_index_sha256"],
        "source_files": {str(p.relative_to(ROOT.parent)): sha256_file(p)
                         for p in source_files},
        "inputs": inputs,
    }
    manifest = {**body, "identity_sha256": digest(body)}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("frozen_evaluation_configuration_changed_use_new_output")
    else:
        write_json(manifest_path, manifest)
    return manifest, rows


def parse_answer(text: str, options: list[dict]) -> tuple[str | None, str | None]:
    """Read an explicit final answer; never infer it from GT or explanation."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean, flags=re.IGNORECASE).strip()
    labels = {o["label"] for o in options}
    try:
        value = json.loads(clean)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict):
        answer = value.get("answer")
        reasoning = value.get("reasoning")
        if not isinstance(reasoning, str):
            reasoning = None
        if isinstance(answer, str):
            candidate = answer.strip().strip("().: ").upper()
            if candidate in labels:
                return candidate, reasoning
        return None, reasoning
    candidate = clean.strip("().: ").upper()
    return (candidate if candidate in labels else None), None


def infer_one(args, row: dict, payload_sha: str) -> dict:
    client = new_client(args)  # Per-request client keeps metadata thread-safe.
    parameters = parameters_for(args)
    start = time.monotonic()
    record = {
        "sample_id": row["sample_id"], "split": row["split"],
        "requested_model": args.model, "payload_sha256": payload_sha,
        "started_at": now(), "reference_answer_visible": False,
        "api_calls": 0, "automatic_retry": False,
    }
    try:
        messages = messages_for(row, args.root)
        if digest(client.build_payload(messages, **parameters)) != payload_sha:
            raise ValueError("prepared_model_request_changed")
        record["api_calls"] = 1
        text = client.chat(messages, **parameters)
        reasoning_content = client.last_reasoning_content
        if (os.environ.get(args.api_key_env) in text
                or reasoning_content and os.environ.get(args.api_key_env) in reasoning_content):
            raise ValueError("credential_in_response")
        actual_model = client.last_response_identity.get("model")
        answer, reasoning = parse_answer(text, row["options"])
        status = ("model_mismatch" if actual_model != args.model
                  else "ok" if answer is not None else "parse_error")
        record.update({
            "status": status, "model_response": text, "model_reasoning": reasoning,
            "reasoning_content": reasoning_content,
            "thinking_enabled": parameters["enable_thinking"],
            "prediction": answer, "actual_model": actual_model,
            "response_identity": client.last_response_identity,
            "usage": client.last_usage,
            "finish_reason": client.last_generation.get("finish_reason"),
        })
    except Exception as exc:
        # Provider exception messages may contain credentials or request details.
        cause = exc
        http_status = None
        while cause is not None:
            if isinstance(cause, urllib.error.HTTPError):
                http_status = cause.code
                break
            cause = cause.__cause__
        record.update({
            "status": "api_error", "error_type": type(exc).__name__,
            "error_http_status": http_status,
            "model_response": None, "model_reasoning": None, "prediction": None,
            "reasoning_content": None, "thinking_enabled": parameters["enable_thinking"],
            "actual_model": None, "usage": {},
        })
    record.update({"finished_at": now(), "latency_seconds": time.monotonic() - start})
    return record


def load_predictions(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate_prediction_records")
    return {row["sample_id"]: row for row in rows}


def score(root: Path, output: Path) -> dict:
    """Join local GT only after inference, retaining hidden GT as null."""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(root / "data" / "inputs.jsonl") != manifest["input_index_sha256"]:
        raise ValueError("scoring_input_index_changed")
    if sha256_file(root / "data" / "references.jsonl") != manifest["reference_index_sha256"]:
        raise ValueError("scoring_reference_index_changed")
    rows = select_inputs(root, manifest["splits"])
    predictions = load_predictions(output / "predictions.jsonl")
    expected = {r["sample_id"] for r in rows}
    if set(predictions) - expected:
        raise ValueError("predictions_contain_unknown_samples")
    references = {r["sample_id"]: r for r in read_jsonl(root / "data" / "references.jsonl")}
    results = []
    for row in rows:
        prediction = predictions.get(row["sample_id"], {
            "status": "not_attempted", "prediction": None,
            "model_response": None, "model_reasoning": None, "api_calls": 0,
            "reasoning_content": None, "thinking_enabled": manifest["parameters"]["enable_thinking"],
        })
        reference = references[row["sample_id"]]
        correct = None
        if reference["gt_available"] and prediction["status"] != "not_attempted":
            correct = (prediction["status"] == "ok"
                       and prediction["prediction"] == reference["gt_answer"])
        result = {
            **row, **prediction, **reference, "is_correct": correct,
            "prediction_text": next(
                (o["text"] for o in row["options"]
                 if o["label"] == prediction["prediction"]), None),
            "evaluation_identity_sha256": manifest["identity_sha256"],
        }
        results.append(result)
    write_jsonl(output / "results.jsonl", results)
    split_metrics = {}
    for split in manifest["splits"]:
        group = [row for row in results if row["split"] == split]
        labeled = [row for row in group if row["gt_available"]]
        correct = sum(row["is_correct"] is True for row in labeled)
        labeled_complete = all(row["status"] != "not_attempted" for row in labeled)
        score_available = bool(labeled) and labeled_complete
        split_metrics[split] = {
            "total_questions": len(group),
            "attempted": sum(row["status"] != "not_attempted" for row in group),
            "valid_predictions": sum(row["status"] == "ok" for row in group),
            "parse_errors": sum(row["status"] == "parse_error" for row in group),
            "api_errors": sum(row["status"] == "api_error" for row in group),
            "model_mismatches": sum(row["status"] == "model_mismatch" for row in group),
            "not_attempted": sum(row["status"] == "not_attempted" for row in group),
            "gt_available": len(labeled), "gt_hidden": len(group) - len(labeled),
            "correct": correct if labeled else None,
            "accuracy": correct / len(labeled) if score_available else None,
            "accuracy_percent": 100 * correct / len(labeled) if score_available else None,
            "scoring_status": ("scored" if score_available else
                               "partial" if labeled else "gt_hidden"),
        }
    labeled_all = [row for row in results if row["gt_available"]]
    correct_all = sum(row["is_correct"] is True for row in labeled_all)
    labeled_complete = bool(labeled_all) and all(
        row["status"] != "not_attempted" for row in labeled_all)
    attempted = sum(row["status"] != "not_attempted" for row in results)
    operational_failure = any(
        row["status"] in {"api_error", "model_mismatch"} for row in results)
    summary = {
        "schema_version": SCHEMA, "model": manifest["model"], "subset": SUBSET,
        "status": ("stopped" if operational_failure else
                   "complete" if attempted == len(results) else "partial"),
        "updated_at": now(), "total_questions": len(results), "attempted": attempted,
        "labeled_questions": len(labeled_all), "correct": correct_all,
        "labeled_accuracy": correct_all / len(labeled_all) if labeled_complete else None,
        "splits": split_metrics,
        "accuracy_formula": "correct / all GT-available questions in the selected split",
        "invalid_or_failed_answers_count_as_incorrect": True,
        "hidden_gt_accuracy": None, "automatic_retry": False,
        "reference_answer_visible": False, "parameters": manifest["parameters"],
        "thinking_enabled": manifest["parameters"]["enable_thinking"],
        "responses_with_reasoning_content": sum(
            bool(row.get("reasoning_content")) for row in results),
        "evaluation_identity_sha256": manifest["identity_sha256"],
        "api_calls": sum(row.get("api_calls", 0) for row in results),
        "usage": {
            key: sum((row.get("usage") or {}).get(key, 0) for row in results)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }
    write_json(output / "summary.json", summary)
    test_rows = [row for row in results if row["split"] == "test"]
    if test_rows:
        write_jsonl(output / "test_predictions.jsonl", [
            {"idx": row["idx"], "prediction": row["prediction"],
             "status": row["status"]} for row in test_rows])
    return summary


def run(args) -> dict:
    manifest, rows = prepare_manifest(args)
    credential = resolve_api_key(args)
    os.environ[args.api_key_env] = credential
    output = args.output
    prediction_path = output / "predictions.jsonl"
    predictions = load_predictions(prediction_path)
    payloads = {r["sample_id"]: r["payload_sha256"] for r in manifest["inputs"]}
    attempts = read_jsonl(output / "attempts.jsonl")
    attempted_ids = [r["sample_id"] for r in attempts]
    if len(set(attempted_ids)) != len(attempted_ids):
        raise ValueError("duplicate_model_attempts")
    if set(predictions) - set(attempted_ids) or set(attempted_ids) - set(payloads):
        raise ValueError("attempt_prediction_identity_mismatch")
    unknown = set(attempted_ids) - set(predictions)
    if unknown:
        raise ValueError("indeterminate_previous_calls_no_automatic_retry")
    if predictions and not args.resume:
        raise ValueError("existing_run_requires_resume")
    if any(r["status"] in {"api_error", "model_mismatch"} for r in predictions.values()):
        raise ValueError("previous_service_failure_requires_manual_diagnosis")
    for sample_id, prediction in predictions.items():
        if prediction["payload_sha256"] != payloads[sample_id]:
            raise ValueError("resumed_prediction_payload_changed")
    todo = iter(row for row in rows if row["sample_id"] not in predictions)
    stopped = False
    pending = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        def submit_next() -> bool:
            row = next(todo, None)
            if row is None:
                return False
            append_jsonl(output / "attempts.jsonl", {
                "sample_id": row["sample_id"], "started_at": now(),
                "payload_sha256": payloads[row["sample_id"]],
                "evaluation_identity_sha256": manifest["identity_sha256"],
            })
            future = pool.submit(infer_one, args, row, payloads[row["sample_id"]])
            pending[future] = row["sample_id"]
            return True

        for _ in range(args.concurrency):
            if not submit_next():
                break
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                pending.pop(future)
                result = future.result()
                append_jsonl(prediction_path, result)
                predictions[result["sample_id"]] = result
                stopped |= result["status"] in {"api_error", "model_mismatch"}
                progress = {
                    "status": "stopping" if stopped else "running",
                    "completed": len(predictions), "total": len(rows),
                    "last_sample_id": result["sample_id"],
                    "last_status": result["status"], "updated_at": now(),
                }
                write_json(output / "progress.json", progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
            if not stopped:
                while len(pending) < args.concurrency and submit_next():
                    pass
    # Neither references.jsonl nor its GT is used by prepare_manifest/infer_one.
    summary = score(args.root, output)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "score"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--thinking", action=argparse.BooleanOptionalAction, default=True,
        help="Enable thinking (default); use --no-thinking to disable it.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    default_output = "qwen3.8-flash" if args.thinking else "qwen3.8-flash-no-thinking"
    args.output = (args.output or args.root / "results" / default_output).resolve()
    if not 1 <= args.concurrency <= 4 or len(set(args.splits)) != len(args.splits):
        parser.error("concurrency must be 1..4 and splits must be unique")
    if args.model != MODEL:
        parser.error("this evaluation is configured for qwen3.8-flash")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "operator.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "preflight":
            manifest, rows = prepare_manifest(args)
            print(json.dumps({
                "status": "ready", "questions": len(rows), "images": 2 * len(rows),
                "model_requests": 0, "identity_sha256": manifest["identity_sha256"],
            }, ensure_ascii=False))
        elif args.mode == "score":
            print(json.dumps(score(args.root, args.output), ensure_ascii=False))
        else:
            result = run(args)
            if result["status"] != "complete":
                raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "stopped", "error_type": type(exc).__name__}),
              file=sys.stderr)
        raise SystemExit(1)
