#!/usr/bin/env python3
"""Read explicit final answers and score without new API calls.

Keep original predictions and strict scoring intact. Write derived predictions,
GT comparisons, metrics and a parsing receipt into a separate parsed directory.
Answer extraction takes only model text and options, never ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import evaluate as e

PARSER_SCHEMA = "blink-explicit-final-answer/1"


def parse_final_answer(text: str, options: list[dict]):
    answer, reasoning = e.parse_answer(text, options)
    if answer is not None:
        return answer, reasoning, "original_parser"
    clean = text.strip()
    blocks = list(re.finditer(
        r"```(?:json)?\s*([\s\S]*?)```", clean, flags=re.IGNORECASE,
    ))
    if blocks and blocks[-1].end() == len(clean):
        candidate = blocks[-1].group(1).strip()
        try:
            value = json.loads(candidate)
        except ValueError:
            value = None
        if isinstance(value, dict):
            parsed, explanation = e.parse_answer(candidate, options)
            if parsed is not None:
                return parsed, explanation, "final_fenced_json"
        return None, reasoning, "unparsed"
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", clean):
        candidate = clean[match.start():]
        try:
            value, end = decoder.raw_decode(candidate)
        except ValueError:
            continue
        if isinstance(value, dict) and not candidate[end:].strip():
            parsed, explanation = e.parse_answer(candidate[:end], options)
            if parsed is not None:
                candidates.append((parsed, explanation))
    if len(candidates) == 1:
        return *candidates[0], "final_json_after_prose"
    final_line = re.sub(r"[*`]", "", clean.splitlines()[-1]).strip() if clean else ""
    option_line = re.fullmatch(
        r"(?:(?:final\s+)?answer\s*:\s*)?"
        r"(?:\(([A-Z])\)|([A-Z])(?:[.)])?)"
        r"(?:\s+(.+?))?[.!]?",
        final_line, flags=re.IGNORECASE,
    )
    if option_line:
        label = (option_line.group(1) or option_line.group(2)).upper()
        suffix = (option_line.group(3) or "").strip().rstrip(".!").casefold()
        for option in options:
            if label == option["label"] and (
                not suffix or suffix == option.get("text", "").strip().casefold()
            ):
                explanation = "\n".join(clean.splitlines()[:-1]).strip() or None
                return label, explanation, "final_option_line"
    marked_labels = set()
    for statement in re.finditer(
        r"^(?:therefore,\s*)?(?:the\s+)?(?:final\s+)?answer\s*"
        r"(?::|is)\s*(.+?)\s*$",
        clean, flags=re.IGNORECASE | re.MULTILINE,
    ):
        value = statement.group(1).strip().rstrip(".!").strip("*` ").casefold()
        for option in options:
            if value in {
                option["label"].casefold(),
                option.get("text", "").strip().casefold(),
            }:
                marked_labels.add(option["label"])
    if len(marked_labels) == 1:
        return marked_labels.pop(), clean, "explicit_answer_statement"
    return None, reasoning, "unparsed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=(
        e.ROOT / "results/qwen3.8-flash-no-thinking"))
    parser.add_argument("--parsed-output", type=Path)
    args = parser.parse_args()
    raw = args.output.resolve()
    output = (args.parsed_output or raw / "parsed").resolve()
    if raw == output:
        raise ValueError("parsed_output_must_preserve_raw_predictions")
    manifest = json.loads((raw / "manifest.json").read_text())
    original_summary = json.loads((raw / "summary.json").read_text())
    if original_summary["status"] != "complete":
        raise ValueError("finish_inference_before_final_rescoring")
    rows = e.read_jsonl(raw / "predictions.jsonl")
    inputs = {
        row["sample_id"]: row for row in e.select_inputs(e.ROOT, manifest["splits"])
    }
    raw_sha = e.sha256_file(raw / "predictions.jsonl")
    parser_sha = e.sha256_file(Path(__file__))
    receipt_path = output / "answer_parsing_receipt.json"
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if (existing["raw_predictions_sha256"] != raw_sha
                or existing["parser_source_sha256"] != parser_sha):
            raise ValueError("postprocessing_identity_changed_use_new_parsed_output")
    parsed_rows = []
    methods = Counter()
    for original in rows:
        row = dict(original)
        method = "original_parser"
        row["raw_status"] = original["status"]
        row["raw_prediction"] = original["prediction"]
        if original["status"] == "parse_error":
            answer, reasoning, method = parse_final_answer(
                original["model_response"], inputs[original["sample_id"]]["options"],
            )
            if answer is not None:
                row.update({
                    "prediction": answer, "model_reasoning": reasoning, "status": "ok",
                })
        row["answer_parse_method"] = method
        row["answer_parser_schema"] = PARSER_SCHEMA
        methods[method] += 1
        parsed_rows.append(row)
    e.write_json(output / "manifest.json", manifest)
    e.write_jsonl(output / "predictions.jsonl", parsed_rows)
    summary = e.score(e.ROOT, output)
    receipt = {
        "schema_version": PARSER_SCHEMA,
        "inference_identity_sha256": manifest["identity_sha256"],
        "raw_predictions_sha256": raw_sha,
        "parser_source_sha256": parser_sha,
        "parsed_predictions_sha256": e.sha256_file(output / "predictions.jsonl"),
        "results_sha256": e.sha256_file(output / "results.jsonl"),
        "summary_sha256": e.sha256_file(output / "summary.json"),
        "parsing_methods": dict(methods),
        "recovered_explicit_answers": sum(
            r["raw_status"] == "parse_error" and r["status"] == "ok"
            for r in parsed_rows),
        "remaining_parse_errors": sum(r["status"] == "parse_error" for r in parsed_rows),
        "raw_predictions_preserved": True,
        "ground_truth_used_for_answer_extraction": False,
        "ground_truth_access": "separate local scoring stage",
        "new_model_requests": 0,
    }
    e.write_json(receipt_path, receipt)
    print(json.dumps({
        "status": summary["status"],
        "questions": summary["total_questions"],
        "val": summary["splits"].get("val"),
        "test": summary["splits"].get("test"),
        "recovered_explicit_answers": receipt["recovered_explicit_answers"],
        "remaining_parse_errors": receipt["remaining_parse_errors"],
        "new_model_requests": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
