"""Parent-side scoring only. All selected episodes remain in the denominator."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

if not __package__:
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    __package__ = "MindcubeRunner"

from .common import digest, read, sha, validate_preflight, write
from .data import pair_identity


def load_answers(path, items, *, expected_sha256):
    if sha(path) != expected_sha256:
        raise ValueError("host_answer_file_changed")
    answers = read(path)
    if not isinstance(answers, dict) or set(answers) != {r["sample_id"] for r in items}:
        raise ValueError("host_answers_require_exact_selection")
    for item in items:
        if answers[item["sample_id"]] not in item["labels"]:
            raise ValueError("host_answer_letter_invalid")
    return answers


def summarize(items, arms, results, *, run_identity, answers=None):
    expected = {(item["sample_id"], arm): item for item in items for arm in arms}
    seen = {}
    for row in results:
        key = (row["sample_id"], row["arm"])
        if key not in expected or key in seen:
            raise ValueError("unexpected_or_duplicate_result")
        if (row["run_identity"] != run_identity
                or row["pair_identity"] != pair_identity(expected[key])
                or row["receipt_sha256"] != digest(
                    {k: v for k, v in row.items() if k != "receipt_sha256"})):
            raise ValueError("result_identity_changed")
        if row.get("reference_answer_visible") is not False or row.get("real_labels_available") is not False:
            raise ValueError("label_visibility_changed")
        if row["status"] == "answered" and row["answer"] not in expected[key]["labels"]:
            raise ValueError("illegal_completed_answer")
        seen[key] = row
    summary = {"schema": "mindcube-host-summary/1", "run_identity": run_identity,
               "selected_questions": len(items), "scored": answers is not None, "arms": {}}
    for arm in arms:
        rows = [seen.get((item["sample_id"], arm)) for item in items]
        statuses = Counter(row["status"] if row else "not_run" for row in rows)
        answered = sum(bool(row and row["status"] == "answered") for row in rows)
        correct = (sum(bool(row and row["status"] == "answered" and
                            row["answer"] == answers[item["sample_id"]])
                       for item, row in zip(items, rows)) if answers is not None else None)
        summary["arms"][arm] = {
            "denominator": len(items), "statuses": dict(statuses), "answered": answered,
            "legal_answer_rate": answered / len(items), "correct": correct,
            "accuracy": correct / len(items) if correct is not None else None,
            "request_count": sum(row["request_count"] for row in rows if row),
        }
    if set(arms) == {"skill", "direct"}:
        summary["paired_answered"] = sum(
            all(seen.get((item["sample_id"], arm), {}).get("status") == "answered" for arm in arms)
            for item in items)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--answers-sha256", required=True)
    args = parser.parse_args()
    root = Path(args.output)
    frozen = read(root / "preflight.json")
    validate_preflight(frozen)
    items = frozen["items"]
    answers = load_answers(args.answers, items, expected_sha256=args.answers_sha256)
    results = [read(path) for path in sorted((root / "episodes").glob("*/*/result.json"))]
    value = summarize(items, frozen["arms"], results, run_identity=frozen["run_identity"], answers=answers)
    write(root / "scored_summary.json", value, exclusive=True)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
