"""Validate fixed experiment panels without importing services or labels."""
from collections import Counter

from skill3d.routing.task_classifier import canonical_task


def validate_panel(sampling, expected):
    rows = sampling["questions"]
    if (sampling.get("n_questions") != expected or len(rows) != expected
            or len({str(row["qa_id"]) for row in rows}) != expected
            or sampling.get("seed") != 137):
        raise ValueError("frozen panel size, identifiers or seed changed")
    if any(row.get(key) not in (None, "") for row in rows for key in (
            "ground_truth", "reference_answer", "answer", "label")):
        raise ValueError("public panel contains reference labels")
    actual = dict(Counter(canonical_task(row["question_type"]) for row in rows))
    if actual != sampling["type_counts"]:
        raise ValueError("frozen panel task counts changed")
    if expected == 40 and set(actual.values()) != {5}:
        raise ValueError("original forty-question balanced panel changed")
    return rows
