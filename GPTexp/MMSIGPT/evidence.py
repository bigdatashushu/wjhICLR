"""Bounded public evidence summaries; never a substitute for executed receipts."""
from __future__ import annotations

EVIDENCE_VERSION = "mmsigpt-public-evidence/1"
FIELDS = {"observations", "reference_frame", "relation", "selected_option", "tool_rationale"}


def capture(value, *, n_views, labels, answer=None):
    record = {"schema": EVIDENCE_VERSION, "status": "missing", "summary": None,
              "issues": [], "answer_consistent": None, "facts_verified": False}
    if value is None:
        return record
    record["status"] = "invalid"
    if (type(value) is not dict or any(type(key) is not str for key in value)
            or set(value) != FIELDS):
        record["issues"].append("evidence_fields_invalid")
        return record
    summary = {}
    for field in ("reference_frame", "relation", "selected_option", "tool_rationale"):
        item = value[field]
        limit = 1 if field == "selected_option" else 400
        if type(item) is not str or not item.strip() or len(item) > limit:
            record["issues"].append(field + "_invalid")
        else:
            summary[field] = item
    rows = value["observations"]
    if type(rows) not in (list, tuple) or not 1 <= len(rows) <= 8:
        record["issues"].append("observations_invalid")
    else:
        observed = []
        for row in rows:
            if (
                type(row) is not dict or any(type(key) is not str for key in row)
                or set(row) != {"view_index", "fact"}
                or type(row["view_index"]) is not int
                or not 0 <= row["view_index"] < n_views
                or type(row["fact"]) is not str
                or not row["fact"].strip() or len(row["fact"]) > 300
            ):
                record["issues"].append("observation_invalid")
                continue
            observed.append({"view_index": row["view_index"], "fact": row["fact"]})
        summary["observations"] = observed
    selected = summary.get("selected_option")
    if selected is not None and selected not in labels:
        record["issues"].append("selected_option_invalid")
    if selected in labels and answer is not None:
        record["answer_consistent"] = selected == answer
        if selected != answer:
            record["issues"].append("selected_option_answer_mismatch")
    record["summary"] = summary
    if not record["issues"]:
        record["status"] = "provided"
    return record
