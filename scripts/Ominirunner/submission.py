"""Bounded rejected model declarations, separate from observed tool evidence."""
from __future__ import annotations

import json

from .common import digest, encoded

VERSION = "omni-rejected-submission/1"
MAX_CHARS = 2400
NOTICE = ("Rejected declarations are unverified model hypotheses, not TOOL FACTS. "
          "Only the reported rejection was checked; other fields are not certified. "
          "Repair from current RGB/usable facts; new evidence may change the answer or entity list.")


def size(value):
    # Match the exact escaped JSON delivered to the model.
    return len(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def fallback(snapshot, reason="context_budget"):
    return {k: snapshot[k] for k in (
        "version", "round", "status", "source_sha256")} | {
        "omitted": [{"field": "submission_and_rejection", "reason": reason}]}


def capture(round_index, answer, evidence, errors):
    """Snapshot only a rejected submission from a successfully terminated cell."""
    payload = {"answer": answer, "evidence": evidence, "rejection_errors": errors}
    snapshot = {"version": VERSION, "round": round_index,
                "status": "rejected_model_hypothesis", "source_sha256": digest(payload),
                **json.loads(encoded(payload))}
    if size(snapshot) <= MAX_CHARS:
        return snapshot
    omitted = []
    # Remove whole fields, retaining their identities; never truncate strings,
    # floats, individual entities or binding records into a different declaration.
    candidates = [("rejection_errors", snapshot, "rejection_errors"),
                  ("answer", snapshot, "answer")]
    if isinstance(snapshot["evidence"], dict):
        candidates += [(f"evidence.{key}", snapshot["evidence"], key)
                       for key in snapshot["evidence"]]
    else:
        candidates += [("evidence", snapshot, "evidence")]
    candidates.sort(key=lambda row: (-size(row[1][row[2]]), row[0]))
    for field, parent, key in candidates:
        value = parent.pop(key)
        omitted.append({"field": field, "sha256": digest(value), "reason": "submission_budget"})
        snapshot["omitted"] = omitted
        if size(snapshot) <= MAX_CHARS:
            return snapshot
    return fallback(snapshot, "submission_budget")


def delivery(snapshot, budget=None):
    """Evidence packing has priority; drop the whole snapshot when it won't fit."""
    if snapshot is None:
        return None
    if budget is None or size(snapshot) <= budget:
        return snapshot
    compact = fallback(snapshot)
    return compact if size(compact) <= budget else None
