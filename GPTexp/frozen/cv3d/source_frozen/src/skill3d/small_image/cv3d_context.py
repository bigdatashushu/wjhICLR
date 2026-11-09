"""Compact model feedback and bounded repair for CV3D; full evidence stays in traces."""
from collections import defaultdict
from copy import deepcopy
import json

VERSION = "cv3d-context/2"
FEEDBACK_VERSION = "cv3d-observations/2"
MAX_NOTE_CHARS = 400
HINTS = {
    "program_parse_error": "Return valid Python defining solve(ctx), with no prose outside the program.",
    "program_missing_action": "End solve(ctx) with global YieldObservations or ReturnAnswer; return/print alone does not submit.",
    "interface_misuse": "Use r['status'], r['data'][field], r['result_id']. ReturnAnswer, AnswerPayload and YieldObservations are global names, not ctx or ctx.tools members.",
    "invalid_answer": "Submit one supplied option letter using AnswerPayload(value=letter, unit='option', basis='mixed' or 'visual_estimate').",
    "invalid_observation_reference": "Yield existing result_id strings, not response dictionaries, invented IDs or variable names.",
    "no_answer": "The program finished without an answer. Use the observations and submit or explicitly yield.",
    "python_error": "Fix the indicated Python line. Recreate local variables and use the exact documented dictionary keys and global controls.",
}
REPAIRABLE_RUNTIME = frozenset({
    "AttributeError", "NameError", "UnboundLocalError", "KeyError", "TypeError",
    "IndexError", "ValueError", "ZeroDivisionError", "SyntaxError",
})


def correctable_guard(violations):
    """Reject this program; only known API notation mistakes can request a new one."""
    permitted = {
        "Access to unsafe attributes is forbidden: data",
        *(f"Unknown tool in the tools namespace: {name}"
          for name in ("ReturnAnswer", "YieldObservations", "AnswerPayload")),
    }
    return bool(violations) and set(violations) <= permitted


def compact_observations(observations, rounds):
    """Hoist equal metadata only; every numeric value, source and ID is recoverable."""
    items = deepcopy(observations)
    common = {}
    for key in ("status", "schema_version", "evidence_version"):
        if items and all(item.get(key) == items[0].get(key) and key in item for item in items):
            common[key] = items[0][key]
            for item in items:
                item.pop(key)
    grouped = defaultdict(list)
    for item in items:
        grouped[item["source_tool"]].append(item)
        item["round"] = rounds[item["result_id"]]
    constants = {}
    for tool, records in grouped.items():
        shared = {}
        for key in ("unit", "coordinate_convention", "definition", "coverage"):
            values = [r["data"] for r in records]
            if (all(isinstance(v, dict) and key in v for v in values)
                    and all(v[key] == values[0][key] for v in values)):
                shared[key] = values[0][key]
                for value in values:
                    value.pop(key)
        if shared:
            constants[tool] = shared
    return {"schema": FEEDBACK_VERSION, "common": common,
            "tool_constants": constants, "observations": items}


class CompactContext:
    def __init__(self):
        self.observations = []
        self.origins = {}
        self.history = []
        self.notes = []
        self.last_error = None

    def add_observations(self, observations, round_number):
        for observation in observations:
            rid = observation["result_id"]
            if rid in self.origins:
                existing = next(item for item in self.observations if item["result_id"] == rid)
                if existing != observation:
                    raise ValueError("observation identity changed")
                continue
            self.observations.append(deepcopy(observation))
            self.origins[rid] = round_number

    def record(self, round_number, status, *, error_code=None, cell=None, note="",
               result_ids=()):
        self.history.append({"round": round_number, "outcome": status})
        self.last_error = None
        if error_code:
            self.last_error = {"code": error_code, "hint": HINTS[error_code]}
            if cell is not None:
                self.last_error.update(
                    error_type=cell.error_type, line=cell.error_line,
                    program_line=cell.error_source_line[:240])
        if note.strip():
            clipped = note.strip()[:MAX_NOTE_CHARS]
            # Identical notes about the same results add no new context; keep the latest round.
            self.notes = [item for item in self.notes
                          if item["text"] != clipped or item["result_ids"] != list(result_ids)]
            self.notes.append({
                "round": round_number, "kind": "model_note_unverified",
                "result_ids": list(result_ids), "text": clipped,
            })

    def state(self):
        return {"previous_rounds": deepcopy(self.history), "model_notes": deepcopy(self.notes),
                "last_error": deepcopy(self.last_error)}

    def feedback(self):
        return compact_observations(self.observations, self.origins)

    def feedback_chars(self):
        return len(json.dumps(self.feedback(), ensure_ascii=True, allow_nan=False))
