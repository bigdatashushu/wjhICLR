"""Bounded input diagnostics; recovery always starts in a fresh cell."""
from __future__ import annotations

import inspect
import json
import math

from skill3d.schemas.tool import FailureCause
from skill3d.tools.contract import DomainValueError

VERSION = "omni-local-input-recovery/1"
INPUT_CODE = "omni_invalid_input"
TOOLS = {"point_depth", "pair_point_distance", "view_candidates", "view_depth_summary"}


def input_error(tool, field, expected, actual=None):
    """Only call for located argument errors, never evidence or quality failures."""
    if isinstance(actual, str):
        actual = actual[:80]
    elif isinstance(actual, (list, tuple)):
        actual = [v if type(v) in (int, float) and abs(v) < 1e100 else "<invalid>"
                  for v in actual[:2]]
    elif (type(actual) not in (int, float, type(None))
          or not math.isfinite(actual or 0) or abs(actual or 0) >= 1e100):
        actual = "<invalid>"
    record = {"field": field, "expected": expected, "actual": actual}
    message = json.dumps(record, allow_nan=False, separators=(",", ":"))
    return DomainValueError(
        tool, message,
        failure_cause=FailureCause(code=INPUT_CODE, scope="local_call", message=message))


def cell_failure(cell, receipts, registry):
    """Share worker/replay classification without trusting a failed answer."""
    errors = [{"code": cell["error_code"], "type": cell["error_type"],
               "line": cell["error_line"]}]
    fatal = bool(cell["answer_untrusted"] or cell["error_code"] in ("timeout", "tool_contract"))
    violations = cell.get("contract_violations", [])
    failed = [r for r in receipts if r["status"] != "ok" or r.get("error")
              or r.get("invalidated_by")]
    if (cell["error_code"] != "tool_contract" or cell["error_type"] != "DomainValueError"
            or not cell["answer_untrusted"] or not failed or violations != failed):
        return errors, fatal
    details = []
    for row in failed:
        cause = row.get("failure_cause") or {}
        if (row["tool"] not in TOOLS or row["status"] != "failed"
                or row.get("error_code") != "domain_value" or row.get("invalidated_by")
                or row.get("missing_artifacts") or cause.get("scope") != "local_call"
                or any(cause.get(k) for k in ("shared_premises", "object_ids", "result_ids", "audit_ids"))
                or cause.get("code") not in (INPUT_CODE, "invalid_tool_arguments")):
            return errors, fatal
        if cause["code"] == INPUT_CODE:
            try:
                diagnostic = json.loads(cause["message"])
            except (ValueError, TypeError, KeyError):
                return errors, fatal
            if (not isinstance(diagnostic, dict)
                    or set(diagnostic) != {"field", "expected", "actual"}
                    or not isinstance(diagnostic["field"], str)):
                return errors, fatal
        else:
            parameters = inspect.signature(registry.get(row["tool"]).fn).parameters
            allowed = [key for key in parameters if key not in ("handle", "scene")]
            supplied = row["args"]
            missing = [key for key in allowed if parameters[key].default is inspect.Parameter.empty
                       and key not in supplied]
            unexpected = sorted(str(key)[:80] for key in supplied if key not in allowed)
            # Positional binding failures lack positional receipts; do not guess.
            diagnostic = {"field": "arguments", "allowed_fields": allowed,
                          "missing_fields": missing, "unexpected_fields": unexpected}
        details.append({"code": cause["code"], "tool": row["tool"],
                        "failed_result_id": row["result_id"], **diagnostic,
                        "repair": "Correct this call in a new solve cell. The failed result and "
                                  "any answer from its cell remain untrusted. Use targets as input; "
                                  "intended_targets is returned output."})
    return details, False
