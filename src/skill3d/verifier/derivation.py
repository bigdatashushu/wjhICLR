"""Deterministic replay of the declarative answer derivation contract."""

from __future__ import annotations

import json
import math
import string
from typing import Any, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from skill3d.schemas import (
    CANONICAL_UNIT_BY_QUESTION_TYPE,
    AnswerPayload,
    ToolResult,
)
from skill3d.tools.category_match import matches, option_category
from skill3d.tools.ranking_contract import ranking_problems


class DerivationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    reason: str
    result_id: str = ""


class DerivationReplayResult(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    passed: bool
    performed: bool
    op: str = ""
    input_result_ids: list[str] = Field(default_factory=list)
    expected_value: Any = None
    computed_value: Any = None
    expected_unit: str = ""
    computed_unit: str = ""
    issues: list[DerivationIssue] = Field(default_factory=list)


class DerivationReplayError(ValueError):
    def __init__(self, code: str, reason: str, *, result_id: str = "") -> None:
        super().__init__(reason)
        self.code = code
        self.result_id = result_id


_ALLOWED_PARAMETERS = {
    "field": frozenset({"field"}),
    "convert": frozenset({"field", "from_unit", "to_unit"}),
    "count": frozenset({"field"}),
    "sort": frozenset({"field", "key", "descending", "labels"}),
    "argmin": frozenset({"field", "key", "labels"}),
    "option_map": frozenset({"field", "mapping"}),
}

_CONVERSION_FACTORS = {
    ("m", "m"): 1.0,
    ("m", "cm"): 100.0,
    ("cm", "m"): 0.01,
    ("cm", "cm"): 1.0,
    ("m2", "m2"): 1.0,
    ("count", "count"): 1.0,
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _finite_json(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_json(item)
                   for key, item in value.items())
    return False


def _decode_result(result: ToolResult) -> Any:
    if result.status != "ok" or result.error is not None:
        raise DerivationReplayError(
            "result_not_successful",
            f"Result {result.result_id} does not have status ok",
            result_id=result.result_id,
        )
    if result.invalidated_by:
        raise DerivationReplayError(
            "result_invalidated",
            f"Result {result.result_id} has been invalidated",
            result_id=result.result_id,
        )
    try:
        value = json.loads(result.value, parse_constant=_reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DerivationReplayError(
            "result_not_json",
            f"Result {result.result_id} is not replayable JSON: {exc}",
            result_id=result.result_id,
        ) from exc
    if not _finite_json(value):
        raise DerivationReplayError(
            "result_not_finite_json",
            f"Result {result.result_id} contains non-finite or unsupported values",
            result_id=result.result_id,
        )
    return value


def _field_tokens(raw: Any) -> list[str | int]:
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        parts = raw.split(".")
        if any(not part for part in parts):
            raise DerivationReplayError("invalid_parameters", "field path contains an empty segment")
        return [int(part) if part.isdigit() else part for part in parts]
    if isinstance(raw, list) and all(isinstance(part, (str, int)) for part in raw):
        return list(raw)
    raise DerivationReplayError(
        "invalid_parameters", "field must be a dot-separated string or a list of strings/integers")


def _lookup(value: Any, raw_field: Any) -> Any:
    current = value
    for token in _field_tokens(raw_field):
        if isinstance(current, Mapping) and isinstance(token, str):
            if token not in current:
                raise DerivationReplayError(
                    "field_missing", f"Result has no field {token!r}")
            current = current[token]
            continue
        if isinstance(current, list) and isinstance(token, int):
            if token < 0 or token >= len(current):
                raise DerivationReplayError(
                    "field_missing", f"List index {token} is out of range")
            current = current[token]
            continue
        raise DerivationReplayError(
            "field_type_mismatch",
            f"Cannot read path segment {token!r} from {type(current).__name__}")
    return current


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DerivationReplayError(
            "type_mismatch", f"{label} must be a finite number; got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise DerivationReplayError("non_finite", f"{label} is not a finite number")
    return number


def _unit_for_field(tool: str, raw_field: Any) -> str:
    tokens = _field_tokens(raw_field)
    leaf = str(tokens[-1]) if tokens else ""
    if leaf == "count" or leaf.startswith("n_"):
        return "count"
    if leaf.endswith("_m2"):
        return "m2"
    if leaf.endswith("_m") or leaf.endswith("_metric"):
        return "m"
    if leaf in {"closest_category", "direction", "category", "option"}:
        return "option"
    if not leaf and tool == "exists_in_scene":
        return "option"
    return ""


def _validate_parameters(op: str, raw: Any) -> dict:
    if raw is None:
        params: dict = {}
    elif isinstance(raw, dict):
        params = dict(raw)
    else:
        raise DerivationReplayError(
            "invalid_parameters", "derivation.parameters must be an object")
    extra = set(params) - _ALLOWED_PARAMETERS[op]
    if extra:
        raise DerivationReplayError(
            "invalid_parameters", f"{op} contains unregistered parameters {sorted(extra)}")
    return params


def _labels(params: dict, size: int) -> Optional[list[Any]]:
    raw = params.get("labels")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != size:
        raise DerivationReplayError(
            "invalid_parameters", f"labels must be a list of length {size}")
    if not all(isinstance(value, (str, int, float)) and not isinstance(value, bool)
               for value in raw):
        raise DerivationReplayError(
            "invalid_parameters", "labels must contain only strings or finite numbers")
    if not all(_finite_json(value) for value in raw):
        raise DerivationReplayError("invalid_parameters", "labels contains non-finite numbers")
    return list(raw)


def _sortable(values: Sequence[Any], label: str) -> None:
    kinds = {"number" if isinstance(value, (int, float)) and not isinstance(value, bool)
             else "string" if isinstance(value, str) else "other"
             for value in values}
    if "other" in kinds or len(kinds) != 1:
        raise DerivationReplayError(
            "type_mismatch", f"{label} must contain either all numbers or all strings")
    if "number" in kinds:
        for index, value in enumerate(values):
            _number(value, f"{label}[{index}]")


def _execute(
    op: str,
    decoded: list[Any],
    tools: list[str],
    params: dict,
) -> tuple[Any, str]:
    field = params.get("field")
    if op == "field":
        if len(decoded) != 1:
            raise DerivationReplayError("invalid_arity", "field must reference exactly one result")
        return _lookup(decoded[0], field), _unit_for_field(tools[0], field)

    if op == "convert":
        if len(decoded) != 1:
            raise DerivationReplayError("invalid_arity", "convert must reference exactly one result")
        source_unit = str(params.get("from_unit") or "")
        target_unit = str(params.get("to_unit") or "")
        factor = _CONVERSION_FACTORS.get((source_unit, target_unit))
        if factor is None:
            raise DerivationReplayError(
                "invalid_parameters",
                f"Unsupported unit conversion {source_unit!r} -> {target_unit!r}")
        inferred = _unit_for_field(tools[0], field)
        if not inferred:
            raise DerivationReplayError(
                "unit_unverified", "Cannot determine the source unit for convert from tool result fields")
        if inferred != source_unit:
            raise DerivationReplayError(
                "unit_mismatch",
                f"Field unit is {inferred}, but derivation declares from_unit={source_unit}")
        value = _number(_lookup(decoded[0], field), "convert input")
        return value * factor, target_unit

    if op == "count":
        if len(decoded) != 1:
            raise DerivationReplayError("invalid_arity", "count must reference exactly one result")
        value = _lookup(decoded[0], field)
        if not isinstance(value, (list, dict)):
            raise DerivationReplayError(
                "type_mismatch", "count can only count elements in a JSON list or object")
        return len(value), "count"

    if op in {"sort", "argmin"}:
        key = params.get("key")
        descending = params.get("descending", False)
        if not isinstance(descending, bool):
            raise DerivationReplayError(
                "invalid_parameters", "sort.descending must be a boolean")
        if len(decoded) == 1:
            source = _lookup(decoded[0], field)
            if isinstance(source, dict):
                labels = list(source)
                values = list(source.values())
            elif isinstance(source, list):
                labels = _labels(params, len(source))
                if labels is None:
                    labels = list(range(len(source)))
                values = [_lookup(item, key) for item in source]
            else:
                raise DerivationReplayError(
                    "type_mismatch", f"A single input to {op} must decode to a list or object")
        else:
            labels = _labels(params, len(decoded))
            if labels is None:
                raise DerivationReplayError(
                    "invalid_parameters", f"{op} requires labels when referencing multiple results")
            values = [_lookup(value, field) for value in decoded]
        if not values:
            raise DerivationReplayError("empty_input", f"{op} cannot process an empty collection")
        _sortable(values, f"{op} input")
        order = sorted(
            range(len(values)),
            key=lambda index: values[index],
            reverse=descending if op == "sort" else False,
        )
        if op == "argmin":
            if len(order) > 1 and values[order[0]] == values[order[1]]:
                raise DerivationReplayError("ambiguous_minimum", "argmin has tied minimum values")
            return labels[order[0]], "option"
        return [labels[index] for index in order], "option"

    if op == "option_map":
        if len(decoded) != 1:
            raise DerivationReplayError(
                "invalid_arity", "option_map must reference exactly one result")
        mapping = params.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise DerivationReplayError(
                "invalid_parameters", "option_map.mapping must be a nonempty object")
        source = _lookup(decoded[0], field)
        if isinstance(source, bool) or not isinstance(source, (str, int, float)):
            raise DerivationReplayError(
                "type_mismatch", "option_map input must be a string or number")
        key = str(int(source)) if isinstance(source, float) and source.is_integer() else str(source)
        if key not in mapping:
            raise DerivationReplayError(
                "mapping_missing", f"option_map.mapping has no key {key!r}")
        value = mapping[key]
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise DerivationReplayError(
                "type_mismatch", "option_map output must be a string or number")
        return value, "option"

    raise DerivationReplayError("unsupported_op", f"Unsupported derivation operation {op!r}")


def _numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            return None
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _equal_value(expected: Any, computed: Any, unit: str) -> bool:
    if unit in {"count", "m", "cm", "m2"}:
        left, right = _numeric_value(expected), _numeric_value(computed)
        if left is None or right is None:
            return False
        if unit == "count":
            return left.is_integer() and right.is_integer() and int(left) == int(right)
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(computed, (list, dict)):
        if not isinstance(expected, str):
            return False
        try:
            expected = json.loads(expected, parse_constant=_reject_constant)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return expected == computed
    left, right = str(expected).strip(), str(computed).strip()
    if unit == "option" and len(left) == len(right) == 1:
        return left.upper() == right.upper()
    return left == right


def replay_derivation(
    payload: Optional[AnswerPayload],
    results: Sequence[ToolResult],
    *,
    question_type: str = "",
    options: Optional[Sequence[str]] = None,
) -> DerivationReplayResult:
    """Replay one declared operation and verify its value and unit against the answer."""
    if payload is None:
        return DerivationReplayResult(
            passed=False,
            performed=False,
            issues=[DerivationIssue(
                code="answer_payload_missing",
                reason="ReturnAnswer did not preserve a verifiable AnswerPayload",
            )],
        )
    issues = [
        DerivationIssue(code="payload_contract", reason=problem)
        for problem in payload.problems()
    ]
    expected_unit = str(CANONICAL_UNIT_BY_QUESTION_TYPE.get(question_type, "") or "")
    if expected_unit and payload.unit != expected_unit:
        issues.append(DerivationIssue(
            code="canonical_unit_mismatch",
            reason=f"Question type {question_type} requires unit={expected_unit}; got {payload.unit}",
        ))
    derivation = payload.derivation
    referenced = set(payload.used_result_ids) | set(payload.derivation_inputs())
    ranks = []
    for result in results:
        if result.result_id not in referenced or result.tool != "relative_distance_rank":
            continue
        try:
            value = _decode_result(result)
            problems = ranking_problems(value, options)
            if problems:
                issues.append(DerivationIssue(
                    code="incomplete_ranking", reason="; ".join(problems),
                    result_id=result.result_id))
            else:
                ranks.append(value)
        except DerivationReplayError as exc:
            issues.append(DerivationIssue(
                code=exc.code, reason=str(exc), result_id=exc.result_id))
    # A model-provided mapping/labels array must not swap the measured winner.
    if options and payload.basis == "tool_derived" and question_type == "object_rel_distance":
        for rank in ranks:
            winner = next(i for i, option in enumerate(options)
                          if matches(option_category(option), rank["closest_category"]))
            if str(payload.value).strip().upper() != string.ascii_uppercase[winner]:
                issues.append(DerivationIssue(
                    code="ranking_option_mismatch",
                    reason="The answer option does not match the closest category in the complete ranking"))
    # Visual/mixed submissions still have an answer domain even without a
    # declared calculation. Validate before the no-derivation early return.
    if payload.unit in {"count", "m", "cm", "m2"}:
        number = _numeric_value(payload.value)
        if number is None:
            issues.append(DerivationIssue(
                code="answer_not_finite", reason="The answer must be a finite number"))
        elif number < 0:
            issues.append(DerivationIssue(
                code="answer_negative", reason="Distance, size, area, and count must be nonnegative"))
        elif payload.unit == "count" and not number.is_integer():
            issues.append(DerivationIssue(
                code="answer_not_integer", reason="A count answer must be an integer"))
    elif payload.unit == "option" and options and (
        derivation is None or derivation.get("op") != "sort"
    ):
        allowed = set(string.ascii_uppercase[:len(options)])
        if str(payload.value).strip().upper() not in allowed:
            issues.append(DerivationIssue(
                code="option_out_of_range",
                reason=f"The answer must be a valid option letter in {sorted(allowed)}"))
    if derivation is None:
        return DerivationReplayResult(
            passed=not issues,
            performed=False,
            expected_value=payload.value,
            expected_unit=payload.unit,
            issues=issues,
        )

    op = str(derivation.get("op") or "")
    input_ids = [str(value) for value in derivation.get("input_result_ids") or []]
    result_by_id: dict[str, ToolResult] = {}
    duplicate_ids: set[str] = set()
    for result in results:
        if result.result_id in result_by_id:
            duplicate_ids.add(result.result_id)
        result_by_id[result.result_id] = result
    if duplicate_ids:
        issues.append(DerivationIssue(
            code="duplicate_result_id",
            reason=f"The tool ledger contains duplicate result_id values: {sorted(duplicate_ids)}",
        ))
    if len(input_ids) != len(set(input_ids)):
        issues.append(DerivationIssue(
            code="duplicate_input_result_id",
            reason="derivation.input_result_ids must not contain duplicates",
        ))

    computed: Any = None
    computed_unit = ""
    blocking_codes = {
        "payload_contract", "duplicate_result_id", "duplicate_input_result_id",
        "incomplete_ranking",
    }
    if not any(issue.code in blocking_codes for issue in issues):
        try:
            params = _validate_parameters(op, derivation.get("parameters"))
            decoded: list[Any] = []
            tools: list[str] = []
            for result_id in input_ids:
                result = result_by_id.get(result_id)
                if result is None:
                    raise DerivationReplayError(
                        "result_missing",
                        f"Result {result_id!r} referenced by derivation does not exist",
                        result_id=result_id,
                    )
                decoded.append(_decode_result(result))
                tools.append(str(result.tool or result.source_tool))
            computed, computed_unit = _execute(op, decoded, tools, params)
            if not _finite_json(computed):
                raise DerivationReplayError(
                    "computed_value_invalid", "derivation replay produced a non-serializable or non-finite value")
            if not computed_unit:
                issues.append(DerivationIssue(
                    code="unit_unverified",
                    reason="Cannot determine the replay result unit from registered operations and tool fields",
                ))
            elif computed_unit != payload.unit:
                issues.append(DerivationIssue(
                    code="derived_unit_mismatch",
                    reason=f"derivation produced unit={computed_unit}, but the answer declares unit={payload.unit}",
                ))
            if not _equal_value(payload.value, computed, payload.unit):
                issues.append(DerivationIssue(
                    code="value_mismatch",
                    reason=f"derivation recomputed value {computed!r} does not match answer {payload.value!r}",
                ))
            if payload.unit == "option" and options and not isinstance(computed, (list, dict)):
                answer = str(computed).strip().upper()
                allowed = set(string.ascii_uppercase[:len(options)])
                if answer not in allowed:
                    issues.append(DerivationIssue(
                        code="option_out_of_range",
                        reason=f"derivation produced option {answer!r}; valid options are {sorted(allowed)}",
                    ))
        except DerivationReplayError as exc:
            issues.append(DerivationIssue(
                code=exc.code, reason=str(exc), result_id=exc.result_id))

    return DerivationReplayResult(
        passed=not issues,
        performed=True,
        op=op,
        input_result_ids=input_ids,
        expected_value=payload.value,
        computed_value=computed,
        expected_unit=payload.unit,
        computed_unit=computed_unit,
        issues=issues,
    )


__all__ = [
    "DerivationIssue",
    "DerivationReplayError",
    "DerivationReplayResult",
    "replay_derivation",
]
