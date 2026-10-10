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
            f"结果 {result.result_id} 状态不是 ok",
            result_id=result.result_id,
        )
    if result.invalidated_by:
        raise DerivationReplayError(
            "result_invalidated",
            f"结果 {result.result_id} 已失效",
            result_id=result.result_id,
        )
    try:
        value = json.loads(result.value, parse_constant=_reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DerivationReplayError(
            "result_not_json",
            f"结果 {result.result_id} 不是可重放 JSON: {exc}",
            result_id=result.result_id,
        ) from exc
    if not _finite_json(value):
        raise DerivationReplayError(
            "result_not_finite_json",
            f"结果 {result.result_id} 含非有限或不支持的值",
            result_id=result.result_id,
        )
    return value


def _field_tokens(raw: Any) -> list[str | int]:
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        parts = raw.split(".")
        if any(not part for part in parts):
            raise DerivationReplayError("invalid_parameters", "field 路径含空片段")
        return [int(part) if part.isdigit() else part for part in parts]
    if isinstance(raw, list) and all(isinstance(part, (str, int)) for part in raw):
        return list(raw)
    raise DerivationReplayError(
        "invalid_parameters", "field 必须是点分隔字符串或字符串/整数列表")


def _lookup(value: Any, raw_field: Any) -> Any:
    current = value
    for token in _field_tokens(raw_field):
        if isinstance(current, Mapping) and isinstance(token, str):
            if token not in current:
                raise DerivationReplayError(
                    "field_missing", f"结果中不存在字段 {token!r}")
            current = current[token]
            continue
        if isinstance(current, list) and isinstance(token, int):
            if token < 0 or token >= len(current):
                raise DerivationReplayError(
                    "field_missing", f"列表索引 {token} 越界")
            current = current[token]
            continue
        raise DerivationReplayError(
            "field_type_mismatch",
            f"无法从 {type(current).__name__} 读取路径片段 {token!r}")
    return current


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DerivationReplayError(
            "type_mismatch", f"{label} 必须是有限数值，实际为 {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise DerivationReplayError("non_finite", f"{label} 不是有限数值")
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
            "invalid_parameters", "derivation.parameters 必须是对象")
    extra = set(params) - _ALLOWED_PARAMETERS[op]
    if extra:
        raise DerivationReplayError(
            "invalid_parameters", f"{op} 含未登记参数 {sorted(extra)}")
    return params


def _labels(params: dict, size: int) -> Optional[list[Any]]:
    raw = params.get("labels")
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != size:
        raise DerivationReplayError(
            "invalid_parameters", f"labels 必须是长度为 {size} 的列表")
    if not all(isinstance(value, (str, int, float)) and not isinstance(value, bool)
               for value in raw):
        raise DerivationReplayError(
            "invalid_parameters", "labels 只能包含字符串或有限数值")
    if not all(_finite_json(value) for value in raw):
        raise DerivationReplayError("invalid_parameters", "labels 含非有限数值")
    return list(raw)


def _sortable(values: Sequence[Any], label: str) -> None:
    kinds = {"number" if isinstance(value, (int, float)) and not isinstance(value, bool)
             else "string" if isinstance(value, str) else "other"
             for value in values}
    if "other" in kinds or len(kinds) != 1:
        raise DerivationReplayError(
            "type_mismatch", f"{label} 必须全部为数值或全部为字符串")
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
            raise DerivationReplayError("invalid_arity", "field 必须且只能引用一个结果")
        return _lookup(decoded[0], field), _unit_for_field(tools[0], field)

    if op == "convert":
        if len(decoded) != 1:
            raise DerivationReplayError("invalid_arity", "convert 必须且只能引用一个结果")
        source_unit = str(params.get("from_unit") or "")
        target_unit = str(params.get("to_unit") or "")
        factor = _CONVERSION_FACTORS.get((source_unit, target_unit))
        if factor is None:
            raise DerivationReplayError(
                "invalid_parameters",
                f"不支持的单位换算 {source_unit!r} -> {target_unit!r}")
        inferred = _unit_for_field(tools[0], field)
        if not inferred:
            raise DerivationReplayError(
                "unit_unverified", "无法从工具返回字段确定 convert 的源单位")
        if inferred != source_unit:
            raise DerivationReplayError(
                "unit_mismatch",
                f"字段单位为 {inferred}，但 derivation 声明 from_unit={source_unit}")
        value = _number(_lookup(decoded[0], field), "convert 输入")
        return value * factor, target_unit

    if op == "count":
        if len(decoded) != 1:
            raise DerivationReplayError("invalid_arity", "count 必须且只能引用一个结果")
        value = _lookup(decoded[0], field)
        if not isinstance(value, (list, dict)):
            raise DerivationReplayError(
                "type_mismatch", "count 只能统计 JSON list 或 object 的元素数")
        return len(value), "count"

    if op in {"sort", "argmin"}:
        key = params.get("key")
        descending = params.get("descending", False)
        if not isinstance(descending, bool):
            raise DerivationReplayError(
                "invalid_parameters", "sort.descending 必须是布尔值")
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
                    "type_mismatch", f"{op} 的单个输入必须解析为 list 或 object")
        else:
            labels = _labels(params, len(decoded))
            if labels is None:
                raise DerivationReplayError(
                    "invalid_parameters", f"{op} 引用多个结果时必须提供 labels")
            values = [_lookup(value, field) for value in decoded]
        if not values:
            raise DerivationReplayError("empty_input", f"{op} 不能处理空集合")
        _sortable(values, f"{op} 输入")
        order = sorted(
            range(len(values)),
            key=lambda index: values[index],
            reverse=descending if op == "sort" else False,
        )
        if op == "argmin":
            return labels[order[0]], "option"
        return [labels[index] for index in order], "option"

    if op == "option_map":
        if len(decoded) != 1:
            raise DerivationReplayError(
                "invalid_arity", "option_map 必须且只能引用一个结果")
        mapping = params.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise DerivationReplayError(
                "invalid_parameters", "option_map.mapping 必须是非空对象")
        source = _lookup(decoded[0], field)
        if isinstance(source, bool) or not isinstance(source, (str, int, float)):
            raise DerivationReplayError(
                "type_mismatch", "option_map 输入必须是字符串或数值")
        key = str(int(source)) if isinstance(source, float) and source.is_integer() else str(source)
        if key not in mapping:
            raise DerivationReplayError(
                "mapping_missing", f"option_map.mapping 不包含键 {key!r}")
        value = mapping[key]
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise DerivationReplayError(
                "type_mismatch", "option_map 输出必须是字符串或数值")
        return value, "option"

    raise DerivationReplayError("unsupported_op", f"不支持的 derivation 操作 {op!r}")


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
                reason="ReturnAnswer 未保留可验证的 AnswerPayload",
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
            reason=f"题型 {question_type} 要求 unit={expected_unit}，实际为 {payload.unit}",
        ))
    derivation = payload.derivation
    # Visual/mixed submissions still have an answer domain even without a
    # declared calculation. Validate before the no-derivation early return.
    if payload.unit in {"count", "m", "cm", "m2"}:
        number = _numeric_value(payload.value)
        if number is None:
            issues.append(DerivationIssue(
                code="answer_not_finite", reason="答案必须是有限数值"))
        elif number < 0:
            issues.append(DerivationIssue(
                code="answer_negative", reason="距离、尺寸、面积与计数不能为负"))
        elif payload.unit == "count" and not number.is_integer():
            issues.append(DerivationIssue(
                code="answer_not_integer", reason="计数答案必须是整数"))
    elif payload.unit == "option" and options and (
        derivation is None or derivation.get("op") != "sort"
    ):
        allowed = set(string.ascii_uppercase[:len(options)])
        if str(payload.value).strip().upper() not in allowed:
            issues.append(DerivationIssue(
                code="option_out_of_range",
                reason=f"答案必须是合法选项字母 {sorted(allowed)}"))
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
            reason=f"工具台账存在重复 result_id: {sorted(duplicate_ids)}",
        ))
    if len(input_ids) != len(set(input_ids)):
        issues.append(DerivationIssue(
            code="duplicate_input_result_id",
            reason="derivation.input_result_ids 不得重复",
        ))

    computed: Any = None
    computed_unit = ""
    blocking_codes = {
        "payload_contract", "duplicate_result_id", "duplicate_input_result_id",
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
                        f"derivation 引用的结果 {result_id!r} 不存在",
                        result_id=result_id,
                    )
                decoded.append(_decode_result(result))
                tools.append(str(result.tool or result.source_tool))
            computed, computed_unit = _execute(op, decoded, tools, params)
            if not _finite_json(computed):
                raise DerivationReplayError(
                    "computed_value_invalid", "derivation 重放结果不可序列化或包含非有限值")
            if not computed_unit:
                issues.append(DerivationIssue(
                    code="unit_unverified",
                    reason="无法从登记操作和工具字段确定重放结果单位",
                ))
            elif computed_unit != payload.unit:
                issues.append(DerivationIssue(
                    code="derived_unit_mismatch",
                    reason=f"derivation 产出 unit={computed_unit}，答案声明 unit={payload.unit}",
                ))
            if not _equal_value(payload.value, computed, payload.unit):
                issues.append(DerivationIssue(
                    code="value_mismatch",
                    reason=f"derivation 重算值 {computed!r} 与答案 {payload.value!r} 不一致",
                ))
            if payload.unit == "option" and options and not isinstance(computed, (list, dict)):
                answer = str(computed).strip().upper()
                allowed = set(string.ascii_uppercase[:len(options)])
                if answer not in allowed:
                    issues.append(DerivationIssue(
                        code="option_out_of_range",
                        reason=f"derivation 产出选项 {answer!r}，合法范围为 {sorted(allowed)}",
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
