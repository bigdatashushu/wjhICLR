"""Versioned model boundary; raw ToolResult storage and M11 paths stay unchanged."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from skill3d.schemas.tool import ToolResult

TOOL_RESPONSE_VERSION = "tool-response/2.0"
TOOL_RESPONSE_FIELDS = frozenset({
    "result_id", "source_tool", "status", "data", "schema_version", "evidence_version",
})


class ModelToolResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result_id: str
    source_tool: str
    status: Literal["ok"] = "ok"
    data: Any
    schema_version: Literal["tool-response/2.0"] = TOOL_RESPONSE_VERSION
    evidence_version: str


def model_tool_response(result: ToolResult) -> dict:
    """Expose successful data without mutating it or trusting business-level IDs.

    Failed calls continue to raise at the kernel. An inner data.status such as
    incomplete_candidates/fault remains intact and is not a transport status.
    The copy is decoded from the same serialized value used by the old boundary.
    """
    if (result.status != "ok" or result.error is not None or result.error_code
            or result.invalidated_by or not result.result_id
            or result.authorization.get("allowed") is False):
        raise ValueError("Only valid successful ToolResults can enter the model boundary")
    return ModelToolResponse(
        result_id=result.result_id, source_tool=result.source_tool or result.tool,
        data=json.loads(result.value), evidence_version=result.evidence_version,
    ).model_dump()


def decode_recorded_response(value: Any, *, schema_version: str) -> Any:
    """Read an explicitly identified recorded model return, never guess its shape.

    Legacy ToolResult.value itself is always raw and needs no envelope decoder.
    This helper is for archived *model-visible* values and offline replay only.
    New online runs use only TOOL_RESPONSE_VERSION.
    """
    if schema_version == "tool-response/1.0":
        return value
    if schema_version != TOOL_RESPONSE_VERSION:
        raise ValueError("Unsupported recorded tool response version")
    return ModelToolResponse.model_validate(value).data
