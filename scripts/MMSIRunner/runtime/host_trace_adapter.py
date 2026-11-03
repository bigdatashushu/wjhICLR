"""Store failed tool diagnostics without using the model-visible success boundary."""
from skill3d.schemas.tool_response import model_tool_response


def trace_result(result):
    failed = (result.status != "ok" or result.error is not None or result.error_code
              or result.invalidated_by or not result.result_id
              or result.authorization.get("allowed") is False)
    if not failed:
        return model_tool_response(result)
    return {
        "schema_version": "host-failed-tool-trace/1",
        "result_id": result.result_id,
        "source_tool": result.source_tool or result.tool,
        "status": result.status,
        "error_code": result.error_code,
        "invalidated": bool(result.invalidated_by),
        "authorization_allowed": result.authorization.get("allowed"),
        "model_visible": False,
    }
