"""M4: distinguish output loops and failed programs from useful evidence updates."""

from types import SimpleNamespace

import pytest

from skill3d.online.recovery import (
    deterministic_cell_failure, evidence_state_fingerprint, field_repair_instructions,
)
from skill3d.schemas import ToolResult
from skill3d.synthesis.program_assembler import (
    executable_ast_fingerprint, streaming_degenerate_reason,
)


def test_ast_identity_ignores_only_comments_formatting_and_docstrings():
    original = '"""module docs"""\ndef solve(ctx):\n    """docs"""\n    return 3\n'
    equivalent = "# changed comment\ndef solve(ctx):\n    return (3)\n"
    assert executable_ast_fingerprint(original) == executable_ast_fingerprint(equivalent)
    assert executable_ast_fingerprint(original) != executable_ast_fingerprint(
        "def solve(ctx):\n    return 4\n")
    # A string used as an actual expression is not a function/module docstring.
    assert executable_ast_fingerprint('x = "docs"\n') != executable_ast_fingerprint(
        'x = "different"\n')
    with pytest.raises(SyntaxError):
        executable_ast_fingerprint("def broken(")


def test_streaming_repetition_requires_completed_continuous_output():
    repeated = "# I cannot infer a reliable numeric answer from these observations.\n"
    assert streaming_degenerate_reason(repeated * 12)
    assert streaming_degenerate_reason(repeated * 7) is None
    assert streaming_degenerate_reason("\n".join(f"value_{i} = {i}" for i in range(250))) is None
    assert streaming_degenerate_reason("pass\n" * 50) is None
    assert streaming_degenerate_reason("value = compute_something(12345)\n" * 12) is None
    block = "# Retain all validated evidence and avoid unsupported measurements.\n"
    block += "# Use the original images to make an explicitly declared visual estimate.\n"
    assert streaming_degenerate_reason(block * 6)


def _state(results=()):
    return SimpleNamespace(tool_results=list(results))


def test_evidence_identity_ignores_duplicate_receipts_but_keeps_new_facts():
    result = ToolResult(tool="detect_objects", result_id="r1",
                        value='{"new_object_ids": [], "detections": []}')
    base = evidence_state_fingerprint(None, None, _state([result]))
    duplicate = result.model_copy(update={"result_id": "r2"})
    assert base == evidence_state_fingerprint(None, None, _state([result, duplicate]))
    changed = result.model_copy(update={"value": '{"new_object_ids": ["chair-1"]}'})
    assert base != evidence_state_fingerprint(None, None, _state([changed]))
    invalid = result.model_copy(update={"invalidated_by": ["geometry_3d"]})
    assert base != evidence_state_fingerprint(None, None, _state([invalid]))


def test_obj_id_repair_uses_actual_detection_result_without_inventing_mapping():
    result = ToolResult(tool="detect_objects", result_id="det-real",
                        value='{"new_object_ids": [], "detections": [{"category_name": "chair"}]}')
    cell = SimpleNamespace(error_type="KeyError", error_message="'obj_id'",
                           error_source_line="oid = detection['obj_id']")
    feedback = field_repair_instructions(cell, [result])
    assert "detections" in feedback and "no obj_id" in feedback
    assert "new_object_ids=[]" in feedback and "list_objects" in feedback
    assert "does not mean" in feedback and "list order" in feedback
    assert "det-real" in feedback
    assert field_repair_instructions(cell, []) == ""


def test_transient_service_failure_is_not_a_deterministic_program_loop():
    cell = SimpleNamespace(error_type="KeyError", error_code="violation_runtime")
    assert deterministic_cell_failure(cell, [])
    fault = ToolResult(tool="detect_objects", value='{"status":"fault","service_healthy":false}')
    assert not deterministic_cell_failure(cell, [fault])
    timeout = ToolResult(tool="detect_objects", status="failed",
                         error="ConnectionTimeout from detector")
    assert not deterministic_cell_failure(cell, [timeout])
    cell.error_code = "timeout"
    assert not deterministic_cell_failure(cell, [])
