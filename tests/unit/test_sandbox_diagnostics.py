"""M3 full local diagnostics, including source lines and normal control flow."""

import pytest

from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.schemas import SceneState
from skill3d.tools import REGISTRY
from skill3d.tools.scene_handle import SceneHandle


def kernel():
    return RestrictedNamespaceKernel(
        REGISTRY, SceneHandle(SceneState(artifact_ref="fixture", scene_route="fallback_2d_only")))


def test_nested_error_has_source_call_chain_and_full_output():
    source = "def inner():\n    return {}['obj_id']\ndef outer():\n    return inner()\nprint('x' * 5000)\nouter()\n"
    result = kernel().run_cell(source)
    assert result.error_type == "KeyError"
    assert "in inner" in result.traceback_text and "in outer" in result.traceback_text
    assert "return {}['obj_id']" in result.traceback_text
    assert result.error_line == 2 and result.program_sha256
    assert len(result.stdout) == 5001 and len(result.stdout_tail) == 4096


@pytest.mark.parametrize("source,expected", [
    ("def broken(:\n", "SyntaxError"),
    ("raise RuntimeError('nested message')\n", "RuntimeError"),
    ("raise TimeoutError('timeout message')\n", "TimeoutError"),
    ("robust_distance('missing', 'other')\n", "ToolContractError"),
])
def test_exception_categories_keep_tracebacks(source, expected):
    result = kernel().run_cell(source)
    assert result.error_type.endswith(expected) or expected in result.traceback_text
    assert result.traceback_text and result.error_message and result.error_line


@pytest.mark.parametrize("source,kind", [
    ('ReturnAnswer("A")\n', "answer"),
    ('YieldObservations([], "observe")\n', "yield"),
])
def test_control_termination_is_not_an_exception_log(source, kind):
    result = kernel().run_cell(source)
    assert result.terminated == kind
    assert result.traceback_text == "" and result.error_type == ""
