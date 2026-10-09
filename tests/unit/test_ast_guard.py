"""M9 AST 白名单单测（§4 M9 字段 11）。"""

import pytest

from skill3d.sandbox.ast_guard import ast_guard
import skill3d.tools  # noqa: F401  # 注册几何 Tool


def test_reject_os_system():
    r = ast_guard("import os\nos.system('rm -rf /')")
    assert not r.ok


def test_reject_subprocess():
    r = ast_guard("import subprocess\nsubprocess.run(['ls'])")
    assert not r.ok


def test_reject_requests():
    r = ast_guard("import requests\nrequests.get('http://x')")
    assert not r.ok


def test_reject_open_host_path():
    r = ast_guard("f = open('/etc/passwd')\nprint(f.read())")
    assert not r.ok


def test_reject_eval_exec():
    assert not ast_guard("eval('1+1')").ok
    assert not ast_guard("exec('x=1')").ok
    assert not ast_guard("__import__('os')").ok


def test_reject_unregistered_function():
    r = ast_guard("result = my_custom_hack(1, 2)\nReturnAnswer(result)")
    assert not r.ok
    assert any("REGISTRY 外" in v for v in r.violations)


def test_reject_file_write_methods():
    assert not ast_guard("import numpy as np\nnp.save('x.npy', np.zeros(3))").ok
    assert not ast_guard("df.to_csv('x.csv')").ok


def test_reject_reserved_reassignment():
    assert not ast_guard("show = lambda x: x").ok
    assert not ast_guard("ReturnAnswer = print").ok
    assert not ast_guard("tools = {}").ok
    assert not ast_guard("scene.summary = 'changed'").ok
    assert not ast_guard("ctx.scene.summary = 'changed'").ok


@pytest.mark.parametrize("program", [
    'points = scene.object_points("chair")\nReturnAnswer(len(points))',
    "scale = scene.metric_scale\nReturnAnswer(scale)",
    "scene.set_point_map(None)\nReturnAnswer('A')",
    'points = ctx.scene.object_points("chair")\nReturnAnswer(len(points))',
])
def test_reject_scene_handle_geometry_bypass(program):
    result = ast_guard(program)
    assert not result.ok
    assert any("scene" in violation for violation in result.violations)


def test_allows_only_read_only_scene_metadata():
    program = (
        "route = scene.scene_route\n"
        "scope = scene.question_tool_scope\n"
        "summary = ctx.scene.summary\n"
        "ReturnAnswer('A')\n"
    )
    result = ast_guard(program)
    assert result.ok, result.violations


def test_ctx_tools_uses_the_same_registry_allowlist():
    allowed = ast_guard(
        'value = ctx.tools.euclidean_distance([0, 0, 0], [1, 0, 0])\n'
        "ReturnAnswer(value)\n")
    assert allowed.ok, allowed.violations
    assert allowed.allowed_tool_calls == ["euclidean_distance"]
    denied = ast_guard("value = ctx.tools.not_registered()\nReturnAnswer(value)\n")
    assert not denied.ok


def test_reject_dunder_access():
    assert not ast_guard("x = (1).__class__").ok


def test_legal_program_passes():
    program = """
import numpy as np
import math

p_a = [0.0, 0.0, 0.0]
p_b = [3.0, 4.0, 0.0]
d = euclidean_distance(point_a=p_a, point_b=p_b)
d2 = tools.euclidean_distance(point_a=p_a, point_b=p_b)
ok = exists_in_scene(name="chair")
acc = 0.0
for i in range(3):
    acc += float(i)
print("dist", d, math.sqrt(acc))
ReturnAnswer("A")
"""
    r = ast_guard(program)
    assert r.ok, r.violations
    assert "euclidean_distance" in r.allowed_tool_calls


def test_user_defined_helper_function_allowed():
    program = """
import numpy as np

def norm(v):
    return float(np.linalg.norm(np.asarray(v)))

n = norm([1.0, 2.0, 2.0])
ReturnAnswer(n)
"""
    r = ast_guard(program)
    assert r.ok, r.violations


def test_syntax_error_rejected():
    assert not ast_guard("def broken(:").ok
