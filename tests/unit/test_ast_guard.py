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
