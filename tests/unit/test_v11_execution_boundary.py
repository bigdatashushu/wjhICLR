"""Behavioral regressions for generated-code IO, state and execution limits."""

import signal

import numpy as np
import pytest

from skill3d.sandbox.ast_guard import ast_guard
from skill3d.sandbox.kernel import RestrictedNamespaceKernel
from skill3d.tools import REGISTRY
from test_geometry_tools import _handle, _scene


def kernel(**kwargs):
    return RestrictedNamespaceKernel(REGISTRY, _handle(_scene()), **kwargs)


@pytest.mark.parametrize("expression", [
    "np.fromfile(PATH, dtype=np.uint8)",
    "np.load(PATH)",
    "np.memmap(PATH)",
    "np.ctypeslib.load_library('x', '.')",
])
def test_numerical_import_never_grants_host_file_io(tmp_path, expression):
    sentinel = tmp_path / "private-label"
    sentinel.write_text("42", encoding="utf-8")
    code = "import numpy as np\nx = " + expression.replace("PATH", repr(str(sentinel)))
    assert not ast_guard(code).ok
    # Even a direct kernel call cannot recover the prohibited NumPy capability.
    result = kernel().run_cell(code)
    assert result.error_code
    assert result.answer is None


@pytest.mark.parametrize("code", [
    "from numpy import fromfile as reader",
    "import scipy.io as io",
    "from numpy import *",
    "import math as ReturnAnswer",
    "x = __builtins__",
    "x = '{0.__self__._scene}'.format(show)",
    "tools.euclidean_distance = print",
])
def test_aliases_and_host_introspection_are_rejected(code):
    assert not ast_guard(code).ok


def test_kernel_does_not_receive_unrestricted_builtins():
    result = kernel().run_cell("open('private-label')")
    assert result.error_code == "violation_runtime"
    assert "NameError" in result.error


def test_solve_output_and_timeout_are_both_captured():
    k = kernel(cell_timeout_s=0.05)
    original = signal.getsignal(signal.SIGALRM)
    result = k.run_cell(
        "def solve(ctx):\n"
        "    print('inside-solve')\n"
        "    while True:\n"
        "        try:\n"
        "            pass\n"
        "        except Exception:\n"
        "            pass\n"
    )
    assert result.error_code == "timeout"
    assert result.stdout_tail == "inside-solve\n"
    assert signal.getsignal(signal.SIGALRM) is original
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


@pytest.mark.parametrize("mutation", [
    "frames[0][:] = 0",
    "frames[0].setflags(write=True)",
    "ctx.frames[0].fill(0)",
])
def test_frozen_pixels_cannot_change_input_or_future_observations(mutation):
    source = np.ones((2, 2, 3), dtype=np.uint8)
    k = kernel(frames=[source])
    result = k.run_cell("def solve(ctx):\n    " + mutation + "\n    ReturnAnswer('A')")
    assert result.error_code == "violation_runtime"
    np.testing.assert_array_equal(source, np.ones_like(source))
    np.testing.assert_array_equal(k._ns["frames"][0], source)


def test_numerical_modules_do_not_share_writable_state_between_arms():
    result = kernel().run_cell("import math\nmath.pi = 1")
    assert result.error_code == "violation_runtime"
    good = kernel().run_cell(
        "from numpy.linalg import norm\n"
        "import numpy as np\n"
        "def solve(ctx):\n"
        "    ReturnAnswer(float(norm(np.asarray([3., 4.]))))"
    )
    assert good.error_code is None
    assert good.answer == "5"
