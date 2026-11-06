"""Trusted probes bypass the AST and exercise the actual Linux OS boundary."""
import json
import platform
import subprocess
import sys

import pytest

from sandbox_linux import linux_command
from sandbox_process import minimal_environment

pytestmark = pytest.mark.skipif(platform.system() != "Linux", reason="Linux OS policy evidence")


def test_native_file_fd_network_process_and_no_marker_bypass(tmp_path):
    bootstrap = """import sys
sys.path[:0] = PATHS
import os, socket, ctypes, json
from sandbox_linux import seal_linux
libc = ctypes.CDLL(None, use_errno=True)
libc.open.argtypes = [ctypes.c_char_p, ctypes.c_int]
libc.open.restype = ctypes.c_int
fd = os.open('/usr/bin/true', os.O_RDONLY)
seal = seal_linux()
checks = {}
for name, fn in [('file', lambda: open('/usr/bin/true', 'rb')),
                 ('fd', lambda: os.read(fd, 20)),
                 ('write', lambda: open('/tmp/probe', 'w')),
                 ('network', lambda: socket.socket()),
                 ('fork', lambda: os.fork()),
                 ('exec', lambda: os.execv('/usr/bin/true', ['true']))]:
    try:
        fn()
        checks[name] = False
    except OSError:
        checks[name] = True
checks['native_open'] = libc.open(b'/usr/bin/true', 0) == -1
print(json.dumps({'seal': seal, 'checks': checks}))
"""
    from sandbox_linux import ROOT
    bootstrap = bootstrap.replace("PATHS", repr([str(ROOT), str(ROOT.parent / "src")]))
    result = subprocess.run(linux_command(sys.executable, bootstrap),
                            env=minimal_environment(tmp_path), capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert all(report["checks"].values()), report
    assert report["seal"]["namespaces_verified"] and report["seal"]["syscall_allowlist"]
    # A manually supplied historical bwrap marker is insufficient.
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c",
         "import sys; sys.path[:0]=" + repr([str(ROOT), str(ROOT.parent / "src")])
         + "; from sandbox_linux import seal_linux; seal_linux()"],
        env={**minimal_environment(tmp_path), "EMBODIED_BWRAP": "1"},
        capture_output=True, timeout=10)
    assert result.returncode != 0
