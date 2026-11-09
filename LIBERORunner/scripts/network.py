"""Optional process-local IPv4 routing for hosts with unreachable IPv6."""
from __future__ import annotations

import os
import runpy
import socket
import sys


def configure():
    if os.environ.get("LIBERO_IPV4_ONLY") != "1":
        return
    original = socket.getaddrinfo

    def ipv4(host, port, family=0, type=0, proto=0, flags=0):
        return original(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4


if __name__ == "__main__":
    configure()
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name="__main__", alter_sys=True)
