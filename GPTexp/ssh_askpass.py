#!/usr/bin/python3
"""Return a transient SSH password to OpenSSH's private askpass pipe."""
import os
import sys

value = os.environ.get("GPTEXP_SSH_AUTH", "")
if not value or any(c in value for c in ("\n", "\r", "\0")):
    raise SystemExit(1)
sys.stdout.write(value + "\n")
