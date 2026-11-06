#!/usr/bin/env python3
"""OpenSSH-only transient authentication pipe; never run directly."""
import os
import sys

value = os.environ.get("EMBODIED_SSH_AUTH", "")
if not value or any(character in value for character in "\n\r\0"):
    raise SystemExit(1)
sys.stdout.write(value + "\n")
