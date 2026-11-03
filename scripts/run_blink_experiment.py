#!/usr/bin/env python3
"""Compatibility entrypoint for scripts/BLINKRunner/run.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from BLINKRunner.run import entrypoint, main


if __name__ == "__main__":
    entrypoint()
