"""Frozen paired evaluations with an independent Responses transport."""
from pathlib import Path
import sys

_vendor = Path(__file__).resolve().parent / "vendor"
if _vendor.is_dir():
    sys.path.insert(0, str(_vendor))
