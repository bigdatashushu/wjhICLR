#!/usr/bin/env python3
"""VSI1000 Qwen Flash experiment entrypoint."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from skill3d.infra.experiment_campaign import main

if __name__ == "__main__":
    raise SystemExit(main())
