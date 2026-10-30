#!/usr/bin/env python3
"""Internal adapter for a frozen unified experiment; use run_experiment.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill3d.infra.experiment_stages_v11 import main


if __name__ == "__main__":
    raise SystemExit(main())
