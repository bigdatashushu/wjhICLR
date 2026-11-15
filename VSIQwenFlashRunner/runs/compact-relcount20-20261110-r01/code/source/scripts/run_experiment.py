#!/usr/bin/env python3
"""One-command, dataset-selectable v11 experiment launcher (no GPU imports)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skill3d.infra.experiment_campaign import main


if __name__ == "__main__":
    raise SystemExit(main())
