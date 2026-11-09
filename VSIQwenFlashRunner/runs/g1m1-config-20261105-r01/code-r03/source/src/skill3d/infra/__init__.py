"""Local orchestration and read-only shared infrastructure."""

from pathlib import Path
from skill3d import __path__ as _roots

__path__ = [str(Path(root) / "infra") for root in _roots if (Path(root) / "infra").is_dir()]
