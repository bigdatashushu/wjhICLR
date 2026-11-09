"""Local learning workflow and read-only shared evolution utilities."""

from pathlib import Path
from skill3d import __path__ as _roots

__path__ = [str(Path(root) / "evolution") for root in _roots
            if (Path(root) / "evolution").is_dir()]
