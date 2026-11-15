"""Local fit-set execution and read-only shared evaluation utilities."""

from pathlib import Path
from skill3d import __path__ as _roots

__path__ = [str(Path(root) / "evaluation") for root in _roots
            if (Path(root) / "evaluation").is_dir()]
