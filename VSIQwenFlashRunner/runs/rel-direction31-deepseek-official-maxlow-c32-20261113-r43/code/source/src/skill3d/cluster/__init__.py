"""Runner-owned transport with explicit read-only shared cluster dependencies."""

from pathlib import Path
from skill3d import __path__ as _roots

__path__ = [str(Path(root) / "cluster") for root in _roots
            if (Path(root) / "cluster").is_dir()]
