"""Runner-owned prompts with explicit read-only shared synthesis dependencies."""

from pathlib import Path
from skill3d import __path__ as _roots

__path__ = [str(Path(root) / "synthesis") for root in _roots
            if (Path(root) / "synthesis").is_dir()]
