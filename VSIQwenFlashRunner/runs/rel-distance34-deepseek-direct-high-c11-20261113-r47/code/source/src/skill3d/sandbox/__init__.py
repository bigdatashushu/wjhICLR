"""Local static guard and read-only shared sandbox runtime."""

from pathlib import Path
from skill3d import __path__ as _roots

__path__ = [str(Path(root) / "sandbox") for root in _roots if (Path(root) / "sandbox").is_dir()]
