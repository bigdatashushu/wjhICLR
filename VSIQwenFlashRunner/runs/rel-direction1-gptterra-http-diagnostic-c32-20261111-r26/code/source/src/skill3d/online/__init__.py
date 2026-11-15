"""Local solver and read-only shared configuration utilities."""

from pathlib import Path
from skill3d import __path__ as _roots

__path__ = [str(Path(root) / "online") for root in _roots if (Path(root) / "online").is_dir()]
