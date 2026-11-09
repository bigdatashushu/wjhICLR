"""Shared frozen contracts; record both the reused and new implementation."""
from pathlib import Path

from target_detection_r05.sources import (
    ContractError, LIMITS, OperationalError, REVISION, ROOT, Sources,
    canonical, digest, file_digest, implementation_hashes as parent_hashes,
)


def implementation_hashes():
    rows = {"target_detection_r05/" + name: value for name, value in parent_hashes().items()}
    rows.update({"target_detection_qwen3vl/" + p.name: file_digest(p)
                 for p in sorted(Path(__file__).parent.glob("*.py"))})
    return rows
