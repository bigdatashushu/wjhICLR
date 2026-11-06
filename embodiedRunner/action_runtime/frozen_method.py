"""Immutable embodied methods, independent of VSI task slots/active snapshots."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

from skill3d.schemas.skill import normalize_skill_markdown

HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FrozenMethod:
    skill_id: str
    version: str
    environment: str
    control_profile: str
    required_capabilities: tuple[str, ...]
    optional_capabilities: tuple[str, ...]
    tools: tuple[str, ...]
    skill_md: str
    content_sha256: str
    manifest_sha256: str
    source_ref: str

    def __post_init__(self):
        validate_method_identity(self)


def validate_method_identity(method):
    if (not isinstance(method.skill_id, str) or not method.skill_id
            or not isinstance(method.version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", method.version)
            or method.environment not in {"eb_navigation", "eb_alfred", "eb_habitat", "eb_manipulation"}
            or method.control_profile != "embodied-actions/1"
            or not isinstance(method.skill_md, str) or not method.skill_md.strip()
            or method.skill_md != normalize_skill_markdown(method.skill_md)):
        raise ValueError("invalid frozen method metadata or body")
    for values in (method.required_capabilities, method.optional_capabilities, method.tools):
        if (type(values) is not tuple or any(not isinstance(v, str) or not v for v in values)
                or len(set(values)) != len(values)):
            raise ValueError("frozen method capabilities/tools must be immutable unique names")
    if (not isinstance(method.content_sha256, str) or not isinstance(method.manifest_sha256, str)
            or HASH.fullmatch(method.content_sha256) is None
            or HASH.fullmatch(method.manifest_sha256) is None
            or hashlib.sha256(method.skill_md.encode("utf-8")).hexdigest() != method.content_sha256):
        raise ValueError("frozen method content identity mismatch")


def load_frozen_method(root, *, source_ref, metadata, manifest_sha256):
    root = Path(root).resolve()
    path = (root / source_ref).resolve()
    if Path(source_ref).is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError("frozen method source escapes its library")
    required = {"method_id", "version", "environment", "control_profile", "required_capabilities",
                "optional_capabilities", "tools", "content_sha256"}
    if set(metadata) != required or any(not isinstance(metadata[name], (list, tuple)) for name in (
            "required_capabilities", "optional_capabilities", "tools")):
        raise ValueError("unexpected frozen method metadata")
    body = normalize_skill_markdown(path.read_text(encoding="utf-8"))
    return FrozenMethod(
        metadata["method_id"], metadata["version"], metadata["environment"],
        metadata["control_profile"], tuple(metadata["required_capabilities"]),
        tuple(metadata["optional_capabilities"]), tuple(metadata["tools"]),
        body, metadata["content_sha256"], manifest_sha256, source_ref)
