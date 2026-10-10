"""Frozen v11 method delivery budget and deterministic lookup identity."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from skill3d.skills.delivery import DEFAULT_METHOD_CONTEXT_MAX_CHARS


@dataclass(frozen=True)
class RetrievalPolicy:
    method_context_max_chars: int = DEFAULT_METHOD_CONTEXT_MAX_CHARS
    label: str = "ret-v11-deterministic-1"
    source: str = "default"

    def __post_init__(self) -> None:
        if self.method_context_max_chars <= 0:
            raise ValueError("retrieval.method_context_max_chars 必须为正整数")

    def canonical(self) -> dict:
        return {
            "mode": "v11_unique_active_by_question_type",
            "ranking": False,
            "version_selection": False,
            "method_context_max_chars": self.method_context_max_chars,
        }

    def sha256(self) -> str:
        payload = json.dumps(self.canonical(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def version(self) -> str:
        return self.label or f"ret-{self.sha256()[:12]}"

    def to_dict(self) -> dict:
        return {**self.canonical(), "version": self.version(),
                "sha256": self.sha256(), "source": self.source, "label": self.label}


def retrieval_policy_from_config(cfg: Mapping | None) -> RetrievalPolicy:
    section = (cfg or {}).get("retrieval")
    if section is None:
        return RetrievalPolicy()
    if not isinstance(section, Mapping):
        raise ValueError("retrieval 必须是 mapping")
    unknown = set(section) - {"config_version", "method_context_max_chars"}
    if unknown:
        raise ValueError(f"v11 不支持检索配置字段: {sorted(unknown)}")
    return RetrievalPolicy(
        method_context_max_chars=int(section.get(
            "method_context_max_chars", DEFAULT_METHOD_CONTEXT_MAX_CHARS)),
        label=str(section.get("config_version", "ret-v11-deterministic-1")),
        source="config",
    )
