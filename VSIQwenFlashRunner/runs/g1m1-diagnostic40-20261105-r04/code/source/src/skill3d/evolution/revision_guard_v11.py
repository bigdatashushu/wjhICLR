"""Scoped lexical correction; native leakage and validation modules stay unchanged."""
from __future__ import annotations

from types import FunctionType

from skill3d.evolution.leakage import (
    _QA_ID_PATTERN, leakage_scan_text as _native_leakage_scan_text,
)

_ORDINARY_VERBS = frozenset({"sampled", "questioned"})


def leakage_scan_text(text: str, forbidden_sample_ids: set[str] | None = None) -> list[str]:
    hits = _native_leakage_scan_text(text, forbidden_sample_ids=forbidden_sample_ids)
    if "qa_id_pattern" not in hits:
        return hits
    matches = list(_QA_ID_PATTERN.finditer(text))
    if matches and all(match.group().lower() in _ORDINARY_VERBS for match in matches):
        # A literal forbidden ID, even "sampled", remains a separate hit.
        return [hit for hit in hits if hit != "qa_id_pattern"]
    return hits


def scoped_static_validator(native):
    namespace = dict(native.__globals__)
    namespace["leakage_scan_text"] = leakage_scan_text
    scoped = FunctionType(native.__code__, namespace, native.__name__,
                          native.__defaults__, native.__closure__)
    scoped.__kwdefaults__ = (
        dict(native.__kwdefaults__) if native.__kwdefaults__ is not None else None)
    scoped.__annotations__ = dict(native.__annotations__)
    return scoped
