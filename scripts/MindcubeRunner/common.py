from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path

from . import VERSION


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("duplicate_json_key")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)


def write(path, value, *, exclusive=False):
    from .traces import atomic
    atomic(path, value, exclusive=exclusive)


@dataclass(frozen=True)
class Config:
    model: str = "qwen3.8-flash"
    enable_thinking: bool = True
    temperature: float = 0.0
    seed: int = 137
    max_tokens: int = 8192
    timeout_s: float = 600
    cell_timeout_s: float = 30
    max_program_rounds: int = 5
    max_context_chars: int = 24000

    def __post_init__(self):
        if self.model != "qwen3.8-flash" or type(self.enable_thinking) is not bool:
            raise ValueError("invalid_model_configuration")
        if type(self.max_program_rounds) is not int or self.max_program_rounds != 5:
            raise ValueError("five_program_slots_required")
        for name in ("max_tokens", "max_context_chars"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("invalid_episode_limits")
        if self.max_tokens > 8192:
            raise ValueError("invalid_episode_limits")
        if type(self.seed) is not int:
            raise ValueError("invalid_seed")
        for name in ("temperature", "timeout_s", "cell_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("invalid_numeric_configuration")
        if self.temperature != 0 or min(self.timeout_s, self.cell_timeout_s) <= 0:
            raise ValueError("invalid_sampling_or_timeout")

    def public(self):
        return asdict(self)

    def kwargs(self):
        return {name: getattr(self, name) for name in
                ("enable_thinking", "temperature", "seed", "max_tokens")}

    def request_limit(self, arm):
        if arm not in ("direct", "skill"):
            raise ValueError("unknown_arm")
        return 1 if arm == "direct" else self.max_program_rounds + 1


def skill_text():
    return (Path(__file__).parent / "static_skill.txt").read_text(encoding="utf-8").strip()


def source_identity():
    root = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob("*.py"))
    files.append(Path(__file__).parent / "static_skill.txt")
    # Freeze shared runtime dependencies as well as the adapter.
    files.extend((root / "src/skill3d").rglob("*.py"))
    return {str(path.relative_to(root)): sha(path) for path in sorted(files)}


def validate_source(source):
    root = Path(__file__).resolve().parents[2]
    if source != source_identity():
        raise ValueError("frozen_source_changed")
    for name in source:
        if Path(name).is_absolute() or ".." in Path(name).parts or not (root / name).is_file():
            raise ValueError("invalid_source_binding")


def validate_preflight(frozen):
    if (frozen.get("schema") != "mindcube-tinybench-preflight/1"
            or frozen.get("run_identity") != digest(
                {k: v for k, v in frozen.items() if k != "run_identity"})):
        raise ValueError("frozen_preflight_changed")
