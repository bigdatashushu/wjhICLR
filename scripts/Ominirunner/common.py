"""Omni-only frozen contracts and content identities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

VERSION = "omni-static-depth-versus-official-direct/2"
PRIOR_IDS = ("15", "28", "38", "0", "3", "31", "2", "21", "33", "10", "81", "106")
# Added questions were selected from public type/ID bindings, before any labels.
ADDED_IDS = ("190", "375", "411", "222", "49", "479", "417", "350")
IDS = PRIOR_IDS + ADDED_IDS
MANIFEST_SHA = "ca0d46e1ea137844933da8fc4d785a5e5acb72ad2aea7ab24e3d3a334d95fc9c"
BINDINGS_SHA = "43c0deaf3f06931a15e2c68158b6c709670ef7936f39e46d7dca1eb6fa7987c3"


@dataclass(frozen=True)
class Config:
    model: str = "qwen3.8-flash"
    enable_thinking: bool = True
    temperature: float = 0.0
    seed: int = 137
    max_tokens: int = 8192
    timeout_s: int = 600
    max_program_rounds: int = 5
    max_tool_calls: int = 8
    max_measurements: int = 6
    text_caps: tuple = (4000, 6000, 6000, 6000, 6000, 4500)
    cell_timeout_s: int = 30

    def __post_init__(self):
        if (self.max_program_rounds < 1
                or len(self.text_caps) != self.max_program_rounds + 1
                or not 0 < self.max_measurements <= self.max_tool_calls):
            raise ValueError("invalid_episode_limits")

    def request_limit(self, arm):
        if arm not in ("direct", "skill"):
            raise ValueError("unknown_arm")
        return 1 if arm == "direct" else self.max_program_rounds + 1

    @property
    def trial_request_cap(self):
        return len(IDS) * (self.request_limit("direct") + self.request_limit("skill"))

    def public(self):
        return json.loads(json.dumps(asdict(self)))

    def kwargs(self):
        return {k: getattr(self, k) for k in (
            "enable_thinking", "temperature", "seed", "max_tokens")}


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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded(value).decode() + "\n")
        return
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(encoded(value) + b"\n")
    temp.replace(path)


def skill_text():
    text = (Path(__file__).parent / "static_skill.txt").read_text().strip()
    if not text or len(text) > 1800:
        raise ValueError("static_skill_length")
    return text


def validate_source(source):
    root = Path(__file__).resolve().parents[2]
    if "scripts/Ominirunner/worker.py" not in source:
        raise ValueError("worker_pin_missing")
    for name, expected in source.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or sha(root / path) != expected:
            raise ValueError("frozen_source_changed")
