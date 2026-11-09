from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

PROTOCOL = "cv3d-original-depth-native-tools/1"


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_sha256(path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json_bytes(value) + b"\n"
    if exclusive:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class InferenceConfig:
    model: str = "qwen3.8-flash"
    thinking: bool = True
    temperature: float = 0.0
    seed: int = 137
    max_tokens: int = 8192
    timeout_s: float = 180.0

    def __post_init__(self):
        if not self.model or type(self.thinking) is not bool:
            raise ValueError("invalid model or thinking setting")
        if (type(self.seed) is not int or type(self.max_tokens) is not int
                or self.max_tokens < 1):
            raise ValueError("invalid seed or token limit")
        if any(not math.isfinite(v) for v in (self.temperature, self.timeout_s)):
            raise ValueError("nonfinite inference configuration")
        if self.temperature < 0 or self.timeout_s <= 0:
            raise ValueError("invalid temperature or timeout")

    def public_dict(self):
        return asdict(self)
