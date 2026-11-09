from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


class RunnerError(RuntimeError):
    """An actionable local configuration or execution error."""


@dataclass(frozen=True)
class Config:
    model: str = "gpt-6.1-terra"
    base_url: str = "https://xcode.best/v1"
    reasoning_effort: str = "high"
    timeout_seconds: int = 3600
    max_output_tokens: int = 4096
    max_requests: int = 600
    libero_root: str = "third_party/LIBERO"
    suites: tuple[str, ...] = SUITES[:4]
    task_ids: tuple[int, ...] | None = None
    task_order_index: int = 0
    seed: int = 42
    # None means read the installed ORIGINAL LIBERO Hydra defaults.
    episodes_per_task: int | None = None
    max_steps: int | None = None
    image_height: int | None = None
    image_width: int | None = None
    image_orientation: str = "raw"
    render_gpu_device_id: int = -1

    def __post_init__(self):
        for name in ("timeout_seconds", "max_output_tokens", "max_requests",
                     "episodes_per_task", "max_steps", "image_height", "image_width"):
            value = getattr(self, name)
            if value is None and name in {
                "episodes_per_task", "max_steps", "image_height", "image_width"
            }:
                continue
            if type(value) is not int or value <= 0:
                raise RunnerError(f"{name} must be a positive integer.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise RunnerError("seed must be an integer in [0, 2**32).")
        if type(self.task_order_index) is not int or not 0 <= self.task_order_index < 21:
            raise RunnerError("task_order_index must be an integer in [0, 20].")
        if (not isinstance(self.suites, (tuple, list)) or not self.suites
                or any(s not in SUITES for s in self.suites)
                or len(set(self.suites)) != len(self.suites)):
            raise RunnerError("suites must contain unique supported suite names.")
        if self.task_ids is not None and (
            not isinstance(self.task_ids, (tuple, list)) or not self.task_ids
            or any(type(i) is not int or i < 0 for i in self.task_ids)
            or len(set(self.task_ids)) != len(self.task_ids)
        ):
            raise RunnerError("task_ids must contain unique nonnegative integers.")
        if "libero_90" in self.suites and self.task_order_index != 0:
            raise RunnerError("LIBERO-90 only supports task_order_index=0.")
        if self.image_orientation not in ("raw", "flip_vertical", "rotate_180"):
            raise RunnerError("Unknown image_orientation.")
        if self.reasoning_effort not in ("low", "medium", "high"):
            raise RunnerError("Unknown reasoning_effort.")
        if not isinstance(self.libero_root, str) or not self.libero_root:
            raise RunnerError("libero_root must be a path.")
        if type(self.render_gpu_device_id) is not int or self.render_gpu_device_id < -1:
            raise RunnerError("render_gpu_device_id must be -1 or a GPU index.")
        # Validate the endpoint and exact model string without loading a key.
        self.api_profile().require_model()

    @property
    def upstream_root(self) -> Path:
        path = Path(self.libero_root).expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()

    def api_profile(self):
        from xcodeapi import Profile
        return Profile("responses", self.base_url, self.model, "XCODE_GPT_API_KEY",
                       timeout_seconds=self.timeout_seconds)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path, **overrides) -> Config:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunnerError("Cannot read the Runner JSON configuration.") from exc
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(Config)}:
        raise RunnerError("Configuration has unknown fields.")
    data.update({k: v for k, v in overrides.items() if v is not None})
    for name in ("suites", "task_ids"):
        if isinstance(data.get(name), list):
            data[name] = tuple(data[name])
    return Config(**data)
