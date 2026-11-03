"""Local protocol configuration; shared dataset runners are not extended."""
from dataclasses import dataclass
import math

from . import VERSION


@dataclass(frozen=True)
class RunnerConfig:
    prompt_profile: str = VERSION
    evaluation_role: str = "test_development"
    max_requests: int = 6
    max_solve_rounds: int = 5
    max_tokens: int = 8192
    cell_timeout_s: float = 30.0
    max_feedback_chars: int = 24000
    seed: int = 137
    enable_thinking: bool = True
    original_image_bytes: bool = True
    send_depth_image: bool = True
    explicit_final_answer: bool = True
    capture_responses: bool = True
    expected_model: str = "qwen3.8-flash"

    def __post_init__(self):
        if (self.prompt_profile != VERSION or self.evaluation_role != "test_development"
                or type(self.max_requests) is not int or self.max_requests != 6
                or type(self.max_solve_rounds) is not int or self.max_solve_rounds != 5):
            raise ValueError("CV3Drunner requires five program slots plus one finalization")
        for name in ("enable_thinking", "original_image_bytes", "send_depth_image",
                     "explicit_final_answer", "capture_responses"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"invalid {name}")
        if not self.original_image_bytes or not self.explicit_final_answer:
            raise ValueError("original bytes and explicit finalization are required")
        if (type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 8192
                or type(self.max_feedback_chars) is not int
                or not 256 <= self.max_feedback_chars <= 64000
                or type(self.seed) is not int):
            raise ValueError("invalid token, feedback or seed setting")
        if (type(self.cell_timeout_s) not in (int, float)
                or not math.isfinite(self.cell_timeout_s) or self.cell_timeout_s <= 0
                or not isinstance(self.expected_model, str)):
            raise ValueError("invalid timeout or expected model")
