"""§5.1 输入 / Episode Schema。"""

from typing import Literal, Optional

from . import Spec


class InputFrame(Spec):
    frame_idx: int
    timestamp: float
    width: int = 640
    height: int = 480
    blur_var: float
    overexposed_ratio: float
    underexposed_ratio: float
    quality_ok: bool


class VSIBenchEpisode(Spec):
    qa_id: str  # 官方 id
    scene_name: str  # scene 分层切分依据
    dataset: str  # scannet / scannetpp / arkitscenes
    question_type: str  # 8 题型官方 10 值之一
    question: str
    options: Optional[list[str]]  # MCA 有，NA 无
    ground_truth: str  # 仅评测用；GPT-6 在线/离线均不可见
    frames: list[InputFrame]
    split: Literal["induction", "inner_validation", "outer_holdout", "final_test"]


class DataSplitConfig(Spec):
    induction_scene_ids: list[str]
    inner_validation_scene_ids: list[str]
    outer_holdout_scene_ids: list[str]
    final_test_scene_ids: list[str]  # 系统不可见，仅记录存在
    task_type_stratification: bool = True
    split_version: str
    contamination_check_log_ref: str


class InputGateVerdict(Spec):
    """M2 输入门禁判定（§4 M2 字段 6）。"""

    level: Literal["pass", "locally_degraded", "overall_unusable"]
    degraded_frame_ids: list[int]
    action: Literal["proceed", "drop_and_refill", "fallback_2d_only", "unanswerable"]
