"""§5.1 输入 / Episode Schema（含 §4 M1 统一固定 FrameSet）。"""

from typing import Literal, Optional

from . import Spec


class FrameSet(Spec):
    """统一固定帧集（§4 M1 / 硬约束 21）：全链路共用同一 32 帧与同一 `frame_set_hash`。

    在线链 M2/M3/M5/M7/M8 与所有 baseline/消融必须使用**同一个** FrameSet；
    M2 只做被动观测（不删/不换/不补/不重排帧），禁止双帧集。

    - `frame_ids`：与官方均匀采样对齐的**物理帧索引**（32 个、唯一、严格递增），
      同时是全链路的对齐键（`frame_ids[i]` = 第 i 个规范槽位对应的物理帧）；
    - `source_frame_indices`：来源视频帧号。直接从原始视频解码时与 `frame_ids` 相同；
      若上游先把帧抽成图片再喂进来（jsonl 数据源），此处记源视频帧号，
      而 `frame_ids` 记落盘/传输帧序号；
    - `frame_set_hash = sha256(json.dumps(frame_ids, sort_keys=True))`
      —— 纯算术、确定性、可复现、不读题目（无答案泄漏风险）。
    """

    frame_ids: list[int]
    source_frame_indices: list[int]
    timestamps: list[float]
    frame_set_hash: str
    n_frames: int = 32
    n_total_frames: int = 0
    fps: float = 0.0


class InputFrame(Spec):
    """单帧输入记录。

    M2（被动质量观测）写 `blur_var / overexposed_ratio / underexposed_ratio /
    degradation_flags / quality_weight`；`quality_ok` 只表示"未打劣化标记"，
    **不触发任何删帧/换帧/补帧动作**（硬约束 21）。
    """

    frame_idx: int
    timestamp: float
    width: int = 640
    height: int = 480
    blur_var: float
    overexposed_ratio: float
    underexposed_ratio: float
    quality_ok: bool
    # M2 被动观测产物（不改帧集，仅用于置信度 / route 判定 / 归因 / 消融）
    degradation_flags: list[str] = []
    quality_weight: float = 1.0


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
    frame_set: Optional[FrameSet] = None  # M1 冻结的帧集身份（硬约束 21）


class DataSplitConfig(Spec):
    induction_scene_ids: list[str]
    inner_validation_scene_ids: list[str]
    outer_holdout_scene_ids: list[str]
    final_test_scene_ids: list[str]  # 系统不可见，仅记录存在
    task_type_stratification: bool = True
    split_version: str
    contamination_check_log_ref: str


class InputGateVerdict(Spec):
    """M2 输入门禁判定（§4 M2 字段 6，**被动观测**）。

    M2 不再有"删帧/补帧"动作（旧 `drop_and_refill` 已废弃，硬约束 21）：
    - `level=pass` / `locally_degraded` 一律 `action="proceed"`，只带 flag/weight；
    - 仅**输入合法性**（无法解码 / 空帧 / 尺寸非法 / 损坏）→ `unanswerable`，
      这类帧导致整 episode `unavailable`，不进入 split（§4 M1）。
    """

    level: Literal["pass", "locally_degraded", "overall_unusable"]
    degraded_frame_ids: list[int]
    action: Literal["proceed", "fallback_2d_only", "unanswerable"]
    # 被动观测产物：逐帧劣化原因与整体质量权重（不改变帧集本身）
    degradation_flags: list[str] = []
    quality_weight: float = 1.0
    # §4 M2 字段 6 要求的三个输出之一：quality_score = 无 flag 帧占比（[0,1]）。
    # 与 quality_weight 口径不同——前者数"多少帧干净"，后者是"加权后的整体置信"。
    quality_score: float = 1.0
    hard_fail_frame_ids: list[int] = []   # 输入合法性硬失败帧（空帧/尺寸非法/损坏）
    n_frames: int = 0
    frame_set_hash: str = ""
