"""§5.1 输入 / Episode Schema（含 §4 M1 统一固定 FrameSet）。"""

from typing import Literal, Optional

from pydantic import model_validator

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
    # ---- v9 §5.2 点名的其余字段（"FrameSet 至少包含"）----
    # `frame_ids` 是**规划**的采样槽位；`readable_frame_ids` 是其中真正可解码的子集。
    # 当前实现里解码失败即输入合法性硬失败（硬约束 21），因此正常路径下两者相同；
    # 一旦按 §5.2"用可读帧继续"放开，这里就是承载"缺失掩码"的地方。
    # 缺省从 `frame_ids` 补齐（见 validator），保证既有构造点语义不变。
    schema_version: str = "frame-set/1.0"
    episode_id: str = ""
    dataset_id: str = ""
    video_id: str = ""
    scene_name: str = ""
    frame_refs: list[str] = []
    decode_status: str = "ok"
    readable_frame_ids: list[int] = []
    preprocessing_version: str = ""

    @model_validator(mode="after")
    def _consistency(self) -> "FrameSet":
        """§5.2：数组字段长度匹配，帧索引唯一且有序；可读帧必须是规划帧的子集。"""
        if not self.readable_frame_ids:
            # 未显式给出 = 全部规划帧都可读（当前实现的真实语义）
            object.__setattr__(self, "readable_frame_ids", list(self.frame_ids))
        if len(self.source_frame_indices) != len(self.frame_ids):
            raise ValueError("FrameSet: source_frame_indices 与 frame_ids 长度不匹配")
        if len(self.timestamps) != len(self.frame_ids):
            raise ValueError("FrameSet: timestamps 与 frame_ids 长度不匹配")
        if self.frame_refs and len(self.frame_refs) != len(self.frame_ids):
            raise ValueError("FrameSet: frame_refs 与 frame_ids 长度不匹配")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("FrameSet: frame_ids 必须唯一")
        if sorted(self.frame_ids) != list(self.frame_ids):
            raise ValueError("FrameSet: frame_ids 必须有序（§5.2）")
        extra = set(self.readable_frame_ids) - set(self.frame_ids)
        if extra:
            raise ValueError(f"FrameSet: readable_frame_ids 不在 frame_ids 内: {sorted(extra)}")
        return self

    def cache_identity(self) -> str:
        """§5.2：缓存身份必须含**源标识**，不能只看帧索引列表。

        规范原文："源标识及内容校验值参与缓存身份，**禁止仅凭相同的帧索引列表跨视频
        复用**。" 两个不同视频若采样出相同的 32 个索引，`frame_set_hash` 会相同，
        因此这里把 dataset／video／scene 身份与内容哈希合成一个缓存键。
        """
        import hashlib
        import json as _json

        return hashlib.sha256(_json.dumps({
            "dataset_id": self.dataset_id, "video_id": self.video_id,
            "scene_name": self.scene_name, "frame_set_hash": self.frame_set_hash,
        }, sort_keys=True).encode("utf-8")).hexdigest()


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
    # 在线求解不可见；M12 评分和 induction 的离线 Skill 修订可以读取并审计。
    ground_truth: str
    frames: list[InputFrame]
    split: Literal["induction", "inner_validation", "outer_holdout", "final_test"]
    frame_set: Optional[FrameSet] = None  # M1 冻结的帧集身份（硬约束 21）


class InputErrorRecord(Spec):
    """预登记题目的输入失败事实；不伪造图像，也不从评测分母删除。"""

    status: Literal["input_error"] = "input_error"
    qa_id: str
    scene_name: str
    dataset: str
    question_type: str
    question: str
    options: Optional[list[str]] = None
    ground_truth: str
    split: Literal["induction", "inner_validation", "outer_holdout", "final_test"]
    reason: str
    source_attempts: list[dict] = []

    def as_episode(self) -> VSIBenchEpisode:
        """Build metadata-only input; M2 terminates before reconstruction or inference."""
        return VSIBenchEpisode(
            qa_id=self.qa_id,
            scene_name=self.scene_name,
            dataset=self.dataset,
            question_type=self.question_type,
            question=self.question,
            options=self.options,
            ground_truth=self.ground_truth,
            frames=[],
            split=self.split,
            frame_set=None,
        )


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
    - 仅**输入合法性**（无法解码 / 空帧 / 尺寸非法 / 损坏）→ `unanswerable`
      控制事件；runner 生成 `input_error` 零分结果行并保留在 split 分母。
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
