"""VSI-Bench 8 题型识别（§4 M7）。

官方 meta `question_type` 共 **10 个取值**（8 题型 + `object_rel_direction` 的
easy/medium/hard 三档变体，§1.2 已核验）：

| 规范题型（8） | 官方 question_type 取值 | 作答形式 |
|---|---|---|
| object_counting | object_counting | NA（数值，MRA） |
| object_abs_distance | object_abs_distance | NA |
| object_size_estimation | object_size_estimation | NA |
| room_size_estimation | room_size_estimation | NA |
| object_rel_distance | object_rel_distance | MCA（有 options） |
| object_rel_direction | object_rel_direction_easy / _medium / _hard | MCA |
| route_planning | route_planning | MCA |
| obj_appearance_order | obj_appearance_order | MCA |

MCA/NA 分流以**官方 meta 的 `options` 字段**为准（4 MCA + 4 NA，§1.2）：
有选项 → Accuracy；无数值 → MRA。已用真实 meta（5130 行）核对：
`object_counting` / `object_abs_distance` / `object_size_estimation` /
`room_size_estimation` 的 `options` 恒为 null，其余六类恒有 options。

历史缺口：本模块原用非官方的短名（`relative_direction`/`route_plan`/
`absolute_distance`…）且把 `object_counting`/`relative_distance` 的 MCA/NA 判反，
真实 meta 上 10 个取值有 9 个直接 `UnknownQuestionTypeError` → M7 全链失效。
现以官方取值为准（§1.2 / B-5、B-6）：只接受**官方 10 个取值**与规范聚合名
`object_rel_direction`；其它非官方短名（`relative_direction` / `route_plan` /
`absolute_distance` / `object_count` …）一律抛 `UnknownQuestionTypeError`。
宽松别名会把上游数据错误（拼错的题型名）静默映射成错误题型，污染 8 任务聚合与
Skill 检索，故不再保留。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from skill3d.schemas import VSIBenchEpisode

# 官方 meta 的 10 个 question_type 取值（§1.2）
QUESTION_TYPE_VALUES: tuple[str, ...] = (
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "route_planning",
    "obj_appearance_order",
)

# 8 个规范题型（§16.1 主表"8 任务分别准确率"按此聚合）
TASK_TYPES: tuple[str, ...] = (
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
)

# MCA（多选，Accuracy）/ NA（数值，MRA）——按官方 options 字段核定（4+4）
MC_TASKS: frozenset[str] = frozenset({
    "object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order",
})
NA_TASKS: frozenset[str] = frozenset({
    "object_counting", "object_abs_distance", "object_size_estimation",
    "room_size_estimation",
})

# 需要 metric 尺度才能作答的题（G-11：scale_confidence=low 时降级 2D-only）
MEASUREMENT_TASKS: frozenset[str] = frozenset({
    "object_abs_distance", "object_size_estimation", "room_size_estimation",
})
# 尺寸类题（原 G8 拒答门已删除；此处仅作题型分类）
SIZE_TASKS: frozenset[str] = frozenset({
    "object_size_estimation", "room_size_estimation",
})

# 官方取值 → 规范题型（**只认官方值**，§1.2 / B-5）
# - 官方 meta 的 10 个取值各自映射到 8 个规范题型（rel_direction 三档归一类）；
# - `object_rel_direction` 是规范聚合名（非 meta 取值），也接受，便于人工/Skill 侧引用。
# 非官方短名（relative_direction / route_plan / absolute_distance / object_count …）
# **不接受**：宽松别名会把上游拼错的题型静默映射成错误题型，污染聚合与 Skill 检索。
_ALIASES: dict[str, str] = {}
for _t in TASK_TYPES:
    _ALIASES[_t] = _t
for _q in QUESTION_TYPE_VALUES:
    _ALIASES[_q] = ("object_rel_direction" if _q.startswith("object_rel_direction") else _q)

# 向后兼容别名（旧代码 import 的名字）：规范题型集合，语义见 MC_TASKS/NA_TASKS
MCA_TYPES: frozenset[str] = MC_TASKS
NA_TYPES: frozenset[str] = NA_TASKS


class UnknownQuestionTypeError(ValueError):
    pass


@dataclass(frozen=True)
class TaskClassification:
    """题型判定结果。

    - `question_type`：原始官方取值（如 `object_rel_direction_hard`），落盘/报告用；
    - `task`：规范题型（8 类之一），聚合统计与 SkillSpec.task_type 用；
    - `is_mca`：True=Accuracy（精确匹配），False=MRA（数值相对准确率）。
    """

    question_type: str
    task: str
    is_mca: bool

    def __post_init__(self) -> None:
        if self.task not in MC_TASKS and self.task not in NA_TASKS:
            raise UnknownQuestionTypeError(f"未知规范题型: {self.task!r}")


def canonical_task(question_type: str) -> str:
    """官方取值 / 别名 → 规范题型（8 类之一）。"""
    qt = str(question_type).strip().lower()
    task = _ALIASES.get(qt)
    if task is None:
        raise UnknownQuestionTypeError(
            f"未知 VSI-Bench question_type: {question_type!r}；"
            f"官方取值见 QUESTION_TYPE_VALUES（{len(QUESTION_TYPE_VALUES)} 个）"
        )
    return task


def is_measurement_task(task_or_question_type: str) -> bool:
    """是否需要 metric 尺度（G-11 门控用）。"""
    return canonical_task(task_or_question_type) in MEASUREMENT_TASKS


def is_size_task(task_or_question_type: str) -> bool:
    """是否尺寸类题（保留作题型分类工具；原 G8 门控已随指标删除）。"""
    return canonical_task(task_or_question_type) in SIZE_TASKS


def classify_question_type(question_type: str) -> TaskClassification:
    qt = str(question_type).strip().lower()
    task = canonical_task(qt)
    return TaskClassification(question_type=qt, task=task, is_mca=task in MC_TASKS)


def classify(episode_or_type: Union[VSIBenchEpisode, str]) -> TaskClassification:
    """优先使用 episode.question_type meta；字符串输入亦支持（规则入口）。"""
    qt = (
        episode_or_type.question_type
        if isinstance(episode_or_type, VSIBenchEpisode)
        else episode_or_type
    )
    return classify_question_type(qt)
