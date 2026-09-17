"""VSI-Bench 8 题型识别（§4 M7）。

以官方 meta `question_type` 字段为准 + 规则做 MCA/NA 分流：
- MCA（选项题）: object_counting / relative_direction / route_plan / appearance_order
- NA（数值题）:  absolute_distance / object_size / room_size / relative_distance
官方 question_type 共 10 值（8 题型 + 变体），未知值抛 UnknownQuestionTypeError。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from skill3d.schemas import VSIBenchEpisode

# VSI-Bench 官方 question_type → 作答形式（以官方 meta 值为准）
MCA_TYPES = frozenset(
    {"object_counting", "relative_direction", "route_plan", "appearance_order"}
)
NA_TYPES = frozenset(
    {"absolute_distance", "object_size", "room_size", "relative_distance"}
)


class UnknownQuestionTypeError(ValueError):
    pass


@dataclass(frozen=True)
class TaskClassification:
    question_type: str
    is_mca: bool  # True=MCA 精确匹配；False=NA MRA


def classify_question_type(question_type: str) -> TaskClassification:
    qt = question_type.strip().lower()
    if qt in MCA_TYPES:
        return TaskClassification(question_type=qt, is_mca=True)
    if qt in NA_TYPES:
        return TaskClassification(question_type=qt, is_mca=False)
    raise UnknownQuestionTypeError(f"未知 VSI-Bench question_type: {question_type!r}")


def classify(episode_or_type: Union[VSIBenchEpisode, str]) -> TaskClassification:
    """优先使用 episode.question_type meta；字符串输入亦支持（规则入口）。"""
    qt = (
        episode_or_type.question_type
        if isinstance(episode_or_type, VSIBenchEpisode)
        else episode_or_type
    )
    return classify_question_type(qt)
