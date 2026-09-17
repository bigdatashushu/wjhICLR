"""Skill Retriever（§4 M7）：硬过滤 + 语义排序。

硬过滤为真实逻辑：
- requires_artifacts 满足（由 scene.route 推导可用产物集合）
- minimum_quality 达标（scene 质量分）
- supported_coordinate_frames 含 scene.frame
- metric_scale_required == True 时 scene.scale_known 必须为 True

LanceDB 向量检索与 bge-reranker 均 lazy import；不可用时退化为
"纯硬过滤 + 关键词打分"。无命中 → 返回空列表（上层走空 Skill baseline）。
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from skill3d.schemas import RetrievedSkill, SceneState, SkillSpec

from .task_classifier import classify_question_type

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3  # TODO_CALIBRATE

# route → 可用重建产物集合（硬约束 16：路由在重建之后）
_ROUTE_ARTIFACTS: dict[str, set[str]] = {
    "full_3d": {"point_map", "c2w", "intrinsics", "depth_maps", "objects", "confidence"},
    "fallback_2d_only": {"frames", "intrinsics"},
    "unanswerable": set(),
}


def hard_filter(
    skill: SkillSpec,
    scene: SceneState,
    scene_quality: Optional[float] = None,
    task_type: Optional[str] = None,
) -> bool:
    """§4 M7 伪代码 hard_filter 的真实实现。"""
    available = _ROUTE_ARTIFACTS.get(scene.route, set())
    if not set(skill.requires_artifacts).issubset(available):
        return False
    if task_type is not None and skill.task_type != task_type:
        return False
    if scene_quality is not None and scene_quality < skill.minimum_quality:
        return False
    if scene.frame not in skill.supported_coordinate_frames:
        return False
    if skill.metric_scale_required and not scene.scale_known:
        return False
    return True


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _keyword_score(question: str, skill: SkillSpec) -> float:
    """退化打分：question 与 (task_type + description) 的 token 重叠率。"""
    q = _tokenize(question)
    if not q:
        return 0.0
    s = _tokenize(f"{skill.task_type} {skill.description}")
    return len(q & s) / len(q)


def _try_lancedb_candidates(question: str, limit: int = 50) -> Optional[list[SkillSpec]]:
    """LanceDB 向量检索候选（lazy import；不可用返回 None 走退化路径）。TODO: embedding 模型未定"""
    try:
        import lancedb  # noqa: F401
    except ImportError:
        return None
    # TODO: LanceDB 连接/表名/embedding 模型需与 Memory Store 集成时确定
    return None


def _try_rerank(question: str, skills: list[SkillSpec]) -> Optional[list[float]]:
    """bge-reranker-v2-m3 语义排序（lazy import；不可用返回 None）。TODO: 模型路径"""
    try:
        from FlagEmbedding import FlagReranker  # noqa: F401
    except ImportError:
        return None
    # TODO: bge-reranker-v2-m3 权重路径与推理后端 TODO_USER_INPUT
    return None


def retrieve(
    question: str,
    scene: SceneState,
    skills: Sequence[SkillSpec],
    question_type: Optional[str] = None,
    scene_quality: Optional[float] = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedSkill]:
    """retrieve(question, scene) -> list[RetrievedSkill]；无命中返回空列表。"""
    task_type: Optional[str] = None
    if question_type is not None:
        task_type = classify_question_type(question_type).question_type

    # LanceDB 候选（可用时）；否则使用传入的 skills 全集
    candidates = _try_lancedb_candidates(question)
    pool: Sequence[SkillSpec] = candidates if candidates is not None else skills

    hard = [
        s for s in pool if hard_filter(s, scene, scene_quality=scene_quality, task_type=task_type)
    ]
    if not hard:
        return []  # 无 Skill 命中 → 上层走空 Skill（baseline direct program generation）

    scores = _try_rerank(question, hard)
    if scores is None:
        scores = [_keyword_score(question, s) for s in hard]

    ranked = sorted(zip(hard, scores), key=lambda x: x[1], reverse=True)
    return [
        RetrievedSkill(skill_semver=f"{s.skill_id}@{s.semver}", score=float(sc), hard_filter_passed=True)
        for s, sc in ranked[:top_k]
    ]
